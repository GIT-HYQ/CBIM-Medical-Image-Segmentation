import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from inference.utils import get_inference
from metric.utils import calculate_distance, calculate_dice, calculate_dice_split, calculate_iou
import numpy as np
from .utils import concat_all_gather, remove_wrap_arounds
import logging
import pdb
from utils import is_master
from tqdm import tqdm
import SimpleITK as sitk

import cv2
import os
import math

def save_images(img, msk, msk_pred, name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy() * 255
    msk = msk.permute(1,2,0).detach().cpu().numpy() * 255
    msk_pred = msk_pred.permute(1,2,0).detach().cpu().numpy() * 255 
    image_path = save_path + name.replace('.png', '_src.png')
    mask_path = save_path + name.replace('.png', '_mask.png')
    pred_path = save_path + name.replace('.png', '_pred.png')
    cv2.imwrite(image_path, img)
    cv2.imwrite(mask_path, msk)
    cv2.imwrite(pred_path, msk_pred)


def validation(net, dataloader, args, mode='Evaluating'):
    
    net.eval()

    dice_list = []
    ASD_list = []
    HD_list = []
    IoU_list = []
    ACC_list = []
    SPE_list = []
    SEN_list = []
    for i in range(args.classes-1): # background is not including in validation
        dice_list.append([])
        ASD_list.append([])
        HD_list.append([])
        IoU_list.append([])
        ACC_list.append([])
        SPE_list.append([])
        SEN_list.append([])

    inference = get_inference(args)
    
    logging.info(mode)

    with torch.no_grad():
        iterator = tqdm(dataloader)
        for (images, labels, spacing, name) in iterator:
            # spacing here is used for distance metrics calculation
            
            torch.cuda.empty_cache() # large images may cause OOM error, release cache first to mitigate it
            images, labels = images.cuda().float(), labels.to(torch.int8).cuda()
            if args.dimension == '2d':
                images = images.permute(1, 0, 2, 3)
            
            if args.amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = inference(net, images, args)
            else:
                pred = inference(net, images, args)

            _, label_pred = torch.max(pred, dim=1)
            label_pred = label_pred.to(torch.int8)
            
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)
            
            if args.save and mode == 'Testing':
                save_path = args.cp_dir + "/output/"
                save_images(images, labels, label_pred, name[0], save_path)

            #tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            # comment this for fast debugging (HD and ASD computation for large 3D images is slow)
            tmp_ASD_list = np.zeros(args.classes-1)
            tmp_HD_list = np.zeros(args.classes-1)

            tmp_ASD_list =  np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)
        
            # The dice evaluation is based on the whole image. If image size too big, might cause gpu OOM.
            # Use calculate_dice_split instead if got OOM, it will evaluate patch by patch to reduce gpu memory consumption.
            #dice, _, _ = calculate_dice(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            # dice, _, _ = calculate_dice_split(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            iou, dice2, acc, spe, sen = calculate_iou(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            # print("dice:", dice)
            # print("dice2:", dice2)

            # exclude background
            # dice = dice.cpu().numpy()[1:]
            # print("dice:", dice)

            unique_cls = torch.unique(labels)
            for cls in range(0, args.classes-1):
                if cls+1 in unique_cls: 
                    # in case some classes are missing in the GT
                    # only classes appear in the GT are used for evaluation
                    ASD_list[cls].append(tmp_ASD_list[cls])
                    HD_list[cls].append(tmp_HD_list[cls])
                    # dice_list[cls].append(dice[cls])
                    dice_list[cls].append(dice2)
                    IoU_list[cls].append(iou)
                    ACC_list[cls].append(acc)
                    SPE_list[cls].append(spe)
                    SEN_list[cls].append(sen)

    out_dice = []
    out_ASD = []
    out_HD = []
    out_IoU = []
    out_ACC = []
    out_SPE = []
    out_SEN = []
    for cls in range(0, args.classes-1):
        out_dice.append(np.array(dice_list[cls]).mean())
        out_ASD.append(np.array(ASD_list[cls]).mean())
        out_HD.append(np.array(HD_list[cls]).mean())
        out_IoU.append(np.array(IoU_list[cls]).mean())
        out_ACC.append(np.array(ACC_list[cls]).mean())
        out_SPE.append(np.array(SPE_list[cls]).mean())
        out_SEN.append(np.array(SEN_list[cls]).mean())

    return np.array(out_dice), np.array(out_ASD), np.array(out_HD), np.array(out_IoU), np.array(out_ACC), np.array(out_SPE), np.array(out_SEN)




def validation_ddp(net, dataloader, args):
    
    net.eval()

    dice_list = []
    ASD_list = []
    HD_list = []
    unique_labels_list = []

    inference = get_inference(args)

    logging.info(f"Evaluating")
    
    with torch.no_grad():
        iterator = tqdm(dataloader) if is_master(args) else dataloader
        for (images, labels, spacing) in iterator:
            # spacing here is used for distance metrics calculation
            
            torch.cuda.empty_cache()
            images, labels = images.cuda(args.proc_idx).float(), labels.to(torch.int8).cuda(args.proc_idx)
            if args.dimension == '2d':
                images = images.permute(1, 0, 2, 3)
            
            if args.amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = inference(net, images, args)
            else:
                pred = inference(net, images, args)

            _, label_pred = torch.max(pred, dim=1)
            
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)
 

            #tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            # comment this for fast debugging. (HD and ASD computation for large 3D images are slow)
            tmp_ASD_list = np.zeros(args.classes-1)
            tmp_HD_list = np.zeros(args.classes-1)

            tmp_ASD_list =  np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)

            # The dice evaluation is based on the whole image. If image size too big, might cause gpu OOM. Put tensors to cpu if needed.
            tmp_dice_list, _, _ = calculate_dice_split(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            #tmp_dice_list, _, _ = calculate_dice(label_pred.view(-1, 1).cpu(), labels.view(-1, 1).cpu(), args.classes)


            unique_labels = torch.unique(labels).cpu().numpy()
            unique_labels =  np.pad(unique_labels, (100-len(unique_labels), 0), 'constant', constant_values=0)
            # the length of padding is just a randomly picked number (most medical tasks don't have over 100 classes)
            # The padding here is because the all_gather in DDP requires the tensors in gpus have the same shape

            tmp_dice_list = tmp_dice_list.unsqueeze(0)
            unique_labels = np.expand_dims(unique_labels, axis=0)
            tmp_ASD_list = np.expand_dims(tmp_ASD_list, axis=0)
            tmp_HD_list = np.expand_dims(tmp_HD_list, axis=0)

            if args.distributed:
                # gather results from all gpus
                tmp_dice_list = concat_all_gather(tmp_dice_list)
                
                unique_labels = torch.from_numpy(unique_labels).cuda()
                unique_labels = concat_all_gather(unique_labels)
                unique_labels = unique_labels.cpu().numpy()
                
                tmp_ASD_list = torch.from_numpy(tmp_ASD_list).cuda()
                tmp_ASD_list = concat_all_gather(tmp_ASD_list)
                tmp_ASD_list = tmp_ASD_list.cpu().numpy()

                tmp_HD_list = torch.from_numpy(tmp_HD_list).cuda()
                tmp_HD_list = concat_all_gather(tmp_HD_list)
                tmp_HD_list = tmp_HD_list.cpu().numpy()


            tmp_dice_list = tmp_dice_list.cpu().numpy()[:, 1:] # exclude background
            for idx in range(len(tmp_dice_list)):  # get the result for each sample
                ASD_list.append(tmp_ASD_list[idx])
                HD_list.append(tmp_HD_list[idx])
                dice_list.append(tmp_dice_list[idx])
                unique_labels_list.append(unique_labels[idx])
    
    # Due to the DistributedSampler pad samples to make data evenly distributed to all gpus,
    # we need to remove the padded samples for correct evaluation.
    if args.distributed:
        world_size = dist.get_world_size()
        dataset_len = len(dataloader.dataset)

        padding_size = 0 if (dataset_len % world_size) == 0 else world_size - (dataset_len % world_size)
        
        for _ in range(padding_size):
            ASD_list.pop()
            HD_list.pop()
            dice_list.pop()
            unique_labels_list.pop()
    

    out_dice = []
    out_ASD = []
    out_HD = []
    for cls in range(0, args.classes-1):
        out_dice.append([])
        out_ASD.append([])
        out_HD.append([])

    for idx in range(len(dice_list)):
        for cls in range(0, args.classes-1):
            if cls+1 in unique_labels_list[idx]:
                out_dice[cls].append(dice_list[idx][cls])
                out_ASD[cls].append(ASD_list[idx][cls])
                out_HD[cls].append(HD_list[idx][cls])
    
    out_dice_mean, out_ASD_mean, out_HD_mean = [], [], []
    for cls in range(0, args.classes-1):
        out_dice_mean.append(np.array(out_dice[cls]).mean())
        out_ASD_mean.append(np.array(out_ASD[cls]).mean())
        out_HD_mean.append(np.array(out_HD[cls]).mean())

    return np.array(out_dice_mean), np.array(out_ASD_mean), np.array(out_HD_mean)

def validation_ddp_with_large_images(net, dataloader, args):
    
    net.eval()

    dice_list = []
    ASD_list = []
    HD_list = []
    unique_labels_list = []

    inference = get_inference(args)

    logging.info(f"Evaluating")
    
    with torch.no_grad():
        iterator = tqdm(dataloader) if is_master(args) else dataloader
        for (images, labels, spacing) in iterator:
            # spacing here is used for distance metrics calculation
            
            torch.cuda.empty_cache()
            images, labels = images.cuda(args.proc_idx).float(), labels.to(torch.int8).cuda(args.proc_idx)
            if args.dimension == '2d':
                images = images.permute(1, 0, 2, 3)
            
            B, _, D, H, W = images.shape
            
            z_len = 600.
            if D > z_len:
                num_z_chunks = math.ceil(D / z_len)
                z_chunk_len = math.ceil(D / num_z_chunks)

                image_chunk_list = []
                for i in range(num_z_chunks):
                    image_chunk_list.append(images[:, :, i*z_chunk_len: (i+1)*z_chunk_len, :, :])
                
                label_pred_list = []
                for image_chunk in image_chunk_list:
                    if args.amp:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            label_pred = inference(net, image_chunk, args)
                            _, label_pred = torch.max(label_pred, dim=1)
                    else:
                        label_pred = inference(net, image_chunk, args)
                        _, label_pred = torch.max(label_pred, dim=1)
                    label_pred_list.append(label_pred)
                    
                label_pred = torch.cat(label_pred_list, dim=1)

            else:
                if args.amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        pred = inference(net, images, args)
                else:
                    pred = inference(net, images, args)

                _, label_pred = torch.max(pred, dim=1)
                
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0).to(torch.int8)
                labels = labels.squeeze(0).squeeze(0)
 

            #tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            # comment this for fast debugging. (HD and ASD computation for large 3D images are slow)
            tmp_ASD_list = np.zeros(args.classes-1)
            tmp_HD_list = np.zeros(args.classes-1)

            tmp_ASD_list =  np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)

            # The dice evaluation is based on the whole image. If image size too big, might cause gpu OOM. Put tensors to cpu if needed.
            tmp_dice_list, _, _ = calculate_dice_split(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            #tmp_dice_list, _, _ = calculate_dice(label_pred.view(-1, 1).cpu(), labels.view(-1, 1).cpu(), args.classes)


            unique_labels = torch.unique(labels).cpu().numpy()
            unique_labels =  np.pad(unique_labels, (100-len(unique_labels), 0), 'constant', constant_values=0)
            # the length of padding is just a randomly picked number (most medical tasks don't have over 100 classes)
            # The padding here is because the all_gather in DDP requires the tensors in gpus have the same shape

            tmp_dice_list = tmp_dice_list.unsqueeze(0)
            unique_labels = np.expand_dims(unique_labels, axis=0)
            tmp_ASD_list = np.expand_dims(tmp_ASD_list, axis=0)
            tmp_HD_list = np.expand_dims(tmp_HD_list, axis=0)

            if args.distributed:
                # gather results from all gpus
                tmp_dice_list = concat_all_gather(tmp_dice_list)
                
                unique_labels = torch.from_numpy(unique_labels).cuda()
                unique_labels = concat_all_gather(unique_labels)
                unique_labels = unique_labels.cpu().numpy()
                
                tmp_ASD_list = torch.from_numpy(tmp_ASD_list).cuda()
                tmp_ASD_list = concat_all_gather(tmp_ASD_list)
                tmp_ASD_list = tmp_ASD_list.cpu().numpy()

                tmp_HD_list = torch.from_numpy(tmp_HD_list).cuda()
                tmp_HD_list = concat_all_gather(tmp_HD_list)
                tmp_HD_list = tmp_HD_list.cpu().numpy()


            tmp_dice_list = tmp_dice_list.cpu().numpy()[:, 1:] # exclude background
            for idx in range(len(tmp_dice_list)):  # get the result for each sample
                ASD_list.append(tmp_ASD_list[idx])
                HD_list.append(tmp_HD_list[idx])
                dice_list.append(tmp_dice_list[idx])
                unique_labels_list.append(unique_labels[idx])
    
    # Due to the DistributedSampler pad samples to make data evenly distributed to all gpus,
    # we need to remove the padded samples for correct evaluation.
    if args.distributed:
        world_size = dist.get_world_size()
        dataset_len = len(dataloader.dataset)

        padding_size = 0 if (dataset_len % world_size) == 0 else world_size - (dataset_len % world_size)
        
        for _ in range(padding_size):
            ASD_list.pop()
            HD_list.pop()
            dice_list.pop()
            unique_labels_list.pop()
    

    out_dice = []
    out_ASD = []
    out_HD = []
    for cls in range(0, args.classes-1):
        out_dice.append([])
        out_ASD.append([])
        out_HD.append([])

    for idx in range(len(dice_list)):
        for cls in range(0, args.classes-1):
            if cls+1 in unique_labels_list[idx]:
                out_dice[cls].append(dice_list[idx][cls])
                out_ASD[cls].append(ASD_list[idx][cls])
                out_HD[cls].append(HD_list[idx][cls])
    
    out_dice_mean, out_ASD_mean, out_HD_mean = [], [], []
    for cls in range(0, args.classes-1):
        out_dice_mean.append(np.array(out_dice[cls]).mean())
        out_ASD_mean.append(np.array(out_ASD[cls]).mean())
        out_HD_mean.append(np.array(out_HD[cls]).mean())

    return np.array(out_dice_mean), np.array(out_ASD_mean), np.array(out_HD_mean)


