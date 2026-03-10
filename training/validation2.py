import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from inference.utils import get_inference
from metric.utils import calculate_distance, calculate_cldice_metric, calculate_dice_split, calculate_iou, calculate_iou_multiclass
import numpy as np
from .utils import concat_all_gather, remove_wrap_arounds
import logging
import pdb
from utils import is_master
from tqdm import tqdm
import SimpleITK as sitk
import cv2
import os
import torchvision.utils as vutils

def scale_image_max(image):
    # 将图像转换为浮点数格式
    image_float = image.astype(np.float32)

    # 找到像素值范围
    min_val, max_val = np.min(image_float), np.max(image_float)

    # 将像素值缩放到0-255之间
    image_normalized = cv2.normalize(image_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 返回归一化后的图像
    return image_normalized

def save_images2(img, msk, msk_pred, name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = scale_image_max(img)
    msk = msk.permute(1,2,0).detach().cpu().numpy()
    msk = scale_image_max(msk)
    msk_pred = msk_pred.permute(1,2,0).detach().cpu().numpy()
    msk_pred = scale_image_max(msk_pred)
    image_path = os.path.join(save_path, name.replace('.png', '_src.png'))
    mask_path = os.path.join(save_path, name.replace('.png', '_mask.png'))
    pred_path = os.path.join(save_path, name.replace('.png', '_pred.png'))
    cv2.imwrite(image_path, img)
    cv2.imwrite(mask_path, msk)
    cv2.imwrite(pred_path, msk_pred)

def save_images(img, msk, msk_pred, name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy() * 255
    msk = msk.permute(1,2,0).detach().cpu().numpy() * 255
    msk_pred = msk_pred.permute(1,2,0).detach().cpu().numpy() * 255 
    image_path = os.path.join(save_path, name.replace('.png', '_src.png'))
    mask_path = os.path.join(save_path, name.replace('.png', '_mask.png'))
    pred_path = os.path.join(save_path, name.replace('.png', '_pred.png'))
    cv2.imwrite(image_path, img)
    cv2.imwrite(mask_path, msk)
    cv2.imwrite(pred_path, msk_pred)

def visualize_results(writer, step, image, gt, max_v, pred, model=None, phase='Val'):
    """
    image: 原图 [B, 1, H, W]
    gt: 标签 [B, 1, H, W]
    max_v: Frangi最大响应图 [B, 1, H, W]
    pred: 模型预测概率图/掩码 [B, 1, H, W]
    """
    v_min = max_v.min().item()
    v_max = max_v.max().item()
    v_mean = max_v.mean().item()
    
    # 计算非零像素占比，判断是否有响应
    # 这里的 1e-7 是一个极小的阈值
    non_zero_ratio = (max_v > 1e-7).float().mean().item()
    
    logging.info(f"🔍 [Debug Prior] Epoch {step} | Max: {v_max:.8e} | Mean: {v_mean:.8e} | Min: {v_min:.8e} | Active Pixels: {non_zero_ratio:.2%}")

    # 如果发现 v_max 极其微小（比如小于 1e-5），在可视化前可以手动放大，方便观察
    if v_max < 1e-4 and v_max > 0:
        logging.warning(f"⚠️ Prior response is extremely weak. Applying 100x boost for visualization.")
    
    with torch.no_grad():
        # 1. 原图归一化 (仅用于显示)
        img_show = (image[0:1] - image[0:1].min()) / (image[0:1].max() - image[0:1].min() + 1e-8)
        
        # 2. 预测图处理 (绝对严谨逻辑)
        if pred.shape[1] > 1:
            # 多分类：取血管通道(1)，不重新归一化，保留原始置信度
            pred_show = torch.softmax(pred, dim=1)[0:1, 1:2, :, :]
        else:
            # 二分类：Sigmoid，保留 [0, 1] 概率
            pred_show = torch.sigmoid(pred[0:1])

        # 3. 血管先验 (HGPG 输入)
        # max_v 本身在 [0, 1] 之间，直接取第一个样本
        vessel_prior = max_v[0:1]

        # 4. 拼接 (Image | GT | HGPG_Prior | Prediction)
        # 注意：这里我们不给 pred_show 做 min-max，亮度越亮代表模型越确信
        comparison = torch.cat([img_show, gt[0:1].float(), vessel_prior, pred_show], dim=3)

        # 5. 输出到 TensorBoard
        grid = vutils.make_grid(comparison, normalize=False)
        writer.add_image(f'{phase}/Structural_Evaluation', grid, step)

def validation(net, dataloader, args, mode='Evaluating', writer=None, epoch=0):
    
    net.eval()

    # 使用字典管理所有指标列表
    metrics_log = {
        'Dice': [], 'ASD': [], 'HD': [], 'IoU': [],
        'ACC': [], 'SPE': [], 'SEN': [], 'clDice': []
    }
    # 初始化每个类别的列表
    for key in metrics_log.keys():
        metrics_log[key] = [[] for _ in range(args.classes - 1)]

    inference = get_inference(args)
    
    logging.info(mode)

    with torch.no_grad():
        iterator = tqdm(dataloader)
        for i, (images, labels, spacing, name) in enumerate(iterator):
            # spacing here is used for distance metrics calculation
            
            inputs, labels = images.float().cuda(), labels.cuda().to(torch.int8)
            
            if args.dimension == '2d':
                inputs = inputs.permute(1, 0, 2, 3)

            # 对于 medformer_hgpg，inference 现在返回 (prob, prior)
            max_v_vis = args.model == 'medformer_hgpg' or args.model == 'medformer_hgpg_graph'
            max_v = None
            if max_v_vis:
                pred, max_v = inference(net, inputs, args)
            else:
                pred = inference(net, inputs, args)
                

            if writer and max_v_vis and i == 0:
                # max_v = net.geometric_analyzer(inputs)           
                visualize_results(writer, epoch, image=inputs, gt=labels, max_v=max_v, pred=pred)

            _, label_pred = torch.max(pred, dim=1)
            label_pred = label_pred.to(torch.int8)
            
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)
            
            if args.save and mode == 'Testing':
                save_path = args.save_path if args.save_path is not None else args.cp_dir + "/preds"
                save_images2(inputs, labels, label_pred, name[0], save_path)

            tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            # comment this for fast debugging (HD and ASD computation for large 3D images is slow)
            #tmp_ASD_list = np.zeros(args.classes-1)
            #tmp_HD_list = np.zeros(args.classes-1)

            tmp_ASD_list =  np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)
        
            # The dice evaluation is based on the whole image. If image size too big, might cause gpu OOM.
            # Use calculate_dice_split instead if got OOM, it will evaluate patch by patch to reduce gpu memory consumption.
            #dice, _, _ = calculate_dice(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            # dice, _, _ = calculate_dice_split(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            if args.classes == 2:   # 多分类引入了calculate_iou_multiclass，但为了保持之前2分类的逻辑，加了判断
                iou, dice2, acc, spe, sen = calculate_iou(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            else:
                iou, dice2, acc, spe, sen = calculate_iou_multiclass(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)

            # --- 3. 新增 clDice 计算 ---
            # 转为 Numpy 进行骨架化评估
            np_pred = label_pred.cpu().numpy()
            np_labels = labels.cpu().numpy()
            # 这里的逻辑仅演示二分类(血管/背景)，如果是多分类需对各类别单独做 mask
            tmp_clDice = calculate_cldice_metric(np_pred, np_labels)

            unique_cls = torch.unique(labels)
            for cls in range(0, args.classes-1):
                if cls+1 in unique_cls: 
                    # in case some classes are missing in the GT
                    # only classes appear in the GT are used for evaluation
                    metrics_log['ASD'][cls].append(tmp_ASD_list[cls])
                    metrics_log['HD'][cls].append(tmp_HD_list[cls])
                    metrics_log['Dice'][cls].append(dice2)
                    metrics_log['IoU'][cls].append(iou)
                    metrics_log['ACC'][cls].append(acc)
                    metrics_log['SPE'][cls].append(spe)
                    metrics_log['SEN'][cls].append(sen)
                    metrics_log['clDice'][cls].append(tmp_clDice)

    # --- 聚合结果为 Dict ---
    # 计算每个类别在所有 batch 上的平均值，再返回一个包含各指标数组的字典
    performance_dict = {}
    for key, value_list in metrics_log.items():
        # 结果是 shape 为 (classes-1,) 的 numpy 数组
        performance_dict[key] = np.array([np.array(cls_list).mean() for cls_list in value_list])

    return performance_dict


def validation_without_calc(net, dataloader, args, mode='Evaluating', writer=None, epoch=0):
    
    net.eval()

    # 使用字典管理所有指标列表
    metrics_log = {
        'Dice': [], 'ASD': [], 'HD': [], 'IoU': [],
        'ACC': [], 'SPE': [], 'SEN': [], 'clDice': []
    }
    # 初始化每个类别的列表
    for key in metrics_log.keys():
        metrics_log[key] = [[] for _ in range(args.classes - 1)]

    inference = get_inference(args)
    
    logging.info(mode)

    with torch.no_grad():
        iterator = tqdm(dataloader)
        for i, (images, labels, spacing, name) in enumerate(iterator):
            # spacing here is used for distance metrics calculation
            
            inputs, labels = images.float().cuda(), labels.cuda().to(torch.int8)
            
            if args.dimension == '2d':
                inputs = inputs.permute(1, 0, 2, 3)

            # 对于 medformer_hgpg，inference 现在返回 (prob, prior)
            max_v_vis = args.model == 'medformer_hgpg' or args.model == 'medformer_hgpg_graph'
            max_v = None
            if max_v_vis:
                pred, max_v = inference(net, inputs, args)
            else:
                pred = inference(net, inputs, args)
                

            if writer and max_v_vis and i == 0:
                # max_v = net.geometric_analyzer(inputs)           
                visualize_results(writer, epoch, image=inputs, gt=labels, max_v=max_v, pred=pred)

            _, label_pred = torch.max(pred, dim=1)
            label_pred = label_pred.to(torch.int8)
            
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)
            
            if args.save and mode == 'Testing':
                save_path = args.save_path if args.save_path is not None else args.cp_dir + "/preds"
                save_images2(inputs, labels, label_pred, name[0], save_path)

            tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            # comment this for fast debugging (HD and ASD computation for large 3D images is slow)
            #tmp_ASD_list = np.zeros(args.classes-1)
            #tmp_HD_list = np.zeros(args.classes-1)

            tmp_ASD_list =  np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)
        
            # The dice evaluation is based on the whole image. If image size too big, might cause gpu OOM.
            # Use calculate_dice_split instead if got OOM, it will evaluate patch by patch to reduce gpu memory consumption.
            #dice, _, _ = calculate_dice(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            # dice, _, _ = calculate_dice_split(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            # if args.classes == 2:   # 多分类引入了calculate_iou_multiclass，但为了保持之前2分类的逻辑，加了判断
            #     iou, dice2, acc, spe, sen = calculate_iou(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            # else:
            #     iou, dice2, acc, spe, sen = calculate_iou_multiclass(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)

            # --- 3. 新增 clDice 计算 ---
            # 转为 Numpy 进行骨架化评估
            # np_pred = label_pred.cpu().numpy()
            # np_labels = labels.cpu().numpy()
            # 这里的逻辑仅演示二分类(血管/背景)，如果是多分类需对各类别单独做 mask
            # tmp_clDice = calculate_cldice_metric(np_pred, np_labels)

            unique_cls = torch.unique(labels)
            for cls in range(0, args.classes-1):
                if cls+1 in unique_cls: 
                    # in case some classes are missing in the GT
                    # only classes appear in the GT are used for evaluation
                    metrics_log['ASD'][cls].append(tmp_ASD_list[cls])
                    metrics_log['HD'][cls].append(tmp_HD_list[cls])
                    # metrics_log['Dice'][cls].append(dice2)
                    # metrics_log['IoU'][cls].append(iou)
                    # metrics_log['ACC'][cls].append(acc)
                    # metrics_log['SPE'][cls].append(spe)
                    # metrics_log['SEN'][cls].append(sen)
                    # metrics_log['clDice'][cls].append(tmp_clDice)

    # --- 聚合结果为 Dict ---
    # 计算每个类别在所有 batch 上的平均值，再返回一个包含各指标数组的字典
    performance_dict = {}
    for key, value_list in metrics_log.items():
        # 结果是 shape 为 (classes-1,) 的 numpy 数组
        performance_dict[key] = np.array([np.array(cls_list).mean() for cls_list in value_list])

    return performance_dict


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
            
            inputs, labels = images.cuda(args.proc_idx).float(), labels.cuda(args.proc_idx).long()
            
            if args.dimension == '2d':
                inputs = inputs.permute(1, 0, 2, 3)
            
            pred = inference(net, inputs, args)

            _, label_pred = torch.max(pred, dim=1)
            
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)
 

            tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            # comment this for fast debugging. (HD and ASD computation for large 3D images are slow)
            #tmp_ASD_list = np.zeros(args.classes-1)
            #tmp_HD_list = np.zeros(args.classes-1)

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


