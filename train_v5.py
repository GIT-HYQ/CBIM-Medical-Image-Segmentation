import builtins
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from training.utils import update_ema_variables
from training.losses import DiceLoss
from training.validation import validation
from training.utils import (
    exp_lr_scheduler_with_warmup, 
    log_evaluation_result, 
    get_optimizer, 
    filter_validation_results
)
import yaml
import argparse
import time
import math
import sys
import pdb
import warnings

import matplotlib.pyplot as plt

from utils import (
    configure_logger,
    save_configure,
    AverageMeter,
    ProgressMeter,
    resume_load_optimizer_checkpoint,
    resume_load_model_checkpoint,
)

import types
import collections
from random import shuffle

warnings.filterwarnings("ignore", category=UserWarning)
from datetime import datetime


# --- [新增 import] ---
from skimage.morphology import skeletonize

class TopoLoss(nn.Module):
    def __init__(self, pos_weight=25.0): # 64x64 下血管更稀疏，略微调高权重
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).cuda())
        self.recent_skel_gt = None 

    def generate_skeleton_gt(self, mask_tensor):
        masks = mask_tensor.detach().cpu().numpy()
        skel_batch = []
        for i in range(masks.shape[0]):
            single_mask = masks[i, 0] > 0
            if single_mask.sum() == 0:
                skel = np.zeros_like(single_mask)
            else:
                try:
                    skel = skeletonize(single_mask)
                except:
                    skel = np.zeros_like(single_mask)
            skel_batch.append(skel)
        return torch.from_numpy(np.array(skel_batch)).unsqueeze(1).float().to(mask_tensor.device)

    def forward(self, skel_pred, gt_mask):
        # 1. 生成全分辨率骨架并存储以便可视化
        gt_skel_full = self.generate_skeleton_gt(gt_mask)
        self.recent_skel_gt = gt_skel_full 
        
        # 2. 将 GT 对齐到预测的分辨率 (64x64)
        target_size = skel_pred.shape[-2:]
        gt_skel_small = F.interpolate(gt_skel_full, size=target_size, mode='nearest')
        
        return self.bce(skel_pred, gt_skel_small)


def train_net(net, args, ema_net=None, fold_idx=0):

    ################################################################################
    # Dataset Creation
    trainset = get_dataset(args, mode='train', fold_idx=fold_idx)
    
    trainLoader = data.DataLoader(
        trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        pin_memory=(args.aug_device != 'gpu'), 
        num_workers=args.num_workers, 
        persistent_workers=(args.num_workers>0)
    )

    valset = get_dataset(args, mode='val', fold_idx=fold_idx)
    valLoader = data.DataLoader(valset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

    testset = get_dataset(args, mode='test', fold_idx=fold_idx)
    testLoader = data.DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)
    
    logging.info(f"Created Dataset and DataLoader")

    ################################################################################
    # Initialize tensorboard, optimizer and etc
    writer = SummaryWriter(f"{args.log_path}{args.unique_name}/fold_{fold_idx}")

    optimizer = get_optimizer(args, net)

    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weight).cuda().float())
    criterion_dl = DiceLoss()
    
    # --- [新增代码] 初始化拓扑损失 ---
    criterion_topo = TopoLoss().cuda()
    # -------------------------------

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    ################################################################################
    # Start training
    best_Dice = np.zeros(args.classes)
    best_IoU = np.zeros(args.classes)
    best_ACC = np.zeros(args.classes)
    best_SPE = np.zeros(args.classes)
    best_SEN = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000

    
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=5, max_epoch=args.epochs)
        logging.info(f"Current lr: {exp_scheduler:.4e}")
        
        # train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, scaler, args)
        # --- [修改传参] 将 criterion_topo 传入 train_epoch ---
        train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, criterion_topo, scaler, args)
        # ---------------------------------------------------

        ########################################################################################
        # Evaluation, save checkpoint and log training info
        net_for_eval = ema_net if args.ema else net 
        
        # # save the latest checkpoint, including net, ema_net, and optimizer
        # torch.save({
        #     'epoch': epoch+1,
        #     'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
        #     'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
        #     'optimizer_state_dict': optimizer.state_dict(),
        # }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_latest.pth")
   
        if (epoch+1) % args.val_freq == 0:
            # args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # save the latest checkpoint, including net, ema_net, and optimizer
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
                'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{args.cp_dir}/fold_{fold_idx}_latest.pth")

            dice_list_test, ASD_list_test, HD_list_test, IoU_list_test, ACC_list_test, SPE_list_test, SEN_list_test = validation(net_for_eval, valLoader, args)
            dice_list_test, ASD_list_test, HD_list_test = filter_validation_results(dice_list_test, ASD_list_test, HD_list_test, args) # filter results for some dataset, e.g. amos_mr
            log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, IoU_list_test, ACC_list_test, SPE_list_test, SEN_list_test, 'test', epoch, args)
            
            if dice_list_test.mean() >= best_Dice.mean():
                best_Dice = dice_list_test
                best_HD = HD_list_test
                best_ASD = ASD_list_test
                best_IoU = IoU_list_test
                best_ACC = ACC_list_test
                best_SPE = SPE_list_test
                best_SEN = SEN_list_test

                # Save the checkpoint with best performance
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
                    'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{args.cp_dir}/fold_{fold_idx}_best.pth")
            
            logging.info(f"Evaluation epoch:{epoch+1} Done")
            logging.info(f"Dice: {dice_list_test.mean():.4f}/Best Dice: {best_Dice.mean():.4f}, Best IoU:{best_IoU.mean():.4f}, Best ACC:{best_ACC.mean():.4f}")
            logging.info(f"Best SPE:{best_SPE.mean():.4f}, Best SEN:{best_SEN.mean():.4f}")
    
        writer.add_scalar('LR', exp_scheduler, epoch+1)
    
    # test best.pth
    best_pth = f"{args.cp_dir}/fold_{fold_idx}_best.pth"
    if os.path.exists(best_pth):
        checkpoint = torch.load(best_pth)
        # print(checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        if args.ema:
            ema_net.load_state_dict(checkpoint['ema_model_state_dict'])
        net_for_eval = ema_net if args.ema else net 
        # test_Dice, test_ASD, test_HD, test_IoU, test_ACC, test_SPE, test_SEN = validation(net_for_eval, testLoader, args, mode="Testing")
        best_Dice, best_ASD, best_HD, best_IoU, best_ACC, best_SPE, best_SEN = validation(net_for_eval, testLoader, args, mode="Testing")
        best_epoch = checkpoint['epoch']
        # logging.info(f"Testing epoch:{best_epoch} Done")
        # logging.info(f"Test Dice: {test_Dice.mean():.4f}, Test IoU:{test_IoU.mean():.4f}, Test ACC:{test_ACC.mean():.4f}")
        # logging.info(f"Test SPE:{test_SPE.mean():.4f}, Best SEN:{test_SEN.mean():.4f}")

        logging.info(f"Test epoch:{best_epoch} Done")
        logging.info(f"Best Dice: {best_Dice.mean():.4f}, Best IoU:{best_IoU.mean():.4f}, Best ACC:{best_ACC.mean():.4f}")
        logging.info(f"Best SPE:{best_SPE.mean():.4f}, Best SEN:{best_SEN.mean():.4f}")

    
    return best_Dice, best_HD, best_ASD, best_IoU, best_ACC, best_SPE, best_SEN


def train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, criterion_topo, scaler, args):
    batch_time = AverageMeter("Time", ":6.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")
    topo_loss_meter = AverageMeter("TopoLoss", ":.4f")

    progress = ProgressMeter(
        len(trainLoader) if args.dimension=='2d' else args.iter_per_epoch, 
        [batch_time, epoch_loss, topo_loss_meter], 
        prefix="Epoch: [{}]".format(epoch+1),
    )   
    
    net.train()
    tic = time.time()
    iter_num_per_epoch = 0 

    # --- [修改：梯度累积参数设置] ---
    # 由于 batch_size=2 显存限制，累积 4 步达到实际 batch_size=8 的效果
    accumulation_steps = 1
    optimizer.zero_grad() 

    for i, inputs in enumerate(trainLoader):
        img, label = inputs[0], inputs[1].long()
        if args.aug_device != 'gpu':
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        step = i + epoch * len(trainLoader)

        # --- [内部函数：计算总 Loss] ---
        def compute_total_loss(result, label, topo_weight=0.1):
            loss_seg = 0
            loss_topo = 0
            vis_dict = {}

            # 根据 Mediator 的输出结构：result[-1] 通常是 (skel_logits, topo_bias)
            if isinstance(result, (list, tuple)):
                seg_pred = result[0]
                # 这里假设 net 返回的是 (seg_out, (skel_logits, topo_bias))
                # 所以 result[-1][0] 是我们需要监督的 64x64 Logits
                topo_output = result[-1]
                skel_pred = topo_output[0] if isinstance(topo_output, (list, tuple)) else topo_output
                
                # 1. 主分割损耗 (Dice + CrossEntropy)
                loss_seg = criterion(seg_pred, label.squeeze(1)) + criterion_dl(seg_pred, label)
                
                # 2. 拓扑损耗 (使用我们分离出来的 64x64 分支)
                loss_topo = criterion_topo(skel_pred, label)

                vis_dict['seg_pred'] = seg_pred
                vis_dict['skel_pred'] = skel_pred
                vis_dict['skel_gt'] = criterion_topo.recent_skel_gt
            else:
                loss_seg = criterion(result, label.squeeze(1)) + criterion_dl(result, label)
                vis_dict['seg_pred'] = result

            total_loss = loss_seg + topo_weight * loss_topo
            return total_loss, loss_topo, vis_dict

        current_topo_weight = getattr(args, 'topo_weight', 0.1)

        # --- [核心修改：混合精度 + 梯度累积逻辑] ---
        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = net(img)
                loss, loss_t, vis_data = compute_total_loss(result, label, current_topo_weight)
                # 损失除以累积步数进行平均
                loss = loss / accumulation_steps 
            
            scaler.scale(loss).backward()

            # 每达到累积步数执行一次优化
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            result = net(img)
            loss, loss_t, vis_data = compute_total_loss(result, label, current_topo_weight)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # EMA 更新
        if args.ema:
            update_ema_variables(net, ema_net, args.ema_alpha, step)

        # 更新 Meter (记录原始 Loss 大小)
        epoch_loss.update(loss.item() * accumulation_steps, img.shape[0])
        topo_val = loss_t.item() if torch.is_tensor(loss_t) else loss_t
        topo_loss_meter.update(topo_val, img.shape[0])

        # --- [可视化逻辑：增强 64x64 骨架图展示] ---
        if i % args.print_freq == 0:
            progress.display(i)
            
            if 'skel_pred' in vis_data:
                fig, axs = plt.subplots(1, 5, figsize=(20, 4))
                
                # 1. 原图
                axs[0].imshow(img[0, 0].detach().cpu().numpy(), cmap='gray')
                axs[0].set_title('Image')
                
                # 2. 标签
                axs[1].imshow(label[0, 0].detach().cpu().numpy(), cmap='gray')
                axs[1].set_title('GT Mask')
                
                # 3. 分割预测
                pred_mask = torch.argmax(vis_data['seg_pred'], dim=1)[0].detach().cpu().numpy()
                axs[2].imshow(pred_mask, cmap='jet')
                axs[2].set_title('Pred Seg')
                
                # 4. 骨架 GT (由 TopoLoss 实时生成)
                axs[3].imshow(vis_data['skel_gt'][0, 0].detach().cpu().numpy(), cmap='bone')
                axs[3].set_title('Skel GT (512x512)')
                
                # 5. 拓扑分支预测 (64x64)
                # 经过 Sigmoid 映射到 0-1 概率空间，使用 magma 色图增强细微结构的可见度
                skel_prob = torch.sigmoid(vis_data['skel_pred'][0, 0]).detach().cpu().numpy()
                axs[4].imshow(skel_prob, cmap='magma', vmin=0, vmax=0.8) # 调低 vmax 让细血管更明显
                axs[4].set_title('Topo Pred (64x64)')

                for ax in axs: ax.axis('off')
                plt.tight_layout()
                
                writer.add_figure('Debug/Training_Visuals', fig, global_step=step)
                plt.close(fig) 

        batch_time.update(time.time() - tic)
        tic = time.time()
        
        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break

    # 每个 Epoch 结束记录平均值
    writer.add_scalar('Train/Loss', epoch_loss.avg, epoch+1)
    writer.add_scalar('Train/TopoLoss', topo_loss_meter.avg, epoch+1)

def get_parser():
    parser = argparse.ArgumentParser(description='CBIM Medical Image Segmentation')
    parser.add_argument('--dataset', type=str, default='acdc', help='dataset name')
    parser.add_argument('--model', type=str, default='unet', help='model name')
    parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
    parser.add_argument('--amp', action='store_true', help='if use the automatic mixed precision for faster training')
    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile, only supported by pytorch2.0')

    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./exp/', help='checkpoint path')
    parser.add_argument('--log_path', type=str, default='./log/', help='log path')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')
    
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--reproduce_seed', type=int, default=42)
    parser.add_argument('--save', action='store_true', help='save images')
    parser.add_argument('--save_path', type=str, default=None, help='save images path')
    parser.add_argument('--test_root', type=str, default=None, help='testset root dir')
    parser.add_argument('--topo_weight', type=float, default=0.5, help='weight for topological loss')
    
    
    args = parser.parse_args()

    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    print('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args
    


def init_network(args):
    net = get_model(args, pretrain=args.pretrain)

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        for p in ema_net.parameters():
            p.requires_grad_(False)
        logging.info("Use EMA model for evaluation")
    else:
        ema_net = None
    
    if args.resume:
        resume_load_model_checkpoint(net, ema_net, args)
    
    

    if args.torch_compile:
        net = torch.compile(net)
    return net, ema_net 

def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    torch.backends.benchmark = False
    torch.backends.deterministic = True
    # torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    args.log_path = args.log_path + '%s/'%args.dataset
    

    if args.reproduce_seed is not None:
        # random.seed(args.reproduce_seed)
        # np.random.seed(args.reproduce_seed)
        # torch.manual_seed(args.reproduce_seed)

        # if hasattr(torch, 'set_deterministic'):
        #     torch.set_deterministic(True)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        set_seed(args.reproduce_seed)
   
    Dice_list, HD_list, ASD_list, IoU_list, ACC_list, SPE_list, SEN_list = [], [], [], [], [], [], []

    for fold_idx in range(args.k_fold):
        
        args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(args.cp_dir, exist_ok=True)
        configure_logger(0, args.cp_dir+f"/fold_{fold_idx}.txt")
        save_configure(args)
        logging.info(
            f"\nDataset: {args.dataset},\n"
            + f"Model: {args.model},\n"
            + f"Dimension: {args.dimension}"
        )

        net, ema_net = init_network(args)

        net.cuda()
        if args.ema:
            ema_net.cuda()
        logging.info(f"Created Model")
        best_Dice, best_HD, best_ASD, best_IoU, best_ACC, best_SPE, best_SEN = train_net(net, args, ema_net, fold_idx=fold_idx)

        logging.info(f"Training and evaluation on Fold {fold_idx} is done")

        Dice_list.append(best_Dice)
        HD_list.append(best_HD)
        ASD_list.append(best_ASD)
        IoU_list.append(best_IoU)
        ACC_list.append(best_ACC)
        SPE_list.append(best_SPE)
        SEN_list.append(best_SEN)

    

    ############################################################################################3
    # Save the cross validation results
    total_Dice = np.vstack(Dice_list)
    total_HD = np.vstack(HD_list)
    total_ASD = np.vstack(ASD_list)
    total_IoU = np.vstack(IoU_list)
    total_ACC = np.vstack(ACC_list)
    total_SPE = np.vstack(SPE_list)
    total_SEN = np.vstack(SEN_list)
    

    with open(f"{args.cp_dir}/cross_validation.txt",  'w') as f:
        np.set_printoptions(precision=4, suppress=True) 
        f.write('Dice\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {Dice_list[i]}\n")
        f.write(f"Each Class Dice Avg: {np.mean(total_Dice, axis=0)}\n")
        f.write(f"Each Class Dice Std: {np.std(total_Dice, axis=0)}\n")
        f.write(f"All classes Dice Avg: {total_Dice.mean()}\n")
        f.write(f"All classes Dice Std: {np.mean(total_Dice, axis=1).std()}\n")

        f.write("\n")

        f.write('Iou\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {IoU_list[i]}\n")
        f.write(f"Each Class Iou Avg: {np.mean(total_IoU, axis=0)}\n")
        f.write(f"Each Class Iou Std: {np.std(total_IoU, axis=0)}\n")
        f.write(f"All classes Iou Avg: {total_IoU.mean()}\n")
        f.write(f"All classes Iou Std: {np.mean(total_IoU, axis=1).std()}\n")

        f.write("\n")

        f.write('ACC\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {ACC_list[i]}\n")
        f.write(f"Each Class ACC Avg: {np.mean(total_ACC, axis=0)}\n")
        f.write(f"Each Class ACC Std: {np.std(total_ACC, axis=0)}\n")
        f.write(f"All classes ACC Avg: {total_ACC.mean()}\n")
        f.write(f"All classes ACC Std: {np.mean(total_ACC, axis=1).std()}\n")

        f.write("\n")

        f.write('SPE\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {SPE_list[i]}\n")
        f.write(f"Each Class SPE Avg: {np.mean(total_SPE, axis=0)}\n")
        f.write(f"Each Class SPE Std: {np.std(total_SPE, axis=0)}\n")
        f.write(f"All classes SPE Avg: {total_SPE.mean()}\n")
        f.write(f"All classes SPE Std: {np.mean(total_SPE, axis=1).std()}\n")

        f.write("\n")
    
        f.write('SEN\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {SEN_list[i]}\n")
        f.write(f"Each Class SEN Avg: {np.mean(total_SEN, axis=0)}\n")
        f.write(f"Each Class SEN Std: {np.std(total_SEN, axis=0)}\n")
        f.write(f"All classes SEN Avg: {total_SEN.mean()}\n")
        f.write(f"All classes SEN Std: {np.mean(total_SEN, axis=1).std()}\n")

        f.write("\n")

        f.write("HD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {HD_list[i]}\n")
        f.write(f"Each Class HD Avg: {np.mean(total_HD, axis=0)}\n")
        f.write(f"Each Class HD Std: {np.std(total_HD, axis=0)}\n")
        f.write(f"All classes HD Avg: {total_HD.mean()}\n")
        f.write(f"All classes HD Std: {np.mean(total_HD, axis=1).std()}\n")

        f.write("\n")

        f.write("ASD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {ASD_list[i]}\n")
        f.write(f"Each Class ASD Avg: {np.mean(total_ASD, axis=0)}\n")
        f.write(f"Each Class ASD Std: {np.std(total_ASD, axis=0)}\n")
        f.write(f"All classes ASD Avg: {total_ASD.mean()}\n")
        f.write(f"All classes ASD Std: {np.mean(total_ASD, axis=1).std()}\n")




    print(f'All {args.k_fold} folds done.')

    sys.exit(0)
