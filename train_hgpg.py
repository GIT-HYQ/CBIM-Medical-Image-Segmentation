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
from training.losses import DiceLoss, CombinedGeometricLoss
from training.validation2 import validation
from training.utils import (
    exp_lr_scheduler_with_warmup, 
    log_evaluation_result_dict, 
    get_optimizer, 
    filter_validation_results_dict
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


def print_erformance_dict(perf):
    """
    格式化打印最佳性能字典
    """
    if perf is None:
        logging.info("No best performance recorded yet.")
        return

    logging.info("\n" + "="*80)
    logging.info(f"{'Metric':<15} | {'Mean Value':<15} | {'Class-wise Detail'}")
    logging.info("-" * 80)

    for metric, values in perf.items():
        # values 是一个 shape 为 (C,) 的 numpy 数组
        mean_val = values.mean()
        # 将每个类别的得分转为字符串，方便展示
        if len(values) > 1:
            detail_str = ", ".join([f"C{i+1}: {v:.4f}" for i, v in enumerate(values)])
        else:
            detail_str = f"{values[0]:.4f}"
            
        logging.info(f"{metric:<15} | {mean_val:<15.4f} | {detail_str}")

    logging.info("="*80 + "\n")

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
    # criterion_dl = DiceLoss()
    criterion_dl = CombinedGeometricLoss() # 增强连通性
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    ################################################################################
    # Start training
    best_perf = None  # 用于存储最佳模型发生时的所有指标字典
    best_epoch = 0

    
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=5, max_epoch=args.epochs)
        logging.info(f"Current lr: {exp_scheduler:.4e}")
        
        train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, scaler, args)
        
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

            perf = validation(net_for_eval, valLoader, args, mode='Evaluating', writer=writer, epoch=epoch+1)
            perf = filter_validation_results_dict(perf, args) # filter results for some dataset, e.g. amos_mr
            log_evaluation_result_dict(writer, perf, 'Val', epoch, args)

            # 3. 综合评分判定最佳模型 (对于血管分割，Dice 和 clDice 同等重要)
            current_Dice = perf['Dice'].mean()
            # current_clDice = perf['clDice'].mean()
            # current_score = (current_Dice + current_clDice) / 2
            best_Dice = best_perf['Dice'].mean() if best_perf else 0
            if current_Dice >= best_Dice:
                best_perf = perf
                best_epoch = epoch+1
                # Save the checkpoint with best performance
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
                    'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{args.cp_dir}/fold_{fold_idx}_best.pth")
            
            logging.info(f"Evaluation epoch:{epoch+1} Done best epoch:{best_epoch}")
            print_erformance_dict(best_perf)
    
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
        test_perf = validation(net_for_eval, testLoader, args, mode='Testing', writer=writer, epoch=epoch+1)
        best_epoch = checkpoint['epoch']

        logging.info(f"Test best epoch:{best_epoch}")
        print_erformance_dict(test_perf)

    return test_perf


def train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, scaler, args):
    batch_time = AverageMeter("Time", ":6.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")
    progress = ProgressMeter(
        len(trainLoader) if args.dimension=='2d' else args.iter_per_epoch, 
        [batch_time, epoch_loss], 
        prefix="Epoch: [{}]".format(epoch+1),
    )   
    
    net.train()

    tic = time.time()
    iter_num_per_epoch = 0 
    for i, inputs in enumerate(trainLoader):
        img, label = inputs[0], inputs[1].long()
        if args.aug_device != 'gpu':
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

    
        # uncomment this for visualize the input images and labels for debug
        '''
        img = img.cpu()
        print(img.mean())
        label = label.cpu()
        for idx in range(img.shape[0]):
            plt.subplot(3,2,1)
            plt.imshow(img[idx, 0, 64, :, :].numpy())
            plt.subplot(3,2,2)
            plt.imshow(label[idx, 0, 64, :, :].numpy())
            
            plt.subplot(3,2,3)
            plt.imshow(img[idx, 0, :, 64, :].cpu().numpy())
            plt.subplot(3,2,4)
            plt.imshow(label[idx, 0, :, 64, :].numpy())
            
            plt.subplot(3,2,5)
            plt.imshow(img[idx, 0, :, :, 64].cpu().numpy())
            plt.subplot(3,2,6)
            plt.imshow(label[idx, 0, :, :, 64].numpy())
           

            plt.savefig('./result/PtranslateX_idx%d.png'%idx)

            #plt.show()
        '''
        step = i + epoch * len(trainLoader) # global steps
    
        optimizer.zero_grad()
        
        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = net(img)

                loss = 0

                if isinstance(result, tuple) or isinstance(result, list):
                    # if use deep supervision, add all loss together
                    for j in range(len(result)):
                        loss += args.aux_weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
                else:
                    loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            result = net(img)
            # print(result.shape, label.shape, label.squeeze(1).shape)
            loss = 0 
            if isinstance(result, tuple) or isinstance(result, list):
                # If use deep supervision, add all loss together 
                for j in range(len(result)):
                    loss += args.aux_weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
            else:
                loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)


            loss.backward()
            optimizer.step()
        if args.ema:
            update_ema_variables(net, ema_net, args.ema_alpha, step)

        epoch_loss.update(loss.item(), img.shape[0])
        batch_time.update(time.time() - tic)
        tic = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)

        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break


        writer.add_scalar('Train/Loss', epoch_loss.avg, epoch+1)



 


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
    parser.add_argument('--guidance_l2', action='store_true', default=False, help='enable guidance level 2')
    parser.add_argument('--guidance_l3', action='store_true', default=False, help='enable guidance level 3')
    parser.add_argument('--gated_hgm', action='store_true', default=False, help='enable gated hgm')

    
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

    # 初始化存储所有 fold 结果的字典
    all_folds_metrics = {
        'Dice': [], 'clDice': [], 'HD': [], 'ASD': [], 
        'IoU': [], 'ACC': [], 'SPE': [], 'SEN': []
    }

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
        test_perf = train_net(net, args, ema_net, fold_idx=fold_idx)

        logging.info(f"Training and evaluation on Fold {fold_idx} is done")

        # --- 核心修复：正确读取 test_perf 中的各项指标 ---
        for key in all_folds_metrics.keys():
            all_folds_metrics[key].append(test_perf[key])    

    ############################################################################################
    # 保存测试结果 (Test Summary)
    ############################################################################################
    
    summary_path = f"{args.cp_dir}/test_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Test Results ({args.k_fold} folds)\n")
        f.write(f"Model: {args.model} | Dataset: {args.dataset}\n")
        f.write("="*60 + "\n\n")

        np.set_printoptions(precision=4, suppress=True)

        # 遍历所有指标进行统一统计
        for metric_name, data_list in all_folds_metrics.items():
            # 将列表转换为 numpy 矩阵 [k_fold, classes-1]
            total_data = np.vstack(data_list)
            
            f.write(f"--- {metric_name} ---\n")
            
            # 记录每一个 fold 的结果
            for i in range(args.k_fold):
                f.write(f"Fold {i}: {data_list[i]}\n")
            
            # 计算统计量
            class_avg = np.mean(total_data, axis=0)
            class_std = np.std(total_data, axis=0)
            overall_avg = total_data.mean()
            # 计算 Fold 间的波动（SCI论文常用：先求各fold均值，再求均值的标准差）
            fold_wise_avg = np.mean(total_data, axis=1)
            overall_std = fold_wise_avg.std()

            f.write(f"Each Class {metric_name} Avg: {class_avg}\n")
            f.write(f"Each Class {metric_name} Std: {class_std}\n")
            f.write(f"All classes {metric_name} Avg: {overall_avg:.4f}\n")
            f.write(f"All classes {metric_name} Std (across folds): {overall_std:.4f}\n")
            f.write("\n")

    print(f'All {args.k_fold} folds done. Results saved to {summary_path}')

    sys.exit(0)
