import logging
import os
import random

import torch
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from torch.utils import data

from training.validation import validation

import yaml
import argparse
import sys
import warnings

from utils import (
    configure_logger,
    save_configure,
)

warnings.filterwarnings("ignore", category=UserWarning)
from datetime import datetime


def test_net(net, args, ema_net=None, fold_idx=0):

    ################################################################################
    # Dataset Creation
    testset = get_dataset(args, mode='val', fold_idx=fold_idx)
    testLoader = data.DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)
    
    logging.info(f"Created Dataset and DataLoader")

    # Start testing
    best_Dice = np.zeros(args.classes)
    best_IoU = np.zeros(args.classes)
    best_ACC = np.zeros(args.classes)
    best_SPE = np.zeros(args.classes)
    best_SEN = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000

    
    checkpoint = torch.load(args.load)
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
        best_Dice, best_HD, best_ASD, best_IoU, best_ACC, best_SPE, best_SEN = test_net(net, args, ema_net, fold_idx=fold_idx)

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
    

    with open(f"{args.cp_dir}/test.txt",  'w') as f:
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
