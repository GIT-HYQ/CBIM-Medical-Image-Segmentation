import torch
import torch.nn as nn
import torch.nn.functional as F
from . import metrics
import numpy as np
import pdb
from sklearn.metrics import confusion_matrix

def calculate_distance(label_pred, label_true, spacing, C, percentage=95):
    # the input args are torch tensors
    if label_pred.is_cuda:
        label_pred = label_pred.cpu()
        label_true = label_true.cpu()

    label_pred = label_pred.numpy()
    label_true = label_true.numpy()
    spacing = spacing.numpy()

    ASD_list = np.zeros(C-1)
    HD_list = np.zeros(C-1)

    for i in range(C-1):
        tmp_surface = metrics.compute_surface_distances(label_true==(i+1), label_pred==(i+1), spacing)
        dis_gt_to_pred, dis_pred_to_gt = metrics.compute_average_surface_distance(tmp_surface)
        ASD_list[i] = (dis_gt_to_pred + dis_pred_to_gt) / 2 

        HD = metrics.compute_robust_hausdorff(tmp_surface, percentage)
        HD_list[i] = HD

    return ASD_list, HD_list



def calculate_dice_split(pred, target, C, block_size=64*64*64):
    
    assert pred.shape[0] == target.shape[0]
    N = pred.shape[0]
    total_sum = torch.zeros(C).to(pred.device)
    total_intersection = torch.zeros(C).to(pred.device)
    
    split_num = N // block_size
    for i in range(split_num):
        dice, intersection, summ = calculate_dice(pred[i*block_size:(i+1)*block_size, :], target[i*block_size:(i+1)*block_size, :], C)
        total_intersection += intersection
        total_sum += summ
    if N % block_size != 0:
        dice, intersection, summ = calculate_dice(pred[(i+1)*block_size:, :], target[(i+1)*block_size:, :], C)
        total_intersection += intersection
        total_sum += summ

    dice = 2 * total_intersection / (total_sum + 1e-5)

    return dice, total_intersection, total_sum


        
        





def calculate_dice(pred, target, C): 
    # pred and target are torch tensor
    target = target.long()
    pred = pred.long()
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.) 

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.) 

    intersection= pred_mask * target_mask
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)
    summ = summ.sum(0).type(torch.float32)
    
    summ += 1e-5 
    dice = 2 * intersection / summ

    return dice, intersection, summ


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    # corr = torch.sum(SR == GT, dim=0)
    # tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    tensor_size = SR.numel()
    acc = float(corr)/float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    # SE = float(torch.sum(TP, dim=0))/(float(torch.sum(TP+FN, dim=0)) + 1e-6)
    return SE


def get_specificity(SR, GT, threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    # SP = float(torch.sum(TN, dim=0))/(float(torch.sum(TN+FP, dim=0)) + 1e-6)
    return SP


def get_precision(SR, GT, threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT== torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    # PC = float(torch.sum(TP, dim=0))/(float(torch.sum(TP+FP, dim=0)) + 1e-6)
    return PC

def calculate_iou_multiclass(pred, target, C):
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    N = pred.shape[0]

    ious = []
    f1s = []
    sensitivities = []
    specificities = []

    for c in range(1, C):
        pred_c = (pred == c)
        target_c = (target == c)
        TP = (pred_c & target_c).sum().item()
        FP = (pred_c & (~target_c)).sum().item()
        FN = ((~pred_c) & target_c).sum().item()
        TN = ((~pred_c) & (~target_c)).sum().item()

        union = TP + FP + FN
        iou = TP / union if union != 0 else 0
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

        ious.append(iou)
        f1s.append(f1)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    miou = np.mean(ious)
    f1_or_dsc = np.mean(f1s)
    accuracy = (pred == target).sum().item() / N
    specificity = np.mean(specificities)
    sensitivity = np.mean(sensitivities)

    return miou, f1_or_dsc, accuracy, specificity, sensitivity

def calculate_iou(pred, target, C): 
    # pred and target are torch tensor
    target = target.long()
    pred = pred.long()
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.) 

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.) 


    pred_mask_ = torch.tensor(pred_mask)[:,1]
    target_mask_ = torch.tensor(target_mask)[:,1]


    if target_mask.is_cuda:
        target_mask_1 = np.array(target_mask_.cpu()).reshape(-1)
        pred_mask_1 = np.array(pred_mask_.cpu()).reshape(-1)
    confusion = confusion_matrix(target_mask_1, pred_mask_1)

    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    return miou, f1_or_dsc, accuracy, specificity, sensitivity

