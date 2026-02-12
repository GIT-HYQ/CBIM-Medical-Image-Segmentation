import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb 

class DiceLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets):
        N = preds.size(0)
        C = preds.size(1)
        

        P = F.softmax(preds, dim=1)
        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)

        class_mask = torch.zeros(preds.shape).to(preds.device)
        class_mask.scatter_(1, targets, 1.) 

        ones = torch.ones(preds.shape).to(preds.device)
        P_ = ones - P 
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        self.alpha = FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) / ((FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) + FN.transpose(0, 1).reshape(C, -1).sum(dim=(1))) + smooth)
    
        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8) 
        #print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP.transpose(0, 1).reshape(C, -1), dim=(1)).float()
        den = num + self.alpha * torch.sum(FP.transpose(0, 1).reshape(C, -1), dim=(1)).float() + self.beta * torch.sum(FN.transpose(0, 1).reshape(C, -1), dim=(1)).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss

        loss = 1 - dice
        loss = loss.sum()

        if self.size_average:
            loss /= C

        return loss

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        if alpha is None:
            self.alpha = torch.ones(class_num)
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.size_average = size_average

    def forward(self, preds, targets):
        N = preds.size(0)
        C = preds.size(1)

        targets = targets.unsqueeze(1)
        P = F.softmax(preds, dim=1)
        log_P = F.log_softmax(preds, dim=1)

        class_mask = torch.zeros(preds.shape).to(preds.device)
        class_mask.scatter_(1, targets, 1.)
        
        if targets.size(1) == 1:
            # squeeze the chaneel for target
            targets = targets.squeeze(1)
        alpha = self.alpha[targets.data].to(preds.device)

        probs = (P * class_mask).sum(1)
        log_probs = (log_P * class_mask).sum(1)
        
        batch_loss = -alpha * (1-probs).pow(self.gamma)*log_probs

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSkeletonTopologyLoss(nn.Module):
    """
    Soft-clDice Loss: 拓扑感知损失函数
    适用于管状结构（血管、道路、神经纤维）的细长分支提取。
    通过形态学池化（Morphological Pooling）模拟骨架化过程。
    """
    def __init__(self, iter=3, smooth=1e-5, exclude_background=True):
        super(SoftSkeletonTopologyLoss, self).__init__()
        self.iter = iter
        self.smooth = smooth
        self.exclude_background = exclude_background

    def soft_dilation(self, x):
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def soft_erosion(self, x):
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

    def get_soft_skeleton(self, x):
        """
        通过迭代腐蚀和膨胀的残差来提取软骨架。
        """
        for _ in range(self.iter):
            erosion = self.soft_erosion(x)
            dilation = self.soft_dilation(erosion)
            # 提取结构残留：原始图像与开运算结果的差值
            skeleton = F.relu(x - dilation)
            x = erosion
        return skeleton

    def forward(self, preds, targets):
        """
        preds: 模型输出的 Logits [B, C, H, W]
        targets: 标签索引 [B, 1, H, W]
        """
        # 1. 转换为概率空间
        if preds.shape[1] > 1:
            probs = F.softmax(preds, dim=1)[:, 1:2, :, :] # 取血管通道
        else:
            probs = torch.sigmoid(preds)
            
        targets = targets.float()

        # 2. 提取预测图和真实标签的软骨架
        skel_pred = self.get_soft_skeleton(probs)
        skel_true = self.get_soft_skeleton(targets)

        # 3. 计算 Tprec (Topological Precision) 和 Tsens (Topological Sensitivity)
        # Tprec: 预测骨架在真实掩码上的覆盖率
        tprec = (torch.sum(skel_pred * targets) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        # Tsens: 真实骨架在预测掩码上的覆盖率
        tsens = (torch.sum(skel_true * probs) + self.smooth) / (torch.sum(skel_true) + self.smooth)

        # 4. 计算 clDice
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)
        
        return 1.0 - cl_dice

# --- 用于 SCI 论文展示的组合损失函数 ---
class CombinedGeometricLoss(nn.Module):
    def __init__(self, dice_weight=1.0, topo_weight=0.5, iter=3):
        super().__init__()
        self.dice_loss = DiceLoss() # 你原有的 DiceLoss
        self.topo_loss = SoftSkeletonTopologyLoss(iter=iter)
        self.dice_weight = dice_weight
        self.topo_weight = topo_weight

    def forward(self, preds, targets):
        l_dice = self.dice_loss(preds, targets)
        l_topo = self.topo_loss(preds, targets)
        
        # 建议动态平衡：前期靠 Dice 确定位置，后期靠 Topo 优化连通性
        return self.dice_weight * l_dice + self.topo_weight * l_topo


if __name__ == '__main__':
    
    DL = DiceLoss()
    FL = FocalLoss(10)
    
    pred = torch.randn(2, 10, 128, 128)
    target = torch.zeros((2, 1, 128, 128)).long()

    dl_loss = DL(pred, target)
    fl_loss = FL(pred, target)

    print('2D:', dl_loss.item(), fl_loss.item())

    pred = torch.randn(2, 10, 64, 128, 128)
    target = torch.zeros(2, 1, 64, 128, 128).long()

    dl_loss = DL(pred, target)
    fl_loss = FL(pred, target)

    print('3D:', dl_loss.item(), fl_loss.item())

    
