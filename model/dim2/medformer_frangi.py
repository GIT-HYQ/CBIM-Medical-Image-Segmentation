import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import get_block
from .medformer_utils import down_block, up_block, inconv, SemanticMapFusion
import pdb

# ==========================================
# 新增：优化后的 GPU 版 Frangi 提取器模块
# ==========================================
class OptimizedFrangiFilter(nn.Module):
    def __init__(self, sigmas=(1, 2, 4), beta=0.5, c=15, black_white=True):
        super().__init__()
        self.sigmas = sorted(sigmas)
        self.beta = 2 * (beta ** 2)
        self.c = 2 * (c ** 2)
        self.black_white = black_white
        
        # 预计算所有尺度的 Hessian 卷积核并注册为 buffer
        for sigma in self.sigmas:
            s_round = int(math.ceil(3 * sigma))
            r = torch.arange(-s_round, s_round + 1)
            # 显式指定 indexing 避免警告
            y, x = torch.meshgrid(r, r, indexing='ij')
            
            sigma_sq = sigma**2
            g = torch.exp(-(x**2 + y**2) / (2 * sigma_sq)) / (2 * math.pi * sigma_sq)
            
            # 高斯二阶导核
            dxx = (x**2 / sigma_sq - 1) / sigma_sq * g
            dxy = (x * y) / (sigma_sq**2) * g
            dyy = (y**2 / sigma_sq - 1) / sigma_sq * g
            
            kernel = torch.stack([dxx, dxy, dyy], dim=0).unsqueeze(1)
            self.register_buffer(f'kernel_sigma_{sigma}', kernel.float())

    def _eig2image(self, dxx, dxy, dyy):
        # 判别式计算特征值
        tmp = torch.sqrt((dxx - dyy)**2 + 4 * dxy**2)
        mu1 = 0.5 * (dxx + dyy + tmp)
        mu2 = 0.5 * (dxx + dyy - tmp)
        
        # 排序使得 |l1| < |l2| (l2是主曲率)
        check = torch.abs(mu1) > torch.abs(mu2)
        lambda1 = torch.where(check, mu2, mu1)
        lambda2 = torch.where(check, mu1, mu2)
        return lambda1, lambda2

    def forward(self, x):
        all_scales = []
        for sigma in self.sigmas:
            kernel = getattr(self, f'kernel_sigma_{sigma}')
            padding = kernel.shape[-1] // 2
            h_out = F.conv2d(x, kernel, padding=padding)
            
            dxx, dxy, dyy = h_out[:, 0:1], h_out[:, 1:2], h_out[:, 2:3]
            # 尺度归一化
            dxx, dxy, dyy = dxx * (sigma**2), dxy * (sigma**2), dyy * (sigma**2)
            
            l1, l2 = self._eig2image(dxx, dxy, dyy)
            l1 = torch.where(l1 == 0, torch.tensor(1e-10, device=x.device), l1)
            
            rb = (l2 / l1)**2
            s2 = l1**2 + l2**2
            vesselness = torch.exp(-rb / self.beta) * (1 - torch.exp(-s2 / self.c))
            
            # 针对亮/暗结构过滤
            if self.black_white:
                vesselness = torch.where(l2 < 0, vesselness, torch.zeros_like(vesselness))
            else:
                vesselness = torch.where(l2 > 0, vesselness, torch.zeros_like(vesselness))
            
            all_scales.append(vesselness)

        multi_scale_feat = torch.cat(all_scales, dim=1)
        max_vesselness, _ = torch.max(multi_scale_feat, dim=1, keepdim=True)
        return multi_scale_feat, max_vesselness


# ==========================================
# 修改后的 MedFormer 主类
# ==========================================
class MedFormerFrangi(nn.Module):
    def __init__(self, in_chan, num_classes, base_chan=32, map_size=8, conv_block='BasicBlock', 
                 conv_num=[2,1,0,0, 0,1,2,2], trans_num=[0,1,2,2, 2,1,0,0], 
                 num_heads=[1,4,8,16, 8,4,1,1], fusion_depth=2, fusion_dim=512, 
                 fusion_heads=16, expansion=4, attn_drop=0., proj_drop=0., 
                 proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.ReLU, aux_loss=False):
        super().__init__()
        
        # 1. 初始化 Frangi 几何增强模块
        # sigmas=(1, 2, 4) 覆盖了从细微分支到较粗主干的范围
        self.frangi_pre = OptimizedFrangiFilter(sigmas=(1, 2, 4))
        
        # 2. 方案一：初始化空间注意力门控 (Gate)
        self.frangi_gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        chan_num = [2*base_chan, 4*base_chan, 8*base_chan, 16*base_chan, 
                        8*base_chan, 4*base_chan, 2*base_chan, base_chan]
        dim_head = [chan_num[i]//num_heads[i] for i in range(8)]
        conv_block = get_block(conv_block)

        # 3. 方案二：特征注入
        # 修改 self.inc 的输入通道：in_chan (原图) + 3 (三个尺度的 Frangi 响应)
        self.inc = inconv(in_chan + 3, base_chan, norm=norm, act=act)
        
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block, norm=norm, act=act, map_generate=False)
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block, heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block, heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block, heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.map_fusion = SemanticMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)

        self.up1 = up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block, heads=num_heads[4], dim_head=dim_head[4], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)
        self.up2 = up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block, heads=num_heads[5], dim_head=dim_head[5], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)
         
        self.up3 = up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6], conv_block, norm=norm, act=act, map_shortcut=False)
        self.up4 = up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7], conv_block, norm=norm, act=act, map_shortcut=False)

        self.outc = nn.Conv2d(chan_num[7], num_classes, kernel_size=1)

        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_out = nn.Conv2d(chan_num[5], num_classes, kernel_size=1)

    def forward(self, x):
        # --- 几何先验提取 ---
        # multi_f: 多尺度特征注入 [B, 3, H, W]
        # max_v: 空间注意力掩码 [B, 1, H, W]
        multi_f, max_v = self.frangi_pre(x)
        
        # --- 方案二：输入侧特征拼接 ---
        x_with_geo = torch.cat([x, multi_f], dim=1)
        x0 = self.inc(x_with_geo)
        
        # --- 编码器 (Encoder) ---
        x1, _ = self.down1(x0)
        x2, map2 = self.down2(x1)
        x3, map3 = self.down3(x2)
        x4, map4 = self.down4(x3)
        
        map_list = [map2, map3, map4]
        map_list = self.map_fusion(map_list)
        
        # --- 方案一：Skip Connection 注意力引导函数 ---
        # 确保注意力图尺度与特征图尺度匹配
        def gate_feature(feat, mask):
            gate_mask = F.interpolate(mask, size=feat.shape[2:], mode='bilinear', align_corners=True)
            gate_weight = self.frangi_gate(gate_mask)
            return feat * gate_weight

        # --- 解码器 (Decoder) + 几何注意力引导 ---
        # 对每一个来自 Encoder 的 Skip 信号应用 Frangi Gate，过滤非管状背景噪声
        out, semantic_map = self.up1(x4, gate_feature(x3, max_v), map_list[2], map_list[1])
        out, semantic_map = self.up2(out, gate_feature(x2, max_v), semantic_map, map_list[0])

        if self.aux_loss:
            aux_out = self.aux_out(out)
            aux_out = F.interpolate(aux_out, size=x.shape[-2:], mode='bilinear', align_corners=True)

        out, semantic_map = self.up3(out, gate_feature(x1, max_v), semantic_map, None)
        out, semantic_map = self.up4(out, gate_feature(x0, max_v), semantic_map, None)

        out = self.outc(out)

        if self.aux_loss:
            return [out, aux_out]
        else:
            return out