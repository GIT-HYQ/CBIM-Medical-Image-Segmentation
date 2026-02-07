import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import get_block
from .medformer_utils import down_block, up_block, inconv, SemanticMapFusion
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import get_block
from .medformer_utils import down_block, up_block, inconv, SemanticMapFusion

# =========================================================================
# Module: GPU-Accelerated Frangi Spectral Filter (FSF)
# Description: Extracts multi-scale tubular geometric priors via Hessian 
#              eigenanalysis to provide explicit morphological constraints.
# =========================================================================
class FrangiSpectralFilter(nn.Module):
    def __init__(self, sigmas=(1, 2, 4), beta=0.5, c=15, black_white=True):
        super().__init__()
        self.sigmas = sorted(sigmas)
        self.beta = 2 * (beta ** 2)
        self.c = 2 * (c ** 2)
        self.black_white = black_white
        
        # Pre-compute and register Hessian kernels for computational efficiency
        for sigma in self.sigmas:
            s_round = int(math.ceil(3 * sigma))
            r = torch.arange(-s_round, s_round + 1)
            y, x = torch.meshgrid(r, r, indexing='ij')
            
            sigma_sq = sigma**2
            g = torch.exp(-(x**2 + y**2) / (2 * sigma_sq)) / (2 * math.pi * sigma_sq)
            dxx = (x**2 / sigma_sq - 1) / sigma_sq * g
            dxy = (x * y) / (sigma_sq**2) * g
            dyy = (y**2 / sigma_sq - 1) / sigma_sq * g
            
            kernel = torch.stack([dxx, dxy, dyy], dim=0).unsqueeze(1)
            self.register_buffer(f'kernel_sigma_{sigma}', kernel.float())

    def _eigen_analysis(self, dxx, dxy, dyy):
        """Perform eigen-decomposition of the Hessian matrix."""
        tmp = torch.sqrt((dxx - dyy)**2 + 4 * dxy**2)
        mu1 = 0.5 * (dxx + dyy + tmp)
        mu2 = 0.5 * (dxx + dyy - tmp)
        # Reorder eigenvalues s.t. |l1| < |l2|
        check = torch.abs(mu1) > torch.abs(mu2)
        lambda1 = torch.where(check, mu2, mu1)
        lambda2 = torch.where(check, mu1, mu2)
        return lambda1, lambda2

    def forward(self, x):
        vesselness_scales = []
        for sigma in self.sigmas:
            kernel = getattr(self, f'kernel_sigma_{sigma}')
            padding = kernel.shape[-1] // 2
            h_out = F.conv2d(x, kernel, padding=padding)
            
            # Scale-normalized derivatives
            dxx, dxy, dyy = h_out[:, 0:1] * (sigma**2), h_out[:, 1:2] * (sigma**2), h_out[:, 2:3] * (sigma**2)
            
            l1, l2 = self._eigen_analysis(dxx, dxy, dyy)
            l1 = torch.where(l1 == 0, torch.tensor(1e-10, device=x.device), l1)
            
            rb = (l2 / l1)**2
            s2 = l1**2 + l2**2
            v = torch.exp(-rb / self.beta) * (1 - torch.exp(-s2 / self.c))
            
            # Bright vessel foreground constraint
            v = torch.where(l2 < 0 if self.black_white else l2 > 0, v, torch.zeros_like(v))
            vesselness_scales.append(v)
            
        # Max-pooling across scales to derive the Integrated Geometric Prior (IGP)
        integrated_prior, _ = torch.max(torch.cat(vesselness_scales, dim=1), dim=1, keepdim=True)
        return integrated_prior

# =========================================================================
# MedFormer with Hierarchical Geometric Prior Guidance (HGPG)
# =========================================================================
class MedFormerHGPG(nn.Module):
    def __init__(self, in_chan, num_classes, base_chan=32, map_size=8, conv_block='BasicBlock', 
                 conv_num=[2,1,0,0, 0,1,2,2], trans_num=[0,1,2,2, 2,1,0,0], 
                 num_heads=[1,4,8,16, 8,4,1,1], fusion_depth=2, fusion_dim=512, 
                 fusion_heads=16, expansion=4, attn_drop=0., proj_drop=0., 
                 proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.ReLU, aux_loss=False,
                 enable_guidance_lvl2=True, enable_guidance_lvl3=True):
        super().__init__()
        
        # Geometry Analysis Engine
        self.geometric_analyzer = FrangiSpectralFilter(sigmas=(1, 2, 4))
        self.enable_guidance_lvl2 = enable_guidance_lvl2
        self.enable_guidance_lvl3 = enable_guidance_lvl3
        
        chan_num = [2*base_chan, 4*base_chan, 8*base_chan, 16*base_chan, 
                    8*base_chan, 4*base_chan, 2*base_chan, base_chan]
        dim_head = [chan_num[i]//num_heads[i] for i in range(8)]
        conv_block = get_block(conv_block)

        # 1. Early-stage Multi-modal Fusion (Input + Prior)
        self.inc = inconv(in_chan + 1, base_chan, norm=norm, act=act)
        
        # 2. Hierarchical Guidance Modules (HGM)
        # Guidance at Level 2 (Mid-level representation)
        if self.enable_guidance_lvl2:
            self.hgm_lvl2 = nn.Sequential(
                nn.Conv2d(1, chan_num[0], kernel_size=1),
                norm(chan_num[0])
            )
            self.alpha_lvl2 = nn.Parameter(torch.zeros(1))
            
        # Guidance at Level 3 (High-level semantic representation)
        if self.enable_guidance_lvl3:
            self.hgm_lvl3 = nn.Sequential(
                nn.Conv2d(1, chan_num[1], kernel_size=1),
                norm(chan_num[1])
            )
            self.alpha_lvl3 = nn.Parameter(torch.zeros(1))

        # Standard Encoder Layers
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block, norm=norm, act=act, map_generate=False)
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block, heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block, heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block, heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.map_fusion = SemanticMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)

        # Decoder Path (Implicitly guided via skip connections)
        self.up1 = up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block, heads=num_heads[4], dim_head=dim_head[4], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)
        self.up2 = up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block, heads=num_heads[5], dim_head=dim_head[5], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)
        self.up3 = up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6], conv_block, norm=norm, act=act, map_shortcut=False)
        self.up4 = up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7], conv_block, norm=norm, act=act, map_shortcut=False)

        self.outc = nn.Conv2d(chan_num[7], num_classes, kernel_size=1)
        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_out = nn.Conv2d(chan_num[5], num_classes, kernel_size=1)

    def forward(self, x):
        # I. Geometric Prior Extraction
        geometric_prior = self.geometric_analyzer(x) # [B, 1, H, W]
        
        # II. Encoder with Hierarchical Modulation
        # Level 0 & 1: Initial fusion through concatenation
        x0 = self.inc(torch.cat([x, geometric_prior], dim=1)) 
        x1, _ = self.down1(x0) 
        
        # Level 2: Structural Modulation
        if self.enable_guidance_lvl2:
            prior_lvl2 = F.avg_pool2d(geometric_prior, kernel_size=2)
            x1 = x1 + self.alpha_lvl2 * self.hgm_lvl2(prior_lvl2)
        x2, map2 = self.down2(x1) 
        
        # Level 3: Semantic Modulation
        if self.enable_guidance_lvl3:
            prior_lvl3 = F.avg_pool2d(geometric_prior, kernel_size=4)
            x2 = x2 + self.alpha_lvl3 * self.hgm_lvl3(prior_lvl3)
        x3, map3 = self.down3(x2) 
        
        x4, map4 = self.down4(x3)
        
        # III. Decoder & Feature Reconstruction
        map_list = self.map_fusion([map2, map3, map4])
        out, semantic_map = self.up1(x4, x3, map_list[2], map_list[1])
        out, semantic_map = self.up2(out, x2, semantic_map, map_list[0])

        if self.aux_loss:
            aux_out = self.aux_out(out)
            aux_out = F.interpolate(aux_out, size=x.shape[-2:], mode='bilinear', align_corners=True)

        out, semantic_map = self.up3(out, x1, semantic_map, None)
        out, semantic_map = self.up4(out, x0, semantic_map, None)

        out = self.outc(out)
        
        return (out, aux_out) if self.training and self.aux_loss else (out, geometric_prior)