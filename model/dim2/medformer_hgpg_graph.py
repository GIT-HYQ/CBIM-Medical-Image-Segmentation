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
from .FrangiFilter2d import FrangiFilter2d
from .vessel_graph import AdaptiveVesselGraphBottleneck

# =========================================================================
# Gated Modulation Module for Hierarchical Guidance
# =========================================================================
class GatedModulationModule(nn.Module):
    def __init__(self, in_channels, use_learnable_scale=True):
        """
        Args:
            in_channels: number of feature channels
            use_learnable_scale: whether to use learnable scaling factor
        """
        super().__init__()
        self.use_learnable_scale = use_learnable_scale
        
        # Transform geometric prior to channel-wise gates
        self.conv = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        # Learnable scaling factor to balance prior strength
        if use_learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = 1.0

    def forward(self, x, prior):
        """
        Args:
            x: feature maps [B, C, H, W]
            prior: geometric prior map [B, 1, H, W], normalized to [0, 1]
        Returns:
            modulated features [B, C, H, W]
        """
        # Generate channel-wise attention gates
        gate = self.conv(prior)
        
        # Apply learnable scaling
        if self.use_learnable_scale:
            gate = gate * self.scale
        
        # Residual gating: x * (1 + gate)
        # This ensures gradient flow even when gate → 0
        return x * (1 + gate)


# =========================================================================
# MedFormer with Hierarchical Geometric Prior Guidance (HGPG)
# =========================================================================
class MedFormerHGPG_Graph(nn.Module):
    def __init__(self, in_chan, num_classes, base_chan=32, map_size=8, conv_block='BasicBlock', 
                 conv_num=[2,1,0,0, 0,1,2,2], trans_num=[0,1,2,2, 2,1,0,0], 
                 num_heads=[1,4,8,16, 8,4,1,1], fusion_depth=2, fusion_dim=512, 
                 fusion_heads=16, expansion=4, attn_drop=0., proj_drop=0., 
                 proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.ReLU, aux_loss=False,
                 enable_guidance_lvl2=True, enable_guidance_lvl3=True, gated_hgm=False):
        super().__init__()
        
        # Geometry Analysis Engine
        # self.geometric_analyzer = FrangiSpectralFilter3(sigmas=(1, 2, 4))
        self.geometric_analyzer = FrangiFilter2d()
        self.enable_guidance_lvl2 = enable_guidance_lvl2
        self.enable_guidance_lvl3 = enable_guidance_lvl3
        self.gated_hgm = gated_hgm
        
        chan_num = [2*base_chan, 4*base_chan, 8*base_chan, 16*base_chan, 
                    8*base_chan, 4*base_chan, 2*base_chan, base_chan]
        dim_head = [chan_num[i]//num_heads[i] for i in range(8)]
        conv_block = get_block(conv_block)

        # 1. Early-stage Multi-modal Fusion (Input + Prior)
        self.inc = inconv(in_chan + 1, base_chan, norm=norm, act=act)
        
        # 2. Hierarchical Guidance Modules (HGM)
        self.initHGM(norm, chan_num[0], chan_num[1])

        # Standard Encoder Layers
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block, norm=norm, act=act, map_generate=False)
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block, heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block, heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block, heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        # 1. 实例化我们的融合模块
        # 假设 down4 输出的通道数是 base_chan * 16 (即 512)
        num_nodes = 512  # 查看onenote的解释
        self.bottleneck_channels = base_chan * 16 
        self.avg_bottleneck = AdaptiveVesselGraphBottleneck(
            in_channels=self.bottleneck_channels, 
            num_nodes=num_nodes
        )

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
    
    def initHGM(self, norm, chan_lvl2, chan_lvl3):            
        # Guidance at Level 2 (Mid-level representation)
        if self.enable_guidance_lvl2:
            if self.gated_hgm:
                self.hgm_lvl2 = GatedModulationModule(chan_lvl2, use_learnable_scale=True)
            else:
                self.hgm_lvl2 = nn.Sequential(
                    nn.Conv2d(1, chan_lvl2, kernel_size=1),
                    norm(chan_lvl2)
                )
                self.alpha_lvl2 = nn.Parameter(torch.zeros(1))
            
        # Guidance at Level 3 (High-level semantic representation)
        if self.enable_guidance_lvl3:
            if self.gated_hgm:
                self.hgm_lvl3 = GatedModulationModule(chan_lvl3, use_learnable_scale=True)
            else:
                self.hgm_lvl3 = nn.Sequential(
                    nn.Conv2d(1, chan_lvl3, kernel_size=1),
                    norm(chan_lvl3)
                )
                self.alpha_lvl3 = nn.Parameter(torch.zeros(1))

    def guidanceLevel2(self, geometric_prior, x1):
        # Level 2: Structural Modulation
        if self.enable_guidance_lvl2:
            if self.gated_hgm:
                prior_lvl2 = F.max_pool2d(geometric_prior, kernel_size=2) # 改用 MaxPool 保留强信号
                x1 = self.hgm_lvl2(x1, prior_lvl2)
            else:
                prior_lvl2 = F.avg_pool2d(geometric_prior, kernel_size=2)
                x1 = x1 + self.alpha_lvl2 * self.hgm_lvl2(prior_lvl2)
        return x1
    
    def guidanceLevel3(self, geometric_prior, x2):
        # Level 3: Semantic Modulation
        if self.enable_guidance_lvl3:
            if self.gated_hgm:
                prior_lvl3 = F.max_pool2d(geometric_prior, kernel_size=4) # 改用 MaxPool
                x2 = self.hgm_lvl3(x2, prior_lvl3)
            else:
                prior_lvl3 = F.avg_pool2d(geometric_prior, kernel_size=4)
                x2 = x2 + self.alpha_lvl3 * self.hgm_lvl3(prior_lvl3)
        return x2
    
    def adaptiveVesselGraph(self, x4, geometric_prior):
        # 1. 将先验池化到与 x4 相同的大小 (e.g., 32x32)
        prior_low = F.adaptive_max_pool2d(geometric_prior, x4.shape[-2:])
        
        # 2. 执行自适应图推理
        return self.avg_bottleneck(x4, prior_low)

    def forward(self, x):
        # I. Geometric Prior Extraction
        geometric_prior = self.geometric_analyzer(x) # [B, 1, H, W]
        
        # II. Encoder with Hierarchical Modulation
        # Level 0 & 1: Initial fusion through concatenation
        x0 = self.inc(torch.cat([x, geometric_prior], dim=1)) 
        x1, _ = self.down1(x0) 
        
        # Level 2: Structural Modulation
        x1 = self.guidanceLevel2(geometric_prior, x1)

        x2, map2 = self.down2(x1) 
        
        # Level 3: Semantic Modulation
        x2 = self.guidanceLevel3(geometric_prior, x2)

        x3, map3 = self.down3(x2) 
        
        x4, map4 = self.down4(x3)

        # --- 集成 AdaptiveVesselGraph ---
        x4 = self.adaptiveVesselGraph(x4, geometric_prior)
        
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