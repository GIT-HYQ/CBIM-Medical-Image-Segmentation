import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_block
from .medformer_utils import down_block, up_block, inconv, SemanticMapFusion
import pdb


class TopologicalMediator(nn.Module):
    def __init__(self, in_channels, out_channels, map_size=8):
        """
        in_channels: Encoder 最深层特征通道数
        out_channels: map4 的通道数
        map_size: 语义图的大小 (必须与 MedFormer down_block 中的 map_size 一致)
        """
        super().__init__()
        
        # 1. 骨架预测分支
        self.skel_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid() 
        )
        
        # 2. 拓扑注入投影
        self.topo_proj = nn.Conv2d(1, out_channels, kernel_size=1)
        
        # 3. 降采样
        # 注意：这里会强制将尺寸变为 map_size x map_size
        self.pool = nn.AdaptiveAvgPool2d((map_size, map_size))

    def forward(self, x):
        # x: [B, C, H, W]
        
        # 1. 生成骨架图 (用于 Loss)
        skel_pred = self.skel_head(x) 
        
        # 2. 生成拓扑偏置
        topo_feat = self.pool(skel_pred) # [B, 1, M, M]
        topo_feat = self.topo_proj(topo_feat) # [B, Out_C, M, M]
        
        # --- [关键修改] ---
        # 不要 Flatten！保持与 map4 一样的 [B, C, M, M] 形状
        topo_bias = topo_feat 
        # -----------------
        
        return skel_pred, topo_bias


class MedFormerV2(nn.Module):

    def __init__(self, in_chan, num_classes, base_chan=32, map_size=8, conv_block='BasicBlock', conv_num=[2,1,0,0, 0,1,2,2], trans_num=[0,1,2,2, 2,1,0,0], num_heads=[1,4,8,16, 8,4,1,1], fusion_depth=2, fusion_dim=512, fusion_heads=16, expansion=4, attn_drop=0., proj_drop=0., proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.ReLU, aux_loss=False):
        super().__init__()
        
        chan_num = [2*base_chan, 4*base_chan, 8*base_chan, 16*base_chan, 
                        8*base_chan, 4*base_chan, 2*base_chan, base_chan]
        dim_head = [chan_num[i]//num_heads[i] for i in range(8)]
        conv_block = get_block(conv_block)

        # self.inc and self.down1 forms the conv stem
        self.inc = inconv(in_chan, base_chan, norm=norm, act=act)
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block, norm=norm, act=act, map_generate=False)
        
        # down2 down3 down4 apply the B-MHA blocks
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block, heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block, heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block, heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        # --- [新增代码] 初始化拓扑中介 ---
        # chan_num[3] 是 x4 的通道数 (Encoder最深层)
        # 我们假设 map4 的通道数与特征通道数一致 (chan_num[3])
        self.topo_mediator = TopologicalMediator(
            in_channels=chan_num[3], 
            out_channels=chan_num[3], 
            map_size=map_size
        )
        # -------------------------------
        
        self.map_fusion = SemanticMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)

        self.up1 = up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block, heads=num_heads[4], dim_head=dim_head[4], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)
        self.up2 = up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block, heads=num_heads[5], dim_head=dim_head[5], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)
         
         # up3 up4 form the conv decoder
        self.up3 = up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6], conv_block, norm=norm, act=act, map_shortcut=False)
        self.up4 = up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7], conv_block, norm=norm, act=act, map_shortcut=False)
        
        self.outc = nn.Conv2d(chan_num[7], num_classes, kernel_size=1)

        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_out = nn.Conv2d(chan_num[5], num_classes, kernel_size=1)

    def forward(self, x):
        
        x0 = self.inc(x)
        x1, _ = self.down1(x0)
        x2, map2 = self.down2(x1)
        x3, map3 = self.down3(x2)
        x4, map4 = self.down4(x3) # x4 是 Bottleneck 特征, map4 是最深层的语义图
        
        # --- [新增代码] 拓扑注入 ---
        # 1. 计算拓扑
        skel_pred, topo_bias = self.topo_mediator(x4)
        
        # 2. 注入到 map4 中
        # map4 的形状通常是 [B, C, N], topo_bias 也是 [B, C, N]
        # 如果维度不匹配 (例如 map4 是 [B, N, C])，可能需要 permute，但根据 MedFormer 惯例这里通常是匹配的
        map4 = map4 + topo_bias 
        # ------------------------
        
        map_list = [map2, map3, map4]
        map_list = self.map_fusion(map_list)
        
        out, semantic_map = self.up1(x4, x3, map_list[2], map_list[1])
        out, semantic_map = self.up2(out, x2, semantic_map, map_list[0])

        if self.aux_loss:
            aux_out = self.aux_out(out)
            aux_out = F.interpolate(aux_out, size=x.shape[-2:], mode='bilinear', align_corners=True)

        out, semantic_map = self.up3(out, x1, semantic_map, None)
        out, semantic_map = self.up4(out, x0, semantic_map, None)

        out = self.outc(out)

        # 如果有 aux_loss，返回 [out, aux, skel]
        if self.aux_loss:
            # --- [修改返回值] 训练模式下返回骨架图用于 Loss ---
            if self.training:
                return [out, aux_out, skel_pred]
            else:
                return [out, aux_out]
        else:
            # --- [修改返回值] 训练模式下返回骨架图用于 Loss ---
            if self.training:
                return [out, skel_pred] # 返回 list 以便区分
            else:
                return out