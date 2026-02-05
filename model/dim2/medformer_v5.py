import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_block
from .medformer_utils import down_block, up_block, inconv, SemanticMapFusion


class TopologicalMediator(nn.Module):
    def __init__(self, in_channels, out_channels, topo_size=64, original_map_size=3):
        super().__init__()
        self.topo_size = topo_size
        self.original_map_size = original_map_size

        # 骨架预测头：在 64x64 尺度上提取拓扑特征
        # 增加了一层卷积以增强从 16x16 上采样后的特征表达能力
        self.skel_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        )
        
        # 拓扑特征投影：将 1 通道的预测映射回主干维度
        self.topo_proj = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x 为 Stage 4 输出 [B, C, 16, 16]
        
        # 1. 显式上采样：从 16x16 提升到 64x64，为 TopoLoss 提供高精视野
        x_up = F.interpolate(x, size=(self.topo_size, self.topo_size), 
                             mode='bilinear', align_corners=True)
        
        # 2. 预测骨架 Logits (输出 [B, 1, 64, 64]，用于 TopoLoss)
        skel_logits = self.skel_head(x_up) 
        
        # 3. 产生用于主干融合的 3x3 偏置 (兼容原有 map_size=3)
        # 通过 Sigmoid 增强特征稳定性，再通过池化压回 3x3
        topo_prob = torch.sigmoid(skel_logits)
        topo_bias_small = F.adaptive_avg_pool2d(topo_prob, (self.original_map_size, self.original_map_size))
        topo_bias = self.topo_proj(topo_bias_small)
        
        return skel_logits, topo_bias


class MedFormerV5(nn.Module):

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

        # --- [修改 2] 初始化 Mediator 和 强融合层 ---
        # 这里的 chan_num[3] 对应 x4 的通道数 (例如 512)
        self.topo_mediator = TopologicalMediator(
            in_channels=chan_num[3], 
            out_channels=chan_num[3], 
            original_map_size=map_size # 传入 MedFormer 的 map_size 配置
        )
        
        # 强融合层：Concat(map4, topo) -> Conv -> map4
        # 输入通道是 2倍 (map4原本的 + topo原本的)
        self.topo_fusion_conv = nn.Sequential(
            nn.Conv2d(chan_num[3] * 2, chan_num[3], kernel_size=1, bias=False),
            norm(chan_num[3]),
            act(inplace=True)
        )
        # -------------------------------------------
        
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

        # --- [修改 3] Forward 逻辑改进 ---
        # 1. 获取拓扑特征
        skel_pred, topo_bias = self.topo_mediator(x4)
        
        # 2. 安全检查：强制对齐尺寸
        # 如果 map4 是 [B, C, 3, 3], 而 topo_bias 是 [B, C, 8, 8], 强行插值对齐到 map4
        if topo_bias.shape[-2:] != map4.shape[-2:]:
            topo_bias = F.interpolate(topo_bias, size=map4.shape[-2:], mode='bilinear', align_corners=True)
        
        # 3. 强融合 (Concat + Conv)
        # 将原始 map4 和 拓扑特征 拼接
        combined_map = torch.cat([map4, topo_bias], dim=1) # [B, 2*C, H, W]
        # 通过卷积融合回原始维度
        map4 = self.topo_fusion_conv(combined_map) # [B, C, H, W]
        # -------------------------------
        
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