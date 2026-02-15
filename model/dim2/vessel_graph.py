import torch
import torch.nn as nn
import torch.nn.functional as F

class VesselGraphSampler(nn.Module):
    """专门负责从像素特征空间采样血管关键点"""
    def __init__(self, num_nodes=512):
        super().__init__()
        self.num_nodes = num_nodes

    def forward(self, x, prior):
        b, c, h, w = x.shape
        # 1. 在先验图上选点
        prior_flat = prior.view(b, -1)
        k = min(self.num_nodes, h * w)
        _, indices = torch.topk(prior_flat, k, dim=1) 

        # 2. 采样特征 [B, C, K]
        x_flat = x.view(b, c, -1)
        node_feats = torch.gather(x_flat, 2, indices.unsqueeze(1).expand(-1, c, -1))
        
        # 3. 计算坐标 (归一化到 0-1)，用于空间距离引导
        y = torch.div(indices, w, rounding_mode='floor').float() / (h - 1)
        x_coord = (indices % w).float() / (w - 1)
        coords = torch.stack([y, x_coord], dim=-1) # [B, K, 2]

        return node_feats.permute(0, 2, 1), indices, coords

class VesselGraphBottleneck(nn.Module):
    def __init__(self, in_channels, num_nodes=512):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 实例化 Sampler
        self.sampler = VesselGraphSampler(num_nodes=num_nodes)
        
        # 1. 特征压缩
        self.node_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 2. 图推理
        self.attn_query = nn.Linear(in_channels // 2, in_channels // 4)
        self.attn_key = nn.Linear(in_channels // 2, in_channels // 4)
        
        self.gcn = nn.Sequential(
            nn.Linear(in_channels // 2, in_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channels // 2, in_channels // 2)
        )

        # 3. 通道恢复 (改用可学习卷积，效果更好)
        self.proj_back = nn.Conv2d(in_channels // 2, in_channels, 1)
        
        # 4. 融合门控
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, prior):
        b, c, h, w = x.shape
        
        # --- A. 调用独立 Sampler ---
        # 首先对特征进行降维投影，再采样
        feat_reduced = self.node_proj(x)
        c_p = feat_reduced.shape[1]
        
        node_feats, indices, coords = self.sampler(feat_reduced, prior)

        # --- B. 图连接与推理 ---
        # 引入物理距离约束：防止相隔太远的像素建立非法连接
        dist_mat = torch.cdist(coords, coords) # [B, K, K]
        dist_mask = torch.exp(-dist_mat ** 2 / 0.1) # 只有距离近的点有较高权重
        
        q = self.attn_query(node_feats)
        k_vec = self.attn_key(node_feats)
        adj = torch.matmul(q, k_vec.transpose(1, 2)) / (q.shape[-1]**0.5)
        adj = F.softmax(adj, dim=-1) * dist_mask # 关键：融合空间距离

        node_feats = torch.matmul(adj, node_feats)
        node_feats = self.gcn(node_feats) # [B, K, C_p]

        # --- C. 特征重投影 ---
        graph_out_flat = torch.zeros(b, c_p, h * w, device=x.device)
        graph_out_flat.scatter_(2, indices.unsqueeze(1).expand(-1, c_p, -1), node_feats.permute(0, 2, 1))
        
        graph_out = graph_out_flat.view(b, c_p, h, w)
        graph_out_final = self.proj_back(graph_out)
        
        # --- D. 门控融合 ---
        gate = self.fusion_gate(x)
        return x + gate * graph_out_final


class AdaptiveVesselGraphBottleneck(nn.Module):
    def __init__(self, in_channels, num_nodes=512):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 1. 结构探测器 (来自 Adaptive)：自学习的显著性图，用于辅助采样
        self.saliency_detector = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        
        # 2. 采样器 (来自 Bottleneck)：负责坐标与特征提取
        self.sampler = VesselGraphSampler(num_nodes=num_nodes)
        
        # 3. 特征处理与图推理 (来自 Bottleneck)
        self.node_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.attn_query = nn.Linear(in_channels // 2, in_channels // 4)
        self.attn_key = nn.Linear(in_channels // 2, in_channels // 4)
        
        self.gcn = nn.Sequential(
            nn.Linear(in_channels // 2, in_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channels // 2, in_channels // 2)
        )
        
        self.proj_back = nn.Conv2d(in_channels // 2, in_channels, 1)

        # 4. 融合门控 (来自 Adaptive)：动态决定图特征的注入比例
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, external_prior=None):
        b, c, h, w = x.shape
        
        # --- 步骤 1: 生成采样图 (融合外部先验与自学习显著性) ---
        learned_prior = self.saliency_detector(x)
        sampling_map = external_prior * learned_prior if external_prior is not None else learned_prior
        
        # --- 步骤 2: 投影与采样 ---
        feat_reduced = self.node_proj(x)
        c_p = feat_reduced.shape[1]
        
        # 使用融合后的采样图进行采样
        node_feats, indices, coords = self.sampler(feat_reduced, sampling_map)

        # --- 步骤 3: 拓扑推理 (带空间距离约束) ---
        dist_mat = torch.cdist(coords, coords)
        # 距离掩码：确保只有物理上合理的点建立连接
        dist_mask = torch.exp(-dist_mat ** 2 / 0.1) 
        
        q = self.attn_query(node_feats)
        k = self.attn_key(node_feats)
        
        adj = torch.matmul(q, k.transpose(1, 2)) / (q.shape[-1]**0.5)
        adj = F.softmax(adj, dim=-1) * dist_mask # 融合特征相似度与空间连通性
        
        node_feats = torch.matmul(adj, node_feats)
        node_feats = self.gcn(node_feats)

        # --- 步骤 4: 重投影与通道恢复 ---
        graph_out_flat = torch.zeros(b, c_p, h * w, device=x.device)
        graph_out_flat.scatter_(2, indices.unsqueeze(1).expand(-1, c_p, -1), node_feats.permute(0, 2, 1))
        
        graph_out = self.proj_back(graph_out_flat.view(b, c_p, h, w))
        
        # --- 步骤 5: 柔性门控融合 ---
        # 根据原始特征和图特征的交互，动态计算融合权重
        gate = self.fusion_gate(torch.cat([x, graph_out], dim=1))
        return x + gate * graph_out