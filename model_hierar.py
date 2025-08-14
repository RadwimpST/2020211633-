# model_hierar.py
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15

# 新：改用 TimeEncoder（PatchTST）作为时间编码器
from TimeEncoder import PatchTST

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (简化版，仅做输入标准化，不做反标准化)。
    对每个样本、每个 ROI、每个通道，沿时间维做标准化，可选可学习仿射。
    期望输入: x [B, C, T, V, M]
    """
    def __init__(self, num_channels: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            # 形状 [1, C, 1]，在 [B*M*V, C, T] 上广播
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
            self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, V, M]
        B, C, T, V, M = x.shape
        # 合并 (B, V, M) 维，逐"样本×ROI×人"独立标准化
        x_bvm = x.permute(0, 3, 4, 1, 2).contiguous().view(B * V * M, C, T)
        mean = x_bvm.mean(dim=-1, keepdim=True)
        std  = x_bvm.std(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_bvm - mean) / (std + self.eps)
        if self.affine:
            x_norm = self.gamma * x_norm + self.beta
        # 还原回 [B, C, T, V, M]
        x = x_norm.view(B, V, M, C, T).permute(0, 3, 4, 1, 2).contiguous()
        return x
class LayerNorm(nn.Module):
    def __init__(self, feature_size, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(feature_size))
        self.b_2 = nn.Parameter(torch.ones(feature_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class HieraFormer(nn.Module):
    """
    多头注意力（支持稀疏注意力 entmax15），用于：
    1) 节点特征交互（IsMulti=True）
    2) 预测节点分配/融合权重（IsMulti=False）
    """
    def __init__(self, input_dim, output_dim, embedding_dim,
                 add_self=False, normalize_embedding=False, dropout=0.0, bias=True,
                 IsMulti=True, mul_num=3):
        super().__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout and dropout > 1e-3:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.h = mul_num  # head 数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.softmax = nn.Softmax(dim=-1)
        self.IsMulti = IsMulti
        self.d_k = int(input_dim / self.h)
        self.dropout_x = nn.Dropout(p=0.1)

        if IsMulti:
            self.linears = nn.ModuleList()
            for _ in range(3):
                Mul_outdim = int(input_dim / mul_num) * mul_num
                self.linears.append(nn.Linear(input_dim, Mul_outdim))
            self.norm = LayerNorm(Mul_outdim)
        else:
            self.linears = nn.ModuleList()
            for i in range(3):
                self.linears.append(nn.Linear(self.embedding_dim, self.embedding_dim))
            self.norm = LayerNorm(self.embedding_dim)

        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
            nn.init.uniform_(self.bias, -0.1, 0.1)
        else:
            self.bias = None

    def attention(self, query, key, value, dropout=None):
        d_k = query.size(-1)
        energy = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
        # 稀疏注意力
        attention = entmax15(energy, dim=-1)
        p_attn = attention  # 修复：先赋值，避免未定义
        if dropout is not None and dropout > 1e-3:
            p_attn = self.dropout_x(attention)
        return torch.matmul(p_attn, value)

    def forward(self, x):
        # 期望输入 (B, C, N)；内部会转为 (B, N, C)
        B, _, _ = x.size()
        x = x.transpose(1, 2)  # (B, N, C)
        x1 = x
        if self.IsMulti:
            q, k, v = [
                l(x_a).view(B, -1, self.h, self.d_k).transpose(1, 2)
                for l, x_a in zip(self.linears, (x, x, x))
            ]
            x = self.attention(q, k, v, dropout=self.dropout)
            x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        else:
            q, k, v = [l(x_a) for l, x_a in zip(self.linears, (x, x, x))]
            x = self.attention(q, k, v, dropout=self.dropout)

        x = self.dropout_x(x)
        x_norm = x1 + self.norm(x)
        return x_norm  # (B, N, C)


class model_hierar(nn.Module):
    """
    用 PatchTST（TimeEncoder）替代 ST-GCFE：
    输入:  x 形状 (B, V, T)
    输出:  分类 logits (B, num_classes)
    """
    def __init__(self, args, concat=True, seq_len=187, num_nodes=90, output_channels=2):
        super().__init__()
        self.args = args
        self.num_pooling = args.num_pooling
        embedding_dim = args.embedding_dim
        assign_ratio = args.assign_ratio
        assign_ratio_1 = args.assign_ratio_1
        partroi = args.partroi
        self.num_pooling = args.num_pooling
        self.mult_num = args.mult_num
        self.dropout = args.dropout
        max_num_nodes = partroi
        self.node = max_num_nodes
        self.concat = concat
        add_self = not concat

        # === 新：时间编码器（PatchTST） ===
        # 说明：你现在手动在 TimeEncoder.PatchTST 的 __init__ 里设置参数，
        # 这里直接无参构造；要求其 forward(x: B,T,V) -> (B,C,V)，默认 C=64。
        # self.time_encoder = PatchTST()
        self.time_encoder = PatchTST(
            seq_len=seq_len,
            num_nodes=num_nodes,
            # 其他你已有的开关：patch_len/stride/rope/rpb/… 等
        )

        # === 下游层级注意力堆叠，要求 time_encoder 输出通道为 64 ===
        channel = 64
        self.input_channel = channel
        assign_input_dim = channel

        inter_channel = int(max_num_nodes * assign_ratio)     # N * r
        inter_channel_1 = 128

        self.conv1 = nn.Conv1d(int(self.input_channel / self.mult_num) * self.mult_num, inter_channel, 1)
        self.conv2 = nn.Conv1d(int(self.input_channel / self.mult_num) * self.mult_num, inter_channel_1, 1)
        self.bn1 = nn.BatchNorm1d(inter_channel)
        self.bn2 = nn.BatchNorm1d(inter_channel_1)
        self.act = nn.ReLU()
        self.relu = nn.ReLU(inplace=True)

        assign_dims = []
        self.conv_first_after_pool = nn.ModuleList()
        for i in range(self.num_pooling):
            if i == 0:
                self.pred_input_dim = inter_channel_1
            else:
                self.pred_input_dim = int(inter_channel / self.mult_num) * self.mult_num
            conv_first2 = self.build_hiera_layers(self.pred_input_dim, embedding_dim, add_self,
                                                  normalize=True, dropout=self.dropout,
                                                  IsMulti=True, mul_num=self.mult_num)
            self.conv_first_after_pool.append(conv_first2)

        self.conv_first = self.build_hiera_layers(channel, embedding_dim, add_self,
                                                 normalize=True, dropout=self.dropout,
                                                 IsMulti=True, mul_num=self.mult_num)

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(self.num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first = self.build_hiera_layers(assign_input_dim, assign_dim, add_self,
                                                        normalize=True, dropout=self.dropout, IsMulti=False)
            if i == 0:
                assign_input_dim = inter_channel
            else:
                assign_input_dim = int(inter_channel / self.mult_num) * self.mult_num
            assign_dim = int(assign_dim * assign_ratio_1)
            self.assign_conv_first_modules.append(assign_conv_first)

        self.softmax = nn.Softmax(dim=-1)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(int(inter_channel_1 / self.mult_num) * self.mult_num, output_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def build_hiera_layers(self, input_dim, embedding_dim, add_self,
                           normalize=False, dropout=0.0, mul_num=8, IsMulti=True):
        return HieraFormer(input_dim=input_dim,
                           output_dim=self.input_channel,
                           embedding_dim=embedding_dim,
                           add_self=add_self,
                           normalize_embedding=normalize,
                           bias=True,
                           dropout=dropout,
                           IsMulti=IsMulti,
                           mul_num=mul_num)

    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).to(x.device)
        return bn_module(x)

    def gcn_forward(self, x, conv_first, embedding_mask=None):
        # 注意：这里的命名沿用旧代码，实际已不再用 GCN
        # 期望 x: (B, N, C)，内部转为 (B, C, N) 以复用 HieraFormer 的输入约定
        x = x.permute(0, 2, 1)
        x_tensor = conv_first(x)  # (B, N, C)
        if embedding_mask is not None:
            x_tensor = x * embedding_mask
        return x_tensor

    def forward(self, x):
        """
        x: (B, V, T)
        """
        B0, V, T = x.size()

        M = 1  # 兼容原多人维，当前固定为 1
        x = x.permute(0, 2, 1).contiguous()

        # 1) 时间编码：PatchTST -> (B, C, V) 或 (B, V, C)
        x = x.unsqueeze(1).unsqueeze(4)  # [B, 1, T, V, 1]


        z = self.time_encoder(x)
        if z.dim() != 3:
            raise RuntimeError(f"TimeEncoder must return 3D tensor, got {z.shape}")
        # 兼容两种布局
        if z.size(1) == self.input_channel and z.size(2) == V:
            # (B, C, V)
            x_feat = z.permute(0, 2, 1)  # -> (B, V, C)
        elif z.size(1) == V:
            # (B, V, C)
            x_feat = z  # already (B, V, C)
        else:
            raise RuntimeError(f"Unexpected TimeEncoder output shape {z.shape}, "
                               f"expect (B,{self.input_channel},{V}) or (B,{V},{self.input_channel})")

        out_all = []

        # 2) 初始层的节点注意力
        hierarchical_tensor = self.gcn_forward(x_feat, self.conv_first)  # (B, N, C)
        out, _ = torch.max(hierarchical_tensor, dim=1)
        out_all.append(out)

        # 3) 层级节点融合
        if self.num_pooling == 0:
            hierarchical_tensor = self.dp1(F.relu(self.bn1(self.conv1(hierarchical_tensor.permute(0, 2, 1)))))
            hierarchical_tensor = hierarchical_tensor.permute(0, 2, 1)  # (B, N, C)
        else:
            x_mid = x_feat
            for i in range(self.num_pooling):
                x_mid = self.dp1(F.relu(self.bn1(self.conv1(x_mid.permute(0, 2, 1))))).permute(0, 2, 1)
                NodeAssign = self.gcn_forward(x_mid, self.assign_conv_first_modules[i])  # (B, N, C) as weights
                # 使用 NodeAssign 作为软分配（保持与原实现一致）
                x_pool = torch.matmul(NodeAssign.transpose(1, 2), hierarchical_tensor)  # (B, C, N') * (B, N, C) -> (B, C, C)??
                # 为了与原网络对应：x_pool 应该是 (B, N', C)。这里沿用原代码逻辑：
                if i == 0:
                    x_pool = self.dp1(F.relu(self.bn2(self.conv2(x_pool.permute(0, 2, 1))))).permute(0, 2, 1)
                hierarchical_tensor = self.gcn_forward(x_pool, self.conv_first_after_pool[i])  # (B, N', C)

        # 4) 全局池化 & 分类
        _, N_after, C_after = hierarchical_tensor.size()
        x_final = hierarchical_tensor.view(B0, M, N_after, C_after).mean(2).mean(1)  # (B, C)
        logits = self.linear2(x_final)  # (B, num_classes)
        return logits
