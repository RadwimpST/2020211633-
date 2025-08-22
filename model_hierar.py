# model_hierar.py — MVP 版：PatchTST → 软分配S → Pre-QKV + 模块化注意力(共享主干 + 稳定FiLM) → 模块读出(mean) → 分类
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15
from TimeEncoder import PatchTST
from Allocator import Allocator


# ------------------------- 基础层：LayerNorm -------------------------
class LayerNorm(nn.Module):
    def __init__(self, feature_size, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(feature_size))
        self.b_2 = nn.Parameter(torch.ones(feature_size))
        # self.b_2 = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# ------------------------- 节点注意力：HieraFormer -------------------------
class HieraFormer(nn.Module):
    """多头注意力（稀疏注意力 entmax15），用于节点维交互。
    约定：输入 (B, C, N) → 输出 (B, N, C)
    仅保留多头路径（IsMulti=True）。
    """
    def __init__(self, input_dim, embedding_dim, dropout=0.0, mul_num=8):
        super().__init__()
        self.h = mul_num
        self.d_k = input_dim // self.h
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 1e-3 else None
        self.dropout_x = nn.Dropout(p=0.1)

        # 三个线性层：Q/K/V
        self.qkv = nn.ModuleList()
        for _ in range(3):
            Mul_outdim = int(input_dim / mul_num) * mul_num
            self.qkv.append(nn.Linear(input_dim, Mul_outdim))
        self.norm = LayerNorm(Mul_outdim)

    def attention(self, query, key, value):
        d_k = query.size(-1)
        energy = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
        p_attn = entmax15(energy, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value)

    def forward(self, x):
        # 期望 x: (B, C, N)
        B, C, N = x.size()
        x = x.transpose(1, 2)  # (B, N, C)
        x_res = x
        q, k, v = [
            l(xa).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for l, xa in zip(self.qkv, (x, x, x))
        ]
        x = self.attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        x = self.dropout_x(x)
        x = x_res + self.norm(x)
        return x  # (B, N, C)


# ------------------------- 主模型：MVP 软分配 + 模块化 -------------------------
class model_hierar(nn.Module):
    """MVP：时间编码(PatchTST) → 软分配Allocator(S) → Pre-QKV(√S门控) + 模块FiLM(稳定) → 模块读出(mean) → 分类

    输入:  x (B, V, T)
    输出:  logits (B, num_classes)
    仅使用软分配，不加入CLS/动态残差/模块间注意力等二期特性。
    """
    def __init__(self, args, concat=True, seq_len=187, num_nodes=90, output_channels=2):
        super().__init__()
        self.args = args
        self.dropout = float(getattr(args, 'dropout', 0.0))
        self.mult_num = int(getattr(args, 'mult_num', 8))
        self.if_norm = int(getattr(args, 'if_norm', 0))  # MVP 默认 0；若为 1，将在每模块上做通道 MLP

        # 模块数（K），默认 4，可通过 --num_modules 指定
        self.K = int(getattr(args, 'num_modules', 4))

        # === 时间编码器（PatchTST） ===
        self.time_encoder = PatchTST(seq_len=seq_len, num_nodes=num_nodes)
        channel = 128  # PatchTST 输出对齐的通道数
        self.input_channel = channel

        # === 软分配器（返回 S） ===
        self.allocator = Allocator(
            d=channel,
            k=self.K,
            use_mlp=True,
            use_bias=False,
            use_cosine=True,
            tau_init=float(getattr(args, 'tau_init', 1.5)),
            tau_min=float(getattr(args, 'tau_min', 0.5)),
            ema_momentum=None,
            use_ema_for_logits=False,
            normalize_prototypes=False,
        )

        # === 模块身份嵌入 & FiLM（稳定版，仅依赖E_k） ===
        self.mod_embed = nn.Embedding(self.K, channel)
        self.film_mlp = nn.Linear(channel, 2 * channel)

        # === 节点注意力（共享主干） ===
        self.conv_first = HieraFormer(input_dim=channel,
                                      embedding_dim=int(getattr(args, 'embedding_dim', 128)),
                                      dropout=self.dropout,
                                      mul_num=self.mult_num)

        # === 可选通道 MLP（每模块上应用） ===
        partroi = int(getattr(args, 'partroi', num_nodes))
        assign_ratio = float(getattr(args, 'assign_ratio', 0.5))
        inter_channel = max(1, int(partroi * assign_ratio))
        self.inter_channel = inter_channel
        if self.if_norm == 1:
            in_ch = int(self.input_channel / self.mult_num) * self.mult_num
            self.conv1 = nn.Conv1d(in_ch, inter_channel, 1)
            self.bn1 = nn.BatchNorm1d(inter_channel)
            self.relu = nn.ReLU(inplace=True)
            self.dp1 = nn.Dropout(p=self.dropout)
            cls_in_dim = inter_channel
        else:
            cls_in_dim = channel

        # === 分类头 ===
        self.linear2 = nn.Linear(cls_in_dim, int(output_channels), bias=False)

    # 小包装：把 (B,N,C) → (B,C,N) 送入注意力，再取回 (B,N,C)
    def gcn_forward(self, x, conv_first):
        x = x.permute(0, 2, 1)  # (B,C,N)
        x_tensor = conv_first(x)  # (B,N,C)
        return x_tensor

    @torch.no_grad()
    def infer_assign(self, x):
        """
        仅计算并返回软分配矩阵 S（不做后续注意力/分类）
        输入 x: (B, V, T)
        返回 S: (B, V, K)
        """
        self.eval()
        B, V, T = x.size()
        xt = x.permute(0, 2, 1).unsqueeze(1).unsqueeze(4)  # (B,1,T,V,1)
        z = self.time_encoder(xt)
        if z.size(1) == self.input_channel and z.size(2) == V:
            H = z.permute(0, 2, 1)                          # (B,V,C)
        elif z.size(1) == V:
            H = z                                           # (B,V,C)
        else:
            raise RuntimeError(f"TimeEncoder output shape {tuple(z.shape)} unexpected")
        # 仅取软分配
        S, _, _ = self.allocator(H)          # (B,V,K)
        return S


    def forward(self, x):
        # x: (B, V, T)
        B, V, T = x.size()

        # 1) 时间编码（PatchTST 需要 5D: (B,C,T,V,M)）
        xt = x.permute(0, 2, 1).unsqueeze(1).unsqueeze(4)  # (B,1,T,V,1)
        z = self.time_encoder(xt)
        # 对齐为 (B, V, C)
        if z.size(1) == self.input_channel and z.size(2) == V:
            H = z.permute(0, 2, 1)
        elif z.size(1) == V:
            H = z
        else:
            raise RuntimeError(f"TimeEncoder output shape {tuple(z.shape)} unexpected; expect (B,{self.input_channel},{V}) or (B,{V},{self.input_channel})")
        # H: (B, V, C)

        # 2) 软分配 S（仅软，不做硬路由）
        S, logits_assign, extras = self.allocator(H)
        # S: (B, V, K)

        # 3) FiLM（稳定：仅依赖模块身份 E_k）
        E = self.mod_embed.weight  # (K, C)
        E = E.unsqueeze(0).expand(B, -1, -1)  # (B, K, C)
        film_params = self.film_mlp(E)  # (B, K, 2C)
        C = self.input_channel
        gamma = 1.0 + 0.1 * torch.tanh(film_params[:, :, :C])     # (B, K, C)
        beta  = 0.1 * torch.tanh(film_params[:, :, C:])           # (B, K, C)

        # 4) Pre-QKV：FiLM + √S 预门控，堆到 batch 维
        H_tile = H.unsqueeze(1).expand(-1, self.K, -1, -1)        # (B, K, V, C)
        H_film = gamma.unsqueeze(2) * H_tile + beta.unsqueeze(2)  # (B, K, V, C)
        S_kv = S.transpose(1, 2)                                  # (B, K, V)
        m = torch.sqrt(S_kv.clamp_min(1e-8)).unsqueeze(-1)        # (B, K, V, 1)
        H_hat = H_film * m                                        # (B, K, V, C)

        # (B*K, V, C) → 注意力（共享）
        H_hat_flat = H_hat.reshape(B * self.K, V, C)
        O_flat = self.gcn_forward(H_hat_flat, self.conv_first)     # (B*K, V, C)

        # 可选：通道 MLP（每模块应用）
        if self.if_norm == 1:
            y = O_flat.permute(0, 2, 1)                           # (B*K, C, V)
            y = self.dp1(self.relu(self.bn1(self.conv1(y))))      # (B*K, C*, V)
            O_flat = y.permute(0, 2, 1)                           # (B*K, V, C*)
            C_out = self.inter_channel
        else:
            C_out = C

        # 还原 K 维
        O = O_flat.view(B, self.K, V, C_out)                      # (B, K, V, C_out)

        # 5) 模块读出：S 加权归一化（与路由一致）
        weights = S_kv.sum(dim=2, keepdim=True).clamp_min(1e-6)   # (B, K, 1)
        M = (S_kv.unsqueeze(-1) * O).sum(dim=2) / weights         # (B, K, C_out)

        # 6) 跨模块聚合（mean）→ 分类
        g = M.mean(dim=1)                                         # (B, C_out)
        logits = self.linear2(g)                                  # (B, num_classes)
        return logits
