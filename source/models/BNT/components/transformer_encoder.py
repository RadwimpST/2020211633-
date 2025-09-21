from torch.nn import TransformerEncoderLayer
from torch import Tensor
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpretableTransformerEncoder(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[Tensor] = None

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights

# === [ADD] Minimal two-branch encoder built on original BNT skeleton (Post-LN) ===


# ---------- FiLM (per-module) ----------
class FiLM(nn.Module):
    """
    FiLM: y = gamma_m ⊙ x + beta_m
    mods: (B, N) int64 in [1..K] or [0..K-1]
    """
    def __init__(self, num_modules: int, d_model: int, film_dim: int = 64, mods_start_from_one: bool = True):
        super().__init__()
        self.num_modules = num_modules
        self.mods_start_from_one = mods_start_from_one
        emb_sz = num_modules + (1 if mods_start_from_one else 0)  # 预留 idx 0
        self.embed = nn.Embedding(emb_sz, film_dim)
        self.to_gamma = nn.Linear(film_dim, d_model)
        self.to_beta  = nn.Linear(film_dim, d_model)
        # 温和初始化：γ≈1, β≈0
        nn.init.zeros_(self.to_gamma.bias); nn.init.zeros_(self.to_beta.bias)
        nn.init.zeros_(self.to_beta.weight)
        with torch.no_grad():
            self.to_gamma.weight.data.zero_()
    def forward(self, x: torch.Tensor, mods: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D), mods: (B,N)
        if self.mods_start_from_one:
            mods = mods.clamp_min(1)
        e = self.embed(mods)                   # (B,N,F)
        gamma = 1.0 + self.to_gamma(e)         # (B,N,D)
        beta  = self.to_beta(e)                # (B,N,D)
        return gamma * x + beta

# ---------- Entmax (α in (1, 2]) ----------
def entmax_bisect(x: torch.Tensor, alpha: float = 1.5, dim: int = -1, n_iter: int = 50, tol: float = 1e-6):
    """
    通用 entmax 的简洁二分实现（支持任意 α∈(1,2]；α=1->softmax，α=2->sparsemax）
    参考 Martins & Astudillo 2016 / Peters et al. 2019 的闭式/二分思路。
    """
    if alpha <= 1.0 + 1e-6:
        return torch.softmax(x, dim=dim)
    if alpha >= 2.0 - 1e-6:
        # sparsemax（投影到概率单纯形）——很简化的实现
        z = x - x.max(dim=dim, keepdim=True).values
        z_sorted, _ = torch.sort(z, descending=True, dim=dim)
        k = torch.arange(1, z.size(dim)+1, device=x.device, dtype=x.dtype).view(
            *([1]*(z.dim()-1)), -1
        ).transpose(dim, -1)
        cssv = z_sorted.cumsum(dim=dim)
        cond = 1 + k * z_sorted > cssv
        k_z = cond.sum(dim=dim, keepdim=True).clamp(min=1)
        tau = (cssv.gather(dim, k_z-1) - 1) / k_z
        p = torch.clamp(z - tau, min=0)
        p = p / (p.sum(dim=dim, keepdim=True) + 1e-12)
        return p

    # α∈(1,2) → 二分求解 τ，使得 sum(((α-1)*x - τ)+^{1/(α-1)}) = 1
    d = x.size(dim)
    x = x - x.max(dim=dim, keepdim=True).values          # 中心化更稳定
    # 二分上下界：足够宽即可保证包住解
    tau_lo = x.max(dim=dim, keepdim=True).values - 10.0
    tau_hi = x.max(dim=dim, keepdim=True).values + 10.0
    for _ in range(n_iter):
        tau_mid = (tau_lo + tau_hi) / 2
        p_mid = torch.clamp((alpha - 1) * x - tau_mid, min=0) ** (1.0 / (alpha - 1))
        s = p_mid.sum(dim=dim, keepdim=True)
        # s>1 → τ 偏小（p 偏大）→ 提高下界；反之降低上界
        tau_lo = torch.where(s > 1, tau_mid, tau_lo)
        tau_hi = torch.where(s > 1, tau_hi, tau_mid)
        if (tau_hi - tau_lo).abs().max().item() < tol:
            break
    p = torch.clamp((alpha - 1) * x - tau_hi, min=0) ** (1.0 / (alpha - 1))
    p = p / (p.sum(dim=dim, keepdim=True) + 1e-12)
    return p

class EntmaxMultiheadAttention(nn.Module):
    """
    与 nn.MultiheadAttention 行为对齐，但把 softmax 换成 entmax(alpha)。
    仅用于“模块间”分支；保持 batch_first=True。
    """
    def __init__(self, embed_dim: int, num_heads: int, alpha: float = 1.3, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.alpha = alpha
        self.in_proj_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.in_proj_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.in_proj_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj  = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        # x: (B,N,D)
        B, N, D = x.shape
        q = self.in_proj_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,N,dh)
        k = self.in_proj_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.in_proj_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,N,N)
        if attn_mask is not None:
            # 加性掩码：允许=0，禁止=-1e9
            if attn_mask is not None:
                # 允许的形状: (N,N)、(B,N,N)、(B,1,N,N)、(1,1,N,N)
                if attn_mask.dim() == 2:  # (N,N)
                    mask = attn_mask.unsqueeze(0).unsqueeze(0)  # → (1,1,N,N)
                elif attn_mask.dim() == 3:  # (B,N,N)
                    # 与当前 batch 对齐后在 head 维度上广播
                    if attn_mask.size(0) != x.size(0) and attn_mask.size(0) == 1:
                        mask = attn_mask.expand(x.size(0), -1, -1)  # (B,N,N)
                    else:
                        mask = attn_mask
                    mask = mask.unsqueeze(1)  # → (B,1,N,N)
                elif attn_mask.dim() == 4:  # (B,1,N,N) 或 (B,H,N,N)
                    mask = attn_mask
                    # 若提供的是 (B,H,N,N)，也能直接广播到 (B,H,N,N)
                else:
                    raise ValueError("attn_mask must be (N,N), (B,N,N) or (B,1,N,N)")

                mask = mask.to(dtype=scores.dtype, device=scores.device)
                scores = scores + mask

        # 稳定化：减行内 max
        scores = scores - scores.max(dim=-1, keepdim=True).values

        attn = entmax_bisect(scores, alpha=self.alpha, dim=-1)  # (B,H,N,N)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)                              # (B,H,N,dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)     # (B,N,D)
        out = self.out_proj(out)                                 # (B,N,D)
        return out, attn

# ---------- Node-level gate ----------
class NodeGate(nn.Module):
    """ 每节点一门：g(x)∈(0,1)，形状 (B,N,1) """
    def __init__(self, d_model: int, hidden: int = 0, dropout: float = 0.0, bias_init: float = -0.3):
        super().__init__()
        if hidden and hidden > 0:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
            )
            nn.init.constant_(self.mlp[-1].bias, bias_init)
        else:
            self.mlp = nn.Linear(d_model, 1)
            nn.init.constant_(self.mlp.bias, bias_init)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(x))  # (B,N,1)

# ---------- Two-branch encoder layer (Post-LN, same as original skeleton) ----------
class TwoBranchEncoderLayer(nn.Module):
    """
    与原 BNT 的 TransformerEncoderLayer 残差/归一化顺序保持一致（Post-LN）。
    仅在:
      - 模块内: 注意力前加 FiLM
      - 模块间: Softmax -> Entmax(α)
      - 融合: gate(global/node) 或 concat
    其他保持不变。
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        entmax_alpha: float = 1.3,
        film_dim: int = 64,
        fuse: str = "gate",                  # "gate" | "concat"
        gate_granularity: str = "global",    # "global" | "node"
        gate_hidden: int = 0,
        gate_dropout: float = 0.0,
    ):
        super().__init__()
        self.fuse = fuse
        self.gate_granularity = gate_granularity

        # intra: 标准 MHA（保持不变）
        self.mha_intra = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # inter: entmax MHA（仅 softmax->entmax）
        self.mha_inter = EntmaxMultiheadAttention(d_model, nhead, alpha=entmax_alpha, dropout=dropout)

        # FiLM
        self.film = FiLM(num_modules=9999, d_model=d_model, film_dim=film_dim, mods_start_from_one=True)

        # 融合参数
        if fuse == "gate":
            if gate_granularity == "global":
                self.alpha = nn.Parameter(torch.zeros(2))  # softmax -> [w_intra, w_inter]
            elif gate_granularity == "node":
                self.alpha = None
                self.node_gate = NodeGate(d_model, hidden=gate_hidden, dropout=gate_dropout, bias_init=-0.3)
            else:
                raise ValueError("gate_granularity must be 'global' or 'node'")
        elif fuse == "concat":
            self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        else:
            raise ValueError("fuse must be 'gate' or 'concat'")

        # 与原版一致的 Post-LN 次序
        self.dropout1 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout  = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2    = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def set_num_modules(self, K: int):
        # 让 FiLM 的 embedding 足够大；原始标签一般 1..K
        self.film.num_modules = K

    def forward(self, x: torch.Tensor, attn_mask_intra: torch.Tensor, attn_mask_inter: torch.Tensor, mods: torch.Tensor):
        """
        x: (B,N,D); mods: (B,N)
        attn_mask_*: (N,N) or (B,N,N) 加性掩码（允许=0, 禁止=-1e9）
        """
        # ---- Self-Attn (two branches) ----
        # 模块内：FiLM 调制后做标准注意力
        y_intra = self.film(x, mods)
        z_intra, _ = self.mha_intra(y_intra, y_intra, y_intra, attn_mask=attn_mask_intra)

        # 模块间：entmax 注意力
        z_inter, _ = self.mha_inter(x, attn_mask=attn_mask_inter)

        # 融合（仅改变这里）
        if self.fuse == "gate":
            if self.gate_granularity == "global":
                w = torch.softmax(self.alpha, dim=0)  # [2]
                z = w[0] * z_intra + w[1] * z_inter
            else:  # node-level
                g = self.node_gate(x)                # (B,N,1) 来自当前输入特征
                z = (1.0 - g) * z_intra + g * z_inter
        else:  # concat
            z = self.proj(torch.cat([z_intra, z_inter], dim=-1))

        # ---- Post-LN 残差结构（保持与原版一致）----
        x = x + self.dropout1(z)
        x = self.norm1(x)

        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(y)
        x = self.norm2(x)
        return x

# ---------- 堆叠若干层 ----------
class ModularTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int = 1,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        entmax_alpha: float = 1.3,
        film_dim: int = 64,
        fuse: str = "gate",
        gate_granularity: str = "global",
        gate_hidden: int = 0,
        gate_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoBranchEncoderLayer(
                d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                entmax_alpha=entmax_alpha, film_dim=film_dim,
                fuse=fuse, gate_granularity=gate_granularity,
                gate_hidden=gate_hidden, gate_dropout=gate_dropout
            ) for _ in range(num_layers)
        ])

    def set_num_modules(self, K: int):
        for lyr in self.layers:
            lyr.set_num_modules(K)

    def forward(self, x, attn_mask_intra, attn_mask_inter, mods):
        for lyr in self.layers:
            x = lyr(x, attn_mask_intra, attn_mask_inter, mods)
        return x
# === [END ADD] ===============================================================
