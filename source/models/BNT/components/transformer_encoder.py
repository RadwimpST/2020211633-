from torch.nn import TransformerEncoderLayer
from torch import Tensor
from typing import Optional
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

# ========== Entmax（简化版） ==========
def _entmax_bisect(logits, alpha=1.3, n_iter=50, tol=1e-6):
    # logits: [B, L, S] 或 [*, S]
    # 返回 entmax 概率，与 softmax 形状相同
    shape = logits.shape
    x = logits.reshape(-1, shape[-1])

    # 参考 Peters et al. Entmax: sparse probability mappings; 这里给一个常用的二分法实现
    d = x.size(1)
    x = x / max(1.0, abs(alpha - 1.0))  # 粗略缩放提高数值稳定
    max_x, _ = x.max(dim=1, keepdim=True)

    x = x - max_x  # 平移不变性
    tau_lo = (x.min(dim=1, keepdim=True).values - 1.0).detach()
    tau_hi = (x.max(dim=1, keepdim=True).values - 1e-6).detach()

    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.clamp(x - tau, min=0) ** (1.0 / (alpha - 1.0))
        s = p.sum(dim=1, keepdim=True)
        f = s ** (alpha - 1.0)
        too_big = (f > 1.0)
        tau_lo = torch.where(too_big, tau, tau_lo)
        tau_hi = torch.where(too_big, tau_hi, tau)
        if (tau_hi - tau_lo).max() < tol:
            break
    tau = (tau_lo + tau_hi) / 2.0
    p = torch.clamp(x - tau, min=0) ** (1.0 / (alpha - 1.0))
    p_sum = p.sum(dim=1, keepdim=True) + 1e-12
    p = p / p_sum
    return p.reshape(shape)

class EntmaxScaledDotProduct(nn.Module):
    def __init__(self, alpha=1.3, dropout=0.0):
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q,k,v: [B, L, d], [B, S, d], [B, S, d]
        d = q.size(-1)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)  # [B,L,S]
        if attn_mask is not None:
            scores = scores + attn_mask  # additive mask: allowed 0 / banned -1e9

        attn = _entmax_bisect(scores, alpha=self.alpha)          # [B,L,S]
        attn = self.dropout(attn)
        out = torch.bmm(attn, v)                                 # [B,L,d]
        return out, attn

class EntmaxMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, alpha=1.3, dropout=0.0, batch_first=True):
        super().__init__()
        assert batch_first, "only batch_first=True supported here"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.core = EntmaxScaledDotProduct(alpha=alpha, dropout=dropout)

    def _split(self, x):
        # x: [B, L, d_model] -> [B*nhead, L, d_head]
        B, L, _ = x.shape
        x = x.view(B, L, self.nhead, self.d_head).transpose(1, 2).contiguous()
        return x.view(B * self.nhead, L, self.d_head)

    def _merge(self, x, B):
        # x: [B*nhead, L, d_head] -> [B, L, d_model]
        x = x.view(B, self.nhead, x.size(1), self.d_head).transpose(1, 2).contiguous()
        return x.view(B, x.size(1), self.d_model)

    def forward(self, x, attn_mask=None):
        # x: [B, L, d_model]; attn_mask: [L,S] or [1,L,S] or broadcastable
        B, L, _ = x.shape
        Q = self._split(self.q_proj(x))
        K = self._split(self.k_proj(x))
        V = self._split(self.v_proj(x))
        if attn_mask is not None:
            # 扩展到多头 batch： [1,L,S] -> [B*nhead, L, S]
            attn_mask = attn_mask.unsqueeze(0) if attn_mask.dim()==2 else attn_mask
            attn_mask = attn_mask.expand(B, -1, -1).repeat(self.nhead, 1, 1)
        out, _ = self.core(Q, K, V, attn_mask=attn_mask)
        out = self._merge(out, B)
        return self.out_proj(out)

# ========== FiLM（按模块条件化） ==========
class FiLM(nn.Module):
    def __init__(self, num_modules, d_model, film_dim=64, place="pre_qkv"):
        super().__init__()
        self.place = place
        self.embed = nn.Embedding(num_modules + 1, film_dim)  # 模块ID从1..K
        self.mlp   = nn.Linear(film_dim, 2 * d_model)
        nn.init.zeros_(self.mlp.weight); nn.init.zeros_(self.mlp.bias)  # γ≈1, β≈0 的效果由下面的 +1 实现

    def forward(self, x, mods):
        """
        x: [B,N,d]; mods: [N] （群体模板，批内所有样本相同）
        返回 γ⊙x + β，γ=1+Δγ
        """
        B, N, d = x.shape
        h = self.embed(mods)           # [N,film_dim]
        gb = self.mlp(h)               # [N,2d]
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + gamma            # 初始化≈1
        gamma = gamma.unsqueeze(0).expand(B, N, d)
        beta  = beta.unsqueeze(0).expand(B, N, d)
        return gamma * x + beta

# ========== 两路编码器层 + 融合 ==========
class ModularEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, entmax_alpha=1.3, film_dim=64, fuse="gate", gate_granularity="global", dropout=0.1):
        super().__init__()
        self.fuse = fuse
        self.gate_granularity = gate_granularity

        # 分支
        self.mha_intra = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.mha_inter = EntmaxMultiheadAttention(d_model, nhead, alpha=entmax_alpha)

        self.film = FiLM(num_modules=9999, d_model=d_model, film_dim=film_dim)  # num_modules 会在 build 时 reset
        # 融合参数
        if fuse == "gate" or fuse == "sum":
            self.alpha = nn.Parameter(torch.zeros(2))  # softmax -> [w_intra, w_inter]
        elif fuse == "concat":
            self.proj = nn.Linear(2 * d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def set_num_modules(self, K):
        # 让 FiLM 知道真实 K
        self.film = FiLM(num_modules=K, d_model=self.norm1.normalized_shape[0], film_dim=self.film.mlp.in_features)

    def forward(self, x, attn_mask_intra, attn_mask_inter, mods):
        # Pre-LN
        y = self.norm1(x)

        # Intra 分支：FiLM + MHA(intra mask)
        y_intra = self.film(y, mods)
        z_intra, _ = self.mha_intra(y_intra, y_intra, y_intra, attn_mask=attn_mask_intra)

        # Inter 分支：entmax MHA（inter mask）
        z_inter = self.mha_inter(y, attn_mask=attn_mask_inter)

        # 融合
        if self.fuse in ("gate", "sum"):
            w = F.softmax(self.alpha, dim=0)  # [2]
            z = w[0] * z_intra + w[1] * z_inter
        else:  # concat
            z = torch.cat([z_intra, z_inter], dim=-1)
            z = self.proj(z)

        # 残差 + FFN
        x = x + self.dropout(z)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class ModularTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, entmax_alpha=1.3, film_dim=64, film_place="pre_qkv", fuse="gate", gate_granularity="global", dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ModularEncoderLayer(d_model, nhead, entmax_alpha, film_dim, fuse, gate_granularity, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask_intra, attn_mask_inter, mods):
        # 确保 mask 可广播成 [1,N,N]
        if attn_mask_intra.dim() == 2:
            attn_mask_intra = attn_mask_intra  # [N,N]
            attn_mask_inter = attn_mask_inter
        for layer in self.layers:
            # 第一次调用时把 K 告诉 FiLM
            if isinstance(layer.film, FiLM) and layer.film.embed.num_embeddings == 10000:
                K = int(mods.max().item())
                layer.set_num_modules(K)
            x = layer(x, attn_mask_intra, attn_mask_inter, mods)
        return x
