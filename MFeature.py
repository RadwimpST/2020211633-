# MFeature.py
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_heads(x: torch.Tensor, h: int):
    # x: [B, N, D] -> [B, h, N, d]
    B, N, D = x.shape
    d = D // h
    return x.view(B, N, h, d).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: torch.Tensor):
    # x: [B, h, N, d] -> [B, N, h*d]
    B, h, N, d = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, N, h * d)


def _masked_softmax_with_log(logits: torch.Tensor, log_mask_add: torch.Tensor, dim: int = -1, dropout_p: float = 0.0):
    """Stable softmax with additive log(mask+eps). log_mask_add is broadcastable to logits.
    Works for both hard (0/1) and soft masks in [0,1]."""
    scores = logits + log_mask_add
    scores = scores - scores.max(dim=dim, keepdim=True).values
    attn = scores.exp()
    denom = attn.sum(dim=dim, keepdim=True) + 1e-8
    attn = attn / denom
    if dropout_p > 0.0 and attn.requires_grad:
        attn = F.dropout(attn, p=dropout_p, training=True)
    return attn


class MLPZeroLast(nn.Module):
    """2-layer MLP with GELU; last linear weight/bias initialized to zero."""
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or max(in_dim, out_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or (4 * d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc2(self.dropout(F.gelu(self.fc1(x))))
        return x


class IntraBlock(nn.Module):
    """模块内（masked）自注意力 + FiLM（两处），全部 Pre-Norm。"""
    def __init__(self, d_model: int, heads: int, d_embed: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.h = heads
        self.dk = d_model // heads

        # FiLM 调制：两个门，最后一层零初始化（初始≈恒等）
        self.film1 = MLPZeroLast(d_embed, 2 * d_model, hidden=2 * d_model, dropout=dropout)
        self.film2 = MLPZeroLast(d_embed, 2 * d_model, hidden=2 * d_model, dropout=dropout)

        # QKV 投影
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = FFN(d_model, dropout=dropout)

        # Norm & Dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, e_node: torch.Tensor, conn: torch.Tensor):
        """
        x:      [B, V, D]
        e_node: [B, V, d_e]    （由 assign 与模块嵌入表 E 生成）
        conn:   [B, V, V]      （同模块的连通权重，硬/软均可；范围建议[0,1]）
        """
        B, V, D = x.shape
        h, dk = self.h, self.dk

        # === PreNorm + FiLM (1) ===
        y = self.ln1(x)
        gamma1, beta1 = self.film1(e_node).chunk(2, dim=-1)  # [B,V,D]x2
        y = y * (1.0 + gamma1) + beta1

        # === Masked MHSA（仅同模块） ===
        Q = _split_heads(self.Wq(y), h)     # [B,h,V,dk]
        K = _split_heads(self.Wk(y), h)     # [B,h,V,dk]
        Vv = _split_heads(self.Wv(y), h)    # [B,h,V,dk]

        logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dk)  # [B,h,V,V]
        # 采用加 log(mask+eps) 的方式实现硬/软掩码
        log_mask_add = torch.log(conn.clamp(min=0.0, max=1.0) + 1e-6).unsqueeze(1)  # [B,1,V,V]
        attn = _masked_softmax_with_log(logits, log_mask_add, dim=-1, dropout_p=self.dropout.p)

        y = torch.matmul(attn, Vv)  # [B,h,V,dk]
        y = _merge_heads(y)         # [B,V,D]
        y = self.proj(y)
        x = x + self.dropout(y)

        # === PreNorm + FiLM (2) + FFN ===
        y2 = self.ln2(x)
        gamma2, beta2 = self.film2(e_node).chunk(2, dim=-1)
        y2 = y2 * (1.0 + gamma2) + beta2
        y2 = self.ffn(y2)
        x = x + self.dropout(y2)
        return x


class ModuleCLSPool(nn.Module):
    """每个模块一个 CLS_k，用 CLS→节点的 masked 注意力聚合为模块 token。"""
    def __init__(self, k_modules: int, d_model: int, heads: int, use_e_for_cls: bool = True, d_embed: int = 16, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0
        self.K = k_modules
        self.h = heads
        self.dk = d_model // heads
        self.use_e_for_cls = use_e_for_cls

        self.cls0 = nn.Parameter(torch.zeros(1, k_modules, d_model))  # 基向量
        nn.init.normal_(self.cls0, mean=0.0, std=0.02)

        self.E2D = nn.Linear(d_embed, d_model) if use_e_for_cls else None

        # Q (from CLS), K/V (from nodes)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.ln = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, assign: torch.Tensor, E: Optional[torch.Tensor] = None, detach_mask: bool = True):
        """
        x:      [B, V, D]  （节点特征，来自 Intra 之后）
        assign: [B, V, K]  （Z_tilde 或 S）
        E:      [K, d_e]   （可选模块嵌入表，用于调制 CLS；若 None 则仅用 cls0）
        """
        B, V, D = x.shape
        K = self.K

        # 构造 batch 化 CLS
        cls = self.cls0.expand(B, K, D)  # [B,K,D]
        if self.use_e_for_cls and (E is not None):
            cls = cls + self.E2D(E).unsqueeze(0).expand(B, K, D)

        # 掩码：CLS 只看本模块的节点
        mask_kv = assign.transpose(1, 2)  # [B,K,V]
        if detach_mask:
            mask_kv = mask_kv.detach()
        # count/空模块诊断
        count_per_k = mask_kv.sum(dim=-1)  # [B,K]
        empty_mask = (count_per_k <= 1e-6)  # [B,K]

        # 注意力（CLS 做 Q；节点做 K,V）
        q = _split_heads(self.Wq(self.ln(cls)), self.h)         # [B,h,K,dk]
        k = _split_heads(self.Wk(x), self.h)                    # [B,h,V,dk]
        v = _split_heads(self.Wv(x), self.h)                    # [B,h,V,dk]

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)  # [B,h,K,V]
        log_mask_add = torch.log(mask_kv.clamp(min=0.0, max=1.0) + 1e-6).unsqueeze(1)  # [B,1,K,V]
        attn = _masked_softmax_with_log(logits, log_mask_add, dim=-1, dropout_p=self.dropout.p)

        m = torch.matmul(attn, v)                   # [B,h,K,dk]
        m = _merge_heads(m)                         # [B,K,D]

        # 对空模块回退：若某模块无成员，则直接取 CLS
        if empty_mask.any():
            # 将对应位置替换为 cls；其余保持聚合结果
            m = torch.where(empty_mask.unsqueeze(-1), cls, m)

        # 仅对 CLS token 自己做 FFN（更省算）
        m = m + self.dropout(self.ffn(self.ln(m)))
        return m, {"count_per_k": count_per_k, "empty_mask": empty_mask}


class InterBlock(nn.Module):
    """模块 token 上的标准自注意力（不使用 FiLM），Pre-Norm。"""
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.dk = d_model // heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.ffn = FFN(d_model, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, m: torch.Tensor):
        # m: [B,K,D]
        q = _split_heads(self.Wq(self.ln1(m)), self.h)
        k = _split_heads(self.Wk(m), self.h)
        v = _split_heads(self.Wv(m), self.h)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = _merge_heads(out)
        m = m + self.dropout(self.proj(out))

        f = self.ffn(self.ln2(m))
        m = m + self.dropout(f)
        return m


class GlobalReadout(nn.Module):
    """全局 CLS 读出：g 作为查询，对 K 个模块 token 做注意力，得到 r。"""
    def __init__(self, d_model: int, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.dk = d_model // heads

        self.g = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.g, mean=0.0, std=0.02)

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, m: torch.Tensor):
        # m: [B,K,D]
        B, K, D = m.shape
        g = self.g.expand(B, 1, D)  # [B,1,D]

        q = _split_heads(self.Wq(self.ln(g)), self.h)   # [B,h,1,dk]
        k = _split_heads(self.Wk(m), self.h)            # [B,h,K,dk]
        v = _split_heads(self.Wv(m), self.h)            # [B,h,K,dk]

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)  # [B,h,1,K]
        alpha = logits.softmax(dim=-1)
        alpha = self.dropout(alpha)

        r = torch.matmul(alpha, v).squeeze(-2)  # [B,h,dk]
        r = _merge_heads(r.unsqueeze(-2)).squeeze(-2)   # [B,D]
        return r


class MFeature(nn.Module):
    """
    模块级特征提取器：
      ① Intra（FiLM + masked MHSA）× L_intra
      ② Module-CLS 池化（masked）
      ③ Inter（MHSA on module tokens）× L_inter
      ④ Global-CLS 读出
    """
    def __init__(
        self,
        k_modules: int,
        d_model: int = 64,
        d_embed: int = 16,                 # 节点 FiLM 的 e_node 维度
        heads_intra: int = 4,
        heads_cls: int = 4,
        heads_inter: int = 4,
        layers_intra: int = 2,
        layers_inter: int = 1,
        dropout: float = 0.1,
        use_e_for_cls: bool = True,
        d_ff: Optional[int] = None
    ):
        super().__init__()
        self.K = k_modules
        self.d = d_model

        # Intra blocks
        self.intra = nn.ModuleList([
            IntraBlock(d_model=d_model, heads=heads_intra, d_embed=d_embed, dropout=dropout)
            for _ in range(layers_intra)
        ])

        # CLS pooling
        self.cls_pool = ModuleCLSPool(k_modules, d_model, heads=heads_cls, use_e_for_cls=use_e_for_cls, d_embed=d_embed, dropout=dropout)

        # Inter blocks
        self.inter = nn.ModuleList([
            InterBlock(d_model=d_model, heads=heads_inter, dropout=dropout)
            for _ in range(layers_inter)
        ])

        # Global readout
        self.readout = GlobalReadout(d_model=d_model, heads=max(1, heads_inter // 2), dropout=dropout)

        # 产生 e_node 的简单线性层（由 assign 与模块嵌入表 E 生成后，通常还会过一层）
        # 这里我们只在 forward 内按你的公式 e_node = assign @ E 构造，因此不在 __init__ 里加额外参数。

    @torch.no_grad()
    def _build_conn(self, assign: torch.Tensor) -> torch.Tensor:
        """根据分配矩阵构造节点间连通权（硬/软均可）。"""
        # [B,V,K] @ [B,K,V] -> [B,V,V]；对 soft 情况得到 [0,1] 之间的期望连通
        conn = torch.matmul(assign, assign.transpose(1, 2))
        # clip to [0,1] 以稳健（硬分配会是{0,1}）
        return conn.clamp_(0.0, 1.0)

    def forward(
        self,
        H: torch.Tensor,                  # [B,V,D]
        assign: torch.Tensor,             # [B,V,K]  (Z_tilde 或 S)
        *,
        E: Optional[torch.Tensor] = None, # [K,d_e]  模块嵌入表（可选）
        detach_mask: bool = True,         # True: 掩码不回传到分配器
        return_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        B, V, D = H.shape
        assert D == self.d, f"D mismatch: got {D}, expected {self.d}"
        A = assign.detach() if detach_mask else assign

        # === 构造 Intra 所需的连通权与节点嵌入（e_node = A @ E） ===
        conn = self._build_conn(A)                 # [B,V,V]
        if E is None:
            # 若未提供模块嵌入表，用 one-hot 聚合得到 e_node 的占位（不引入新信息，但保持接口一致）
            # 这里直接用模块概率作为特征：e_node = A @ I = A
            e_node = A  # [B,V,K] 作为调制输入
            d_e = A.shape[-1]
        else:
            e_node = torch.matmul(A, E)            # [B,V,d_e]
            d_e = E.shape[-1]

        x = H
        # 若 e_node 的维度与 IntraBlock 期望不一致，可线性映射（此处省略，默认 E/assign 已匹配 d_embed）

        # === ① 模块内更新（多层） ===
        for blk in self.intra:
            x = blk(x, e_node, conn)

        # === ② Module-CLS 池化 ===
        m, diag_pool = self.cls_pool(x, A, E=E, detach_mask=detach_mask)  # [B,K,D]

        # === ③ 模块间融合（多层） ===
        for blk in self.inter:
            m = blk(m)  # [B,K,D]

        # === ④ 全局读出 ===
        r = self.readout(m)  # [B,D]

        out = {
            "m_tokens": m,          # [B,K,D]
            "readout": r,           # [B,D]
            "node_out": x,          # [B,V,D]  （Intra 之后的节点特征）
        }
        # 诊断信息（供 train.py 记录/早停）
        out["diag"] = {
            "count_per_k": diag_pool["count_per_k"],   # [B,K]
            "empty_mask": diag_pool["empty_mask"],     # [B,K] bool
        }
        if return_tokens is False:
            out.pop("m_tokens", None)
        return out
