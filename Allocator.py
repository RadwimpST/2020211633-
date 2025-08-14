
import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gumbel_noise_like(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """Sample Gumbel(0, 1) noise with the same shape/device/dtype as x."""
    # torch.rand_like returns [0,1); clamp for numerical stability
    u = torch.rand_like(x).clamp_(min=eps, max=1.0 - eps)
    return -torch.log(-torch.log(u + eps) + eps)


class RoutingHead(nn.Module):
    """A tiny MLP + LayerNorm head to decorrelate routing from classification.

    Input/Output shape: (B, V, D) -> (B, V, D)
    """

    def __init__(self, d: int, hidden: Optional[int] = None, use_mlp: bool = True):
        super().__init__()
        self.use_mlp = use_mlp
        self.pre_norm = nn.LayerNorm(d)
        if use_mlp:
            h = hidden or max(64, d * 2)
            self.mlp = nn.Sequential(
                nn.Linear(d, h),
                nn.GELU(),
                nn.Linear(h, d),
            )
            self.post_norm = nn.LayerNorm(d)
        else:
            self.mlp = nn.Identity()
            self.post_norm = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        if isinstance(self.mlp, nn.Sequential):
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B, V, D)
        H = self.pre_norm(H)
        H = self.mlp(H)
        H = self.post_norm(H)
        return H


class Allocator(nn.Module):
    """Straight-Through Gumbel-Softmax allocator (prototype-based).

    Args:
        d: feature dimension D of node embeddings from the time encoder.
        k: number of modules (clusters). Must be fixed at init time.
        use_mlp: whether to apply a small routing MLP head before scoring.
        mlp_hidden: hidden size of the routing MLP. Defaults to 2*D (min 64).
        use_bias: whether to learn a per-module bias added to logits.
        use_cosine: if True, use cosine similarity for scoring; else scaled dot-product.
        tau_init: initial temperature for softmax.
        tau_min: minimum temperature used by external scheduler; stored here for logging.
        ema_momentum: if not None (e.g., 0.99), maintain EMA prototypes for stability.
        use_ema_for_logits: if True, compute logits against EMA prototypes (no gradient).
        normalize_prototypes: if True, keep prototypes L2-normalized after each update.

    Inputs (forward):
        H: Tensor (B, V, D) node features.
        tau: Optional float temperature. If None, fall back to self.tau.
        hard: use hard one-hot assignments in the forward path (ST trick). Default True.
        gumbel_noise: add Gumbel noise when training. Default follows self.training.

    Returns:
        Z_tilde: (B, V, K) straight-through hard assignments (forward hard, backward soft).
        S: (B, V, K) soft assignments (probabilities).
        logits: (B, V, K) raw logits before softmax (returned for logging/diagnostics).
        extras: dict with prototype tensor, usage statistics, entropies, tau actually used.
    """

    def __init__(
        self,
        d: int, # 节点特征维度
        k: int = 7,
        *,
        use_mlp: bool = True, # 否启用路由小 MLP（LN→MLP→LN）来解耦“路由”与“分类”。建议开。
        mlp_hidden: Optional[int] = None, # 路由 MLP 的隐藏维
        use_bias: bool = True, # logits 是否加每个模块的可学习偏置 b_k。
        use_cosine: bool = False,
        tau_init: float = 1.5,  # 初始温度 τ（越大越“软”）
        tau_min: float = 0.3,  # τ 的下界
        ema_momentum: Optional[float] = None,  # 若设为 0.99/0.995，将维护原型的 EMA 副本以增稳；不需要就传 None。
        use_ema_for_logits: bool = False, # 打分时是否用 EMA 原型（不走梯度）计算 logits。一般前期可开，后期可关。
        normalize_prototypes: bool = False, # 每次前向前对原型做 L2 归一化（把它们约束在单位球面上）。
        eps: float = 1e-6, # 聚合/归一化时避免除零的小数。
    ) -> None:
        super().__init__()
        assert k > 0 and d > 0, "Allocator: k and d must be positive."

        self.d = d
        self.k = k
        self.use_bias = use_bias
        self.use_cosine = use_cosine
        self.tau = float(tau_init)
        self.tau_min = float(tau_min)
        self.normalize_prototypes = normalize_prototypes
        self.eps = float(eps)

        # routing head
        self.routing = RoutingHead(d, mlp_hidden, use_mlp=use_mlp)

        # prototypes & bias
        self.prototypes = nn.Parameter(torch.empty(k, d))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(k))
        else:
            self.register_parameter('bias', None)

        # optional EMA
        self.ema_momentum = ema_momentum
        self.use_ema_for_logits = use_ema_for_logits
        if ema_momentum is not None:
            self.register_buffer('prototypes_ema', torch.empty(k, d), persistent=True)
            self.register_buffer('ema_initialized', torch.tensor(False), persistent=True)
        else:
            self.register_buffer('prototypes_ema', None, persistent=False)
            self.register_buffer('ema_initialized', None, persistent=False)

        self._reset_parameters()

    # ------------------------------ public utils ------------------------------

    @torch.no_grad()
    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    @torch.no_grad()
    def decay_tau(self, factor: float) -> float:
        """Multiply current tau by `factor` and clamp to >= tau_min. Returns new tau."""
        self.tau = max(self.tau * float(factor), self.tau_min)
        return self.tau

    # ------------------------------ init helpers ------------------------------

    def _reset_parameters(self) -> None:
        # Prototypes initialized with Xavier; bias zeros.
        nn.init.xavier_uniform_(self.prototypes)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # Initialize EMA buffer if used.
        if self.ema_momentum is not None:
            self.prototypes_ema.copy_(self.prototypes.detach())
            self.ema_initialized.fill_(True)

    # ------------------------------ EMA helpers -------------------------------

    @torch.no_grad()
    def _ema_update(self) -> None:
        if self.ema_momentum is None:
            return
        m = float(self.ema_momentum)
        if not bool(self.ema_initialized):
            self.prototypes_ema.copy_(self.prototypes.detach())
            self.ema_initialized.fill_(True)
            return
        self.prototypes_ema.mul_(m).add_(self.prototypes.detach(), alpha=1.0 - m)

    # ------------------------------- forward ----------------------------------

    def forward(
        self,
        H: torch.Tensor,
        *,
        tau: Optional[float] = None,
        hard: bool = True,
        gumbel_noise: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute assignments for node features H."""
        assert H.dim() == 3, f"Allocator.forward expects H of shape (B,V,D); got {tuple(H.shape)}"
        B, V, D = H.shape
        assert D == self.d, f"Allocator: feature dim mismatch, got D={D}, expected {self.d}"

        # (1) routing head
        H_tilde = self.routing(H)  # (B, V, D)

        # Optionally normalize prototypes after each update (keeps them on unit sphere)
        if self.normalize_prototypes: # 把原型向量拉回单位球，但这不是训练目标的一部分，所以不让它产生梯度。
            with torch.no_grad():
                self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)

        # (2) scoring against prototypes
        P = self.prototypes
        if self.ema_momentum is not None and self.use_ema_for_logits:
            P = self.prototypes_ema  # no grad path

        if self.use_cosine:
            Hn = F.normalize(H_tilde, dim=-1)
            Pn = F.normalize(P, dim=-1)
            logits = torch.einsum('bvd,kd->bvk', Hn, Pn)  # cosine similarity in [-1,1]
        else:
            scale = 1.0 / math.sqrt(self.d)
            logits = torch.einsum('bvd,kd->bvk', H_tilde, P) * scale

        if self.bias is not None:
            logits = logits + self.bias.view(1, 1, self.k)

        # (3) ST-Gumbel-Softmax
        if tau is None:
            tau = self.tau
        tau = max(float(tau), 1e-6)

        if gumbel_noise is None:
            gumbel_noise = self.training
        if gumbel_noise:
            g = _gumbel_noise_like(logits)
            logits_soft = (logits + g) / tau
        else:
            logits_soft = logits / tau

        S = F.softmax(logits_soft, dim=-1)  # (B, V, K)
        if hard:
            # one-hot along K
            idx = S.argmax(dim=-1, keepdim=True)
            Z = torch.zeros_like(S).scatter_(-1, idx, 1.0)
            Z_tilde = Z + (S - Z).detach()
        else:
            Z_tilde = S  # purely soft, if desired

        # (4) EMA update (after computing grad flow paths)
        if self.training and (self.ema_momentum is not None):
            self._ema_update()

        # statistics for diagnostics
        with torch.no_grad(): # 这些只是日志/正则的输入，不需要梯度
            usage = S.mean(dim=(0, 1))  # (K,)
            # entropy per node
            ent = -(S.clamp_min(1e-12).log() * S).sum(dim=-1).mean()  # scalar

        extras: Dict[str, torch.Tensor] = {
            'usage_per_k': usage,         # (K,)
            'mean_entropy': ent.unsqueeze(0),  # (1,)
            'tau': torch.tensor(tau, device=H.device, dtype=H.dtype).unsqueeze(0),
            'prototypes': P.detach(),     # (K, D)
        }

        return Z_tilde, S, logits, extras

    # ----------------------------- aggregation --------------------------------

    @staticmethod
    def aggregate(H: torch.Tensor, assign: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate node features H (B,V,D) into K modules using assignment weights.

        Args:
            H: (B, V, D) node features.
            assign: (B, V, K) assignment weights (soft or ST-soft).

        Returns:
            H_mod: (B, K, D) aggregated (weighted mean) features per module.
            weights: (B, K) sum of assignment weights per module (before normalization).
        """
        assert H.dim() == 3 and assign.dim() == 3, "aggregate expects H:(B,V,D), assign:(B,V,K)"
        B, V, D = H.shape
        _, V2, K = assign.shape
        assert V == V2, "aggregate: V mismatch between H and assign"

        # Weighted sum over nodes
        H_sum = torch.einsum('bvk,bvd->bkd', assign, H)  # (B, K, D)
        weights = assign.sum(dim=1)  # (B, K)
        H_mod = H_sum / (weights.clamp_min(eps).unsqueeze(-1))
        return H_mod, weights
