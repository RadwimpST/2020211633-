from torch.nn import TransformerEncoderLayer
from torch import Tensor
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F

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