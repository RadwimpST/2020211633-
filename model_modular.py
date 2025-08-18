
import torch
import torch.nn as nn
from typing import Dict, Optional, Literal

# Local modules
from TimeEncoder import PatchTST
from Allocator import Allocator
from MFeature import MFeature


def _bvt_to_bctvm(x: torch.Tensor) -> torch.Tensor:
    """
    Convert [B, V, T] -> [B, C=1, T, V, M=1] expected by PatchTST.
    """
    assert x.dim() == 3, f"expected x[B,V,T], got shape {tuple(x.shape)}"
    B, V, T = x.shape
    x = x.permute(0, 2, 1).contiguous()  # [B, T, V]
    x = x.unsqueeze(1).unsqueeze(-1)     # [B, 1, T, V, 1]
    return x


class ClassifierHead(nn.Module):
    """
    Simple MLP head for classification on top of the global readout vector.
    """
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(d_model * 2, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelModular(nn.Module):
    """
    Top-level modular model that wires together:
        TimeEncoder (PatchTST) -> Allocator (ST-Gumbel) -> MFeature (module extractor) -> Classifier.

    Responsibilities:
        - define the end-to-end forward pass from x[B,V,T] to logits[B,C].
        - DO NOT compute any losses/regularizers inside; return all intermediates for train.py.
    """
    def __init__(
        self,
        *,
        seq_len: int,
        num_nodes: int,
        num_classes: int,
        k_modules: int,
        d_model: int = 64,            # must match PatchTST out_channels and MFeature.d_model
        d_embed: int = 16,            # module embedding dim used by MFeature FiLM/CLS
        use_module_embed: bool = True,
        dropout: float = 0.1,
        # pass-through kwargs
        encoder_kwargs: Optional[dict] = None,
        allocator_kwargs: Optional[dict] = None,
        mfeature_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        encoder_kwargs = dict(encoder_kwargs or {})
        allocator_kwargs = dict(allocator_kwargs or {})
        mfeature_kwargs = dict(mfeature_kwargs or {})

        # ---- Time Encoder (PatchTST) ----
        # Ensure out_channels == d_model for downstream modules
        self.encoder = PatchTST(
            seq_len=seq_len,
            num_nodes=num_nodes,
            out_channels=d_model,
            **encoder_kwargs
        )

        # ---- Allocator ----
        self.allocator = Allocator(
            d=d_model,
            k=k_modules,
            **allocator_kwargs
        )

        # ---- Module Feature Extractor ----
        self.mfeature = MFeature(
            k_modules=k_modules,
            d_model=d_model,
            d_embed=d_embed,
            **mfeature_kwargs
        )

        # ---- Optional module embedding table E ----
        self.module_embed: Optional[nn.Parameter]
        if use_module_embed:
            self.module_embed = nn.Parameter(torch.zeros(k_modules, d_embed))
            nn.init.normal_(self.module_embed, mean=0.0, std=0.02)
        else:
            self.register_parameter('module_embed', None)

        # ---- Classifier head ----
        self.head = ClassifierHead(d_model, num_classes, dropout=dropout)

        # Keep shapes for reference
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.k_modules = k_modules
        self.d_model = d_model
        self.d_embed = d_embed
        self.num_classes = num_classes

    @torch.no_grad()
    def set_tau(self, tau: float) -> None:
        """Convenience wrapper to set allocator temperature from train.py."""
        if hasattr(self.allocator, 'set_tau'):
            self.allocator.set_tau(tau)

    @torch.no_grad()
    def decay_tau(self, factor: float) -> float:
        """Convenience wrapper to decay allocator temperature from train.py."""
        if hasattr(self.allocator, 'decay_tau'):
            return self.allocator.decay_tau(factor)
        return float('nan')

    def forward(
        self,
        x_bvt: torch.Tensor,                                 # [B, V, T]
        *,
        assign_source: Literal['hard', 'soft'] = 'hard',
        detach_mask: bool = True,
        return_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-end forward.

        Args:
            x_bvt: Input fMRI time series [B, V, T].
            assign_source: 'hard' -> use Z_tilde; 'soft' -> use S for MFeature masking.
            detach_mask: if True, do not backprop through the assignment mask in MFeature.
            return_tokens: if False, 'm_tokens' is dropped from outputs.

        Returns (dict):
            logits: [B, C]
            H: [B, V, D]                   # node features after time encoder
            assign_hard: [B, V, K]
            assign_soft: [B, V, K]
            alloc_logits: [B, V, K]
            alloc_extras: {usage_per_k[K], mean_entropy[1], tau[1], prototypes[K,D]}
            m_tokens: [B, K, D]            # (if return_tokens=True)
            readout: [B, D]
            mfeature_diag: {count_per_k[B,K], empty_mask[B,K]}
        """
        # 1) Time encoding
        x_bctvm = _bvt_to_bctvm(x_bvt)                      # [B,1,T,V,1]
        H_bdv = self.encoder(x_bctvm)                       # [B, D, V]
        H = H_bdv.permute(0, 2, 1).contiguous()             # [B, V, D]

        # 2) Allocator (ST-Gumbel)
        Z_tilde, S, alloc_logits, alloc_extras = self.allocator(H)  # shapes: [B,V,K], [B,V,K], [B,V,K], dict

        # 3) Choose assignment source for module extractor
        if assign_source == 'hard':
            A = Z_tilde
        elif assign_source == 'soft':
            A = S
        else:
            raise ValueError(f"assign_source must be 'hard' or 'soft', got {assign_source}")

        # 4) Module-level feature extractor
        E = self.module_embed if self.module_embed is not None else None
        mf_out = self.mfeature(H, A, E=E, detach_mask=detach_mask, return_tokens=return_tokens)
        m_tokens = mf_out.get('m_tokens', None)  # [B,K,D] or None
        readout = mf_out['readout']              # [B,D]
        mdiag = mf_out.get('diag', {})

        # 5) Classifier
        logits = self.head(readout)             # [B,C]

        out: Dict[str, torch.Tensor] = {
            'logits': logits,
            'H': H,
            'assign_hard': Z_tilde,
            'assign_soft': S,
            'alloc_logits': alloc_logits,
            'readout': readout,
            'mfeature_diag': mdiag,
        }
        # alloc_extras is a dict of tensors; attach as a sub-dict
        out['alloc_extras'] = alloc_extras
        if return_tokens and (m_tokens is not None):
            out['m_tokens'] = m_tokens
        return out
