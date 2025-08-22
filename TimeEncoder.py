import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class PatchEmbedding(nn.Module):
    """
    1D Patch Embedding for time series.

    Input:  (B, C, T)
    Output: (B, L, D)   # L = number of patches, D = d_model

    Optional residual path inside the patch embedding block can be toggled
    via `use_residual` (default: False).
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        patch_len: int,
        stride: int,
        dropout: float = 0.0,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_len, stride=stride, bias=True)
        self.use_residual = use_residual
        # Residual branch projects the same raw window with an independent projection.
        # Kept separate so it can be turned on/off explicitly.
        self.res_proj = (
            nn.Conv1d(in_channels, d_model, kernel_size=patch_len, stride=stride, bias=False)
            if use_residual
            else None
        )
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(d_model)  # token-wise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        tokens = self.proj(x).transpose(1, 2)  # (B, L, D)
        if self.use_residual:
            res = self.res_proj(x).transpose(1, 2)  # (B, L, D)
            tokens = tokens + res
        tokens = self.dropout(tokens)
        return self.norm(tokens)


class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer with the two REQUIRED residual paths:
      1) x + Dropout(MultiHeadAttention(LayerNorm(x)))
      2) x + Dropout(FFN(LayerNorm(x)))

    This is a Pre-LN formulation which typically trains more stably.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention block with residual
        y = self.attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + self.drop1(y)

        # Feed-forward block with residual
        y = self.ff(self.norm2(x))
        x = x + self.drop2(y)
        return x


class PatchTST(nn.Module):
    """
    TimeEncoder (PatchTST-style) to replace ST-GCFE.

    Expected input (to match existing pipeline): (B, C, T, V, M)
      - B: batch size
      - C: channels per node (usually 1 for rs-fMRI)
      - T: temporal length
      - V: number of nodes/ROIs
      - M: persons (kept for API parity; often 1)

    Output: (B, out_channels, V)

    "分块式规格表"（简要）
    ------------------------------------------------------------
    - Patchify: Conv1d(kernel_size=patch_len, stride=stride)  -> tokens (L, D)
    - Patch Embedding Residual: switchable (default False)
    - Positional Encoding: learnable per-token (length = num_patches)
    - Transformer Encoder: n_layers × [Pre-LN MHA + residual, Pre-LN FFN + residual]
    - Token Pooling: mean over tokens
    - Head: Linear(D -> out_channels)
    ------------------------------------------------------------
    """

    def __init__(
        self,
        *,
        seq_len: int,              # full T before patching
        num_nodes: int,            # V
        in_channels: int = 1,      # C
        patch_len: int = 48,
        stride: int = 16,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.2,
        out_channels: int = 128,
        use_patch_residual: bool = False,

        input_norm: str = "bn",  # {"bn", "revin"}
        num_person: int = 1,
        revin_affine: bool = True,
        revin_eps: float = 1e-5,
        roi_IDstd: float = 0.02, # ROI位置嵌入的初始化标准差，均值为0
    ) -> None:
        super().__init__()
        assert seq_len > 0 and patch_len > 0 and stride > 0
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.out_channels = out_channels
        self.use_patch_residual = use_patch_residual


        # compute number of patches deterministically for PE
        self.num_patches = self._num_patches(seq_len, patch_len, stride)
        if self.num_patches <= 0:
            raise ValueError("num_patches must be positive; check seq_len/patch_len/stride")

        self.input_norm = input_norm.lower()
        self.num_person = num_person
        if self.input_norm == "bn":
            # 等价于 ST-GCFE: nn.BatchNorm1d(num_person * in_channels * num_nodes)
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)
        elif self.input_norm == "revin":
            self.revin = RevIN(num_channels=in_channels, affine=revin_affine, eps=revin_eps)
        else:
            raise ValueError("input_norm must be 'bn' or 'revin'")

        # Embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            dropout=dropout,
            use_residual=use_patch_residual,
        )



        # Encoder stack
        if d_ff is None:
            d_ff = 4 * d_model
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # Projection head to the channel expected by downstream HieraFormer
        self.head = nn.Linear(d_model, out_channels)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # === Learned ROI ID Embedding（绝对位置/身份） ===
        self.roi_embed = nn.Embedding(num_nodes, out_channels)
        nn.init.normal_(self.roi_embed.weight, mean=0.0, std=roi_IDstd)
    @staticmethod
    def _num_patches(seq_len: int, patch_len: int, stride: int) -> int:
        return 1 + (seq_len - patch_len) // stride if seq_len >= patch_len else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, V, M)
        returns: (B, out_channels, V)
        """
        B, C, T, V, M = x.shape
        if self.input_norm == "bn":
            # [B, C, T, V, M] -> [B, M*V*C, T] 做 BN1d -> 还原
            x_bn = x.permute(0, 4, 3, 1, 2).contiguous().view(B, M * V * C, T)
            x_bn = self.data_bn(x_bn)
            x = x_bn.view(B, M, V, C, T).permute(0, 3, 4, 2, 1).contiguous()
        elif self.input_norm == "revin":
            x = self.revin(x)
        # (B, C, V, T, M) -> merge V and M into batch for weight sharing across nodes
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(B * V * M, C, T)

        # Patchify & embed -> (B*V*M, L, D)
        tokens = self.patch_embed(x)
        L = tokens.size(1)

        # Transformer encoder
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.final_norm(tokens)  # (B*V*M, L, D)

        # Pool across tokens and project
        feat = tokens.mean(dim=1)  # (B*V*M, D)
        feat = self.head(feat)      # (B*V*M, out_channels)

        # 先还原为 (B, V, M, D)，逐 ROI 相加身份向量，再按你的管线继续
        feat = feat.view(B, V, M, self.out_channels)  # (B, V, M, D)
        ids = torch.arange(V, device=feat.device)     # [0..V-1]
        roi_bias = self.roi_embed(ids)                # (V, D)
        feat = feat + roi_bias.unsqueeze(0).unsqueeze(2)  # (1, V, 1, D) 广播

        # 回到原有路径：转为 (B, D, V, M) 并对 M 求均值 → (B, D, V)
        feat = feat.permute(0, 3, 1, 2).contiguous()  # (B, D, V, M)
        feat = feat.mean(dim=3)                       # (B, D, V)
        return feat
