
# loss_utils.py
# A collection of pure loss/regularizer and scheduler utilities for modular fMRI models.
# All functions are framework-agnostic (no parameter creation), take tensors as inputs,
# and return scalar tensors suitable for logging and backprop.

from typing import Optional, Literal
import math
import torch
import torch.nn.functional as F


def _reduce(x: torch.Tensor, reduction: Literal['none','mean','sum'] = 'mean') -> torch.Tensor:
    if reduction == 'none':
        return x
    if reduction == 'sum':
        return x.sum()
    return x.mean()


def ce_loss(
    logits: torch.Tensor,                  # [B, C]
    target: torch.Tensor,                  # [B] (long) or [B, C] (one-hot / prob)
    *, label_smoothing: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,  # [C]
    reduction: Literal['none','mean','sum'] = 'mean',
) -> torch.Tensor:
    """
    Cross-entropy with optional label smoothing and class weights.
    """
    B, C = logits.shape
    if target.dtype == torch.long:
        if label_smoothing and label_smoothing > 0.0:
            # Convert to smoothed one-hot
            with torch.no_grad():
                smooth = torch.full((B, C), fill_value=label_smoothing / (C - 1),
                                    device=logits.device, dtype=logits.dtype)
                smooth.scatter_(1, target.view(-1,1), 1.0 - label_smoothing)
            logp = F.log_softmax(logits, dim=-1)
            loss = -(smooth * logp)
            if class_weights is not None:
                loss = loss * class_weights.unsqueeze(0)
            loss = loss.sum(dim=-1)  # [B]
            return _reduce(loss, reduction)
        else:
            return F.cross_entropy(logits, target, weight=class_weights, reduction=reduction)
    else:
        # target given as one-hot or soft labels
        logp = F.log_softmax(logits, dim=-1)
        tgt = target
        if label_smoothing and label_smoothing > 0.0:
            with torch.no_grad():
                uni = torch.full_like(tgt, 1.0 / C)
                tgt = (1.0 - label_smoothing) * tgt + label_smoothing * uni
        loss = -(tgt * logp)  # [B,C]
        if class_weights is not None:
            loss = loss * class_weights.unsqueeze(0)
        loss = loss.sum(dim=-1)  # [B]
        return _reduce(loss, reduction)


def balance_loss(
    usage_per_k: torch.Tensor,             # [K] or [B,K]
    *, target: Optional[torch.Tensor] = None,   # [K]; if None -> uniform
    reduction: Literal['none','mean','sum'] = 'mean',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL divergence between usage distribution and target (default uniform).
    usage_per_k can be unnormalized; we normalize internally along K.
    """
    u = usage_per_k
    if u.dim() == 2:
        # average over batch first (treat each row as a distribution estimate)
        u = u / (u.sum(dim=-1, keepdim=True) + eps)
        kl = (u * (torch.log(u + eps) - math.log(1.0 / u.size(-1)))).sum(dim=-1)  # [B]
        return _reduce(kl, reduction)
    else:
        u = u / (u.sum() + eps)
        if target is None:
            # uniform
            K = u.numel()
            kl = (u * (torch.log(u + eps) - math.log(1.0 / K))).sum()
        else:
            t = target / (target.sum() + eps)
            kl = (u * (torch.log((u + eps) / (t + eps)))).sum()
        return _reduce(kl, reduction)


def sharp_loss(
    S: torch.Tensor,                        # [B,V,K]
    *, reduction: Literal['none','mean','sum'] = 'mean',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Encourage sharp (low-entropy) assignments by minimizing mean entropy of S.
    L_sharp = mean_{B,V} sum_k S * log(S + eps)
    """
    ent = (S * torch.log(S + eps)).sum(dim=-1)    # [B,V]
    return _reduce(ent, reduction)


def _pairwise_cos(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # returns cosine similarities [N,M]
    A = F.normalize(A, dim=-1, eps=eps)
    B = F.normalize(B, dim=-1, eps=eps)
    return A @ B.T


def intra_compact_loss(
    H: torch.Tensor,                        # [B,V,D]
    S: torch.Tensor,                        # [B,V,K]   (soft assignments)
    P: torch.Tensor,                        # [K,D]     (prototypes)
    *, metric: Literal['l2','cos'] = 'l2',
    normalize: bool = False,
    detach_S: bool = False,
    detach_P: bool = False,
    reduction: Literal['none','mean','sum'] = 'mean',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Encourage nodes to be close to their (soft) assigned prototypes.
    L_intra = mean_{b,v} sum_k S[b,v,k] * d(H[b,v], P[k])
    where d is squared L2 or (1 - cosine).
    """
    if detach_S:
        S = S.detach()
    if detach_P:
        P_d = P.detach()
    else:
        P_d = P

    B, V, D = H.shape
    K = P.shape[0]

    if normalize and metric in ('l2','cos'):
        Hn = F.normalize(H, dim=-1, eps=eps)
        Pn = F.normalize(P_d, dim=-1, eps=eps)
    else:
        Hn = H
        Pn = P_d

    # Compute distances: [B,V,K]
    if metric == 'l2':
        # (h - p)^2 = h^2 + p^2 - 2 h.p
        h2 = (Hn * Hn).sum(dim=-1, keepdim=True)          # [B,V,1]
        p2 = (Pn * Pn).sum(dim=-1).unsqueeze(0).unsqueeze(0)  # [1,1,K]
        hp = torch.matmul(Hn, Pn.T)                        # [B,V,K]
        dist = (h2 + p2 - 2.0 * hp).clamp_min(0.0)
    elif metric == 'cos':
        sim = torch.matmul(F.normalize(Hn, dim=-1, eps=eps), F.normalize(Pn, dim=-1, eps=eps).T)
        dist = 1.0 - sim
    else:
        raise ValueError("metric must be 'l2' or 'cos'")

    loss = (S * dist).sum(dim=-1)   # [B,V]
    return _reduce(loss, reduction)


def inter_separate_loss(
    P: torch.Tensor,                         # [K,D]
    *, mode: Literal['orth','cos_exp'] = 'orth',
    gamma: float = 0.1,
    normalize: bool = True,
    reduction: Literal['none','mean','sum'] = 'mean',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Encourage prototypes to be mutually separated.
    - 'orth': || P^T P - I ||_F^2 (after L2 normalization if normalize=True)
    - 'cos_exp': sum_{i!=j} exp( cos(p_i, p_j) / gamma )
    """
    K, D = P.shape
    if normalize:
        Pn = F.normalize(P, dim=-1, eps=eps)
    else:
        Pn = P

    if mode == 'orth':
        G = Pn @ Pn.T                                    # [K,K]
        I = torch.eye(K, device=P.device, dtype=P.dtype)
        loss = ((G - I) ** 2).sum()
        return _reduce(loss, reduction)
    elif mode == 'cos_exp':
        S = Pn @ Pn.T                                    # [K,K], cosine if normalized
        S = S - torch.diag_embed(torch.diag(S))          # zero-out diagonal
        loss = torch.exp(S / gamma).sum()
        return _reduce(loss, reduction)
    else:
        raise ValueError("mode must be 'orth' or 'cos_exp'")


def empty_module_penalty(
    count_per_k: torch.Tensor,              # [B,K] or [K]
    *, min_count: float = 1.0,
    mode: Literal['hinge','count'] = 'hinge',
    reduction: Literal['none','mean','sum'] = 'mean',
) -> torch.Tensor:
    """
    Penalize empty or under-utilized modules.
    - If mode='hinge': mean( relu(min_count - mean_count_per_k) )
    - If mode='count': mean( 1.0 * (mean_count_per_k <= 1e-6) )
    """
    c = count_per_k
    if c.dim() == 2:
        c = c.mean(dim=0)          # average over batch -> [K]
    if mode == 'hinge':
        loss = F.relu(min_count - c)  # [K]
        return _reduce(loss, reduction)
    elif mode == 'count':
        loss = (c <= 1e-6).to(c.dtype)  # [K]
        return _reduce(loss, reduction)
    else:
        raise ValueError("mode must be 'hinge' or 'count'")


def prototype_norm_loss(
    P: torch.Tensor,                         # [K,D]
    *, target_norm: Optional[float] = None,
    reduction: Literal['none','mean','sum'] = 'mean',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Optional regularizer to keep prototype norms bounded or near a target norm.
    If target_norm is None, encourage unit norm via (||p||_2 - 1)^2.
    """
    norms = torch.linalg.norm(P, dim=-1)  # [K]
    if target_norm is None:
        loss = (norms - 1.0) ** 2
    else:
        loss = (norms - float(target_norm)) ** 2
    return _reduce(loss, reduction)


# ---- Schedulers --------------------------------------------------------------

def schedule_linear(
    step: int,
    *, start: float, end: float, total_steps: int,
    clamp: bool = True,
) -> float:
    """Linear schedule from start to end across total_steps (inclusive of step=0)."""
    if total_steps <= 0:
        return end
    t = step / float(total_steps)
    val = (1 - t) * start + t * end
    if clamp:
        lo, hi = (min(start, end), max(start, end))
        val = float(max(lo, min(hi, val)))
    return float(val)


def schedule_cosine(
    step: int,
    *, start: float, end: float, total_steps: int,
    clamp: bool = True,
) -> float:
    """Cosine schedule between start and end across total_steps."""
    if total_steps <= 0:
        return end
    t = step / float(total_steps)
    val = end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))
    if clamp:
        lo, hi = (min(start, end), max(start, end))
        val = float(max(lo, min(hi, val)))
    return float(val)
