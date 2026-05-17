"""Audit loss for LEQA (Phase 3).

Per FINAL_PROPOSAL.md section 3.4::

    L_audit = mean_i[ sum_t softmax_norm(|a_{t,i}|) * (1 - q_{t,i}) * clip(|delta_rel_i|, 0, c) ]

where:
  - a_{t,i}: reasoner attention over evidence token t for node i, shape (N, T)
  - q_{t,i}: LoRA quality score in [0, 1], shape (N, T)
  - delta_rel_i: rel-branch residual, shape (N,)
  - c: clip bound (default 2.0)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_audit_loss(
    reasoner_attention: Tensor,
    lora_q: Tensor,
    delta_rel: Tensor,
    clip: float = 2.0,
) -> Tensor:
    """Compute the LEQA audit loss.

    Args:
        reasoner_attention: Per-token attention weights ``(N, T)``.
        lora_q: Per-token quality scores in [0, 1] from LoRA ``(N, T)``.
        delta_rel: Relation-branch residual ``(N,)``.
        clip: Upper clip bound for ``|delta_rel|``.

    Returns:
        Scalar loss tensor with gradient connectivity to all three inputs.
    """
    if reasoner_attention.dim() != 2:
        raise ValueError(
            f"reasoner_attention must be 2D (N, T); got {tuple(reasoner_attention.shape)}"
        )
    if lora_q.shape != reasoner_attention.shape:
        raise ValueError(
            f"lora_q shape {tuple(lora_q.shape)} != "
            f"reasoner_attention shape {tuple(reasoner_attention.shape)}"
        )
    if delta_rel.dim() != 1 or delta_rel.shape[0] != reasoner_attention.shape[0]:
        raise ValueError(
            f"delta_rel must be 1D with length N={reasoner_attention.shape[0]}; "
            f"got shape {tuple(delta_rel.shape)}"
        )

    # softmax_norm(|a_{t,i}|) over token dimension
    abs_attention = reasoner_attention.abs()
    attn_weights = F.softmax(abs_attention, dim=-1)  # (N, T)

    # (1 - q_{t,i}): penalty for low-quality evidence
    quality_penalty = 1.0 - lora_q  # (N, T)

    # clip(|delta_rel_i|, 0, c)
    clipped_delta = delta_rel.abs().clamp(max=clip)  # (N,)

    # Per-node loss: sum_t attn_weights * quality_penalty * clipped_delta
    per_token_loss = attn_weights * quality_penalty  # (N, T)
    per_node_loss = per_token_loss.sum(dim=-1)  # (N,)
    per_node_loss = per_node_loss * clipped_delta  # (N,)

    return per_node_loss.mean()


__all__ = ["compute_audit_loss"]
