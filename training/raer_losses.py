"""RAER teacher loss.

The canonical RAER teacher uses a single supervised objective over the frozen
base plus bounded relation-evidence residual:

    L = BCE(final_logit, y)

Relation gates, residual magnitude, and dominance are logged as diagnostics,
not used as auxiliary optimization terms.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _zero_like(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def build_relation_evidence_distribution(
    relation_features: Tensor,
    num_relations: int,
    rel_stat_dim: int = 9,
    tau: float = 1.5,
    eps: float = 1e-8,
) -> Tensor:
    """Build a score-blind diagnostic distribution from relation statistics."""
    if relation_features.ndim != 2:
        raise ValueError(f"relation_features must be rank-2, got {tuple(relation_features.shape)}")
    num_relations = int(num_relations)
    rel_stat_dim = int(rel_stat_dim)
    expected = num_relations * rel_stat_dim
    if relation_features.shape[1] != expected:
        raise ValueError(
            f"relation_features dim mismatch: {relation_features.shape[1]} != {expected}"
        )

    rel = relation_features.detach().float().view(-1, num_relations, rel_stat_dim)
    degree = rel[..., 0].abs()
    if rel_stat_dim > 2:
        degree = degree + rel[..., 1].clamp_min(0.0) + rel[..., 2].clamp_min(0.0)
    deviation = rel[..., 3].clamp_min(0.0) if rel_stat_dim > 3 else torch.zeros_like(degree)
    inconsistency = (1.0 - rel[..., 4]).clamp_min(0.0) if rel_stat_dim > 4 else torch.zeros_like(degree)
    z_fraction = rel[..., 5].clamp_min(0.0) if rel_stat_dim > 5 else torch.zeros_like(degree)
    proto_margin = rel[..., 8].abs() if rel_stat_dim > 8 else torch.zeros_like(degree)

    strength = (
        0.5 * degree
        + deviation
        + 0.75 * inconsistency
        + z_fraction
        + proto_margin
    )
    strength = strength - strength.mean(dim=-1, keepdim=True)
    return F.softmax(strength / max(float(tau), eps), dim=-1).detach()


def compute_raer_loss(
    outputs: dict[str, Tensor],
    y: Tensor,
    train_mask: Tensor,
    pos_weight: Tensor | float | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compute canonical RAER BCE loss and diagnostics."""
    final_logit = outputs["final_logit"]
    delta_rel = outputs.get("delta_rel")
    rel_gate = outputs.get("relation_gate")

    device = final_logit.device
    train_bool = train_mask.to(device=device).view(-1).to(torch.bool)
    y_f = y.to(device=device, dtype=torch.float32).view(-1)
    if pos_weight is None:
        pw: Tensor | None = None
    elif isinstance(pos_weight, Tensor):
        pw = pos_weight.to(device=device)
    else:
        pw = torch.tensor(float(pos_weight), device=device)

    if bool(train_bool.any()):
        loss = F.binary_cross_entropy_with_logits(
            final_logit[train_bool],
            y_f[train_bool],
            pos_weight=pw,
        )
    else:
        loss = _zero_like(final_logit)

    with torch.no_grad():
        if bool(train_bool.any()) and delta_rel is not None:
            mean_abs_delta_rel = float(delta_rel[train_bool].abs().mean().item())
        else:
            mean_abs_delta_rel = 0.0

        if bool(train_bool.any()) and rel_gate is not None and rel_gate.numel() > 0:
            eps = 1e-8
            entropy = -(rel_gate * torch.log(rel_gate.clamp_min(eps))).sum(dim=-1)
            mean_gate_entropy = float(entropy[train_bool].mean().item())
            num_relations = rel_gate.shape[1]
            if num_relations >= 2:
                dominance = (1.0 - entropy / math.log(num_relations)).clamp(0.0, 1.0)
                mean_dominance_rho = float(dominance[train_bool].mean().item())
            else:
                mean_dominance_rho = 0.0
        else:
            mean_gate_entropy = 0.0
            mean_dominance_rho = 0.0

    stats = {
        "total": float(loss.detach().item()),
        "l_cls": float(loss.detach().item()),
        "mean_abs_delta_rel": mean_abs_delta_rel,
        "mean_gate_entropy": mean_gate_entropy,
        "mean_dominance_rho": mean_dominance_rho,
    }
    return loss, stats


__all__ = [
    "build_relation_evidence_distribution",
    "compute_raer_loss",
]
