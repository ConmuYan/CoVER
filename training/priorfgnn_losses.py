"""PriorF-GNN loss functions used by RAER-FD strong-base experiments.

Faithful copy of PriorF-GNN/lghgcl/models/losses.py:
- focal_loss_with_logits: numerically-stable Focal Loss on raw logits
- sdcl_loss_v4: Structure-Discrepancy Contrastive Loss (SupCon variant)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-6


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    p_t = p_t.clamp(min=_EPS, max=1.0 - _EPS)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_weight = alpha_t * (1.0 - p_t).pow(gamma)
    loss = focal_weight * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def sdcl_loss_v4(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    hsd_scores: torch.Tensor,
    temperature: float = 0.1,
    num_anchors: int = 128,
    quantile: float = 0.3,
    num_rounds: int = 1,
) -> torch.Tensor:
    N = embeddings.size(0)
    if N < 4:
        return embeddings.new_zeros(())

    z = F.normalize(embeddings, dim=-1)
    labels_long = labels.long()

    q_hi = torch.quantile(hsd_scores.float(), 1.0 - quantile)
    hi_mask = hsd_scores >= q_hi

    hi_fraud = (hi_mask & (labels_long == 1)).nonzero(as_tuple=True)[0]
    hi_normal = (hi_mask & (labels_long == 0)).nonzero(as_tuple=True)[0]

    if len(hi_fraud) < 2 or len(hi_normal) < 2:
        return embeddings.new_zeros(())

    half_k = num_anchors // 2
    n_f = min(half_k, len(hi_fraud))
    n_n = min(half_k, len(hi_normal))

    losses = []
    for _ in range(num_rounds):
        sel_f = hi_fraud[torch.randperm(len(hi_fraud), device=z.device)[:n_f]]
        sel_n = hi_normal[torch.randperm(len(hi_normal), device=z.device)[:n_n]]
        sel = torch.cat([sel_f, sel_n])

        k = len(sel)
        if k < 4:
            continue

        z_sel = z[sel]
        y_sel = labels_long[sel]

        sim = z_sel @ z_sel.T / temperature

        self_mask = ~torch.eye(k, dtype=torch.bool, device=z.device)
        labels_eq = y_sel.unsqueeze(0) == y_sel.unsqueeze(1)
        pos_mask = labels_eq & self_mask

        pos_count = pos_mask.float().sum(dim=1)
        valid = pos_count > 0
        if valid.sum() < 2:
            continue

        sim_max = sim.detach().max(dim=1, keepdim=True).values
        sim = sim - sim_max

        exp_sim = torch.exp(sim) * self_mask.float()
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        log_prob = sim - log_denom

        mean_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / (pos_count + 1e-8)
        round_loss = -mean_log_prob[valid].mean()

        log_k = torch.log(torch.tensor(max(k - 1, 2), dtype=torch.float32))
        losses.append(round_loss / log_k)

    if not losses:
        return embeddings.new_zeros(())

    return torch.stack(losses).mean()
