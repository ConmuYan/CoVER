from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_reasoner_loss(
    outputs: dict[str, Tensor],
    y: Tensor,
    targets: dict[str, Tensor],
    accepted_mask: Tensor,
    base_logit: Tensor | None = None,
    pos_weight: Tensor | None = None,
    lambda_evi: float = 0.5,
    use_type_loss: bool = True,
    use_evidence_loss: bool = True,
    residual_l2_weight: float = 0.0,
    max_shift_penalty_weight: float = 0.0,
    max_abs_shift: float = 2.0,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the reasoner training loss.

    The total loss is::

        task_loss
        + lambda_evi * evidence_loss
        + residual_l2_weight * residual_l2
        + max_shift_penalty_weight * shift_penalty

    where ``residual_l2 = mean((final_logit - base_logit)^2)`` and
    ``shift_penalty = mean(relu(|final_logit - base_logit| - max_abs_shift)^2)``.

    All optional terms default to 0 so the function is backward-compatible.
    """
    final_logit = outputs["final_logit"]
    type_logits = outputs["type_logits"]
    pos_logits = outputs["pos_logits"]
    neg_logits = outputs["neg_logits"]

    task_loss = F.binary_cross_entropy_with_logits(
        final_logit, y.float(), pos_weight=pos_weight,
    )

    loss_dict: dict[str, float] = {
        "task": task_loss.item(),
        "type": 0.0,
        "pos": 0.0,
        "neg": 0.0,
        "residual_l2": 0.0,
        "shift_penalty": 0.0,
    }

    if accepted_mask.sum() > 0:
        idx = accepted_mask.bool()

        type_loss = torch.tensor(0.0, device=task_loss.device)
        pos_loss = torch.tensor(0.0, device=task_loss.device)
        neg_loss = torch.tensor(0.0, device=task_loss.device)

        if use_type_loss:
            type_loss = F.cross_entropy(type_logits[idx], targets["risk_type_id"][idx])
            loss_dict["type"] = type_loss.item()

        if use_evidence_loss:
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits[idx], targets["pos_mask"][idx].float(),
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits[idx], targets["neg_mask"][idx].float(),
            )
            loss_dict["pos"] = pos_loss.item()
            loss_dict["neg"] = neg_loss.item()

        evi_loss = type_loss + pos_loss + neg_loss
        total_loss = task_loss + lambda_evi * evi_loss
    else:
        total_loss = task_loss

    if base_logit is not None and residual_l2_weight > 0:
        residual_l2 = ((final_logit - base_logit.view(-1)) ** 2).mean()
        total_loss = total_loss + residual_l2_weight * residual_l2
        loss_dict["residual_l2"] = residual_l2.item()

    if base_logit is not None and max_shift_penalty_weight > 0:
        shift = (final_logit - base_logit.view(-1)).abs() - max_abs_shift
        shift_penalty = (F.relu(shift) ** 2).mean()
        total_loss = total_loss + max_shift_penalty_weight * shift_penalty
        loss_dict["shift_penalty"] = shift_penalty.item()

    return total_loss, loss_dict
