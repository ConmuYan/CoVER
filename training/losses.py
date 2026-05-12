from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_reasoner_loss(
    outputs: dict[str, Tensor],
    y: Tensor,
    targets: dict[str, Tensor],
    accepted_mask: Tensor,
    pos_weight: Tensor | None = None,
    lambda_evi: float = 0.5,
    use_type_loss: bool = True,
    use_evidence_loss: bool = True,
) -> tuple[Tensor, dict[str, float]]:
    final_logit = outputs["final_logit"]
    type_logits = outputs["type_logits"]
    pos_logits = outputs["pos_logits"]
    neg_logits = outputs["neg_logits"]

    task_loss = F.binary_cross_entropy_with_logits(
        final_logit, y.float(), pos_weight=pos_weight,
    )

    loss_dict = {"task": task_loss.item(), "type": 0.0, "pos": 0.0, "neg": 0.0}

    if accepted_mask.sum() == 0:
        return task_loss, loss_dict

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

    return total_loss, loss_dict
