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
    """Compute the reasoner training loss (legacy two-term version).

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


def _compute_det_loss(
    final_logit: Tensor,
    y: Tensor,
    base_logits: Tensor,
    train_mask: Tensor,
    pos_weight: float | None,
    correction_weight: float,
) -> tuple[Tensor, dict[str, int]]:
    """L_det: Correction-aware final detection loss with per-sample weights."""
    # Only use train labels for correction weights
    y_train = y[train_mask]
    final_train = final_logit[train_mask]
    base_train = base_logits[train_mask]

    # Build per-sample correction weights
    correction_weights = torch.ones_like(y_train, dtype=torch.float)

    base_prob = torch.sigmoid(base_train)
    base_wrong_fn = (y_train == 1) & (base_prob < 0.5)
    base_wrong_fp = (y_train == 0) & (base_prob >= 0.5)
    base_high_loss = base_wrong_fn | base_wrong_fp

    correction_weights[base_wrong_fn] *= correction_weight
    correction_weights[base_wrong_fp] *= correction_weight

    # base high-loss nodes get extra weight
    if base_high_loss.any():
        correction_weights[base_high_loss] *= 1.5

    # BCE with per-sample weights
    pw = None
    if pos_weight is not None:
        pw = torch.tensor(pos_weight, device=final_train.device)

    per_sample_loss = F.binary_cross_entropy_with_logits(
        final_train, y_train.float(), pos_weight=pw, reduction="none",
    )
    det_loss = (per_sample_loss * correction_weights).mean()

    fn_count = int(base_wrong_fn.sum().item())
    fp_count = int(base_wrong_fp.sum().item())

    return det_loss, {"fn_count": fn_count, "fp_count": fp_count}


def _compute_err_loss(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    accepted_mask: Tensor,
    direction_ids: Tensor | None = None,
    lambda_direction: float = 1.0,
) -> tuple[Tensor, float]:
    """L_err: Contract-verified ERR distillation loss for accepted nodes only."""
    if accepted_mask.sum() == 0:
        return torch.tensor(0.0, device=outputs["final_logit"].device), 0.0

    idx = accepted_mask.bool()
    type_logits = outputs["type_logits"][idx]
    pos_logits = outputs["pos_logits"][idx]
    neg_logits = outputs["neg_logits"][idx]

    type_loss = F.cross_entropy(type_logits, targets["risk_type_id"][idx])
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, targets["pos_mask"][idx].float(),
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits, targets["neg_mask"][idx].float(),
    )

    err_loss = type_loss + pos_loss + neg_loss

    direction_ce_val = 0.0
    if direction_ids is not None and lambda_direction > 0:
        dir_ids_acc = direction_ids[idx]
        valid = dir_ids_acc >= 0
        if valid.any():
            dir_logits = outputs.get("direction_logits")
            if dir_logits is not None:
                valid_indices = valid.nonzero(as_tuple=True)[0]
                direction_ce = F.cross_entropy(
                    dir_logits[idx][valid_indices],
                    dir_ids_acc[valid_indices],
                )
                err_loss = err_loss + lambda_direction * direction_ce
                direction_ce_val = direction_ce.item()

    return err_loss, direction_ce_val


def _compute_signed_loss(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    accepted_mask: Tensor,
    signed_margin: float = 0.2,
    direction_ids: Tensor | None = None,
) -> Tensor:
    """L_signed: DuoKD-inspired signed evidence separation with direction awareness.

    direction_ids values: 0=increase_risk, 1=decrease_risk, 2=uncertain.
    - increase_risk: standard margin (pos_head(support) > pos_head(counter))
    - decrease_risk: reversed margin (pos_head(counter) > pos_head(support))
    - uncertain: weight=0 (skip signed for this node)
    """
    if accepted_mask.sum() == 0:
        return torch.tensor(0.0, device=outputs["final_logit"].device)

    idx = accepted_mask.bool()
    pos_logits = outputs["pos_logits"][idx]
    neg_logits = outputs["neg_logits"][idx]
    pos_mask = targets["pos_mask"][idx]
    neg_mask = targets["neg_mask"][idx]

    pos_head = pos_logits  # (N_acc, num_slots)
    neg_head = neg_logits  # (N_acc, num_slots)

    total_loss = torch.tensor(0.0, device=pos_logits.device)
    num_terms = 0

    for i in range(pos_logits.shape[0]):
        # Direction weight and margin direction
        if direction_ids is not None:
            dir_id = direction_ids[idx][i].item()
            if dir_id == 2:  # uncertain
                continue  # skip signed loss for uncertain nodes
            margin_sign = -1.0 if dir_id == 1 else 1.0  # decrease_risk reverses
        else:
            margin_sign = 1.0

        supp_slots = pos_mask[i].bool()
        cnt_slots = neg_mask[i].bool()
        has_supp = supp_slots.any()
        has_cnt = cnt_slots.any()

        if has_supp and has_cnt:
            supp_vals_p = pos_head[i, supp_slots]
            cnt_vals_p = pos_head[i, cnt_slots]
            margin_p = F.relu(signed_margin - margin_sign * (supp_vals_p.unsqueeze(1) - cnt_vals_p.unsqueeze(0)))
            total_loss = total_loss + margin_p.mean()

            cnt_vals_n = neg_head[i, cnt_slots]
            supp_vals_n = neg_head[i, supp_slots]
            margin_n = F.relu(signed_margin + margin_sign * (cnt_vals_n.unsqueeze(1) - supp_vals_n.unsqueeze(0)))
            total_loss = total_loss + margin_n.mean()
            num_terms += 2
        elif has_supp:
            target = torch.ones_like(pos_head[i, supp_slots])
            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                pos_head[i, supp_slots], target,
            )
            num_terms += 1
        elif has_cnt:
            target = torch.ones_like(neg_head[i, cnt_slots])
            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                neg_head[i, cnt_slots], target,
            )
            num_terms += 1

    if num_terms == 0:
        return torch.tensor(0.0, device=pos_logits.device)

    return total_loss / num_terms


def _compute_corr_loss(
    final_logit: Tensor,
    base_logits: Tensor,
    y: Tensor,
    targets: dict[str, Tensor],
    accepted_mask: Tensor,
    train_mask: Tensor,
    risk_type_names: list[str] | None,
    correction_margin: float = 0.05,
    anchor_weight: float = 0.001,
    max_shift_penalty_weight: float = 0.001,
    max_allowed_shift: float = 0.3,
    direction_ids: Tensor | None = None,
) -> tuple[Tensor, int, int, dict[str, float]]:
    """L_corr: Selective evidence-corrective residual loss with direction agreement.

    Direction-label agreement:
    - Base FN + direction=increase_risk → agrees (both say more fraud-like)
    - Base FP + direction=decrease_risk → agrees (both say more benign-like)
    - All other combos → disagrees or uncertain → use anchor behavior

    Returns (corr_loss, fn_count, fp_count, direction_stats).
    """
    residual_shift = final_logit - base_logits.detach()

    y_train = y[train_mask]
    shift_train = residual_shift[train_mask]
    base_train = base_logits[train_mask]
    base_prob = torch.sigmoid(base_train)

    base_fn = (y_train == 1) & (base_prob < 0.5)
    base_fp = (y_train == 0) & (base_prob >= 0.5)
    base_correct = ~base_fn & ~base_fp

    corr_loss = torch.tensor(0.0, device=final_logit.device)

    # Build direction agreement factor per train node
    dir_stats: dict[str, float] = {
        "direction_correction_agreement": 0.0,
    }

    # Helper: compute agreement factor for correction nodes
    def _agreement_factor(fn_or_fp_mask: Tensor, expected_dir: int) -> tuple[Tensor, float]:
        """Return per-node correction weights and agreement fraction.

        expected_dir: 0=increase_risk for FN, 1=decrease_risk for FP.
        Nodes with matching direction get weight 1.0 (strong correction).
        Nodes with disagreeing/uncertain direction get weight 0.1 (anchor).
        """
        n_nodes = int(fn_or_fp_mask.sum().item())
        if n_nodes == 0 or direction_ids is None:
            return None, 0.0

        dir_train = direction_ids[train_mask]
        node_dirs = dir_train[fn_or_fp_mask]
        agrees = (node_dirs == expected_dir)
        agrees_count = int(agrees.sum().item())

        weights = torch.full((n_nodes,), 0.1, device=fn_or_fp_mask.device)
        weights[agrees] = 1.0

        agreement_frac = agrees_count / n_nodes if n_nodes > 0 else 0.0
        return weights, agreement_frac

    # Correction: base FN → encourage positive shift (direction=increase_risk agrees)
    fn_agreement = 0.0
    if base_fn.any():
        fn_shift = shift_train[base_fn]
        fn_weights, fn_agreement = _agreement_factor(base_fn, expected_dir=0)

        if fn_weights is not None and fn_weights.numel() == fn_shift.numel():
            fn_loss = (fn_weights * F.relu(correction_margin - fn_shift)).mean()
        else:
            fn_loss = F.relu(correction_margin - fn_shift).mean()
        corr_loss = corr_loss + fn_loss

    # Correction: base FP → encourage negative shift (direction=decrease_risk agrees)
    fp_agreement = 0.0
    if base_fp.any():
        fp_shift = shift_train[base_fp]
        fp_weights, fp_agreement = _agreement_factor(base_fp, expected_dir=1)

        if fp_weights is not None and fp_weights.numel() == fp_shift.numel():
            fp_loss = (fp_weights * F.relu(correction_margin + fp_shift)).mean()
        else:
            fp_loss = F.relu(correction_margin + fp_shift).mean()
        corr_loss = corr_loss + fp_loss

    correction_count = int(base_fn.sum().item()) + int(base_fp.sum().item())
    total_agreement = 0.0
    if correction_count > 0:
        total_fn = int(base_fn.sum().item())
        total_fp = int(base_fp.sum().item())
        total_agreement = (fn_agreement * total_fn + fp_agreement * total_fp) / correction_count
    dir_stats["direction_correction_agreement"] = total_agreement

    # Inheritance: base correct nodes → penalize large |shift|
    if base_correct.any() and anchor_weight > 0:
        anchor_loss = (shift_train[base_correct] ** 2).mean()
        corr_loss = corr_loss + anchor_weight * anchor_loss

    # Weak/uncertain evidence → penalize large |shift|
    if risk_type_names is not None and "risk_type_id" in targets:
        weak_id = None
        for ri, rn in enumerate(risk_type_names):
            if rn == "weak_or_uncertain_evidence":
                weak_id = ri
                break
        if weak_id is not None:
            weak_mask = (targets["risk_type_id"][train_mask] == weak_id) & accepted_mask[train_mask].bool()
            if weak_mask.any():
                weak_loss = (shift_train[weak_mask] ** 2).mean()
                corr_loss = corr_loss + anchor_weight * weak_loss

    # Safety: penalize |shift| > max_allowed_shift
    if max_shift_penalty_weight > 0:
        overshoot = (shift_train.abs() - max_allowed_shift)
        penalty = (F.relu(overshoot) ** 2).mean()
        corr_loss = corr_loss + max_shift_penalty_weight * penalty

    fn_count = int(base_fn.sum().item())
    fp_count = int(base_fp.sum().item())

    return corr_loss, fn_count, fp_count, dir_stats


def compute_cvscd_loss(
    outputs: dict[str, Tensor],
    y: Tensor,
    targets: dict[str, Tensor],
    train_mask: Tensor,
    base_logits: Tensor,
    risk_type_names: list[str] | None = None,
    pos_weight: float | None = None,
    lambda_det: float = 1.0,
    lambda_err: float = 0.3,
    lambda_signed: float = 0.1,
    lambda_corr: float = 0.1,
    correction_weight: float = 2.0,
    signed_margin: float = 0.2,
    correction_margin: float = 0.05,
    anchor_weight: float = 0.001,
    max_shift_penalty_weight: float = 0.001,
    max_allowed_shift: float = 0.3,
    direction_ids: Tensor | None = None,
    lambda_direction: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """CV-SCD four-term Stage3 loss with direction awareness.

    L_CoVER = L_det + lambda_err * L_err + lambda_signed * L_signed + lambda_corr * L_corr

    Direction effects are folded into existing terms, NOT a 5th term.

    Args:
        outputs: Reasoner outputs (final_logit, type_logits, pos_logits, neg_logits,
                 direction_logits).
        y: Labels for the full graph.
        targets: Dict with risk_type_id, pos_mask, neg_mask, accepted_mask.
        train_mask: Boolean mask for training nodes.
        base_logits: Base detector logits (will be detached internally).
        risk_type_names: List of risk type names for weak_or_uncertain detection.
        pos_weight: Positive class weight for BCE.
        lambda_det: Weight for detection loss.
        lambda_err: Weight for ERR distillation loss.
        lambda_signed: Weight for signed evidence separation loss.
        lambda_corr: Weight for corrective residual loss.
        correction_weight: Weight multiplier for base FN/FP nodes in L_det.
        signed_margin: Margin for signed evidence separation.
        correction_margin: Margin for corrective residual.
        anchor_weight: Weight for inheritance anchor penalty.
        max_shift_penalty_weight: Weight for safety shift penalty.
        max_allowed_shift: Maximum allowed absolute shift.
        direction_ids: Per-node direction IDs (0=increase_risk, 1=decrease_risk,
                       2=uncertain). None or all-uncertain → no direction effects.
        lambda_direction: Weight for direction CE loss inside L_err.

    Returns:
        (total_loss, stats_dict) with all loss components and diagnostics.
    """
    # Detach base_logits to prevent gradient flow
    base_logits = base_logits.detach()

    final_logit = outputs["final_logit"]
    accepted_mask = targets["accepted_mask"]

    # L_det: Correction-aware detection loss
    det_loss, det_counts = _compute_det_loss(
        final_logit, y, base_logits, train_mask, pos_weight, correction_weight,
    )

    # L_err: ERR distillation (accepted ERR nodes only) + direction CE
    err_loss, direction_ce_val = _compute_err_loss(
        outputs, targets, accepted_mask, direction_ids, lambda_direction,
    )

    # L_signed: Signed evidence separation with direction awareness
    signed_loss = _compute_signed_loss(
        outputs, targets, accepted_mask, signed_margin, direction_ids,
    )

    # L_corr: Selective corrective residual with direction agreement
    corr_loss, fn_corr, fp_corr, dir_stats = _compute_corr_loss(
        final_logit, base_logits, y, targets, accepted_mask, train_mask,
        risk_type_names, correction_margin, anchor_weight,
        max_shift_penalty_weight, max_allowed_shift, direction_ids,
    )

    total_loss = (
        lambda_det * det_loss
        + lambda_err * err_loss
        + lambda_signed * signed_loss
        + lambda_corr * corr_loss
    )

    # Compute residual shift diagnostics (over train nodes)
    with torch.no_grad():
        shift = final_logit[train_mask] - base_logits[train_mask]
        shift_mean = float(shift.mean().item())
        shift_max_abs = float(shift.abs().max().item()) if shift.numel() > 0 else 0.0

        # Direction-specific diagnostics
        direction_names = ["increase_risk", "decrease_risk", "uncertain"]
        direction_label_dist: dict[str, float] = {}
        residual_shift_by_direction: dict[str, float] = {}
        correction_nodes_by_direction: dict[str, float] = {}

        if direction_ids is not None:
            dir_train = direction_ids[train_mask]
            for di, dname in enumerate(direction_names):
                d_mask = dir_train == di
                direction_label_dist[dname] = float(d_mask.sum().item())
                if d_mask.any():
                    residual_shift_by_direction[dname] = float(shift[d_mask].mean().item())
                else:
                    residual_shift_by_direction[dname] = 0.0
                # Correction nodes by direction
                base_prob_t = torch.sigmoid(base_logits[train_mask])
                base_fn_t = (y[train_mask] == 1) & (base_prob_t < 0.5)
                base_fp_t = (y[train_mask] == 0) & (base_prob_t >= 0.5)
                corr_t = base_fn_t | base_fp_t
                corr_dir = corr_t & d_mask
                correction_nodes_by_direction[dname] = float(corr_dir.sum().item())

    accepted_err_count = int(accepted_mask.sum().item())

    stats: dict[str, float] = {
        "det_loss": det_loss.item(),
        "err_loss": err_loss.item(),
        "signed_loss": signed_loss.item(),
        "corr_loss": corr_loss.item(),
        "total_loss": total_loss.item(),
        "correction_node_count": float(det_counts["fn_count"] + det_counts["fp_count"]),
        "inheritance_node_count": float(fn_corr + fp_corr),
        "accepted_err_count": float(accepted_err_count),
        "residual_shift_mean": shift_mean,
        "residual_shift_max_abs": shift_max_abs,
        "false_negative_correction_count": float(det_counts["fn_count"]),
        "false_positive_correction_count": float(det_counts["fp_count"]),
    }

    # Merge direction diagnostics
    if direction_ids is not None:
        stats["direction_ce_loss"] = direction_ce_val
        stats.update({f"direction_label_{k}": v for k, v in direction_label_dist.items()})
        stats.update({f"residual_shift_{k}": v for k, v in residual_shift_by_direction.items()})
        stats.update({f"correction_nodes_{k}": v for k, v in correction_nodes_by_direction.items()})
        stats.update(dir_stats)

    return total_loss, stats
