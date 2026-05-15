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
    strength_ids: Tensor | None = None,
    polarity_ids: Tensor | None = None,
    base_correct_anchor_weight: float | None = None,
    uncertain_anchor_weight: float | None = None,
    decrease_anchor_weight: float | None = None,
    fp_anchor_weight: float | None = None,
    non_fn_positive_anchor_weight: float | None = None,
    non_accepted_anchor_weight: float | None = None,
    enable_fp_negative_correction: bool = False,
    residual_cap: float | None = None,
) -> tuple[Tensor, int, int, dict[str, float]]:
    """L_corr: FN-focused evidence-corrective residual loss.

    Strong positive correction is only applied to accepted train base FN nodes
    with increase_risk, moderate/strong evidence, and non-benign payload polarity.
    FP nodes are anchored in the current structure-only setting unless explicitly
    enabled for future accepted FP + decrease_risk evidence.

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
    base_correct_anchor_weight = anchor_weight if base_correct_anchor_weight is None else base_correct_anchor_weight
    uncertain_anchor_weight = 10.0 * anchor_weight if uncertain_anchor_weight is None else uncertain_anchor_weight
    decrease_anchor_weight = 10.0 * anchor_weight if decrease_anchor_weight is None else decrease_anchor_weight
    fp_anchor_weight = 10.0 * anchor_weight if fp_anchor_weight is None else fp_anchor_weight
    non_fn_positive_anchor_weight = 10.0 * anchor_weight if non_fn_positive_anchor_weight is None else non_fn_positive_anchor_weight
    non_accepted_anchor_weight = anchor_weight if non_accepted_anchor_weight is None else non_accepted_anchor_weight

    accepted_train_mask = accepted_mask[train_mask].bool()
    dir_train = direction_ids[train_mask] if direction_ids is not None else None
    strength_train = strength_ids[train_mask] if strength_ids is not None else None
    polarity_train = polarity_ids[train_mask] if polarity_ids is not None else None
    strong_enough = strength_train >= 1 if strength_train is not None else torch.ones_like(base_fn)
    not_benign = polarity_train != 1 if polarity_train is not None else torch.ones_like(base_fn)
    pos_agree = (
        accepted_train_mask & base_fn & (dir_train == 0) & strong_enough & not_benign
        if dir_train is not None else torch.zeros_like(base_fn)
    )
    neg_agree = (
        accepted_train_mask & base_fp & (dir_train == 1) & bool(enable_fp_negative_correction)
        if dir_train is not None else torch.zeros_like(base_fp)
    )
    correction_count = int((accepted_train_mask & (base_fn | base_fp)).sum().item())
    total_agreement = 0.0
    if correction_count > 0:
        total_agreement = int((pos_agree | neg_agree).sum().item()) / correction_count

    pos_corr_loss = torch.tensor(0.0, device=final_logit.device)
    neg_corr_loss = torch.tensor(0.0, device=final_logit.device)
    if pos_agree.any():
        pos_corr_loss = F.relu(correction_margin - shift_train[pos_agree]).mean()
    if neg_agree.any():
        neg_corr_loss = F.relu(correction_margin + shift_train[neg_agree]).mean()

    corr_loss = torch.tensor(0.0, device=final_logit.device)
    if pos_agree.any() and neg_agree.any():
        corr_loss = corr_loss + 0.5 * (pos_corr_loss + neg_corr_loss)
    elif pos_agree.any():
        corr_loss = corr_loss + pos_corr_loss
    elif neg_agree.any():
        corr_loss = corr_loss + neg_corr_loss

    # Inheritance: base correct nodes → penalize large |shift|
    if base_correct.any() and base_correct_anchor_weight > 0:
        anchor_loss = (shift_train[base_correct] ** 2).mean()
        corr_loss = corr_loss + base_correct_anchor_weight * anchor_loss

    if base_fp.any() and fp_anchor_weight > 0:
        fp_anchor = (shift_train[base_fp] ** 2).mean()
        corr_loss = corr_loss + fp_anchor_weight * fp_anchor

    # Directional anchors: non-agreement decrease/uncertain evidence must not
    # follow a global positive residual drift.
    uncertain_mask = torch.zeros_like(base_fn)
    decrease_mask = torch.zeros_like(base_fn)
    if dir_train is not None and accepted_train_mask.any():
        decrease_mask = accepted_train_mask & (dir_train == 1) & ~neg_agree
        if decrease_mask.any() and decrease_anchor_weight > 0:
            decrease_anchor = (F.relu(shift_train[decrease_mask]) ** 2).mean()
            corr_loss = corr_loss + decrease_anchor_weight * decrease_anchor

        uncertain_mask = accepted_train_mask & (dir_train == 2)
        if uncertain_mask.any() and uncertain_anchor_weight > 0:
            uncertain_anchor = (shift_train[uncertain_mask] ** 2).mean()
            corr_loss = corr_loss + uncertain_anchor_weight * uncertain_anchor

        disagree_mask = accepted_train_mask & (base_fn | base_fp) & ~(pos_agree | neg_agree)
        if disagree_mask.any():
            disagree_anchor = (shift_train[disagree_mask] ** 2).mean()
            corr_loss = corr_loss + anchor_weight * disagree_anchor

    non_accepted_mask = ~accepted_train_mask
    if non_accepted_mask.any() and non_accepted_anchor_weight > 0:
        non_accepted_anchor = (F.relu(shift_train[non_accepted_mask]) ** 2).mean()
        corr_loss = corr_loss + non_accepted_anchor_weight * non_accepted_anchor

    non_fn_positive_mask = ~pos_agree
    non_fn_positive_shift_penalty = torch.tensor(0.0, device=final_logit.device)
    if non_fn_positive_mask.any() and non_fn_positive_anchor_weight > 0:
        non_fn_positive_shift_penalty = (F.relu(shift_train[non_fn_positive_mask]) ** 2).mean()
        corr_loss = corr_loss + non_fn_positive_anchor_weight * non_fn_positive_shift_penalty

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

    fn_count = int(pos_agree.sum().item())
    fp_count = int(neg_agree.sum().item())

    cap = residual_cap if residual_cap is not None else max_allowed_shift
    near_pos_cap = float((shift_train >= 0.9 * cap).float().mean().item()) if cap > 0 and shift_train.numel() else 0.0
    near_neg_cap = float((shift_train <= -0.9 * cap).float().mean().item()) if cap > 0 and shift_train.numel() else 0.0
    dir_stats: dict[str, float] = {
        "direction_correction_agreement": total_agreement,
        "positive_correction_node_count": float(pos_agree.sum().item()),
        "negative_correction_node_count": float(neg_agree.sum().item()),
        "fn_positive_correction_node_count": float(pos_agree.sum().item()),
        "fp_anchor_node_count": float(base_fp.sum().item()),
        "uncertain_anchor_node_count": float(uncertain_mask.sum().item()),
        "decrease_anchor_node_count": float(decrease_mask.sum().item()),
        "non_accepted_anchor_node_count": float(non_accepted_mask.sum().item()),
        "non_fn_positive_shift_penalty": float(non_fn_positive_shift_penalty.item()),
        "direction_balance_warning": 1.0 if bool(pos_agree.any()) and not bool(neg_agree.any()) else 0.0,
        "residual_cap": float(cap),
        "near_positive_cap_fraction": near_pos_cap,
        "near_negative_cap_fraction": near_neg_cap,
        "residual_saturation_warning": 1.0 if near_pos_cap > 0.5 else 0.0,
        "residual_shift_status_FN": float(shift_train[base_fn].mean().item()) if base_fn.any() else 0.0,
        "residual_shift_status_FP": float(shift_train[base_fp].mean().item()) if base_fp.any() else 0.0,
        "residual_shift_status_TP": float(shift_train[(y_train == 1) & ~base_fn].mean().item()) if ((y_train == 1) & ~base_fn).any() else 0.0,
        "residual_shift_status_TN": float(shift_train[(y_train == 0) & ~base_fp].mean().item()) if ((y_train == 0) & ~base_fp).any() else 0.0,
        "residual_shift_group_fn_positive": float(shift_train[pos_agree].mean().item()) if pos_agree.any() else 0.0,
        "residual_shift_group_fp_anchor": float(shift_train[base_fp].mean().item()) if base_fp.any() else 0.0,
        "residual_shift_group_other": float(shift_train[~pos_agree].mean().item()) if (~pos_agree).any() else 0.0,
    }

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
    strength_ids: Tensor | None = None,
    polarity_ids: Tensor | None = None,
    lambda_direction: float = 1.0,
    base_correct_anchor_weight: float | None = None,
    uncertain_anchor_weight: float | None = None,
    decrease_anchor_weight: float | None = None,
    fp_anchor_weight: float | None = None,
    non_fn_positive_anchor_weight: float | None = None,
    non_accepted_anchor_weight: float | None = None,
    enable_fp_negative_correction: bool = False,
    residual_cap: float | None = None,
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
        strength_ids, polarity_ids, base_correct_anchor_weight,
        uncertain_anchor_weight, decrease_anchor_weight, fp_anchor_weight,
        non_fn_positive_anchor_weight, non_accepted_anchor_weight,
        enable_fp_negative_correction, residual_cap,
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


def _zero_loss(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _get_output(outputs: dict[str, Tensor], key: str, fallback: str) -> Tensor:
    value = outputs.get(key)
    if value is not None:
        return value
    return outputs[fallback]


def _to_pos_weight(value: float | Tensor | None, device: torch.device) -> Tensor | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        return value.to(device)
    return torch.tensor(float(value), device=device)


def compute_pairrank_loss(
    final_logit: Tensor,
    y: Tensor,
    train_mask: Tensor,
    base_logits: Tensor,
    temperature: float = 0.1,
    max_hard_negatives: int = 512,
) -> tuple[Tensor, dict[str, float]]:
    """Pairwise train-positive vs hard-train-negative ranking loss."""
    train = train_mask.bool()
    labels = y.long()
    pos_mask = train & (labels == 1)
    neg_mask = train & (labels == 0)

    if not bool(pos_mask.any()) or not bool(neg_mask.any()):
        return _zero_loss(final_logit), {
            "pairrank_pos_count": float(pos_mask.sum().item()),
            "pairrank_hard_neg_count": 0.0,
            "ranking_gap_pos_vs_hard_neg": 0.0,
        }

    neg_idx = neg_mask.nonzero(as_tuple=True)[0]
    hard_score = torch.maximum(
        torch.sigmoid(base_logits.detach()),
        torch.sigmoid(final_logit.detach()),
    )
    hard_values = hard_score[neg_idx]
    k = min(
        int(max_hard_negatives),
        int(neg_idx.numel()),
        max(int(pos_mask.sum().item()) * 4, 1),
    )
    selected_neg_idx = neg_idx[torch.topk(hard_values, k=k).indices]

    pos_scores = final_logit[pos_mask]
    neg_scores = final_logit[selected_neg_idx]
    tau = max(float(temperature), 1e-6)
    loss = F.softplus(-(pos_scores[:, None] - neg_scores[None, :]) / tau).mean()
    gap = pos_scores.mean() - neg_scores.mean()

    return loss, {
        "pairrank_pos_count": float(pos_scores.numel()),
        "pairrank_hard_neg_count": float(neg_scores.numel()),
        "ranking_gap_pos_vs_hard_neg": float(gap.detach().item()),
    }


def compute_cve_loss(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    accepted_mask: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Contract-verified evidence loss over accepted ERR nodes only."""
    final_logit = outputs["final_logit"]
    accepted = accepted_mask.bool()
    zero = _zero_loss(final_logit)
    stats = {
        "risk_type_ce": 0.0,
        "direction_ce": 0.0,
        "strength_ce": 0.0,
        "support_mask_bce": 0.0,
        "counter_mask_bce": 0.0,
        "accepted_cve_count": float(accepted.sum().item()),
    }
    if not bool(accepted.any()):
        return zero, stats

    risk_logits = _get_output(outputs, "risk_type_logits", "type_logits")
    support_logits = _get_output(outputs, "support_mask_logits", "pos_logits")
    counter_logits = _get_output(outputs, "counter_mask_logits", "neg_logits")
    direction_logits = outputs.get("direction_logits")
    strength_logits = outputs.get("strength_logits")

    risk_loss = F.cross_entropy(risk_logits[accepted], targets["risk_type_id"][accepted])
    support_loss = F.binary_cross_entropy_with_logits(
        support_logits[accepted],
        targets["pos_mask"][accepted].float(),
    )
    counter_loss = F.binary_cross_entropy_with_logits(
        counter_logits[accepted],
        targets["neg_mask"][accepted].float(),
    )

    direction_loss = zero
    if direction_logits is not None and "direction_id" in targets:
        direction_ids = targets["direction_id"]
        valid = accepted & (direction_ids >= 0)
        if bool(valid.any()):
            direction_loss = F.cross_entropy(direction_logits[valid], direction_ids[valid])

    strength_loss = zero
    if strength_logits is not None and "strength_id" in targets:
        strength_ids = targets["strength_id"]
        valid = accepted & (strength_ids >= 0)
        if bool(valid.any()):
            strength_loss = F.cross_entropy(strength_logits[valid], strength_ids[valid])

    total = risk_loss + direction_loss + strength_loss + support_loss + counter_loss
    stats.update({
        "risk_type_ce": float(risk_loss.detach().item()),
        "direction_ce": float(direction_loss.detach().item()),
        "strength_ce": float(strength_loss.detach().item()),
        "support_mask_bce": float(support_loss.detach().item()),
        "counter_mask_bce": float(counter_loss.detach().item()),
    })
    return total, stats


def _project_teacher_latents(teacher: Tensor, latent_dim: int) -> tuple[Tensor, str]:
    teacher_dim = teacher.shape[-1]
    if teacher_dim == latent_dim:
        return teacher, "identity"
    if teacher_dim > latent_dim:
        return teacher[..., :latent_dim], f"truncate:{teacher_dim}->{latent_dim}"
    pad = latent_dim - teacher_dim
    return F.pad(teacher, (0, pad)), f"zero_pad:{teacher_dim}->{latent_dim}"


def _supervised_contrastive_loss(
    z_student: Tensor,
    contrast_ids: Tensor,
    temperature: float,
) -> Tensor:
    if z_student.shape[0] <= 1:
        return _zero_loss(z_student)
    labels = contrast_ids.view(-1)
    valid = labels >= 0
    if int(valid.sum().item()) <= 1:
        return _zero_loss(z_student)

    z = F.normalize(z_student[valid], dim=-1)
    labels = labels[valid]
    logits = z @ z.t() / max(float(temperature), 1e-6)
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive = (labels[:, None] == labels[None, :]) & ~eye
    if not bool(positive.any()):
        return _zero_loss(z_student)

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~eye).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    per_anchor = -(log_prob * positive.float()).sum(dim=1) / positive.float().sum(dim=1).clamp_min(1.0)
    anchors = positive.any(dim=1)
    return per_anchor[anchors].mean()


def compute_latent_loss(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    beta_contrastive: float = 0.0,
    contrast_temperature: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Cosine latent distillation over accepted nodes with teacher latents."""
    z_student = outputs["z_student"]
    zero = _zero_loss(z_student)
    mask = targets.get("teacher_latent_mask")
    teacher = targets.get("teacher_latents")
    stats = {
        "latent_cosine_similarity": 0.0,
        "latent_teacher_count": 0.0,
        "latent_projection_needed": 0.0,
        "latent_contrastive_loss": 0.0,
    }
    if mask is None or teacher is None:
        return zero, stats

    valid = mask.bool()
    if not bool(valid.any()):
        return zero, stats

    teacher = teacher.to(device=z_student.device, dtype=z_student.dtype)
    teacher_projected, projection = _project_teacher_latents(teacher, z_student.shape[-1])
    cosine = F.cosine_similarity(z_student[valid], teacher_projected[valid], dim=-1)
    cosine_loss = (1.0 - cosine).mean()
    contrastive_loss = zero

    if beta_contrastive > 0 and "contrast_class_id" in targets:
        contrastive_loss = _supervised_contrastive_loss(
            z_student[valid],
            targets["contrast_class_id"][valid],
            contrast_temperature,
        )

    total = cosine_loss + float(beta_contrastive) * contrastive_loss
    stats.update({
        "latent_cosine_similarity": float(cosine.detach().mean().item()),
        "latent_teacher_count": float(valid.sum().item()),
        "latent_projection_needed": 0.0 if projection == "identity" else 1.0,
        "latent_contrastive_loss": float(contrastive_loss.detach().item()),
    })
    return total, stats


def compute_intervention_loss(
    final_logit: Tensor,
    base_logits: Tensor,
    y: Tensor,
    targets: dict[str, Tensor],
    train_mask: Tensor,
    delta_scale: float = 2.0,
    max_allowed_shift: float = 0.2,
    anchor_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Bounded residual intervention with evidence-gated correction."""
    base_detached = base_logits.detach()
    residual = final_logit - base_detached
    train = train_mask.bool()
    accepted = targets.get("accepted_mask", torch.zeros_like(y, dtype=torch.bool)).bool()
    direction_ids = targets.get("direction_id", torch.full_like(y.long(), -1))
    strength_ids = targets.get("strength_id", torch.full_like(y.long(), -1))
    polarity_ids = targets.get("polarity_id", torch.full_like(y.long(), -1))

    base_prob = torch.sigmoid(base_detached)
    labels = y.long()
    base_fn = train & (labels == 1) & (base_prob < 0.5)
    base_fp = train & (labels == 0) & (base_prob >= 0.5)
    strong_enough = strength_ids >= 1

    positive = accepted & base_fn & (direction_ids == 0) & strong_enough
    negative = accepted & base_fp & (direction_ids == 1) & strong_enough & (polarity_ids == 1)
    intervene = train & (positive | negative)
    anchor = train & ~intervene

    target = (float(delta_scale) * (y.float() - base_prob)).clamp(
        -float(max_allowed_shift),
        float(max_allowed_shift),
    )

    intervention_loss = _zero_loss(final_logit)
    if bool(intervene.any()):
        intervention_loss = F.smooth_l1_loss(residual[intervene], target[intervene])

    anchor_loss = _zero_loss(final_logit)
    if bool(anchor.any()):
        anchor_loss = (residual[anchor] ** 2).mean()

    cap_penalty = _zero_loss(final_logit)
    if bool(train.any()):
        overshoot = F.relu(residual[train].abs() - float(max_allowed_shift))
        cap_penalty = (overshoot ** 2).mean()

    total = intervention_loss + float(anchor_weight) * anchor_loss + cap_penalty
    return total, {
        "intervention_huber": float(intervention_loss.detach().item()),
        "intervention_anchor": float(anchor_loss.detach().item()),
        "bounded_shift_penalty": float(cap_penalty.detach().item()),
        "positive_intervention_count": float(positive.sum().item()),
        "negative_intervention_count": float(negative.sum().item()),
        "anchor_count": float(anchor.sum().item()),
    }


def compute_cover_lift_loss(
    outputs: dict[str, Tensor],
    y: Tensor,
    targets: dict[str, Tensor],
    train_mask: Tensor,
    base_logits: Tensor,
    pos_weight: float | Tensor | None = None,
    lambda_cve: float = 0.3,
    lambda_latent: float = 0.2,
    lambda_intervene: float = 0.05,
    lambda_pairrank: float = 0.2,
    pairrank_temperature: float = 0.1,
    delta_scale: float = 2.0,
    max_allowed_shift: float = 0.2,
    latent_contrastive_weight: float = 0.0,
    contrast_temperature: float = 0.1,
    lambda_gate_sparse: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """CoVER-LIFT Stage3 loss.

    The function uses train labels only for detection, pair ranking, and
    residual intervention. Accepted/rejected gating is carried only by
    ``targets["accepted_mask"]`` and ``targets["teacher_latent_mask"]``.
    """
    final_logit = outputs["final_logit"]
    device = final_logit.device
    base_detached = base_logits.detach()
    train = train_mask.bool()
    accepted_mask = targets.get(
        "accepted_mask",
        torch.zeros_like(final_logit, dtype=torch.bool),
    )

    if bool(train.any()):
        bce_loss = F.binary_cross_entropy_with_logits(
            final_logit[train],
            y[train].float(),
            pos_weight=_to_pos_weight(pos_weight, device),
        )
    else:
        bce_loss = _zero_loss(final_logit)

    pairrank_loss, pairrank_stats = compute_pairrank_loss(
        final_logit,
        y,
        train,
        base_detached,
        temperature=pairrank_temperature,
    )
    ap_det_loss = bce_loss + float(lambda_pairrank) * pairrank_loss

    cve_loss, cve_stats = compute_cve_loss(outputs, targets, accepted_mask)
    latent_loss, latent_stats = compute_latent_loss(
        outputs,
        targets,
        beta_contrastive=latent_contrastive_weight,
        contrast_temperature=contrast_temperature,
    )
    intervene_loss, intervene_stats = compute_intervention_loss(
        final_logit,
        base_detached,
        y,
        targets,
        train,
        delta_scale=delta_scale,
        max_allowed_shift=max_allowed_shift,
    )
    gate_sparse_loss = outputs.get("relation_gate_sparse_loss", _zero_loss(final_logit))

    total_loss = (
        ap_det_loss
        + float(lambda_cve) * cve_loss
        + float(lambda_latent) * latent_loss
        + float(lambda_intervene) * intervene_loss
        + float(lambda_gate_sparse) * gate_sparse_loss
    )

    with torch.no_grad():
        residual = final_logit[train] - base_detached[train]
        if residual.numel() > 0:
            shift_mean = float(residual.mean().item())
            shift_abs_mean = float(residual.abs().mean().item())
            shift_max_abs = float(residual.abs().max().item())
            near_cap = float((residual.abs() >= 0.9 * float(max_allowed_shift)).float().mean().item())
        else:
            shift_mean = shift_abs_mean = shift_max_abs = near_cap = 0.0

    stats: dict[str, float] = {
        "total_loss": float(total_loss.detach().item()),
        "ap_det_loss": float(ap_det_loss.detach().item()),
        "bce_loss": float(bce_loss.detach().item()),
        "pairrank_loss": float(pairrank_loss.detach().item()),
        "cve_loss": float(cve_loss.detach().item()),
        "latent_loss": float(latent_loss.detach().item()),
        "intervene_loss": float(intervene_loss.detach().item()),
        "gate_sparse_loss": float(gate_sparse_loss.detach().item()),
        "lambda_gate_sparse": float(lambda_gate_sparse),
        "residual_shift_mean": shift_mean,
        "residual_shift_abs_mean": shift_abs_mean,
        "residual_shift_max_abs": shift_max_abs,
        "near_cap_fraction": near_cap,
    }
    stats.update(cve_stats)
    stats.update(latent_stats)
    stats.update(pairrank_stats)
    stats.update(intervene_stats)
    return total_loss, stats


def compute_cover_judge_loss(
    outputs: dict[str, Tensor],
    y: Tensor,
    targets: dict[str, Tensor],
    train_mask: Tensor,
    base_logits: Tensor,
    pos_weight: float | Tensor | None = None,
    lambda_rank: float = 0.2,
    pairrank_temperature: float = 0.1,
    lambda_judge: float = 0.1,
    lambda_llm_reg: float = 0.001,
    llm_alpha_reg: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    final_logit = outputs["final_logit"]
    device = final_logit.device
    train = train_mask.bool()
    if bool(train.any()):
        det_loss = F.binary_cross_entropy_with_logits(
            final_logit[train],
            y[train].float(),
            pos_weight=_to_pos_weight(pos_weight, device),
        )
    else:
        det_loss = _zero_loss(final_logit)

    rank_loss, rank_stats = compute_pairrank_loss(
        final_logit,
        y,
        train,
        base_logits.detach(),
        temperature=pairrank_temperature,
    )
    judge_mask = targets.get("judge_mask", torch.zeros_like(y, dtype=torch.bool)).bool() & train
    judge_only = outputs.get("judge_only_logit")
    judge_loss = _zero_loss(final_logit)
    if judge_only is not None and bool(judge_mask.any()):
        judge_loss = F.binary_cross_entropy_with_logits(
            judge_only[judge_mask],
            y[judge_mask].float(),
            pos_weight=_to_pos_weight(pos_weight, device),
        )

    llm_residual = outputs.get("llm_residual", _zero_loss(final_logit))
    if isinstance(llm_residual, Tensor) and llm_residual.ndim > 0:
        llm_reg = (llm_residual[train] ** 2).mean() if bool(train.any()) else _zero_loss(final_logit)
    else:
        llm_reg = _zero_loss(final_logit)

    alpha = outputs.get("alpha_llm")
    alpha_reg = _zero_loss(final_logit)
    if isinstance(alpha, Tensor) and alpha.ndim > 0 and bool(train.any()):
        alpha_reg = (alpha[train] ** 2).mean()

    total = (
        det_loss
        + float(lambda_rank) * rank_loss
        + float(lambda_judge) * judge_loss
        + float(lambda_llm_reg) * llm_reg
        + float(llm_alpha_reg) * alpha_reg
    )
    with torch.no_grad():
        alpha_mean = float(alpha[train].mean().item()) if isinstance(alpha, Tensor) and alpha.ndim > 0 and bool(train.any()) else 0.0
        delta = outputs.get("delta_llm")
        delta_abs_max = float(delta[train].abs().max().item()) if isinstance(delta, Tensor) and delta.ndim > 0 and bool(train.any()) else 0.0
        rel_delta = final_logit - outputs.get("rel_only_logit", final_logit).detach()
        final_vs_rel_abs_mean = float(rel_delta[train].abs().mean().item()) if bool(train.any()) else 0.0
    stats = {
        "total_loss": float(total.detach().item()),
        "det_loss": float(det_loss.detach().item()),
        "pairrank_loss": float(rank_loss.detach().item()),
        "judge_loss": float(judge_loss.detach().item()),
        "llm_reg_loss": float(llm_reg.detach().item()),
        "llm_alpha_reg_loss": float(alpha_reg.detach().item()),
        "lambda_judge": float(lambda_judge),
        "lambda_llm_reg": float(lambda_llm_reg),
        "judge_train_count": float(judge_mask.sum().item()),
        "alpha_llm_mean": alpha_mean,
        "delta_llm_max_abs": delta_abs_max,
        "final_vs_rel_abs_mean": final_vs_rel_abs_mean,
    }
    stats.update(rank_stats)
    return total, stats
