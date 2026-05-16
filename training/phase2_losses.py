"""Phase2 4-term loss for CoVER-REL.

Total loss::

    L = L_cls + λ_int · L_intervention + λ_sparse · L_sparse + λ_align · L_align

Term details
------------

L_cls
    Standard pos-weighted BCE on ``final_logit[train]`` against the labels.

L_intervention  (base-anchored correction penalty)
    ``mean_train((z - b)²)``.  Since the forward is
    ``z = b + Δ_rel + α · Δ_llm``, this is implemented as
    ``mean_train((Δ_rel + α · Δ_llm)²)``.

    The deprecated ``lambda_trust`` name is accepted as an alias for one
    release.  ``eta_llm`` is kept as a no-op kwarg for backward compatibility.

L_sparse  (evidence-to-gate consistency)
    Build a fixed, non-trainable relation evidence distribution
    ``π_evidence`` from the relation feature packet, then minimise
    ``KL(π_evidence || g)``.  This keeps the "sparse" loss slot but changes
    the optimisation direction from entropy minimisation to evidence
    consistency: the gate is concentrated only when the evidence itself is
    concentrated.

L_align  (judge-tilted evidence alignment)
    Accepted judge records do not overwrite relation evidence with a one-hot
    routing target.  Instead, they tilt ``π_evidence`` toward the
    judge-identified key relation and align the gate to the tilted target via
    KL.  The judge verifies/tilts evidence; it is not a router teacher.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor


# Alignment weight for each evidence_strength bucket.  Uncertain/missing
# strengths are skipped by ``build_judge_relation_targets``.
_STRENGTH_TO_ALIGN_WEIGHT: dict[str, float] = {
    "strong": 1.0,
    "moderate": 0.5,
    "weak": 0.2,
}


def build_judge_relation_targets(
    jsonl_path: Path,
    num_nodes: int,
    relation_names: Iterable[str],
) -> dict[str, Tensor]:
    """Build per-node key-relation CE targets from an ``accepted_judge.jsonl`` file.

    Skip rules (the node simply has ``has_target=False`` afterwards):

    * ``verdict == "uncertain"``
    * ``evidence_strength`` missing or not in {strong, moderate, weak}
    * ``key_relation`` not in ``relation_names``
    * ``node_id`` outside ``[0, num_nodes)``

    For accepted records we store the cited key relation index and a scalar
    weight from the evidence-strength bucket::

        strong   → 1.0
        moderate → 0.5
        weak     → 0.2

    Args:
        jsonl_path: Path to ``accepted_judge.jsonl``.  A missing file returns
            an all-zero / all-False target (acts like "no judge alignment").
        num_nodes: Total number of graph nodes (defines target shape).
        relation_names: Ordered relation list (case-insensitive); the index
            of ``key_relation`` in this list determines the CE target index.

    Returns:
        Dict with::

            key_idx         (N,)   int64, -1 where no target
            w_weight        (N,)   float32
            has_target_mask (N,)   bool

        A one-hot ``q_target`` field is also returned as a deprecated
        compatibility alias for legacy scripts that have not yet moved from
        KL to CE.  New loss code ignores it.
    """
    rel_names = [str(name).upper() for name in relation_names]
    if not rel_names:
        raise ValueError("relation_names must be non-empty")
    R = len(rel_names)
    rel_index = {name: i for i, name in enumerate(rel_names)}

    n = int(num_nodes)
    key_idx = torch.full((n,), -1, dtype=torch.long)
    q_target = torch.zeros((n, R), dtype=torch.float32)
    w_weight = torch.zeros(n, dtype=torch.float32)
    has_target = torch.zeros(n, dtype=torch.bool)

    path = Path(jsonl_path)
    if not path.exists():
        return {
            "key_idx": key_idx,
            "q_target": q_target,
            "w_weight": w_weight,
            "has_target_mask": has_target,
        }

    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                node_id = int(rec["node_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if node_id < 0 or node_id >= n:
                continue

            verdict = str(rec.get("verdict", "uncertain")).lower()
            if verdict == "uncertain":
                continue

            strength_raw = rec.get("evidence_strength")
            if strength_raw is None:
                continue
            strength = str(strength_raw).lower()
            if strength not in _STRENGTH_TO_ALIGN_WEIGHT:
                continue

            key_rel = str(rec.get("key_relation", "")).upper()
            if key_rel not in rel_index:
                continue

            k_idx = rel_index[key_rel]
            key_idx[node_id] = int(k_idx)
            q_target[node_id, :] = 0.0
            q_target[node_id, k_idx] = 1.0
            w_weight[node_id] = float(_STRENGTH_TO_ALIGN_WEIGHT[strength])
            has_target[node_id] = True

    return {
        "key_idx": key_idx,
        "q_target": q_target,
        "w_weight": w_weight,
        "has_target_mask": has_target,
    }


def _zero_like(reference: Tensor) -> Tensor:
    """Return a differentiable zero scalar that shares ``reference``'s device/graph."""
    return reference.sum() * 0.0


def build_relation_evidence_distribution(
    relation_features: Tensor,
    num_relations: int,
    rel_stat_dim: int = 9,
    tau: float = 1.5,
    eps: float = 1e-8,
) -> Tensor:
    """Build fixed relation-evidence distribution from relation statistics.

    The distribution is intentionally score-blind and non-trainable.  It uses
    the existing anonymous relation statistics rather than ``Head_r`` outputs,
    avoiding self-reinforcing gate collapse.  The strength is a compact blend
    of degree evidence, feature deviation, neighbor inconsistency, anomalous
    z-score fraction, and prototype-margin magnitude.
    """
    if relation_features.ndim != 2:
        raise ValueError(f"relation_features must be rank-2, got {tuple(relation_features.shape)}")
    R = int(num_relations)
    D = int(rel_stat_dim)
    if R <= 0 or D <= 0:
        raise ValueError("num_relations and rel_stat_dim must be positive")
    expected = R * D
    if relation_features.shape[1] != expected:
        raise ValueError(
            f"relation_features dim mismatch: {relation_features.shape[1]} != {expected}"
        )
    rel = relation_features.detach().float().view(-1, R, D)

    degree = rel[..., 0].abs()
    if D > 2:
        degree = degree + rel[..., 1].clamp_min(0.0) + rel[..., 2].clamp_min(0.0)
    deviation = rel[..., 3].clamp_min(0.0) if D > 3 else torch.zeros_like(degree)
    if D > 4:
        # Lower neighbor cosine means weaker local consistency and stronger
        # relation evidence for a correction.  Clamp avoids rewarding cosine>1
        # artefacts while preserving negative-cosine inconsistency.
        inconsistency = (1.0 - rel[..., 4]).clamp_min(0.0)
    else:
        inconsistency = torch.zeros_like(degree)
    z_fraction = rel[..., 5].clamp_min(0.0) if D > 5 else torch.zeros_like(degree)
    proto_margin = rel[..., 8].abs() if D > 8 else torch.zeros_like(degree)

    strength = (
        0.5 * degree
        + 1.0 * deviation
        + 0.75 * inconsistency
        + 1.0 * z_fraction
        + 1.0 * proto_margin
    )
    # Sample-wise centering makes the softmax depend on relative relation
    # evidence for the node, not global feature scale.
    strength = strength - strength.mean(dim=-1, keepdim=True)
    return F.softmax(strength / max(float(tau), eps), dim=-1).detach()


def compute_phase2_loss(
    outputs: dict[str, Tensor],
    y: Tensor,
    train_mask: Tensor,
    judge_align: dict[str, Tensor] | None,
    pos_weight: Tensor | float | None = None,
    lambda_int: float | None = None,
    lambda_trust: float | None = None,
    lambda_sparse: float = 1e-3,
    lambda_align: float = 1e-3,
    eta_llm: float = 2.0,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the Phase2 4-term loss.

    Args:
        outputs: Output dict from ``CoVERRelReasoner.forward``.  Must contain
            ``final_logit``, ``relation_gate``, ``delta_rel``, ``delta_llm``,
            ``alpha_llm``, ``judge_used_mask``, ``relation_strength``.
        y: Labels ``(N,)`` (0/1).
        train_mask: Boolean mask ``(N,)`` selecting train nodes.
        judge_align: Output of ``build_judge_relation_targets`` or ``None``.
            When ``None`` (or no eligible nodes) ``L_align = 0``.
        pos_weight: Positive-class weight tensor / scalar for BCE (typically
            ``N_neg / N_pos`` over train).  When ``None``, vanilla BCE is used.
        lambda_int, lambda_sparse, lambda_align: Term weights.
        lambda_trust: Deprecated alias for ``lambda_int``.  Accepted for one
            release so legacy configs continue to run.
        eta_llm: Deprecated no-op kept for one release.

    Returns:
        ``(total_loss, stats_dict)``.  Stats include each term plus
        ``mean_abs_delta_rel``, ``mean_alpha_llm``,
        ``mean_alpha_llm_accepted``, ``mean_alpha_llm_rejected``,
        ``mean_gate_entropy``, ``mean_dominance_rho``, ``judge_align_count``.
    """
    if lambda_int is None:
        if lambda_trust is not None:
            warnings.warn(
                "lambda_trust is deprecated for Phase2; use lambda_int. "
                "The alias still maps to L_intervention for this release.",
                DeprecationWarning,
                stacklevel=2,
            )
            lambda_int = float(lambda_trust)
        else:
            lambda_int = 3e-3
    if eta_llm != 2.0:
        warnings.warn(
            "eta_llm is deprecated and ignored by L_intervention.",
            DeprecationWarning,
            stacklevel=2,
        )

    final_logit = outputs["final_logit"]
    rel_gate = outputs["relation_gate"]            # (N, R)
    delta_rel = outputs["delta_rel"]               # (N,)
    delta_llm = outputs["delta_llm"]               # (N,)
    alpha_llm = outputs["alpha_llm"]               # (N,)
    judge_used = outputs["judge_used_mask"]        # (N,) bool
    relation_strength = outputs["relation_strength"]  # (N, R)
    relation_evidence_pi = outputs.get("relation_evidence_pi")

    device = final_logit.device
    train_bool = train_mask.to(device=device).view(-1).to(torch.bool)
    y_f = y.to(device=device, dtype=torch.float32).view(-1)
    if pos_weight is None:
        pw: Tensor | None = None
    elif isinstance(pos_weight, Tensor):
        pw = pos_weight.to(device=device)
    else:
        pw = torch.tensor(float(pos_weight), device=device)

    has_train = bool(train_bool.any())
    R = relation_strength.shape[1]
    eps = 1e-8

    # ------------------------------------------------------------------ L_cls
    if has_train:
        l_cls = F.binary_cross_entropy_with_logits(
            final_logit[train_bool], y_f[train_bool], pos_weight=pw
        )
    else:
        l_cls = _zero_like(final_logit)

    # ---------------------------------------------------------- L_intervention
    intervention = delta_rel + alpha_llm * delta_llm
    if has_train:
        interv_t = intervention[train_bool]
        l_intervention = (interv_t * interv_t).mean()
    else:
        l_intervention = _zero_like(final_logit)

    # ---------------------------------------------------------------- L_sparse
    # Gate entropy H(g_i) -- compute on full graph for stats.
    log_g_full = torch.log(rel_gate + eps)
    ent_full = -(rel_gate * log_g_full).sum(dim=-1)            # (N,)
    log_r = math.log(max(R, 2))
    if relation_evidence_pi is None:
        warnings.warn(
            "relation_evidence_pi missing; falling back to detached relation_strength softmax. "
            "Final CoVER configs should pass fixed evidence distributions.",
            DeprecationWarning,
            stacklevel=2,
        )
        pi_evidence = F.softmax(relation_strength.detach(), dim=-1)
    else:
        pi_evidence = relation_evidence_pi.to(device=device, dtype=final_logit.dtype).detach()
        if pi_evidence.shape != rel_gate.shape:
            raise ValueError(
                f"relation_evidence_pi shape mismatch: {tuple(pi_evidence.shape)} != {tuple(rel_gate.shape)}"
            )
        pi_evidence = pi_evidence / pi_evidence.sum(dim=-1, keepdim=True).clamp_min(eps)

    if has_train and R >= 2:
        log_pi = torch.log(pi_evidence + eps)
        l_sparse_per_node = (pi_evidence * (log_pi - log_g_full)).sum(dim=-1)
        l_sparse = l_sparse_per_node[train_bool].mean()
        ent_pi = -(pi_evidence * log_pi).sum(dim=-1)
        rho = (1.0 - ent_pi / log_r).clamp(min=0.0, max=1.0)
    else:
        # Single-relation graph: KL and dominance are both undefined/zero.
        rho = torch.zeros_like(ent_full)
        l_sparse = _zero_like(final_logit)

    # ----------------------------------------------------------------- L_align
    judge_align_count = 0
    gate_key_agreement = 0.0
    l_align: Tensor
    if judge_align is not None:
        w_weight = judge_align["w_weight"].to(device=device, dtype=final_logit.dtype)
        has_target = (
            judge_align["has_target_mask"].to(device=device).view(-1).to(torch.bool)
        )
        if "key_idx" in judge_align:
            key_idx = judge_align["key_idx"].to(device=device).view(-1).to(torch.long)
        elif "q_target" in judge_align:
            warnings.warn(
                "judge_align['q_target'] without key_idx is deprecated; using argmax(q_target).",
                DeprecationWarning,
                stacklevel=2,
            )
            key_idx = judge_align["q_target"].to(device=device).argmax(dim=-1).view(-1).to(torch.long)
        else:
            key_idx = torch.full_like(w_weight, -1, dtype=torch.long)
        eligible = train_bool & has_target & (key_idx >= 0) & (key_idx < R)
        judge_align_count = int(eligible.sum().item())
        if judge_align_count > 0:
            g = rel_gate[eligible]                              # (M, R)
            w = w_weight[eligible]                              # (M,)
            k = key_idx[eligible]                               # (M,)
            base_pi = pi_evidence[eligible].clamp_min(eps)
            tilt_logits = torch.log(base_pi)
            tilt_logits = tilt_logits.scatter_add(
                1,
                k.view(-1, 1),
                w.view(-1, 1).to(dtype=tilt_logits.dtype),
            )
            q_judge = F.softmax(tilt_logits, dim=-1)
            l_align_per_node = (
                q_judge * (torch.log(q_judge + eps) - torch.log(g + eps))
            ).sum(dim=-1)
            l_align = l_align_per_node.mean()
            with torch.no_grad():
                gate_key_agreement = float((g.argmax(dim=-1) == k).float().mean().item())
        else:
            l_align = _zero_like(final_logit)
    else:
        l_align = _zero_like(final_logit)

    total = (
        l_cls
        + float(lambda_int) * l_intervention
        + float(lambda_sparse) * l_sparse
        + float(lambda_align) * l_align
    )

    # --------------------------------------------------------------------- stats
    with torch.no_grad():
        if has_train:
            mean_abs_delta_rel = float(delta_rel[train_bool].abs().mean().item())
            mean_alpha_llm = float(alpha_llm[train_bool].mean().item())
            mean_intervention = float(intervention[train_bool].abs().mean().item())
            judge_train = judge_used & train_bool
            non_judge_train = (~judge_used) & train_bool
            base_logit = final_logit - intervention
            base_pred = base_logit[train_bool] >= 0
            y_train_bool = y_f[train_bool] >= 0.5
            base_correct_train = torch.zeros_like(train_bool)
            base_correct_train[train_bool] = base_pred == y_train_bool
            base_wrong_train = train_bool & (~base_correct_train)
            mean_alpha_acc = (
                float(alpha_llm[judge_train].mean().item())
                if bool(judge_train.any())
                else 0.0
            )
            mean_alpha_rej = (
                float(alpha_llm[non_judge_train].mean().item())
                if bool(non_judge_train.any())
                else 0.0
            )
            mean_intervention_acc = (
                float(intervention[judge_train].abs().mean().item())
                if bool(judge_train.any())
                else 0.0
            )
            mean_intervention_rej = (
                float(intervention[non_judge_train].abs().mean().item())
                if bool(non_judge_train.any())
                else 0.0
            )
            mean_intervention_base_correct = (
                float(intervention[base_correct_train].abs().mean().item())
                if bool(base_correct_train.any())
                else 0.0
            )
            mean_intervention_base_wrong = (
                float(intervention[base_wrong_train].abs().mean().item())
                if bool(base_wrong_train.any())
                else 0.0
            )
            mean_gate_entropy = float(ent_full[train_bool].mean().item())
            if R >= 2:
                mean_dominance_rho = float(rho[train_bool].mean().item())
                evidence_gate_kl = float(l_sparse_per_node[train_bool].mean().item())
                mean_evidence_entropy = float(ent_pi[train_bool].mean().item())
            else:
                mean_dominance_rho = 0.0
                evidence_gate_kl = 0.0
                mean_evidence_entropy = 0.0
        else:
            mean_abs_delta_rel = 0.0
            mean_alpha_llm = 0.0
            mean_intervention = 0.0
            mean_alpha_acc = 0.0
            mean_alpha_rej = 0.0
            mean_intervention_acc = 0.0
            mean_intervention_rej = 0.0
            mean_intervention_base_correct = 0.0
            mean_intervention_base_wrong = 0.0
            mean_gate_entropy = 0.0
            mean_dominance_rho = 0.0
            evidence_gate_kl = 0.0
            mean_evidence_entropy = 0.0

    stats: dict[str, float] = {
        "total": float(total.detach().item()),
        "l_cls": float(l_cls.detach().item()),
        "l_intervention": float(l_intervention.detach().item()),
        "l_trust": float(l_intervention.detach().item()),
        "l_sparse": float(l_sparse.detach().item()),
        "l_align": float(l_align.detach().item()),
        "mean_abs_delta_rel": mean_abs_delta_rel,
        "mean_intervention": mean_intervention,
        "mean_intervention_judge_accepted": mean_intervention_acc,
        "mean_intervention_judge_rejected": mean_intervention_rej,
        "mean_intervention_base_correct": mean_intervention_base_correct,
        "mean_intervention_base_wrong": mean_intervention_base_wrong,
        "mean_alpha_llm": mean_alpha_llm,
        "mean_alpha_llm_accepted": mean_alpha_acc,
        "mean_alpha_llm_rejected": mean_alpha_rej,
        "mean_gate_entropy": mean_gate_entropy,
        "mean_dominance_rho": mean_dominance_rho,
        "mean_evidence_entropy": mean_evidence_entropy,
        "mean_evidence_gate_kl": evidence_gate_kl,
        "judge_align_count": float(judge_align_count),
        "gate_key_agreement": gate_key_agreement,
    }
    return total, stats


__all__ = [
    "build_judge_relation_targets",
    "build_relation_evidence_distribution",
    "compute_phase2_loss",
]
