"""Phase2 4-term loss for CoVER-REL.

Total loss::

    L = L_cls + λ_trust · L_trust + λ_sparse · L_sparse + λ_align · L_align

Term details
------------

L_cls
    Standard pos-weighted BCE on ``final_logit[train]`` against the labels.

L_trust  (bounded residual penalty)
    ``mean_train(Δ_rel² + η_llm · α_llm · Δ_llm²)``.

    The LLM term is multiplied by ``α_llm`` (NOT ``α_llm²``) so the penalty
    only fires when the gate is actually open — that way ``use_judge=False`` /
    rejected nodes contribute exactly zero to the LLM trust cost.

L_sparse  (dominance-conditioned entropy penalty)
    For each train node compute the gate entropy ``H(g_i)`` and the
    relation-strength dominance ``ρ_i = top1 - top2`` of
    ``relation_strength``.  Normalise ``ρ`` by its train max (so
    ``ρ_norm ∈ [0, 1]``) and minimise ``mean_train(ρ_norm · H)``.  When one
    relation clearly dominates, the entropy is pushed down — when the
    relations look indistinguishable, the gate is allowed to stay diffuse.

L_align  (judge soft target KL)
    Build a relation-distribution target from the accepted judge JSONL
    (``build_judge_relation_targets``) and align the gate to it via a
    weighted forward KL ``w_i · KL(q_target || g)``, averaged over nodes
    that have a target and are in train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor


# (q_key, w_weight) for each evidence_strength bucket.  q_others is then
# (1 - q_key) / (R - 1) so the row sums to 1.
_STRENGTH_TO_WEIGHTS: dict[str, tuple[float, float]] = {
    "strong":   (0.90, 1.0),
    "moderate": (0.75, 0.5),
    "weak":     (0.55, 0.2),
}


def build_judge_relation_targets(
    jsonl_path: Path,
    num_nodes: int,
    relation_names: Iterable[str],
) -> dict[str, Tensor]:
    """Build per-node soft KL targets from an ``accepted_judge.jsonl`` file.

    Skip rules (the node simply has ``has_target=False`` afterwards):

    * ``verdict == "uncertain"``
    * ``evidence_strength`` missing or not in {strong, moderate, weak}
    * ``key_relation`` not in ``relation_names``
    * ``node_id`` outside ``[0, num_nodes)``

    For accepted records we distribute mass as::

        q[k_idx]    = q_key
        q[other]    = (1 - q_key) / (R - 1)
        w_weight    = w
        has_target  = True

    with ``(q_key, w)`` per strength bucket::

        strong   → (0.90, 1.0)
        moderate → (0.75, 0.5)
        weak     → (0.55, 0.2)

    Args:
        jsonl_path: Path to ``accepted_judge.jsonl``.  A missing file returns
            an all-zero / all-False target (acts like "no judge alignment").
        num_nodes: Total number of graph nodes (defines target shape).
        relation_names: Ordered relation list (case-insensitive); the index
            of ``key_relation`` in this list determines which column of
            ``q_target`` carries the dominant mass.

    Returns:
        Dict with::

            q_target        (N, R) float32
            w_weight        (N,)   float32
            has_target_mask (N,)   bool
    """
    rel_names = [str(name).upper() for name in relation_names]
    if not rel_names:
        raise ValueError("relation_names must be non-empty")
    R = len(rel_names)
    rel_index = {name: i for i, name in enumerate(rel_names)}

    n = int(num_nodes)
    q_target = torch.zeros((n, R), dtype=torch.float32)
    w_weight = torch.zeros(n, dtype=torch.float32)
    has_target = torch.zeros(n, dtype=torch.bool)

    path = Path(jsonl_path)
    if not path.exists():
        return {
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
            if strength not in _STRENGTH_TO_WEIGHTS:
                continue

            key_rel = str(rec.get("key_relation", "")).upper()
            if key_rel not in rel_index:
                continue

            q_key, w = _STRENGTH_TO_WEIGHTS[strength]
            k_idx = rel_index[key_rel]
            if R > 1:
                q_other = (1.0 - q_key) / (R - 1)
                q_target[node_id, :] = q_other
                q_target[node_id, k_idx] = q_key
            else:
                # Degenerate single-relation case — the only proper distribution
                # places all mass on the lone relation; ignore q_key.
                q_target[node_id, 0] = 1.0
            w_weight[node_id] = float(w)
            has_target[node_id] = True

    return {
        "q_target": q_target,
        "w_weight": w_weight,
        "has_target_mask": has_target,
    }


def _zero_like(reference: Tensor) -> Tensor:
    """Return a differentiable zero scalar that shares ``reference``'s device/graph."""
    return reference.sum() * 0.0


def compute_phase2_loss(
    outputs: dict[str, Tensor],
    y: Tensor,
    train_mask: Tensor,
    judge_align: dict[str, Tensor] | None,
    pos_weight: Tensor | float | None = None,
    lambda_trust: float = 3e-3,
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
        lambda_trust, lambda_sparse, lambda_align: Term weights.
        eta_llm: Relative penalty on the LLM trust term inside ``L_trust``.

    Returns:
        ``(total_loss, stats_dict)``.  Stats include each term plus
        ``mean_abs_delta_rel``, ``mean_alpha_llm``,
        ``mean_alpha_llm_accepted``, ``mean_alpha_llm_rejected``,
        ``mean_gate_entropy``, ``mean_dominance_rho``, ``judge_align_count``.
    """
    final_logit = outputs["final_logit"]
    rel_gate = outputs["relation_gate"]            # (N, R)
    delta_rel = outputs["delta_rel"]               # (N,)
    delta_llm = outputs["delta_llm"]               # (N,)
    alpha_llm = outputs["alpha_llm"]               # (N,)
    judge_used = outputs["judge_used_mask"]        # (N,) bool
    relation_strength = outputs["relation_strength"]  # (N, R)

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

    # ----------------------------------------------------------------- L_trust
    if has_train:
        dr = delta_rel[train_bool]
        dl = delta_llm[train_bool]
        al = alpha_llm[train_bool]
        l_trust = (dr * dr + float(eta_llm) * al * (dl * dl)).mean()
    else:
        l_trust = _zero_like(final_logit)

    # ---------------------------------------------------------------- L_sparse
    # Gate entropy H(g_i) -- compute on full graph for stats; reduce on train.
    log_g_full = torch.log(rel_gate + eps)
    ent_full = -(rel_gate * log_g_full).sum(dim=-1)            # (N,)

    if has_train and R >= 2:
        sorted_s, _ = torch.sort(relation_strength, dim=-1, descending=True)
        rho = sorted_s[:, 0] - sorted_s[:, 1]                   # (N,)
        # Use train-max so the normaliser is data-driven and stable.
        rho_train_max = rho[train_bool].max().detach()
        rho_norm = rho / (rho_train_max + eps)
        l_sparse = (rho_norm[train_bool] * ent_full[train_bool]).mean()
    else:
        # Single-relation graph: entropy is constant 0 and dominance is undefined.
        rho = torch.zeros_like(ent_full)
        l_sparse = _zero_like(final_logit)

    # ----------------------------------------------------------------- L_align
    judge_align_count = 0
    l_align: Tensor
    if judge_align is not None:
        q_target = judge_align["q_target"].to(device=device, dtype=final_logit.dtype)
        w_weight = judge_align["w_weight"].to(device=device, dtype=final_logit.dtype)
        has_target = (
            judge_align["has_target_mask"].to(device=device).view(-1).to(torch.bool)
        )
        eligible = train_bool & has_target
        judge_align_count = int(eligible.sum().item())
        if judge_align_count > 0:
            q = q_target[eligible]                              # (M, R)
            g = rel_gate[eligible]                              # (M, R)
            w = w_weight[eligible]                              # (M,)
            log_q = torch.log(q + eps)
            log_g_e = torch.log(g + eps)
            kl_per_node = (q * (log_q - log_g_e)).sum(dim=-1)   # (M,)
            l_align = (w * kl_per_node).mean()
        else:
            l_align = _zero_like(final_logit)
    else:
        l_align = _zero_like(final_logit)

    total = (
        l_cls
        + float(lambda_trust) * l_trust
        + float(lambda_sparse) * l_sparse
        + float(lambda_align) * l_align
    )

    # --------------------------------------------------------------------- stats
    with torch.no_grad():
        if has_train:
            mean_abs_delta_rel = float(delta_rel[train_bool].abs().mean().item())
            mean_alpha_llm = float(alpha_llm[train_bool].mean().item())
            judge_train = judge_used & train_bool
            non_judge_train = (~judge_used) & train_bool
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
            mean_gate_entropy = float(ent_full[train_bool].mean().item())
            if R >= 2:
                mean_dominance_rho = float(rho[train_bool].mean().item())
            else:
                mean_dominance_rho = 0.0
        else:
            mean_abs_delta_rel = 0.0
            mean_alpha_llm = 0.0
            mean_alpha_acc = 0.0
            mean_alpha_rej = 0.0
            mean_gate_entropy = 0.0
            mean_dominance_rho = 0.0

    stats: dict[str, float] = {
        "total": float(total.detach().item()),
        "l_cls": float(l_cls.detach().item()),
        "l_trust": float(l_trust.detach().item()),
        "l_sparse": float(l_sparse.detach().item()),
        "l_align": float(l_align.detach().item()),
        "mean_abs_delta_rel": mean_abs_delta_rel,
        "mean_alpha_llm": mean_alpha_llm,
        "mean_alpha_llm_accepted": mean_alpha_acc,
        "mean_alpha_llm_rejected": mean_alpha_rej,
        "mean_gate_entropy": mean_gate_entropy,
        "mean_dominance_rho": mean_dominance_rho,
        "judge_align_count": float(judge_align_count),
    }
    return total, stats


__all__ = ["build_judge_relation_targets", "compute_phase2_loss"]
