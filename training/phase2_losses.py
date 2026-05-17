"""Phase 2 3-term loss for CoVER-REL — Phase 3 cleanup version.

Total loss::

    L = L_cls + λ_int · L_intervention + λ_sparse · L_sparse

The fourth term ``λ_align · L_align`` (judge-tilted evidence alignment)
was removed in Commit 1 v2 (Phase 3 cleanup) after 5-seed paired t-test
showed Δ = -0.0004, p = 0.32 (n.s.) on YelpChi-BWGNN.  See
``PHASE2_DEPRECATION_PLAN_CORRECTION.md`` for the rationale.

Term details
------------

L_cls
    Standard pos-weighted BCE on ``final_logit[train]`` against the labels.

L_intervention  (base-anchored correction penalty)
    ``mean_train((z - b)²)`` = ``mean_train((Δ_rel + α · Δ_llm)²)``.
    With the judge path removed, ``α = Δ_llm ≡ 0`` so this collapses to
    ``mean_train(Δ_rel²)``.

    The deprecated ``lambda_trust`` name is accepted as an alias for one
    release.  ``eta_llm`` is kept as a no-op kwarg for backward compatibility.

L_sparse  (evidence-to-gate consistency)
    KL(π_evidence || g) where π_evidence is a fixed, non-trainable
    relation-evidence distribution built from the relation feature packet.

Backward compatibility
----------------------

* ``compute_phase2_loss`` keeps ``judge_align``, ``lambda_align``, and
  ``eta_llm`` kwargs so legacy callers can import / construct loss
  configs.  Passing a non-None ``judge_align`` or ``lambda_align > 0``
  raises ``NotImplementedError``.
* ``build_judge_relation_targets`` is preserved as a stub that raises
  ``NotImplementedError`` so legacy ``from training.phase2_losses import
  build_judge_relation_targets`` continues to import without error.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor


_DEPRECATION_MSG_ALIGN = (
    "L_align (judge-tilted evidence alignment) removed in Commit 1 v2 "
    "(Phase 3 cleanup). Per RESEARCH_BRIEF.md §3.2 it was 5-seed "
    "falsified (paired t = -0.0004, p = 0.32 n.s.). LEQA replacement "
    "coming in Commit 2. See PHASE2_DEPRECATION_PLAN_CORRECTION.md."
)


def build_judge_relation_targets(*args, **kwargs):
    """Deprecated stub.

    The full implementation was removed in Commit 1 v2 because the
    associated ``L_align`` loss was 5-seed falsified.  The stub keeps
    legacy imports unbroken while making any actual call explicitly
    fail.
    """
    raise NotImplementedError(_DEPRECATION_MSG_ALIGN)


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

    Score-blind and non-trainable.  Uses anonymous relation statistics
    rather than ``Head_r`` outputs to avoid self-reinforcing gate collapse.
    Strength blends degree evidence, feature deviation, neighbor
    inconsistency, anomalous z-score fraction, and prototype-margin
    magnitude.
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
    strength = strength - strength.mean(dim=-1, keepdim=True)
    return F.softmax(strength / max(float(tau), eps), dim=-1).detach()


def compute_phase2_loss(
    outputs: dict[str, Tensor],
    y: Tensor,
    train_mask: Tensor,
    judge_align: dict[str, Tensor] | None = None,
    pos_weight: Tensor | float | None = None,
    lambda_int: float | None = None,
    lambda_trust: float | None = None,
    lambda_sparse: float = 1e-3,
    lambda_align: float = 0.0,
    eta_llm: float = 2.0,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the Phase 2 3-term loss (post-Commit-1-v2 cleanup).

    Args:
        outputs: Output dict from ``CoVERRelReasoner.forward``.  Must contain
            ``final_logit``, ``relation_gate``, ``delta_rel``, ``delta_llm``,
            ``alpha_llm``, ``judge_used_mask``, ``relation_strength``.
        y: Labels ``(N,)`` (0/1).
        train_mask: Boolean mask ``(N,)`` selecting train nodes.
        judge_align: **Deprecated.** Must be ``None``.  Passing non-None
            raises ``NotImplementedError``.
        pos_weight: Positive-class weight tensor / scalar for BCE.
        lambda_int, lambda_sparse: Term weights.
        lambda_trust: Deprecated alias for ``lambda_int``.  Accepted for one
            release so legacy configs continue to run.
        lambda_align: **Deprecated.** Must be ``0.0``.  Any value > 0
            raises ``NotImplementedError``.
        eta_llm: Deprecated no-op.

    Returns:
        ``(total_loss, stats_dict)``.
    """
    # ----- Reject deprecated judge alignment -----
    if judge_align is not None or float(lambda_align) > 0.0:
        raise NotImplementedError(_DEPRECATION_MSG_ALIGN)

    if lambda_int is None:
        if lambda_trust is not None:
            warnings.warn(
                "lambda_trust is deprecated for Phase 2; use lambda_int. "
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
    log_g_full = torch.log(rel_gate + eps)
    ent_full = -(rel_gate * log_g_full).sum(dim=-1)
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
        rho = torch.zeros_like(ent_full)
        l_sparse = _zero_like(final_logit)

    total = (
        l_cls
        + float(lambda_int) * l_intervention
        + float(lambda_sparse) * l_sparse
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
    }
    return total, stats


__all__ = [
    "build_judge_relation_targets",
    "build_relation_evidence_distribution",
    "compute_phase2_loss",
]
