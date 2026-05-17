"""Phase 2 cls-only loss for CoVER-REL — post 5-seed honest cleanup.

Total loss::

    L = L_cls

The 4-term legacy form ``L = L_cls + λ_int·L_int + λ_sparse·L_sparse + λ_align·L_align``
was empirically dismantled in three steps (all 5-seed paired t-tests on
YelpChi-BWGNN, see ``artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md``):

* ``L_align`` (judge-tilted KL): Δ = −0.0004, p = 0.32  → removed Commit 1 v2.
* ``L_intervention`` (base-anchored Δ_rel² penalty): Δ = +0.0006, p = 0.81
  (single addition to cls), Δ = +0.0015, p = 0.36 (with L_sparse) → removed here.
* ``L_sparse`` (KL(π_evidence ‖ g)): Δ = +0.0025, p = 0.0609 (closest to bar
  but still ns) → removed here.

The architecture (frozen base + per-relation expert MLP + schema softmax gate
+ bounded tanh residual) explains the full +0.1042 AUPRC lift from base to
canonical (A0 → L0, t = +30.4, p ≈ 7e-6); the four loss terms collectively add
nothing the cls-only baseline didn't already get.

Observability note
------------------

The companion documentation table for SAGE-YelpChi shows the discarded
``L_int + L_sparse`` config produced a ~41% std reduction (0.089 → 0.052).
That is **not** captured in this loss — it was a stability side-effect of the
4-term regularization on a high-variance base. Future runs that need that
stability should be reported as a "variance-reduction regularizer" ablation,
not folded back into the canonical loss.

Backward compatibility
----------------------

* ``compute_phase2_loss`` keeps ``judge_align``, ``lambda_int``, ``lambda_trust``,
  ``lambda_sparse``, ``lambda_align``, ``eta_llm`` kwargs so legacy configs and
  the trainer in ``scripts/train_phase2_reasoner.py`` can call without source
  edits.  Non-default values trigger a one-shot ``DeprecationWarning`` and
  are then ignored.
* ``build_judge_relation_targets`` is preserved as a stub that raises
  ``NotImplementedError`` so legacy imports continue to load.
* The returned ``stats`` dict preserves all observability fields
  (``l_cls``, ``l_intervention``, ``l_sparse``, ``l_trust``,
  ``mean_abs_delta_rel``, ``mean_gate_entropy``, ``mean_dominance_rho``, …)
  so downstream diagnostics, plots, and CSV aggregators do not break.
  Removed-term values are reported as ``0.0`` rather than raising KeyError.
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
    "falsified (paired t = -0.0004, p = 0.32 n.s.)."
)


def build_judge_relation_targets(*args, **kwargs):
    """Deprecated stub kept for legacy imports.

    The full implementation was removed when its associated ``L_align`` loss
    was 5-seed falsified.  Calling this raises ``NotImplementedError``.
    """
    raise NotImplementedError(_DEPRECATION_MSG_ALIGN)


def _zero_like(reference: Tensor) -> Tensor:
    """Return a differentiable zero scalar sharing ``reference``'s device/graph."""
    return reference.sum() * 0.0


def build_relation_evidence_distribution(
    relation_features: Tensor,
    num_relations: int,
    rel_stat_dim: int = 9,
    tau: float = 1.5,
    eps: float = 1e-8,
) -> Tensor:
    """Build a fixed relation-evidence distribution from relation statistics.

    Score-blind and non-trainable.  Originally used as the target for the
    L_sparse KL penalty before that loss term was 5-seed-falsified (p≈0.06).
    Preserved here for diagnostic / explainability use by the trainer
    (e.g., evidence-gate KL agreement is still logged as an observability
    metric even though no gradient flows through it).
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
    inconsistency = (1.0 - rel[..., 4]).clamp_min(0.0) if D > 4 else torch.zeros_like(degree)
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
    lambda_sparse: float = 0.0,
    lambda_align: float = 0.0,
    eta_llm: float = 2.0,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the Phase 2 cls-only loss.

    Args:
        outputs: Output dict from ``CoVERRelReasoner.forward``.  Reads
            ``final_logit`` (required), and the optional observability fields
            ``relation_gate``, ``delta_rel``, ``relation_strength`` (used only
            for stats — no contribution to ``total``).
        y: Labels ``(N,)`` (0/1).
        train_mask: Boolean mask ``(N,)`` selecting train nodes.
        judge_align: **Deprecated.** Must be ``None``.  Passing non-None
            raises ``NotImplementedError``.
        pos_weight: Positive-class weight tensor / scalar for BCE.
        lambda_int, lambda_trust, lambda_sparse, lambda_align, eta_llm:
            **Deprecated no-ops.**  Accepted to preserve legacy config / CLI
            compatibility.  A non-zero / non-default value emits a one-shot
            ``DeprecationWarning`` and is then ignored.  The 5-seed paired
            t-test on YelpChi-BWGNN (table
            ``artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md``)
            found none of these terms statistically significant against cls-only.

    Returns:
        ``(total_loss, stats_dict)`` where ``total_loss = L_cls``.  The stats
        dict reports observability fields (mean |Δ_rel|, gate entropy,
        dominance ρ) plus ``l_intervention = l_sparse = l_trust = 0.0`` for
        downstream CSV-aggregator compatibility.
    """
    if judge_align is not None or float(lambda_align) > 0.0:
        raise NotImplementedError(_DEPRECATION_MSG_ALIGN)

    # ---- One-shot deprecation warnings for legacy weight kwargs --------
    _warn_if_set("lambda_int", lambda_int, default=None)
    _warn_if_set("lambda_trust", lambda_trust, default=None)
    _warn_if_set("lambda_sparse", lambda_sparse, default=0.0)
    if eta_llm != 2.0:
        warnings.warn(
            "eta_llm is deprecated and ignored by the cls-only loss.",
            DeprecationWarning,
            stacklevel=2,
        )

    final_logit = outputs["final_logit"]
    delta_rel = outputs.get("delta_rel")              # (N,) optional
    rel_gate = outputs.get("relation_gate")           # (N, R) optional
    relation_strength = outputs.get("relation_strength")  # (N, R) optional

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

    # ---- L_cls (the only loss term) -----------------------------------
    if has_train:
        l_cls = F.binary_cross_entropy_with_logits(
            final_logit[train_bool], y_f[train_bool], pos_weight=pw
        )
    else:
        l_cls = _zero_like(final_logit)

    total = l_cls

    # ---- Observability stats (NOT contributing to total) --------------
    with torch.no_grad():
        if has_train and delta_rel is not None:
            mean_abs_delta_rel = float(delta_rel[train_bool].abs().mean().item())
            mean_intervention = mean_abs_delta_rel  # alias for backward CSV compat
        else:
            mean_abs_delta_rel = 0.0
            mean_intervention = 0.0

        if has_train and rel_gate is not None and rel_gate.numel() > 0:
            eps = 1e-8
            log_g = torch.log(rel_gate + eps)
            ent = -(rel_gate * log_g).sum(dim=-1)
            mean_gate_entropy = float(ent[train_bool].mean().item())
            R = rel_gate.shape[1]
            if R >= 2:
                rho = (1.0 - ent / math.log(R)).clamp(min=0.0, max=1.0)
                mean_dominance_rho = float(rho[train_bool].mean().item())
            else:
                mean_dominance_rho = 0.0
        else:
            mean_gate_entropy = 0.0
            mean_dominance_rho = 0.0

    # ---- Stats dict (backward-compatible keys) ------------------------
    stats: dict[str, float] = {
        "total": float(total.detach().item()),
        "l_cls": float(l_cls.detach().item()),
        "l_intervention": 0.0,           # removed (5-seed p=0.81 single, 0.36 combined)
        "l_trust": 0.0,                  # alias of l_intervention, also removed
        "l_sparse": 0.0,                 # removed (5-seed p=0.0609, closest to bar but fail)
        "mean_abs_delta_rel": mean_abs_delta_rel,
        "mean_intervention": mean_intervention,
        "mean_intervention_judge_accepted": 0.0,
        "mean_intervention_judge_rejected": 0.0,
        "mean_intervention_base_correct": 0.0,
        "mean_intervention_base_wrong": 0.0,
        "mean_alpha_llm": 0.0,
        "mean_alpha_llm_accepted": 0.0,
        "mean_alpha_llm_rejected": 0.0,
        "mean_gate_entropy": mean_gate_entropy,
        "mean_dominance_rho": mean_dominance_rho,
        "mean_evidence_entropy": 0.0,    # L_sparse evidence-prior is gone
        "mean_evidence_gate_kl": 0.0,    # L_sparse KL value is gone
    }
    return total, stats


# Suppress repeated deprecation warnings per kwarg name.
_WARNED: set[str] = set()


def _warn_if_set(name: str, value, default) -> None:
    if value is None:
        return
    if value == default:
        return
    if name in _WARNED:
        return
    _WARNED.add(name)
    warnings.warn(
        f"{name} is deprecated for the cls-only Phase 2 loss and is ignored. "
        f"Per artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md the "
        f"4-term loss is statistically indistinguishable from cls-only at 5 seeds.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = [
    "build_judge_relation_targets",
    "build_relation_evidence_distribution",
    "compute_phase2_loss",
]
