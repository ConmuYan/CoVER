"""Smoke tests for cls-only Phase 2 loss.

The legacy 4-term test suite (L_intervention / L_sparse / L_align dynamics)
was deleted alongside the corresponding loss terms after the 5-seed paired
t-test on YelpChi-BWGNN (artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md)
found none of them statistically significant against cls-only:

    L_align    : Δ = -0.0004, p = 0.32   (removed Commit 1 v2)
    L_int      : Δ = +0.0006, p = 0.81   (removed Commit 5 cls-only)
    L_sparse   : Δ = +0.0025, p = 0.061  (removed Commit 5 cls-only)
    L_int+sp   : Δ = +0.0015, p = 0.36   (removed Commit 5 cls-only)

This file only verifies the cls-only loss is well-behaved and that legacy
kwargs are silently ignored as deprecation no-ops.
"""
from __future__ import annotations

import warnings

import pytest
import torch

from training.phase2_losses import (
    build_judge_relation_targets,
    compute_phase2_loss,
)


def _fake_outputs(n: int = 32, R: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "final_logit": torch.randn(n, requires_grad=True),
        "delta_rel": torch.randn(n, requires_grad=True) * 0.5,
        "relation_gate": torch.softmax(torch.randn(n, R), dim=-1),
        "relation_strength": torch.rand(n, R),
    }


def test_cls_only_loss_finite_and_backprop():
    out = _fake_outputs()
    y = (torch.rand(out["final_logit"].shape[0]) > 0.7).long()
    train_mask = torch.ones_like(y, dtype=torch.bool)
    loss, stats = compute_phase2_loss(out, y, train_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert out["final_logit"].grad is not None
    assert abs(stats["total"] - stats["l_cls"]) < 1e-6  # cls-only invariant
    assert stats["l_intervention"] == 0.0
    assert stats["l_sparse"] == 0.0


def test_observability_stats_populated():
    out = _fake_outputs()
    y = (torch.rand(out["final_logit"].shape[0]) > 0.7).long()
    train_mask = torch.ones_like(y, dtype=torch.bool)
    _, stats = compute_phase2_loss(out, y, train_mask)
    assert stats["mean_abs_delta_rel"] > 0.0
    assert 0.0 <= stats["mean_gate_entropy"] <= 2.0
    assert 0.0 <= stats["mean_dominance_rho"] <= 1.0


def test_legacy_kwargs_silently_ignored_with_warning():
    out = _fake_outputs()
    y = torch.zeros(out["final_logit"].shape[0], dtype=torch.long)
    train_mask = torch.ones_like(y, dtype=torch.bool)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss_a, _ = compute_phase2_loss(out, y, train_mask)
        loss_b, _ = compute_phase2_loss(
            out, y, train_mask,
            lambda_int=3e-3,
            lambda_sparse=1e-3,
            lambda_trust=3e-3,
        )
    assert torch.allclose(loss_a, loss_b)  # legacy kwargs must not change loss
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_lambda_align_still_rejected():
    out = _fake_outputs()
    y = torch.zeros(out["final_logit"].shape[0], dtype=torch.long)
    train_mask = torch.ones_like(y, dtype=torch.bool)
    with pytest.raises(NotImplementedError):
        compute_phase2_loss(out, y, train_mask, lambda_align=1e-3)


def test_build_judge_relation_targets_stub_raises():
    with pytest.raises(NotImplementedError):
        build_judge_relation_targets()


def test_loss_unchanged_by_removed_evidence_pi():
    """relation_evidence_pi is no longer consumed by the cls-only loss."""
    out_no_pi = _fake_outputs()
    out_with_pi = _fake_outputs()
    out_with_pi["relation_evidence_pi"] = torch.softmax(
        torch.randn(out_with_pi["final_logit"].shape[0], 3), dim=-1
    )
    y = torch.zeros(out_no_pi["final_logit"].shape[0], dtype=torch.long)
    train_mask = torch.ones_like(y, dtype=torch.bool)
    loss_no_pi, _ = compute_phase2_loss(out_no_pi, y, train_mask)
    loss_with_pi, _ = compute_phase2_loss(out_with_pi, y, train_mask)
    assert torch.allclose(loss_no_pi, loss_with_pi)
