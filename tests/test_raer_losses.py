"""Smoke tests for the canonical RAER teacher loss."""
from __future__ import annotations

import torch

from training.raer_losses import (
    build_relation_evidence_distribution,
    compute_raer_loss,
)


def _fake_outputs(n: int = 32, num_relations: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "final_logit": torch.randn(n, requires_grad=True),
        "delta_rel": torch.randn(n, requires_grad=True) * 0.5,
        "relation_gate": torch.softmax(torch.randn(n, num_relations), dim=-1),
        "relation_strength": torch.rand(n, num_relations),
    }


def test_raer_loss_finite_and_backprop():
    out = _fake_outputs()
    y = (torch.rand(out["final_logit"].shape[0]) > 0.7).long()
    train_mask = torch.ones_like(y, dtype=torch.bool)
    loss, stats = compute_raer_loss(out, y, train_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert out["final_logit"].grad is not None
    assert abs(stats["total"] - stats["l_cls"]) < 1e-6


def test_raer_loss_diagnostics_populated():
    out = _fake_outputs()
    y = (torch.rand(out["final_logit"].shape[0]) > 0.7).long()
    train_mask = torch.ones_like(y, dtype=torch.bool)
    _, stats = compute_raer_loss(out, y, train_mask)
    assert stats["mean_abs_delta_rel"] > 0.0
    assert 0.0 <= stats["mean_gate_entropy"] <= 2.0
    assert 0.0 <= stats["mean_dominance_rho"] <= 1.0


def test_empty_train_mask_returns_differentiable_zero():
    out = _fake_outputs()
    y = torch.zeros(out["final_logit"].shape[0], dtype=torch.long)
    train_mask = torch.zeros_like(y, dtype=torch.bool)
    loss, stats = compute_raer_loss(out, y, train_mask)
    assert torch.isfinite(loss)
    assert stats["total"] == 0.0
    loss.backward()
    assert out["final_logit"].grad is not None


def test_relation_evidence_distribution_shape_and_normalization():
    rel_features = torch.randn(16, 27)
    pi = build_relation_evidence_distribution(rel_features, num_relations=3)
    assert pi.shape == (16, 3)
    torch.testing.assert_close(pi.sum(dim=-1), torch.ones(16))
