"""Unit tests for the Phase2 4-term loss functions.

Tests cover: loss finiteness, L_trust (pure relation and with alpha),
L_sparse (dominance), L_align (accepted-only, zero-when-no-judge),
build_judge_relation_targets, and KL correctness.
Uses small synthetic tensors (N=16, R=3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.phase2_losses import (
    compute_phase2_loss,
    build_judge_relation_targets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 16
R = 3
REL_NAMES = ["RUR", "RSR", "RTR"]


def _make_outputs(n: int = N, with_judge: bool = False):
    """Synthesize reasoner output dict matching CoVERRelReasoner API."""
    out = {
        "final_logit": torch.randn(n),
        "rel_only_logit": torch.randn(n),
        "delta_rel": 0.5 * torch.tanh(torch.randn(n)),
        "delta_llm": torch.zeros(n),
        "alpha_llm": torch.zeros(n),
        "relation_gate": F.softmax(torch.randn(n, R), dim=1),
        "relation_strength": torch.rand(n, R),
        "judge_used_mask": torch.zeros(n, dtype=torch.bool),
        "fused_rel_h": torch.randn(n, 16),
    }
    if with_judge:
        out["alpha_llm"] = 0.05 * torch.sigmoid(torch.randn(n))
        out["delta_llm"] = 0.3 * torch.tanh(torch.randn(n))
        out["judge_used_mask"] = torch.ones(n, dtype=torch.bool)
    return out


def _make_labels(n: int = N):
    return torch.randint(0, 2, (n,)).float()


def _make_train_mask(n: int = N):
    return torch.ones(n, dtype=torch.bool)


def _make_pos_weight():
    return torch.tensor(1.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_loss_components_finite():
    """All loss components must be finite (no NaN/Inf)."""
    out = _make_outputs(with_judge=True)
    y = _make_labels()
    train_mask = _make_train_mask()

    # Build a synthetic judge_align dict
    judge_align = {
        "q_target": F.softmax(torch.randn(N, R), dim=1),
        "w_weight": torch.ones(N),
        "has_target_mask": torch.ones(N, dtype=torch.bool),
    }

    loss, stats = compute_phase2_loss(
        outputs=out,
        y=y,
        train_mask=train_mask,
        judge_align=judge_align,
        pos_weight=_make_pos_weight(),
        lambda_trust=3e-3,
        lambda_sparse=1e-3,
        lambda_align=1e-3,
    )

    assert torch.isfinite(loss), f"total loss not finite: {loss}"
    for key in ["l_cls", "l_trust", "l_sparse", "l_align"]:
        assert key in stats, f"missing stat key {key}"
        assert torch.isfinite(torch.tensor(stats[key])), f"{key} not finite: {stats[key]}"


def test_l_trust_pure_relation():
    """When alpha=0 (no judge), L_trust = mean(delta_rel^2)."""
    delta_rel = torch.tensor([0.1, -0.2, 0.3, 0.0])
    out = _make_outputs(n=4, with_judge=False)
    out["delta_rel"] = delta_rel
    out["alpha_llm"] = torch.zeros(4)
    out["delta_llm"] = torch.zeros(4)

    y = torch.zeros(4)
    train_mask = torch.ones(4, dtype=torch.bool)

    _, stats = compute_phase2_loss(
        outputs=out,
        y=y,
        train_mask=train_mask,
        judge_align=None,
        pos_weight=_make_pos_weight(),
        lambda_trust=1.0,
        lambda_sparse=0.0,
        lambda_align=0.0,
    )

    expected_trust = (delta_rel ** 2).mean().item()
    assert abs(stats["l_trust"] - expected_trust) < 1e-5, (
        f"l_trust={stats['l_trust']:.6f} != expected={expected_trust:.6f}"
    )


def test_l_trust_with_alpha():
    """When alpha > 0, L_trust includes eta_llm * alpha * delta_llm^2 term."""
    out = _make_outputs(n=8, with_judge=True)
    dr = torch.full((8,), 0.1)
    dl = torch.full((8,), 0.2)
    al = torch.full((8,), 0.05)
    out["delta_rel"] = dr
    out["delta_llm"] = dl
    out["alpha_llm"] = al

    y = torch.zeros(8)
    train_mask = torch.ones(8, dtype=torch.bool)
    eta_llm = 2.0

    _, stats = compute_phase2_loss(
        outputs=out,
        y=y,
        train_mask=train_mask,
        judge_align=None,
        pos_weight=_make_pos_weight(),
        lambda_trust=1.0,
        lambda_sparse=0.0,
        lambda_align=0.0,
        eta_llm=eta_llm,
    )

    # L_trust = mean(dr^2 + eta_llm * al * dl^2)
    expected = (dr * dr + eta_llm * al * (dl * dl)).mean().item()
    assert abs(stats["l_trust"] - expected) < 1e-5, (
        f"l_trust={stats['l_trust']:.6f} != expected={expected:.6f}"
    )


def test_l_sparse_dominance():
    """L_sparse: uniform gate [1/3,1/3,1/3] should have higher sparsity loss than
    peaked gate [1,0,0] when relation_strength has clear dominance."""
    # --- Uniform gate, equal strengths → high entropy, high rho_norm → high L_sparse
    out_uniform = _make_outputs(n=4)
    out_uniform["relation_gate"] = torch.ones(4, R) / R
    # Equal strengths → rho = 0 → rho_norm = 0 → L_sparse = 0
    # Actually need DIFFERENT strengths for rho to matter.
    # Use [1,0,0] strengths for both, but different gates.
    out_uniform["relation_strength"] = torch.tensor([[1.0, 0.5, 0.2]] * 4)

    # --- Peaked gate → lower entropy
    out_peaked = _make_outputs(n=4)
    peaked = torch.zeros(4, R)
    peaked[:, 0] = 1.0
    out_peaked["relation_gate"] = peaked
    out_peaked["relation_strength"] = torch.tensor([[1.0, 0.5, 0.2]] * 4)

    y = torch.zeros(4)
    train_mask = torch.ones(4, dtype=torch.bool)

    _, stats_uniform = compute_phase2_loss(
        outputs=out_uniform, y=y, train_mask=train_mask,
        judge_align=None, pos_weight=_make_pos_weight(),
        lambda_trust=0.0, lambda_sparse=1.0, lambda_align=0.0,
    )
    _, stats_peaked = compute_phase2_loss(
        outputs=out_peaked, y=y, train_mask=train_mask,
        judge_align=None, pos_weight=_make_pos_weight(),
        lambda_trust=0.0, lambda_sparse=1.0, lambda_align=0.0,
    )

    assert stats_uniform["l_sparse"] > stats_peaked["l_sparse"], (
        f"uniform sparse={stats_uniform['l_sparse']:.6f} should > "
        f"peaked sparse={stats_peaked['l_sparse']:.6f}"
    )


def test_l_align_only_accepted():
    """L_align should only consider nodes with has_target_mask=True AND train_mask."""
    out = _make_outputs(n=8, with_judge=True)
    y = torch.zeros(8)
    train_mask = torch.ones(8, dtype=torch.bool)

    # Only first 4 nodes have judge alignment targets
    judge_align = {
        "q_target": F.softmax(torch.randn(8, R), dim=1),
        "w_weight": torch.ones(8),
        "has_target_mask": torch.zeros(8, dtype=torch.bool),
    }
    judge_align["has_target_mask"][:4] = True

    _, stats = compute_phase2_loss(
        outputs=out, y=y, train_mask=train_mask,
        judge_align=judge_align, pos_weight=_make_pos_weight(),
        lambda_trust=0.0, lambda_sparse=0.0, lambda_align=1.0,
    )

    # L_align should be > 0 since we have some target nodes
    assert stats["l_align"] > 0.0, "l_align should be > 0 with accepted target nodes"
    assert stats["judge_align_count"] == 4.0, (
        f"judge_align_count={stats['judge_align_count']}, expected 4"
    )


def test_l_align_zero_when_no_judge():
    """L_align must be 0 when judge_align is None."""
    out = _make_outputs(n=8, with_judge=False)
    y = torch.zeros(8)
    train_mask = torch.ones(8, dtype=torch.bool)

    _, stats = compute_phase2_loss(
        outputs=out, y=y, train_mask=train_mask,
        judge_align=None, pos_weight=_make_pos_weight(),
        lambda_trust=0.0, lambda_sparse=0.0, lambda_align=1.0,
    )

    assert stats["l_align"] == 0.0, f"l_align={stats['l_align']} should be 0 without judge"
    assert stats["judge_align_count"] == 0.0


def test_build_judge_relation_targets_strong(tmp_path):
    """Strong quality scores produce peaked target distribution with has_target=True."""
    jsonl = tmp_path / "accepted_judge.jsonl"
    rec = {
        "node_id": 2,
        "verdict": "fraud",
        "evidence_strength": "strong",
        "key_relation": "RUR",
    }
    jsonl.write_text(json.dumps(rec) + "\n")

    result = build_judge_relation_targets(
        jsonl_path=jsonl,
        num_nodes=8,
        relation_names=REL_NAMES,
    )

    assert result["has_target_mask"][2].item() is True, "node 2 should have target"
    target = result["q_target"][2]
    assert target.shape == (R,), f"target shape {target.shape}"
    # Must be valid probability distribution
    assert torch.allclose(target.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(target >= 0.0)
    # RUR (idx 0) should carry q_key=0.90 for "strong"
    assert abs(target[0].item() - 0.90) < 1e-5, f"RUR mass={target[0]:.4f}, expected 0.90"
    # Others get (1-0.90)/2 = 0.05
    assert abs(target[1].item() - 0.05) < 1e-5
    assert abs(target[2].item() - 0.05) < 1e-5
    # Weight should be 1.0 for "strong"
    assert abs(result["w_weight"][2].item() - 1.0) < 1e-5


def test_build_judge_uncertain(tmp_path):
    """Uncertain verdict should produce has_target=False."""
    jsonl = tmp_path / "accepted_judge.jsonl"
    rec = {
        "node_id": 3,
        "verdict": "uncertain",
        "evidence_strength": "strong",
        "key_relation": "RUR",
    }
    jsonl.write_text(json.dumps(rec) + "\n")

    result = build_judge_relation_targets(
        jsonl_path=jsonl,
        num_nodes=8,
        relation_names=REL_NAMES,
    )

    assert result["has_target_mask"][3].item() is False, (
        "uncertain verdict should not produce a target"
    )


def test_kl_correctness():
    """Manual KL divergence must match L_align output."""
    # Create a known gate distribution and a known target
    gate_probs = F.softmax(torch.tensor([[2.0, 0.5, -1.0]]), dim=1)  # (1, R)
    target_probs = torch.tensor([[0.7, 0.2, 0.1]])

    out = _make_outputs(n=1, with_judge=True)
    out["relation_gate"] = gate_probs  # (1, R)

    judge_align = {
        "q_target": target_probs,      # (1, R)
        "w_weight": torch.ones(1),      # weight = 1
        "has_target_mask": torch.ones(1, dtype=torch.bool),
    }

    y = torch.zeros(1)
    train_mask = torch.ones(1, dtype=torch.bool)

    _, stats = compute_phase2_loss(
        outputs=out, y=y, train_mask=train_mask,
        judge_align=judge_align, pos_weight=_make_pos_weight(),
        lambda_trust=0.0, lambda_sparse=0.0, lambda_align=1.0,
    )

    # Manual KL(target || gate)
    eps = 1e-8
    manual_kl = (target_probs * (torch.log(target_probs + eps) - torch.log(gate_probs + eps))).sum(dim=1).mean()

    assert abs(stats["l_align"] - manual_kl.item()) < 1e-4, (
        f"l_align={stats['l_align']:.6f} != manual_kl={manual_kl.item():.6f}"
    )
