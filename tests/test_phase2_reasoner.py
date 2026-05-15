"""Unit tests for the Phase2 CoVERRelReasoner model.

Tests cover: forward shapes, softmax gate properties, judge masking,
delta/alpha bounds, and relation strength output.
Uses small synthetic tensors (N=16, R=3, judge_dim=10).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cover_rel_reasoner import CoVERRelReasoner


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

N = 16
R = 3
Z_DIM = 32
REL_STAT_DIM = 9
REL_DIM = R * REL_STAT_DIM  # 27
JUDGE_DIM = 10
DELTA_REL_MAX = 2.0
DELTA_LLM_MAX = 0.75
ALPHA_MAX = 0.10


def _make_reasoner(use_judge: bool = False, alpha_max: float = 0.0) -> CoVERRelReasoner:
    return CoVERRelReasoner(
        base_z_dim=Z_DIM,
        relation_names=["RUR", "RSR", "RTR"],
        anchor_relation="RUR",
        rel_stat_dim=REL_STAT_DIM,
        rel_hidden_dim=16,
        rel_num_layers=2,
        rel_dropout=0.0,
        tau_gate=1.0,
        delta_rel_max=DELTA_REL_MAX,
        use_judge=use_judge,
        judge_feature_dim=JUDGE_DIM if use_judge else 0,
        judge_hidden_dim=16,
        judge_dropout=0.0,
        delta_llm_max=DELTA_LLM_MAX,
        alpha_max=alpha_max,
        alpha_bias_init=-3.0,
    )


def _inputs(n: int = N):
    z = torch.randn(n, Z_DIM)
    base_logit = torch.randn(n)
    rel = torch.randn(n, REL_DIM)
    return z, base_logit, rel


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forward_shape_no_judge():
    """E0: relation-only reasoner produces correct output shapes."""
    reasoner = _make_reasoner(use_judge=False)
    z, base_logit, rel = _inputs()

    out = reasoner(z, base_logit, rel)

    assert out["final_logit"].shape == (N,), f"final_logit shape {out['final_logit'].shape}"
    assert out["rel_only_logit"].shape == (N,)
    assert "relation_gate" in out, "missing relation_gate"
    assert out["relation_gate"].shape == (N, R), f"relation_gate shape {out['relation_gate'].shape}"
    assert "relation_strength" in out, "missing relation_strength"
    assert out["relation_strength"].shape == (N, R)
    assert "delta_rel" in out, "missing delta_rel"
    assert out["delta_rel"].shape == (N,)
    # No judge → alpha/delta_llm should be zero tensors
    assert torch.allclose(out["alpha_llm"], torch.zeros(N), atol=1e-7)
    assert torch.allclose(out["delta_llm"], torch.zeros(N), atol=1e-7)


def test_forward_shape_with_judge():
    """E2: judge-enabled reasoner adds alpha_llm and delta_llm."""
    reasoner = _make_reasoner(use_judge=True, alpha_max=ALPHA_MAX)
    z, base_logit, rel = _inputs()
    judge = torch.randn(N, JUDGE_DIM)
    judge_mask = torch.ones(N, dtype=torch.bool)

    out = reasoner(z, base_logit, rel, judge_features=judge, judge_mask=judge_mask)

    assert out["final_logit"].shape == (N,)
    assert "alpha_llm" in out, "missing alpha_llm"
    assert out["alpha_llm"].shape == (N,), f"alpha_llm shape {out['alpha_llm'].shape}"
    assert "delta_llm" in out, "missing delta_llm"
    assert out["delta_llm"].shape == (N,), f"delta_llm shape {out['delta_llm'].shape}"
    assert out["judge_used_mask"].shape == (N,)
    assert out["fused_rel_h"].shape[0] == N


def test_softmax_gate_sums_to_one():
    """Softmax gate weights must sum to 1 for each node."""
    reasoner = _make_reasoner(use_judge=False)
    z, base_logit, rel = _inputs()

    out = reasoner(z, base_logit, rel)
    gate = out["relation_gate"]

    assert gate.shape == (N, R)
    sums = gate.sum(dim=1)
    assert torch.allclose(sums, torch.ones(N), atol=1e-5), (
        f"gate sums not 1: min={sums.min():.6f}, max={sums.max():.6f}"
    )


def test_judge_missing_alpha_zero():
    """When judge_mask is all-zero, alpha_llm must be all-zero."""
    reasoner = _make_reasoner(use_judge=True, alpha_max=ALPHA_MAX)
    z, base_logit, rel = _inputs()
    judge = torch.randn(N, JUDGE_DIM)
    judge_mask = torch.zeros(N, dtype=torch.bool)

    out = reasoner(z, base_logit, rel, judge_features=judge, judge_mask=judge_mask)

    assert torch.allclose(out["alpha_llm"], torch.zeros(N), atol=1e-6), (
        f"alpha_llm not zero when judge_mask=0: max={out['alpha_llm'].max():.6f}"
    )


def test_judge_partial_mask():
    """Partial judge_mask: alpha=0 for masked-out nodes, possibly >0 for masked-in."""
    reasoner = _make_reasoner(use_judge=True, alpha_max=ALPHA_MAX)
    z, base_logit, rel = _inputs()
    judge = torch.randn(N, JUDGE_DIM)
    judge_mask = torch.zeros(N, dtype=torch.bool)
    judge_mask[:N // 2] = True

    out = reasoner(z, base_logit, rel, judge_features=judge, judge_mask=judge_mask)

    # Masked-out nodes must have alpha=0
    assert torch.allclose(
        out["alpha_llm"][N // 2:],
        torch.zeros(N - N // 2),
        atol=1e-6,
    ), "alpha_llm not zero for masked-out nodes"
    # All alpha values must be non-negative
    assert torch.all(out["alpha_llm"] >= -1e-6), "negative alpha_llm"


def test_delta_bounded():
    """|delta_rel| ≤ delta_rel_max and |delta_llm| ≤ delta_llm_max (with tolerance)."""
    reasoner = _make_reasoner(use_judge=True, alpha_max=ALPHA_MAX)
    z, base_logit, rel = _inputs(n=64)
    judge = torch.randn(64, JUDGE_DIM)
    judge_mask = torch.ones(64, dtype=torch.bool)

    out = reasoner(z, base_logit, rel, judge_features=judge, judge_mask=judge_mask)

    tol = 1e-5
    delta_rel = out["delta_rel"]
    assert torch.all(delta_rel.abs() <= DELTA_REL_MAX + tol), (
        f"|delta_rel| max={delta_rel.abs().max():.6f} > {DELTA_REL_MAX}"
    )

    delta_llm = out["delta_llm"]
    assert torch.all(delta_llm.abs() <= DELTA_LLM_MAX + tol), (
        f"|delta_llm| max={delta_llm.abs().max():.6f} > {DELTA_LLM_MAX}"
    )


def test_alpha_bounded():
    """alpha_llm ≤ alpha_max (with tolerance)."""
    reasoner = _make_reasoner(use_judge=True, alpha_max=ALPHA_MAX)
    z, base_logit, rel = _inputs()
    judge = torch.randn(N, JUDGE_DIM)
    judge_mask = torch.ones(N, dtype=torch.bool)

    out = reasoner(z, base_logit, rel, judge_features=judge, judge_mask=judge_mask)

    tol = 1e-5
    assert torch.all(out["alpha_llm"] <= ALPHA_MAX + tol), (
        f"alpha_llm max={out['alpha_llm'].max():.6f} > {ALPHA_MAX}"
    )


def test_initial_alpha_small():
    """At init, mean alpha should be ≤ alpha_max * 0.1 (conservative init via bias=-3)."""
    reasoner = _make_reasoner(use_judge=True, alpha_max=ALPHA_MAX)
    reasoner.eval()
    z, base_logit, rel = _inputs(n=128)
    judge = torch.randn(128, JUDGE_DIM)
    judge_mask = torch.ones(128, dtype=torch.bool)

    with torch.no_grad():
        out = reasoner(z, base_logit, rel, judge_features=judge, judge_mask=judge_mask)

    mean_alpha = out["alpha_llm"].mean().item()
    # sigmoid(-3) ≈ 0.0474, so mean alpha ≈ 0.10 * 0.0474 ≈ 0.00474
    threshold = ALPHA_MAX * 0.1
    assert mean_alpha <= threshold + 1e-4, (
        f"mean alpha at init={mean_alpha:.6f} > {threshold} (alpha_max * 0.1)"
    )


def test_relation_strength_shape():
    """relation_strength output has shape (N, R) and is non-negative."""
    reasoner = _make_reasoner(use_judge=False)
    z, base_logit, rel = _inputs()

    out = reasoner(z, base_logit, rel)

    assert "relation_strength" in out
    strength = out["relation_strength"]
    assert strength.shape == (N, R), f"relation_strength shape {strength.shape}, expected ({N}, {R})"
    # Strengths are |Head_r(h_i,r)| so must be non-negative
    assert torch.all(strength >= -1e-6), "negative relation_strength"
