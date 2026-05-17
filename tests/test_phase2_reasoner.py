"""Unit tests for the cls-only Phase 2 CoVERRelReasoner.

Covers: forward shapes, softmax-gate simplex property, |Δ_rel| ≤ δ_max
bound, bounded-by-construction starting point (Δ_rel ≈ 0 from zero-init heads),
relation_strength shape/non-negativity, uniform-gate fallback, and the
evidence-group mask.

All judge / α·Δ_llm tests live in tests/test_reasoner_no_judge_path.py and
verify that any judge-on configuration raises NotImplementedError. There is
no "skipped" judge surface here — the canonical reasoner is rel-only by
construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cover_rel_reasoner import CoVERRelReasoner


N = 16
R = 3
Z_DIM = 32
REL_STAT_DIM = 9
REL_DIM = R * REL_STAT_DIM  # 27
DELTA_REL_MAX = 2.0


def _make_reasoner(
    *,
    gate_mode: str = "softmax",
    evidence_groups: list[str] | None = None,
) -> CoVERRelReasoner:
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
        gate_mode=gate_mode,
        evidence_groups=evidence_groups,
    )


def _inputs(n: int = N):
    z = torch.randn(n, Z_DIM)
    base_logit = torch.randn(n)
    rel = torch.randn(n, REL_DIM)
    return z, base_logit, rel


# ─────────────────────────── shape + invariants ──────────────────────────────

def test_forward_shapes_cls_only():
    """rel-only reasoner produces correct output shapes (no judge fields)."""
    reasoner = _make_reasoner()
    z, base_logit, rel = _inputs()

    out = reasoner(z, base_logit, rel)

    assert out["final_logit"].shape == (N,)
    assert out["rel_only_logit"].shape == (N,)
    assert out["relation_gate"].shape == (N, R)
    assert out["relation_strength"].shape == (N, R)
    assert out["delta_rel"].shape == (N,)
    # Backward-compat zero tensors for downstream diag code (rel-only contract)
    assert torch.allclose(out["alpha_llm"], torch.zeros(N), atol=1e-7)
    assert torch.allclose(out["delta_llm"], torch.zeros(N), atol=1e-7)
    assert out["judge_used_mask"].dtype == torch.bool
    assert not out["judge_used_mask"].any()


def test_softmax_gate_sums_to_one():
    reasoner = _make_reasoner(gate_mode="softmax")
    out = reasoner(*_inputs())
    sums = out["relation_gate"].sum(dim=1)
    assert torch.allclose(sums, torch.ones(N), atol=1e-5), (
        f"gate sums not 1: min={sums.min():.6f}, max={sums.max():.6f}"
    )


def test_uniform_gate_is_one_over_R():
    reasoner = _make_reasoner(gate_mode="uniform")
    out = reasoner(*_inputs())
    expected = torch.full((N, R), 1.0 / R)
    assert torch.allclose(out["relation_gate"], expected, atol=1e-6)


def test_delta_rel_bounded_by_delta_rel_max():
    reasoner = _make_reasoner()
    z, base_logit, rel = _inputs(n=64)
    out = reasoner(z, base_logit, rel)
    tol = 1e-5
    assert torch.all(out["delta_rel"].abs() <= DELTA_REL_MAX + tol), (
        f"|delta_rel| max={out['delta_rel'].abs().max():.6f} > {DELTA_REL_MAX}"
    )


def test_initial_delta_rel_is_zero_by_zero_init_heads():
    """At init: per-relation residual heads are zero, so Δ_rel ≡ 0 and
    final_logit ≡ base_logit. This is the architectural do-nothing baseline."""
    reasoner = _make_reasoner()
    reasoner.eval()
    z, base_logit, rel = _inputs()
    with torch.no_grad():
        out = reasoner(z, base_logit, rel)
    assert torch.allclose(out["delta_rel"], torch.zeros(N), atol=1e-6)
    assert torch.allclose(out["final_logit"], base_logit, atol=1e-6)


def test_relation_strength_non_negative():
    reasoner = _make_reasoner()
    out = reasoner(*_inputs())
    assert torch.all(out["relation_strength"] >= -1e-6)


# ─────────────────────────── evidence-group mask ─────────────────────────────

def test_evidence_groups_subset_zeros_dropped_dims():
    """When a group is dropped, the corresponding per-relation dims are
    masked to zero before being fed to the experts."""
    reasoner = _make_reasoner(evidence_groups=["A", "B"])  # drop C (dims 6-8)
    # Internal mask should zero indices 6,7,8 of each per-relation chunk.
    mask = reasoner.evidence_mask  # shape (9,)
    assert mask.shape == (REL_STAT_DIM,)
    assert torch.all(mask[:6] == 1.0)
    assert torch.all(mask[6:9] == 0.0)


def test_evidence_groups_default_is_all():
    reasoner = _make_reasoner()  # default = [A, B, C]
    assert reasoner.evidence_groups == ["A", "B", "C"]
    assert torch.all(reasoner.evidence_mask == 1.0)


# ─────────────────────────── error contracts ─────────────────────────────────

def test_bad_anchor_raises():
    with pytest.raises(ValueError, match="anchor_relation"):
        CoVERRelReasoner(
            base_z_dim=Z_DIM,
            relation_names=["RUR", "RSR"],
            anchor_relation="UPU",  # not in list
        )


def test_bad_gate_mode_raises():
    with pytest.raises(ValueError, match="gate_mode"):
        CoVERRelReasoner(
            base_z_dim=Z_DIM,
            relation_names=["RUR", "RSR"],
            anchor_relation="RUR",
            gate_mode="sparsemax",  # unsupported
        )


def test_empty_evidence_groups_raises():
    with pytest.raises(ValueError, match="evidence_groups"):
        CoVERRelReasoner(
            base_z_dim=Z_DIM,
            relation_names=["RUR", "RSR"],
            anchor_relation="RUR",
            evidence_groups=[],
        )
