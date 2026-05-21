"""T1 — Teacher internal-head exposure tests.

Verifies that ``RAERTeacher.forward(..., return_heads=True)`` exposes
diagnostic heads for the teacher's internal relation decomposition:

* ``logit``         (N,)
* ``s_per_r``       (N, R)        per-relation pre-gate scalar Head_r(h_{i,r})
* ``c_per_r``       (N, R)        gate-weighted contribution g_r * s_r
* ``gate``          (N, R)        alias of ``relation_gate``
* ``p``             (N,)          σ(logit)
* ``delta_pre_tanh``(N,)          Σ_r c_per_r (used by ranking-stability test)

Critical invariant (reconstruction lemma):

    Σ_r c_per_r[i, r] == u_i == atanh(Δ_rel[i] / δ_max)

Equivalently:

    δ_max * tanh(Σ_r c_per_r) == Δ_rel

This is the canonical sanity check that the exposed heads truly correspond
to the teacher's internal pre-tanh sum.  CBR-Flash no longer matches these
heads; it distills only the teacher's final residual policy.
"""

from __future__ import annotations

import math

import pytest
import torch

from models.raer_teacher import RAERTeacher


def _make_teacher(num_relations: int = 3, hidden_dim: int = 16) -> RAERTeacher:
    rel_names = [f"R{i + 1}" for i in range(num_relations)]
    return RAERTeacher(
        base_z_dim=8,
        relation_names=rel_names,
        anchor_relation=rel_names[0],
        rel_stat_dim=9,
        rel_hidden_dim=hidden_dim,
        rel_num_layers=2,
        rel_dropout=0.0,
        tau_gate=0.7,
        delta_rel_max=2.0,
    )


def _random_inputs(n: int = 23, base_z_dim: int = 8, num_relations: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    base_z = torch.randn(n, base_z_dim, generator=g)
    base_logit = torch.randn(n, generator=g) * 0.5
    rel_features = torch.randn(n, num_relations * 9, generator=g) * 0.3
    return base_z, base_logit, rel_features


def test_return_heads_false_exposes_canonical_outputs():
    teacher = _make_teacher()
    teacher.eval()
    base_z, base_logit, rel_features = _random_inputs()

    with torch.no_grad():
        out_default = teacher(base_z, base_logit, rel_features)
        out_explicit = teacher(base_z, base_logit, rel_features, return_heads=False)

    expected_keys = {
        "final_logit", "rel_only_logit", "relation_gate", "delta_rel",
        "fused_rel_h", "relation_strength",
    }
    assert set(out_default.keys()) == expected_keys
    assert set(out_explicit.keys()) == expected_keys

    for k in expected_keys:
        assert torch.equal(out_default[k], out_explicit[k]), f"output drift on {k!r}"


def test_return_heads_true_exposes_diagnostic_heads():
    teacher = _make_teacher(num_relations=3)
    teacher.eval()
    base_z, base_logit, rel_features = _random_inputs(n=37, num_relations=3)

    with torch.no_grad():
        out = teacher(base_z, base_logit, rel_features, return_heads=True)

    # Extra keys added by return_heads=True.
    new_keys = {"logit", "s_per_r", "c_per_r", "gate", "p", "delta_pre_tanh"}
    assert new_keys.issubset(out.keys()), f"missing new heads: {new_keys - set(out.keys())}"

    n = base_z.shape[0]
    r = 3
    assert out["logit"].shape == (n,)
    assert out["s_per_r"].shape == (n, r)
    assert out["c_per_r"].shape == (n, r)
    assert out["gate"].shape == (n, r)
    assert out["p"].shape == (n,)
    assert out["delta_pre_tanh"].shape == (n,)


def test_logit_alias_matches_final_logit():
    teacher = _make_teacher()
    teacher.eval()
    base_z, base_logit, rel_features = _random_inputs()
    with torch.no_grad():
        out = teacher(base_z, base_logit, rel_features, return_heads=True)
    assert torch.equal(out["logit"], out["final_logit"])


def test_gate_alias_matches_relation_gate():
    teacher = _make_teacher()
    teacher.eval()
    base_z, base_logit, rel_features = _random_inputs()
    with torch.no_grad():
        out = teacher(base_z, base_logit, rel_features, return_heads=True)
    assert torch.equal(out["gate"], out["relation_gate"])


def test_p_is_sigmoid_of_logit():
    teacher = _make_teacher()
    teacher.eval()
    base_z, base_logit, rel_features = _random_inputs()
    with torch.no_grad():
        out = teacher(base_z, base_logit, rel_features, return_heads=True)
    torch.testing.assert_close(out["p"], torch.sigmoid(out["logit"]))


def test_c_per_r_reconstructs_pre_tanh_sum():
    """Critical invariant: Σ_r c_per_r[i, r] = u_i (pre-tanh).

    Equivalently, δ_max * tanh(Σ_r c_per_r) = Δ_rel.  This proves the
    exposed heads correspond to the teacher's actual internal computation
    graph — not a re-derivation that may diverge under non-trivial
    activations / normalisation.
    """
    teacher = _make_teacher()
    teacher.eval()
    # Drive heads with non-trivial inputs so all of s, gate, c are non-zero.
    # head_delta is zero-init so we have to actually train a few steps first
    # for the test to be non-trivial. Instead, manually perturb the
    # relation_heads weights to non-zero.
    for head in teacher.relation_heads.values():
        torch.nn.init.normal_(head.weight, mean=0.0, std=0.3)
        torch.nn.init.normal_(head.bias, mean=0.0, std=0.1)
    base_z, base_logit, rel_features = _random_inputs()

    with torch.no_grad():
        out = teacher(base_z, base_logit, rel_features, return_heads=True)

    # Reconstruct from exposed heads
    reconstructed_u = out["c_per_r"].sum(dim=-1)
    reconstructed_delta = teacher.delta_rel_max * torch.tanh(reconstructed_u)

    torch.testing.assert_close(out["delta_pre_tanh"], reconstructed_u, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reconstructed_delta, out["delta_rel"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out["final_logit"] - base_logit, out["delta_rel"],
                               rtol=1e-5, atol=1e-6)


def test_gate_rows_sum_to_one():
    teacher = _make_teacher()
    teacher.eval()
    base_z, base_logit, rel_features = _random_inputs()
    with torch.no_grad():
        out = teacher(base_z, base_logit, rel_features, return_heads=True)
    row_sums = out["gate"].sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-6)


def test_s_per_r_matches_head_outputs_under_known_weights():
    """If we zero gate and identity-init heads, c_per_r reduces to s_per_r / R
    (under uniform gate)."""
    teacher = RAERTeacher(
        base_z_dim=8,
        relation_names=["R1", "R2", "R3"],
        anchor_relation="R1",
        rel_stat_dim=9,
        rel_hidden_dim=16,
        rel_num_layers=2,
        rel_dropout=0.0,
        tau_gate=0.7,
        delta_rel_max=2.0,
        gate_mode="uniform",   # forces g = 1/R uniformly
    )
    for head in teacher.relation_heads.values():
        torch.nn.init.normal_(head.weight, mean=0.0, std=0.5)
    teacher.eval()
    base_z, base_logit, rel_features = _random_inputs()
    with torch.no_grad():
        out = teacher(base_z, base_logit, rel_features, return_heads=True)
    # Under uniform gate: c = (1/R) * s
    expected_c = out["s_per_r"] / 3.0
    torch.testing.assert_close(out["c_per_r"], expected_c, rtol=1e-5, atol=1e-6)


def test_no_nan_or_inf_in_heads():
    teacher = _make_teacher()
    teacher.eval()
    base_z, base_logit, rel_features = _random_inputs()
    # Inject a few extreme inputs.
    base_logit[0] = 1e3
    base_logit[1] = -1e3
    rel_features[2] = 1e3
    with torch.no_grad():
        out = teacher(base_z, base_logit, rel_features, return_heads=True)
    for k in ("logit", "s_per_r", "c_per_r", "gate", "p", "delta_pre_tanh"):
        assert torch.isfinite(out[k]).all(), f"non-finite values in {k!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
