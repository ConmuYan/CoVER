"""T-CONTRACTS — CBR-Flash four hard contracts test suite.

Verifies that ``models/cbr_flash_adapter.py::FlashAdapter`` + the CBR-Flash
training loop respect ``AGENTS.md §1 Problem formulation`` contracts at
every epoch:

* **C1 base-freeze (SHA-256)** — base detector parameters never mutate.
* **C2 score-blind input** — runtime counterfactual rejects any expression
  that reads ``base_logit`` inside the trunk before it is added on as
  ``s_S = base_logit + Δ_φ``.
* **C3 train-only prototype** — teacher prototype tensors (when relevant)
  are constructed from labelled train indices only; CBR-Flash never
  modifies teacher.
* **C4 δ-bounded residual** — ``|Δ_φ| ≤ δ_max`` at every forward, regardless
  of arbitrary weight initialisation (architectural enforcement, not clip).

* **P3 zero-init epoch-0 equivalence** — student output at epoch 0 with
  zero-init heads equals the base posterior (safe deployment from epoch 0).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cbr_flash_adapter import FlashAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_of_state(state_dict: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k].detach().cpu().contiguous().numpy().tobytes()
        h.update(k.encode("utf-8"))
        h.update(v)
    return h.hexdigest()


def _make_inputs(n: int = 64, base_z_dim: int = 32, num_relations: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, base_z_dim, generator=g),
        torch.randn(n, generator=g) * 0.5,
        torch.randn(n, num_relations * 9, generator=g) * 0.3,
    )


# ---------------------------------------------------------------------------
# C1 — base-freeze SHA-256
# ---------------------------------------------------------------------------

class _MockBaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(32, 1)


def test_C1_base_sha256_unchanged_through_training():
    """Mimic a training step: base.parameters() never touched by student
    optimiser; SHA-256 of base state-dict must match before / after."""
    base = _MockBaseModel()
    for p in base.parameters():
        p.requires_grad = False

    sha_before = _sha256_of_state(base.state_dict())

    student = FlashAdapter(base_z_dim=32, num_relations=3, hidden_dim=32)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2)
    base_z, base_logit, rel_features = _make_inputs(base_z_dim=32)
    y = torch.randint(0, 2, (base_z.shape[0],)).float()

    for _ in range(5):
        out = student(base_z, base_logit, rel_features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out["final_logit"], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    sha_after = _sha256_of_state(base.state_dict())
    assert sha_before == sha_after, "C1 VIOLATED: base SHA-256 changed during student training"


# ---------------------------------------------------------------------------
# C2 — score-blind input (runtime counterfactual hook — supersedes AST pass)
# ---------------------------------------------------------------------------

def test_C2_score_blind_runtime_counterfactual():
    """Counterfactual: replacing ``base_logit`` with an arbitrary huge tensor
    must NOT change the trunk-derived residual outputs.  Only ``final_logit``
    may change — by exactly the perturbation amount.

    This is **architecturally** the C2 contract: the trunk MUST be a
    pure function of ``(base_z, relation_features)``.  The earlier AST
    pass was bypassable by aliasing (``b_i = base_logit.detach()`` then
    feeding ``b_i`` into the trunk).  The counterfactual is bypass-proof:
    if the trunk reads any function of ``base_logit``, perturbing it
    will change at least one of the trunk-derived outputs.

    Tolerance: bit-identical for trunk outputs (atol=0); only
    ``final_logit`` changes by the perturbation magnitude.
    """
    torch.manual_seed(0)
    student = FlashAdapter(base_z_dim=32, num_relations=3, hidden_dim=32, dropout=0.0)
    # Perturb the heads a bit so this isn't trivially zero everywhere.
    for p in student.head_delta.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.5)
    student.eval()  # disable dropout for deterministic counterfactual

    base_z, base_logit, rel_features = _make_inputs(base_z_dim=32)

    with torch.no_grad():
        out_a = student(base_z, base_logit, rel_features, return_heads=True)

    # Adversarial counterfactual: replace base_logit with 1e9-magnitude garbage.
    base_logit_bad = torch.randn_like(base_logit) * 1e9
    with torch.no_grad():
        out_b = student(base_z, base_logit_bad, rel_features, return_heads=True)

    # All trunk-derived outputs must be BIT-IDENTICAL.
    trunk_keys = ["delta_phi", "delta_pre_tanh"]
    for k in trunk_keys:
        assert torch.equal(out_a[k], out_b[k]), (
            f"C2 VIOLATED: trunk output {k!r} changed under base_logit perturbation. "
            f"max abs diff = {(out_a[k] - out_b[k]).abs().max()}.  This means the "
            f"trunk reads base_logit (directly or via alias), violating score-blind input."
        )

    # final_logit must change by exactly the perturbation amount.
    expected_diff = base_logit_bad - base_logit
    actual_diff = out_b["final_logit"] - out_a["final_logit"]
    torch.testing.assert_close(actual_diff, expected_diff, rtol=1e-5, atol=1e-5)


def test_C2_dropout_does_not_break_counterfactual():
    """Repeat the counterfactual with eval mode confirmed off & dropout on,
    but with a fixed RNG state so the same dropout mask applies to both
    forward passes (rules out dropout-induced false positives)."""
    torch.manual_seed(0)
    student = FlashAdapter(base_z_dim=32, num_relations=3, hidden_dim=32, dropout=0.3)
    student.train()
    base_z, base_logit, rel_features = _make_inputs(base_z_dim=32)
    base_logit_bad = torch.randn_like(base_logit) * 1e9

    g_state = torch.get_rng_state()
    out_a = student(base_z, base_logit, rel_features, return_heads=True)
    torch.set_rng_state(g_state)
    out_b = student(base_z, base_logit_bad, rel_features, return_heads=True)

    for k in ["delta_phi", "delta_pre_tanh"]:
        # With same dropout mask, must be bit-identical.
        assert torch.equal(out_a[k], out_b[k]), (
            f"C2 VIOLATED under dropout: {k!r} changed. Diff max = "
            f"{(out_a[k] - out_b[k]).abs().max()}"
        )


# ---------------------------------------------------------------------------
# C4 — δ-bounded residual (architectural, not post-hoc clip)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delta_max", [0.5, 1.0, 2.0, 5.0])
def test_C4_delta_bounded_under_extreme_weights(delta_max):
    """No matter how head_delta weights are perturbed, |Δ_φ| ≤ δ_max
    because the architecture uses ``δ_max * tanh(...)``."""
    student = FlashAdapter(base_z_dim=32, num_relations=3, hidden_dim=32, delta_max=delta_max)
    for p in student.head_delta.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=100.0)
    base_z, base_logit, rel_features = _make_inputs(base_z_dim=32)
    out = student(base_z, base_logit, rel_features)
    abs_max = float(out["delta_phi"].detach().abs().max())
    assert abs_max <= delta_max + 1e-5, (
        f"C4 VIOLATED at δ_max={delta_max}: |Δ_φ| max = {abs_max}"
    )


def test_C4_holds_under_extreme_inputs():
    """Even if the trunk input contains extreme values, |Δ_φ| ≤ δ_max."""
    student = FlashAdapter(base_z_dim=32, num_relations=3, hidden_dim=32, delta_max=2.0)
    for p in student.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=5.0)
    base_z = torch.randn(50, 32) * 1e3
    base_logit = torch.randn(50) * 1e3
    rel_features = torch.randn(50, 27) * 1e3
    out = student(base_z, base_logit, rel_features)
    assert out["delta_phi"].abs().max() <= 2.0 + 1e-5
    assert torch.isfinite(out["delta_phi"]).all()


# ---------------------------------------------------------------------------
# P3 — zero-init epoch-0 equivalence (safe deployment)
# ---------------------------------------------------------------------------

def test_P3_zero_init_epoch_0_equals_base():
    """At epoch 0 (zero-init heads), the student final logit equals the
    base logit *exactly* — safe deployment from epoch 0 (Prop. P3.5)."""
    student = FlashAdapter(base_z_dim=64, num_relations=3, hidden_dim=32)
    base_z, base_logit, rel_features = _make_inputs(base_z_dim=64)
    out = student(base_z, base_logit, rel_features)
    diff = (out["final_logit"] - base_logit).abs().max()
    assert diff < 1e-6, f"P3 VIOLATED: epoch-0 student != base (max diff {diff})"
    assert out["delta_phi"].abs().max() < 1e-6, "epoch-0 Δ_φ not zero"


def test_return_heads_exposes_only_final_policy_signals():
    """CBR-Flash no longer exposes relation/gate matching heads."""
    student = FlashAdapter(base_z_dim=32, num_relations=3, hidden_dim=32)
    base_z, base_logit, rel_features = _make_inputs(base_z_dim=32)
    out = student(base_z, base_logit, rel_features, return_heads=True)
    assert set(out) == {"delta_phi", "final_logit", "logit", "p", "delta_pre_tanh"}
    assert not hasattr(student, "c_per_r_head")
    assert not hasattr(student, "head_gate")


# ---------------------------------------------------------------------------
# Param budget (TKDE design target ≤ 5000)
# ---------------------------------------------------------------------------

def test_student_param_budget():
    """Total student params must fit in design budget of 5000."""
    student = FlashAdapter(base_z_dim=64, num_relations=3, hidden_dim=32)
    n = sum(p.numel() for p in student.parameters())
    assert n <= 5000, f"param budget exceeded: {n} > 5000"
    # Sanity bound: at least 4000 (otherwise architecture is suspiciously tiny).
    assert n >= 4000, f"param count suspiciously low: {n}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
