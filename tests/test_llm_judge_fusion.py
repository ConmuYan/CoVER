from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.reasoner import EvidenceReasoner
from training.losses import compute_cover_judge_loss


def _reasoner() -> EvidenceReasoner:
    return EvidenceReasoner(
        z_dim=8,
        num_slots=20,
        hidden_dim=16,
        rho=0.1,
        relation_dim=0,
        use_llm_judge=True,
        judge_feature_dim=12,
        fusion_mode="gated_llm_residual",
        llm_delta_scale=1.0,
    )


def test_llm_judge_fusion_reduces_to_rel_only_when_judge_missing():
    reasoner = _reasoner()
    z = torch.randn(5, 8)
    base = torch.randn(5)
    evidence = torch.zeros(5, 20, dtype=torch.long)
    judge = torch.randn(5, 12)
    mask = torch.zeros(5, dtype=torch.bool)

    out = reasoner(z, base, evidence, judge_features=judge, judge_mask=mask)
    assert torch.allclose(out["final_logit"], out["rel_only_logit"])
    assert torch.allclose(out["alpha_llm"], torch.zeros_like(out["alpha_llm"]))


def test_alpha_llm_is_finite_and_bounded():
    reasoner = _reasoner()
    z = torch.randn(5, 8)
    base = torch.randn(5)
    evidence = torch.zeros(5, 20, dtype=torch.long)
    judge = torch.randn(5, 12)
    mask = torch.ones(5, dtype=torch.bool)

    out = reasoner(z, base, evidence, judge_features=judge, judge_mask=mask)
    assert torch.isfinite(out["alpha_llm"]).all()
    assert torch.all((out["alpha_llm"] >= 0) & (out["alpha_llm"] <= 1))


def test_alpha_max_caps_llm_alpha():
    reasoner = EvidenceReasoner(
        z_dim=8,
        num_slots=20,
        hidden_dim=16,
        rho=0.1,
        use_llm_judge=True,
        judge_feature_dim=12,
        fusion_mode="gated_llm_residual",
        alpha_max=0.5,
    )
    z = torch.randn(5, 8)
    base = torch.randn(5)
    evidence = torch.zeros(5, 20, dtype=torch.long)
    judge = torch.randn(5, 12)
    mask = torch.ones(5, dtype=torch.bool)

    out = reasoner(z, base, evidence, judge_features=judge, judge_mask=mask)
    assert torch.all(out["alpha_llm"] <= 0.5)


def test_strength_aware_alpha_downweights_uncertain_verdict():
    base_kwargs = {
        "z_dim": 8,
        "num_slots": 20,
        "hidden_dim": 16,
        "rho": 0.1,
        "use_llm_judge": True,
        "judge_feature_dim": 12,
        "fusion_mode": "gated_llm_residual",
        "alpha_max": 1.0,
    }
    reasoner = EvidenceReasoner(**base_kwargs, strength_aware_alpha=True)
    z = torch.randn(3, 8)
    base = torch.randn(3)
    evidence = torch.zeros(3, 20, dtype=torch.long)
    judge = torch.zeros(3, 12)
    judge[:, 2] = 1.0  # uncertain verdict
    judge[:, 5] = 1.0  # strong evidence
    mask = torch.ones(3, dtype=torch.bool)

    out = reasoner(z, base, evidence, judge_features=judge, judge_mask=mask)
    assert torch.all(out["alpha_llm"] <= 0.2)


def test_llm_residual_regularization_is_finite_and_rejected_excluded():
    reasoner = _reasoner()
    z = torch.randn(6, 8)
    base = torch.randn(6)
    evidence = torch.zeros(6, 20, dtype=torch.long)
    judge = torch.randn(6, 12)
    judge_mask = torch.tensor([True, False, True, False, False, False])
    outputs = reasoner(z, base, evidence, judge_features=judge, judge_mask=judge_mask)
    loss, stats = compute_cover_judge_loss(
        outputs,
        y=torch.tensor([1, 0, 1, 0, 0, 1]),
        targets={"judge_mask": judge_mask},
        train_mask=torch.ones(6, dtype=torch.bool),
        base_logits=base,
    )

    assert torch.isfinite(loss)
    assert stats["judge_train_count"] == 2.0
    assert stats["llm_reg_loss"] >= 0.0
