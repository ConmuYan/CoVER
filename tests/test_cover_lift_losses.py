from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.vocab import get_evidence_slots, get_reason_types
from models.reasoner import EvidenceReasoner
from training.losses import (
    compute_cover_lift_loss,
    compute_cve_loss,
    compute_latent_loss,
    compute_pairrank_loss,
)


def _targets(n: int) -> dict[str, torch.Tensor]:
    slots = len(get_evidence_slots())
    targets = {
        "risk_type_id": torch.zeros(n, dtype=torch.long),
        "direction_id": torch.zeros(n, dtype=torch.long),
        "strength_id": torch.ones(n, dtype=torch.long),
        "polarity_id": torch.zeros(n, dtype=torch.long),
        "pos_mask": torch.zeros(n, slots),
        "neg_mask": torch.zeros(n, slots),
        "accepted_mask": torch.zeros(n, dtype=torch.float),
        "teacher_latents": torch.zeros(n, 6),
        "teacher_latent_mask": torch.zeros(n, dtype=torch.bool),
        "contrast_class_id": torch.zeros(n, dtype=torch.long),
    }
    targets["pos_mask"][:, 0] = 1.0
    targets["neg_mask"][:, 1] = 1.0
    return targets


def _outputs(n: int) -> dict[str, torch.Tensor]:
    slots = len(get_evidence_slots())
    types = len(get_reason_types())
    return {
        "final_logit": torch.linspace(-0.3, 0.3, n, requires_grad=True),
        "risk_type_logits": torch.randn(n, types, requires_grad=True),
        "type_logits": torch.randn(n, types, requires_grad=True),
        "direction_logits": torch.randn(n, 3, requires_grad=True),
        "strength_logits": torch.randn(n, 3, requires_grad=True),
        "support_mask_logits": torch.randn(n, slots, requires_grad=True),
        "counter_mask_logits": torch.randn(n, slots, requires_grad=True),
        "pos_logits": torch.randn(n, slots, requires_grad=True),
        "neg_logits": torch.randn(n, slots, requires_grad=True),
        "z_student": torch.randn(n, 4, requires_grad=True),
    }


def test_cover_lift_loss_terms_are_finite_and_base_logits_detached():
    n = 6
    outputs = _outputs(n)
    targets = _targets(n)
    targets["accepted_mask"][:3] = 1.0
    targets["teacher_latent_mask"][:2] = True
    y = torch.tensor([1, 0, 1, 0, 1, 0])
    train_mask = torch.ones(n, dtype=torch.bool)
    base_logits = torch.linspace(-1.0, 1.0, n, requires_grad=True)

    loss, stats = compute_cover_lift_loss(
        outputs,
        y,
        targets,
        train_mask,
        base_logits,
        pos_weight=1.0,
    )
    assert torch.isfinite(loss)
    assert stats["ap_det_loss"] >= 0
    assert stats["cve_loss"] >= 0
    assert stats["latent_loss"] >= 0
    assert stats["intervene_loss"] >= 0

    loss.backward()
    assert base_logits.grad is None


def test_zero_safe_cve_latent_and_pairrank():
    n = 4
    outputs = _outputs(n)
    targets = _targets(n)
    cve_loss, _ = compute_cve_loss(outputs, targets, targets["accepted_mask"])
    latent_loss, _ = compute_latent_loss(outputs, targets)
    pairrank_loss, pair_stats = compute_pairrank_loss(
        outputs["final_logit"],
        torch.zeros(n, dtype=torch.long),
        torch.ones(n, dtype=torch.bool),
        torch.zeros(n),
    )

    assert cve_loss.item() == 0.0
    assert latent_loss.item() == 0.0
    assert pairrank_loss.item() == 0.0
    assert pair_stats["pairrank_hard_neg_count"] == 0.0


def test_pairrank_uses_train_labels_only():
    final_logit = torch.tensor([0.2, 0.1, 2.0])
    base_logits = torch.zeros(3)
    y = torch.tensor([0, 0, 1])
    train_mask = torch.tensor([True, True, False])

    loss, stats = compute_pairrank_loss(final_logit, y, train_mask, base_logits)

    assert loss.item() == 0.0
    assert stats["pairrank_pos_count"] == 0.0


def test_bounded_penalty_and_anchor_apply_without_supported_correction():
    n = 3
    outputs = _outputs(n)
    outputs["final_logit"] = torch.tensor([2.0, -2.0, 1.5], requires_grad=True)
    targets = _targets(n)
    y = torch.tensor([1, 0, 1])
    base_logits = torch.zeros(n)
    train_mask = torch.ones(n, dtype=torch.bool)

    loss, stats = compute_cover_lift_loss(
        outputs,
        y,
        targets,
        train_mask,
        base_logits,
        max_allowed_shift=0.2,
    )

    assert torch.isfinite(loss)
    assert stats["bounded_shift_penalty"] > 0.0
    assert stats["intervention_anchor"] > 0.0
    assert stats["positive_intervention_count"] == 0.0
    assert stats["negative_intervention_count"] == 0.0


def test_fp_correction_not_forced_for_fraud_dominant_evidence():
    n = 1
    outputs = _outputs(n)
    outputs["final_logit"] = torch.tensor([-0.2], requires_grad=True)
    targets = _targets(n)
    targets["accepted_mask"][0] = 1.0
    targets["direction_id"][0] = 1
    targets["strength_id"][0] = 2
    targets["polarity_id"][0] = 0

    _, stats = compute_cover_lift_loss(
        outputs,
        y=torch.tensor([0]),
        targets=targets,
        train_mask=torch.tensor([True]),
        base_logits=torch.tensor([2.0]),
    )

    assert stats["negative_intervention_count"] == 0.0
    assert stats["anchor_count"] == 1.0


def test_reasoner_latent_outputs_and_rho_zero_recovers_base():
    n = 5
    reasoner = EvidenceReasoner(z_dim=8, rho=0.0, latent_dim=7)
    z = torch.randn(n, 8)
    base = torch.randn(n)
    evidence = torch.zeros(n, len(get_evidence_slots()), dtype=torch.long)

    outputs = reasoner(z, base, evidence)

    assert torch.allclose(outputs["final_logit"], base)
    assert torch.allclose(outputs["residual"], torch.zeros_like(base))
    assert outputs["z_student"].shape == (n, 7)
    assert outputs["strength_logits"].shape == (n, 3)
    assert outputs["risk_type_logits"].shape == outputs["type_logits"].shape
    assert outputs["support_mask_logits"].shape == outputs["pos_logits"].shape
    assert outputs["counter_mask_logits"].shape == outputs["neg_logits"].shape


def test_train_one_epoch_cover_lift_with_and_without_teacher_latents():
    from scripts.train_stage3 import add_teacher_latents_to_targets, train_one_epoch

    n = 6
    z = torch.randn(n, 8)
    base = torch.zeros(n)
    y = torch.tensor([1, 0, 1, 0, 1, 0])
    train_mask = torch.ones(n, dtype=torch.bool)
    targets = _targets(n)
    targets["evidence_token_ids"] = torch.zeros(n, len(get_evidence_slots()), dtype=torch.long)
    targets["accepted_mask"][:2] = 1.0
    targets["teacher_latent_mask"][:2] = True

    reasoner = EvidenceReasoner(z_dim=8, latent_dim=4)
    optimizer = torch.optim.Adam(reasoner.parameters(), lr=0.001)
    config = {"reasoner": {"loss_mode": "cover_lift", "pos_weight": 1.0}}
    loss_with, _ = train_one_epoch(reasoner, z, base, targets, y, train_mask, optimizer, config)

    add_teacher_latents_to_targets(targets, None, None)
    loss_without, _ = train_one_epoch(reasoner, z, base, targets, y, train_mask, optimizer, config)

    assert loss_with >= 0.0
    assert loss_without >= 0.0


def test_stage3_train_and_evaluate_are_llm_free():
    import scripts.evaluate as evaluate_script
    import scripts.train_stage3 as train_stage3_script

    train_src = inspect.getsource(train_stage3_script)
    eval_src = inspect.getsource(evaluate_script)
    forbidden = ("llm_teacher", "Qwen", "AutoTokenizer", "AutoModel", "transformers")

    for token in forbidden:
        assert token not in train_src
        assert token not in eval_src
