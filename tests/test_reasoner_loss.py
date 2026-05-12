import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.losses import compute_reasoner_loss
from evidence.vocab import get_evidence_slots, get_reason_types


def _make_outputs(num_nodes=10):
    num_slots = len(get_evidence_slots())
    num_types = len(get_reason_types())
    return {
        "final_logit": torch.randn(num_nodes),
        "type_logits": torch.randn(num_nodes, num_types),
        "pos_logits": torch.randn(num_nodes, num_slots),
        "neg_logits": torch.randn(num_nodes, num_slots),
    }


def _make_targets(num_nodes=10):
    num_slots = len(get_evidence_slots())
    return {
        "risk_type_id": torch.randint(0, len(get_reason_types()), (num_nodes,)),
        "pos_mask": torch.randint(0, 2, (num_nodes, num_slots)).float(),
        "neg_mask": torch.randint(0, 2, (num_nodes, num_slots)).float(),
    }


def test_accepted_mask_all_zero():
    outputs = _make_outputs()
    y = torch.zeros(10)
    targets = _make_targets()
    accepted_mask = torch.zeros(10)

    loss, loss_dict = compute_reasoner_loss(outputs, y, targets, accepted_mask)

    assert loss_dict["type"] == 0.0
    assert loss_dict["pos"] == 0.0
    assert loss_dict["neg"] == 0.0


def test_accepted_mask_has_samples():
    outputs = _make_outputs()
    y = torch.zeros(10)
    targets = _make_targets()
    accepted_mask = torch.zeros(10)
    accepted_mask[0:5] = 1.0

    loss, loss_dict = compute_reasoner_loss(outputs, y, targets, accepted_mask)

    assert loss_dict["type"] > 0.0
    assert loss_dict["pos"] > 0.0
    assert loss_dict["neg"] > 0.0


def test_rejected_not_in_loss():
    outputs = _make_outputs(5)
    y = torch.zeros(5)
    targets = _make_targets(5)
    accepted_mask = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])

    loss, loss_dict = compute_reasoner_loss(outputs, y, targets, accepted_mask)

    assert loss_dict["task"] > 0.0


def test_use_type_loss_false():
    outputs = _make_outputs()
    y = torch.zeros(10)
    targets = _make_targets()
    accepted_mask = torch.ones(10)

    loss, loss_dict = compute_reasoner_loss(
        outputs, y, targets, accepted_mask, use_type_loss=False,
    )

    assert loss_dict["type"] == 0.0


def test_use_evidence_loss_false():
    outputs = _make_outputs()
    y = torch.zeros(10)
    targets = _make_targets()
    accepted_mask = torch.ones(10)

    loss, loss_dict = compute_reasoner_loss(
        outputs, y, targets, accepted_mask, use_evidence_loss=False,
    )

    assert loss_dict["pos"] == 0.0
    assert loss_dict["neg"] == 0.0


def test_residual_l2_weight_adds_term():
    outputs = _make_outputs()
    y = torch.zeros(10)
    targets = _make_targets()
    accepted_mask = torch.zeros(10)
    base_logit = torch.randn(10)

    loss_no_reg, dict_no_reg = compute_reasoner_loss(
        outputs, y, targets, accepted_mask, base_logit=base_logit, residual_l2_weight=0.0,
    )
    loss_with_reg, dict_with_reg = compute_reasoner_loss(
        outputs, y, targets, accepted_mask, base_logit=base_logit, residual_l2_weight=0.01,
    )

    assert dict_no_reg["residual_l2"] == 0.0
    assert dict_with_reg["residual_l2"] > 0.0
    assert loss_with_reg > loss_no_reg


def test_shift_penalty_weight_adds_term():
    outputs = _make_outputs()
    y = torch.zeros(10)
    targets = _make_targets()
    accepted_mask = torch.zeros(10)
    base_logit = torch.zeros(10)

    outputs["final_logit"] = torch.tensor([3.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    loss_no_pen, dict_no_pen = compute_reasoner_loss(
        outputs, y, targets, accepted_mask, base_logit=base_logit,
        max_shift_penalty_weight=0.0, max_abs_shift=2.0,
    )
    loss_with_pen, dict_with_pen = compute_reasoner_loss(
        outputs, y, targets, accepted_mask, base_logit=base_logit,
        max_shift_penalty_weight=0.01, max_abs_shift=2.0,
    )

    assert dict_no_pen["shift_penalty"] == 0.0
    assert dict_with_pen["shift_penalty"] > 0.0
    assert loss_with_pen > loss_no_pen


def test_default_weights_dont_change_old_behavior():
    outputs = _make_outputs()
    y = torch.zeros(10)
    targets = _make_targets()
    accepted_mask = torch.ones(10)

    loss, loss_dict = compute_reasoner_loss(outputs, y, targets, accepted_mask)

    assert loss_dict["residual_l2"] == 0.0
    assert loss_dict["shift_penalty"] == 0.0
