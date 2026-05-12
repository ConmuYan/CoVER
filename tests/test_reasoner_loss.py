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
