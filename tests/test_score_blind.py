import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.schema import CalibrationChannel, ReasoningChannel, EvidenceCard, ERR
from evidence.prompt import build_teacher_payload, assert_score_blind_payload


def _make_card(node_id=0, base_score=0.5):
    return EvidenceCard(
        node_id=node_id,
        detector_name="gcn",
        calibration=CalibrationChannel(base_score=base_score, uncertainty=0.3),
        reasoning=ReasoningChannel(
            degree_level="medium",
            neighbor_consistency="high",
            feature_neighbor_discrepancy="low",
            detector_signal="normal",
            detector_signal_strength="weak",
            counter_signal="benign_neighbor_signal_high",
            allowed_support_ids=["neighbor_1", "neighbor_2"],
            allowed_counter_ids=["counter_1"],
        ),
    )


def test_payload_no_score_leakage():
    card = _make_card(base_score=0.95)
    payload = build_teacher_payload(card)
    assert "base_score" not in payload
    assert "score" not in payload
    assert "logit" not in payload
    assert "prob" not in payload


def test_payload_has_reasoning():
    card = _make_card()
    payload = build_teacher_payload(card)
    assert "reasoning" in payload
    assert "degree_level" in payload["reasoning"]


def test_score_blind_passes():
    payload = {
        "node_id": 0,
        "detector_name": "gcn",
        "reasoning": {
            "degree_level": "medium",
            "neighbor_consistency": "high",
        },
    }
    assert_score_blind_payload(payload)


def test_score_blind_fails_on_base_score():
    payload = {
        "node_id": 0,
        "base_score": 0.92,
        "reasoning": {},
    }
    with pytest.raises(ValueError, match="Score leakage"):
        assert_score_blind_payload(payload)


def test_score_blind_fails_on_nested_logit():
    payload = {
        "node_id": 0,
        "reasoning": {
            "details": {
                "logit": 1.5,
            },
        },
    }
    with pytest.raises(ValueError, match="Score leakage"):
        assert_score_blind_payload(payload)


def test_score_blind_fails_on_nested_confidence():
    payload = {
        "node_id": 0,
        "items": [{"confidence": 0.9}],
    }
    with pytest.raises(ValueError, match="Score leakage"):
        assert_score_blind_payload(payload)


def test_calibration_has_score_but_payload_does_not():
    card = _make_card(base_score=0.95)
    assert card.calibration.base_score == 0.95
    payload = build_teacher_payload(card)
    assert "base_score" not in str(payload)
