import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.prompt import build_llm_messages, assert_score_blind_payload


def _make_payload():
    return {
        "node_id": 0,
        "detector_name": "bwgnn",
        "reasoning": {
            "degree_level": "high",
            "neighbor_consistency": "low",
            "feature_neighbor_discrepancy": "high",
            "detector_signal": "high_frequency_response_high",
            "detector_signal_strength": "strong",
            "counter_signal": "benign_neighbor_signal_low",
            "allowed_support_ids": ["degree_level", "detector_signal", "detector_signal_strength"],
            "allowed_counter_ids": ["counter_signal"],
        },
    }


def test_messages_no_score_leakage():
    payload = _make_payload()
    messages = build_llm_messages(payload)
    user_msg = messages[1]["content"]
    payload_section = user_msg.split("Payload:")[1] if "Payload:" in user_msg else user_msg
    forbidden = ["base_score", "logit", "probability", "confidence"]
    for word in forbidden:
        assert word not in payload_section.lower()


def test_messages_has_allowed_ids():
    payload = _make_payload()
    messages = build_llm_messages(payload)
    user_msg = messages[1]["content"]
    assert "allowed_support_ids" in user_msg
    assert "allowed_counter_ids" in user_msg


def test_messages_has_risk_types():
    payload = _make_payload()
    messages = build_llm_messages(payload)
    user_msg = messages[1]["content"]
    assert "structural_discrepancy" in user_msg
    assert "spectral_anomaly" in user_msg


def test_messages_no_calibration():
    payload = _make_payload()
    messages = build_llm_messages(payload)
    full_text = str(messages).lower()
    assert "calibration" not in full_text


def test_score_leakage_raises():
    payload = {
        "node_id": 0,
        "base_score": 0.9,
        "reasoning": {"allowed_support_ids": [], "allowed_counter_ids": []},
    }
    with pytest.raises(ValueError, match="Score leakage"):
        build_llm_messages(payload)


def test_nested_score_leakage_raises():
    payload = {
        "node_id": 0,
        "reasoning": {
            "details": {"logit": 1.5},
            "allowed_support_ids": [],
            "allowed_counter_ids": [],
        },
    }
    with pytest.raises(ValueError, match="Score leakage"):
        build_llm_messages(payload)
