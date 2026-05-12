import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.llm_teacher import OfflineLLMTeacher
from evidence.verifier import EvidenceContractVerifier, load_contracts
from evidence.schema import CalibrationChannel, ReasoningChannel, EvidenceCard


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
            "allowed_support_ids": ["degree_level", "detector_signal", "detector_signal_strength", "feature_neighbor_discrepancy"],
            "allowed_counter_ids": ["counter_signal"],
        },
    }


def _make_card():
    return EvidenceCard(
        node_id=0,
        detector_name="bwgnn",
        calibration=CalibrationChannel(base_score=0.5, uncertainty=0.3),
        reasoning=ReasoningChannel(
            degree_level="high",
            neighbor_consistency="low",
            feature_neighbor_discrepancy="high",
            detector_signal="high_frequency_response_high",
            detector_signal_strength="strong",
            counter_signal="benign_neighbor_signal_low",
            allowed_support_ids=["degree_level", "detector_signal", "detector_signal_strength", "feature_neighbor_discrepancy"],
            allowed_counter_ids=["counter_signal"],
        ),
    )


@pytest.fixture
def verifier():
    contracts = load_contracts()
    return EvidenceContractVerifier(contracts, enable_label_compatibility=False)


def test_retry_prompt_score_blind():
    teacher = OfflineLLMTeacher(backend="mock")
    payload = _make_payload()
    err, metadata = teacher.generate(payload)

    from evidence.prompt import build_retry_messages
    messages = build_retry_messages(payload, err, ["contract_required_not_satisfied"])

    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    user_text = " ".join(user_msgs).lower()

    assert "base_score" not in user_text or "do not mention" in user_text
    assert "logit" not in user_text or "do not mention" in user_text


def test_mock_verifier_retry_accepts(verifier):
    teacher = OfflineLLMTeacher(backend="mock", enable_verifier_retry=True, max_verifier_retries=1)
    payload = _make_payload()
    card = _make_card()

    err, metadata = teacher.generate_with_verifier_retry(payload, verifier, card)
    assert err is not None
    assert metadata.get("final_status") in ("accepted", "accepted_after_retry")


def test_retry_metadata_structure(verifier):
    teacher = OfflineLLMTeacher(backend="mock", enable_verifier_retry=True, max_verifier_retries=1)
    payload = _make_payload()
    card = _make_card()

    err, metadata = teacher.generate_with_verifier_retry(payload, verifier, card)
    assert "attempts" in metadata
    assert "final_status" in metadata
    assert "verifier_retries" in metadata
