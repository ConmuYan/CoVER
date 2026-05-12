import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.llm_teacher import OfflineLLMTeacher
from evidence.verifier import EvidenceContractVerifier, load_contracts


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


@pytest.fixture
def mock_teacher():
    return OfflineLLMTeacher(backend="mock")


@pytest.fixture
def verifier():
    contracts = load_contracts()
    return EvidenceContractVerifier(contracts, enable_label_compatibility=False)


def test_mock_teacher_returns_err(mock_teacher):
    payload = _make_payload()
    err, metadata = mock_teacher.generate(payload)
    assert err is not None
    assert err.risk_type is not None


def test_mock_teacher_metadata(mock_teacher):
    payload = _make_payload()
    err, metadata = mock_teacher.generate(payload)
    assert metadata["backend"] == "mock"
    assert metadata["parsed_ok"] is True
    assert metadata["raw_output"] is not None


def test_mock_teacher_passes_verifier(mock_teacher, verifier):
    payload = _make_payload()
    err, metadata = mock_teacher.generate(payload)

    from evidence.schema import CalibrationChannel, ReasoningChannel, EvidenceCard
    card = EvidenceCard(
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
            allowed_support_ids=["degree_level", "detector_signal", "detector_signal_strength"],
            allowed_counter_ids=["counter_signal"],
        ),
    )

    accepted, reasons = verifier.verify(err, card)
    assert accepted


def test_mock_teacher_spectral_anomaly():
    teacher = OfflineLLMTeacher(backend="mock")
    payload = {
        "node_id": 0,
        "detector_name": "bwgnn",
        "reasoning": {
            "degree_level": "medium",
            "neighbor_consistency": "high",
            "feature_neighbor_discrepancy": "low",
            "detector_signal": "high_frequency_response_high",
            "detector_signal_strength": "strong",
            "counter_signal": "benign_neighbor_signal_high",
            "allowed_support_ids": ["detector_signal", "detector_signal_strength"],
            "allowed_counter_ids": ["counter_signal"],
        },
    }
    err, metadata = teacher.generate(payload)
    assert err.risk_type == "spectral_anomaly"
    assert "detector_signal" in err.supporting_evidence
