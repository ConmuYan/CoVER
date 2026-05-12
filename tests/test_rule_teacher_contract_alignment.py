import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.schema import CalibrationChannel, ReasoningChannel, EvidenceCard
from evidence.rule_teacher import RuleTeacher
from evidence.verifier import EvidenceContractVerifier, load_contracts
from evidence.prompt import build_teacher_payload, assert_score_blind_payload


def _make_card(
    detector_signal="normal",
    detector_signal_strength="weak",
    feature_neighbor_discrepancy="low",
    neighbor_consistency="high",
    degree_level="medium",
):
    return EvidenceCard(
        node_id=0,
        detector_name="bwgnn",
        calibration=CalibrationChannel(base_score=0.5, uncertainty=0.3),
        reasoning=ReasoningChannel(
            degree_level=degree_level,
            neighbor_consistency=neighbor_consistency,
            feature_neighbor_discrepancy=feature_neighbor_discrepancy,
            detector_signal=detector_signal,
            detector_signal_strength=detector_signal_strength,
            counter_signal="benign_neighbor_signal_high",
            allowed_support_ids=[
                "degree_level", "neighbor_consistency", "feature_neighbor_discrepancy",
                "detector_signal", "detector_signal_strength",
            ],
            allowed_counter_ids=["counter_signal"],
        ),
    )


@pytest.fixture
def teacher():
    return RuleTeacher()


@pytest.fixture
def verifier():
    contracts = load_contracts()
    return EvidenceContractVerifier(contracts, enable_label_compatibility=False)


def test_weak_strength_no_spectral_anomaly(teacher):
    card = _make_card(
        detector_signal="high_frequency_response_high",
        detector_signal_strength="weak",
    )
    err = teacher.generate(card)
    assert err.risk_type != "spectral_anomaly"


def test_strong_high_freq_spectral_anomaly(teacher):
    card = _make_card(
        detector_signal="high_frequency_response_high",
        detector_signal_strength="strong",
    )
    err = teacher.generate(card)
    assert err.risk_type == "spectral_anomaly"
    assert "detector_signal" in err.supporting_evidence


def test_spectral_anomaly_includes_detector_signal(teacher, verifier):
    card = _make_card(
        detector_signal="high_frequency_response_high",
        detector_signal_strength="strong",
    )
    err = teacher.generate(card)
    assert "detector_signal" in err.supporting_evidence
    accepted, reasons = verifier.verify(err, card)
    assert accepted


def test_feature_structure_conflict_includes_field(teacher, verifier):
    card = _make_card(feature_neighbor_discrepancy="high")
    err = teacher.generate(card)
    assert err.risk_type == "feature_structure_conflict"
    assert "feature_neighbor_discrepancy" in err.supporting_evidence
    accepted, reasons = verifier.verify(err, card)
    assert accepted


def test_structural_discrepancy_includes_field(teacher, verifier):
    card = _make_card(
        detector_signal="embedding_neighbor_discrepancy_high",
        detector_signal_strength="strong",
    )
    err = teacher.generate(card)
    assert err.risk_type == "structural_discrepancy"
    assert "detector_signal" in err.supporting_evidence
    accepted, reasons = verifier.verify(err, card)
    assert accepted


def test_generated_err_passes_verifier(teacher, verifier):
    card = _make_card(
        detector_signal="high_frequency_response_high",
        detector_signal_strength="strong",
    )
    err = teacher.generate(card)
    accepted, reasons = verifier.verify(err, card)
    assert accepted


def test_score_blind_unaffected(teacher):
    card = _make_card(
        detector_signal="high_frequency_response_high",
        detector_signal_strength="strong",
    )
    err = teacher.generate(card)
    payload = build_teacher_payload(card)
    assert_score_blind_payload(payload)


def test_case_a_generic_structural_discrepancy(teacher, verifier):
    card = _make_card(
        degree_level="high",
        neighbor_consistency="medium",
        detector_signal="embedding_neighbor_discrepancy_high",
        detector_signal_strength="strong",
    )
    err = teacher.generate(card)
    assert err.risk_type in ["structural_discrepancy", "spectral_anomaly"]
    accepted, reasons = verifier.verify(err, card)
    assert accepted


def test_case_b_camouflage_neighbor(teacher, verifier):
    card = _make_card(
        neighbor_consistency="low",
        detector_signal_strength="moderate",
    )
    err = teacher.generate(card)
    if err.risk_type == "camouflage_neighbor":
        assert "neighbor_consistency" in err.supporting_evidence
    accepted, reasons = verifier.verify(err, card)
    assert accepted


def test_case_c_feature_structure_conflict(teacher, verifier):
    card = _make_card(feature_neighbor_discrepancy="high")
    err = teacher.generate(card)
    assert err.risk_type == "feature_structure_conflict"
    assert "feature_neighbor_discrepancy" in err.supporting_evidence
    accepted, reasons = verifier.verify(err, card)
    assert accepted


def test_case_d_bwgnn_spectral_anomaly(teacher, verifier):
    card = _make_card(
        detector_signal="high_frequency_response_high",
        detector_signal_strength="strong",
    )
    err = teacher.generate(card)
    assert err.risk_type == "spectral_anomaly"
    assert "detector_signal" in err.supporting_evidence
    accepted, reasons = verifier.verify(err, card)
    assert accepted


def test_case_e_weak_high_frequency_no_spectral(teacher, verifier):
    card = _make_card(
        detector_signal="high_frequency_response_low",
        detector_signal_strength="weak",
    )
    err = teacher.generate(card)
    assert err.risk_type != "spectral_anomaly"
    accepted, reasons = verifier.verify(err, card)
    assert accepted


def test_case_f_invalid_hidden_compliance_fails(verifier):
    from evidence.schema import ERR

    card = _make_card(
        detector_signal="high_frequency_response_high",
        detector_signal_strength="strong",
    )
    err = ERR(
        node_id=0,
        risk_type="spectral_anomaly",
        supporting_evidence=["degree_level", "neighbor_consistency"],
        counter_evidence=["counter_signal"],
        summary="test",
    )
    accepted, reasons = verifier.verify(err, card)
    assert not accepted
    assert "contract_required_not_satisfied" in reasons


def test_case_g_score_blind_payload(teacher):
    card = _make_card(
        detector_signal="high_frequency_response_high",
        detector_signal_strength="strong",
    )
    payload = build_teacher_payload(card)
    assert_score_blind_payload(payload)

    payload_str = str(payload).lower()
    forbidden = ["base_score", "score", "logit", "prob", "probability", "confidence"]
    for word in forbidden:
        assert word not in payload_str
