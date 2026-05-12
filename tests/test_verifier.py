import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.schema import CalibrationChannel, ReasoningChannel, EvidenceCard, ERR
from evidence.verifier import EvidenceContractVerifier, load_contracts


def _make_card(
    degree_level="medium",
    neighbor_consistency="high",
    feature_neighbor_discrepancy="low",
    detector_signal="normal",
    detector_signal_strength="weak",
    counter_signal="benign_neighbor_signal_high",
    allowed_support_ids=None,
    allowed_counter_ids=None,
):
    if allowed_support_ids is None:
        allowed_support_ids = [
            "degree_level", "neighbor_consistency", "feature_neighbor_discrepancy",
            "detector_signal", "detector_signal_strength", "neighbor_1", "neighbor_2",
        ]
    if allowed_counter_ids is None:
        allowed_counter_ids = ["counter_signal", "counter_1"]

    return EvidenceCard(
        node_id=0,
        detector_name="gcn",
        calibration=CalibrationChannel(base_score=0.5, uncertainty=0.3),
        reasoning=ReasoningChannel(
            degree_level=degree_level,
            neighbor_consistency=neighbor_consistency,
            feature_neighbor_discrepancy=feature_neighbor_discrepancy,
            detector_signal=detector_signal,
            detector_signal_strength=detector_signal_strength,
            counter_signal=counter_signal,
            allowed_support_ids=allowed_support_ids,
            allowed_counter_ids=allowed_counter_ids,
        ),
    )


def _make_err(
    risk_type="weak_or_uncertain_evidence",
    supporting_evidence=None,
    counter_evidence=None,
):
    if supporting_evidence is None:
        supporting_evidence = ["neighbor_1"]
    if counter_evidence is None:
        counter_evidence = ["counter_1"]

    return ERR(
        node_id=0,
        risk_type=risk_type,
        supporting_evidence=supporting_evidence,
        counter_evidence=counter_evidence,
        summary="test",
    )


@pytest.fixture
def verifier():
    contracts = load_contracts()
    return EvidenceContractVerifier(contracts, enable_label_compatibility=False)


@pytest.fixture
def verifier_with_label():
    contracts = load_contracts()
    return EvidenceContractVerifier(contracts, enable_label_compatibility=True)


def test_valid_err_accepted(verifier):
    card = _make_card()
    err = _make_err()
    accepted, reasons = verifier.verify(err, card)
    assert accepted
    assert reasons == []


def test_unavailable_evidence_rejected(verifier):
    card = _make_card()
    err = _make_err(supporting_evidence=["nonexistent_field"])
    accepted, reasons = verifier.verify(err, card)
    assert not accepted
    assert "unavailable_evidence" in reasons


def test_counter_signal_as_support_rejected(verifier):
    card = _make_card(allowed_support_ids=["neighbor_1"])
    err = _make_err(supporting_evidence=["counter_1"])
    accepted, reasons = verifier.verify(err, card)
    assert not accepted
    assert "invalid_support_role" in reasons


def test_support_as_counter_rejected(verifier):
    card = _make_card(allowed_counter_ids=["counter_1"])
    err = _make_err(counter_evidence=["neighbor_1"])
    accepted, reasons = verifier.verify(err, card)
    assert not accepted
    assert "invalid_counter_role" in reasons


def test_overlap_rejected(verifier):
    card = _make_card()
    err = _make_err(
        supporting_evidence=["neighbor_1"],
        counter_evidence=["neighbor_1"],
    )
    accepted, reasons = verifier.verify(err, card)
    assert not accepted
    assert "support_counter_overlap" in reasons


def test_score_leakage_rejected(verifier):
    card = _make_card(allowed_support_ids=["base_score"])
    err = _make_err(supporting_evidence=["base_score"])
    accepted, reasons = verifier.verify(err, card)
    assert not accepted
    assert "score_leakage" in reasons


def test_contract_required_not_satisfied(verifier):
    card = _make_card(degree_level="low", neighbor_consistency="high", detector_signal="normal")
    err = _make_err(risk_type="structural_discrepancy", supporting_evidence=["neighbor_1"])
    accepted, reasons = verifier.verify(err, card)
    assert not accepted
    assert "contract_required_not_satisfied" in reasons


def test_contract_forbidden_hit(verifier):
    card = _make_card(neighbor_consistency="high")
    err = _make_err(risk_type="camouflage_neighbor", supporting_evidence=["neighbor_1"])
    accepted, reasons = verifier.verify(err, card)
    assert not accepted
    assert "contract_forbidden_hit" in reasons


def test_weak_or_uncertain_accepted(verifier):
    card = _make_card()
    err = _make_err(risk_type="weak_or_uncertain_evidence")
    accepted, reasons = verifier.verify(err, card)
    assert accepted
    assert reasons == []


def test_label_compatibility_disabled(verifier):
    card = _make_card(detector_signal="embedding_neighbor_discrepancy_high", detector_signal_strength="strong")
    err = _make_err(risk_type="spectral_anomaly", supporting_evidence=["detector_signal", "neighbor_1"])
    accepted, reasons = verifier.verify(err, card, label=0)
    assert accepted


def test_label_compatibility_enabled_benign_rejects_fraud_type(verifier_with_label):
    card = _make_card(detector_signal="embedding_neighbor_discrepancy_high", detector_signal_strength="strong")
    err = _make_err(risk_type="spectral_anomaly", supporting_evidence=["detector_signal", "neighbor_1"])
    accepted, reasons = verifier_with_label.verify(err, card, label=0)
    assert not accepted
    assert "label_incompatible" in reasons


def test_label_compatibility_enabled_fraud_allows_fraud_type(verifier_with_label):
    card = _make_card(detector_signal="embedding_neighbor_discrepancy_high", detector_signal_strength="strong")
    err = _make_err(risk_type="spectral_anomaly", supporting_evidence=["detector_signal", "neighbor_1"])
    accepted, reasons = verifier_with_label.verify(err, card, label=1)
    assert accepted


def test_structural_discrepancy_accepted(verifier):
    card = _make_card(degree_level="high")
    err = _make_err(
        risk_type="structural_discrepancy",
        supporting_evidence=["degree_level", "neighbor_1"],
    )
    accepted, reasons = verifier.verify(err, card)
    assert accepted


def test_feature_structure_conflict_accepted(verifier):
    card = _make_card(feature_neighbor_discrepancy="high")
    err = _make_err(
        risk_type="feature_structure_conflict",
        supporting_evidence=["feature_neighbor_discrepancy", "neighbor_1"],
    )
    accepted, reasons = verifier.verify(err, card)
    assert accepted
