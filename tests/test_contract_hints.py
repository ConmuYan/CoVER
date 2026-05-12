import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.contract_hints import (
    get_contract_hint_for_reason_type,
    get_all_contract_hints,
    get_retry_hint,
    CONTRACT_HINTS,
)
from evidence.schema import ERR


def test_all_contract_hints_contains_all_types():
    hints = get_all_contract_hints()
    for risk_type in CONTRACT_HINTS:
        assert risk_type in hints or risk_type.replace("_", " ") in hints


def test_spectral_anomaly_hint_contains_detector_signal():
    hint = get_contract_hint_for_reason_type("spectral_anomaly")
    assert "detector_signal" in hint


def test_feature_structure_conflict_hint_contains_field():
    hint = get_contract_hint_for_reason_type("feature_structure_conflict")
    assert "feature_neighbor_discrepancy" in hint


def test_camouflage_neighbor_hint_contains_consistency():
    hint = get_contract_hint_for_reason_type("camouflage_neighbor")
    assert "neighbor_consistency" in hint


def test_weak_uncertain_hint():
    hint = get_contract_hint_for_reason_type("weak_or_uncertain_evidence")
    assert "weak" in hint.lower() or "insufficient" in hint.lower()


def test_hints_no_score_leakage():
    all_hints = get_all_contract_hints()
    forbidden = ["base_score", "logit", "probability", "confidence"]
    for word in forbidden:
        assert word not in all_hints.lower()


def test_retry_hint_contract_not_satisfied():
    err = ERR(
        node_id=0,
        risk_type="spectral_anomaly",
        supporting_evidence=["degree_level"],
        counter_evidence=[],
        summary="test",
    )
    payload = {"reasoning": {"allowed_support_ids": ["detector_signal"], "allowed_counter_ids": []}}
    hint = get_retry_hint(["contract_required_not_satisfied"], err, payload)
    assert "spectral_anomaly" in hint
    assert "detector_signal" in hint


def test_retry_hint_unavailable_evidence():
    hint = get_retry_hint(["unavailable_evidence"], None, {"reasoning": {"allowed_support_ids": ["a"], "allowed_counter_ids": ["b"]}})
    assert "allowed_support_ids" in hint
