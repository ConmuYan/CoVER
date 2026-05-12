import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.json_utils import parse_llm_err


def test_pure_json():
    raw = '{"risk_type": "spectral_anomaly", "supporting_evidence": ["detector_signal"], "counter_evidence": ["counter_signal"], "summary": "test"}'
    err = parse_llm_err(raw, node_id=0)
    assert err.risk_type == "spectral_anomaly"
    assert err.supporting_evidence == ["detector_signal"]


def test_fenced_json():
    raw = '''```json
{"risk_type": "camouflage_neighbor", "supporting_evidence": ["neighbor_consistency"], "counter_evidence": [], "summary": "test"}
```'''
    err = parse_llm_err(raw, node_id=1)
    assert err.risk_type == "camouflage_neighbor"


def test_embedded_json():
    raw = 'Here is the result: {"risk_type": "weak_or_uncertain_evidence", "supporting_evidence": [], "counter_evidence": [], "summary": "uncertain"}'
    err = parse_llm_err(raw, node_id=2)
    assert err.risk_type == "weak_or_uncertain_evidence"


def test_think_block():
    raw = '<think>Let me analyze...</think>\n{"risk_type": "structural_discrepancy", "supporting_evidence": ["degree_level"], "counter_evidence": [], "summary": "high degree"}'
    err = parse_llm_err(raw, node_id=3)
    assert err.risk_type == "structural_discrepancy"


def test_missing_risk_type():
    raw = '{"supporting_evidence": [], "counter_evidence": []}'
    with pytest.raises(ValueError, match="Missing required field"):
        parse_llm_err(raw, node_id=0)


def test_invalid_supporting_type():
    raw = '{"risk_type": "test", "supporting_evidence": "not_a_list", "counter_evidence": []}'
    with pytest.raises(ValueError, match="supporting_evidence must be list"):
        parse_llm_err(raw, node_id=0)


def test_invalid_counter_type():
    raw = '{"risk_type": "test", "supporting_evidence": [], "counter_evidence": 123}'
    with pytest.raises(ValueError, match="counter_evidence must be list"):
        parse_llm_err(raw, node_id=0)
