from __future__ import annotations

import sys
from pathlib import Path

from tests.test_llm_judge_packet import _packet

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.judge_verifier import verify_judge_output


def _valid_output():
    return {
        "node_id": 7,
        "verdict": "fake",
        "evidence_strength": "moderate",
        "key_relation": "RUR",
        "supporting_evidence": ["relation_evidence.RUR.degree_bucket"],
        "counter_evidence": [],
        "uncertainty_factors": [],
        "short_explanation": "RUR relation evidence is moderately fake-like.",
    }


def test_judge_verifier_accepts_grounded_output():
    result = verify_judge_output(_packet(), _valid_output())
    assert result.accepted
    assert result.record is not None
    assert result.record["short_explanation"]


def test_judge_verifier_rejects_unavailable_evidence_field():
    output = _valid_output()
    output["supporting_evidence"] = ["relation_evidence.RUR.missing_field"]
    result = verify_judge_output(_packet(), output)
    assert not result.accepted
    assert any("unavailable_supporting_evidence" in reason for reason in result.reasons)


def test_judge_verifier_rejects_invalid_key_relation():
    output = _valid_output()
    output["key_relation"] = "UPU"
    result = verify_judge_output(_packet(), output)
    assert not result.accepted
    assert "invalid_key_relation" in result.reasons


def test_judge_verifier_rejects_forbidden_explanation_terms():
    output = _valid_output()
    output["short_explanation"] = "The base score says this is fake."
    result = verify_judge_output(_packet(), output)
    assert not result.accepted
    assert "short_explanation_forbidden_terms" in result.reasons or any(
        "forbidden_text" in reason for reason in result.reasons
    )
