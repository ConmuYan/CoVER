from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evidence.evidence_packet import FORBIDDEN_PACKET_KEYS, judge_available_fields


VERDICTS = {"fake", "real", "uncertain"}
STRENGTHS = {"weak", "moderate", "strong"}
REQUIRED_FIELDS = (
    "node_id",
    "verdict",
    "evidence_strength",
    "key_relation",
    "supporting_evidence",
    "counter_evidence",
    "uncertainty_factors",
    "short_explanation",
)
FORBIDDEN_TEXT_TERMS = (
    "base_score",
    "base score",
    "base_prob",
    "base probability",
    "base_logit",
    "base logit",
    "confidence",
    "base prediction",
    "target label",
    "ground truth",
    "test label",
    "train label",
    "validation label",
    "fn",
    "fp",
    "base_error",
    "base error",
)


@dataclass
class JudgeVerificationResult:
    accepted: bool
    reasons: list[str]
    record: dict[str, Any] | None = None


def verify_judge_output(packet: dict[str, Any], output: dict[str, Any]) -> JudgeVerificationResult:
    reasons: list[str] = []
    if not isinstance(output, dict):
        return JudgeVerificationResult(False, ["output_not_object"])

    for field in REQUIRED_FIELDS:
        if field not in output:
            reasons.append(f"missing_{field}")

    if output.get("node_id") != packet.get("node_id"):
        reasons.append("node_id_mismatch")
    if output.get("verdict") not in VERDICTS:
        reasons.append("invalid_verdict")
    if output.get("evidence_strength") not in STRENGTHS:
        reasons.append("invalid_evidence_strength")

    relation_schema = {str(name).upper() for name in packet.get("relation_schema", [])}
    if str(output.get("key_relation", "")).upper() not in relation_schema:
        reasons.append("invalid_key_relation")

    available = set(judge_available_fields(packet))
    for field_name in ("supporting_evidence", "counter_evidence", "uncertainty_factors"):
        values = output.get(field_name)
        if not isinstance(values, list):
            reasons.append(f"{field_name}_not_list")
            continue
        for value in values:
            if not isinstance(value, str) or value not in available:
                reasons.append(f"unavailable_{field_name}:{value}")

    try:
        _assert_no_forbidden(output)
    except ValueError as exc:
        reasons.append(str(exc))

    explanation = output.get("short_explanation", "")
    if not isinstance(explanation, str):
        reasons.append("short_explanation_not_string")
    elif len(explanation.split()) > 80:
        reasons.append("short_explanation_too_long")
    elif _contains_forbidden_text(explanation):
        reasons.append("short_explanation_forbidden_terms")

    if reasons:
        return JudgeVerificationResult(False, sorted(dict.fromkeys(reasons)))

    record = {
        "node_id": int(output["node_id"]),
        "verdict": str(output["verdict"]),
        "evidence_strength": str(output["evidence_strength"]),
        "key_relation": str(output["key_relation"]).upper(),
        "supporting_evidence": [str(v) for v in output.get("supporting_evidence", [])],
        "counter_evidence": [str(v) for v in output.get("counter_evidence", [])],
        "uncertainty_factors": [str(v) for v in output.get("uncertainty_factors", [])],
        "short_explanation": str(output.get("short_explanation", "")),
    }
    return JudgeVerificationResult(True, [], record)


def audit_forbidden_fields(packet: dict[str, Any], prompt: str, output: dict[str, Any] | None = None) -> dict[str, Any]:
    checks = {
        "packet_forbidden_field_free": True,
        "prompt_forbidden_field_free": True,
        "output_forbidden_field_free": True,
        "violations": [],
    }
    for name, obj in (("packet", packet), ("output", output or {})):
        try:
            _assert_no_forbidden(obj)
        except ValueError as exc:
            checks[f"{name}_forbidden_field_free"] = False
            checks["violations"].append(str(exc))
    if _contains_forbidden_text(prompt):
        checks["prompt_forbidden_field_free"] = False
        checks["violations"].append("prompt_forbidden_terms")
    checks["passed"] = not checks["violations"]
    return checks


def _assert_no_forbidden(obj: Any, path: str = "root") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            if key_lower in FORBIDDEN_PACKET_KEYS:
                raise ValueError(f"forbidden_field:{path}.{key}")
            _assert_no_forbidden(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            _assert_no_forbidden(value, f"{path}[{idx}]")
    elif isinstance(obj, str) and _contains_forbidden_text(obj):
        raise ValueError(f"forbidden_text:{path}")


def _contains_forbidden_text(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in FORBIDDEN_TEXT_TERMS)


__all__ = [
    "JudgeVerificationResult",
    "audit_forbidden_fields",
    "verify_judge_output",
]
