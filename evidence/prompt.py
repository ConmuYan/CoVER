from __future__ import annotations

import json
from typing import Any

from evidence.schema import ERR

SCORE_LEAKAGE_KEYS = frozenset({
    "base_score",
    "score",
    "logit",
    "logits",
    "prob",
    "probs",
    "probability",
    "probabilities",
    "confidence",
    "prediction",
    "pred",
    "label",
    "y",
    "target",
})

# Lazy imports – populated by vocab after worker-1 adds the constants
_DIRECTION_VALUES = None
_STRENGTH_VALUES = None


def _get_direction_values() -> list[str]:
    global _DIRECTION_VALUES
    if _DIRECTION_VALUES is None:
        from evidence.vocab import EVIDENCE_DIRECTIONS
        _DIRECTION_VALUES = EVIDENCE_DIRECTIONS
    return _DIRECTION_VALUES


def _get_strength_values() -> list[str]:
    global _STRENGTH_VALUES
    if _STRENGTH_VALUES is None:
        from evidence.vocab import EVIDENCE_STRENGTHS
        _STRENGTH_VALUES = EVIDENCE_STRENGTHS
    return _STRENGTH_VALUES

VALID_RISK_TYPES = [
    "structural_discrepancy",
    "camouflage_neighbor",
    "spectral_anomaly",
    "feature_structure_conflict",
    "relation_or_burst_anomaly",
    "weak_or_uncertain_evidence",
]

SYSTEM_PROMPT = (
    "You are a strict evidence reasoning engine for graph fraud detection.\n"
    "You must output exactly one valid JSON object.\n"
    "Do not output markdown.\n"
    "Do not explain your reasoning outside JSON.\n"
    "Do not invent evidence fields.\n"
    "Do not mention scores, probabilities, logits, confidence, or model predictions."
)


def build_teacher_payload(card) -> dict[str, Any]:
    return card.to_teacher_payload()


def assert_score_blind_payload(payload: dict[str, Any]) -> None:
    _check_recursive(payload, path="root")


def assert_score_blind_payload_str(text: str) -> None:
    import re
    text_lower = text.lower()
    for key in SCORE_LEAKAGE_KEYS:
        if len(key) <= 2:
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, text_lower):
                raise ValueError(f"Score leakage detected in payload string: '{key}'")
        else:
            if key in text_lower:
                raise ValueError(f"Score leakage detected in payload string: '{key}'")


def build_llm_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    from evidence.contract_hints import get_all_contract_hints

    assert_score_blind_payload(payload)

    reasoning = payload.get("reasoning", {})
    allowed_support = reasoning.get("allowed_support_ids", [])
    allowed_counter = reasoning.get("allowed_counter_ids", [])

    payload_json = _safe_payload_str(payload)
    assert_score_blind_payload_str(payload_json)

    contract_hints = get_all_contract_hints()

    user_content = (
        f"Given the following score-blind structural evidence payload, generate an ERR.\n"
        f"\nRules:\n"
        f"- supporting_evidence can only contain: {allowed_support}\n"
        f"- counter_evidence can only contain: {allowed_counter}\n"
        f"- You must only cite fields that appear in allowed_support_ids or allowed_counter_ids.\n"
        f"- Do not invent evidence fields.\n"
        f"- Do not mention score, probability, confidence, logit, base_score, or model prediction.\n"
        f"- If you cannot satisfy the required evidence for a risk_type, choose weak_or_uncertain_evidence.\n"
        f"- Output exactly one JSON object and nothing else.\n"
        f"- Summary must be one sentence without scores.\n"
        f"\nrisk_type must be one of: {', '.join(VALID_RISK_TYPES)}\n"
        f"\nRisk-type-specific contract rules:\n{contract_hints}\n"
        f"\nPayload:\n{payload_json}\n"
        f"\nOutput JSON schema:\n"
        f'{{\n'
        f'  "risk_type": "...",\n'
        f'  "supporting_evidence": ["..."],\n'
        f'  "counter_evidence": ["..."],\n'
        f'  "summary": "one short sentence"\n'
        f'}}\n'
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_retry_messages(
    payload: dict[str, Any],
    previous_err: ERR | None,
    reject_reasons: list[str],
) -> list[dict[str, str]]:
    from evidence.contract_hints import get_retry_hint

    assert_score_blind_payload(payload)

    original_messages = build_llm_messages(payload)
    retry_hint = get_retry_hint(reject_reasons, previous_err, payload)

    previous_err_str = ""
    if previous_err is not None:
        import json
        previous_err_str = json.dumps({
            "risk_type": previous_err.risk_type,
            "supporting_evidence": previous_err.supporting_evidence,
            "counter_evidence": previous_err.counter_evidence,
        }, indent=2)

    retry_content = (
        f"Your previous output was rejected by the verifier.\n"
        f"\nReject reasons: {reject_reasons}\n"
        f"\nPrevious ERR:\n{previous_err_str}\n"
        f"\nContract hints:\n{retry_hint}\n"
        f"\nPlease generate a new ERR that satisfies the contract rules.\n"
        f"Output exactly one JSON object and nothing else."
    )

    if previous_err_str:
        assert_score_blind_payload_str(previous_err_str)

    return original_messages + [{"role": "user", "content": retry_content}]


def build_contrastive_directional_messages(
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    from evidence.vocab import EVIDENCE_DIRECTIONS, EVIDENCE_STRENGTHS

    assert_score_blind_payload(payload)

    reasoning = payload.get("reasoning", {})
    allowed_support = reasoning.get("allowed_support_ids", [])
    allowed_counter = reasoning.get("allowed_counter_ids", [])

    fraud_proto = payload.get("fraud_prototype", {})
    benign_proto = payload.get("benign_prototype", {})

    categorical_fields = {
        k: v for k, v in reasoning.items()
        if k not in ("allowed_support_ids", "allowed_counter_ids")
        and isinstance(v, str)
    }

    target_desc = json.dumps(categorical_fields, ensure_ascii=False, indent=2)
    fraud_desc = json.dumps(fraud_proto, ensure_ascii=False, indent=2)
    benign_desc = json.dumps(benign_proto, ensure_ascii=False, indent=2)

    system_content = (
        "You are a strict evidence reasoning engine for graph fraud detection.\n"
        "You perform contrastive evidence comparison between a target node and structural prototypes.\n"
        "You must output exactly one valid JSON object.\n"
        "Do not output markdown.\n"
        "Do not explain your reasoning outside JSON.\n"
        "Do not invent evidence fields.\n"
        "Do not mention scores, probabilities, logits, confidence, or model predictions."
    )

    user_content = (
        "You are not given any base model score, probability, logit, confidence, or prediction.\n\n"
        "=== Target Node Reasoning Fields (categorical only) ===\n"
        f"{target_desc}\n\n"
        "=== Fraud Structural Prototype ===\n"
        f"{fraud_desc}\n\n"
        "=== Benign Structural Prototype ===\n"
        f"{benign_desc}\n\n"
        f"Allowed supporting_evidence fields: {allowed_support}\n"
        f"Allowed counter_evidence fields: {allowed_counter}\n\n"
        f"Valid risk_type values: {', '.join(VALID_RISK_TYPES)}\n"
        f"Valid evidence_direction values: {', '.join(EVIDENCE_DIRECTIONS)}\n"
        f"Valid evidence_strength values: {', '.join(EVIDENCE_STRENGTHS)}\n\n"
        "=== evidence_direction Definitions ===\n"
        "- increase_risk: The score-blind structural evidence is fraud-like. "
        "Fraud-like evidence dominates benign-like counter-evidence.\n"
        "- decrease_risk: The score-blind structural evidence is benign-like, "
        "normal-structure-like, or contradicts fraud. Benign-like counter-evidence "
        "dominates fraud-like anomaly evidence.\n"
        "- uncertain: Fraud-like and benign-like evidence are mixed, weak, or insufficient.\n\n"
        "=== Decision Rubric ===\n"
        "1. Do not choose increase_risk only because one anomaly token exists.\n"
        "2. Choose increase_risk only when fraud-like evidence clearly dominates.\n"
        "3. Choose decrease_risk when benign-like / normal-structure / counter evidence dominates, "
        "even if minor anomaly evidence exists.\n"
        "4. Choose uncertain when both sides are mixed.\n\n"
        "=== Synthetic Examples ===\n"
        "Example 1 (fraud-like → increase_risk):\n"
        '  "supporting_evidence": ["degree_level", "detector_signal", "feature_neighbor_discrepancy"],\n'
        '  "counter_evidence": [],\n'
        '  "evidence_direction": "increase_risk",\n'
        '  "reasoning": "High degree, strong detector signal, and feature-neighbor discrepancy all align with fraud prototype. No benign-like counter-evidence present."\n\n'
        "Example 2 (benign-like → decrease_risk):\n"
        '  "supporting_evidence": [],\n'
        '  "counter_evidence": ["degree_level", "neighbor_consistency"],\n'
        '  "evidence_direction": "decrease_risk",\n'
        '  "reasoning": "Degree level and neighbor consistency match benign prototype. No fraud-like anomaly evidence present."\n\n'
        "Example 3 (mixed → uncertain):\n"
        '  "supporting_evidence": ["detector_signal"],\n'
        '  "counter_evidence": ["neighbor_consistency"],\n'
        '  "evidence_direction": "uncertain",\n'
        '  "reasoning": "Detector signal suggests fraud, but neighbor consistency suggests benign. Signals are mixed and neither side dominates."\n\n'
        "=== Rules ===\n"
        "- Compare the target node against both prototypes to determine direction.\n"
        "- supporting_evidence can only cite fields from allowed_support_ids.\n"
        "- counter_evidence can only cite fields from allowed_counter_ids.\n"
        "- Each uncertainty_factor must reference an available reasoning field.\n"
        "- Do not mention score, probability, confidence, logit, base_score, or model prediction.\n"
        "- Output exactly one JSON object and nothing else.\n"
        "- summary must be one sentence without scores.\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "risk_type": "...",\n'
        '  "evidence_direction": "increase_risk|decrease_risk|uncertain",\n'
        '  "evidence_strength": "weak|moderate|strong",\n'
        '  "supporting_evidence": ["..."],\n'
        '  "counter_evidence": ["..."],\n'
        '  "uncertainty_factors": ["..."],\n'
        '  "summary": "one short sentence"\n'
        "}"
    )

    assert_score_blind_payload_str(target_desc)
    assert_score_blind_payload_str(fraud_desc)
    assert_score_blind_payload_str(benign_desc)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_contrastive_directional_retry_messages(
    payload: dict[str, Any],
    previous_err: ERR | None,
    reject_reasons: list[str],
) -> list[dict[str, str]]:
    assert_score_blind_payload(payload)

    original_messages = build_contrastive_directional_messages(payload)

    previous_err_str = ""
    if previous_err is not None:
        prev_dict: dict[str, Any] = {
            "risk_type": previous_err.risk_type,
            "supporting_evidence": previous_err.supporting_evidence,
            "counter_evidence": previous_err.counter_evidence,
            "summary": previous_err.summary,
        }
        if hasattr(previous_err, "evidence_direction"):
            prev_dict["evidence_direction"] = getattr(previous_err, "evidence_direction", "uncertain")
        if hasattr(previous_err, "evidence_strength"):
            prev_dict["evidence_strength"] = getattr(previous_err, "evidence_strength", "weak")
        if hasattr(previous_err, "uncertainty_factors"):
            prev_dict["uncertainty_factors"] = getattr(previous_err, "uncertainty_factors", [])
        previous_err_str = json.dumps(prev_dict, indent=2)

    direction_hint = ""
    if "invalid_direction" in reject_reasons:
        direction_hint = "\nevidence_direction must be one of: increase_risk, decrease_risk, uncertain"
    if "invalid_strength" in reject_reasons:
        direction_hint += "\nevidence_strength must be one of: weak, moderate, strong"
    if "direction_consistency" in reject_reasons:
        direction_hint += "\nCheck that your evidence_direction is consistent with your evidence lists."

    retry_content = (
        f"Your previous output was rejected by the verifier.\n"
        f"\nReject reasons: {reject_reasons}\n"
        f"\nPrevious ERR:\n{previous_err_str}\n"
        f"{direction_hint}\n"
        f"\nPlease generate a new ERR that satisfies all rules.\n"
        f"Output exactly one JSON object and nothing else."
    )

    if previous_err_str:
        assert_score_blind_payload_str(previous_err_str)

    return original_messages + [{"role": "user", "content": retry_content}]


def _safe_payload_str(payload: dict[str, Any]) -> str:
    import json
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _check_recursive(obj: Any, path: str) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in SCORE_LEAKAGE_KEYS:
                raise ValueError(
                    f"Score leakage detected at {path}.{key}: "
                    f"'{key}' is a forbidden field in teacher payload"
                )
            _check_recursive(value, path=f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _check_recursive(item, path=f"{path}[{i}]")
