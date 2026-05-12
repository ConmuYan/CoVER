from __future__ import annotations

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

    assert_score_blind_payload_str(retry_content)

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
