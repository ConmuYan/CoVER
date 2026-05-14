from __future__ import annotations

import json
import re

from evidence.schema import ERR


def parse_llm_err(raw_output: str, node_id: int) -> ERR:
    cleaned = _remove_think_blocks(raw_output)
    json_obj = _extract_json(cleaned)

    if not isinstance(json_obj, dict):
        raise ValueError(f"Expected JSON object, got {type(json_obj)}")

    if "risk_type" not in json_obj:
        raise ValueError("Missing required field: risk_type")

    risk_type = json_obj["risk_type"]
    if not isinstance(risk_type, str):
        raise ValueError(f"risk_type must be str, got {type(risk_type)}")

    supporting = json_obj.get("supporting_evidence", [])
    if not isinstance(supporting, list):
        raise ValueError(f"supporting_evidence must be list, got {type(supporting)}")
    if not all(isinstance(s, str) for s in supporting):
        raise ValueError("supporting_evidence must be list[str]")

    counter = json_obj.get("counter_evidence", [])
    if not isinstance(counter, list):
        raise ValueError(f"counter_evidence must be list, got {type(counter)}")
    if not all(isinstance(s, str) for s in counter):
        raise ValueError("counter_evidence must be list[str]")

    summary = json_obj.get("summary", "")

    evidence_direction = json_obj.get("evidence_direction", "uncertain")
    evidence_strength = json_obj.get("evidence_strength", "weak")
    uncertainty_raw = json_obj.get("uncertainty_factors", [])
    uncertainty_factors = uncertainty_raw if isinstance(uncertainty_raw, list) else []

    err_kwargs: dict = dict(
        node_id=node_id,
        risk_type=risk_type,
        supporting_evidence=supporting,
        counter_evidence=counter,
        summary=str(summary),
    )

    import dataclasses
    if dataclasses.is_dataclass(ERR) and hasattr(ERR, "evidence_direction"):
        err_kwargs["evidence_direction"] = evidence_direction
    if dataclasses.is_dataclass(ERR) and hasattr(ERR, "evidence_strength"):
        err_kwargs["evidence_strength"] = evidence_strength
    if dataclasses.is_dataclass(ERR) and hasattr(ERR, "uncertainty_factors"):
        err_kwargs["uncertainty_factors"] = [
            f for f in uncertainty_factors if isinstance(f, str)
        ]

    return ERR(**err_kwargs)


def _remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1).strip())

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    return json.loads(text)
