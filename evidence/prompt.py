from __future__ import annotations

from typing import Any

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


def build_teacher_payload(card) -> dict[str, Any]:
    return card.to_teacher_payload()


def assert_score_blind_payload(payload: dict[str, Any]) -> None:
    _check_recursive(payload, path="root")


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
