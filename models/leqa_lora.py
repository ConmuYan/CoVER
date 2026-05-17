"""LEQA (LLM-as-Evidence-Quality-Auditor) model.

Wraps a LoRA-Qwen3 to produce per-token quality scores via structured
JSON generation.  The model emits::

    {"token_quality": [
        {"token": "<name>", "q": <float>, "reason": "noisy|contradictory|insufficient|none"},
        ...
    ]}

A parser converts the JSON into:
  - ``q_tensor``  : ``(N_packets, T)``  quality scores in [0, 1]
  - ``reason_tensor``: ``(N_packets, T)``  reason label indices

Token vocabulary comes from ``evidence/vocab.py`` constants and
``evidence/relation_features.py`` stat names.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import torch
from torch import Tensor

from evidence.vocab import EVIDENCE_SLOTS
from evidence.relation_features import RELATION_STAT_NAMES

logger = logging.getLogger(__name__)

# ---- LEQA token vocabulary (T=37 = 10 card fields + 3 relations * 9 stats) ----

LEQA_CARD_FIELDS: list[str] = list(EVIDENCE_SLOTS[:10])

LEQA_REASON_LABELS = ["none", "noisy", "contradictory", "insufficient"]
LEQA_REASON_TO_ID = {r: i for i, r in enumerate(LEQA_REASON_LABELS)}


def build_leqa_token_names(relation_names: list[str]) -> list[str]:
    """Build canonical LEQA token name list.

    For YelpChi (RUR, RSR, RTR): 10 + 3*9 = 37 tokens.
    For Amazon (UPU, USU, UVU): 10 + 3*9 = 37 tokens.
    """
    tokens = list(LEQA_CARD_FIELDS)
    for rel in relation_names:
        for stat in RELATION_STAT_NAMES:
            tokens.append(f"{rel.upper()}_{stat}")
    return tokens


def get_default_token_names() -> list[str]:
    """Default token names for YelpChi (RUR, RSR, RTR)."""
    return build_leqa_token_names(["RUR", "RSR", "RTR"])


# ---- Prompt template (score-blind, per FINAL_PROPOSAL.md section 3.3) ----

LEQA_SYSTEM_PROMPT = (
    "You are an evidence quality auditor. For each evidence token in the "
    "packet, emit a quality score in [0, 1] and an optional reason in "
    "{noisy, contradictory, insufficient}. You may NEVER see, infer, or "
    "reference: base_score, base_prob, base_logit, confidence, label, "
    "split identity, ground truth, FN/FP, final prediction. If you "
    "reference any of these, your response will be rejected."
)

LEQA_USER_TEMPLATE = "EVIDENCE PACKET:\n{packet_json}\n\nRETURN:\n"

LEQA_OUTPUT_SCHEMA_HINT = (
    '{{"token_quality": [{{"token": "<name>", "q": <float 0-1>, '
    '"reason": "noisy"|"contradictory"|"insufficient"|"none"}}, ...]}}'
)


def format_leqa_prompt(packet_json: str) -> str:
    """Format a full prompt for the LEQA model."""
    return (
        f"<|im_start|>system\n{LEQA_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{LEQA_USER_TEMPLATE.format(packet_json=packet_json)}"
        f"{LEQA_OUTPUT_SCHEMA_HINT}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---- JSON output parser ----

def parse_leqa_json(
    raw_text: str,
    token_names: list[str],
    default_q: float = 1.0,
    default_reason: str = "none",
) -> tuple[list[float], list[int]]:
    """Parse LEQA JSON output into per-token quality + reason lists.

    Returns:
        (q_list, reason_list) each of length len(token_names).
        Missing tokens get default_q=1.0 and default_reason="none".
    """
    T = len(token_names)
    q_list = [default_q] * T
    reason_list = [LEQA_REASON_TO_ID[default_reason]] * T

    token_to_idx = {name: i for i, name in enumerate(token_names)}

    # Try to extract JSON from the text
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match is None:
        return q_list, reason_list

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return q_list, reason_list

    entries = parsed.get("token_quality", [])
    if not isinstance(entries, list):
        return q_list, reason_list

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        token_name = str(entry.get("token", ""))
        if token_name not in token_to_idx:
            continue
        idx = token_to_idx[token_name]
        q_val = entry.get("q", default_q)
        try:
            q_val = float(q_val)
            q_val = max(0.0, min(1.0, q_val))
        except (ValueError, TypeError):
            q_val = default_q
        q_list[idx] = q_val

        reason = str(entry.get("reason", default_reason)).lower().strip()
        reason_list[idx] = LEQA_REASON_TO_ID.get(reason, 0)

    return q_list, reason_list


def parse_leqa_batch(
    raw_texts: list[str],
    token_names: list[str],
    default_q: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Parse a batch of LEQA JSON outputs into tensors.

    Returns:
        q_tensor:      (N, T)  float in [0, 1]
        reason_tensor: (N, T)  long reason indices
    """
    N = len(raw_texts)
    T = len(token_names)
    q_tensor = torch.full((N, T), default_q, dtype=torch.float32)
    reason_tensor = torch.zeros((N, T), dtype=torch.long)

    for i, text in enumerate(raw_texts):
        q_list, reason_list = parse_leqa_json(text, token_names, default_q)
        q_tensor[i] = torch.tensor(q_list, dtype=torch.float32)
        reason_tensor[i] = torch.tensor(reason_list, dtype=torch.long)

    return q_tensor, reason_tensor


# ---- Training data formatting ----

def build_training_example(
    packet_json: str,
    token_names: list[str],
    q_labels: list[float],
    reason_labels: list[str],
) -> str:
    """Build a single training example (prompt + expected output).

    Returns the full text for causal LM training.
    """
    entries = []
    for name, q, reason in zip(token_names, q_labels, reason_labels):
        entries.append({
            "token": name,
            "q": round(q, 3),
            "reason": reason,
        })
    output_json = json.dumps({"token_quality": entries}, indent=None)

    return (
        f"<|im_start|>system\n{LEQA_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{LEQA_USER_TEMPLATE.format(packet_json=packet_json)}"
        f"{LEQA_OUTPUT_SCHEMA_HINT}<|im_end|>\n"
        f"<|im_start|>assistant\n{output_json}<|im_end|>\n"
    )


class LEQAModel:
    """High-level wrapper for LEQA inference.

    Wraps the LoRA-Qwen3 model and provides methods for:
    - Single/batch inference
    - JSON parsing to quality tensors
    - Forbidden-field audit on outputs

    This class supports ``use_judge=False`` in ``CoVERRelReasoner``:
    gradient flow to LoRA happens later in Block 2 joint training.
    In Commit 2, the LoRA is warmed up offline.
    """

    def __init__(
        self,
        token_names: list[str] | None = None,
        default_q: float = 1.0,
    ):
        self.token_names = token_names or get_default_token_names()
        self.T = len(self.token_names)
        self.default_q = default_q
        self._token_to_idx = {n: i for i, n in enumerate(self.token_names)}

    @property
    def num_tokens(self) -> int:
        return self.T

    def parse_outputs(
        self,
        raw_texts: list[str],
    ) -> tuple[Tensor, Tensor]:
        """Parse raw LLM text outputs into quality tensors.

        Returns:
            q_tensor:      (N, T)  float in [0, 1]
            reason_tensor: (N, T)  long reason label indices
        """
        return parse_leqa_batch(raw_texts, self.token_names, self.default_q)

    def uniform_quality(self, n: int) -> Tensor:
        """Return uniform q=1 tensor for E0 (no LEQA) baseline."""
        return torch.ones(n, self.T, dtype=torch.float32)

    def random_quality(self, n: int) -> Tensor:
        """Return random q ~ Uniform[0,1] tensor for E4' falsification."""
        return torch.rand(n, self.T, dtype=torch.float32)


# ---- Forbidden-field audit on LEQA outputs ----

FORBIDDEN_OUTPUT_FIELDS = frozenset({
    "base_score", "base_prob", "base_logit", "confidence",
    "label", "split", "FN", "FP", "ground_truth", "final_prediction",
    "base_probability", "base_prediction", "target_label",
    "train_label", "val_label", "test_label", "split_name",
    "split_identity",
})


def audit_leqa_output(raw_text: str) -> bool:
    """Return True if the LEQA output text contains NO forbidden fields."""
    lower = raw_text.lower()
    for field in FORBIDDEN_OUTPUT_FIELDS:
        if field.lower() in lower:
            return False
    return True


__all__ = [
    "LEQA_CARD_FIELDS",
    "LEQA_REASON_LABELS",
    "LEQA_REASON_TO_ID",
    "LEQAModel",
    "build_leqa_token_names",
    "get_default_token_names",
    "format_leqa_prompt",
    "parse_leqa_json",
    "parse_leqa_batch",
    "build_training_example",
    "audit_leqa_output",
    "FORBIDDEN_OUTPUT_FIELDS",
]
