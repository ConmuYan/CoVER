from __future__ import annotations

import torch

from evidence.schema import ERR, ReasoningChannel

EVIDENCE_SLOTS = [
    "degree_level",
    "neighbor_consistency",
    "feature_neighbor_discrepancy",
    "detector_signal",
    "detector_signal_strength",
    "counter_signal",
]

REASON_TYPES = [
    "structural_discrepancy",
    "camouflage_neighbor",
    "spectral_anomaly",
    "feature_structure_conflict",
    "relation_or_burst_anomaly",
    "weak_or_uncertain_evidence",
]

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<MISSING>"]

VALUE_VOCAB: dict[str, int] = {}
_idx = 0
for token in SPECIAL_TOKENS:
    VALUE_VOCAB[token] = _idx
    _idx += 1

_all_values = set()
for slot in EVIDENCE_SLOTS:
    _all_values.update(["low", "medium", "high", "normal", "strong", "weak",
                        "embedding_neighbor_discrepancy_high",
                        "high_frequency_response_high",
                        "spectral_energy_shift_high",
                        "bandpass_response_high",
                        "benign_neighbor_signal_low",
                        "benign_neighbor_signal_high"])

for val in sorted(_all_values):
    if val not in VALUE_VOCAB:
        VALUE_VOCAB[val] = _idx
        _idx += 1

REASON_TYPE_TO_ID = {name: i for i, name in enumerate(REASON_TYPES)}


def get_evidence_slots() -> list[str]:
    return EVIDENCE_SLOTS.copy()


def get_reason_types() -> list[str]:
    return REASON_TYPES.copy()


def encode_reasoning(reasoning: dict) -> torch.LongTensor:
    tokens = []
    for slot in EVIDENCE_SLOTS:
        value = reasoning.get(slot, "<MISSING>")
        tokens.append(VALUE_VOCAB.get(value, VALUE_VOCAB["<UNK>"]))
    return torch.tensor(tokens, dtype=torch.long)


def encode_err_targets(err: ERR) -> dict:
    risk_type_id = REASON_TYPE_TO_ID.get(err.risk_type, REASON_TYPE_TO_ID["weak_or_uncertain_evidence"])

    num_slots = len(EVIDENCE_SLOTS)
    pos_mask = torch.zeros(num_slots, dtype=torch.float)
    neg_mask = torch.zeros(num_slots, dtype=torch.float)

    for i, slot in enumerate(EVIDENCE_SLOTS):
        if slot in err.supporting_evidence:
            pos_mask[i] = 1.0
        if slot in err.counter_evidence:
            neg_mask[i] = 1.0

    return {
        "risk_type_id": risk_type_id,
        "pos_mask": pos_mask,
        "neg_mask": neg_mask,
    }


def get_num_values() -> int:
    return len(VALUE_VOCAB)
