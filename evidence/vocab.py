from __future__ import annotations

import torch

from evidence.schema import ERR

EVIDENCE_SLOTS = [
    "degree_level",
    "neighbor_consistency",
    "feature_neighbor_discrepancy",
    "detector_signal",
    "detector_signal_strength",
    "counter_signal",
    "degree_percentile_bucket",
    "neighbor_degree_skew_bucket",
    "two_hop_consistency_bucket",
    "feature_neighbor_cosine_bucket",
    "embedding_neighbor_cosine_bucket",
    "feature_embedding_disagreement_bucket",
    "bwgnn_low_band_energy_bucket",
    "bwgnn_mid_band_energy_bucket",
    "bwgnn_high_band_energy_bucket",
    "bwgnn_high_low_energy_ratio_bucket",
    "message_residual_bucket",
    "closer_to_fraud_prototype",
    "closer_to_benign_prototype",
    "prototype_conflict_level",
]

RELATION_TOKEN_PREFIXES = ("RUR", "RSR", "RTR", "UPU", "USU", "UVU")
RELATION_FRAUD_SUFFIXES = (
    "DEGREE_TOP10",
    "FEATURE_DEVIATION_HIGH",
    "NEIGHBOR_CONSISTENCY_LOW",
    "FRAUD_PROTO_CLOSE",
    "ZSCORE_HIGH_COUNT",
)
RELATION_BENIGN_SUFFIXES = (
    "DEGREE_LOW",
    "BENIGN_PROTO_CLOSE",
)

TOKEN_POLARITY_FRAUD = frozenset({
    "HF_RATIO_TOP10",
    "HF_RATIO_HIGH",
    "BAND_ENERGY_CONFLICT_HIGH",
    "LOW_HIGH_BAND_MISMATCH",
    "FEAT_NEIGH_COS_BOTTOM10",
    "EMB_NEIGH_COS_BOTTOM10",
    "FEATURE_EMBED_DISAGREE_HIGH",
    "PROTO_FRAUD_CLOSE",
    "NORMAL_STRUCTURE_DIST_HIGH",
    "NORMAL_PATTERN_DEVIATION_HIGH",
    "INTERFERING_EDGE_RATIO_HIGH",
    "CLEAN_VIEW_SHIFT_HIGH",
    "RAW_TO_CLEAN_CONFLICT",
    "ANON_FEATURE_RELATION_ZSCORE_HIGH",
} | {
    f"{prefix}_{suffix}"
    for prefix in RELATION_TOKEN_PREFIXES
    for suffix in RELATION_FRAUD_SUFFIXES
})

TOKEN_POLARITY_BENIGN = frozenset({
    "HF_RATIO_LOW",
    "FEAT_NEIGH_COS_TOP20",
    "EMB_NEIGH_COS_TOP20",
    "BAND_ENERGY_STABLE",
    "NORMAL_STRUCTURE_DIST_LOW",
    "LOW_INTERFERENCE_EDGE_RATIO",
    "CLEAN_VIEW_STABLE",
    "NEIGHBOR_CONSISTENCY_HIGH",
    "PROTO_BENIGN_CLOSE",
    "FEATURE_EMBED_AGREE_HIGH",
    "TWO_HOP_CONSISTENCY_HIGH",
    "LOW_HIGH_BAND_MATCH",
} | {
    f"{prefix}_{suffix}"
    for prefix in RELATION_TOKEN_PREFIXES
    for suffix in RELATION_BENIGN_SUFFIXES
})

TOKEN_POLARITY_NEUTRAL = frozenset({
    "PROTO_CONFLICT_HIGH",
    "LOCAL_CURVATURE_OUTLIER_HIGH",
    "EDGE_CURVATURE_VAR_HIGH",
    "RELATION_PROTO_CONFLICT_HIGH",
})

TOKEN_POLARITY_MAP: dict[str, str] = {}
for _tok in TOKEN_POLARITY_FRAUD:
    TOKEN_POLARITY_MAP[_tok] = "fraud"
for _tok in TOKEN_POLARITY_BENIGN:
    TOKEN_POLARITY_MAP[_tok] = "benign"
for _tok in TOKEN_POLARITY_NEUTRAL:
    TOKEN_POLARITY_MAP[_tok] = "neutral"

GRAPH_EVIDENCE_TOKENS = sorted(
    TOKEN_POLARITY_FRAUD | TOKEN_POLARITY_BENIGN | TOKEN_POLARITY_NEUTRAL
)

OPTIONAL_GRAPH_TOKENS: list[str] = []

REASON_TYPES = [
    "structural_discrepancy",
    "camouflage_neighbor",
    "spectral_anomaly",
    "feature_structure_conflict",
    "relation_or_burst_anomaly",
    "weak_or_uncertain_evidence",
]

EVIDENCE_DIRECTIONS = ["increase_risk", "decrease_risk", "uncertain"]
EVIDENCE_STRENGTHS = ["weak", "moderate", "strong"]

DIRECTION_TO_ID = {d: i for i, d in enumerate(EVIDENCE_DIRECTIONS)}
STRENGTH_TO_ID = {s: i for i, s in enumerate(EVIDENCE_STRENGTHS)}

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<MISSING>"]

VALUE_VOCAB: dict[str, int] = {}
_idx = 0
for token in SPECIAL_TOKENS:
    VALUE_VOCAB[token] = _idx
    _idx += 1

_all_values = set()
for slot in EVIDENCE_SLOTS:
    _all_values.update(["low", "medium", "high", "normal", "strong", "weak", "unknown",
                        "embedding_neighbor_discrepancy_high",
                        "high_frequency_response_high",
                        "high_frequency_response_medium",
                        "high_frequency_response_low",
                        "spectral_energy_shift_high",
                        "bandpass_response_high",
                        "benign_neighbor_signal_low",
                        "benign_neighbor_signal_high",
                        "moderate"])

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


def encode_direction_target(err: ERR) -> dict:
    direction_id = DIRECTION_TO_ID.get(err.evidence_direction, DIRECTION_TO_ID["uncertain"])
    return {"direction_id": direction_id}


def get_direction_num_classes() -> int:
    return len(EVIDENCE_DIRECTIONS)
