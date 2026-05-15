from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.evidence_packet import build_judge_packet, judge_available_fields
from scripts.generate_llm_judge import build_judge_messages


def _packet():
    return build_judge_packet(
        node_id=7,
        dataset="yelpchi",
        relation_schema=["RUR", "RSR", "RTR"],
        primary_relation="RUR",
        relation_fusion_source="cover_rel_anchor_gate_nollm",
        relation_stats=np.zeros(27, dtype=np.float32),
        relation_stat_names=[
            "log_degree_norm",
            "degree_top10",
            "degree_low",
            "feature_l2_deviation_z",
            "neighbor_cosine",
            "zscore_high_fraction",
            "fraud_proto_dist_z",
            "benign_proto_dist_z",
            "fraud_minus_benign_margin_z",
        ],
        relation_tokens=["RUR_DEGREE_LOW", "RSR_FEATURE_DEVIATION_HIGH"],
        gate_values={"RUR": 1.0, "RSR": 0.1, "RTR": 0.0},
    )


def test_judge_packet_excludes_forbidden_model_and_label_fields():
    packet = _packet()
    text = str(packet).lower()
    for forbidden in (
        "base_score",
        "base_prob",
        "base_logit",
        "confidence",
        "base_prediction",
        "target_label",
        "split_identity",
        "ground_truth",
    ):
        assert forbidden not in text


def test_judge_packet_available_fields_are_grounding_ids():
    packet = _packet()
    fields = judge_available_fields(packet)
    assert "relation_evidence.RUR.degree_bucket" in fields
    assert "gate_evidence.optional_relation_gate_buckets.RSR" in fields
    assert packet["allowed_support_fields"] == fields


def test_judge_prompt_has_no_base_specific_forbidden_terms():
    prompt = str(build_judge_messages(_packet())).lower()
    for forbidden in ("base_score", "base_prob", "base_logit", "confidence", "target_label"):
        assert forbidden not in prompt
