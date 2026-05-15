from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.evidence_packet import (
    assert_packet_score_blind,
    build_evidence_packet,
    make_packet_context,
    serialize_evidence_packet,
)


def _context():
    features = np.array(
        [
            [0.0, 0.4, 0.9],
            [0.2, 0.5, 0.8],
            [0.9, 0.1, 0.1],
            [0.8, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    return make_packet_context(
        features,
        {
            "net_rur": sparse.csr_matrix(
                np.array(
                    [
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                    ],
                    dtype=np.float32,
                )
            ),
            "net_rsr": sparse.csr_matrix((4, 4), dtype=np.float32),
            "net_rtr": sparse.csr_matrix((4, 4), dtype=np.float32),
        },
    )


def test_evidence_packet_uses_feature_buckets_and_relation_summary_without_scores():
    evidence_card = {
        "node_id": 0,
        "calibration": {"base_score": 0.99, "confidence": 0.9},
        "reasoning": {
            "neighbor_consistency": "low",
            "feature_embedding_disagreement_bucket": "high",
            "evidence_polarity": "fraud_dominant",
            "allowed_support_ids": ["degree_level"],
            "allowed_counter_ids": ["counter_signal"],
        },
    }
    teacher_payload = {
        "node_id": 0,
        "summary": "must not be copied",
        "reasoning": {
            "degree_level": "medium",
            "detector_signal": "high_frequency_response_high",
            "fraud_token_count": 2,
        },
        "fraud_like_reference_cases": [
            {"node_id": 2, "tokens": {"HF_RATIO_HIGH": "active", "IGNORED": "inactive"}}
        ],
        "benign_like_reference_cases": [
            {"node_id": 1, "tokens": {"FEATURE_EMBED_AGREE_HIGH": "active"}}
        ],
        "fraud_prototype_summary": {
            "distinctive_tokens": [{"field": "HF_RATIO_HIGH", "value": "active", "log_odds": 3.0}]
        },
    }

    packet = build_evidence_packet(0, _context(), evidence_card, teacher_payload)
    text = serialize_evidence_packet(packet)

    assert packet["score_blind"] is True
    assert packet["target_feature_summary"]["raw_review_text_available"] is False
    assert packet["target_feature_summary"]["feature_buckets"]["feature_00"] == "low"
    assert packet["relation_context_summary"]["same_user"]["neighbor_count"] == 1
    assert packet["graph_structural_summary"]["neighbor_consistency"] == "low"
    assert packet["retrieved_context"]["fraud_like_reference_cases"] == [
        {"active_tokens": ["HF_RATIO_HIGH"]}
    ]
    assert packet["retrieved_context"]["hard_positive_patterns"] == [
        {"field": "HF_RATIO_HIGH", "value": "active"}
    ]
    assert "base_score" not in text
    assert "confidence" not in text
    assert "must not be copied" not in text
    assert "target_label" not in text


def test_evidence_packet_rejects_forbidden_fields():
    with pytest.raises(ValueError, match="Forbidden"):
        assert_packet_score_blind({"target_label": 1})

    with pytest.raises(ValueError, match="Forbidden"):
        assert_packet_score_blind({"nested": {"base_logit": 2.0}})
