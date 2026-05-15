from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.judge_encoder import encode_judge_records


def _record(explanation: str):
    return {
        "node_id": 1,
        "verdict": "fake",
        "evidence_strength": "strong",
        "key_relation": "RUR",
        "supporting_evidence": ["relation_evidence.RUR.degree_bucket"],
        "counter_evidence": [],
        "uncertainty_factors": [],
        "short_explanation": explanation,
    }


def test_judge_encoder_ignores_short_explanation_for_training_features():
    kwargs = {
        "num_nodes": 4,
        "relation_names": ["RUR", "RSR", "RTR"],
        "evidence_fields": ["relation_evidence.RUR.degree_bucket"],
        "primary_relation": "RUR",
    }
    features_a, mask_a, meta_a = encode_judge_records([_record("first explanation")], **kwargs)
    features_b, mask_b, meta_b = encode_judge_records([_record("different explanation")], **kwargs)

    assert torch.allclose(features_a, features_b)
    assert torch.equal(mask_a, mask_b)
    assert meta_a["short_explanation_used_for_loss"] is False
    assert meta_b["summary_used"] is False


def test_judge_encoder_sets_accepted_mask_and_counts():
    features, mask, meta = encode_judge_records(
        [_record("ok")],
        num_nodes=4,
        relation_names=["RUR", "RSR", "RTR"],
        evidence_fields=["relation_evidence.RUR.degree_bucket"],
        primary_relation="RUR",
    )

    assert features.shape[0] == 4
    assert mask.tolist() == [False, True, False, False]
    assert meta["num_accepted"] == 1
    assert meta["base_score_used"] is False
    assert meta["target_label_used"] is False
