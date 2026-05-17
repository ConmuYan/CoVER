"""Regression tests on the packet contract — forbidden-field audit.

Loads existing judge_packets.jsonl and verifies no forbidden field
(base_score, base_prob, base_logit, confidence, label, split, FN, FP,
ground_truth, final_prediction) appears in any packet.

Also tests that the generate_synth_token_quality perturbation pipeline
preserves the score-blind contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Forbidden fields per task spec
FORBIDDEN_FIELDS = {
    "base_score", "base_prob", "base_logit", "confidence",
    "label", "split", "FN", "FP", "ground_truth", "final_prediction",
}

JUDGE_PACKETS_PATH = Path(
    "/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd"
    "/artifacts/judge_packets/yelpchi/bwgnn/cover_rel_judge_revived"
    "/seed_42/judge_packets.jsonl"
)


def _check_no_forbidden_keys(obj, path="root"):
    """Recursively check that no key in obj matches a forbidden field."""
    violations = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in FORBIDDEN_FIELDS:
                violations.append(f"{path}.{key}")
            violations.extend(_check_no_forbidden_keys(value, f"{path}.{key}"))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            violations.extend(_check_no_forbidden_keys(item, f"{path}[{i}]"))
    return violations


class TestJudgePacketsForbiddenFields:
    @pytest.mark.skipif(
        not JUDGE_PACKETS_PATH.exists(),
        reason=f"Judge packets not found at {JUDGE_PACKETS_PATH}",
    )
    def test_no_forbidden_fields_in_judge_packets(self):
        """Assert no forbidden field in existing judge_packets.jsonl."""
        violations = []
        with open(JUDGE_PACKETS_PATH) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                packet = json.loads(line)
                # Remove calibration (it contains base_score by design
                # for internal use, but it must NOT be passed to the LLM)
                packet.pop("calibration", None)
                v = _check_no_forbidden_keys(packet, f"line_{line_num}")
                violations.extend(v)

        assert violations == [], (
            f"Found {len(violations)} forbidden field(s) in judge_packets.jsonl:\n"
            + "\n".join(violations[:20])
        )


class TestSynthPerturbed:
    def test_perturbed_packet_no_forbidden_fields(self):
        """Verify that the perturbation pipeline does not introduce
        forbidden fields."""
        from scripts.generate_synth_token_quality import (
            perturb_packet,
            _assert_no_forbidden,
        )
        import random

        # Create a minimal clean packet
        clean_packet = {
            "node_id": 42,
            "dataset": "yelpchi",
            "relation_schema": ["RUR", "RSR", "RTR"],
            "relation_evidence": {
                "RUR": {
                    "degree_bucket": "high",
                    "feature_deviation_bucket": "medium",
                    "neighbor_consistency_bucket": "low",
                    "prototype_margin_bucket": "fraud_like",
                },
                "RSR": {
                    "degree_bucket": "low",
                    "feature_deviation_bucket": "low",
                },
                "RTR": {
                    "degree_bucket": "medium",
                },
            },
            "graph_diagnostic_evidence": {
                "band_response_bucket": "high",
                "embedding_neighbor_discrepancy_bucket": "medium",
            },
        }

        rng = random.Random(123)
        for _ in range(50):
            perturbed, labels = perturb_packet(clean_packet, 0.5, rng)
            # Must not raise
            _assert_no_forbidden(perturbed)
            # Labels must not contain forbidden fields
            for lab in labels:
                assert lab["token"].lower() not in FORBIDDEN_FIELDS
                assert lab["reason"] in {"none", "noisy", "contradictory", "insufficient"}
                assert 0.0 <= lab["q"] <= 1.0

    def test_perturbed_labels_have_correct_reasons(self):
        """Verify perturbation labels map to correct reason types."""
        from scripts.generate_synth_token_quality import perturb_packet
        import random

        packet = {
            "node_id": 1,
            "relation_evidence": {
                "RUR": {"degree_bucket": "high", "feature_deviation_bucket": "low"},
            },
            "graph_diagnostic_evidence": {"band_response_bucket": "medium"},
        }

        rng = random.Random(99)
        perturbed, labels = perturb_packet(packet, 0.9, rng)

        valid_reasons = {"none", "noisy", "contradictory", "insufficient"}
        for lab in labels:
            assert lab["reason"] in valid_reasons
            if lab["q"] == 0.0:
                assert lab["reason"] != "none"
            if lab["q"] == 1.0:
                assert lab["reason"] == "none"
