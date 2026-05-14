"""Tests for contrastive directional prompt safety and content."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.prompt import (
    build_contrastive_directional_messages,
    assert_score_blind_payload,
    assert_score_blind_payload_str,
)
from evidence.schema import CalibrationChannel, EvidenceCard, ReasoningChannel


def _make_payload() -> dict:
    """Create a valid test payload."""
    return {
        "node_id": 0,
        "detector_name": "bwgnn",
        "reasoning": {
            "degree_level": "high",
            "neighbor_consistency": "low",
            "feature_neighbor_discrepancy": "high",
            "detector_signal": "strong",
            "detector_signal_strength": "strong",
            "counter_signal": "weak",
            "allowed_support_ids": ["degree_level", "detector_signal"],
            "allowed_counter_ids": ["counter_signal"],
            "closer_to_fraud_prototype": "high",
            "closer_to_benign_prototype": "low",
            "prototype_conflict_level": "low",
            "fraud_prototype_matching_fields": ["degree_level"],
            "benign_prototype_matching_fields": [],
        },
        "fraud_prototype": {
            "degree_level": "high",
            "detector_signal": "strong",
        },
        "benign_prototype": {
            "degree_level": "low",
            "detector_signal": "normal",
        },
    }


class TestPromptContainsExamples:
    def test_prompt_contains_synthetic_examples(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "Example 1" in user_content
        assert "Example 2" in user_content
        assert "Example 3" in user_content

    def test_prompt_contains_decrease_risk_definition(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "decrease_risk" in user_content
        assert "benign-like" in user_content.lower() or "benign like" in user_content.lower()

    def test_prompt_contains_decision_rubric(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "Decision Rubric" in user_content
        assert "Do not choose increase_risk only because" in user_content


class TestPromptNoForbiddenTerms:
    def test_no_fn_fp_terminology(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"].lower()

        assert "false_negative" not in user_content
        assert "false_positive" not in user_content
        assert "base_error" not in user_content
        assert "train_fn" not in user_content
        assert "train_fp" not in user_content

    def test_no_label_in_payload(self):
        payload = _make_payload()
        payload["label"] = 1
        with pytest.raises(ValueError, match="Score leakage"):
            build_contrastive_directional_messages(payload)

    def test_no_prediction_in_payload(self):
        payload = _make_payload()
        payload["prediction"] = 1
        with pytest.raises(ValueError, match="Score leakage"):
            build_contrastive_directional_messages(payload)

    def test_no_target_in_payload(self):
        payload = _make_payload()
        payload["target"] = 1
        with pytest.raises(ValueError, match="Score leakage"):
            build_contrastive_directional_messages(payload)


class TestPromptScoreBlind:
    def test_disclaimer_present(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "You are not given any base model score" in user_content

    def test_no_actual_score_values(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "0.8" not in user_content
        assert "0.95" not in user_content
        assert "2.1" not in user_content

    def test_disclaimer_allows_score_words(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "score" in user_content.lower()
        assert "probability" in user_content.lower()
        assert "logit" in user_content.lower()
        assert "confidence" in user_content.lower()


class TestPayloadScoreBlind:
    def test_reject_base_score_field(self):
        payload = _make_payload()
        payload["reasoning"]["base_score"] = 0.95
        with pytest.raises(ValueError, match="Score leakage"):
            assert_score_blind_payload(payload)

    def test_reject_prob_field(self):
        payload = _make_payload()
        payload["reasoning"]["prob"] = 0.8
        with pytest.raises(ValueError, match="Score leakage"):
            assert_score_blind_payload(payload)

    def test_reject_logit_field(self):
        payload = _make_payload()
        payload["reasoning"]["logit"] = 2.1
        with pytest.raises(ValueError, match="Score leakage"):
            assert_score_blind_payload(payload)

    def test_reject_confidence_field(self):
        payload = _make_payload()
        payload["reasoning"]["confidence"] = 0.9
        with pytest.raises(ValueError, match="Score leakage"):
            assert_score_blind_payload(payload)


class TestSyntheticExamplesValidity:
    def test_fraud_example_maps_to_increase_risk(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "fraud-like" in user_content.lower() or "fraud like" in user_content.lower()
        assert "increase_risk" in user_content

    def test_benign_example_maps_to_decrease_risk(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "benign-like" in user_content.lower() or "benign like" in user_content.lower()
        assert "decrease_risk" in user_content

    def test_mixed_example_maps_to_uncertain(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "mixed" in user_content.lower()
        assert "uncertain" in user_content


class TestPromptStructure:
    def test_returns_two_messages(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_no_scores(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        system_content = messages[0]["content"]

        assert "Do not mention scores" in system_content
        assert "probabilities" in system_content
        assert "logits" in system_content
        assert "confidence" in system_content
        assert "model predictions" in system_content

    def test_user_content_has_direction_values(self):
        payload = _make_payload()
        messages = build_contrastive_directional_messages(payload)
        user_content = messages[1]["content"]

        assert "increase_risk" in user_content
        assert "decrease_risk" in user_content
        assert "uncertain" in user_content
