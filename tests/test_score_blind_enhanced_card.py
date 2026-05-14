"""Tests for enhanced score-blind EvidenceCard: no score/logit/prob leaks, new fields."""

from __future__ import annotations

import torch
import pytest

from evidence.adapter import EvidenceAdapter
from evidence.schema import CalibrationChannel, EvidenceCard, ReasoningChannel

BANNED_KEYS = {"base_score", "score", "logit", "prob", "probability", "confidence", "uncertainty"}

NEW_FIELDS = [
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
]


def _make_adapter(num_nodes: int = 10, feat_dim: int = 4) -> EvidenceAdapter:
    x = torch.randn(num_nodes, feat_dim)
    edges = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
    edges += [[(i + 1) % num_nodes, i] for i in range(num_nodes)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return EvidenceAdapter("test_detector", x, edge_index)


def _make_card(adapter: EvidenceAdapter, node_id: int = 0, extras: dict | None = None) -> EvidenceCard:
    logits = torch.randn(adapter.num_nodes, 1)
    embeddings = torch.randn(adapter.num_nodes, 8)
    return adapter._extract_single(node_id, logits, embeddings, extras)


class TestScoreBlindPayload:
    def test_payload_no_banned_keys(self):
        adapter = _make_adapter()
        card = _make_card(adapter)
        payload = card.to_teacher_payload()
        reasoning = payload["reasoning"]

        for key in BANNED_KEYS:
            assert key not in payload, f"Banned key '{key}' found in top-level payload"
            assert key not in reasoning, f"Banned key '{key}' found in reasoning"

    def test_prompt_no_score_keywords(self):
        adapter = _make_adapter()
        card = _make_card(adapter)
        payload = card.to_teacher_payload()
        prompt_text = str(payload).lower()

        for kw in ["base_score", "score", "logit", "prob", "confidence", "uncertainty"]:
            assert kw not in prompt_text, f"Score keyword '{kw}' leaked into prompt text"

    def test_calibration_not_in_payload(self):
        adapter = _make_adapter()
        card = _make_card(adapter)
        payload = card.to_teacher_payload()

        assert "calibration" not in payload
        assert "base_score" not in payload
        assert "uncertainty" not in payload


class TestNewFields:
    def test_all_new_fields_present(self):
        adapter = _make_adapter()
        card = _make_card(adapter)
        payload = card.to_teacher_payload()
        reasoning = payload["reasoning"]

        for field in NEW_FIELDS:
            assert field in reasoning, f"New field '{field}' missing from payload"

    def test_all_new_fields_are_strings(self):
        adapter = _make_adapter()
        card = _make_card(adapter)
        payload = card.to_teacher_payload()
        reasoning = payload["reasoning"]

        for field in NEW_FIELDS:
            assert isinstance(reasoning[field], str), f"Field '{field}' is not a string: {type(reasoning[field])}"

    def test_new_fields_categorical_values(self):
        adapter = _make_adapter()
        card = _make_card(adapter)
        payload = card.to_teacher_payload()
        reasoning = payload["reasoning"]
        valid = {"low", "medium", "high", "unknown"}

        for field in NEW_FIELDS:
            assert reasoning[field] in valid, f"Field '{field}' has invalid value '{reasoning[field]}'"

    def test_no_extras_fields_are_unknown(self):
        adapter = _make_adapter()
        card = _make_card(adapter, extras=None)
        payload = card.to_teacher_payload()
        reasoning = payload["reasoning"]

        bwgnn_fields = [
            "bwgnn_low_band_energy_bucket",
            "bwgnn_mid_band_energy_bucket",
            "bwgnn_high_band_energy_bucket",
            "bwgnn_high_low_energy_ratio_bucket",
            "message_residual_bucket",
        ]
        for field in bwgnn_fields:
            assert reasoning[field] == "unknown", f"Field '{field}' should be 'unknown' without extras, got '{reasoning[field]}'"

    def test_extras_populate_bwgnn_fields(self):
        adapter = _make_adapter()
        extras = {
            "bwgnn_low_band": torch.randn(adapter.num_nodes),
            "bwgnn_mid_band": torch.randn(adapter.num_nodes),
            "bwgnn_high_band": torch.randn(adapter.num_nodes),
            "message_residual": torch.randn(adapter.num_nodes),
        }
        card = _make_card(adapter, extras=extras)
        payload = card.to_teacher_payload()
        reasoning = payload["reasoning"]

        valid = {"low", "medium", "high"}
        for field in ["bwgnn_low_band_energy_bucket", "bwgnn_mid_band_energy_bucket",
                       "bwgnn_high_band_energy_bucket", "message_residual_bucket"]:
            assert reasoning[field] in valid, f"Field '{field}' should be populated from extras, got '{reasoning[field]}'"

    def test_allowed_ids_consistent_with_fields(self):
        adapter = _make_adapter()
        card = _make_card(adapter)
        payload = card.to_teacher_payload()
        reasoning = payload["reasoning"]

        assert isinstance(reasoning["allowed_support_ids"], list)
        assert isinstance(reasoning["allowed_counter_ids"], list)
        assert "degree_level" in reasoning["allowed_support_ids"]
        assert "counter_signal" in reasoning["allowed_counter_ids"]

    def test_backward_compat_existing_fields(self):
        adapter = _make_adapter()
        card = _make_card(adapter)
        payload = card.to_teacher_payload()
        reasoning = payload["reasoning"]

        for field in ["degree_level", "neighbor_consistency", "feature_neighbor_discrepancy",
                       "detector_signal", "detector_signal_strength", "counter_signal"]:
            assert field in reasoning
            assert isinstance(reasoning[field], str)

    def test_isolated_node_handles_gracefully(self):
        num_nodes = 10
        x = torch.randn(num_nodes, 4)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        adapter = EvidenceAdapter("isolated", x, edge_index)
        card = _make_card(adapter, node_id=5)
        payload = card.to_teacher_payload()
        reasoning = payload["reasoning"]

        assert reasoning["neighbor_degree_skew_bucket"] == "unknown"
        assert reasoning["two_hop_consistency_bucket"] == "unknown"
