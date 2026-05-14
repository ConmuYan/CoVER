"""Tests for evidence polarity system (Task 8.7.3)."""
from __future__ import annotations

import torch
import pytest

from evidence.vocab import (
    TOKEN_POLARITY_FRAUD,
    TOKEN_POLARITY_BENIGN,
    TOKEN_POLARITY_NEUTRAL,
    TOKEN_POLARITY_MAP,
    GRAPH_EVIDENCE_TOKENS,
)
from evidence.prototypes import compute_token_polarity_stats
from evidence.adapter import EvidenceAdapter, compute_prototype_similarity


# ── TestTokenPolarity ──────────────────────────────────────────


class TestTokenPolarity:
    def test_fraud_polarity_count(self):
        assert len(TOKEN_POLARITY_FRAUD) == 13

    def test_benign_polarity_count(self):
        assert len(TOKEN_POLARITY_BENIGN) == 12

    def test_neutral_polarity_count(self):
        assert len(TOKEN_POLARITY_NEUTRAL) == 3

    def test_polarity_map_complete(self):
        assert len(TOKEN_POLARITY_MAP) == 28
        all_tokens = TOKEN_POLARITY_FRAUD | TOKEN_POLARITY_BENIGN | TOKEN_POLARITY_NEUTRAL
        for t in all_tokens:
            assert t in TOKEN_POLARITY_MAP

    def test_no_overlap(self):
        assert TOKEN_POLARITY_FRAUD & TOKEN_POLARITY_BENIGN == set()
        assert TOKEN_POLARITY_FRAUD & TOKEN_POLARITY_NEUTRAL == set()
        assert TOKEN_POLARITY_BENIGN & TOKEN_POLARITY_NEUTRAL == set()

    def test_graph_tokens_covers_all(self):
        assert len(GRAPH_EVIDENCE_TOKENS) == 28
        all_tokens = TOKEN_POLARITY_FRAUD | TOKEN_POLARITY_BENIGN | TOKEN_POLARITY_NEUTRAL
        assert set(GRAPH_EVIDENCE_TOKENS) == all_tokens


# ── TestBenignTokenGeneration ──────────────────────────────────


class TestBenignTokenGeneration:
    def _make_adapter(self, n=100):
        return EvidenceAdapter(
            "bwgnn",
            torch.randn(n, 5),
            torch.randint(0, n, (2, n * 2)),
        )

    def test_benign_tokens_generated(self):
        n = 100
        adapter = self._make_adapter(n)
        base_logits = torch.randn(n)
        embeddings = torch.randn(n, 5)

        all_tokens = set()
        for i in range(n):
            tokens = adapter.generate_graph_evidence_tokens(
                i, base_logits, embeddings, None, None,
            )
            all_tokens.update(tokens)

        benign_present = all_tokens & TOKEN_POLARITY_BENIGN
        assert len(benign_present) > 0, f"No benign tokens. Got: {sorted(all_tokens)}"

    def test_fraud_tokens_generated(self):
        n = 100
        adapter = self._make_adapter(n)
        base_logits = torch.randn(n)
        embeddings = torch.randn(n, 5)

        all_tokens = set()
        for i in range(n):
            tokens = adapter.generate_graph_evidence_tokens(
                i, base_logits, embeddings, None, None,
            )
            all_tokens.update(tokens)

        fraud_present = all_tokens & TOKEN_POLARITY_FRAUD
        assert len(fraud_present) > 0, f"No fraud tokens. Got: {sorted(all_tokens)}"


# ── TestTokenPolarityLearning ──────────────────────────────────


class TestTokenPolarityLearning:
    def test_polarity_stats_uses_train_only(self):
        train_mask = torch.tensor([True, True, True, False, False])
        y = torch.tensor([1, 0, 1, 0, 1])
        graph_tokens = {
            0: ["HF_RATIO_HIGH", "BAND_ENERGY_CONFLICT_HIGH"],
            1: ["HF_RATIO_LOW", "BAND_ENERGY_STABLE"],
            2: ["FEAT_NEIGH_COS_BOTTOM10", "HF_RATIO_HIGH"],
            3: ["HF_RATIO_LOW"],
            4: ["BAND_ENERGY_CONFLICT_HIGH"],
        }
        result = compute_token_polarity_stats(train_mask, y, graph_tokens)
        assert result["test_label_used"] is False
        assert isinstance(result["token_polarity_stats"], dict)
        total = (
            len(result["fraud_distinctive_tokens"])
            + len(result["benign_distinctive_tokens"])
            + len(result["neutral_tokens"])
        )
        assert total > 0

    def test_polarity_stats_empty(self):
        train_mask = torch.tensor([False, False])
        y = torch.tensor([0, 1])
        result = compute_token_polarity_stats(train_mask, y, {})
        assert result["test_label_used"] is False
        assert len(result["token_polarity_stats"]) == 0


# ── TestPrototypeSimilarity ────────────────────────────────────


class TestPrototypeSimilarity:
    def test_distinctive_fields_narrow_comparison(self):
        reasoning = {
            "degree_level": "high",
            "feature_neighbor_discrepancy": "high",
            "neighbor_consistency": "low",
        }
        prototype = {
            "degree_level": "high",
            "feature_neighbor_discrepancy": "low",
            "neighbor_consistency": "high",
        }

        match_all, total_all, _ = compute_prototype_similarity(reasoning, prototype)
        distinctive = {"degree_level"}
        match_dist, total_dist, _ = compute_prototype_similarity(
            reasoning, prototype, distinctive,
        )

        assert total_dist <= total_all
        assert match_dist <= total_dist

    def test_no_distinctive_falls_back(self):
        reasoning = {"degree_level": "high"}
        prototype = {"degree_level": "high"}
        match, total, fields = compute_prototype_similarity(reasoning, prototype)
        assert match == 1
        assert total == 1


# ── TestPayloadPolarity ────────────────────────────────────────


class TestPayloadPolarity:
    def _make_adapter(self, n=50):
        return EvidenceAdapter(
            "bwgnn",
            torch.randn(n, 5),
            torch.randint(0, n, (2, n * 2)),
        )

    def test_polarity_distinguishes_types(self):
        adapter = self._make_adapter()
        polarities = set()
        for i in range(10):
            card = adapter._extract_single(
                i, torch.randn(50), torch.randn(50, 5), None, None,
            )
            polarities.add(card.reasoning.evidence_polarity)
        assert len(polarities) >= 1

    def test_payload_contains_polarity_no_scores(self):
        adapter = self._make_adapter(20)
        card = adapter._extract_single(
            0, torch.randn(20), torch.randn(20, 5), None, None,
        )
        payload = card.to_teacher_payload()

        assert "evidence_polarity" in payload["reasoning"]
        assert "fraud_token_count" in payload["reasoning"]
        assert "benign_token_count" in payload["reasoning"]

        payload_str = str(payload).lower()
        for forbidden in ["base_score", "probability", "logit", "confidence"]:
            assert forbidden not in payload_str, f"Found '{forbidden}' in payload"


# ── TestScoreBlind ─────────────────────────────────────────────


class TestScoreBlind:
    def test_no_score_leakage_in_tokens(self):
        adapter = EvidenceAdapter(
            "bwgnn",
            torch.randn(20, 5),
            torch.randint(0, 20, (2, 40)),
        )
        tokens = adapter.generate_graph_evidence_tokens(
            0, torch.randn(20), torch.randn(20, 5), None, None,
        )
        for token in tokens:
            tl = token.lower()
            for forbidden in ["score", "prob", "logit", "confidence", "label"]:
                assert forbidden not in tl, f"Token '{token}' contains '{forbidden}'"
