from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.retrieval import HybridRetriever


def _train_tokens() -> dict[int, dict[str, str]]:
    return {
        0: {"a": "1", "b": "1"},
        1: {"a": "1", "c": "1"},
    }


def _bank_tokens() -> dict[int, dict[str, str]]:
    return {
        10: {"a": "1", "b": "1"},
        11: {"a": "1"},
        12: {"b": "1"},
        13: {"c": "1"},
    }


class TestHybridRetrieval:
    def test_retrieve_top_k(self):
        """Should return at most top_k_final results."""
        retriever = HybridRetriever(top_k_recall=10, top_k_final=1)
        retriever.fit_idf(_train_tokens())

        results = retriever.retrieve({"a": "1", "b": "1"}, _bank_tokens())

        assert len(results) == 1
        assert results[0]["node_id"] == 10

    def test_hybrid_score_computation(self):
        """Score = 0.4 * jaccard + 0.6 * idf_cosine."""
        retriever = HybridRetriever(top_k_recall=10, top_k_final=2, jaccard_weight=0.4, cosine_weight=0.6)
        retriever.fit_idf(_train_tokens())

        results = retriever.retrieve({"a": "1", "b": "1"}, _bank_tokens())

        assert [item["node_id"] for item in results] == [10, 12]
        assert results[0]["score"] == pytest.approx(1.0)

        idf_a = 1.0
        idf_b = math.log((2 + 1) / (1 + 1)) + 1.0
        target_norm = math.sqrt(idf_a**2 + idf_b**2)
        cand_norm = idf_b
        expected_cosine = (idf_b**2) / (target_norm * cand_norm)
        expected_jaccard = 0.5
        expected_score = 0.4 * expected_jaccard + 0.6 * expected_cosine

        assert results[1]["jaccard"] == pytest.approx(expected_jaccard)
        assert results[1]["cosine"] == pytest.approx(expected_cosine)
        assert results[1]["score"] == pytest.approx(expected_score)


class TestRetrievalScoreBlind:
    def test_no_score_leakage(self):
        """No base_score, score, logit, prob, confidence in output."""
        retriever = HybridRetriever(top_k_recall=10, top_k_final=2)
        retriever.fit_idf(_train_tokens())

        output = retriever.retrieve_with_fallback(
            {"a": "1", "b": "1"},
            fraud_bank={10: {"a": "1", "b": "1"}},
            benign_bank={20: {"a": "1"}},
            fn_bank={11: {"b": "1"}},
            fp_bank={21: {"c": "1"}},
        )

        assert set(output) == {"fraud_like_reference_cases", "benign_like_reference_cases"}

        banned = {"base_score", "score", "logit", "prob", "confidence", "probability", "label", "pred"}

        def _collect_keys(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    yield key
                    yield from _collect_keys(value)
            elif isinstance(obj, list):
                for item in obj:
                    yield from _collect_keys(item)

        keys = set(_collect_keys(output))
        assert keys.isdisjoint(banned)


class TestRetrievalStats:
    def test_stats_safety_markers(self):
        """Stats should include test_label_used=False, etc."""
        retriever = HybridRetriever(top_k_recall=7, top_k_final=3, jaccard_weight=0.4, cosine_weight=0.6)

        assert retriever.get_stats() == {
            "metric": "hybrid_token_jaccard_idf_cosine",
            "jaccard_weight": 0.4,
            "cosine_weight": 0.6,
            "top_k_recall": 7,
            "top_k_final": 3,
            "train_only_banks": True,
            "test_label_used": False,
            "score_visible_to_teacher": False,
            "target_label_visible_to_teacher": False,
            "base_prediction_visible_to_teacher": False,
        }
