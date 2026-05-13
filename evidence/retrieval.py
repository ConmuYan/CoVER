from __future__ import annotations

import math
from typing import Any

SCORE_LEAKAGE_KEYS = frozenset({
    "base_score",
    "score",
    "logit",
    "logits",
    "prob",
    "probs",
    "probability",
    "probabilities",
    "confidence",
    "prediction",
    "pred",
    "label",
    "y",
    "target",
})


class HybridRetriever:
    """Hybrid retrieval: Jaccard recall + IDF cosine rerank.

    HARD CONSTRAINTS:
    - IDF computed from train nodes only
    - No test labels
    - No score/prob/logit in output
    """

    def __init__(
        self,
        top_k_recall: int = 50,
        top_k_final: int = 3,
        jaccard_weight: float = 0.4,
        cosine_weight: float = 0.6,
    ):
        self.top_k_recall = max(0, int(top_k_recall))
        self.top_k_final = max(0, int(top_k_final))
        self.jaccard_weight = float(jaccard_weight)
        self.cosine_weight = float(cosine_weight)
        self._idf: dict[str, float] = {}
        self._idf_fitted = False
        self._train_doc_count = 0

    def fit_idf(self, train_tokens: dict[int, dict[str, str]]) -> None:
        """Compute IDF from train tokens only."""
        doc_freq: dict[str, int] = {}
        num_docs = 0

        for tokens in train_tokens.values():
            num_docs += 1
            for token in self._token_set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self._train_doc_count = num_docs
        self._idf = {
            token: math.log((num_docs + 1) / (df + 1)) + 1.0
            for token, df in doc_freq.items()
        }
        self._idf_fitted = True

    def retrieve(
        self,
        target: dict[str, str],
        bank: dict[int, dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Retrieve top-k most similar nodes from bank.

        Returns: List of {node_id, score, jaccard, cosine} sorted by score desc
        """
        target_tokens = self._token_set(target)
        scored = self._score_bank(target_tokens, bank)
        return scored[: self.top_k_final]

    def retrieve_with_fallback(
        self,
        target: dict[str, str],
        fraud_bank: dict[int, dict[str, str]],
        benign_bank: dict[int, dict[str, str]],
        fn_bank: dict[int, dict[str, str]],
        fp_bank: dict[int, dict[str, str]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Retrieve from all banks with LLM-safe names.

        Returns:
            Dict with fraud_like_reference_cases, benign_like_reference_cases
        """
        target_tokens = self._token_set(target)
        fraud_scored = self._score_banks(target_tokens, fraud_bank, fn_bank)
        benign_scored = self._score_banks(target_tokens, benign_bank, fp_bank)

        return {
            "fraud_like_reference_cases": self._strip_scores(fraud_scored[: self.top_k_final]),
            "benign_like_reference_cases": self._strip_scores(benign_scored[: self.top_k_final]),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get retrieval stats with safety markers."""
        return {
            "metric": "hybrid_token_jaccard_idf_cosine",
            "jaccard_weight": self.jaccard_weight,
            "cosine_weight": self.cosine_weight,
            "top_k_recall": self.top_k_recall,
            "top_k_final": self.top_k_final,
            "train_only_banks": True,
            "test_label_used": False,
            "score_visible_to_teacher": False,
            "target_label_visible_to_teacher": False,
            "base_prediction_visible_to_teacher": False,
        }

    def _score_bank(
        self,
        target_tokens: set[str],
        bank: dict[int, dict[str, str]],
    ) -> list[dict[str, Any]]:
        if not bank:
            return []

        candidates: list[tuple[int, set[str], float]] = []
        for node_id, tokens in bank.items():
            candidate_tokens = self._token_set(tokens)
            jaccard = self._jaccard(target_tokens, candidate_tokens)
            candidates.append((node_id, candidate_tokens, jaccard))

        candidates.sort(key=lambda item: (-item[2], item[0]))
        candidates = candidates[: self.top_k_recall] if self.top_k_recall else []

        scored: list[dict[str, Any]] = []
        for node_id, candidate_tokens, jaccard in candidates:
            cosine = self._cosine(target_tokens, candidate_tokens)
            score = self.jaccard_weight * jaccard + self.cosine_weight * cosine
            scored.append({
                "node_id": node_id,
                "score": score,
                "jaccard": jaccard,
                "cosine": cosine,
            })

        scored.sort(key=lambda item: (-item["score"], item["node_id"]))
        return scored

    def _score_banks(
        self,
        target_tokens: set[str],
        *banks: dict[int, dict[str, str]],
    ) -> list[dict[str, Any]]:
        best_by_node: dict[int, dict[str, Any]] = {}

        for bank in banks:
            for item in self._score_bank(target_tokens, bank):
                node_id = int(item["node_id"])
                current = best_by_node.get(node_id)
                if current is None or item["score"] > current["score"]:
                    best_by_node[node_id] = item

        scored = sorted(best_by_node.values(), key=lambda item: (-item["score"], item["node_id"]))
        return scored

    @staticmethod
    def _strip_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "node_id": item["node_id"],
                "jaccard": item["jaccard"],
                "cosine": item["cosine"],
            }
            for item in results
        ]

    @classmethod
    def _token_set(cls, tokens: dict[str, str]) -> set[str]:
        token_set: set[str] = set()
        for field, value in tokens.items():
            field_name = str(field).lower()
            if field_name in SCORE_LEAKAGE_KEYS:
                continue
            if value is None:
                continue
            token_set.add(f"{field}={value}")
        return token_set

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def _cosine(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0

        shared = a & b
        if not shared:
            return 0.0

        norm_a_sq = sum(self._idf_weight(token) ** 2 for token in a)
        norm_b_sq = sum(self._idf_weight(token) ** 2 for token in b)
        if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
            return 0.0

        dot = sum(self._idf_weight(token) ** 2 for token in shared)
        return dot / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))

    def _idf_weight(self, token: str) -> float:
        if not self._idf_fitted:
            return 1.0
        return self._idf.get(token, 1.0)


__all__ = ["HybridRetriever"]
