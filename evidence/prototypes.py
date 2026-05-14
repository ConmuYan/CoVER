"""Train-only prototype bank builder for directional contrastive evidence.

HARD CONSTRAINTS:
- Only uses train nodes (via train_mask)
- No test labels
- No base_score/prob/logit/confidence in output
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

import torch
from torch import Tensor

from evidence.vocab import TOKEN_POLARITY_MAP

SCORE_BLIND_FIELDS = [
    "degree_level",
    "feature_neighbor_discrepancy",
    "detector_signal",
    "detector_signal_strength",
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

FORBIDDEN_FIELDS = {
    "base_score", "score", "logit", "prob", "confidence",
    "prediction", "raw_label", "target",
}


def _extract_tokens(tokens_per_node: dict[int, dict[str, str]]) -> Counter:
    counter: Counter = Counter()
    for tokens in tokens_per_node.values():
        for field, value in tokens.items():
            if value and value != "unknown":
                counter[(field, value)] += 1
    return counter


def _compute_distinctive_tokens(
    target_counter: Counter,
    contrast_counter: Counter,
    target_n: int,
    contrast_n: int,
    laplace_alpha: float = 1.0,
    top_k: int = 10,
    min_log_odds: float = 0.5,
) -> list[dict[str, Any]]:
    """Compute distinctive tokens using IDF-weighted log odds.

    For each token t:
        log_odds(t) = log( P(t | target) / P(t | contrast) )
    where P(t | class) = (count(t, class) + alpha) / (N_class + alpha * V)
    and V = number of unique tokens across both classes.

    Only tokens with log_odds > min_log_odds are returned, up to top_k.
    """
    all_tokens = set(target_counter.keys()) | set(contrast_counter.keys())
    V = len(all_tokens)
    if V == 0 or target_n == 0 or contrast_n == 0:
        return []

    scored: list[tuple[tuple[str, str], float]] = []
    for token in all_tokens:
        t_count = target_counter.get(token, 0)
        c_count = contrast_counter.get(token, 0)

        p_target = (t_count + laplace_alpha) / (target_n + laplace_alpha * V)
        p_contrast = (c_count + laplace_alpha) / (contrast_n + laplace_alpha * V)

        log_odds = math.log(p_target / p_contrast)
        if log_odds > min_log_odds:
            scored.append((token, log_odds))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            "field": tok[0],
            "value": tok[1],
            "log_odds": round(score, 4),
        }
        for tok, score in scored[:top_k]
    ]


def compute_token_polarity_stats(
    train_mask: Tensor,
    y: Tensor,
    graph_tokens: dict[int, list[str]],
) -> dict[str, Any]:
    """Compute per-token polarity statistics from train nodes only.

    For each token in graph evidence tokens:
    - Count occurrences in fraud/benign train nodes
    - Compute P(token|fraud), P(token|benign)
    - Compute log_odds and class_contrast_score

    HARD CONSTRAINTS: Only uses train labels. No test labels.
    """
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    train_labels = y[train_indices]

    fraud_indices = train_indices[train_labels == 1]
    benign_indices = train_indices[train_labels == 0]

    fraud_n = len(fraud_indices)
    benign_n = len(benign_indices)

    all_tokens: set[str] = set()
    fraud_token_counts: dict[str, int] = {}
    benign_token_counts: dict[str, int] = {}

    for idx in fraud_indices:
        nid = idx.item()
        if nid in graph_tokens:
            for token in graph_tokens[nid]:
                all_tokens.add(token)
                fraud_token_counts[token] = fraud_token_counts.get(token, 0) + 1

    for idx in benign_indices:
        nid = idx.item()
        if nid in graph_tokens:
            for token in graph_tokens[nid]:
                all_tokens.add(token)
                benign_token_counts[token] = benign_token_counts.get(token, 0) + 1

    V = len(all_tokens)
    if V == 0 or fraud_n == 0 or benign_n == 0:
        return {
            "token_polarity_stats": {},
            "fraud_distinctive_tokens": [],
            "benign_distinctive_tokens": [],
            "neutral_tokens": [],
            "test_label_used": False,
        }

    laplace_alpha = 1.0
    token_stats: dict[str, dict] = {}
    fraud_distinctive: list[str] = []
    benign_distinctive: list[str] = []
    neutral_list: list[str] = []

    for token in all_tokens:
        fc = fraud_token_counts.get(token, 0)
        bc = benign_token_counts.get(token, 0)

        p_fraud = (fc + laplace_alpha) / (fraud_n + laplace_alpha * V)
        p_benign = (bc + laplace_alpha) / (benign_n + laplace_alpha * V)

        log_odds = math.log(p_fraud / p_benign) if p_benign > 0 else float('inf')
        class_contrast_score = abs(log_odds)

        if log_odds > 0.5:
            polarity = "fraud"
            fraud_distinctive.append(token)
        elif log_odds < -0.5:
            polarity = "benign"
            benign_distinctive.append(token)
        else:
            polarity = "neutral"
            neutral_list.append(token)

        hand_coded = TOKEN_POLARITY_MAP.get(token)
        if hand_coded and hand_coded != polarity:
            logging.getLogger(__name__).warning(
                "Token '%s': learned '%s' conflicts with hand-coded '%s', using learned",
                token, polarity, hand_coded,
            )

        token_stats[token] = {
            "fraud_count": fc,
            "benign_count": bc,
            "p_fraud": round(p_fraud, 6),
            "p_benign": round(p_benign, 6),
            "log_odds": round(log_odds, 4),
            "class_contrast_score": round(class_contrast_score, 4),
            "polarity": polarity,
        }

    return {
        "token_polarity_stats": token_stats,
        "fraud_distinctive_tokens": sorted(fraud_distinctive),
        "benign_distinctive_tokens": sorted(benign_distinctive),
        "neutral_tokens": sorted(neutral_list),
        "test_label_used": False,
    }


def _build_normal_structure_summary(
    train_tokens: dict[int, dict[str, str]],
) -> dict[str, Any]:
    field_values: dict[str, list[str]] = {}
    for tokens in train_tokens.values():
        for field, value in tokens.items():
            if value and value != "unknown":
                field_values.setdefault(field, []).append(value)

    field_modes: dict[str, str] = {}
    for field in SCORE_BLIND_FIELDS:
        vals = field_values.get(field, [])
        if vals:
            field_modes[field] = Counter(vals).most_common(1)[0][0]
        else:
            field_modes[field] = "unknown"

    return {
        "field_modes": field_modes,
        "num_train_nodes": len(train_tokens),
    }


class PrototypeBuilder:
    """Build prototype banks from train set only.

    HARD CONSTRAINTS:
    - Only uses train nodes
    - No test labels
    - No base_score/prob/logit/confidence in output
    """

    def build(
        self,
        train_mask: Tensor,
        y: Tensor,
        evidence_tokens: dict[int, dict[str, str]],
        base_preds: Tensor,
        graph_tokens: dict[int, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Build all prototype banks.

        Args:
            train_mask: Boolean mask for train nodes
            y: Labels (only train labels used)
            evidence_tokens: {node_id: {field: value}} for all nodes
            base_preds: Base model predictions (0/1)

        Returns:
            Dict with:
            - fraud_prototype_bank: {node_id: tokens} for train positive nodes
            - benign_prototype_bank: {node_id: tokens} for train negative nodes
            - train_fn_bank: {node_id: tokens} for train false negatives
            - train_fp_bank: {node_id: tokens} for train false positives
            - fraud_prototype_summary: dict with token_frequencies, distinctive_tokens, num_nodes
            - benign_prototype_summary: dict with token_frequencies, distinctive_tokens, num_nodes
            - normal_structure_summary: dict with field_modes, num_train_nodes
        """
        train_indices = train_mask.nonzero(as_tuple=True)[0]

        train_labels = y[train_indices]
        train_preds = base_preds[train_indices]

        fraud_mask = train_labels == 1
        benign_mask = train_labels == 0

        fraud_indices = train_indices[fraud_mask]
        benign_indices = train_indices[benign_mask]

        fraud_bank: dict[int, dict[str, str]] = {}
        for idx in fraud_indices:
            nid = idx.item()
            if nid in evidence_tokens:
                fraud_bank[nid] = self._filter_score_blind(evidence_tokens[nid])

        benign_bank: dict[int, dict[str, str]] = {}
        for idx in benign_indices:
            nid = idx.item()
            if nid in evidence_tokens:
                benign_bank[nid] = self._filter_score_blind(evidence_tokens[nid])

        # FN (label=1, pred=0) and FP (label=0, pred=1)
        fn_mask = (train_labels == 1) & (train_preds == 0)
        fp_mask = (train_labels == 0) & (train_preds == 1)

        fn_indices = train_indices[fn_mask]
        fp_indices = train_indices[fp_mask]

        fn_bank: dict[int, dict[str, str]] = {}
        for idx in fn_indices:
            nid = idx.item()
            if nid in evidence_tokens:
                fn_bank[nid] = self._filter_score_blind(evidence_tokens[nid])

        fp_bank: dict[int, dict[str, str]] = {}
        for idx in fp_indices:
            nid = idx.item()
            if nid in evidence_tokens:
                fp_bank[nid] = self._filter_score_blind(evidence_tokens[nid])

        fraud_counter = _extract_tokens(fraud_bank)
        benign_counter = _extract_tokens(benign_bank)

        fraud_n = len(fraud_bank)
        benign_n = len(benign_bank)

        fraud_distinctive = _compute_distinctive_tokens(
            fraud_counter, benign_counter, fraud_n, benign_n,
        )
        benign_distinctive = _compute_distinctive_tokens(
            benign_counter, fraud_counter, benign_n, fraud_n,
        )

        fraud_summary = {
            "token_frequencies": {
                f"{k[0]}:{k[1]}": v for k, v in fraud_counter.most_common(20)
            },
            "distinctive_tokens": fraud_distinctive,
            "num_nodes": fraud_n,
        }
        benign_summary = {
            "token_frequencies": {
                f"{k[0]}:{k[1]}": v for k, v in benign_counter.most_common(20)
            },
            "distinctive_tokens": benign_distinctive,
            "num_nodes": benign_n,
        }

        all_train_tokens: dict[int, dict[str, str]] = {}
        for idx in train_indices:
            nid = idx.item()
            if nid in evidence_tokens:
                all_train_tokens[nid] = self._filter_score_blind(evidence_tokens[nid])

        normal_structure_summary = _build_normal_structure_summary(all_train_tokens)

        result = {
            "fraud_prototype_bank": fraud_bank,
            "benign_prototype_bank": benign_bank,
            "train_fn_bank": fn_bank,
            "train_fp_bank": fp_bank,
            "fraud_prototype_summary": fraud_summary,
            "benign_prototype_summary": benign_summary,
            "normal_structure_summary": normal_structure_summary,
        }

        if graph_tokens is not None:
            result["token_polarity_stats_full"] = compute_token_polarity_stats(
                train_mask, y, graph_tokens,
            )

        return result

    @staticmethod
    def _filter_score_blind(tokens: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in tokens.items() if k not in FORBIDDEN_FIELDS}


__all__ = ["PrototypeBuilder", "compute_token_polarity_stats"]
