"""Evaluation metrics for graph fraud detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute fraud detection metrics.

    Args:
        y_true: Ground truth labels (0/1).
        y_pred_prob: Predicted probabilities for positive class.
        k_values: K values for Precision@K and Recall@K.

    Returns:
        Dict with metric names and values.
    """
    if k_values is None:
        k_values = [50, 100, 200]

    metrics: dict[str, float] = {}

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        metrics["roc_auc"] = 0.0

    metrics["auprc"] = average_precision_score(y_true, y_pred_prob)

    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    metrics["f1"] = f1_score(y_true, y_pred_binary, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred_binary, zero_division=0)

    for k in k_values:
        pk, rk = precision_recall_at_k(y_true, y_pred_prob, k)
        metrics[f"precision@{k}"] = pk
        metrics[f"recall@{k}"] = rk

    return metrics


def precision_recall_at_k(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """Compute Precision@K and Recall@K.

    Selects top-K nodes by predicted probability, then computes
    precision and recall among those K nodes.
    """
    k = min(k, len(y_true))
    if k == 0:
        return 0.0, 0.0

    top_k_indices = np.argsort(y_pred_prob)[-k:][::-1]
    y_true_top_k = y_true[top_k_indices]

    tp = y_true_top_k.sum()
    precision_at_k = tp / k
    total_positive = y_true.sum()
    recall_at_k = tp / total_positive if total_positive > 0 else 0.0

    return float(precision_at_k), float(recall_at_k)
