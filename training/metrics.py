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


def g_means(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """Compute G-Mean = sqrt(sensitivity * specificity)."""
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return float(np.sqrt(sensitivity * specificity))


def compute_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[str, float]:
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
    metrics["macro_f1"] = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics["g_means"] = g_means(y_true, y_pred_binary)

    for k in k_values:
        pk, rk = precision_recall_at_k(y_true, y_pred_prob, k)
        metrics[f"precision@{k}"] = pk
        metrics[f"recall@{k}"] = rk

    return metrics


def compute_metrics_with_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Same as compute_metrics but uses *threshold* instead of 0.5.

    Returns dict with all standard metrics plus threshold_used and
    positive_prediction_rate.
    """
    if k_values is None:
        k_values = [50, 100, 200]

    metrics: dict[str, float] = {}

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        metrics["roc_auc"] = 0.0

    metrics["auprc"] = average_precision_score(y_true, y_pred_prob)

    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    metrics["f1"] = f1_score(y_true, y_pred_binary, zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics["g_means"] = g_means(y_true, y_pred_binary)
    metrics["threshold_used"] = float(threshold)
    metrics["positive_prediction_rate"] = float(y_pred_binary.mean()) if y_pred_binary.size > 0 else 0.0

    for k in k_values:
        pk, rk = precision_recall_at_k(y_true, y_pred_prob, k)
        metrics[f"precision@{k}"] = pk
        metrics[f"recall@{k}"] = rk

    return metrics


def find_best_macro_f1_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    n_thresholds: int = 19,
    low: float = 0.05,
    high: float = 0.95,
) -> tuple[float, float]:
    """Scan ``n_thresholds`` thresholds in ``[low, high]`` and return
    ``(best_threshold, best_macro_f1)``.

    Replicates ``get_best_f1`` from the original BWGNN reference
    implementation (Tang et al., ICML 2022,
    https://github.com/squareRoot3/Rethinking-Anomaly-Detection,
    ``main.py``), which sweeps 19 thresholds from 0.05 to 0.95 and
    picks the one with highest macro-F1.

    The default 0.5 threshold is a poor choice for imbalanced fraud-
    detection problems where positive-class prevalence is much less
    than 50%; the original paper's reported macro-F1 numbers are only
    reproducible with this threshold search applied on the validation
    set and then carried over to the test set.
    """
    best_thre, best_mf1 = 0.5, 0.0
    for thres in np.linspace(low, high, n_thresholds):
        preds = (y_pred_prob >= float(thres)).astype(int)
        mf1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if mf1 > best_mf1:
            best_mf1 = float(mf1)
            best_thre = float(thres)
    return best_thre, best_mf1


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
