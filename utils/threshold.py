"""Threshold calibration utilities for binary fraud detection."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ThresholdMetric = Literal["f1", "macro_f1", "balanced_accuracy"]

_THRESHOLDS = np.linspace(0.01, 0.99, 99)


def _prepare_inputs(
    y_true: np.ndarray | list[int] | list[float],
    y_prob: np.ndarray | list[int] | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize inputs and validate binary labels."""
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float).reshape(-1)

    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have the same number of elements.")

    if y_true_arr.size == 0:
        return y_true_arr.astype(int), y_prob_arr

    unique_labels = np.unique(y_true_arr)
    if not np.isin(unique_labels, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    return y_true_arr.astype(int), y_prob_arr


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return binary confusion counts as (tp, fp, fn, tn)."""
    y_true_pos = y_true == 1
    y_pred_pos = y_pred == 1

    tp = int(np.sum(y_true_pos & y_pred_pos))
    fp = int(np.sum((~y_true_pos) & y_pred_pos))
    fn = int(np.sum(y_true_pos & (~y_pred_pos)))
    tn = int(np.sum((~y_true_pos) & (~y_pred_pos)))
    return tp, fp, fn, tn


def _binary_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    total: int,
    actual_pos: int,
    actual_neg: int,
) -> dict[str, float]:
    """Compute binary classification metrics from confusion counts.

    Macro F1 matches sklearn's default label handling: only labels present in
    the union of y_true and y_pred are averaged.
    """
    pred_pos = tp + fp
    pred_neg = total - pred_pos

    precision = float(tp / pred_pos) if pred_pos > 0 else 0.0
    recall = float(tp / actual_pos) if actual_pos > 0 else 0.0
    f1_denom = (2 * tp) + fp + fn
    f1 = float((2 * tp) / f1_denom) if f1_denom > 0 else 0.0

    f1_scores: list[float] = []
    if actual_neg > 0 or pred_neg > 0:
        neg_denom = (2 * tn) + fp + fn
        f1_neg = float((2 * tn) / neg_denom) if neg_denom > 0 else 0.0
        f1_scores.append(f1_neg)
    if actual_pos > 0 or pred_pos > 0:
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    recalls: list[float] = []
    if actual_neg > 0:
        neg_recall_denom = tn + fp
        recalls.append(float(tn / neg_recall_denom) if neg_recall_denom > 0 else 0.0)
    if actual_pos > 0:
        recalls.append(recall)
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0

    positive_prediction_rate = float(pred_pos / total) if total > 0 else 0.0

    return {
        "f1": f1,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": balanced_accuracy,
        "positive_prediction_rate": positive_prediction_rate,
    }


def _metric_from_counts(
    metric: ThresholdMetric,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    total: int,
    actual_pos: int,
    actual_neg: int,
) -> float:
    """Compute a single threshold-selection metric from confusion counts."""
    if metric not in ("f1", "macro_f1", "balanced_accuracy"):
        raise ValueError(f"Unsupported metric: {metric}")
    metrics = _binary_metrics_from_counts(tp, fp, fn, tn, total, actual_pos, actual_neg)
    return metrics[metric]


def find_best_threshold(
    y_true: np.ndarray | list[int] | list[float],
    y_prob: np.ndarray | list[int] | list[float],
    metric: ThresholdMetric = "f1",
) -> tuple[float, float, float]:
    """Find the best threshold on a fixed validation grid.

    The search is performed over ``np.linspace(0.01, 0.99, 99)`` and the
    threshold that maximizes the requested metric is returned together with the
    metric value and positive prediction rate at that threshold.
    """
    y_true_arr, y_prob_arr = _prepare_inputs(y_true, y_prob)

    if y_true_arr.size == 0:
        return 0.5, 0.0, 0.0

    actual_pos = int(np.sum(y_true_arr == 1))
    actual_neg = int(y_true_arr.size - actual_pos)

    best_threshold = 0.5
    best_score = float("-inf")
    best_positive_rate = 0.0

    for threshold in _THRESHOLDS:
        y_pred = (y_prob_arr >= threshold).astype(int)
        tp, fp, fn, tn = _confusion_counts(y_true_arr, y_pred)
        score = _metric_from_counts(metric, tp, fp, fn, tn, y_true_arr.size, actual_pos, actual_neg)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_positive_rate = float(np.mean(y_pred)) if y_pred.size > 0 else 0.0

    return best_threshold, float(best_score if np.isfinite(best_score) else 0.0), best_positive_rate


def evaluate_with_threshold(
    y_true: np.ndarray | list[int] | list[float],
    y_prob: np.ndarray | list[int] | list[float],
    threshold: float,
) -> dict[str, float]:
    """Evaluate binary classification metrics after thresholding probabilities.

    Undefined metrics (for example ROC AUC on single-class targets) are returned
    as ``0.0`` so callers can safely aggregate results without special casing.
    """
    y_true_arr, y_prob_arr = _prepare_inputs(y_true, y_prob)

    if y_true_arr.size == 0:
        return {
            "roc_auc": 0.0,
            "auprc": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "positive_prediction_rate": 0.0,
            "threshold_used": float(threshold),
        }

    y_pred = (y_prob_arr >= threshold).astype(int)
    tp, fp, fn, tn = _confusion_counts(y_true_arr, y_pred)
    actual_pos = int(np.sum(y_true_arr == 1))
    actual_neg = int(y_true_arr.size - actual_pos)
    metrics = _binary_metrics_from_counts(tp, fp, fn, tn, y_true_arr.size, actual_pos, actual_neg)

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        if not np.isfinite(roc_auc):
            roc_auc = 0.0
    except ValueError:
        roc_auc = 0.0

    try:
        auprc = float(average_precision_score(y_true_arr, y_prob_arr))
    except ValueError:
        auprc = 0.0

    return {
        "roc_auc": roc_auc,
        "auprc": auprc,
        "f1": metrics["f1"],
        "macro_f1": metrics["macro_f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "positive_prediction_rate": metrics["positive_prediction_rate"],
        "threshold_used": float(threshold),
    }


def calibrate_and_evaluate(
    y_true_val: np.ndarray | list[int] | list[float],
    y_prob_val: np.ndarray | list[int] | list[float],
    y_true_test: np.ndarray | list[int] | list[float],
    y_prob_test: np.ndarray | list[int] | list[float],
    metric: ThresholdMetric = "f1",
) -> dict[str, float | dict[str, float]]:
    """Calibrate a threshold on validation data and evaluate val/test splits."""
    best_threshold, _, _ = find_best_threshold(y_true_val, y_prob_val, metric=metric)
    val_metrics = evaluate_with_threshold(y_true_val, y_prob_val, best_threshold)
    test_metrics = evaluate_with_threshold(y_true_test, y_prob_test, best_threshold)

    return {
        "best_threshold": best_threshold,
        "val": val_metrics,
        "test": test_metrics,
    }


__all__ = [
    "find_best_threshold",
    "evaluate_with_threshold",
    "calibrate_and_evaluate",
]
