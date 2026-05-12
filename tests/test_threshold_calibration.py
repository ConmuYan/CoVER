from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils.threshold as threshold


@pytest.fixture()
def calibrated_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_val = np.array([0, 1, 1, 1], dtype=int)
    y_prob_val = np.array([0.10, 0.40, 0.45, 0.90], dtype=float)
    y_test = np.array([0, 0, 1, 1], dtype=int)
    y_prob_test = np.array([0.20, 0.30, 0.70, 0.80], dtype=float)
    return y_val, y_prob_val, y_test, y_prob_test


def test_best_threshold_not_selected_using_test_set(monkeypatch: pytest.MonkeyPatch, calibrated_split) -> None:
    y_val, y_prob_val, y_test, y_prob_test = calibrated_split
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    real_find = threshold.find_best_threshold

    def wrapped(y_true, y_prob, metric="f1"):
        calls.append((np.asarray(y_true), np.asarray(y_prob)))
        return real_find(y_true, y_prob, metric=metric)

    monkeypatch.setattr(threshold, "find_best_threshold", wrapped)

    result = threshold.calibrate_and_evaluate(y_val, y_prob_val, y_test, y_prob_test)

    assert len(calls) == 1
    assert np.array_equal(calls[0][0], y_val)
    assert np.array_equal(calls[0][1], y_prob_val)
    assert result["best_threshold"] != 0.5


def test_val_f1_threshold_improves_f1(calibrated_split) -> None:
    y_val, y_prob_val, *_ = calibrated_split

    best_threshold, _, _ = threshold.find_best_threshold(y_val, y_prob_val, metric="f1")
    fixed = threshold.evaluate_with_threshold(y_val, y_prob_val, 0.5)
    calibrated = threshold.evaluate_with_threshold(y_val, y_prob_val, best_threshold)

    assert calibrated["f1"] > fixed["f1"]


def test_fixed_and_calibrated_metrics_both_returned(calibrated_split) -> None:
    y_val, y_prob_val, y_test, y_prob_test = calibrated_split

    fixed = threshold.evaluate_with_threshold(y_test, y_prob_test, 0.5)
    calibrated = threshold.calibrate_and_evaluate(y_val, y_prob_val, y_test, y_prob_test)

    assert fixed["threshold_used"] == 0.5
    assert "val" in calibrated and "test" in calibrated and "best_threshold" in calibrated
    assert calibrated["test"]["threshold_used"] == calibrated["best_threshold"]
    assert "positive_prediction_rate" in fixed
    assert "positive_prediction_rate" in calibrated["test"]


def test_threshold_helpers_handle_empty_arrays() -> None:
    best_threshold, best_score, ppr = threshold.find_best_threshold([], [], metric="f1")
    metrics = threshold.evaluate_with_threshold([], [], 0.5)

    assert best_threshold == 0.5
    assert best_score == 0.0
    assert ppr == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["positive_prediction_rate"] == 0.0
