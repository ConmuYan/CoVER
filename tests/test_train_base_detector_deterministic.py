"""Regression test for train_base_detector.py --deterministic flag.

Runs the script with --debug --deterministic on the tiny dataset and asserts
retraining_metrics.json exists and contains seed/git_hash/test_metrics.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "train_base_detector.py"
CONFIG = PROJECT_ROOT / "configs" / "raer_fd" / "base_detectors" / "yelpchi_sage.yaml"
SEED = 99


def _checkpoint_dir() -> Path:
    # dataset_name comes from config["dataset"]["name"] = "yelpchi", not "tiny"
    return PROJECT_ROOT / "artifacts" / "checkpoints" / "yelpchi" / "sage" / "deterministic_test" / f"seed_{SEED}"


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test artifacts before and after."""
    d = _checkpoint_dir()
    if d.exists():
        shutil.rmtree(d)
    yield
    if d.exists():
        shutil.rmtree(d)


def test_deterministic_flag_creates_retraining_metrics():
    """--deterministic must produce retraining_metrics.json with required keys."""
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--config", str(CONFIG),
            "--debug",
            "--deterministic",
            "--seed", str(SEED),
            "--run_name", "deterministic_test",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"train_base_detector.py failed:\n{result.stderr}\n{result.stdout}"

    retrain_path = _checkpoint_dir() / "retraining_metrics.json"
    assert retrain_path.exists(), f"retraining_metrics.json not found at {retrain_path}"

    with open(retrain_path) as f:
        data = json.load(f)

    assert data["seed"] == SEED
    assert "git_hash" in data
    assert isinstance(data["git_hash"], str)
    assert "test_metrics" in data
    assert isinstance(data["test_metrics"], dict)
    assert data["deterministic"] is True


def test_no_deterministic_flag_skips_retraining_metrics():
    """Without --deterministic, retraining_metrics.json must NOT be created."""
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--config", str(CONFIG),
            "--debug",
            "--seed", str(SEED),
            "--run_name", "deterministic_test",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"train_base_detector.py failed:\n{result.stderr}\n{result.stdout}"

    retrain_path = _checkpoint_dir() / "retraining_metrics.json"
    assert not retrain_path.exists(), "retraining_metrics.json should not exist without --deterministic"
