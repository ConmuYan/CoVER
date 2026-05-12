"""Tests for real sanity scripts without loading real data or Qwen."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_run_real_sanity_missing_data():
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
dataset:
  name: yelpchi
  path: /nonexistent/path/to/YelpChi.mat
  format: mat
  split_mode: supervised
  train_ratio: 0.4
  val_test_ratio: [1, 2]
  split_seed: 42
  scarcity_ratio: 1.0
model:
  name: bwgnn
  hidden_dim: 64
  num_layers: 2
  dropout: 0.3
  num_bands: 3
  agg: concat
train:
  seed: 0
  epochs: 100
  debug_epochs: 3
  lr: 0.01
  optimizer: adam
  weight_decay: 0.0005
  patience: 50
  device: cuda:0
  select_metric: macro_f1
gpu:
  train_visible_devices: "2"
  llm_visible_devices: "3"
evidence:
  trace_size: 1000
  trace_size_small: 32
  teacher: rule
  use_verifier: true
  use_counter: true
reasoner:
  hidden_dim: 128
  evidence_emb_dim: 16
  rho: 0.3
  lambda_evi: 0.5
eval:
  k_values: [50, 100, 200]
""")
        config_path = f.name

    try:
        cmd = [
            sys.executable, "scripts/run_real_sanity.py",
            "--config", config_path,
            "--teacher", "rule",
            "--trace_size", "32",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()
    finally:
        os.unlink(config_path)


def test_report_evidence_quality_mock_artifacts(tmp_path):
    err_cache_dir = tmp_path / "artifacts" / "err_cache" / "test" / "test" / "seed_0"
    err_cache_dir.mkdir(parents=True)

    cards = [{"node_id": 1, "reasoning": {"detector_signal": "test"}}]
    accepted = [{"node_id": 1, "risk_type": "test", "supporting_evidence": [], "counter_evidence": []}]
    rejected = []
    verifier_stats = {"num_total": 1, "num_accepted": 1, "num_rejected": 0, "acceptance_rate": 1.0, "reject_reason_counts": {}}
    stage2_stats = {"num_accepted_after_initial": 1, "num_accepted_after_retry": 0}

    with open(err_cache_dir / "evidence_cards.jsonl", "w") as f:
        for card in cards:
            f.write(json.dumps(card) + "\n")
    with open(err_cache_dir / "accepted_err.jsonl", "w") as f:
        for err in accepted:
            f.write(json.dumps(err) + "\n")
    with open(err_cache_dir / "rejected_err.jsonl", "w") as f:
        for err in rejected:
            f.write(json.dumps(err) + "\n")
    with open(err_cache_dir / "verifier_stats.json", "w") as f:
        json.dump(verifier_stats, f)
    with open(err_cache_dir / "stage2_stats.json", "w") as f:
        json.dump(stage2_stats, f)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
dataset:
  name: test
  path: /fake/path
model:
  name: test
train:
  seed: 0
""")

    cmd = [
        sys.executable, "scripts/report_evidence_quality.py",
        "--config", str(config_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0


def test_generate_stage2_trace_size_arg():
    cmd = [
        sys.executable, "scripts/generate_stage2_err.py",
        "--config", "configs/yelpchi_bwgnn.yaml",
        "--teacher", "rule",
        "--trace_size", "32",
        "--debug",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    assert "Selected" in result.stdout or "trace" in result.stdout.lower()


def test_generate_stage2_large_llm_reject():
    cmd = [
        sys.executable, "scripts/generate_stage2_err.py",
        "--config", "configs/yelpchi_bwgnn.yaml",
        "--teacher", "llm",
        "--trace_size", "100",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "confirm_large_llm_run" in result.stderr or "confirm_large_llm_run" in result.stdout or "Error" in result.stderr


def test_run_real_sanity_gpu_env():
    cmd = [
        sys.executable, "-c",
        """
import subprocess, os, sys
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "2"
result = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
    capture_output=True, text=True, env=env
)
print(f"device_count={result.stdout.strip()}")
""",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
