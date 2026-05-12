from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import (
    get_checkpoint_dir,
    get_err_cache_dir,
    get_reasoner_checkpoint_path,
    get_stage3_metrics_path,
    teacher_to_run_name,
)


def test_sweep_run_name_does_not_overwrite_qwen_rule() -> None:
    assert teacher_to_run_name("llm") == "qwen"
    assert teacher_to_run_name("rule") == "rule"

    rule_path = get_err_cache_dir("yelpchi", "bwgnn", "rule", 0)
    qwen_path = get_err_cache_dir("yelpchi", "bwgnn", "qwen", 0)

    assert rule_path != qwen_path
    assert "rule" in str(rule_path)
    assert "qwen" in str(qwen_path)


def test_rho_lambda_run_path_correct() -> None:
    run_name = "rho_0.0_lambda_0.5"
    ckpt = get_reasoner_checkpoint_path("yelpchi", "gcn", run_name, seed=7)
    metrics = get_stage3_metrics_path("yelpchi", "gcn", run_name, seed=7)
    base_dir = get_checkpoint_dir("yelpchi", "gcn", run_name, seed=7)

    assert run_name in str(ckpt)
    assert run_name in str(metrics)
    assert base_dir.name == "seed_7"
    assert base_dir.parent.name == run_name


def test_rho_zero_configuration_parsed_correctly(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"name": "tiny"},
                "model": {"name": "gcn"},
                "train": {"seed": 0},
                "reasoner": {"rho": 0.0, "lambda_evi": 0.5},
            }
        )
    )

    config = yaml.safe_load(config_path.read_text())

    assert config["reasoner"]["rho"] == 0.0
    assert config["reasoner"]["lambda_evi"] == 0.5
