from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.diagnose_reasoner_outputs import (
    build_gate_residual_stats,
    build_logit_shift,
    build_split_info,
    build_threshold_behavior,
    render_markdown,
)


@pytest.fixture()
def diagnosis_data() -> Data:
    x = torch.randn(6, 3)
    y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    data = Data(x=x, y=y)
    data.train_mask = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)
    data.val_mask = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool)
    data.test_mask = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool)
    return data


@pytest.fixture()
def mock_logits() -> tuple[np.ndarray, np.ndarray]:
    base = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
    final = np.array([0.1, 0.0, 0.5, 0.7, 0.7, 1.3], dtype=float)
    return base, final


def test_diagnosis_report_generation_with_mock_logits(diagnosis_data: Data, mock_logits, tmp_path: Path) -> None:
    base_logit, final_logit = mock_logits
    split_info = build_split_info(diagnosis_data, {"stratified": True})
    logit_shift = build_logit_shift(base_logit, final_logit, diagnosis_data.test_mask.numpy(), "test")
    threshold_behavior = build_threshold_behavior(
        base_prob_val=np.array([0.2, 0.6]),
        final_prob_val=np.array([0.3, 0.7]),
        y_val=np.array([0, 1]),
        base_prob_test=1.0 / (1.0 + np.exp(-base_logit[diagnosis_data.test_mask.numpy()])),
        final_prob_test=1.0 / (1.0 + np.exp(-final_logit[diagnosis_data.test_mask.numpy()])),
        y_test=diagnosis_data.y[diagnosis_data.test_mask].numpy(),
    )
    gate_residual = build_gate_residual_stats(
        {
            "gate": torch.tensor([[0.1], [0.2]]),
            "residual_raw": torch.tensor([[1.0], [-1.0]]),
            "rho": torch.tensor(0.3),
        }
    )

    report = {
        "dataset": "tiny",
        "model": "gcn",
        "run_name": "rule",
        "seed": 0,
        "git_hash": "test",
        "split_info": split_info,
        "logit_shift": logit_shift,
        "threshold_behavior": threshold_behavior,
        "gate_residual": gate_residual,
    }

    json_path = tmp_path / "reasoner_diagnosis.json"
    json_path.write_text(json.dumps(report, indent=2))
    loaded = json.loads(json_path.read_text())
    markdown = render_markdown(loaded)

    assert loaded["split_info"]["stratified"] is True
    assert "Reasoner Diagnosis Report" in markdown
    assert "## A. Split Info" in markdown


def test_final_logit_minus_base_logit_statistics(diagnosis_data: Data, mock_logits) -> None:
    base_logit, final_logit = mock_logits
    stats = build_logit_shift(base_logit, final_logit, diagnosis_data.test_mask.numpy(), "test")

    delta = final_logit[diagnosis_data.test_mask.numpy()] - base_logit[diagnosis_data.test_mask.numpy()]
    assert stats["test_logit_shift"]["mean"] == pytest.approx(float(delta.mean()))
    assert stats["test_logit_shift"]["max"] == pytest.approx(float(delta.max()))


def test_positive_prediction_rate_computation(diagnosis_data: Data, mock_logits) -> None:
    base_logit, final_logit = mock_logits
    behavior = build_threshold_behavior(
        base_prob_val=np.array([0.1, 0.9]),
        final_prob_val=np.array([0.2, 0.8]),
        y_val=np.array([0, 1]),
        base_prob_test=1.0 / (1.0 + np.exp(-base_logit[diagnosis_data.test_mask.numpy()])),
        final_prob_test=1.0 / (1.0 + np.exp(-final_logit[diagnosis_data.test_mask.numpy()])),
        y_test=diagnosis_data.y[diagnosis_data.test_mask].numpy(),
    )

    assert 0.0 <= behavior["base_pos_rate_at_0.5"] <= 1.0
    assert 0.0 <= behavior["final_pos_rate_at_0.5"] <= 1.0
    assert behavior["test_f1_base"]["positive_prediction_rate"] >= 0.0
    assert behavior["test_f1_final"]["positive_prediction_rate"] >= 0.0


def test_split_info_written_to_diagnosis(diagnosis_data: Data) -> None:
    split_info = build_split_info(diagnosis_data, {"stratified": False})
    report = {"split_info": split_info}
    rendered = render_markdown({
        "dataset": "tiny",
        "model": "gcn",
        "run_name": "rule",
        "seed": 0,
        "git_hash": "test",
        "split_info": split_info,
        "logit_shift": {},
        "threshold_behavior": {},
        "gate_residual": {},
    })

    assert report["split_info"]["num_train"] == 2
    assert report["split_info"]["global_pos_rate"] == pytest.approx(0.5)
    assert "Stratified: False" in rendered
