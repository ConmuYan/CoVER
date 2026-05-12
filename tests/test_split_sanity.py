from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.check_split_sanity as check_split_sanity


@pytest.fixture()
def tiny_split() -> Data:
    x = torch.randn(6, 3)
    y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    data = Data(x=x, y=y)
    data.train_mask = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)
    data.val_mask = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.bool)
    data.test_mask = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.bool)
    return data


@pytest.fixture()
def overlap_split() -> Data:
    data = Data(
        x=torch.randn(6, 3),
        y=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
    )
    data.train_mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
    data.val_mask = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool)
    data.test_mask = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.bool)
    return data


def test_mask_overlap_detection_error(overlap_split: Data) -> None:
    overlap = check_split_sanity.check_mask_overlap(
        overlap_split.train_mask,
        overlap_split.val_mask,
        overlap_split.test_mask,
    )

    assert overlap["train_val_overlap"] == 1
    assert overlap["total_overlap"] == 1


def _create_split_file(split_path: Path, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor) -> None:
    """Create a fake split.pt file at the given path."""
    split_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask},
        split_path,
    )


def test_pos_rate_gap_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # y = [0, 0, 1, 0, 1, 1]  (3 positives out of 6)
    # train=[0,1,2,3] pos_rate=0.25, val=[4] pos_rate=1.0, test=[5] pos_rate=1.0
    # gap = 0.75 > 0.02 => WARNING, but no errors => passed=True
    x = torch.randn(6, 3)
    y = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.long)
    data = Data(x=x, y=y)

    train_mask = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)
    val_mask = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.bool)
    test_mask = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool)

    split_path = tmp_path / "artifacts" / "splits" / "tiny" / "stratified_true" / "seed_0" / "split.pt"
    _create_split_file(split_path, train_mask, val_mask, test_mask)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_split_sanity, "load_dataset_without_split", lambda *a, **kw: data.clone())

    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", ["check_split_sanity.py", "--dataset", "tiny", "--seed", "0"])
        check_split_sanity.main()

    report_path = Path("artifacts/reports/tiny/stratified_true/seed_0/split_sanity_report.json")
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["max_pos_rate_gap"] > 0.02
    assert any("max_pos_rate_gap" in w for w in report["warnings"])


def test_split_sanity_report_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # y = [0, 1, 0, 1, 0, 1]  (3 positives out of 6)
    # Balanced split: train=[0,3], val=[1,4], test=[2,5] — all pos_rate=0.5
    x = torch.randn(6, 3)
    y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    data = Data(x=x, y=y)

    train_mask = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.bool)
    val_mask = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool)
    test_mask = torch.tensor([0, 0, 1, 0, 0, 1], dtype=torch.bool)

    split_path = tmp_path / "artifacts" / "splits" / "tiny" / "stratified_true" / "seed_3" / "split.pt"
    _create_split_file(split_path, train_mask, val_mask, test_mask)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_split_sanity, "load_dataset_without_split", lambda *a, **kw: data.clone())
    monkeypatch.setattr(sys, "argv", ["check_split_sanity.py", "--dataset", "tiny", "--seed", "3"])

    with pytest.raises(SystemExit) as exc:
        check_split_sanity.main()

    assert exc.value.code == 0
    report_dir = Path("artifacts/reports/tiny/stratified_true/seed_3")
    json_path = report_dir / "split_sanity_report.json"
    md_path = report_dir / "split_sanity_report.md"

    assert json_path.exists()
    assert md_path.exists()
    report = json.loads(json_path.read_text())
    assert report["passed"] is True
    assert report["split_mode"] == "supervised"
