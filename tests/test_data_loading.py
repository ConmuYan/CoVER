import pytest
import torch
import sys
import json
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_tiny_graph, load_synthetic_graph, load_fraud_dataset
from data.split import generate_masks, apply_scarcity, stratified_split, load_split as load_split_masks
import data.load_fraud as load_fraud_module
import utils.paths as paths


def test_tiny_graph_creation():
    data = load_tiny_graph(num_nodes=50, seed=42)
    assert data.x.shape == (50, 16)
    assert data.y.shape == (50,)
    assert data.y.dtype == torch.long
    assert data.edge_index.shape[0] == 2
    assert hasattr(data, "train_mask")
    assert hasattr(data, "val_mask")
    assert hasattr(data, "test_mask")


def test_synthetic_graph_creation():
    data = load_synthetic_graph(num_nodes=200, num_features=32, seed=42)
    assert data.x.shape == (200, 32)
    assert data.y.shape == (200,)


def test_mask_coverage():
    data = load_tiny_graph(num_nodes=100, seed=42)
    total_masked = data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()
    assert total_masked == 100
    assert not (data.train_mask & data.val_mask).any()
    assert not (data.train_mask & data.test_mask).any()
    assert not (data.val_mask & data.test_mask).any()


def test_mask_determinism():
    data1 = load_tiny_graph(num_nodes=50, seed=123)
    data2 = load_tiny_graph(num_nodes=50, seed=123)
    assert torch.equal(data1.train_mask, data2.train_mask)
    assert torch.equal(data1.val_mask, data2.val_mask)
    assert torch.equal(data1.test_mask, data2.test_mask)


def test_scarcity_reduces_train():
    data = load_tiny_graph(num_nodes=100, seed=42)
    original_train = data.train_mask.sum().item()

    data_scarce = apply_scarcity(data.clone(), scarcity_ratio=0.5, seed=42)
    scarce_train = data_scarce.train_mask.sum().item()

    assert scarce_train < original_train
    assert scarce_train > 0


def test_scarcity_preserves_val_test():
    data = load_tiny_graph(num_nodes=100, seed=42)
    original_val = data.val_mask.sum().item()
    original_test = data.test_mask.sum().item()

    data_scarce = apply_scarcity(data.clone(), scarcity_ratio=0.3, seed=42)

    assert data_scarce.val_mask.sum().item() == original_val
    assert data_scarce.test_mask.sum().item() == original_test


def test_stratified_split():
    data = load_synthetic_graph(num_nodes=200, fraud_ratio=0.1, seed=42)
    data = stratified_split(data, seed=42)

    y_train = data.y[data.train_mask]
    y_all = data.y

    train_fraud_ratio = y_train.float().mean().item()
    all_fraud_ratio = y_all.float().mean().item()

    assert abs(train_fraud_ratio - all_fraud_ratio) < 0.05


def test_load_fraud_dataset_tiny():
    data = load_fraud_dataset("tiny", seed=42)
    assert data.x is not None
    assert data.y is not None
    assert data.train_mask is not None


def test_load_fraud_dataset_synthetic():
    data = load_fraud_dataset("synthetic_small", seed=42)
    assert data.x.shape[0] == 500


def test_generate_masks_custom_ratios():
    data = load_tiny_graph(num_nodes=100, seed=42)
    data = generate_masks(data, seed=42, ratios=(0.6, 0.2, 0.2))

    train_ratio = data.train_mask.sum().item() / 100
    val_ratio = data.val_mask.sum().item() / 100
    test_ratio = data.test_mask.sum().item() / 100

    assert abs(train_ratio - 0.6) < 0.05
    assert abs(val_ratio - 0.2) < 0.05
    assert abs(test_ratio - 0.2) < 0.05


def test_stratified_split_paths_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.chdir(tmp_path)

    y = torch.tensor([1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=torch.long)
    train_mask = torch.tensor([True, True, True, True, False, False, False, False, False])
    val_mask = torch.tensor([False, False, False, False, True, True, False, False, False])
    test_mask = torch.tensor([False, False, False, False, False, False, True, True, True])
    data = SimpleNamespace(y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    split_path_false = load_fraud_module.save_split(data, "demo", 7, "supervised", 0.7, [1, 2], stratified=False)
    split_path_true = load_fraud_module.save_split(data, "demo", 7, "supervised", 0.7, [1, 2], stratified=True)

    assert split_path_false != split_path_true
    assert split_path_false.as_posix().endswith("artifacts/splits/demo/stratified_false/seed_7/split.pt")
    assert split_path_true.as_posix().endswith("artifacts/splits/demo/stratified_true/seed_7/split.pt")

    meta_false = json.loads((split_path_false.parent / "split_meta.json").read_text())
    meta_true = json.loads((split_path_true.parent / "split_meta.json").read_text())

    assert meta_false["stratified"] is False
    assert meta_true["stratified"] is True
    assert meta_true["global_pos_rate"] == pytest.approx(3 / 9)
    assert meta_true["train_pos_rate"] == pytest.approx(1 / 4)
    assert meta_true["val_pos_rate"] == pytest.approx(1 / 2)
    assert meta_true["test_pos_rate"] == pytest.approx(1 / 3)
    assert meta_true["max_pos_rate_gap"] == pytest.approx((1 / 2) - (1 / 4))
    assert meta_true["relative_pos_rate_gap"] == pytest.approx(((1 / 2) - (1 / 4)) / (3 / 9))
    assert meta_true["class_counts_by_split"] == {
        "train": {"pos": 1, "neg": 3},
        "val": {"pos": 1, "neg": 1},
        "test": {"pos": 1, "neg": 2},
    }
    assert isinstance(meta_true["class_counts_by_split"]["train"]["pos"], int)

    loaded = SimpleNamespace(y=y.clone())
    loaded = load_split_masks(loaded, "demo", 7, stratified=True)
    assert torch.equal(loaded.train_mask, train_mask)
    assert torch.equal(loaded.val_mask, val_mask)
    assert torch.equal(loaded.test_mask, test_mask)


def test_load_split_falls_back_to_legacy_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.chdir(tmp_path)

    legacy_dir = tmp_path / "artifacts" / "splits" / "legacy_demo" / "seed_5"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    masks = {
        "train_mask": torch.tensor([True, False, False]),
        "val_mask": torch.tensor([False, True, False]),
        "test_mask": torch.tensor([False, False, True]),
    }
    torch.save(masks, legacy_dir / "split.pt")

    data = SimpleNamespace(y=torch.tensor([0, 1, 0], dtype=torch.long))
    loaded = load_fraud_module.load_split(data, "legacy_demo", 5, stratified=False)

    assert torch.equal(loaded.train_mask, masks["train_mask"])
    assert torch.equal(loaded.val_mask, masks["val_mask"])
    assert torch.equal(loaded.test_mask, masks["test_mask"])


def test_load_fraud_dataset_stratified_switch(monkeypatch):
    data = SimpleNamespace(
        x=torch.randn(4, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y=torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )

    calls = {"generate": 0, "stratified": 0, "save": []}

    def fake_load_from_mat(path):
        return data

    def fake_generate_masks(input_data, seed=0, ratios=(0.7, 0.15, 0.15)):
        calls["generate"] += 1
        input_data.train_mask = torch.tensor([True, True, False, False])
        input_data.val_mask = torch.tensor([False, False, True, False])
        input_data.test_mask = torch.tensor([False, False, False, True])
        return input_data

    def fake_stratified_split(input_data, seed=0, ratios=(0.7, 0.15, 0.15)):
        calls["stratified"] += 1
        input_data.train_mask = torch.tensor([True, False, True, False])
        input_data.val_mask = torch.tensor([False, True, False, False])
        input_data.test_mask = torch.tensor([False, False, False, True])
        return input_data

    def fake_save_split(input_data, dataset, seed, split_mode, train_ratio, val_test_ratio, stratified=False):
        calls["save"].append({"dataset": dataset, "seed": seed, "stratified": stratified})

    monkeypatch.setattr(load_fraud_module, "load_from_mat", fake_load_from_mat)
    monkeypatch.setattr(load_fraud_module, "generate_masks", fake_generate_masks)
    monkeypatch.setattr(load_fraud_module, "stratified_split", fake_stratified_split)
    monkeypatch.setattr(load_fraud_module, "save_split", fake_save_split)

    load_fraud_module.load_fraud_dataset("yelpchi", path="dummy.mat", seed=3, stratified=True)
    load_fraud_module.load_fraud_dataset("yelpchi", path="dummy.mat", seed=3, stratified=False)

    assert calls["stratified"] == 1
    assert calls["generate"] == 1
    assert calls["save"][0]["stratified"] is True
    assert calls["save"][1]["stratified"] is False


def test_new_yelp_style_datasets_use_mat_loader_and_save_split(monkeypatch):
    data = SimpleNamespace(
        x=torch.randn(6, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
    )
    calls = {"paths": [], "datasets": []}

    def fake_load_from_mat(path):
        calls["paths"].append(str(path))
        return SimpleNamespace(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            y=data.y.clone(),
        )

    def fake_stratified_split(input_data, seed=0, ratios=(0.7, 0.15, 0.15)):
        input_data.train_mask = torch.tensor([True, True, False, False, False, False])
        input_data.val_mask = torch.tensor([False, False, True, True, False, False])
        input_data.test_mask = torch.tensor([False, False, False, False, True, True])
        return input_data

    def fake_save_split(input_data, dataset, seed, split_mode, train_ratio, val_test_ratio, stratified=False):
        calls["datasets"].append(dataset)

    monkeypatch.setattr(load_fraud_module, "load_from_mat", fake_load_from_mat)
    monkeypatch.setattr(load_fraud_module, "stratified_split", fake_stratified_split)
    monkeypatch.setattr(load_fraud_module, "save_split", fake_save_split)

    load_fraud_module.load_fraud_dataset("yelpnyc", seed=7, stratified=True)
    load_fraud_module.load_fraud_dataset("yelpzip", seed=7, stratified=True)

    assert calls["paths"] == ["datasets/YelpNYC.mat", "datasets/YelpZip.mat"]
    assert calls["datasets"] == ["yelpnyc", "yelpzip"]


def test_single_relation_dgl_datasets_use_dgl_loader_and_save_split(monkeypatch):
    data = SimpleNamespace(
        x=torch.randn(5, 11),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        y=torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
        hsd=torch.zeros(5),
    )
    calls = {"paths": [], "invert": [], "datasets": []}

    def fake_load_from_dgl(path, hsd_invert=False, hsd_chunk_size=250_000, append_hsd=True):
        calls["paths"].append(str(path))
        calls["invert"].append(bool(hsd_invert))
        return SimpleNamespace(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            y=data.y.clone(),
            hsd=data.hsd.clone(),
        )

    def fake_stratified_split(input_data, seed=0, ratios=(0.7, 0.15, 0.15)):
        input_data.train_mask = torch.tensor([True, True, False, False, False])
        input_data.val_mask = torch.tensor([False, False, True, False, False])
        input_data.test_mask = torch.tensor([False, False, False, True, True])
        return input_data

    def fake_save_split(input_data, dataset, seed, split_mode, train_ratio, val_test_ratio, stratified=False):
        calls["datasets"].append(dataset)

    monkeypatch.setattr(load_fraud_module, "load_from_dgl", fake_load_from_dgl)
    monkeypatch.setattr(load_fraud_module, "stratified_split", fake_stratified_split)
    monkeypatch.setattr(load_fraud_module, "save_split", fake_save_split)

    load_fraud_module.load_fraud_dataset("tfinance", seed=7, stratified=True)
    load_fraud_module.load_fraud_dataset("tsocial", seed=7, stratified=True)

    assert calls["paths"] == ["datasets/tfinance", "datasets/tsocial"]
    assert calls["invert"] == [True, False]
    assert calls["datasets"] == ["tfinance", "tsocial"]
