import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_tiny_graph, load_synthetic_graph, load_fraud_dataset
from data.split import generate_masks, apply_scarcity, stratified_split


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
