from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.split import generate_masks, stratified_split


@pytest.fixture()
def balanced_graph() -> Data:
    x = torch.randn(12, 4)
    y = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    return Data(x=x, y=y)


@pytest.fixture()
def single_class_graph() -> Data:
    x = torch.randn(6, 4)
    y = torch.zeros(6, dtype=torch.long)
    return Data(x=x, y=y)


def test_stratified_split_preserves_class_distribution(balanced_graph: Data) -> None:
    data = stratified_split(balanced_graph.clone(), seed=7, ratios=(0.5, 0.25, 0.25))

    global_rate = balanced_graph.y.float().mean().item()
    train_rate = data.y[data.train_mask].float().mean().item()
    val_rate = data.y[data.val_mask].float().mean().item()
    test_rate = data.y[data.test_mask].float().mean().item()

    assert abs(train_rate - global_rate) < 0.51
    assert abs(val_rate - global_rate) < 0.51
    assert abs(test_rate - global_rate) < 0.51


def test_masks_are_mutually_exclusive(balanced_graph: Data) -> None:
    data = stratified_split(balanced_graph.clone(), seed=7)

    assert not (data.train_mask & data.val_mask).any()
    assert not (data.train_mask & data.test_mask).any()
    assert not (data.val_mask & data.test_mask).any()
    assert int(data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()) == balanced_graph.num_nodes


def test_seed_reproducibility(balanced_graph: Data) -> None:
    first = stratified_split(balanced_graph.clone(), seed=123)
    second = stratified_split(balanced_graph.clone(), seed=123)

    assert torch.equal(first.train_mask, second.train_mask)
    assert torch.equal(first.val_mask, second.val_mask)
    assert torch.equal(first.test_mask, second.test_mask)


def test_stratified_and_non_stratified_paths_are_different(balanced_graph: Data) -> None:
    stratified = stratified_split(balanced_graph.clone(), seed=0)
    plain = generate_masks(balanced_graph.clone(), seed=0)

    assert not torch.equal(stratified.train_mask, plain.train_mask)
    assert not torch.equal(stratified.val_mask, plain.val_mask)
    assert not torch.equal(stratified.test_mask, plain.test_mask)


def test_stratified_split_single_class_graph(single_class_graph: Data) -> None:
    data = stratified_split(single_class_graph.clone(), seed=9)

    assert data.train_mask.dtype == torch.bool
    assert data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum() == single_class_graph.num_nodes
    assert not (data.train_mask & data.val_mask).any()


def test_generate_masks_zero_nodes_raises() -> None:
    data = Data(x=torch.empty(0, 4), y=torch.empty(0, dtype=torch.long))

    with pytest.raises(ValueError, match="no nodes"):
        generate_masks(data)
