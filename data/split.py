"""Train/val/test split utilities for graph datasets.

Provides deterministic, seed-based mask generation for PyG Data objects.
Supports scarcity_ratio for label-scarce experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

try:
    from torch_geometric.data import Data
except ImportError:
    from typing import Any
    Data = Any


def generate_masks(
    data: Data,
    seed: int = 1,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Data:
    """Generate train/val/test boolean masks for a PyG Data object.

    Creates deterministic masks based on the seed.

    Args:
        data: PyG Data object with x or y attribute.
        seed: Random seed for reproducibility.
        ratios: (train_ratio, val_ratio, test_ratio). Must sum to 1.0.

    Returns:
        Data object with train_mask, val_mask, test_mask added.
    """
    train_ratio, val_ratio, test_ratio = ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    num_nodes = _get_num_nodes(data)
    if num_nodes == 0:
        raise ValueError("Data object has no nodes")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)

    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def apply_scarcity(
    data: Data,
    scarcity_ratio: float = 1.0,
    seed: int = 1,
) -> Data:
    """Apply label scarcity to training set.

    When scarcity_ratio < 1.0, only a fraction of training labels are kept.
    Val/test masks are NEVER modified.

    Args:
        data: PyG Data object with train_mask and y.
        scarcity_ratio: Fraction of train labels to keep (0.0, 1.0].
        seed: Random seed for selection.

    Returns:
        Data object with modified train_mask.
    """
    if scarcity_ratio >= 1.0:
        return data

    if not hasattr(data, "train_mask") or data.train_mask is None:
        raise ValueError("Data must have train_mask before applying scarcity")

    train_indices = data.train_mask.nonzero(as_tuple=True)[0]
    num_keep = max(1, int(len(train_indices) * scarcity_ratio))

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_indices), generator=generator)
    keep_indices = train_indices[perm[:num_keep]]

    new_train_mask = torch.zeros_like(data.train_mask)
    new_train_mask[keep_indices] = True
    data.train_mask = new_train_mask

    return data


def stratified_split(
    data: Data,
    seed: int = 1,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Data:
    """Generate stratified train/val/test masks preserving class distribution.

    Args:
        data: PyG Data object with labels (y).
        seed: Random seed.
        ratios: (train, val, test) ratios.

    Returns:
        Data object with stratified masks added.
    """
    train_ratio, val_ratio, _test_ratio = ratios
    labels = data.y
    num_nodes = labels.shape[0]

    generator = torch.Generator().manual_seed(seed)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for label in labels.unique():
        indices = (labels == label).nonzero(as_tuple=True)[0]
        perm = indices[torch.randperm(indices.shape[0], generator=generator)]

        n = perm.shape[0]
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        train_mask[perm[:train_end]] = True
        val_mask[perm[train_end:val_end]] = True
        test_mask[perm[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def _get_num_nodes(data: Data) -> int:
    """Extract number of nodes from a PyG Data object."""
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        return data.num_nodes
    if hasattr(data, "x") and data.x is not None:
        return data.x.shape[0]
    if hasattr(data, "y") and data.y is not None:
        return data.y.shape[0]
    if hasattr(data, "edge_index") and data.edge_index is not None:
        return int(data.edge_index.max()) + 1
    return 0


def _count_positive(labels: torch.Tensor) -> int:
    return int((labels > 0).sum().item())


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def save_split(
    data: Data,
    dataset: str,
    seed: int,
    split_mode: str,
    train_ratio: float,
    val_test_ratio: list[int],
    stratified: bool = False,
):
    from utils.paths import ensure_dir, get_stratified_split_meta_path, get_stratified_split_path

    split_path = get_stratified_split_path(dataset, stratified, seed)
    ensure_dir(split_path.parent)
    split_meta_path = get_stratified_split_meta_path(dataset, stratified, seed)

    torch.save({
        "train_mask": data.train_mask,
        "val_mask": data.val_mask,
        "test_mask": data.test_mask,
    }, split_path)

    y = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    num_train = int(train_mask.sum().item())
    num_val = int(val_mask.sum().item())
    num_test = int(test_mask.sum().item())
    num_total = int(y.shape[0])

    num_pos_total = _count_positive(y)
    num_pos_train = _count_positive(y[train_mask])
    num_pos_val = _count_positive(y[val_mask])
    num_pos_test = _count_positive(y[test_mask])

    num_neg_train = int(num_train - num_pos_train)
    num_neg_val = int(num_val - num_pos_val)
    num_neg_test = int(num_test - num_pos_test)

    pos_rate_train = _safe_rate(num_pos_train, num_train)
    pos_rate_val = _safe_rate(num_pos_val, num_val)
    pos_rate_test = _safe_rate(num_pos_test, num_test)
    global_pos_rate = _safe_rate(num_pos_total, num_total)

    split_pos_rates = [r for r in (pos_rate_train, pos_rate_val, pos_rate_test)]
    max_pos_rate_gap = max(split_pos_rates) - min(split_pos_rates)
    relative_pos_rate_gap = max_pos_rate_gap / global_pos_rate if global_pos_rate > 0 else 0.0

    meta = {
        "dataset": dataset,
        "seed": seed,
        "split_mode": split_mode,
        "train_ratio": train_ratio,
        "val_test_ratio": val_test_ratio,
        "stratified": stratified,
        "num_nodes": num_total,
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "num_pos_train": num_pos_train,
        "num_pos_val": num_pos_val,
        "num_pos_test": num_pos_test,
        "pos_rate_train": pos_rate_train,
        "pos_rate_val": pos_rate_val,
        "pos_rate_test": pos_rate_test,
        "global_pos_rate": global_pos_rate,
        "train_pos_rate": pos_rate_train,
        "val_pos_rate": pos_rate_val,
        "test_pos_rate": pos_rate_test,
        "max_pos_rate_gap": max_pos_rate_gap,
        "relative_pos_rate_gap": relative_pos_rate_gap,
        "class_counts_by_split": {
            "train": {"pos": num_pos_train, "neg": num_neg_train},
            "val": {"pos": num_pos_val, "neg": num_neg_val},
            "test": {"pos": num_pos_test, "neg": num_neg_test},
        },
        "mask_overlap_counts": {
            "train_val": (train_mask & val_mask).sum().item(),
            "train_test": (train_mask & test_mask).sum().item(),
            "val_test": (val_mask & test_mask).sum().item(),
        },
    }

    with open(split_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return split_path


def load_split(data: Data, dataset: str, seed: int, stratified: bool = False) -> Data:
    from utils.paths import get_stratified_split_path

    split_path = get_stratified_split_path(dataset, stratified, seed)
    if not split_path.exists() and not stratified:
        legacy_split_path = Path("artifacts") / "splits" / dataset / f"seed_{seed}" / "split.pt"
        if legacy_split_path.exists():
            split_path = legacy_split_path

    if split_path.exists():
        masks = torch.load(split_path, weights_only=True)
        data.train_mask = masks["train_mask"]
        data.val_mask = masks["val_mask"]
        data.test_mask = masks["test_mask"]
    return data
