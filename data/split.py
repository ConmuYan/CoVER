"""Train/val/test split utilities for graph datasets.

Provides deterministic, seed-based mask generation for PyG Data objects.
Supports scarcity_ratio for label-scarce experiments.
"""

from __future__ import annotations

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
