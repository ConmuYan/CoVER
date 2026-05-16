"""Graph dataset loaders for cover-fd.

Supports synthetic graphs for debug and real datasets (.pt/.pkl/.npz/.mat).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any

from data.split import generate_masks, stratified_split, apply_scarcity, save_split, load_split


def load_tiny_graph(
    num_nodes: int = 50,
    num_features: int = 16,
    fraud_ratio: float = 0.2,
    edge_probability: float = 0.3,
    seed: int = 42,
) -> Data:
    """Create a synthetic tiny graph for testing and smoke runs."""
    generator = torch.Generator().manual_seed(seed)

    x = torch.randn(num_nodes, num_features, generator=generator)

    num_fraud = max(1, int(num_nodes * fraud_ratio))
    y = torch.zeros(num_nodes, dtype=torch.long)
    fraud_indices = torch.randperm(num_nodes, generator=generator)[:num_fraud]
    y[fraud_indices] = 1

    src_list: list[int] = []
    dst_list: list[int] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if torch.rand(1, generator=generator).item() < edge_probability:
                src_list.extend([i, j])
                dst_list.extend([j, i])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long) if src_list else torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return generate_masks(data, seed=seed)


def load_synthetic_graph(
    num_nodes: int = 1000,
    num_features: int = 64,
    fraud_ratio: float = 0.1,
    avg_degree: int = 10,
    seed: int = 42,
) -> Data:
    """Create a larger synthetic graph for pipeline validation."""
    generator = torch.Generator().manual_seed(seed)

    x = torch.randn(num_nodes, num_features, generator=generator)

    num_fraud = max(1, int(num_nodes * fraud_ratio))
    y = torch.zeros(num_nodes, dtype=torch.long)
    fraud_indices = torch.randperm(num_nodes, generator=generator)[:num_fraud]
    y[fraud_indices] = 1

    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    data = Data(x=x, edge_index=edge_index, y=y)
    return generate_masks(data, seed=seed)


def load_from_pt(path: str | Path) -> Data:
    """Load dataset from PyTorch .pt file."""
    obj = torch.load(path, weights_only=False)
    if isinstance(obj, Data):
        return obj
    if isinstance(obj, dict):
        return Data(**obj)
    raise ValueError(f"Unexpected type in .pt file: {type(obj)}")


def load_from_pkl(path: str | Path) -> Data:
    """Load dataset from pickle file."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, Data):
        return obj
    if isinstance(obj, dict):
        return Data(**obj)
    raise ValueError(f"Unexpected type in .pkl file: {type(obj)}")


def load_from_npz(path: str | Path) -> Data:
    """Load dataset from numpy .npz file."""
    arr = np.load(path, allow_pickle=True)
    kwargs = {}
    for key in arr.files:
        val = arr[key]
        if val.dtype == object:
            val = val.item()
        kwargs[key] = torch.as_tensor(val) if isinstance(val, np.ndarray) else val
    return Data(**kwargs)


def load_from_mat(path: str | Path) -> Data:
    """Load dataset from .mat file (YelpChi/Amazon format).

    Adds self-loops to the homogeneous adjacency to match the reference
    BWGNN protocol (``dgl.add_self_loop`` in
    ``external/Rethinking-Anomaly-Detection/dataset.py``). The raw
    ``mat['homo']`` matrix from YelpChi/Amazon does not include
    self-loops; without them, Beta wavelet filtering cannot propagate
    a node's own feature, which reduces AUC by ~1.2 percentage points
    (verified empirically on YelpChi).
    """
    from scipy.io import loadmat

    mat = loadmat(str(path))

    features = mat["features"]
    if hasattr(features, "toarray"):
        features = features.toarray()
    x = torch.tensor(features, dtype=torch.float32)

    label = mat["label"].squeeze()
    y = torch.tensor(label, dtype=torch.long)

    adj = mat["homo"]
    if hasattr(adj, "tocoo"):
        adj = adj.tocoo()
    row = torch.tensor(adj.row, dtype=torch.long)
    col = torch.tensor(adj.col, dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)

    # Add self-loops (paper protocol). Skip nodes that already have one.
    num_nodes = x.shape[0]
    has_self = (row == col)
    nodes_with_self = set(row[has_self].tolist())
    missing_self = [i for i in range(num_nodes) if i not in nodes_with_self]
    if missing_self:
        sl = torch.tensor(missing_self, dtype=torch.long)
        sl_edges = torch.stack([sl, sl], dim=0)
        edge_index = torch.cat([edge_index, sl_edges], dim=1)

    return Data(x=x, edge_index=edge_index, y=y)


_LOADERS = {
    ".pt": load_from_pt,
    ".pkl": load_from_pkl,
    ".npz": load_from_npz,
    ".mat": load_from_mat,
}


def load_fraud_dataset(
    name: str,
    path: str | Path | None = None,
    format: str | None = None,
    seed: int = 0,
    scarcity_ratio: float = 1.0,
    split_mode: str = "supervised",
    train_ratio: float = 0.7,
    val_test_ratio: list[int] | None = None,
    stratified: bool = False,
) -> Data:
    """Load a graph fraud detection dataset.

    Supported datasets:
    - tiny: synthetic tiny graph (50 nodes) for testing
    - synthetic_small: synthetic graph (500 nodes)
    - yelpchi, amazon: real datasets from .mat files
    - Custom: load from path with specified format

    Args:
        name: Dataset name or 'custom'.
        path: Path to dataset file (required for custom).
        format: File format (pt/pkl/npz/mat). Auto-detected if None.
        seed: Random seed for splits.
        scarcity_ratio: Fraction of train labels to keep.
        split_mode: Split mode (supervised/semi-supervised).
        train_ratio: Training set ratio (default 0.7).
        val_test_ratio: Validation to test ratio [val, test] (default [1, 2]).
        stratified: Use stratified split preserving class distribution.

    Returns:
        PyG Data object with x, edge_index, y, train_mask, val_mask, test_mask.
    """
    if val_test_ratio is None:
        val_test_ratio = [1, 2]

    # Calculate actual ratios from train_ratio and val_test_ratio
    val_ratio = (1 - train_ratio) * val_test_ratio[0] / sum(val_test_ratio)
    test_ratio = (1 - train_ratio) * val_test_ratio[1] / sum(val_test_ratio)
    ratios = (train_ratio, val_ratio, test_ratio)

    if name == "tiny":
        data = load_tiny_graph(seed=seed)
        if stratified:
            data = stratified_split(data, seed=seed, ratios=ratios)
    elif name == "synthetic_small":
        data = load_synthetic_graph(num_nodes=500, seed=seed)
        if stratified:
            data = stratified_split(data, seed=seed, ratios=ratios)
    elif name == "synthetic_medium":
        data = load_synthetic_graph(num_nodes=2000, seed=seed)
        if stratified:
            data = stratified_split(data, seed=seed, ratios=ratios)
    elif name in ("yelpchi", "amazon"):
        if path is None:
            raise ValueError(f"Path required for dataset '{name}'")
        data = load_from_mat(path)
        if stratified:
            data = stratified_split(data, seed=seed, ratios=ratios)
        else:
            data = generate_masks(data, seed=seed, ratios=ratios)
        save_split(data, name, seed, split_mode, train_ratio, val_test_ratio, stratified=stratified)
    elif name == "custom":
        if path is None:
            raise ValueError("Path required for custom dataset")
        path = Path(path)
        if format:
            ext = f".{format}"
        else:
            ext = path.suffix
        loader = _LOADERS.get(ext)
        if loader is None:
            raise ValueError(f"Unknown format: {ext}. Supported: {list(_LOADERS.keys())}")
        data = loader(path)
        if not hasattr(data, "train_mask") or data.train_mask is None:
            if stratified:
                data = stratified_split(data, seed=seed, ratios=ratios)
            else:
                data = generate_masks(data, seed=seed, ratios=ratios)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if scarcity_ratio < 1.0:
        data = apply_scarcity(data, scarcity_ratio=scarcity_ratio, seed=seed)

    return data
