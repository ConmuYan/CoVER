"""Graph dataset loaders for RAER-FD.

Supports synthetic graphs, benchmark ``.mat`` files, and DGL single-relation
datasets used by PriorF-GNN.
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


REAL_MAT_DATASETS = {
    "yelpchi": "datasets/YelpChi.mat",
    "yelpnyc": "datasets/YelpNYC.mat",
    "yelpzip": "datasets/YelpZip.mat",
    "amazon": "datasets/Amazon.mat",
}


REAL_DGL_DATASETS = {
    "tfinance": {
        "path": "datasets/tfinance",
        "hsd_invert": True,
    },
    "tsocial": {
        "path": "datasets/tsocial",
        "hsd_invert": False,
    },
}


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
    """Load dataset from .mat file with ``features``, ``label``, and ``homo``.

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


def _ensure_dgl_importable() -> None:
    """Patch DGL graphbolt imports when the compiled extension is unavailable."""
    import sys
    import types

    stubs = [
        "dgl.graphbolt",
        "dgl.graphbolt.base",
        "dgl.graphbolt.dataloader",
        "dgl.graphbolt.impl",
        "dgl.graphbolt.impl.legacy_dataset",
        "dgl.graphbolt.impl.ondisk_dataset",
        "dgl.graphbolt.impl.ondisk_metadata",
    ]
    for name in stubs:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod


def _load_dgl_graphs(path: str):
    _ensure_dgl_importable()
    from dgl.data.utils import load_graphs

    return load_graphs(path)


def _prepare_dgl_features(graph) -> torch.Tensor:
    if "feature" not in graph.ndata:
        raise KeyError("Expected 'feature' in DGL node data")
    features = graph.ndata["feature"].float()
    n_raw_feat = int(features.shape[1])
    n_log_cols = n_raw_feat
    for col_idx in range(n_raw_feat):
        col = features[:, col_idx]
        if col.min() >= 0.0 and col.max() <= 1.0:
            n_log_cols = col_idx
            break
    if n_log_cols > 0:
        features = torch.cat(
            [torch.log1p(features[:, :n_log_cols]), features[:, n_log_cols:]],
            dim=1,
        )
    return features


def _prepare_dgl_labels(graph) -> torch.Tensor:
    if "label" not in graph.ndata:
        raise KeyError("Expected 'label' in DGL node data")
    raw = graph.ndata["label"]
    if raw.ndim == 2 and raw.shape[1] == 2:
        return raw[:, 1].long()
    return raw.long().view(-1)


def _compute_hsd_chunked(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    chunk_size: int = 250_000,
) -> torch.Tensor:
    num_nodes = int(x.shape[0])
    num_edges = int(edge_index.shape[1])
    if num_edges == 0:
        return torch.zeros(num_nodes, dtype=x.dtype)

    row, col = edge_index
    sums = torch.zeros(num_nodes, dtype=x.dtype)
    counts = torch.zeros(num_nodes, dtype=x.dtype)
    chunk = max(int(chunk_size), 1)
    for start in range(0, num_edges, chunk):
        end = min(start + chunk, num_edges)
        row_chunk = row[start:end]
        col_chunk = col[start:end]
        dist = torch.norm(x[row_chunk] - x[col_chunk], p=2, dim=1)
        sums.index_add_(0, row_chunk, dist)
        counts.index_add_(0, row_chunk, torch.ones_like(dist))
    return torch.nan_to_num(sums / counts.clamp_min(1.0), nan=0.0)


def load_from_dgl(
    path: str | Path,
    hsd_invert: bool = False,
    hsd_chunk_size: int = 250_000,
    append_hsd: bool = True,
) -> Data:
    """Load PriorF-GNN DGL single-relation datasets as PyG ``Data``.

    This mirrors the PriorF-GNN DGL loader for ``tfinance`` and ``tsocial``:
    raw count features receive a ``log1p`` transform, labels are converted from
    one-hot if needed, HSD is computed score-blindly from the single relation,
    and HSD is appended as the last feature column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DGL dataset not found: {path}")
    graphs, _ = _load_dgl_graphs(str(path))
    if not graphs:
        raise ValueError(f"No graphs found in DGL dataset: {path}")
    graph = graphs[0]

    features = _prepare_dgl_features(graph)
    labels = _prepare_dgl_labels(graph)
    src, dst = graph.edges()
    edge_index = torch.stack([src.long(), dst.long()], dim=0)

    if append_hsd:
        hsd = _compute_hsd_chunked(features, edge_index, chunk_size=hsd_chunk_size)
        if hsd_invert and hsd.numel() > 0 and hsd.max() > 0:
            hsd = hsd.max() - hsd
        x = torch.cat([features, hsd.unsqueeze(1)], dim=1)
    else:
        hsd = torch.zeros(features.shape[0], dtype=features.dtype)
        x = features
    data = Data(
        x=x,
        edge_index=edge_index,
        y=labels,
        hsd=hsd,
        edge_type=torch.zeros(edge_index.shape[1], dtype=torch.long),
    )
    return data


_LOADERS = {
    ".pt": load_from_pt,
    ".pkl": load_from_pkl,
    ".npz": load_from_npz,
    ".mat": load_from_mat,
    ".dgl": load_from_dgl,
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
    hsd_invert: bool | None = None,
    hsd_chunk_size: int = 250_000,
    append_hsd: bool = True,
) -> Data:
    """Load a graph fraud detection dataset.

    Supported datasets:
    - tiny: synthetic tiny graph (50 nodes) for testing
    - synthetic_small: synthetic graph (500 nodes)
    - yelpchi, yelpnyc, yelpzip, amazon: real datasets from .mat files
    - tfinance, tsocial: real DGL single-relation datasets
    - Custom: load from path with specified format

    Args:
        name: Dataset name or 'custom'.
        path: Path to dataset file (required for custom).
        format: File format (pt/pkl/npz/mat/dgl). Auto-detected if None.
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

    name = name.lower()

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
    elif name in REAL_MAT_DATASETS:
        if path is None:
            path = REAL_MAT_DATASETS[name]
        data = load_from_mat(path)
        if stratified:
            data = stratified_split(data, seed=seed, ratios=ratios)
        else:
            data = generate_masks(data, seed=seed, ratios=ratios)
        save_split(data, name, seed, split_mode, train_ratio, val_test_ratio, stratified=stratified)
    elif name in REAL_DGL_DATASETS:
        defaults = REAL_DGL_DATASETS[name]
        if path is None:
            path = defaults["path"]
        invert = bool(defaults["hsd_invert"] if hsd_invert is None else hsd_invert)
        data = load_from_dgl(
            path,
            hsd_invert=invert,
            hsd_chunk_size=hsd_chunk_size,
            append_hsd=append_hsd,
        )
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
        if ext == ".dgl":
            data = loader(
                path,
                hsd_invert=bool(hsd_invert),
                hsd_chunk_size=hsd_chunk_size,
                append_hsd=append_hsd,
            )
        else:
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
