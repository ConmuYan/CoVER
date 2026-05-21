"""Regenerate PriorF-GNN-style unified data with RAER-FD's 40/20/40 stratified split.

Produces a drop-in unified data file with identical layout to
PriorF-GNN/processed_data/yelpchi/seed_*/data.pt EXCEPT the masks are
generated using RAER-FD's 40/20/40 split protocol (train=0.4,
val=0.2, test=0.4, stratified by label) so we can run PriorF-GNN at
the same data split as the RAER-FD BWGNN baseline.

Output: datasets/unified_404020/<dataset>/seed_<S>/data.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_scatter import scatter_mean


YELP_RELS = {"net_rur": "rur", "net_rtr": "rtr", "net_rsr": "rsr"}
AMAZON_RELS = {"net_upu": "upu", "net_usu": "usu", "net_uvu": "uvu"}


def sparse_to_edge_index(adj: sp.spmatrix) -> torch.Tensor:
    coo = adj.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    ei = torch.stack([row, col], dim=0)
    return ei[:, ei[0] != ei[1]]


def compute_hsd(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.size(1) == 0:
        return torch.zeros(x.size(0))
    row, col = edge_index
    dist = torch.norm(x[row] - x[col], p=2, dim=1)
    hsd = scatter_mean(dist, row, dim=0, dim_size=x.size(0))
    return torch.nan_to_num(hsd, nan=0.0)


def stratified_404020(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """40/20/40 stratified split (RAER-FD protocol).

    Mirrors data/split.py:stratified_split semantics:
      - Step 1: train (40%) vs rest (60%), stratify by y
      - Step 2: from rest, val:test = 1:2 so val is 20% and test is 40%
    """
    n = len(y)
    idx = np.arange(n)
    train_idx, rest_idx = train_test_split(
        idx, train_size=0.4, stratify=y, random_state=seed,
    )
    y_rest = y[rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=1.0 / 3.0, stratify=y_rest, random_state=seed,
    )
    return train_idx, val_idx, test_idx


def build(mat_path: Path, dataset: str, seed: int, out_dir: Path) -> Path:
    mat = sio.loadmat(str(mat_path))
    rel_map = YELP_RELS if dataset == "yelpchi" else AMAZON_RELS

    features = mat["features"]
    if sp.issparse(features):
        features = features.toarray()
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(mat["label"].flatten().astype(np.int64))

    edge_index_dict: dict[str, torch.Tensor] = {}
    edge_indexes: list[torch.Tensor] = []
    for mat_key, rel_name in rel_map.items():
        ei = sparse_to_edge_index(mat[mat_key])
        edge_index_dict[rel_name] = ei
        edge_indexes.append(ei)

    union_edge = torch.cat(edge_indexes, dim=1)
    hsd = compute_hsd(x, union_edge)

    train_idx, val_idx, test_idx = stratified_404020(y.numpy(), seed)
    n = x.size(0)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        y=y,
        hsd=hsd,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        edge_index_dict=edge_index_dict,
    )

    out_path = out_dir / dataset / f"seed_{seed}" / "data.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)

    print(f"[unified-404020] {dataset} seed={seed}")
    print(f"  x: {tuple(x.shape)}  y: {tuple(y.shape)}  hsd: {tuple(hsd.shape)}")
    print(f"  train: {int(train_mask.sum())} ({y[train_mask].float().mean():.4f} fraud)")
    print(f"  val:   {int(val_mask.sum())} ({y[val_mask].float().mean():.4f} fraud)")
    print(f"  test:  {int(test_mask.sum())} ({y[test_mask].float().mean():.4f} fraud)")
    for k, ei in edge_index_dict.items():
        print(f"  rel {k}: {ei.size(1)} directed edges")
    print(f"  saved: {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True)
    parser.add_argument("--dataset", choices=["yelpchi", "amazon"], default="yelpchi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    build(Path(args.mat), args.dataset, args.seed, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
