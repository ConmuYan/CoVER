# SAGE (GraphSAGE) Paper Alignment

## References

- **Hamilton, Ying & Leskovec (2017)**. *Inductive Representation Learning on Large Graphs*. NeurIPS 2017.
  - arXiv: [1706.02216](https://arxiv.org/abs/1706.02216)
  - Stanford PDF: <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>
- **Official GitHub**: [williamleif/GraphSAGE](https://github.com/williamleif/GraphSAGE)
  - Local reference clone: `external/williamleif_GraphSAGE/` (reference-only; never imported)
- **Tang et al. (2022)**. *Rethinking Graph Neural Networks for Anomaly Detection*. ICML 2022.
  - BWGNN paper; uses SAGE as a supervised baseline for fraud detection comparison.

## Hyperparameters Used in This Project

| Parameter | Value | Justification |
|---|---|---|
| `hidden_dim` | 64 | Hamilton et al. Table 2 (Cora/PPI); matches BWGNN baseline hidden for fair comparison |
| `num_layers` | 2 | Paper default (K=2 hops); same as BWGNN `num_layers` |
| `dropout` | 0.5 | Paper Section 3.3 / Algorithm 1 default; differs from BWGNN 0.3 — kept at paper default |
| `lr` | 0.01 | Standard GNN learning rate; matches BWGNN for fair comparison |
| `weight_decay` | 5e-4 | Standard L2 regularisation; matches BWGNN |
| `optimizer` | Adam | Paper default; matches BWGNN |
| `epochs` | 100 | Sufficient for convergence with patience=100; same as BWGNN |
| `train_ratio` | 0.4 | Supervised split; matches BWGNN evaluation protocol |
| `val_test_ratio` | [1, 2] | 1:2 val/test; matches BWGNN |
| `stratified` | true | Class-balanced splits; matches BWGNN |
| `select_metric` | macro_f1 | Primary selection metric for base model; matches BWGNN |
| `patience` | 100 | Early stopping; matches BWGNN |
| `aggregator` | mean | Hamilton et al. Section 3.3 default; `SAGEConv(aggr='mean')` in PyG |
| `activation` | ReLU | Paper default nonlinearity σ; standard in PyG SAGEConv |

## Implementation

The in-repo implementation (`models/gnn.py:SAGEDetector`) uses `torch_geometric.nn.SAGEConv`
with mean aggregation, 2-layer stack, hidden=64, dropout=0.5, and a linear classification head.
This matches the supervised variant described in Hamilton et al. 2017, Section 3.3.

No code from `external/williamleif_GraphSAGE/` is imported — the clone exists solely for
hyperparameter audit trail and reproducibility reference.

## Differences from BWGNN

| Parameter | SAGE | BWGNN | Reason |
|---|---|---|---|
| `dropout` | 0.5 | 0.3 | SAGE paper default vs BWGNN paper default |
| `num_bands` | N/A | 3 | BWGNN-specific spectral parameter |
| `agg` | N/A | concat | BWGNN-specific band aggregation |

All other training hyperparameters (lr, weight_decay, epochs, patience, split) are identical
to ensure a fair same-seed comparison between SAGE and BWGNN base models.

## External Clone Status

The reference clone `external/williamleif_GraphSAGE/` was successfully cloned from
`https://github.com/williamleif/GraphSAGE.git`. It is used for hyperparameter audit only.
