# RAER-FD Dataset Scaling Plan

This note records the current dataset adaptation and the full-graph feasibility
decision for RAER-FD reruns.

## Dataset Families

| Dataset | Format | Relations | Nodes | Edges used by base graph | Features | Fraud rate |
|---|---:|---:|---:|---:|---:|---:|
| YelpChi | `.mat` | RUR, RSR, RTR | 45,954 | 7,693,958 | 32 | mat label |
| Amazon | `.mat` | UPU, USU, UVU | 11,944 | 8,796,784 | 25 | mat label |
| YelpNYC | `.mat` | RUR, RSR, RTR | 359,052 | 154,305,340 | 32 | mat label |
| YelpZip | `.mat` | RUR, RSR, RTR | 608,598 | 186,459,864 | 32 | mat label |
| T-Finance | DGL | EDGE | 39,357 | 42,445,086 | 10 + HSD | 4.58% |
| T-Social | DGL | EDGE | 5,781,065 | 146,211,016 | 10 + HSD | 3.01% |

T-Finance and T-Social follow the PriorF-GNN DGL preprocessing convention:
raw count features are transformed with `log1p`, HSD is appended as the last
feature column, and T-Finance uses `hsd_invert=true`.

## Full-Graph Memory Lower Bounds

The table below is a lower bound, not an expected peak. It counts only common
tensors:

- `edge_index`: two int64 arrays, `16 * E` bytes.
- sparse COO adjacency: `edge_index + float32 values`, `20 * E` bytes.
- one message tensor at hidden size 64: `E * 64 * 4` bytes.

| Dataset | `edge_index` | sparse COO | one 64-d message tensor | Full-graph base training |
|---|---:|---:|---:|---|
| YelpChi | 0.11 GiB | 0.14 GiB | 1.83 GiB | feasible on normal GPUs |
| Amazon | 0.13 GiB | 0.16 GiB | 2.10 GiB | feasible but dense |
| YelpNYC | 2.30 GiB | 2.87 GiB | 36.79 GiB | not recommended |
| YelpZip | 2.78 GiB | 3.47 GiB | 44.46 GiB | not recommended |
| T-Finance | 0.63 GiB | 0.79 GiB | 10.12 GiB | model-dependent, risky for GAT/BWGNN |
| T-Social | 2.18 GiB | 2.72 GiB | 34.86 GiB | not recommended |

GAT and BWGNN can materialize additional edge-level tensors, attention scores,
or band-pass intermediates; actual peak memory can be several times the lower
bound. LREE full-batch also stores sparse normalized adjacencies and neighbor
means, so large-edge datasets should not be treated as ordinary full-graph
runs.

## Rerun Policy

Use the default all-datasets shell scripts only for the compact canonical set:

```text
yelpchi, amazon
```

Use explicit commands for the scaling/generalization datasets:

```text
yelpnyc, yelpzip, tfinance, tsocial
```

Recommended strategy:

| Dataset | Base detector training | RAER-HC evidence | RAER-LREE teacher | CBR-Flash |
|---|---|---|---|---|
| YelpChi | full graph | full graph | full graph | full graph |
| Amazon | full graph | full graph | full graph | full graph |
| YelpNYC | neighbor mini-batch | CPU/chunked | sampled or precomputed | full after teacher cache |
| YelpZip | neighbor mini-batch | CPU/chunked | sampled or precomputed | full after teacher cache |
| T-Finance | full graph only on large-memory GPU; otherwise neighbor mini-batch | CPU/chunked | sampled or precomputed | full after teacher cache |
| T-Social | neighbor mini-batch | CPU/chunked | sampled or precomputed | sampled or chunked |

The active RAER-FD code can now load all six datasets and provides configs for
all six. The large-dataset configs are intentionally not included in the
default batch scripts until a mini-batch base trainer and sampled LREE path are
added.
