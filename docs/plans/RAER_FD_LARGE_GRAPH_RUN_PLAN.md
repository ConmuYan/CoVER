# RAER-FD Large-Graph Run Plan

This plan covers the Work 2 scaling path for `yelpnyc`, `yelpzip`, and
`tsocial` while existing compact full-graph jobs are left undisturbed.

## Scope

- Base detector: SAGE with CPU neighbor mini-batch sampling.
- Relation basis: compact score-blind relation evidence cache.
- Main teacher: RAER with scalable LREE.
- Student: CBR-Flash distilled from the scalable LREE teacher.
- Control teacher: RAER-HC compact relation evidence.

The key change from the first large-graph bridge is that LREE is now scalable:
it learns over cached relation basis features instead of preparing full
per-relation sparse adjacencies on GPU.

## Main Entrypoint

Full Work 2 scaling:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> bash scripts/run_large_graph_raer_fd_sage_lree.sh yelpnyc 42 all
CUDA_VISIBLE_DEVICES=<idle_gpu> bash scripts/run_large_graph_raer_fd_sage_lree.sh yelpzip 42 all
CUDA_VISIBLE_DEVICES=<idle_gpu> bash scripts/run_large_graph_raer_fd_sage_lree.sh tsocial 42 all
```

Single stages:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> bash scripts/run_large_graph_raer_fd_sage_lree.sh yelpnyc 42 base
CUDA_VISIBLE_DEVICES=<idle_gpu> bash scripts/run_large_graph_raer_fd_sage_lree.sh yelpnyc 42 rel
CUDA_VISIBLE_DEVICES=<idle_gpu> bash scripts/run_large_graph_raer_fd_sage_lree.sh yelpnyc 42 teacher
CUDA_VISIBLE_DEVICES=<idle_gpu> bash scripts/run_large_graph_raer_fd_sage_lree.sh yelpnyc 42 student
```

RAER-HC compact control:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> bash scripts/run_large_graph_raer_fd_sage.sh yelpnyc 42 all
```

## Naming

- Base checkpoint:
  `artifacts/checkpoints/{dataset}/sage/base_neighbor_mb/seed_{seed}/base.pt`
- Base output cache:
  `artifacts/base_outputs/{dataset}/sage/seed_{seed}/_override_base_neighbor_mb_seed_{seed}_base.pt`
- Relation basis:
  `artifacts/relation_features/{dataset}/sage/seed_{seed}/all/rel_stats.pt`
- Scalable LREE teacher:
  `artifacts/checkpoints/{dataset}/sage/raer_lree_scalable/seed_{seed}/raer_teacher.pt`
- Scalable LREE extractor:
  `artifacts/checkpoints/{dataset}/sage/raer_lree_scalable/seed_{seed}/scalable_lree.pt`
- CBR-Flash student:
  `artifacts/checkpoints/{dataset}/sage/cbr_flash_lree_scalable/seed_{seed}/cbr_flash_student.pt`
- Control teacher:
  `artifacts/checkpoints/{dataset}/sage/raer_hc_compact/seed_{seed}/raer_teacher.pt`

## Dataset Defaults

| Dataset | Base Training | Fanout | Batch | Work 2 Teacher |
|---|---:|---:|---:|---|
| YelpNYC | CPU neighbor mini-batch | `[15, 10]` | 1024 | `raer_lree_scalable` over RUR/RSR/RTR |
| YelpZip | CPU neighbor mini-batch | `[10, 5]` | 1024 | `raer_lree_scalable` over RUR/RSR/RTR |
| TSocial | CPU neighbor mini-batch | `[15, 10, 5]` | 512 | `raer_lree_scalable` over EDGE |

The sampler expands incoming neighbors by default (`sampling_direction: in`) so
the sampled context follows PyG message flow into root nodes. It does not add
reverse message edges unless a config explicitly sets `add_reverse_edges: true`.

## Why This Scales

- The base detector uses CPU-side neighbor sampling and writes base logits plus
  embeddings once.
- Relation evidence is cached as compact score-blind basis features.
- Scalable LREE learns a small per-relation transform over that cache in
  indexed batches.
- RAER teacher validation, diagnostics, and CBR teacher-cache generation are
  chunked.
- CBR-Flash selection and evaluation are chunked, while the optimization step
  still uses the selected contract-budgeted node set.

## Expected Memory Shape

The dominant persistent tensors are base embeddings, base logits, relation
basis/features, and teacher cache. With 64-dim base embeddings and 9-dim
per-relation evidence, this is practical on a 24GB GPU for the intended
large-graph path. If TSocial hits memory pressure, first reduce:

```yaml
raer_teacher:
  feature_batch_size: 131072
  eval_batch_size: 131072
cbr_flash:
  teacher_cache_batch_size: 131072
  full_batch_size: 131072
  eval_batch_size: 131072
```

Do not fall back to online full-graph LREE adjacency extraction for these
datasets.
