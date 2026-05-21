# RAER-FD

Language: **English** | [中文](README.zh-CN.md)

**RAER-FD** stands for **Relation-Aware Evidence Reasoning and Residual
Distillation for Graph Fraud Detection**.

This repository is Work 2. Work 1, PriorF-GNN, internalizes relation-aware
discrepancy evidence inside a strong detector. RAER-FD externalizes that
principle as an interpretable, bounded, and distillable residual correction
interface over frozen graph fraud detectors.

## Core Insight

Graph fraud errors are often relation-conditioned: a node can look benign
under one relation and suspicious under another. RAER-FD therefore does not
replace a base detector with another unconstrained classifier. It learns a
bounded residual only when score-blind relation evidence supports a correction:

```text
final_logit = base_logit + delta_rel
```

On a saturated relation-aware base such as PriorF-GNN, a near-identity RAER
result is expected and meaningful: it shows the base already internalizes much
of the relation evidence.

## Contributions

1. **RAER teacher**: relation-aware evidence reasoning over a frozen base
   detector with an architectural residual bound.
2. **LREE**: a learnable relation evidence extractor that is score-blind and
   uses train-label prototypes only from the training split.
3. **CBR-Flash**: a contract-budgeted residual distillation student that
   distills the teacher's final residual policy while preserving the same
   residual contract.

## Canonical Layout

```text
configs/raer_fd/
  base_detectors/        # frozen base detector configs
  large_graph/           # YelpNYC/YelpZip/TSocial neighbor mini-batch path
  teacher/raer_hc/       # hand-crafted evidence teacher configs
  teacher/raer_lree/     # LREE teacher configs
  strong_base/           # PriorF-GNN saturation check
  student/               # CBR-Flash configs
  ablations/             # final ablations only

models/
  raer_teacher.py
  cbr_flash_adapter.py
  priorfgnn.py

evidence/
  relation_features.py
  lree.py
  scalable_lree.py

scripts/
  train_base_detector.py
  train_base_detector_minibatch.py
  train_raer_teacher.py
  train_raer_priorfgnn.py
  train_cbr_flash.py
  run_large_graph_raer_fd_sage.sh
  run_large_graph_raer_fd_sage_lree.sh
  run_compact_fullgraph_base_detectors_5seed.sh
  run_compact_fullgraph_relation_features_5seed.sh
  run_compact_fullgraph_raer_teachers_5seed.sh
  run_compact_fullgraph_cbr_flash_5seed.sh
  aggregate_cbr_flash.py
```

Active configs cover `yelpchi`, `amazon`, `yelpnyc`, `yelpzip`, `tfinance`,
and `tsocial`. The default batch scripts intentionally run only the compact
full-graph set (`yelpchi`, `amazon`). Use the scaling plan before rerunning
`yelpnyc`, `yelpzip`, `tfinance`, or `tsocial`.

For the three large datasets, the full Work 2 scaling path uses the compact
relation basis plus scalable LREE:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> bash scripts/run_large_graph_raer_fd_sage_lree.sh yelpnyc 42 all
CUDA_VISIBLE_DEVICES=<idle_gpu> bash scripts/run_large_graph_raer_fd_sage_lree.sh yelpzip 42 all
CUDA_VISIBLE_DEVICES=<idle_gpu> bash scripts/run_large_graph_raer_fd_sage_lree.sh tsocial 42 all
```

`scripts/run_large_graph_raer_fd_sage.sh` remains as the RAER-HC compact
control path.

Historical code, configs, logs, and result artifacts from earlier naming
rounds are archived under:

```text
archive/legacy_raer_migration_20260520/
```

## Quick Start

Train frozen base detectors:

```bash
scripts/run_compact_fullgraph_base_detectors_5seed.sh 0
```

Train RAER-LREE teachers:

```bash
scripts/run_compact_fullgraph_raer_teachers_5seed.sh raer_lree 0
```

Build hand-crafted relation features before RAER-HC runs:

```bash
scripts/run_compact_fullgraph_relation_features_5seed.sh
```

Train CBR-Flash students:

```bash
scripts/run_compact_fullgraph_cbr_flash_5seed.sh 0
```

Single-run RAER-LREE example:

```bash
/data1/mq/conda_envs/gread-core/bin/python scripts/train_raer_teacher.py \
  --config configs/raer_fd/teacher/raer_lree/yelpchi_bwgnn.yaml \
  --seed 42 \
  --device cuda:0 \
  --run_name raer_lree \
  --base_ckpt_path artifacts/checkpoints/yelpchi/bwgnn/base/seed_42/base.pt
```

Single-run CBR-Flash example:

```bash
/data1/mq/conda_envs/gread-core/bin/python scripts/train_cbr_flash.py \
  --config configs/raer_fd/student/cbr_flash_yelpchi_bwgnn.yaml \
  --teacher_ckpt artifacts/checkpoints/yelpchi/bwgnn/raer_lree/seed_42/raer_teacher.pt \
  --teacher_extractor_ckpt artifacts/checkpoints/yelpchi/bwgnn/raer_lree/seed_42/lree.pt \
  --base_ckpt_path artifacts/checkpoints/yelpchi/bwgnn/base/seed_42/base.pt \
  --seed 42 \
  --device cuda:0
```

## Validation

```bash
/data1/mq/conda_envs/gread-core/bin/python -m py_compile \
  models/raer_teacher.py evidence/lree.py evidence/scalable_lree.py models/cbr_flash_adapter.py \
  scripts/train_base_detector.py scripts/train_raer_teacher.py \
  scripts/train_raer_priorfgnn.py scripts/train_cbr_flash.py \
  scripts/train_base_detector_minibatch.py
```

```bash
/data1/mq/conda_envs/gread-core/bin/pytest -q \
  tests/test_raer_teacher.py \
  tests/test_raer_teacher_cbr_heads.py \
  tests/test_cbr_flash_contracts.py \
  tests/test_scalable_lree.py \
  tests/test_raer_losses.py
```

## Documentation

See `docs/plans/RAER_FD_RERUN_PLAN.md` for the rerun workflow and
`docs/plans/RAER_FD_DATASET_SCALING_PLAN.md` for large-dataset memory and
sampling guidance. The large-graph execution path is documented in
`docs/plans/RAER_FD_LARGE_GRAPH_RUN_PLAN.md`. The archive manifest is
`docs/cleanup/RAER_MIGRATION_20260520.md`.
