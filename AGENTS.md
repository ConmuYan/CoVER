# AGENTS.md

This repository is **RAER-FD**:
**Relation-Aware Evidence Reasoning and Residual Distillation for Graph Fraud
Detection**.

## Canonical Story

- **PriorF-GNN** is Work 1. It internalizes relation-aware discrepancy evidence
  inside a strong graph fraud detector.
- **RAER-FD** is Work 2. It externalizes relation-aware evidence as a bounded,
  interpretable, and distillable residual correction interface.
- On a strong PriorF-GNN base, a near-identity RAER result is a saturation
  sanity check: the base already captures much of the relation-aware evidence.

## Contributions

1. **RAER teacher**
   Relation-aware evidence reasoning over a frozen base detector. The teacher
   predicts only a bounded residual:
   `final_logit = base_logit + delta_rel`.

2. **LREE**
   Learnable relation evidence extractor. It is score-blind, computes prototype
   features from train labels only, and does not access base logits.

3. **CBR-Flash**
   Contract-budgeted residual distillation student. It distills the teacher's
   final residual policy under the same residual contract.

## Canonical Paths

| Component | Path |
|---|---|
| Base detector trainer | `scripts/train_base_detector.py` |
| Large base trainer | `scripts/train_base_detector_minibatch.py` |
| RAER teacher | `models/raer_teacher.py` |
| LREE | `evidence/lree.py` |
| Scalable LREE | `evidence/scalable_lree.py` |
| CBR-Flash student | `models/cbr_flash_adapter.py` |
| RAER teacher trainer | `scripts/train_raer_teacher.py` |
| PriorF-GNN RAER trainer | `scripts/train_raer_priorfgnn.py` |
| CBR-Flash trainer | `scripts/train_cbr_flash.py` |
| RAER-FD configs | `configs/raer_fd/` |
| Rerun plan | `docs/plans/RAER_FD_RERUN_PLAN.md` |
| Dataset scaling plan | `docs/plans/RAER_FD_DATASET_SCALING_PLAN.md` |
| Large-graph run plan | `docs/plans/RAER_FD_LARGE_GRAPH_RUN_PLAN.md` |
| Migration manifest | `docs/cleanup/RAER_MIGRATION_20260520.md` |

Historical code, configs, and results are archived under:

```text
archive/legacy_raer_migration_20260520/
```

Do not reintroduce archived naming into active code, configs, run names, tables,
or paper-facing text.

## Invariants

- Keep base detectors frozen in RAER and CBR-Flash experiments.
- Keep evidence score-blind: no base logits as evidence-extractor input.
- Keep prototype evidence train-only.
- Keep residuals bounded by architecture, not by post-hoc clipping.
- Preserve seed, split, dataset, and base-model provenance in every retained
  artifact.
- Active result files use `test_metrics.json`, `raer_summary.json`,
  `cbr_flash_summary.json`, and explicit checkpoint names.
- Default compact full-graph batch scripts cover `yelpchi` and `amazon` only.
  Use the dataset scaling plan before running `yelpnyc`, `yelpzip`,
  `tfinance`, or `tsocial`.
- `yelpnyc`, `yelpzip`, and `tsocial` should use the large-graph path first:
  `scripts/train_base_detector_minibatch.py` plus compact relation basis.
  Full Work 2 scaling uses `raer_lree_scalable` and
  `cbr_flash_lree_scalable`; `raer_hc_compact` is a control path. Do not route
  these datasets through the compact full-graph batch scripts.

## Validation Commands

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

## Editing Policy

- Prefer canonical RAER-FD names in new files.
- Archive historical material instead of deleting it.
- Do not overwrite completed checkpoints, logs, or tables unless the user
  explicitly requests regeneration.
- Do not rewrite dataset paths in archived configs.
