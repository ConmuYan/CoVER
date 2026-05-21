# RAER-FD

语言：[English](README.md) | **中文**

**RAER-FD** 是 **Relation-Aware Evidence Reasoning and Residual
Distillation for Graph Fraud Detection** 的缩写，即面向图欺诈检测的
关系感知证据推理与残差蒸馏框架。

本仓库是 Work 2。Work 1 PriorF-GNN 将关系感知差异证据内化到强检测器中；
Work 2 RAER-FD 则把这类证据外化为一个可解释、有界、可蒸馏的残差纠正接口。

## 核心 Insight

图欺诈检测中的错误常常是关系条件化的：同一节点在某种关系下可能正常，在另一种
关系下可能异常。因此 RAER-FD 不再训练一个无约束的新分类器去覆盖 base，而是在
冻结 base 的前提下，只让 score-blind 的关系证据预测一个有界 residual：

```text
final_logit = base_logit + delta_rel
```

如果 base 已经很强，例如 PriorF-GNN 已经内部建模了关系差异证据，那么 RAER 的
near-identity 结果不是失败，而是强基座饱和性的 sanity check。

## 最终贡献

1. **RAER teacher**：在冻结 base detector 上进行关系感知证据推理，并通过结构设计
   保证 residual 有界。
2. **LREE**：可学习关系证据提取器。它不读取 base logit，只使用训练集标签构造原型，
   生成按关系组织的证据表示。
3. **CBR-Flash**：基于 contract-budgeted residual 的轻量蒸馏学生模型，蒸馏
   teacher 的最终残差策略，并保持同一个 residual contract。

## 规范结构

```text
configs/raer_fd/
  base_detectors/        # 冻结 base detector 配置
  large_graph/           # YelpNYC/YelpZip/TSocial 邻居 mini-batch 路径
  teacher/raer_hc/       # 手工关系证据 teacher
  teacher/raer_lree/     # LREE teacher
  strong_base/           # PriorF-GNN 强基座饱和性检查
  student/               # CBR-Flash
  ablations/             # 最终保留的消融

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

当前 active configs 覆盖 `yelpchi`、`amazon`、`yelpnyc`、`yelpzip`、
`tfinance` 和 `tsocial`。默认批量脚本只跑紧凑 full-graph 集合
（`yelpchi`、`amazon`）；`yelpnyc`、`yelpzip`、`tfinance`、`tsocial`
作为泛化/扩展配置保留，重跑前先参考 scaling plan。

其中三个大图的完整 Work 2 scaling 路径使用 compact relation basis 加
scalable LREE：

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> bash scripts/run_large_graph_raer_fd_sage_lree.sh yelpnyc 42 all
CUDA_VISIBLE_DEVICES=<空闲GPU> bash scripts/run_large_graph_raer_fd_sage_lree.sh yelpzip 42 all
CUDA_VISIBLE_DEVICES=<空闲GPU> bash scripts/run_large_graph_raer_fd_sage_lree.sh tsocial 42 all
```

`scripts/run_large_graph_raer_fd_sage.sh` 保留为 RAER-HC compact 控制组。

早期命名、旧配置、旧结果和负路线材料已经集中归档到：

```text
archive/legacy_raer_migration_20260520/
```

## 快速重跑

训练 base detectors：

```bash
scripts/run_compact_fullgraph_base_detectors_5seed.sh 0
```

训练 RAER-LREE teachers：

```bash
scripts/run_compact_fullgraph_raer_teachers_5seed.sh raer_lree 0
```

训练 RAER-HC 前先构建手工关系证据：

```bash
scripts/run_compact_fullgraph_relation_features_5seed.sh
```

训练 CBR-Flash students：

```bash
scripts/run_compact_fullgraph_cbr_flash_5seed.sh 0
```

单次 RAER-LREE 示例：

```bash
/data1/mq/conda_envs/gread-core/bin/python scripts/train_raer_teacher.py \
  --config configs/raer_fd/teacher/raer_lree/yelpchi_bwgnn.yaml \
  --seed 42 \
  --device cuda:0 \
  --run_name raer_lree \
  --base_ckpt_path artifacts/checkpoints/yelpchi/bwgnn/base/seed_42/base.pt
```

单次 CBR-Flash 示例：

```bash
/data1/mq/conda_envs/gread-core/bin/python scripts/train_cbr_flash.py \
  --config configs/raer_fd/student/cbr_flash_yelpchi_bwgnn.yaml \
  --teacher_ckpt artifacts/checkpoints/yelpchi/bwgnn/raer_lree/seed_42/raer_teacher.pt \
  --teacher_extractor_ckpt artifacts/checkpoints/yelpchi/bwgnn/raer_lree/seed_42/lree.pt \
  --base_ckpt_path artifacts/checkpoints/yelpchi/bwgnn/base/seed_42/base.pt \
  --seed 42 \
  --device cuda:0
```

## 验证

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

## 文档

重跑计划见 `docs/plans/RAER_FD_RERUN_PLAN.md`，大图显存与采样建议见
`docs/plans/RAER_FD_DATASET_SCALING_PLAN.md`，大图执行路径见
`docs/plans/RAER_FD_LARGE_GRAPH_RUN_PLAN.md`，本次迁移归档记录见
`docs/cleanup/RAER_MIGRATION_20260520.md`。
