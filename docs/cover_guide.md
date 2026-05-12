# CoVER-FD 实施指南：Agent 协助下的科研工程工作流

---

## 0. 开发原则

新建一个轻量科研仓库：

```bash
mkdir cover-fd
cd cover-fd
git init
```

核心原则：

```text
能训练 > 工程完整
能复现 > 模块包装
能做消融 > 花哨架构
能保存日志 > 口头结果
能跑 baseline > 只跑 ours
```

方法主线保持三阶段：Base GNN training → offline evidence generation / verification → evidence distillation。CoVER 方案明确采用这三阶段，并强调 LLM 只作为离线 evidence generator，推理阶段不调用 LLM。

---

## 1. 本地目录结构

```text
cover-fd/
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   ├── yelpchi_bwgnn.yaml
│   ├── amazon_bwgnn.yaml
│   ├── yelpchi_gcn.yaml
│   └── ablation.yaml
├── data/
│   ├── __init__.py
│   ├── load_fraud.py
│   └── split.py
├── models/
│   ├── __init__.py
│   ├── gnn.py
│   ├── bwgnn.py
│   └── reasoner.py
├── evidence/
│   ├── __init__.py
│   ├── schema.py
│   ├── adapter.py
│   ├── contracts.yaml
│   ├── verifier.py
│   ├── prompt.py
│   └── llm_teacher.py
├── training/
│   ├── __init__.py
│   ├── losses.py
│   ├── metrics.py
│   └── logger.py
├── scripts/
│   ├── train_stage1.py
│   ├── generate_stage2_err.py
│   ├── train_stage3.py
│   ├── evaluate.py
│   ├── run_scarcity.py
│   └── run_ablation.py
├── tests/
│   ├── test_schema.py
│   ├── test_verifier.py
│   ├── test_reasoner_shapes.py
│   └── test_score_blind.py
└── artifacts/
    ├── checkpoints/
    ├── err_cache/
    ├── logs/
    └── results/
```

`LinguGKD` 的关键启发不是照搬其微调流程，而是借它的 **teacher 输出缓存 + student 蒸馏 + 轻量实验组织**。LinguGKD 论文中明确采用 decoupled stages：先得到 teacher，再缓存 teacher features，最后训练轻量 GNN student。项目也应严格缓存 ERR，训练时不要在线调 LLM。

---

## 2. AGENTS.md：约束 Codex / Coding Agent

在仓库根目录创建 `AGENTS.md`，约束 agent 的行为边界：

````markdown
# AGENTS.md

## Project Goal

This repository implements CoVER-FD: Contract-Verified Evidence Distillation for LLM-Free Fake Review Detection.

The main task is binary graph fraud detection on YelpChi and Amazon. The target pipeline has three stages:

1. Train a base GNN detector.
2. Generate score-blind structural evidence cards, ask an offline LLM teacher to produce ERR records, and verify them with deterministic evidence contracts.
3. Train an evidence-conditioned student reasoner with supervised detection loss and accepted ERR distillation loss.

Do not build a large framework. Keep the code small, explicit, and research-friendly.

## Key Method Constraints

- LLM is offline only.
- No LLM call during model training except the explicit Stage 2 generation script.
- No LLM call during inference.
- The LLM prompt must not contain base_score, probability, logit, confidence, or raw prediction labels.
- ERR summary is for human inspection only and must not be used in loss.
- Rejected ERR records must not contribute to evidence distillation loss.
- If accepted ERR count is zero in a batch, fall back to supervised task loss.
- rho=0 in the reasoner must make final_logit equal base_logit.

## Coding Style

- Prefer simple PyTorch / PyG code.
- Avoid unnecessary abstraction.
- Each module should be under 300 lines unless unavoidable.
- Use type hints for public functions.
- Add docstrings only where they clarify non-obvious logic.
- Use deterministic seeds.
- Save config, seed, git hash, metrics, and checkpoint path for every run.

## Required Tests

Before claiming a task is complete, run:

```bash
pytest -q
python scripts/train_stage1.py --config configs/yelpchi_gcn.yaml --debug
python scripts/generate_stage2_err.py --config configs/yelpchi_gcn.yaml --teacher rule --debug
python scripts/train_stage3.py --config configs/yelpchi_gcn.yaml --debug
python scripts/evaluate.py --config configs/yelpchi_gcn.yaml --debug
```

If a command fails, report the exact error and the file/function likely responsible.

## External Repositories

Reference repositories may be cloned into `external/`, but do not directly copy large code blocks unless the license permits it.

Use external code only for:

- understanding data format
- checking model architecture
- reproducing baseline behavior
- comparing training scripts

Our own implementation should live in this repository.

## Expected Deliverables Per Task

For every coding task, provide:

1. Files changed.
2. What was implemented.
3. How it was tested.
4. Remaining limitations.
5. Next recommended step.
````

该文件可防止 agent 做三类坏事：复制大段外部代码、训练时偷偷调用 LLM、为了"完整性"把项目包装成巨大框架。

---

## 3. 克隆参考仓库

```bash
mkdir external
cd external

git clone https://github.com/ConmuYan/gread-core.git
git clone https://github.com/SXiangHu/LinguGKD.git

cd ..
```

让 Agent 先读，不要改：

```text
请只阅读 external/gread-core 和 external/LinguGKD，不要修改任何文件。
输出一份 repo understanding report，包含：
1. 两个仓库的入口脚本、数据加载方式、模型定义位置、训练流程。
2. 哪些代码/思想可以复用，哪些不能复用。
3. 对 cover-fd 的最小实现建议。
4. 列出每个可复用模块对应的我们项目文件。
不要写代码。
```

要求 agent 区分复用层级：

```text
复用思想
复用接口
复用超参数
复用文件结构
直接复制代码
```

直接复制代码一般放最后，能不复制就不复制。

---

## 4. 编码分 6 个小任务

不要让 Agent 一次做完整系统，按以下 6 步逐步推进。

### Task 1：数据加载与 baseline skeleton

给 Agent 的 prompt：

```text
实现 cover-fd 的第一步：数据加载和最小 GNN baseline。

已知我会提供 YelpChi / Amazon 的数据路径。请先写可配置的数据加载器，不要假设固定路径。

要求：
1. 实现 data/load_fraud.py，支持从 .pt/.pkl/.npz 三种常见格式加载：
   - x
   - edge_index
   - y
   - train_mask / val_mask / test_mask，如果没有则由 split.py 生成。
2. 实现 data/split.py，支持 fixed seed 随机划分与 scarcity ratio。
3. 实现 models/gnn.py，包含 GCN / GraphSAGE / GAT 三个 baseline。
4. 实现 scripts/train_stage1.py，训练二分类 fraud detector。
5. 实现 training/metrics.py，输出 ROC-AUC、AUPRC、F1、Precision@K、Recall@K。
6. 写 tests/test_data_loading.py 和 tests/test_reasoner_shapes.py 的最小测试。

不要实现 evidence、LLM、reasoner。
先保证 stage1 能跑通。
```

验收标准：

```bash
pytest -q
python scripts/train_stage1.py --config configs/yelpchi_gcn.yaml --debug
```

保存结果：

```text
artifacts/checkpoints/yelpchi/gcn/seed_0/base.pt
artifacts/logs/yelpchi/gcn/seed_0/stage1.json
artifacts/results/yelpchi/gcn/seed_0/stage1_metrics.json
```

---

### Task 2：Evidence schema + adapter + rule ERR

先不用 LLM，先用规则 teacher 跑通。

```text
实现 Stage 2 的 score-blind evidence card 与 rule teacher。

要求：
1. evidence/schema.py:
   - EvidenceCard
   - CalibrationChannel
   - ReasoningChannel
   - ERR
2. evidence/adapter.py:
   - 输入 x, edge_index, base_logits, z
   - 输出每个 trace node 的 EvidenceCard
   - calibration 包含 base_score, uncertainty
   - teacher_payload 必须只包含 reasoning channel
3. evidence/prompt.py:
   - build_teacher_payload(card)
   - assert payload 中不出现 score/logit/prob/confidence/base_score
4. scripts/generate_stage2_err.py:
   - 支持 --teacher rule
   - rule teacher 根据 evidence 生成 ERR
   - 保存 jsonl cache
5. tests/test_score_blind.py:
   - 检查 prompt 中没有 score 字段。
```

关键点不是 LLM，而是 **score-blind 边界**。CoVER 文档明确指出 calibration channel 只用于 trace selection，reasoning channel 才是 LLM 可见部分，LLM 不应看到 logits、probability 或 confidence。

---

### Task 3：Verifier

```text
实现 Evidence Contract Verifier。

要求：
1. evidence/contracts.yaml 定义 fraud reason types:
   - structural_discrepancy
   - camouflage_neighbor
   - spectral_anomaly
   - feature_structure_conflict
   - relation_or_burst_anomaly
   - weak_or_uncertain_evidence
2. evidence/verifier.py 实现：
   - schema validity
   - evidence availability
   - role consistency
   - task evidence contract
   - score-blindness check
   - optional label compatibility
3. generate_stage2_err.py 中加入 verifier。
4. 保存 accepted 和 rejected ERR：
   - artifacts/err_cache/.../accepted.jsonl
   - artifacts/err_cache/.../rejected.jsonl
   - artifacts/err_cache/.../verifier_stats.json
5. tests/test_verifier.py 覆盖：
   - valid ERR accept
   - unavailable field reject
   - counter_signal as supporting reject
   - score field reject
   - contract violation reject
```

这一步要做得很扎实，论文可信度就在 verifier。CoVER 方案中 verifier 保证 accepted ERR 至少满足 schema-valid、evidence-closed、role-consistent、score-blind、contract-consistent、label-compatible，但不声称因果忠实。

---

### Task 4：Reasoner + Stage 3 distillation

```text
实现 Stage 3 evidence-conditioned student reasoner。

要求：
1. models/reasoner.py:
   - EvidenceEncoder
   - ReasonTypeHead
   - PositiveEvidenceMaskHead
   - CounterEvidenceMaskHead
   - EvidenceGatedResidualReadout
2. forward 输入：
   - z
   - base_logit
   - evidence_token_ids
3. 输出：
   - final_logit
   - type_logits
   - pos_logits
   - neg_logits
4. training/losses.py:
   - supervised BCE loss on all train nodes
   - type CE loss only on accepted ERR nodes
   - positive mask BCE only on accepted ERR nodes
   - counter mask BCE only on accepted ERR nodes
   - no accepted ERR 时 evidence loss = 0
5. scripts/train_stage3.py:
   - load frozen base checkpoint
   - load accepted ERR cache
   - train reasoner
6. tests:
   - rho=0 final_logit == base_logit
   - rejected ERR 不参与 loss
   - accepted_mask 全 0 不报错
```

CoVER 方案中 student reasoner 的输出包括 final logits、reason type、positive evidence mask、negative evidence mask；并通过 evidence-gated residual readout 让证据影响最终预测，而不是只做解释附件。

---

### Task 5：LLM teacher

```text
实现 offline LLM teacher，不要改变训练脚本。

要求：
1. evidence/llm_teacher.py:
   - 支持 openai-compatible chat completion
   - 支持 local transformers generation 的占位接口
   - temperature=0
   - retry=3
   - JSON parse with repair fallback
2. generate_stage2_err.py 支持 --teacher llm
3. 所有 LLM 原始输出保存：
   - raw_outputs.jsonl
4. 所有 prompt 保存：
   - prompts.jsonl
5. 强制检查 prompt 中不包含 score/logit/prob/confidence/base_score。
6. 训练阶段禁止调用 llm_teacher.py。
```

---

### Task 6：实验脚本和质量报告

```text
实现实验自动化与质量检查。

要求：
1. scripts/run_scarcity.py:
   - train ratios: 5, 10, 20, 40, 100
   - seeds: 0,1,2,3,4
2. scripts/run_ablation.py:
   - full
   - no_verifier
   - no_counter
   - no_type_loss
   - no_evidence_loss
   - no_residual_gate
   - score_visible_prompt
   - rule_teacher
3. scripts/aggregate_results.py:
   - 输出 mean ± std
   - 保存 csv 和 markdown table
4. scripts/check_run_integrity.py:
   - 检查每个 run 是否有 config、seed、checkpoint、metrics、err_cache_hash。
```

DuoKD 的实验对正负知识做了 ablation，去掉 negative supervision 会导致性能下降，尤其在类不平衡场景中能减少非目标类过置信；这正好支持在 fraud detection 中做 `no_counter` 消融。

---

## 5. Agent 使用方式

### A. Reader Agent：只读仓库

```text
你是 repo understanding agent。只读 external/gread-core 和 external/LinguGKD，不要修改文件。
请输出：
- 项目结构
- 入口脚本
- 关键模型
- 数据格式
- 训练流程
- 可借鉴点
- 不应借鉴点
- 对 cover-fd 的实现建议
```

验收：只允许输出报告，不允许改代码。

### B. Builder Agent：一次只实现一个模块

```text
你是 builder agent。只实现 evidence/verifier.py 和 tests/test_verifier.py。
不要改其他文件，除非测试必须。
完成后运行 pytest -q tests/test_verifier.py。
```

验收：必须跑测试。

### C. Reviewer Agent：检查代码质量

```text
你是 reviewer agent。请审查当前 diff，不要改代码。
关注：
1. 是否违反 score-blind 约束。
2. 是否训练阶段调用了 LLM。
3. 是否 rejected ERR 进入 loss。
4. 是否存在数据泄漏。
5. 是否日志足以复现。
6. 是否模块过度抽象。
7. 是否有 shape bug 或 device bug。
给出 blocking issues 和 non-blocking suggestions。
```

比让同一个 Agent 又写又审更可靠。

---

## 6. Git 分支策略

```bash
git checkout -b 00-data-stage1
git checkout -b 01-evidence-card
git checkout -b 02-verifier
git checkout -b 03-reasoner
git checkout -b 04-llm-teacher
git checkout -b 05-experiments
```

每个分支只做一类事情。合并前要求：

```bash
pytest -q
python scripts/train_stage1.py --config configs/yelpchi_gcn.yaml --debug
python scripts/generate_stage2_err.py --config configs/yelpchi_gcn.yaml --teacher rule --debug
python scripts/train_stage3.py --config configs/yelpchi_gcn.yaml --debug
python scripts/evaluate.py --config configs/yelpchi_gcn.yaml --debug
```

---

## 7. 代码质量评价标准

科研代码不需要像工业项目一样复杂，但必须满足以下 10 条：

| # | 标准 | 说明 |
|---|------|------|
| 1 | Reproducibility | 每个 run 保存 config、seed、metrics、checkpoint、git hash |
| 2 | Data safety | train / val / test mask 固定保存，scarcity 只改变 train labels，不动 val/test |
| 3 | Score-blind safety | prompt 和 ERR 不出现 base_score、logit、probability、confidence |
| 4 | LLM-free inference | evaluate.py 不 import llm_teacher.py |
| 5 | Verifier determinism | 同一个 ERR + EvidenceCard，多次 verify 结果一致 |
| 6 | Loss correctness | accepted_mask 控制 evidence loss；rejected ERR 不进入 loss |
| 7 | Ablation controllability | 每个创新点都能通过 config 开关关闭 |
| 8 | Minimal dependency | PyTorch、PyG、sklearn、numpy、yaml、tqdm、transformers 足够 |
| 9 | Debug mode | `--debug` 只跑少量 epoch / 少量节点，5 分钟内发现 shape/device bug |
| 10 | Result aggregation | 自动输出 mean ± std，不手工复制表格 |

---

## 8. 第一天操作指南

第一天不要碰 LLM，目标是把 Stage 1 跑通。

操作顺序：

```bash
conda activate 你的环境名
cd cover-fd

mkdir -p configs data models evidence training scripts tests artifacts
touch README.md AGENTS.md requirements.txt
touch data/__init__.py models/__init__.py evidence/__init__.py training/__init__.py
```

给 Agent 第一个任务：

```text
请根据 AGENTS.md 实现 Task 1：数据加载和 Stage 1 baseline。
我稍后会提供 YelpChi / Amazon 的实际数据路径，所以 config 中请使用占位路径。
要求 debug mode 能用 synthetic small graph 跑通，不依赖真实数据。
完成后运行 pytest -q，并运行：
python scripts/train_stage1.py --config configs/yelpchi_gcn.yaml --debug
```

关键原则：**先 synthetic small graph 跑通**。真实 YelpChi / Amazon 路径晚点给也不影响架构推进。

---

## 9. 配置文件模板

```yaml
# configs/yelpchi_gcn.yaml
project: cover_fd
dataset:
  name: yelpchi
  path: /path/to/yelpchi.pt
  format: pt
  scarcity_ratio: 1.0
  split_seed: 0

model:
  name: gcn
  hidden_dim: 64
  num_layers: 2
  dropout: 0.5

train:
  seed: 0
  epochs: 300
  debug_epochs: 3
  lr: 0.005
  weight_decay: 0.0005
  patience: 50
  device: cuda

evidence:
  trace_size: 2000
  teacher: rule
  use_verifier: true
  use_counter: true

reasoner:
  hidden_dim: 128
  evidence_emb_dim: 16
  rho: 0.3
  lambda_evi: 0.5
  use_type_loss: true
  use_evidence_loss: true
  use_residual_gate: true

eval:
  k_values: [50, 100, 200]
```

---

## 10. 总体路线图

### 第 1 周

- 跑通 Stage 1 + rule ERR + verifier + Stage 3
- 在 synthetic 和 YelpChi 上得到第一版结果

### 第 2 周

- 加 BWGNN
- 跑 YelpChi / Amazon baseline
- 跑 full vs rule teacher vs no verifier

### 第 3 周

- 接入 LLM offline teacher
- 做 verifier stats、acceptance rate、prompt cache
- 跑主结果 5 seeds

### 第 4 周

- scarcity + ablation
- 写论文实验表
- 可选 Cora / PubMed 泛化实验

### 主链闭环

优先保证这条主链闭环：

```text
base.pt
→ evidence_cards.jsonl
→ accepted_err.jsonl / rejected_err.jsonl
→ reasoner.pt
→ final_metrics.json
→ ablation_table.csv
```

只要这条链稳定，论文工作就具备"可训练、可评估、可解释、可复现"的基本盘。
