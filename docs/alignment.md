# CoVER-FD 研究定位与方法-代码-实验对齐分析

---

## 1. 文献定位与代码骨架的对应

CoVER 的核心链路：

```text
LLM-generated rationale
→ structured evidence record
→ contract verification
→ signed evidence distillation
→ LLM-free fraud prediction and explanation
```

与代码模块一一对应：

| 研究概念 | 代码模块 | 已体现在骨架中 |
|----------|----------|:------------:|
| score-blind evidence card | `evidence/schema.py`, `evidence/adapter.py`, `evidence/prompt.py` | 是 |
| offline LLM teacher | `evidence/llm_teacher.py`, `scripts/generate_stage2_err.py` | 是 |
| structured ERR | `ERR` dataclass / JSONL cache | 是 |
| contract verification | `evidence/verifier.py`, `evidence/contracts.yaml` | 是 |
| supporting / counter evidence | `pos_mask`, `neg_mask`, `allowed_support_ids`, `allowed_counter_ids` | 是 |
| evidence distillation | `models/reasoner.py`, `training/losses.py`, `scripts/train_stage3.py` | 是 |
| LLM-free inference | `scripts/evaluate.py` 不调用 LLM | 是 |
| 可解释输出 | reason type + evidence mask + template | 是 |
| verifier / score-blind / counter ablation | `scripts/run_ablation.py` | 是 |

代码结构是围绕研究定位设计的，而非临时拼凑。

---

## 2. 文献 Gap 与代码硬约束的对应

现有 LLM-GNN / LLM-KD 工作的不足：

```text
1. LLM rationale 可能 hallucinate
2. LLM 可能复述 base model score
3. free-form rationale 未经验证就进入训练
```

在 `AGENTS.md` 和代码中已转为硬约束：

```text
- LLM prompt must not contain base_score / probability / logit / confidence.
- Rejected ERR records must not contribute to evidence distillation loss.
- ERR summary is for human inspection only and must not be used in loss.
- No LLM call during inference.
- No LLM call during Stage 3 training.
- Verifier must check schema, availability, role, contract, score-blindness, label compatibility.
```

研究创新点已经落到代码规范里，而非只停留在论文语言上。

---

## 3. 与 DuoKD / LinguGKD 的借鉴关系

### 借鉴 LinguGKD 的部分

```text
teacher output offline cache
student GNN / reasoner training
multi-seed experiment
lightweight research-code style
```

对应代码：

```text
artifacts/err_cache/
scripts/generate_stage2_err.py
scripts/train_stage3.py
scripts/aggregate_results.py
```

### 借鉴 DuoKD 的部分

```text
positive knowledge → supporting_evidence
negative knowledge → counter_evidence
dual supervision → pos_mask / neg_mask
```

对应代码：

```text
ERR.supporting_evidence
ERR.counter_evidence
reasoner.pos_head
reasoner.neg_head
L_pos_mask + L_neg_mask
ablation: no_counter
```

### 未照搬的部分

```text
LinguGKD 的 instruction tuning
DuoKD 的纯 semantic rationale MSE / PPL / textual distance filter
复杂 LLM hidden-state alignment
```

主任务是 YelpChi / Amazon 虚假评论检测，不是 Cora/PubMed 的语义分类主线。

---

## 4. 与硕士论文两个工作的递进关系

论文主线：

```text
工作一：结构差异先验与自适应路由
工作二：结构证据如何被生成、验证、蒸馏并解释
```

代码骨架对应递进：

```text
Stage 1 base detector
    可以接入工作一模型 / BWGNN / GCN / SAGE / GAT

Stage 2 evidence adapter
    把工作一或 BWGNN 产生的结构差异信号转成 evidence card

Stage 3 reasoner
    把 verified evidence 蒸馏为轻量可解释预测模块
```

工作二不是重复做 detector，而是在工作一之上增加：

```text
结构证据化
LLM 受约束生成
合同验证
证据蒸馏
LLM-free explanation
```

对应 `adapter → verifier → reasoner` 结构。

---

## 5. 实验设计对齐

CoVER 成立需要证明：

```text
1. full model > base detector
2. full model > no verifier
3. full model > score-visible prompt
4. full model > no counter evidence
5. scarcity 下更稳
6. evidence mask / reason type 有解释一致性
7. 控制 base score 后仍有 non-redundant 信息
```

对应实验脚本：

```text
scripts/run_ablation.py
scripts/run_scarcity.py
scripts/aggregate_results.py
scripts/check_run_integrity.py
```

消融项：

```text
no_verifier
no_counter
no_type_loss
no_evidence_loss
no_residual_gate
score_visible_prompt
rule_teacher
LLM teacher without contract
schema_only_verifier
```

实验方案与文献分析中的论证逻辑闭合。

---

## 6. 补充消融建议

增加 `schema_only_verifier` 消融，证明 CoVER 不只是 JSON schema，而是 evidence contract。

schema-only verifier 只检查：

```text
valid JSON
risk_type 合法
supporting_evidence / counter_evidence 是 list
```

不检查：

```text
role consistency
contract satisfaction
score-blindness
label compatibility
```

如果 full verifier 明显优于 schema-only verifier，"contract-verified" 贡献更有说服力。

---

## 7. 方案总览

```text
研究定位：
    CoVER-FD 是面向图欺诈检测的合同验证结构证据蒸馏框架。

方法结构：
    score-blind SEC
    offline LLM ERR
    evidence contract verifier
    signed evidence distillation
    evidence-gated residual reasoner
    LLM-free inference

代码结构：
    data/
    models/
    evidence/
    training/
    scripts/

实验结构：
    YelpChi / Amazon 主实验
    scarcity
    ablation
    CEC / non-redundancy
    Cora / PubMed 可选泛化
```

文献 gap、方法创新、代码骨架、实验设计已完全对齐。下一步按 Task 1 → Task 6 编码，不需要再大改整体方向。
