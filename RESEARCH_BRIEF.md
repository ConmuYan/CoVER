# RESEARCH BRIEF — CoVER-FD (Phase 3 Re-Architecting)

> 给 `/idea-discovery` 用的高密度上下文。**所有数字、引用、约束都来自项目内 artifact，不许编造**。
> 本 brief 是 Phase 3 (post-paper-faithful) 的研究起点，目的：把项目对齐到 **工作二**（基于结构证据提示与生成判别协同的可解释推理方法）。

---

## 0. TL;DR — 一句话研究方向

> 在已经验证的 **CoVER-REL relation-evidence reasoner**（Idea 1, work）之上，重新设计 Phase 3：
> **(Idea 2)** 通过 LoRA 微调小参数 LLM（Qwen3-4B）让其对 score-blind 关系证据做**与 rel 分支语义互补**的推理，提供与 GNN reasoner **协同的判别信号 / 反事实证据 / 自然语言归因报告**；
> **(Idea 3)** 提出**新颖的损失函数** 把 LoRA-LLM 与 reasoner 训成 generation–discrimination 协同的闭环；
> 同时把 Phase 2 里已被 5-seed 否决的 `α·Δ_llm` 残差路径与不显著的 `L_align` KL 路径**删除 / 标 deprecated**，避免污染方法叙事与消融判断。

---

## 1. 项目背景与既有积累

### 1.1 项目名与数据
- **项目**：CoVER-FD（Contract-Verified Evidence Reasoning for Fraud / Fake-Review Detection）
- **数据集**：YelpChi（RUR/RSR/RTR 三关系图，~7.7M 边）、Amazon（UPU/USU/UVU 三关系图）
- **任务**：图节点二分类异常检测（fraud / benign）
- **划分协议**：BWGNN 论文协议 train_ratio=0.4 / val:test=1:2 / stratified / 5 seeds [42,123,456,789,2026]
- **base 模型**：BWGNN（主）、GraphSAGE / GCN / GAT（cross-base 验证）
- **LLM**：Qwen3-4B-Instruct-2507，本地路径 `/data1/mq/models/Qwen3-4B-Instruct-2507`
- **硬件**：4×RTX 3090（24GB）

### 1.2 Canonical 当前方法（截至 commit `06da448`）
两阶段 CoVER-REL Reasoner：
```
Phase 1: 训练 fresh base detector (BWGNN/SAGE)，冻结作为 structural prior
Phase 2: 训练统一 base-agnostic Reasoner
         z_i = b_i + Δ_rel,i + α_i · Δ_llm,i
```
- `b_i` — frozen base logit
- `Δ_rel,i` — relation evidence 残差（bounded）
- `Δ_llm,i` — LLM judge 残差（bounded，但实测**没用**）
- `α_i` — judge 门控（实测 mean ≈ 0.17，但 |α·Δ_llm| / |Δ_rel| ≈ 0.24%）

Loss：`L = L_cls + λ_trust·L_trust + λ_sparse·L_sparse + λ_align·L_align`

---

## 2. 已被严格验证的成果（保留为 Idea 1 / 不动）

### 2.1 Idea 1 — Score-blind relation-aware anonymous evidence reasoning（**STRONG GO**）
**Hypothesis**：在多关系欺诈图上，真正有判别力的信号是 *关系级匿名特征证据*（不是 LLM rationale，不是 base score-driven explanation，不是 canonical ERR hidden state）。

**核心实现**：
- 28 个 score-blind graph evidence token（13 fraud / 12 benign / 3 neutral）+ 9 个 EvidenceCard 字段
- Train-only fraud / benign prototype，IDF-weighted polarity learning
- Schema-aware sparse relation gate（YelpChi: RUR-dominant；Amazon: UVU-centered）
- Contract-verified score-blind packet：禁止 `base_score/base_prob/base_logit/confidence/label/split/ground truth/FN-FP/final-prediction`

**5-seed 证据**：
| 评估 | 数值 | 来源 artifact |
|---|---|---|
| BWGNN YelpChi: CoVER vs compute-matched base(400ep) | ΔAUPRC = **+0.0219**, paired t = **+3.03** (p<0.05); AUROC paired t = **+7.28** (p<0.01); G-Means paired t = **+5.97** (p<0.01) | `artifacts/tables/yelpchi_bwgnn_d0_compute_matched_control.md` |
| BWGNN YelpChi: CoVER vs base100ep | ΔAUPRC = +0.1043, paired t = +30.90 (p≈6.5e-6) | `artifacts/tables/yelpchi_bwgnn_judge_revived_5seed.md` |
| GCN YelpChi: Phase2 Gate vs base | ΔAUPRC = **+0.2901**（救活弱 base） | `artifacts/tables/paper_cross_model_results.md` |
| SAGE YelpChi (confirmed 5-seed) | ΔAUPRC = +0.2540, paired t = **+6.82** (p<0.01) | `artifacts/reports/sage_confirmed_vs_sage_baselines.md` |
| BWGNN Amazon | ΔAUPRC = +0.003508（saturated baseline，但 5/5 seed 正） | `artifacts/tables/paper_main_results.md` |

**两层防御**（reviewer-proof）：
1. **base SHA-256 freeze check**：Phase 2 前后 `base.pt / base_logits / base_z` SHA-256 完全一致
2. **compute-matched control**：CoVER(100ep frozen+300ep) vs base(400ep) 完全等算力，CoVER 显著 win

→ **Idea 1 是 Phase 3 的 anchor。绝不动 evidence schema / relation feature extraction / score-blind 边界 / sparse gate / train-only prototype**。

---

## 3. 已被实验否决的设计（Phase 2 要删除 / 标 deprecated）

### 3.1 ❌ LLM Judge as Residual Corrector（`α·Δ_llm`）— DEAD
**5-seed paired t-test（YelpChi BWGNN，`artifacts/tables/yelpchi_bwgnn_judge_revived_5seed.md`）**：
| 对比 | Δ AUPRC | t | p | 结论 |
|---|---|---|---|---|
| L7_revived (2000 packet vLLM) vs L7_legacy (120 packet) | **+0.0000** | +0.03 | 0.976 | 把 judge 数据从 120→2000 也救不活 |
| L7_revived vs A1_revived (rel-only) | **−0.0012** | −0.86 | 0.436 | full model ≤ no-judge model |
| A2_revived (judge-only) vs A0_base | +0.0003 | +7.10 | 0.002 | 绝对增益 = 3e-4，统计幻觉 |

诊断：`|α·Δ_llm|` 全样本均值 = 0.00255 vs `|Δ_rel|` = 1.06278（差 **416 倍**）。

**Phase 3 决定**：删除 `α·Δ_llm` 这个加法残差路径。`Δ_llm` 头、`α` 头、`L_trust` 里的 `η_llm·α·Δ_llm²` 项全砍。

### 3.2 ❌ Loss × Architecture Ablation — 10 cells 全部 ns
**5-seed paired t-test vs L7（`artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md`）**：
所有 L0..L7 配置 AUPRC 差距 ≤ 0.0025，**全部 ns（p ≥ 0.06）**。
→ 现有 `L_int / L_sparse / L_align` 的混合形式没有显著效果；L7 champion 与 cls-only L0 不可区分。

**Phase 3 决定**：
- `L_align`（KL of key_relation）单独 ablation ns（Δ=−0.0004, p=0.32），**标 deprecated**，不作主方法 loss
- `L_sparse`（normalized gate entropy）单独 ablation ns，可保留作 explainability 工具但不主张为 metric driver
- `L_trust = mean(Δ_rel²)` 单独 ablation ns，**重新定义或删除**

### 3.3 ❌ CoVER-DIR / CV-SCD / CoVER-LIFT — 历史负面 routes
- CoVER-DIR：directional evidence 5-seed Δ AUPRC < +0.008，gate 失败
- CV-SCD 4-term loss：3-seed Δ AUPRC ≈ 0
- CoVER-LIFT：canonical ERR hidden state distillation，latent cosine 0.80 但 ΔAUPRC = −0.0003

→ **不要复活任何 DIR/SCD/LIFT/judge-residual route**。

---

## 4. 工作二（北极星目标）对齐表

| 工作二要素 | CoVER 当前 (Phase 2) | Phase 3 目标 |
|---|---|---|
| 结构证据 | ✅ 28 token + 9 field + prototype | 保持，作为 LoRA-LLM 输入序列基础 |
| 序列化 | ✅ JSON-only score-blind packet | 保持；新增 evidence → reasoning chain 序列化 |
| **生成-判别协同** | ❌ judge frozen, no closed loop | **Idea 2 核心：LoRA-Qwen3 与 reasoner 形成 generation↔discrimination 闭环** |
| **联合微调对齐** | ❌ LLM frozen, only reasoner trained | **Idea 2 核心：LoRA-Qwen3 + reasoner joint 训练（或 alternating）** |
| 小参数 LLM | ✅ Qwen3-4B | 保持，加 LoRA adapter |
| 标签 + 异常归因 + 自然语言报告 | ⚠️ 标签✓；归因来自 gate 权重（非 LLM）；`short_explanation` 仅 human-facing 不入 loss | **Idea 2 核心：让 LoRA-Qwen3 端到端输出 (label-supportive rationale, anomaly attribution, NL report)，且报告 faithfulness 进 loss** |

---

## 5. 用户对 Idea 2 / Idea 3 的明确约束

### 5.1 Idea 2（LLM 微调推理）的约束
1. **必须是 LoRA fine-tuning**（不是 prompt engineering / 不是 frozen Qwen）
2. **必须使用 Qwen3-4B 类小 LLM**（不是 GPT-4 / 70B+ 模型）
3. **输入是当前 score-blind relation evidence**（复用 Idea 1 的 evidence packet 与 prototype）
4. **必须与 rel 分支语义互补**（不是再训一个 rel 复制品）；用户原话：「**做一个当前 rel 不同的利用**」
5. **必须协同增强判别能力**（不是只做 explanation）；用户原话：「**协同增强判别能力**」
6. **不能复活 α·Δ_llm 加法残差**（已 5-seed 否决，是 reviewer 攻击面）

### 5.2 Idea 3（新损失函数）的约束
1. **必须新颖**（不是 BCE / KL / L2 / contrastive 套壳）
2. **必须能有效监督 CoVER 框架**（即同时约束 rel + LoRA-LLM + reasoner 的端到端学习）
3. **必须能在 paper 里独立成一个 contribution**（即可单独消融出统计显著贡献）
4. **必须支持 score-blind 安全约束**（rejected judge → 0 contribution; forbidden field audit）

### 5.3 Phase 2 清理指令（用户原话）
> 「之前的 phase2 设计可以大改了，除了 rel 外不 work 的设计去掉避免污染干扰判断和理解」

→ 删除清单：
- `Δ_llm` head + `α` gate head（残差加法）
- `L_align` KL 形式（key_relation → gate 软目标）
- `L_trust = mean(Δ_rel²)`（与新 loss 冲突时优先删除）
- 各种 `--gate_mode {signed_diff_legacy, safe_residual, direct_tanh, aux_only}` 中的 legacy 选项
- `cover-rel-gj/stage3_legacy/` 配置可以保留作为历史复现，但不入主方法

---

## 6. 候选方向草案（Phase 0 deepthink，供 idea-creator 参考）

> 这些只是 Phase 0 brainstorm 锚点，**idea-creator 必须自由扩展**到 8-12 个候选并 pilot。

### Idea 2 候选方向 A：LLM-as-Auxiliary-Discriminator（生成性判别）
LoRA-Qwen3 直接输出二分类 logit `q_i` 与 rationale；与 reasoner 的 `z_i` 通过 *learnable mixture-of-experts gate*（不是固定 α）协同。
新 loss：`L_co = E[KL(softmax(z) ‖ softmax(q)) · 1{disagreement}]` — 仅在 reasoner 与 LLM 不一致样本上反传双向校正信号。

### Idea 2 候选方向 B：Counter-Evidence Generation（反事实证据推理）
LoRA-Qwen3 不输出 label，而是基于 evidence 生成 **counter-hypothesis**：若 reasoner 预测 fraud，LLM 试图生成 benign explanation；反之亦然。Counter-evidence 的 contract-verified 强度反向 calibrate reasoner（不可生成 counter → 高 confidence；轻松生成 counter → 低 confidence）。

### Idea 2 候选方向 C：Reasoning-Chain Conditioned Reasoner（CoT 条件化判别）
LoRA-Qwen3 输出 score-blind chain-of-thought（不含 label，只含 evidence-derived reasoning）。Reasoner 把 CoT token sequence 作为额外 modality 进入 gate / classifier。
新 loss：`L_faith = E[(z(packet) - z(packet without key evidence cited in CoT))²]` — 报告 faithfulness 进 loss。

### Idea 2 候选方向 D：Joint LLM-Reasoner Alternating Fine-tuning
LoRA-Qwen3 与 reasoner alternating step：① reasoner step 用 LLM 当前 rationale 作为 soft prompt；② LLM step 用 reasoner 的 prediction 作为弱监督（reasoner 高置信样本）。
新 loss：`L_consensus = E[(z - q)² + γ · agreement_uncertainty(z, q)]`。

### Idea 3 候选方向（新损失）
- **L_evidence_consistency**：评估 LLM rationale 引用的 evidence subset 与 reasoner gate 高权重 relation 的 IoU。
- **L_intervention_calibrated**：把 guide.md 提到的 `L_intervention = E[(z-b)²]` 升级为 evidence-strength 加权版：`L = E[ w(evidence_strength) · (z-b)² ]`，强证据允许大干预，弱证据约束回退 base。
- **L_disagreement_reweighting**：reasoner-LLM 不一致样本被识别为 *hard sample*，在 BCE 中加权（curriculum-style）。
- **L_faithfulness**：counterfactual masking — 把 LLM rationale 引用的关键 token 从 evidence packet 中删除，重训 reasoner 应预测翻转。

---

## 7. 评估协议（Phase 3 必须沿用，不动）

- **5 seeds**：[42, 123, 456, 789, 2026]，stratified split
- **主指标**：AUPRC（fraud detection 主流）、AUROC、Macro-F1、G-Means
- **统计**：paired t-test (n=5, df=4)；显著阈值 p<0.05
- **数据集**：YelpChi（主，RUR-anchor）、Amazon（次，UVU-anchor saturated）
- **跨 base 验证**：BWGNN（主）、SAGE（次）。GCN/GAT 可作 cross-model 表（已有 5-seed）
- **安全审计**：每次都必须跑 forbidden-field audit + `max_abs_alpha_llm_rejected < 1e-6`（如果 alpha 还存在）；judge output 必须 contract-verified

---

## 8. 约束 / 非目标

### 必须遵守
- 必须 reviewer-proof：所有 Phase 3 claim 都要有 paired t-test + base SHA-256 freeze check
- 必须 score-blind：LLM 看不到 base score / label / split identity
- 必须 LLM-free at inference（如果可能）：如果 inference 时还要调 Qwen，必须解释合理 latency / cost trade-off
- 必须保留历史 artifact：legacy `cover-rel-gj/stage3_legacy/` 不删
- 必须可独立消融出每个 idea 的贡献

### 非目标 / 不追求
- ❌ 不追求 SOTA（项目已明文 `AGENTS.md`: "Do not claim SOTA"）
- ❌ 不追求大模型（>4B）
- ❌ 不追求多轮 LLM agentic loop（latency 不可控）
- ❌ 不引入新数据集（YelpChi + Amazon 已足够撑 paper）
- ❌ 不做 prompt engineering tricks 作为 contribution

---

## 9. 时间 / 算力预算

- **可用 GPU**：4 × RTX 3090（24GB），有时仅 2-3 卡可用（GPU 0/2/3 通用）
- **每个 5-seed 完整实验**：YelpChi BWGNN reasoner ≈ 10-20 min；Amazon ≈ 15-25 min
- **vLLM Qwen3-4B 推理**：~10-30× HF transformers 速度（已实现）
- **LoRA 微调（新加）**：粗估 Qwen3-4B + LoRA r=16 on 2000 packets ≈ 1-3 GPU-hours / epoch / seed
- **Phase 3 总预算**：≤ 40 GPU-hours for pilot + ≤ 80 GPU-hours for full 5-seed × main ablations

---

## 10. Phase 3 期望产物

完成本 brief 对应的 Phase 3 后，应当有：
1. 一份 paper-ready 方法（3 个独立可消融的 contribution：rel + LoRA-LLM-co-discrimination + new loss）
2. 主表：3 contribution × 5 seed × 2 dataset × 1+ base 的 paired t-test
3. Cross-base 验证表（至少 BWGNN + SAGE）
4. faithfulness / explanation 评估（label + attribution + NL report 三件套）
5. 完整 reviewer-proof 防御（SHA-256 freeze + compute-matched control + safety audit）
6. 删除 / deprecated 清单（α·Δ_llm, L_align KL, L_trust 旧形式）

---

## 11. 关键参考文件（idea-creator 必须读）

| 文件 | 用途 |
|---|---|
| `AGENTS.md` | canonical 项目规则、安全约束、命名规范 |
| `docs/cover_main.md` | canonical 两阶段方法描述 |
| `guide.md` | 用户对 full CoVER 哲学的设想（`L_intervention = E[(z-b)²]` 等） |
| `artifacts/tables/yelpchi_bwgnn_d0_compute_matched_control.md` | Idea 1 强证据 |
| `artifacts/tables/yelpchi_bwgnn_judge_revived_5seed.md` | judge 失败的硬证据（防止 idea-creator 复活 residual route） |
| `artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md` | 现有 loss 都 ns（防止 idea-creator 推荐 BCE 套壳） |
| `artifacts/reports/cover_rel_judge_final_conclusion.md` | judge 最终 5-seed conclusion |
| `models/reasoner.py` | 当前 reasoner 架构（需要被改） |
| `training/losses.py` | 当前 loss（需要被改） |
| `evidence/{schema,adapter,vocab,prototypes,prompt}.py` | evidence pipeline（**保留不动**） |
| `external/LinguGKD/` | 蒸馏参考代码 |
| `external/gread-core/` | 完整实现参考 |

---

## 12. Venue / 论文目标

- **优先 venue**：ICLR / NeurIPS / KDD（图异常检测 + LLM 协同推理两条线都接受）
- **退路 venue**：AAAI / IEEE Transactions on Big Data / TKDE
- **论文定位**：方法论文（method paper），不是 benchmark / survey

---

**END OF RESEARCH BRIEF**
