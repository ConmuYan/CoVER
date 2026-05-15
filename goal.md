# Paper Result Consolidation

## Current Goal Status

Task 9 is complete. The project has moved from model iteration to paper/result consolidation.

Final positioning:

| metric | value |
|---|---:|
| main detector | `CoVER-REL-Gate` |
| LLM-assisted research extension | `CoVER-REL-Judge` |
| YelpChi Gate ΔAUPRC vs base | +0.026585 |
| YelpChi Judge ΔAUPRC vs Gate | +0.000010 |
| Amazon Gate ΔAUPRC vs base | +0.003508 |
| Amazon Judge ΔAUPRC vs Gate | +0.000224 |

Main artifacts:

- `artifacts/tables/paper_main_results.csv`
- `artifacts/tables/paper_main_results.md`
- `artifacts/tables/paper_relation_ablation.csv`
- `artifacts/tables/paper_relation_ablation.md`
- `artifacts/tables/paper_gate_judge_summary.csv`
- `artifacts/tables/paper_gate_judge_summary.md`
- `artifacts/tables/paper_negative_routes.csv`
- `artifacts/tables/paper_negative_routes.md`
- `artifacts/paper/README.md`
- `artifacts/paper/method_section_draft.md`
- `artifacts/paper/results_narrative.md`
- `artifacts/paper/appendix_failure_routes.md`

Next recommended goal:

Draft the manuscript using the generated artifacts:
- method section from `artifacts/paper/method_section_draft.md`;
- result narrative from `artifacts/paper/results_narrative.md`;
- appendix motivation from `artifacts/paper/appendix_failure_routes.md`;
- main tables from `artifacts/tables/paper_*.md`.

---

## Historical Previous Goal

# CoVER-REL Schema-Aware Sparse Relation Evidence Gate

## 0. Active Goal

Rebuild the next stage of CoVER into **CoVER-REL-Gate**: a schema-aware, sparse relation evidence framework that uses relation-wise anonymous feature evidence experts under a base-prior reasoner. The method must work on both YelpChi and Amazon without hardcoding YelpChi-specific RUR logic.

The immediate objective is **not** to add more Qwen latent distillation. The current verified performance gains come from relation-aware anonymous feature evidence, not from canonical ERR hidden states. Qwen / verifier / ERR remain useful for explanation and safety, but the next performance path is relation evidence fusion.

---

## 1. Progress Maintenance Update

Append the following block to `progress.md` or the project progress maintenance document.

```markdown
## Task 8.13B Completion — Amazon CoVER-REL 5-Seed Stability

### Status
Accepted. Amazon CoVER-REL 5-seed stability verification is complete.

### Key Results
Amazon CoVER-REL UVU-only is a Strong GO:

| setting | mean ΔAUPRC | mean ΔROC-AUC | mean ΔMacro-F1 | AUPRC positive seeds | mean near-cap |
|---|---:|---:|---:|---:|---:|
| UVU-only | +0.003159 | +0.001213 | +0.000448 | 5/5 | 0.828533 |
| All-rel | +0.002863 | +0.001259 | +0.000808 | 5/5 | 0.909210 |

### Interpretation
- YelpChi relation utility is sharply concentrated in RUR.
  - YelpChi RUR-only 5-seed: mean ΔAUPRC +0.027099, mean ΔROC-AUC +0.007301, mean ΔMacro-F1 +0.002704, AUPRC positive on 5/5 seeds.
- Amazon relation utility is weaker and more distributed, but UVU is the strongest single relation.
  - Amazon UVU-only 5-seed: mean ΔAUPRC +0.003159, AUPRC positive on 5/5 seeds.
- This supports CoVER-REL as a schema-aware relation evidence framework rather than a YelpChi-RUR-specific trick.
- All-rel is positive on Amazon but has higher near-cap residual behavior, so it should not be used as the final candidate without gate/anchor control.

### Current Methodological Conclusion
CoVER-REL should be framed as:

> Given a multi-relation fraud graph schema, CoVER-REL constructs relation-wise anonymous feature evidence experts and learns or selects useful relation evidence under a base-prior LLM-free reasoner.

Dataset-specific relation outcomes:
- YelpChi: strongest relation = RUR.
- Amazon: strongest relation = UVU.
- Therefore, the next method should be a schema-aware sparse relation gate, not a hardcoded RUR method.

### Next Task
Task 8.14: Schema-Aware Sparse Relation Evidence Gate for YelpChi + Amazon.

Goal:
- Preserve the strongest single relation evidence.
- Allow weaker relations to contribute only when useful.
- Avoid ordinary softmax attention that forces useful and noisy relations to compete equally.
- Avoid MoE load balancing because relation utility is intentionally imbalanced.
- Diagnose gate distributions and residual near-cap behavior.
```

---

## 2. Why We Are Pivoting to Schema-Aware Relation Gate

### Verified facts
1. CoVER-LIFT canonical ERR hidden states aligned well with the student latent, but did not improve AUPRC or ranking.
2. Structure-only ERR supervision was too sparse for reliable FN/FP correction.
3. Relation-aware anonymous feature evidence produced the first consistent performance gains:
   - YelpChi RUR-only: strong 5-seed gain.
   - Amazon UVU-only: stable 5-seed gain under a saturated BWGNN baseline.
4. Relation utility is dataset-schema-dependent:
   - YelpChi: RUR dominates.
   - Amazon: UVU is the strongest single relation, while all-rel is positive but more saturated.

### Core pivot
The next method must be:

> **Schema-aware relation expert fusion**, not YelpChi-specific relation engineering.

---

## 3. Literature Grounding and Mapping to CoVER-REL-Gate

| Paper / Venue | DOI / Identifier | Key Innovation | How it maps to CoVER-REL-Gate |
|---|---|---|---|
| **Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling**, NeurIPS 2023 | arXiv:2304.02806. DOI not found in official NeurIPS/arXiv metadata. | Introduces MoE into GNNs; nodes dynamically select aggregation experts to handle graph structural diversity. | Motivates node-wise relation expert selection: each relation-specific evidence encoder is an expert, and each node can use different relation evidence. |
| **Mixture of Weak and Strong Experts on Graphs (Mowst)**, ICLR 2024 | OpenReview ID: wYvuY60SdD; arXiv:2311.05185. DOI not found in OpenReview/ICLR metadata. | Decouples weak self-feature expert and strong GNN expert; uses confidence to conditionally activate experts and reveals soft graph splitting dynamics. | Motivates strong-relation/base-prior preservation and optional activation of weaker relations. We should not average all relations equally. |
| **Can LLMs Find Fraudsters? Multi-level LLM Enhanced Graph Fraud Detection (MLED)**, arXiv 2025 preprint | arXiv:2507.11997. No DOI found; not confirmed as peer-reviewed venue. | Proposes type-level and relation-level LLM enhancers for graph fraud detection; emphasizes relation-level fraud evidence. | Supports the thesis that fraud graphs need relation-level evidence. We implement this without relying on raw text by using relation-aware anonymous feature evidence. |
| **DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs**, AAAI 2026 / arXiv 2025 | arXiv:2507.21653. DOI not found in arXiv/AAAI metadata. | Preserves fine-grained target-node information while summarizing neighbor information coarsely to avoid prompt explosion; reports AUPRC gains. | Supports target-fine / relation-summary design. CoVER-REL uses fine-grained relation stats for the target node and compact relation evidence summaries for explanation. |
| **GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection**, NeurIPS Datasets & Benchmarks 2023 | DOI reported by proceedings.com: 10.52202/075280-1289. arXiv:2306.12251. | Benchmarks supervised graph anomaly detection and finds tree ensembles with simple neighborhood aggregation can outperform specialized GNNs. | Supports the practical value of relation-aware neighborhood feature statistics over more complex but weakly informative latent distillation. |

Notes:
- If a DOI is not found in official metadata, do not invent one. Use arXiv ID / OpenReview ID as the stable identifier.
- The method should cite the venue and identifier honestly.

---

## 4. Mathematical Formulation

### 4.1 Multi-relation fraud graph

Let

\[
G=(V, X, \{E_r\}_{r\in\mathcal{R}}, y)
\]

where:
- \(V\): nodes.
- \(X\in\mathbb{R}^{|V|\times d}\): anonymous handcrafted node features.
- \(\mathcal{R}\): relation schema.
- \(E_r\): edge set for relation \(r\).
- \(y\): labels used only for supervised training and train-only prototype construction.

For YelpChi:

\[
\mathcal{R}_{Yelp}=\{RUR, RSR, RTR\}
\]

For Amazon:

\[
\mathcal{R}_{Amazon}=\{UPU, USU, UVU\}
\]

The method must not hardcode YelpChi relation names. Relation schemas must be dataset-driven.

---

### 4.2 Relation-aware anonymous feature statistics

For node \(i\) and relation \(r\), define relation neighborhood:

\[
\mathcal{N}_r(i)=\{j:(i,j)\in E_r\}.
\]

Compute relation evidence features:

\[
d_{i,r}=|\mathcal{N}_r(i)|
\]

\[
\mu_{i,r}=\frac{1}{|\mathcal{N}_r(i)|}\sum_{j\in\mathcal{N}_r(i)} x_j
\]

\[
\Delta_{i,r}^{L2}=\|x_i-\mu_{i,r}\|_2
\]

\[
c_{i,r}= \cos(x_i,\mu_{i,r})
\]

\[
z_{i,r,k}=\frac{x_{i,k}-\mu_{i,r,k}}{\sigma_{i,r,k}+\epsilon}
\]

\[
zcount_{i,r}=\sum_k \mathbf{1}\{|z_{i,r,k}|>\tau_z\}.
\]

Train-only fraud / benign prototypes for relation \(r\):

\[
p_r^F = \frac{1}{|\mathcal{T}_F|}\sum_{j\in\mathcal{T}_F} \phi_{j,r},
\quad
p_r^B = \frac{1}{|\mathcal{T}_B|}\sum_{j\in\mathcal{T}_B} \phi_{j,r}
\]

where \(\mathcal{T}_F,\mathcal{T}_B\) are train fraud / benign nodes only.

Prototype distances:

\[
D^F_{i,r}=\|\phi_{i,r}-p_r^F\|_2,\quad
D^B_{i,r}=\|\phi_{i,r}-p_r^B\|_2
\]

Prototype margin:

\[
m_{i,r}=D^B_{i,r}-D^F_{i,r}.
\]

A positive margin means relation evidence is closer to fraud prototype than benign prototype.

Relation feature vector:

\[
\phi_{i,r} = [
\log(1+d_{i,r}),
\Delta^{L2}_{i,r},
c_{i,r},
zcount_{i,r},
D^F_{i,r},
D^B_{i,r},
m_{i,r}
].
\]

---

### 4.3 Relation expert encoder

Each relation has an expert:

\[
h_{i,r}=E_r(\phi_{i,r})
\]

where \(E_r\) is a small MLP. For schema generality, experts can share architecture but have relation-specific parameters or relation embeddings.

---

### 4.4 Base prior

Let the frozen base BWGNN produce:

\[
b_i = f_{base}(i),\quad h_i^B = \text{Emb}_{base}(i)
\]

where \(b_i\) is the base logit and \(h_i^B\) is the base embedding. The base logit must be detached in Stage3 losses:

\[
\text{stopgrad}(b_i).
\]

---

### 4.5 Two possible gate modes

#### Mode A: best-relation anchor gate

Use when a strong relation is already empirically identified.

YelpChi:
\[
r^\* = RUR
\]

Amazon:
\[
r^\* = UVU
\]

Fusion:

\[
h_i^{REL}=h_{i,r^\*} + \sum_{r\neq r^\*} g_{i,r} h_{i,r}
\]

with:

\[
g_{i,r}=\sigma(G_r([h_i^B,h_{i,r^\*},h_{i,r},q_{i,r}]))
\]

where \(q_{i,r}\) are relation quality features such as degree, consistency, deviation, prototype margin, and missingness flags.

#### Mode B: base-preserving schema gate

Use as a more dataset-general variant.

\[
h_i^{REL}=\sum_{r\in\mathcal{R}}g_{i,r}h_{i,r}
\]

\[
h_i^{fused}=h_i^B+h_i^{REL}
\]

with:

\[
g_{i,r}=\sigma(G_r([h_i^B,h_{i,r},q_{i,r}])).
\]

This avoids hardcoding an anchor relation and uses the base prior as the guaranteed backbone.

---

### 4.6 Sparse optional gate regularization

Do not use MoE load balancing.

Use sparse optional gate regularization:

\[
\mathcal{L}_{gate}=\lambda_g\cdot \frac{1}{|V_{train}|}\sum_{i\in V_{train}}\sum_{r\in \mathcal{R}_{optional}}g_{i,r}
\]

For base-preserving gate, this can be applied to all relation gates or only weak optional gates after a warmup.

Suggested default:

\[
\lambda_g = 10^{-4}
\]

or

\[
5\times 10^{-4}
\]

if optional relations are overused.

Optional relation dropout:
- Drop optional relation experts with probability 0.1 during training.
- Never drop the base prior.
- In anchor mode, never drop the anchor relation.

---

### 4.7 Final prediction

The reasoner receives:

\[
z_i = [h_i^B, h_i^{REL}, \text{other optional evidence embeddings}]
\]

and outputs residual:

\[
r_i = \rho \cdot \sigma(a_i) \cdot \delta_{scale}\tanh(u_i)
\]

\[
\hat{y}^{logit}_i = b_i + r_i
\]

where \(b_i\) is detached inside residual losses.

---

### 4.8 Training objective

For this task, do not reintroduce Qwen latents.

Use a relation-gated variant of existing Stage3 loss:

\[
\mathcal{L}=
\mathcal{L}_{det}
+\lambda_{rank}\mathcal{L}_{rank}
+\lambda_{anchor}\mathcal{L}_{anchor}
+\lambda_{gate}\mathcal{L}_{gate}
+\lambda_{cap}\mathcal{L}_{cap}
\]

where:
- \(\mathcal{L}_{det}\): weighted BCE on training nodes.
- \(\mathcal{L}_{rank}\): pairwise ranking loss for AUPRC.
- \(\mathcal{L}_{anchor}\): residual anchor for non-intervention nodes.
- \(\mathcal{L}_{gate}\): sparse optional gate regularizer.
- \(\mathcal{L}_{cap}\): bounded residual penalty.

If existing CoVER CVE / ERR losses are enabled, they may remain, but this task should not rely on Qwen latents. The core test is whether relation gate improves or stabilizes relation evidence fusion.

---

## 5. Proof Sketch / Mathematical Rationale

### Proposition 1: Anchor-preserving gate contains the best-single relation hypothesis class

In anchor gate mode:

\[
h_i^{REL}=h_{i,r^\*} + \sum_{r\neq r^\*} g_{i,r}h_{i,r}
\]

If all optional gates are zero:

\[
g_{i,r}=0,\ \forall r\neq r^\*
\]

then:

\[
h_i^{REL}=h_{i,r^\*}.
\]

Therefore the anchor-gated model contains the best-single relation model as a special case.

Implication:
- In the ideal optimization limit, anchor gate should not be less expressive than the best-single relation.
- In finite data, optional gates may overfit; sparse gate regularization and relation dropout are necessary to prevent noisy relation usage.

---

### Proposition 2: Additive anchor gate avoids strong-relation dilution

Softmax attention over relations gives:

\[
h_i^{softmax}=\sum_r \alpha_{i,r}h_{i,r},\quad \sum_r\alpha_{i,r}=1.
\]

If \(h_{i,r^\*}\) is the strongest relation and weaker relations have noisy directions, then \(\alpha_{i,r^\*}<1\) necessarily scales down the useful strong relation representation.

Anchor additive gate gives:

\[
h_i^{anchor}=h_{i,r^\*}+\sum_{r\neq r^\*}g_{i,r}h_{i,r}.
\]

The strongest relation is not forced to share a probability simplex with noisy optional relations.

Implication:
- Additive anchor gate is better suited to empirical relation utility imbalance.
- This matches YelpChi, where RUR is strongly useful and RSR/RTR are weak.
- It also matches Amazon, where UVU is best but gains are small and residual stability matters.

---

### Proposition 3: Sparse optional gating implements evidence-supported relation activation

The sparse gate penalty:

\[
\lambda_g\sum_r g_{i,r}
\]

creates a cost for opening optional relations. An optional relation \(r\) will be used only if its reduction in task loss exceeds the sparsity cost.

For a training node \(i\), optional relation \(r\) is useful if:

\[
\Delta \mathcal{L}_{task}(i,r) > \lambda_g g_{i,r}.
\]

Implication:
- Noisy relations are suppressed.
- Useful relation evidence can still contribute.
- Gate values become interpretable relation-utility scores.

---

### Proposition 4: Base-preserving schema gate is dataset-schema adaptive

Base-preserving gate:

\[
h_i^{fused}=h_i^B+\sum_{r\in\mathcal{R}}g_{i,r}h_{i,r}.
\]

If all relation evidence is unhelpful:

\[
g_{i,r}\rightarrow 0,\quad \forall r
\]

then:

\[
h_i^{fused}\approx h_i^B.
\]

Thus the base-prior fallback is preserved.

If some relation is useful, the model can selectively activate it.

Implication:
- The same formulation covers YelpChi and Amazon.
- It does not require hardcoding RUR or UVU.
- It supports relation schemas beyond YelpChi/Amazon.

---

## 6. Task 8.14 — Detailed Instructions for Codex

### Task Title

**Task 8.14: Schema-Aware Sparse Relation Evidence Gate for YelpChi + Amazon**

### Current Verified Baselines

YelpChi:
- `cover_rel_rur_nollm` 5-seed Strong GO.
- mean ΔAUPRC +0.027099.
- mean ΔROC-AUC +0.007301.
- mean ΔMacro-F1 +0.002704.
- AUPRC positive seeds: 5/5.
- Best relation: RUR.

Amazon:
- `cover_rel_uvu_nollm` 5-seed Strong GO.
- mean ΔAUPRC +0.003159.
- mean ΔROC-AUC +0.001213.
- mean ΔMacro-F1 +0.000448.
- AUPRC positive seeds: 5/5.
- Best relation: UVU.
- `cover_rel_all_nollm` is positive but has high near-cap 0.909210 and should not be the main candidate.

### Goal

Implement and evaluate a schema-aware sparse relation gate.

The gate must:
1. Work for both YelpChi and Amazon.
2. Avoid hardcoded YelpChi-specific RUR logic.
3. Preserve the strongest/base relation evidence.
4. Allow weaker relations to contribute only when useful.
5. Record gate distributions as relation-utility explanations.
6. Maintain LLM-free Stage3 train/eval/inference.
7. Reduce or control near-cap residual behavior.

---

## 7. Implementation Requirements

### 7.1 Files likely modified

- `models/reasoner.py`
- `scripts/train_stage3.py`
- `scripts/evaluate.py`
- `training/losses.py`
- `configs/stage3_cover_rel_gate_nollm.yaml`
- `evidence/relation_features.py` if needed
- `scripts/run_cover_rel_ablation_report.py`
- `scripts/run_cover_rel_5seed_diagnostics.py`
- `tests/test_relation_gate.py`
- `tests/test_relation_features.py`

### 7.2 Add relation fusion modes

Add:

```text
--relation_fusion_mode
```

Supported values:
- `single`
- `concat`
- `anchor_gate`
- `base_additive_gate`

Existing behavior should map to:
- `single`: use a specified relation only.
- `concat`: all-rel concat / previous all-rel behavior.
- `anchor_gate`: best relation anchor + optional sparse gates.
- `base_additive_gate`: generic schema-aware base-preserving gates.

### 7.3 Add config arguments

```text
--anchor_relation
--optional_relations
--lambda_gate_sparse
--relation_dropout
--log_relation_gates
--gate_hidden_dim
--gate_temperature
--gate_warmup_epochs
```

Defaults:
- `lambda_gate_sparse=1e-4`
- `relation_dropout=0.1`
- `gate_hidden_dim=64`
- `gate_temperature=1.0`
- `gate_warmup_epochs=0`

### 7.4 Anchor gate setup

YelpChi:
- anchor_relation: RUR
- optional_relations: RSR, RTR
- stage3_run_name: `cover_rel_anchor_gate_nollm`

Amazon:
- anchor_relation: UVU
- optional_relations: UPU, USU
- stage3_run_name: `cover_rel_anchor_gate_nollm`

Formula:

\[
h_i^{REL}=h_{i,anchor}+\sum_{r\in optional}g_{i,r}h_{i,r}
\]

### 7.5 Base-additive gate setup

For both datasets:

\[
h_i^{REL}=\sum_{r\in\mathcal{R}}g_{i,r}h_{i,r}
\]

\[
h_i^{fused}=h_i^B+h_i^{REL}
\]

stage3_run_name:
- `cover_rel_base_gate_nollm`

### 7.6 Gate diagnostics

Save:
- `relation_gate_stats.json`
- `relation_gate_stats.csv`
- `relation_gate_by_seed.csv`
- `relation_gate_by_label.csv`
- `relation_gate_by_base_status.csv`
- `relation_gate_by_relation.csv`

Report:
- mean gate per relation.
- gate-open rate per relation, using threshold 0.5.
- gate distribution by label.
- gate distribution by base status: TP/TN/FP/FN.
- optional relation contribution norm.
- gate vs residual near-cap correlation.
- gate vs AUPRC improvement.
- gate sparsity penalty value.

---

## 8. Experiments to Run

### 8.1 YelpChi 5-seed

Seeds:
- 42
- 123
- 456
- 789
- 2026

Run:
1. `cover_rel_anchor_gate_nollm`
2. `cover_rel_base_gate_nollm`

Compare against:
- fresh BWGNN.
- `cover_rel_rur_nollm`.
- `cover_rel_all_nollm`.

### 8.2 Amazon 5-seed

Seeds:
- 42
- 123
- 456
- 789
- 2026

Run:
1. `cover_rel_anchor_gate_nollm`
2. `cover_rel_base_gate_nollm`

Compare against:
- fresh BWGNN.
- `cover_rel_uvu_nollm`.
- `cover_rel_all_nollm`.

### 8.3 Optional conservative variant

If near-cap remains high:
- run `cover_rel_anchor_gate_conservative_nollm`
- increase anchor / cap penalty.
- do not change relation features.
- do not add Qwen latents.

---

## 9. Reports to Generate

### YelpChi

- `artifacts/tables/yelpchi_cover_rel_gate_5seed.csv`
- `artifacts/tables/yelpchi_cover_rel_gate_5seed.md`
- `artifacts/reports/yelpchi_cover_rel_gate_5seed_conclusion.md`
- `artifacts/reports/yelpchi_cover_rel_gate_diagnostics_5seed.md`

### Amazon

- `artifacts/tables/amazon_cover_rel_gate_5seed.csv`
- `artifacts/tables/amazon_cover_rel_gate_5seed.md`
- `artifacts/reports/amazon_cover_rel_gate_5seed_conclusion.md`
- `artifacts/reports/amazon_cover_rel_gate_diagnostics_5seed.md`

### Cross-dataset summary

- `artifacts/tables/cover_rel_schema_aware_summary.csv`
- `artifacts/tables/cover_rel_schema_aware_summary.md`
- `artifacts/reports/cover_rel_schema_aware_final_conclusion.md`

The cross-dataset report must explicitly state:
- YelpChi best relation: RUR.
- Amazon best relation: UVU.
- Whether anchor gate improves over or matches best-single relation.
- Whether base-additive gate provides dataset-general behavior.
- Whether gate reduces near-cap residual saturation.
- Whether CoVER-REL can be claimed as schema-aware rather than dataset-specific.

---

## 10. Success Criteria

### Strong GO

For both YelpChi and Amazon:
- gate mean ΔAUPRC >= best-single relation mean ΔAUPRC, and
- Macro-F1 is not worse than best-single by more than 0.005, and
- AUPRC positive seeds >= 4/5.

### Acceptable GO

Gate is acceptable if:
- mean ΔAUPRC is within 0.002 of best-single relation, and
- Macro-F1 is more stable, or
- near-cap fraction is reduced, or
- gate diagnostics are interpretable and optional relations are not always open.

### No-Go

Gate is No-Go if:
- mean ΔAUPRC is lower than best-single by more than 0.002, and
- no stability / near-cap / interpretability advantage is observed, or
- optional relation gates collapse to always-on noisy usage.

---

## 11. Hard Constraints

- Stage3 train/evaluate/inference must remain LLM-free.
- Do not use Qwen latents in this task.
- Do not modify or relax verifier.
- Do not use rejected ERRs in loss.
- Do not use summary in model/loss.
- Do not expose base score/prob/logit/confidence/prediction to any teacher.
- Relation prototypes use train labels only.
- test_label_used=false.
- target_label_used=false in relation feature artifacts.
- base logits detached in losses.
- Relation names must come from dataset schema, not hardcoded YelpChi logic.
- No MoE load balancing loss.
- No softmax gate as the main method unless used only as an ablation.

---

## 12. Required Tests and Verification

Run:
```bash
ruff check <changed files>
pytest -q
python scripts/train_stage1.py --config configs/yelpchi_gcn.yaml --debug
python scripts/generate_stage2_err.py --config configs/yelpchi_gcn.yaml --teacher rule --debug
python scripts/train_stage3.py --config configs/yelpchi_gcn.yaml --debug
python scripts/evaluate.py --config configs/yelpchi_gcn.yaml --debug
```

Add tests:
- `tests/test_relation_gate.py`

Test cases:
1. Anchor gate reduces to best-single when optional gates are zero.
2. Base-additive gate reduces to base branch when all gates are zero.
3. Sparse gate penalty is finite.
4. Relation dropout never drops anchor relation.
5. Relation names are schema-driven.
6. No test labels are used in relation feature/prototype construction.
7. Stage3 remains LLM-free.
8. Gate stats are saved.
9. Gate output shape matches number of relations.
10. No softmax simplex constraint is used in anchor gate.

---

## 13. Codex Execution Prompt

Use this block after setting `/goal`.

```text
/goal Implement CoVER-REL-Gate: a schema-aware sparse relation evidence expert gate for YelpChi and Amazon that preserves best relation/base-prior evidence, conditionally activates weaker relation experts, keeps Stage3 LLM-free, and validates whether relation gating matches or improves best-single relation performance while reducing residual near-cap instability.

Please read goal.md and execute Task 8.14 end-to-end.

Do not stop after planning. First write a concise implementation plan, then implement phase by phase.

Follow these priorities:
1. Implement schema-aware relation gate.
2. Run YelpChi and Amazon 5-seed anchor_gate and base_additive_gate.
3. Generate diagnostics and cross-dataset summary.
4. Decide Strong GO / Acceptable GO / No-Go.

Do not use Qwen latents.
Do not relax verifier.
Do not hardcode YelpChi RUR beyond config-specified anchor_relation.
Keep all relation schema logic dataset-driven.

When finished, return:
- files changed
- commands run
- tests
- YelpChi results
- Amazon results
- gate diagnostics
- cross-dataset conclusion
- final recommendation
```
