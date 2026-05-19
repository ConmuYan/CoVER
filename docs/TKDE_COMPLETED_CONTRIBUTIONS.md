# Completed TKDE Contributions: RAER, LREE, and Flash-RAER Distillation

**Status**: completed-results consolidation.  
**Scope**: only experiments and implementations already present in the repository.  
**Important boundary**: this document does **not** claim on-policy distillation. The
third contribution is the already implemented static/off-policy lightweight
distillation adapter (`RelDistillAdapter`), renamed here as **Flash-RAER
Distillation** to preserve the completed results without relying on the
unimplemented OPD design.

---

## Paper-Level Narrative

Multi-relation graph fraud detection faces a deployment tension: production
detectors are expensive to retrain and must remain auditable, but fraud evidence
is relational and heterogeneous across graph schemas. We resolve this tension
with a sequence of three completed contributions:

| # | Contribution | Novelty axis | Completed evidence |
|---|---|---|---|
| **C1** | **RAER / CoVER-REL**: contract-enforced relation-aware evidence reasoning | framework-level | A frozen-base, score-blind, train-only-prototype, bounded-residual reasoner improves AUPRC on **8/8 base × dataset cells** and yields a base-strength × evidence-type law. |
| **C2** | **LREE**: learnable relational evidence extractor | representation-level | A 13.9K-param per-relation GCN+MLP extractor replaces hand-crafted 9-dim evidence under the same contracts and wins **19/32** cell × metric comparisons. |
| **C3** | **Flash-RAER Distillation**: lightweight static distillation for deployment | compression / deployment-level | A 4,132-param adapter recovers most teacher gains at about **2.60×** head-level speed-up on measured YelpChi cells, with stronger capture when distilled from the LREE teacher. |

The three contributions are sequential but separable. C1 establishes the
contracted reasoning framework. C2 replaces the evidence representation while
leaving the reasoner/loss/contracts unchanged. C3 compresses either C1 or C2
teachers into a lightweight adapter for deployment.

---

# C1. RAER / CoVER-REL: Contract-Enforced Relation-Aware Evidence Reasoning

## C1 Claim

We introduce **RAER**, a contract-enforced relation-aware evidence reasoning
paradigm, and its canonical implementation **CoVER-REL**. CoVER-REL wraps a
frozen multi-relation fraud detector with a bounded residual reasoner that
consumes only score-blind relational evidence. Across YelpChi/Amazon and
BWGNN/SAGE/GCN/GAT, CoVER-REL produces direction-positive AUPRC lift on all
8 base × dataset cells. Its ablations reveal a cell-resolved law: prototype
evidence and schema gating are load-bearing primarily when the base detector has
headroom, while saturated cells show smaller marginal gains.

## Method

Let a frozen base detector emit a scalar base logit \(b_i\) and base embedding
\(\mathbf z_i^{\rm base}\) for node \(i\). For each relation \(r\), CoVER-REL
extracts a 9-dim score-blind evidence vector:

\[
\mathbf E_{i,r}
=
\left[
\underbrace{\text{degree statistics}}_{\text{A: structural}},
\underbrace{\text{feature-neighbor inconsistency}}_{\text{B: local mismatch}},
\underbrace{\text{train-only prototype distances}}_{\text{C: class anchors}}
\right].
\]

The per-relation experts compute:

\[
\mathbf h_{i,r} = {\rm Expert}_r(\mathbf E_{i,r}),
\qquad
s_{i,r} = \mathbf w_r^\top \mathbf h_{i,r} + b_r.
\]

The schema gate is:

\[
\mathbf g_i
=
{\rm softmax}
\left(
\frac{W_g[\mathbf z_i^{\rm base};\mathbf h_{i,1};\ldots;\mathbf h_{i,R}] + \mathbf b_g}{\tau}
\right),
\qquad
\sum_r g_{i,r}=1.
\]

The residual is architecturally bounded:

\[
u_i = \sum_{r=1}^R g_{i,r}s_{i,r},
\qquad
\Delta_i^{\rm rel} = \delta_{\max}\tanh(u_i),
\qquad
z_i = b_i + \Delta_i^{\rm rel},
\qquad
|\Delta_i^{\rm rel}| \le \delta_{\max}.
\]

The canonical loss is the single supervised classification term:

\[
L_{\rm cls}
=
\frac{1}{|\mathcal V_{\rm tr}|}
\sum_{i\in\mathcal V_{\rm tr}}
{\rm BCE}(z_i,y_i;w_+),
\qquad
w_+ =
\frac{1-\pi_{\rm fraud}^{\rm tr}}{\pi_{\rm fraud}^{\rm tr}}.
\]

No LLM judge, auxiliary gate prior, intervention penalty, or align loss is part
of the canonical method.

## Contract Proofs

**Proposition C1.1 (bounded intervention).**  
For every node \(i\), \(|\Delta_i^{\rm rel}|\le\delta_{\max}\).

**Proof.** Since \(\tanh(u)\in(-1,1)\) for all finite \(u\),
\[
|\Delta_i^{\rm rel}|=\delta_{\max}|\tanh(u_i)|<\delta_{\max}.
\]
The bound is architectural, independent of the loss. \(\square\)

**Proposition C1.2 (base-freeze).**  
The base detector is not optimized during Phase 2.

**Proof sketch.** The reasoner receives `base_z.detach()` and
`base_logit.detach()`; gradients from \(L_{\rm cls}\) therefore terminate before
base parameters. The trainer additionally snapshots SHA-256 hashes of `base.pt`,
cached `base_logits`, and `base_z` before and after Phase 2. A mismatch is
reported as `MUTATED`; completed runs report `frozen`. \(\square\)

**Proposition C1.3 (score-blind evidence).**  
\(\mathbf E_{i,r}\) contains no base score, base logit, prediction, final
prediction, split identity, or val/test labels.

**Proof sketch.** The hand-crafted extractor is a function only of
\(\mathbf A^{(r)}\), \(\mathbf X\), `train_mask`, and train labels used solely
for train-only prototypes. The forbidden fields are not arguments of the
extractor and cannot be fetched by construction. \(\square\)

**Proposition C1.4 (train-only prototypes).**  
Prototype construction does not use validation or test labels.

**Proof sketch.** The prototype sets are indexed by
\(\mathcal V_{\rm tr}^{+}\) and \(\mathcal V_{\rm tr}^{-}\) only:
\[
\boldsymbol\mu_+^r =
\frac{1}{|\mathcal V_{\rm tr}^{+,r}|}
\sum_{j\in\mathcal V_{\rm tr}^{+,r}}\bar{\mathbf x}_{j,r},
\qquad
\boldsymbol\mu_-^r =
\frac{1}{|\mathcal V_{\rm tr}^{-,r}|}
\sum_{j\in\mathcal V_{\rm tr}^{-,r}}\bar{\mathbf x}_{j,r}.
\]
Val/test labels are absent from the construction. \(\square\)

## C1 Experiment 1: Base × Dataset Main Lift

Primary metric is AUPRC over 5 seeds. CoVER-REL is direction-positive on all
8 cells.

| Cell | Phase-1 base AUPRC | CoVER-REL AUPRC | Δ |
|---|---:|---:|---:|
| YelpChi-BWGNN | 0.503 | 0.609 | **+0.106** |
| YelpChi-SAGE | 0.455 | 0.600 | **+0.145** |
| YelpChi-GCN | 0.219 | 0.495 | **+0.276** |
| YelpChi-GAT | 0.207 | 0.532 | **+0.325** |
| Amazon-BWGNN | 0.851 | 0.868 | +0.017 |
| Amazon-SAGE | 0.788 | 0.834 | +0.046 |
| Amazon-GCN | 0.258 | 0.467 | **+0.209** |
| Amazon-GAT | 0.322 | 0.459 | **+0.137** |

**Analysis.** The largest gains occur on the weakest bases, especially
YelpChi-GAT and YelpChi-GCN. Strong/saturated cells still move in the right
direction, but the headroom is naturally smaller. This directly supports the
RAER framing: relational evidence is most valuable when the base detector has
not already captured the relevant relation-conditioned structure.

## C1 Experiment 2: Loss and Architecture Falsification

YelpChi-BWGNN, 5 seeds, paired t-test against cls-only \(L_0\).

| Cell | Variant | AUPRC | Δ vs L0 | Verdict |
|---|---|---:|---:|---|
| A0 | base only | 0.5034 ± 0.0155 | -0.1042, t=-31.10 | significant worse |
| L0 | **cls-only canonical** | **0.6076 ± 0.0089** | reference | retained |
| L1 | + intervention | 0.6082 ± 0.0076 | +0.0006, t=+0.39 | n.s. |
| L2 | + sparse gate prior | 0.6100 ± 0.0095 | +0.0025, t=+1.02 | n.s. |
| L3 | + align | 0.6072 ± 0.0097 | -0.0004, t=-0.19 | n.s. |
| L4 | + int + sparse | 0.6091 ± 0.0096 | +0.0015, t=+0.51 | n.s. |
| L5 | + int + align | 0.6079 ± 0.0099 | +0.0003, t=+0.13 | n.s. |
| L6 | + sparse + align | 0.6079 ± 0.0099 | +0.0003, t=+0.12 | n.s. |
| L7 | full 4-term | 0.6076 ± 0.0094 | -0.0000, t=-0.01 | n.s. |
| A1 | rel-only | 0.6090 ± 0.0087 | +0.0014, t=+0.66 | n.s. |
| A2 | judge-only | 0.5034 ± 0.0155 | -0.1042, t=-31.08 | significant worse |

**Analysis.** The lift comes from the relation residual architecture, not from
auxiliary losses. This justifies the cls-only canonical objective and the
negative-route ledger.

## C1 Experiment 3: Component Ablation

8 cells × 4 configs × 5 seeds were run for the three single-switch ablations:
uniform gate, shared expert, and no prototype evidence.

### AUPRC

| Cell | canonical | gate_uniform Δ | shared_expert Δ | no_proto Δ |
|---|---:|---:|---:|---:|
| YelpChi-BWGNN | 0.6089 ± 0.0083 | -0.0029 ns | -0.0026 ns | +0.0002 ns |
| YelpChi-SAGE | 0.6001 ± 0.0105 | **-0.0141 ★★** | -0.0023 ns | **-0.0133 ★★** |
| YelpChi-GCN | 0.4946 ± 0.0149 | +0.0000 ns | +0.0056 ns | **-0.0954 ★★★** |
| YelpChi-GAT | 0.5316 ± 0.0136 | **-0.0244 ★** | **-0.0182 ★** | **-0.1340 ★★★** |
| Amazon-BWGNN | 0.8683 ± 0.0269 | -0.0015 ns | -0.0024 ns | -0.0018 ns |
| Amazon-SAGE | 0.8336 ± 0.0546 | -0.0175 ns | +0.0015 ns | -0.0001 ns |
| Amazon-GCN | 0.4668 ± 0.1356 | +0.0128 ns | -0.0055 ns | -0.0104 ns |
| Amazon-GAT | 0.4592 ± 0.3055 | +0.0189 trend | -0.0069 ns | -0.0020 ns |

### AUROC

| Cell | canonical | gate_uniform Δ | shared_expert Δ | no_proto Δ |
|---|---:|---:|---:|---:|
| YelpChi-BWGNN | 0.8826 ± 0.0043 | **-0.0027 ★★** | -0.0001 ns | +0.0006 ns |
| YelpChi-SAGE | 0.8787 ± 0.0029 | **-0.0070 ★★** | -0.0002 ns | **-0.0056 ★★** |
| YelpChi-GCN | 0.8541 ± 0.0052 | **-0.0048 ★★** | +0.0047 ns | **-0.0523 ★★★** |
| YelpChi-GAT | 0.8632 ± 0.0064 | **-0.0150 ★★** | -0.0009 ns | **-0.0637 ★★★** |
| Amazon-BWGNN | 0.9755 ± 0.0114 | -0.0003 ns | -0.0008 ns | -0.0017 ns |
| Amazon-SAGE | 0.9553 ± 0.0231 | **-0.0044 ★** | +0.0019 ns | -0.0012 ns |
| Amazon-GCN | 0.8646 ± 0.0433 | +0.0070 trend | +0.0024 ns | +0.0052 trend |
| Amazon-GAT | 0.8048 ± 0.1488 | +0.0084 ns | +0.0068 trend | +0.0038 ns |

### Cross-Cell Verdict

| Component | AUPRC sig / 8 | AUROC sig / 8 | M-F1 sig / 8 | G-Means sig / 8 | Main conclusion |
|---|:---:|:---:|:---:|:---:|---|
| `gate_uniform` | 2 | 5 | 3 | 1 | Schema gate is load-bearing, especially on YelpChi. |
| `shared_expert` | 1 | 0 | 0 | 0 | Shared expert is nearly equivalent; relation-specific experts are not essential. |
| `no_proto` | 3 | 3 | 2 | 3 | Prototype evidence is critical on weak YelpChi bases. |

**Contribution alignment.** This is the empirical basis for the
base-strength × evidence-type law. Prototype evidence is decisive on weak
YelpChi bases but inert on saturated Amazon cells and YelpChi-BWGNN; schema
gating matters broadly for ranking quality; independent experts are mostly
compressible into a shared MLP.

## C1 Experiment 4: Compute-Matched Control

YelpChi-BWGNN, 5 seeds. The control compares CoVER on a 100-epoch frozen base
against a base trained alone for 400 epochs.

| Variant | Total compute | AUROC | AUPRC | Macro-F1 | G-Means | F1 |
|---|---:|---:|---:|---:|---:|---:|
| BWGNN base 100ep | 100 | 0.8065 ± 0.012 | 0.4669 ± 0.025 | 0.6031 ± 0.021 | 0.4153 ± 0.041 | 0.2786 ± 0.041 |
| BWGNN base 200ep | 200 | 0.8368 ± 0.004 | 0.5333 ± 0.006 | 0.6612 ± 0.026 | 0.5180 ± 0.052 | 0.3912 ± 0.051 |
| BWGNN base 400ep | 400 | 0.8483 ± 0.006 | 0.5583 ± 0.009 | 0.6800 ± 0.032 | 0.5553 ± 0.066 | 0.4284 ± 0.064 |
| CoVER champion | 400 | **0.8745 ± 0.004** | **0.5802 ± 0.012** | **0.7322 ± 0.007** | **0.7135 ± 0.013** | **0.5440 ± 0.009** |

Paired differences, CoVER minus base 400ep:

| Metric | mean Δ | t | significance |
|---|---:|---:|---|
| AUROC | +0.0263 | +7.28 | p < 0.01 |
| AUPRC | +0.0219 | +3.03 | p < 0.05 |
| Macro-F1 | +0.0522 | +3.49 | p < 0.05 |
| G-Means | +0.1582 | +5.97 | p < 0.01 |
| F1 | +0.1156 | +4.06 | p < 0.05 |

**Analysis.** The improvement is not explained by extra base training. The base
is hash-frozen in Phase 2, and a compute-matched base-only control is still
inferior on all reported metrics.

## C1 Source Artifacts

- `models/cover_rel_reasoner.py`
- `evidence/relation_features.py`
- `training/phase2_losses.py`
- `artifacts/tables/idea1_ablation_FINAL_7cell.md`
- `artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md`
- `artifacts/tables/yelpchi_bwgnn_d0_compute_matched_control.md`
- `tests/test_base_freeze_sha256.py`
- `tests/test_phase2_reasoner.py`
- `tests/test_reasoner_no_judge_path.py`

---

# C2. LREE: Learnable Relational Evidence Extractor

## C2 Claim

We introduce **LREE**, a learnable relational evidence extractor that replaces
the hand-crafted 9-dim relation statistics with a per-relation GCN+MLP encoder
under the same C1 contracts. LREE improves over hand-crafted evidence on
19/32 cell × metric comparisons, with the strongest gains on weak bases.

## Method

For each relation \(r\), LREE builds a symmetric-normalized relation graph and
computes a learned structural embedding:

\[
\mathbf H^{(r)}
=
\tilde{\mathbf A}^{(r)}
{\rm ReLU}\!\left(
\tilde{\mathbf A}^{(r)}\mathbf X W_1^{(r)}
\right)W_2^{(r)}.
\]

It also caches the relation-neighbor mean:

\[
\bar{\mathbf x}_{i,r}
=
\sum_j
\mathbf A^{(r)}_{ij}\mathbf x_j
\Big/
\sum_j \mathbf A^{(r)}_{ij}.
\]

Train-only prototype features are:

\[
d_i^+ = \|\mathbf x_i-\boldsymbol\mu_+\|_2,\quad
d_i^- = \|\mathbf x_i-\boldsymbol\mu_-\|_2,\quad
m_i=d_i^+ - d_i^-,
\]
where prototypes use only train-mask labels. The per-relation MLP receives:

\[
\mathbf q_{i,r}
=
\left[
\mathbf x_i;\bar{\mathbf x}_{i,r};
\mathbf H_{i}^{(r)};
{\rm zscore}(d_i^+);
{\rm zscore}(d_i^-);
{\rm zscore}(m_i)
\right],
\]

and emits a drop-in 9-dim evidence vector:

\[
\mathbf E^{\rm LREE}_{i,r}
=
{\rm MLP}_r(\mathbf q_{i,r})
\in\mathbb R^9.
\]

The downstream CoVER-REL reasoner is unchanged.

## Contract Proofs

**Proposition C2.1 (score-blindness).**  
LREE evidence does not consume base logits, base probabilities, base
predictions, final predictions, or split identities.

**Proof sketch.** The extractor inputs are raw \(\mathbf X\), relation
adjacencies, `train_mask`, and train labels. Base outputs are not arguments of
`LearnedRelationEvidenceExtractor.forward`. \(\square\)

**Proposition C2.2 (train-only prototype preservation).**  
LREE prototype features are computed from train nodes only.

**Proof sketch.** The implementation forms prototype indices as
`train_mask & (train_labels == c)` under `torch.no_grad()`. Val/test labels are
not read. \(\square\)

**Proposition C2.3 (base-freeze and bounded intervention).**  
LREE does not alter the frozen base and does not alter the bounded residual
head.

**Proof sketch.** LREE is an independent evidence module. The base tensors
remain detached in the reasoner, and the same downstream
\(\delta_{\max}\tanh(\cdot)\) residual is used. \(\square\)

## C2 Experiment 1: LREE vs Hand-Crafted Evidence

8 cells × 5 seeds, paired against C1 canonical hand-crafted evidence.

### AUPRC

| Cell | hand-crafted | LREE | Δ | t | sig |
|---|---:|---:|---:|---:|:---:|
| YelpChi-BWGNN | 0.6089 ± 0.0083 | **0.6489 ± 0.0132** | +0.0399 | +11.486 | ★★★ |
| YelpChi-SAGE | 0.6001 ± 0.0105 | **0.6520 ± 0.0072** | +0.0519 | +10.015 | ★★★ |
| YelpChi-GCN | 0.4946 ± 0.0149 | **0.5904 ± 0.0102** | +0.0958 | +12.407 | ★★★ |
| YelpChi-GAT | 0.5316 ± 0.0136 | **0.6340 ± 0.0122** | +0.1024 | +9.612 | ★★★ |
| Amazon-BWGNN | 0.8683 ± 0.0269 | 0.8671 ± 0.0271 | -0.0012 | -0.556 | ns |
| Amazon-SAGE | 0.8336 ± 0.0546 | 0.8511 ± 0.0146 | +0.0175 | +0.719 | ns |
| Amazon-GCN | 0.4668 ± 0.1356 | **0.7006 ± 0.2638** | +0.2337 | +3.796 | ★ |
| Amazon-GAT | 0.4592 ± 0.3055 | 0.5575 ± 0.3810 | +0.0984 | +1.708 | ns |

### AUROC

| Cell | hand-crafted | LREE | Δ | t | sig |
|---|---:|---:|---:|---:|:---:|
| YelpChi-BWGNN | 0.8826 ± 0.0043 | **0.8986 ± 0.0032** | +0.0160 | +16.455 | ★★★ |
| YelpChi-SAGE | 0.8787 ± 0.0029 | **0.8992 ± 0.0020** | +0.0205 | +13.404 | ★★★ |
| YelpChi-GCN | 0.8541 ± 0.0052 | **0.8944 ± 0.0030** | +0.0403 | +13.909 | ★★★ |
| YelpChi-GAT | 0.8632 ± 0.0064 | **0.9017 ± 0.0026** | +0.0385 | +11.242 | ★★★ |
| Amazon-BWGNN | 0.9755 ± 0.0114 | 0.9764 ± 0.0086 | +0.0009 | +0.553 | ns |
| Amazon-SAGE | 0.9553 ± 0.0231 | 0.9624 ± 0.0066 | +0.0071 | +0.771 | ns |
| Amazon-GCN | 0.8646 ± 0.0433 | **0.9221 ± 0.0753** | +0.0575 | +3.627 | ★ |
| Amazon-GAT | 0.8048 ± 0.1488 | 0.8217 ± 0.1751 | +0.0169 | +1.104 | ns |

### Macro-F1

| Cell | hand-crafted | LREE | Δ | t | sig |
|---|---:|---:|---:|---:|:---:|
| YelpChi-BWGNN | 0.7495 ± 0.0059 | **0.7692 ± 0.0065** | +0.0198 | +7.234 | ★★ |
| YelpChi-SAGE | 0.7466 ± 0.0051 | **0.7764 ± 0.0045** | +0.0298 | +15.963 | ★★★ |
| YelpChi-GCN | 0.7150 ± 0.0092 | **0.7707 ± 0.0046** | +0.0557 | +14.856 | ★★★ |
| YelpChi-GAT | 0.7249 ± 0.0065 | **0.7742 ± 0.0036** | +0.0493 | +14.762 | ★★★ |
| Amazon-BWGNN | 0.9174 ± 0.0052 | 0.9175 ± 0.0049 | +0.0002 | +0.152 | ns |
| Amazon-SAGE | 0.8986 ± 0.0311 | 0.9091 ± 0.0121 | +0.0104 | +1.196 | ns |
| Amazon-GCN | 0.7267 ± 0.0536 | **0.8419 ± 0.1191** | +0.1152 | +3.752 | ★ |
| Amazon-GAT | 0.6900 ± 0.1865 | 0.7395 ± 0.2265 | +0.0494 | +1.671 | ns |

### G-Means

| Cell | hand-crafted | LREE | Δ | t | sig |
|---|---:|---:|---:|---:|:---:|
| YelpChi-BWGNN | 0.7301 ± 0.0236 | 0.7621 ± 0.0227 | +0.0319 | +2.392 | trend |
| YelpChi-SAGE | 0.7485 ± 0.0122 | **0.7611 ± 0.0083** | +0.0127 | +6.577 | ★★ |
| YelpChi-GCN | 0.7209 ± 0.0206 | **0.7770 ± 0.0152** | +0.0562 | +6.048 | ★★ |
| YelpChi-GAT | 0.7285 ± 0.0196 | **0.7754 ± 0.0074** | +0.0470 | +5.815 | ★★ |
| Amazon-BWGNN | 0.8851 ± 0.0105 | 0.8834 ± 0.0147 | -0.0016 | -0.694 | ns |
| Amazon-SAGE | 0.8586 ± 0.0423 | 0.8689 ± 0.0206 | +0.0103 | +0.966 | ns |
| Amazon-GCN | 0.6781 ± 0.0698 | **0.8161 ± 0.1317** | +0.1380 | +3.691 | ★ |
| Amazon-GAT | 0.5034 ± 0.3981 | 0.5506 ± 0.4355 | +0.0472 | +1.697 | ns |

### Cross-Cell Summary

| Metric | stat-sig wins | ns wins | losses |
|---|:---:|:---:|:---:|
| AUPRC | 5 | 2 | 1 |
| AUROC | 5 | 3 | 0 |
| Macro-F1 | 5 | 3 | 0 |
| G-Means | 4 | 3 | 1 |
| **Total** | **19** | **11** | **2** |

**Analysis.** LREE strictly dominates on YelpChi across nearly all metrics
and all bases. On Amazon, strong bases are saturated and show no significant
loss; Amazon-GCN gains decisively. This supports C2 as a representation-level
upgrade rather than a new loss or a new downstream classifier.

## C2 Experiment 2: LREE Module Ablation

YelpChi × 4 bases × 3 switches × 5 seeds.

| Switch | Cell | canonical | ablated | Δ | t | sig |
|---|---|---:|---:|---:|---:|:---:|
| drop GCN encoder | YelpChi-BWGNN | 0.6489 ± 0.0132 | 0.6371 ± 0.0092 | -0.0118 | +2.368 | trend |
| drop GCN encoder | YelpChi-SAGE | 0.6520 ± 0.0072 | 0.6454 ± 0.0104 | -0.0066 | +3.489 | ★ |
| drop GCN encoder | YelpChi-GCN | 0.5904 ± 0.0102 | 0.5936 ± 0.0184 | +0.0032 | -0.485 | ns |
| drop GCN encoder | YelpChi-GAT | 0.6340 ± 0.0122 | 0.6297 ± 0.0071 | -0.0043 | +0.694 | ns |
| drop proto features | YelpChi-BWGNN | 0.6489 ± 0.0132 | 0.6561 ± 0.0150 | +0.0072 | -1.050 | ns |
| drop proto features | YelpChi-SAGE | 0.6520 ± 0.0072 | 0.6509 ± 0.0051 | -0.0011 | +0.228 | ns |
| drop proto features | YelpChi-GCN | 0.5904 ± 0.0102 | 0.6106 ± 0.0148 | +0.0202 | -6.462 | ★★ |
| drop proto features | YelpChi-GAT | 0.6340 ± 0.0122 | 0.6469 ± 0.0070 | +0.0129 | -1.645 | ns |
| shared encoder + one-hot | YelpChi-BWGNN | 0.6489 ± 0.0132 | 0.6422 ± 0.0067 | -0.0067 | +1.681 | ns |
| shared encoder + one-hot | YelpChi-SAGE | 0.6520 ± 0.0072 | 0.6486 ± 0.0068 | -0.0034 | +0.911 | ns |
| shared encoder + one-hot | YelpChi-GCN | 0.5904 ± 0.0102 | 0.6015 ± 0.0162 | +0.0111 | -3.971 | ★ |
| shared encoder + one-hot | YelpChi-GAT | 0.6340 ± 0.0122 | 0.6371 ± 0.0167 | +0.0031 | -0.401 | ns |

**Analysis.** The GCN encoder is useful but not uniformly required. Explicit
prototype features are no longer consistently beneficial once the extractor is
learned; on YelpChi-GCN, dropping them helps. This motivates the empirical
"encoder absorbs prototype" interpretation: LREE can learn a substitute for the
hand-crafted prototype subspace, especially on weak bases.

## C2 Source Artifacts

- `evidence/learned_extractor.py`
- `scripts/train_phase2_reasoner.py`
- `configs/phase2_reasoner/ablation/idea2b_learned_extractor_*.yaml`
- `artifacts/tables/idea2b_learned_vs_canonical_8cell_5seed.md`
- `artifacts/tables/idea2b_ablation_4base_5seed.md`

---

# C3. Flash-RAER Distillation: Static Lightweight Adapter for Deployment

## C3 Claim

We introduce **Flash-RAER Distillation**, a lightweight static distillation
procedure that compresses a trained CoVER-REL / LREE teacher into a 4,132-param
adapter. The adapter preserves the deployment contracts at inference: it keeps
the base frozen, consumes score-blind relation evidence, and applies a bounded
residual to the base logit. This contribution is a deployment/compression
result, not an on-policy distillation result.

## Method

The adapter trunk consumes base embedding and relation evidence, but not the
base logit:

\[
\mathbf h_i^\phi
=
{\rm MLP}_\phi([\mathbf z_i^{\rm base};\mathbf E_i]).
\]

The adapter residual and gate are:

\[
\Delta_i^\phi
=
\delta_{\max}\tanh(\mathbf w_\Delta^\top\mathbf h_i^\phi),
\qquad
\mathbf g_i^\phi
=
{\rm softmax}(W_g^\phi\mathbf h_i^\phi).
\]

The final deployment logit is:

\[
z_i^\phi = b_i + \Delta_i^\phi.
\]

Given a frozen teacher with residual \(\Delta_i^T\) and gate \(\mathbf g_i^T\),
the static distillation objective is:

\[
L_{\rm Flash}
=
{\rm BCE}(z_i^\phi,y_i)
+
\lambda_\Delta
\|\Delta_i^\phi - {\rm sg}(\Delta_i^T)\|_2^2
+
\gamma_g
{\rm KL}(\mathbf g_i^\phi \| {\rm sg}(\mathbf g_i^T)).
\]

Here `sg` denotes stop-gradient teacher targets.

## Contract Proofs

**Proposition C3.1 (bounded adapter intervention).**  
For all nodes, \(|\Delta_i^\phi|\le\delta_{\max}\).

**Proof.** Same as C1: \(\Delta_i^\phi=\delta_{\max}\tanh(\cdot)\). \(\square\)

**Proposition C3.2 (frozen base at deployment).**  
The adapter does not modify base parameters and uses \(b_i\) only as an additive
frozen logit.

**Proof sketch.** The implemented adapter detaches `base_z` and uses
`base_logit.detach()` only in `final_logit = base_logit + delta_phi`.
The MLP trunk input is `[base_z; relation_features]`. \(\square\)

**Proposition C3.3 (score-blind inference).**  
At inference, the adapter trunk does not consume base probabilities or final
predictions, and its relation evidence is inherited from C1/C2 score-blind
extractors.

**Proof sketch.** The trunk input excludes `base_logit`; the base logit is used
only for the final bounded additive correction. The evidence tensor is produced
by C1 or C2 extractors. \(\square\)

## C3 Experiment 1: Distillation from Hand-Crafted CoVER-REL Teacher

8 cells × 5 seeds. Teacher is the C1 hand-crafted CoVER-REL reasoner.

| Cell | Base-only | Distill | Teacher | % REL gain | Distill vs Teacher |
|---|---:|---:|---:|---:|---|
| YelpChi-BWGNN | 0.5034 | 0.6008 ± 0.0105 | 0.6089 ± 0.0083 | 92.3% | Δ=-0.0082 ★★ |
| YelpChi-SAGE | 0.4595 | 0.5899 ± 0.0060 | 0.6001 ± 0.0105 | 92.7% | Δ=-0.0102 ★ |
| YelpChi-GCN | 0.2226 | 0.4757 ± 0.0157 | 0.4946 ± 0.0149 | 93.0% | Δ=-0.0189 ★ |
| YelpChi-GAT | 0.2061 | 0.4830 ± 0.0119 | 0.5316 ± 0.0136 | 85.1% | Δ=-0.0486 ★★★ |
| Amazon-BWGNN | 0.8590 | 0.8663 ± 0.0280 | 0.8683 ± 0.0269 | 78.4% | Δ=-0.0020 ns |
| Amazon-SAGE | 0.7894 | 0.8202 ± 0.0640 | 0.8336 ± 0.0546 | 69.8% | Δ=-0.0134 ns |
| Amazon-GCN | 0.2549 | 0.4351 ± 0.0994 | 0.4668 ± 0.1356 | 85.0% | Δ=-0.0318 ns |
| Amazon-GAT | 0.3384 | 0.4553 ± 0.2634 | 0.4592 ± 0.3055 | 96.8% | Δ=-0.0039 ns |

**Analysis.** Static distillation recovers most of the teacher's AUPRC gain,
but the hand-crafted teacher remains significantly better on the four YelpChi
cells. This makes the table a useful compression baseline but not yet a
teacher-equivalent deployment replacement for all settings.

## C3 Experiment 2: Distillation from LREE Teacher

8 cells × 5 seeds. Teacher is the C2 LREE + CoVER-REL reasoner.

### AUPRC

| Cell | Base-only | Flash-RAER | LREE teacher | % LREE REL gain | Adapter vs Teacher |
|---|---:|---:|---:|---:|---|
| YelpChi-BWGNN | 0.5034 | 0.6442 ± 0.0123 | 0.6489 ± 0.0132 | 96.8% | Δ=-0.0047 ★ |
| YelpChi-SAGE | 0.4595 | 0.6422 ± 0.0116 | 0.6520 ± 0.0072 | 94.9% | Δ=-0.0098 ns |
| YelpChi-GCN | 0.2226 | 0.5948 ± 0.0066 | 0.5904 ± 0.0102 | 101.2% | Δ=+0.0044 ns |
| YelpChi-GAT | 0.2061 | 0.6358 ± 0.0150 | 0.6340 ± 0.0122 | 100.4% | Δ=+0.0018 ns |
| Amazon-BWGNN | 0.8590 | 0.8669 ± 0.0291 | 0.8671 ± 0.0271 | 98.1% | Δ=-0.0002 ns |
| Amazon-SAGE | 0.7894 | 0.8501 ± 0.0137 | 0.8511 ± 0.0146 | 98.3% | Δ=-0.0011 ns |
| Amazon-GCN | 0.2549 | 0.7272 ± 0.1959 | 0.7006 ± 0.2638 | 106.0% | Δ=+0.0266 ns |
| Amazon-GAT | 0.3384 | 0.6161 ± 0.3210 | 0.5575 ± 0.3810 | 126.7% | Δ=+0.0585 ns |

### All Metrics Summary

| Metric | Cell | Flash-RAER | LREE teacher | Δ adapter-teacher | t(teacher-adapter) | sig |
|---|---|---:|---:|---:|---:|:---:|
| AUROC | YelpChi-BWGNN | 0.8971 ± 0.0041 | 0.8986 ± 0.0032 | -0.0015 | +1.87 | ns |
| AUROC | YelpChi-SAGE | 0.8963 ± 0.0038 | 0.8992 ± 0.0020 | -0.0029 | +1.19 | ns |
| AUROC | YelpChi-GCN | 0.8932 ± 0.0030 | 0.8944 ± 0.0030 | -0.0012 | +0.79 | ns |
| AUROC | YelpChi-GAT | 0.8998 ± 0.0033 | 0.9017 ± 0.0026 | -0.0019 | +1.13 | ns |
| AUROC | Amazon-BWGNN | 0.9760 ± 0.0098 | 0.9764 ± 0.0086 | -0.0004 | +0.61 | ns |
| AUROC | Amazon-SAGE | 0.9617 ± 0.0105 | 0.9624 ± 0.0066 | -0.0007 | +0.35 | ns |
| AUROC | Amazon-GCN | 0.9298 ± 0.0598 | 0.9221 ± 0.0753 | +0.0077 | -1.11 | ns |
| AUROC | Amazon-GAT | 0.8646 ± 0.1278 | 0.8217 ± 0.1751 | +0.0429 | -1.09 | ns |
| Macro-F1 | YelpChi-BWGNN | 0.7659 ± 0.0062 | 0.7692 ± 0.0065 | -0.0033 | +1.94 | ns |
| Macro-F1 | YelpChi-SAGE | 0.7702 ± 0.0065 | 0.7764 ± 0.0045 | -0.0062 | +1.29 | ns |
| Macro-F1 | YelpChi-GCN | 0.7589 ± 0.0081 | 0.7707 ± 0.0046 | -0.0118 | +3.53 | ★ |
| Macro-F1 | YelpChi-GAT | 0.7694 ± 0.0037 | 0.7742 ± 0.0036 | -0.0048 | +4.33 | ★ |
| Macro-F1 | Amazon-BWGNN | 0.9183 ± 0.0059 | 0.9175 ± 0.0049 | +0.0008 | -1.14 | ns |
| Macro-F1 | Amazon-SAGE | 0.9113 ± 0.0086 | 0.9091 ± 0.0121 | +0.0022 | -0.68 | ns |
| Macro-F1 | Amazon-GCN | 0.8504 ± 0.0929 | 0.8419 ± 0.1191 | +0.0085 | -0.69 | ns |
| Macro-F1 | Amazon-GAT | 0.7405 ± 0.2285 | 0.7395 ± 0.2265 | +0.0010 | -0.69 | ns |
| G-Means | YelpChi-BWGNN | 0.7517 ± 0.0055 | 0.7621 ± 0.0227 | -0.0104 | +0.88 | ns |
| G-Means | YelpChi-SAGE | 0.7708 ± 0.0158 | 0.7611 ± 0.0083 | +0.0096 | -2.65 | trend |
| G-Means | YelpChi-GCN | 0.7584 ± 0.0107 | 0.7770 ± 0.0152 | -0.0186 | +2.43 | trend |
| G-Means | YelpChi-GAT | 0.7740 ± 0.0062 | 0.7754 ± 0.0074 | -0.0015 | +0.55 | ns |
| G-Means | Amazon-BWGNN | 0.8838 ± 0.0162 | 0.8834 ± 0.0147 | +0.0004 | -0.12 | ns |
| G-Means | Amazon-SAGE | 0.8719 ± 0.0113 | 0.8689 ± 0.0206 | +0.0030 | -0.59 | ns |
| G-Means | Amazon-GCN | 0.8026 ± 0.1127 | 0.8161 ± 0.1317 | -0.0135 | +0.89 | ns |
| G-Means | Amazon-GAT | 0.5527 ± 0.4414 | 0.5506 ± 0.4355 | +0.0021 | -0.36 | ns |

| Metric | teacher sig wins | adapter sig wins | non-sig / trend |
|---|---:|---:|---:|
| AUPRC | 1 | 0 | 7 |
| AUROC | 0 | 0 | 8 |
| Macro-F1 | 2 | 0 | 6 |
| G-Means | 0 | 0 | 8 |
| **Total** | **3** | **0** | **29** |

**Analysis.** Distillation is materially stronger when the teacher is the LREE
teacher. The adapter is statistically indistinguishable from the teacher on
29/32 cell × metric comparisons and never significantly beats the teacher,
which is the expected behavior for a faithful compression model. This supports
C3 as a deployment contribution: use C2 for the strongest teacher, then
Flash-RAER for fast inference.

## C3 Experiment 3: Inference Speed

Measured head-level inference on 3 YelpChi cells × 5 seeds.

| Cell | N_test | Base-only ms | LREE teacher ms | Flash-RAER ms | Teacher→Adapter speedup |
|---|---:|---:|---:|---:|---:|
| YelpChi-BWGNN | 18,382 | 0.020 ± 0.003 | 2.281 ± 0.027 | 0.857 ± 0.015 | **2.660 ± 0.034×** |
| YelpChi-SAGE | 18,382 | 0.018 ± 0.000 | 2.225 ± 0.024 | 0.849 ± 0.011 | **2.624 ± 0.009×** |
| YelpChi-GAT | 18,382 | 0.019 ± 0.002 | 2.278 ± 0.052 | 0.876 ± 0.023 | **2.600 ± 0.039×** |

**Analysis.** The adapter provides a consistent 2.60× to 2.66× speed-up on the
measured cells. Combined with the LREE-teacher capture table, this gives a
clean deployment trade-off: near-teacher AUPRC at materially lower head-level
inference cost.

## C3 Source Artifacts

- `models/rel_distill_adapter.py`
- `scripts/train_distill_adapter.py`
- `scripts/run_idea2c_2b_distill_5seed.sh`
- `scripts/aggregate_idea2c_2b_distill.py`
- `scripts/bench_inference_speed.py`
- `artifacts/tables/idea2c_distill_full_benchmark.md`
- `artifacts/tables/idea2c_2b_distill_full_benchmark.md`
- `artifacts/tables/idea2d_2b_speed_benchmark.md`

---

# Integrated Result-to-Contribution Alignment

| Paper claim | Supporting evidence | Interpretation |
|---|---|---|
| RAER safely augments frozen fraud bases | C1 contract proofs; base-freeze SHA checks; 8/8 positive AUPRC cells | The framework solves deployment safety and relation reasoning simultaneously. |
| Relation evidence matters most under base headroom | C1 base-lift table; no-proto and gate ablations | Weak bases receive large rescue; saturated bases show smaller changes. |
| Auxiliary losses are unnecessary | C1 loss falsification table | The canonical method is structurally regularized and cls-only. |
| Learned evidence is a representation-level upgrade | C2 19/32 wins; all YelpChi cells significant | The extractor, not a new loss, accounts for the next performance jump. |
| LREE changes the role of prototypes | C2 module ablation | A learned encoder can absorb some prototype functionality that was explicit in C1. |
| Lightweight deployment is feasible | C3 LREE-teacher distillation and speed table | The adapter recovers near-teacher performance at about 2.6× head-level speed-up. |

## Recommended Paper Framing

The completed paper should present the contributions as:

1. **Framework**: CoVER-REL is a contract-preserving relation-aware residual
   reasoner for frozen fraud bases.
2. **Representation**: LREE learns the relation evidence under the same
   contracts and improves the framework without changing the downstream
   reasoner.
3. **Deployment**: Flash-RAER statically distills the trained teacher into a
   compact adapter, preserving the bounded residual interface and providing
   a measured speed/accuracy trade-off.

Avoid claiming OPD in the completed-results version. Future G-OPD / Honest-OPD
can be added later as a fourth or replacement contribution after implementation
and matched ablations.
