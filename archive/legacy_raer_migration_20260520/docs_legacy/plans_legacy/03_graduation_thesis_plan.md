# Graduation Thesis Writing Plan

## Thesis Theme

The thesis should present a coherent research progression:

> From relation-aware structural discrepancy modeling to relation-evidence residual correction for graph fraud detection.

PriorF-GNN is the first step: it builds a strong detector by internalizing relation-aware structural discrepancy. RAER-FD is the second step: it externalizes relation-aware evidence into a correction, explanation, and distillation interface.

## Proposed Thesis Title

基于关系感知证据推理与残差纠偏的图欺诈检测方法研究

Alternative English title:

Relation-Aware Evidence Reasoning and Residual Correction for Graph Fraud Detection

## Chapter Structure

### Chapter 1: Introduction

#### Goal

Motivate why graph fraud detection requires relation-aware evidence instead of blind neighborhood aggregation.

#### Main Logic

1. Graph fraud detection is important and difficult.
2. Fraud nodes use camouflage and coordinated behavior.
3. Standard GNN smoothing can dilute anomaly signals.
4. Existing methods mostly build stronger detectors but rarely study how to correct existing detectors using explicit relation evidence.
5. This thesis studies relation-aware evidence in two stages:
   - PriorF-GNN: evidence inside detector propagation.
   - RAER-FD: evidence outside detector as residual correction.

#### Contributions

1. A structure-discrepancy-aware graph fraud detector, PriorF-GNN.
2. A two-stage relation-aware evidence residual correction framework, RAER-FD.
3. A learnable relation evidence extractor, LREE.
4. A contract-budgeted residual distillation objective, CBR.
5. A systematic empirical study including strong-base saturation analysis.

#### Key Citations

- CARE-GNN
- GraphConsis
- PC-GNN
- BWGNN
- R-GCN
- Knowledge distillation
- Calibration / residual correction literature

### Chapter 2: Related Work

#### Section 2.1 Graph Fraud and Graph Anomaly Detection

Discuss classic and recent GAD/fraud methods.

Positioning:

- Existing methods detect anomalies directly.
- This thesis also studies how to correct and distill existing detectors.

#### Section 2.2 Camouflage, Heterophily, and High-Frequency Signals

Discuss why fraud graphs violate homophily assumptions.

Positioning:

- PriorF-GNN explicitly models structural discrepancy.
- RAER-FD uses relation evidence to correct detector decisions.

#### Section 2.3 Multi-Relation Graph Learning

Discuss R-GCN, HAN, HGT, and relation-aware fraud models.

Positioning:

- Existing methods use relations inside representation learning.
- RAER-FD uses relations as external evidence for bounded residual intervention.

#### Section 2.4 Knowledge Distillation and Calibration

Discuss standard KD, graph KD, and calibration.

Positioning:

- Standard KD matches logits or representations.
- CBR matches the contract of residual budget allocation.

### Chapter 3: PriorF-GNN

#### Goal

Present Work 1 as the foundation proving relation-aware structural discrepancy is valuable.

#### Method Sections

1. Problem definition.
2. HSD structural discrepancy prior.
3. ASDA adaptive discrepancy-aware routing.
4. Local-global dual-branch fusion.
5. SDCL high-discrepancy supervised contrastive learning.

#### Experiment Sections

1. Main results on Amazon/YelpChi.
2. Ablation: Full, Raw GNN, SCRE, MLP-only, GNN-only, No-SDCL.
3. Label scarcity results.
4. Schedule sensitivity.
5. Qualitative representation analysis.

#### Key Message

PriorF-GNN shows that relation-aware discrepancy is not an auxiliary trick but a core signal for graph fraud detection.

### Chapter 4: RAER-FD Method

#### Goal

Present Work 2 as the main method contribution.

#### Section 4.1 Problem Formulation

Define:

- multi-relation graph,
- frozen base detector,
- base logit `b_i`,
- relation evidence `E_{i,r}`,
- bounded residual correction `Delta_rel_i`.

#### Section 4.2 RAER Teacher

Explain:

- frozen base,
- relation evidence experts,
- schema-aware gate,
- bounded tanh residual,
- cls-only objective.

Core equation:

```text
z_i = b_i + Delta_rel_i
Delta_rel_i = delta_max * tanh(sum_r g_{i,r} Head_r(h_{i,r}))
```

#### Section 4.3 LREE

Explain:

- score-blind input contract,
- train-only prototype contract,
- per-relation GCN signal,
- neighbor feature evidence,
- prototype-relative evidence.

Important caveat:

LREE input is base-independent, but learned parameters are base-conditioned through the residual training objective.

#### Section 4.4 Student Distillation

Explain:

- teacher behavior,
- lightweight adapter,
- distillation targets,
- deployment motivation.

#### Section 4.5 CBR Loss

Explain:

- teacher residual sensitivity,
- student residual budget,
- wasted budget penalty,
- why this differs from standard KD.

### Chapter 5: Experiments

#### Goal

Answer the research questions with a clean evidence chain.

#### Section 5.1 Experimental Setup

Include:

- datasets: YelpChi, Amazon,
- base models: BWGNN, SAGE, GCN, GAT, PriorF-GNN strong base,
- split protocol,
- metrics: AUROC, AUPRC, Macro-F1, G-Means,
- seeds.

#### Section 5.2 RAER Main Results

Show:

- base vs RAER-HC,
- base vs RAER-LREE,
- strongest improvements on weak/headroom cells.

#### Section 5.3 LREE Analysis

Show:

- LREE vs hand-crafted evidence,
- cross-cell win count,
- caveat that internal submodules are not all universally load-bearing.

#### Section 5.4 Component Ablation

Show:

- evidence groups,
- schema gate,
- residual bound,
- cls-only objective.

#### Section 5.5 Distillation and CBR

Show:

- student vs teacher,
- CBR vs vanilla distillation,
- speedup and parameter efficiency.

#### Section 5.6 Strong-Base Saturation Check

Use PriorF-GNN 40/20/40 adaptation.

Report:

- PriorF-GNN base AUPRC: `0.773319`.
- PriorF-GNN + RAER-LREE AUPRC: `0.773347`.
- best epoch: `1`.
- residual near identity.

Interpretation:

> RAER-FD does not invent artificial gains on a strong relation-aware base; it behaves as a selective and conservative residual intervention framework.

### Chapter 6: Conclusion

#### Restate Main Findings

1. Relation-aware discrepancy is central to graph fraud detection.
2. PriorF-GNN internalizes this signal in message passing.
3. RAER-FD externalizes this signal as evidence-conditioned correction.
4. LREE learns useful relation evidence without score leakage.
5. CBR improves residual distillation.

#### Limitations

- Strong relation-aware bases leave little correction headroom.
- LREE learned parameters must be retrained per frozen base.
- Current framework focuses on transductive public fraud benchmarks.

#### Future Work

- hard-node-only residual correction,
- stronger teacher-side CBR regularization,
- cross-dataset evidence transfer,
- online detector monitoring and correction.

## Figure Plan

| ID | Type | Content | Priority |
|---|---|---|---|
| Fig. 1 | Research roadmap | PriorF-GNN internal evidence to RAER-FD external evidence | High |
| Fig. 2 | PriorF-GNN architecture | HSD, ASDA, SDCL | High |
| Fig. 3 | RAER-FD architecture | frozen base, LREE, RAER teacher, student | High |
| Fig. 4 | LREE detail | relation GCN, neighbor mean, prototypes | Medium |
| Fig. 5 | CBR illustration | teacher sensitivity and student residual budget | High |
| Fig. 6 | Strong-base saturation | weak base improves, PriorF-GNN stays near identity | Medium |

## Table Plan

| ID | Content | Source |
|---|---|---|
| Table 1 | Dataset statistics | project configs / datasets |
| Table 2 | PriorF-GNN main results | PriorF-GNN result summaries |
| Table 3 | RAER-FD main RAER results | `idea1_ablation_FINAL_7cell.md` |
| Table 4 | LREE vs hand-crafted evidence | `idea2b_learned_vs_canonical_8cell_5seed.md` |
| Table 5 | Distillation and CBR | `c3_4metric_8cell_5seed.md` |
| Table 6 | PriorF-GNN saturation check | `priorfgnn_idea2b_404020_gpu` JSONs |

## Writing Order

1. Write Chapter 4 method first because the final method is now clear.
2. Write Chapter 5 experiments using the claim-evidence matrix.
3. Write Chapter 3 PriorF-GNN as the foundation chapter.
4. Write Chapter 2 related work after the method wording is stable.
5. Write Chapter 1 introduction last, so the story matches final claims.
6. Write Chapter 6 conclusion and limitations.

## Acceptance Checklist

- Every contribution appears in Introduction, Method, Experiments, and Conclusion.
- Every metric claim maps to a result artifact.
- No deleted or archived route is used as a positive contribution.
- PriorF-GNN result is framed as saturation/selectivity, not failure.
- LREE caching policy is clearly stated.
- Strong claims such as SOTA are only used if external comparison is fully verified.

## Final Thesis Message

The thesis should leave the reader with one clear idea:

> Relation-aware evidence is a reusable scientific object in graph fraud detection: it can guide message passing, correct frozen detectors, explain residual interventions, and constrain distillation.
