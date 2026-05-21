# Work 2 Research Plan: RAER-FD

## Working Title

RAER-FD: Relation-Aware Evidence Reasoning and Contract-Budgeted Residual Distillation for Graph Fraud Detection

## One-Sentence Thesis

RAER-FD treats graph fraud detection errors as relation-conditioned residual correction problems: given a frozen detector, it learns when, through which relation evidence, and by how much the detector should be corrected.

## Research Motivation

Graph fraud detection differs from ordinary node classification because fraud nodes can be camouflaged by normal neighbors, form coordinated suspicious groups, and exploit different relation types in different ways. A base GNN may therefore be correct on most nodes but systematically miscalibrated on nodes where relation evidence contradicts its internal representation.

The central question is:

> Can relation evidence be externalized into an interpretable and bounded correction interface for existing fraud detectors?

## Research Objectives

1. Build a relation-aware evidence reasoning teacher on top of frozen base detectors.
2. Replace hand-crafted evidence with a learnable score-blind relation evidence extractor.
3. Compress the teacher into a lightweight student while preserving useful residual intervention behavior.
4. Show that the framework is selective: it improves weaker bases but stays near identity on already relation-aware strong bases.

## Final Contributions

### Contribution 1: Two-Stage RAER Distillation Framework

RAER-FD proposes a two-stage framework:

1. Teacher stage: learn a relation-aware bounded residual on top of a frozen base detector.
2. Student stage: distill the teacher's correction behavior into a compact adapter.

Formal core:

```text
z_i = b_i + Delta_rel_i
Delta_rel_i = delta_max * tanh(sum_r g_{i,r} Head_r(h_{i,r}))
```

Where:

- `b_i` is the frozen base logit.
- `h_{i,r}` is relation evidence embedding.
- `g_{i,r}` is schema-aware relation gate.
- `Delta_rel_i` is bounded residual correction.

### Contribution 2: Learnable Relation Evidence Extractor

LREE learns score-blind per-relation evidence from:

- raw node features,
- per-relation adjacency,
- neighbor feature means,
- relation-specific GCN encodings,
- train-only class prototypes.

The extractor never sees:

- base logits,
- base embeddings,
- validation labels,
- test labels.

This preserves the key contract: relation evidence should explain and guide correction without leaking target information.

### Contribution 3: Contract-Budgeted Residual Loss

CBR constrains student residual allocation during distillation.

Intuition:

- If the teacher barely intervenes on a node, the student should conserve residual budget.
- If the teacher strongly intervenes, the student may allocate correction capacity.

Core signal:

```text
sensitivity_i = |teacher_logit_i - base_logit_i| / delta_max
waste_i = |student_logit_i - base_logit_i| / delta_max * weight(1 - sensitivity_i)
```

CBR turns distillation from pure logit matching into contract-aware residual allocation.

## Research Questions

| RQ | Question | Expected Evidence |
|---|---|---|
| RQ1 | Does relation-aware residual reasoning improve frozen base detectors? | RAER-HC vs base across 8 cells |
| RQ2 | Does LREE improve over hand-crafted evidence? | RAER-LREE vs RAER-HC |
| RQ3 | Which components are load-bearing? | evidence group, gate, bounded residual ablations |
| RQ4 | Can a lightweight student preserve teacher behavior? | Flash-RAER teacher capture and speed |
| RQ5 | Does CBR improve residual distillation? | CBR-Flash vs vanilla distillation variants |
| RQ6 | What happens on a strong relation-aware base? | PriorF-GNN saturation check |

## Claim-Evidence Matrix

| Claim | Evidence | Status |
|---|---|---|
| RAER improves weaker frozen bases through relation evidence residuals. | `artifacts/tables/idea1_ablation_FINAL_7cell.md` | Supported |
| LREE is stronger than hand-crafted evidence in most settings. | `artifacts/tables/idea2b_learned_vs_canonical_8cell_5seed.md` | Supported |
| LREE internal components should not be overclaimed individually. | `artifacts/tables/idea2b_ablation_4base_5seed.md` | Caveated |
| CBR improves student residual distillation across metrics. | `artifacts/tables/c3_4metric_8cell_5seed.md` | Supported |
| Student adapter gives deployment value. | `artifacts/tables/idea2d_2b_speed_benchmark.md` | Supported |
| RAER-FD is conservative on strong relation-aware bases. | `artifacts/logs/yelpchi/priorfgnn/priorfgnn_idea2b_404020_gpu/seed_42/phase2_summary.json` | Supported as sanity check |

## PriorF-GNN Saturation Interpretation

The PriorF-GNN adaptation result should not be framed as failure.

Observation:

- PriorF-GNN base test AUPRC: `0.773319`.
- PriorF-GNN + RAER-LREE test AUPRC: `0.773347`.
- Best epoch: `1`.
- Gate stayed uniform.
- Residual became a tiny near-constant logit shift.

Interpretation:

PriorF-GNN already internalizes the relation-aware discrepancy evidence that RAER-FD would otherwise use for correction. Therefore, RAER-FD finds little complementary evidence and stays near identity.

Claim:

> RAER-FD is selective and conservative: it corrects when relation evidence provides complementary information, but avoids inventing artificial gains on strong relation-aware detectors.

## LREE Caching Policy

### Cacheable Across Base Models

Hand-crafted relation evidence can be cached across base models under the same:

- dataset,
- split,
- seed,
- train mask,
- relation schema,
- feature preprocessing.

Reason: it uses raw graph/features and train-only prototypes, not base outputs.

### Partially Cacheable

LREE fixed inputs can be cached:

- normalized per-relation adjacency,
- neighbor feature means,
- train-only prototype statistics.

Current implementation builds these online via `extractor.prepare()`, but conceptually they are base-independent.

### Must Be Retrained Per Base

LREE parameters must be retrained for each frozen base.

Reason:

- LREE inputs are score-blind,
- but its gradients come from the residual objective `base_logit + Delta_rel`,
- so the learned extractor adapts to the current base's remaining correction space.

Canonical statement:

> LREE is score-blind in input but base-conditioned through training.

## Experimental Organization

### Main Tables

| Table | Content |
|---|---|
| Table 1 | Main base vs RAER-HC vs RAER-LREE results |
| Table 2 | LREE vs hand-crafted evidence |
| Table 3 | Component ablations |
| Table 4 | Student distillation and CBR |
| Table 5 | Strong-base saturation check |

### Figures

| Figure | Purpose |
|---|---|
| Fig. 1 | Overall RAER-FD framework |
| Fig. 2 | Relation-aware residual teacher |
| Fig. 3 | LREE evidence extractor |
| Fig. 4 | CBR residual budget diagram |
| Fig. 5 | PriorF-GNN saturation analysis |

## Known Limitations

- RAER-FD is not expected to improve every strong base.
- LREE learned parameters are not universally reusable across base detectors.
- On saturated datasets or strong relation-aware detectors, performance gains may be small.
- Some internal LREE submodules are not individually significant in all cells.

## Final Research Positioning

RAER-FD should be presented as:

- not a new base GNN,
- not a universal score booster,
- not an LLM reasoning module,
- but a relation-evidence-conditioned residual intervention and distillation framework.

This positioning is stronger and more defensible than claiming universal SOTA improvements.
