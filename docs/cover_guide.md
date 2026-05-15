# CoVER-REL Technical Guide

This guide describes the final CoVER implementation state. It intentionally omits old speculative tasks and focuses on reproducing or extending the converged method.

## Current Method State

- Main quantitative model: **CoVER-REL-Gate**
- LLM-assisted research extension: **CoVER-REL-Judge**
- Base detector used for final results: **fresh BWGNN**
- Final datasets: **YelpChi** and **Amazon**

Do not restart CoVER-DIR or CoVER-LIFT as the main route. They are negative or diagnostic routes that motivate the final relation-evidence framing.

## Artifact Hierarchy

Key artifact roots:

```text
artifacts/
├── checkpoints/
│   └── {dataset}/bwgnn/{run}/seed_{seed}/
├── logs/
│   └── {dataset}/bwgnn/{run}/seed_{seed}/
├── results/
│   └── {dataset}/bwgnn/{run}/seed_{seed}/
├── relation_features/
│   └── {dataset}/bwgnn/seed_{seed}/{relation_set}/
├── judge_packets/
│   └── {dataset}/bwgnn/cover_rel_judge/seed_{seed}/
├── tables/
└── reports/
```

Paper consolidation artifacts:

```text
artifacts/tables/paper_main_results.md
artifacts/tables/paper_relation_ablation.md
artifacts/tables/paper_gate_judge_summary.md
artifacts/tables/paper_negative_routes.md
artifacts/paper/method_section_draft.md
artifacts/paper/results_narrative.md
artifacts/paper/appendix_failure_routes.md
```

## Final Runs

Use these runs for final reporting:

| Dataset | Purpose | Run |
|---|---|---|
| YelpChi | fresh baseline | `base` |
| YelpChi | best-single relation | `qwen_directional_t200_cover_rel_rur_nollm` |
| YelpChi | main model | `cover_rel_anchor_gate_nollm` |
| YelpChi | Judge extension | `cover_rel_judge_rur_strength_gate` |
| Amazon | fresh baseline | `base` |
| Amazon | best-single relation | `cover_rel_uvu_nollm` |
| Amazon | main model | `cover_rel_anchor_gate_nollm` |
| Amazon | Judge extension | `cover_rel_judge_uvu_strength_gate` |

The main paper claim should use CoVER-REL-Gate. CoVER-REL-Judge is the explanation-oriented LLM-assisted extension.

## Dataset Schemas

YelpChi:

- `RUR`: same user reviews
- `RSR`: same product, same star rating reviews
- `RTR`: same product, same month reviews

Amazon:

- `UPU`: users reviewing at least one same product
- `USU`: users with at least one same star rating within one week
- `UVU`: users with top-5% TF-IDF review text similarity

The method must remain schema-driven. Do not hardcode YelpChi-specific RUR logic into general code.

## Relation Feature Extraction

For every node and relation, CoVER-REL builds anonymous feature evidence from the available `.mat` data. There is no raw review text or raw metadata in the final pipeline.

Per-relation statistics:

- relation degree
- log degree / degree bucket
- relation neighbor feature mean
- node-to-neighbor feature deviation
- neighbor cosine consistency
- z-score outlier count or fraction
- top anonymous feature deviation dimensions
- train-only fraud prototype distance
- train-only benign prototype distance
- fraud-vs-benign prototype margin

Prototype constraints:

- fraud and benign prototypes use train labels only;
- val/test labels must not be used;
- metadata must record `prototype_labels=train_only` and `test_label_used=false`;
- prototype-derived evidence may be used as score-blind diagnostic evidence, but target labels must never enter packets or prompts.

Expected relation feature artifacts:

```text
artifacts/relation_features/{dataset}/bwgnn/seed_{seed}/{relation_set}/rel_stats.pt
artifacts/relation_features/{dataset}/bwgnn/seed_{seed}/{relation_set}/rel_tokens.jsonl
artifacts/relation_features/{dataset}/bwgnn/seed_{seed}/{relation_set}/rel_feature_meta.json
```

## Gate Modes

The reasoner supports several relation fusion modes:

- `single`: use one relation expert.
- `concat`: concatenate all relation statistics and encode them jointly.
- `anchor_gate`: preserve an anchor relation and sparsely gate optional relation experts.
- `base_additive_gate`: add gated relation experts to a base relation representation.

Final model:

- YelpChi: `anchor_gate` with anchor relation `RUR`
- Amazon: `anchor_gate` with anchor relation `UVU`

Interpretation:

- YelpChi is RUR-concentrated. RUR-only is slightly higher in AUPRC, while anchor-gate remains within a narrow margin and reduces near-cap residual behavior.
- Amazon is UVU-centered but more distributed. Anchor-gate improves over UVU-only.

## CoVER-REL-Judge

CoVER-REL-Judge adds a score-blind LLM evidence judge on top of CoVER-REL-Gate.

Pipeline:

1. Build judge packets from relation evidence.
2. Run local Qwen judge offline at inference/evaluation time.
3. Verify structured JSON outputs.
4. Encode accepted judge records into judge features.
5. Fuse judge features through conservative gated residual fusion.

Judge packet fields include:

- dataset name
- relation schema
- primary relation
- relation evidence buckets
- relation token list
- gate evidence buckets
- graph diagnostic evidence buckets
- allowed support/counter fields

The judge output schema is:

```json
{
  "node_id": 0,
  "verdict": "fake | real | uncertain",
  "evidence_strength": "weak | moderate | strong",
  "key_relation": "RUR",
  "supporting_evidence": [],
  "counter_evidence": [],
  "uncertainty_factors": [],
  "short_explanation": ""
}
```

Fusion:

```text
final_logit = relation_logit + alpha_llm * delta_llm
```

If a judge output is missing or rejected:

```text
alpha_llm = 0
final_logit = relation_logit
```

The final Judge variant uses conservative strength-aware alpha. This is for stability and interpretability, not for a large metric gain.

## Safety Constraints

Forbidden fields for any LLM packet or prompt:

- `base_score`
- `base_prob`
- `base_probability`
- `base_logit`
- `confidence`
- base prediction
- final prediction
- target label
- train/val/test label
- split identity
- FN/FP/base-error status
- ground truth

Training and evaluation constraints:

- rejected judge outputs do not enter fusion training;
- `short_explanation` is not used for loss;
- Stage3 training consumes accepted judge features and does not call Qwen;
- relation prototypes must be train-only;
- test labels must never be used in relation feature extraction, packet generation, prompt construction, or verifier decisions;
- claims must be artifact-backed.

## Final Results To Report

| Dataset | Main Model | Gate Delta AUPRC vs Fresh BWGNN | Judge Delta AUPRC vs Gate |
|---|---|---:|---:|
| YelpChi | CoVER-REL-Gate | +0.026585 | +0.000010 |
| Amazon | CoVER-REL-Gate | +0.003508 | +0.000224 |

Report these as improvements over fresh BWGNN. Do not claim state of the art without explicit SOTA comparison.

## Do Not Do Next

- Do not tune CoVER-LIFT as the main method.
- Do not use canonical ERR hidden states as the main signal.
- Do not expose base score, probability, logit, confidence, or prediction to the LLM.
- Do not present CoVER-REL-Judge as the main metric gain.
- Do not claim raw review text or raw metadata is used.
- Do not turn rejected judge outputs or short explanations into training targets.
- Do not add new result numbers unless they come from saved artifacts.

## Extension Guidance

If asked to improve metrics, inspect relation feature and gate diagnostics first:

- relation utility by schema;
- near-cap residual behavior;
- relation gate distributions;
- prototype distance distributions;
- feature scale and z-score distributions.

Only after relation/gate diagnostics are understood should LLM prompts or Judge fusion be revisited.
