# CoVER-REL: Schema-Aware Relation Evidence Reasoning

## Abstract

CoVER-REL is the final CoVER method for fake review detection on YelpChi and Amazon fraud graphs. The central finding is that the useful evidence in these `.mat` benchmarks is not free-form LLM rationale, base-score-driven explanation, or canonical ERR hidden-state distillation. It is the relation-aware anonymous feature evidence induced by the graph construction schema. CoVER-REL treats the base BWGNN as a structural prior, extracts relation-wise evidence views from anonymous handcrafted features and multi-relation neighborhoods, and learns bounded evidence-conditioned interventions through a schema-aware relation gate. CoVER-REL-Gate is the main quantitative model. CoVER-REL-Judge is a research extension that adds a score-blind LLM evidence judge for structured judgement and explanation without making the LLM the main performance source.

## Contributions

1. **Evidence-source reframing.** CoVER reframes fake review detection from generic GNN residual correction or LLM rationale distillation into schema-aware relation evidence reasoning. Rather than asking a student to imitate the base GNN or distill free-form LLM explanations, CoVER identifies relation-wise anonymous feature deviation, neighbor consistency, and train-only prototype margins as the actionable evidence carried by multi-relation fraud graph construction.

2. **Base-prior schema-aware relation intervention.** CoVER-REL introduces a base-prior relation evidence reasoner that treats BWGNN as a frozen structural prior and learns bounded residual interventions from relation-specific evidence experts. Its sparse schema-aware gate preserves the dominant relation evidence when utility is concentrated, while allowing conditional contribution from weaker relations when utility is distributed. This explains the observed patterns: RUR-dominant YelpChi and UVU-centered but more distributed Amazon.

3. **Contract-verified score-blind LLM judgement.** CoVER-REL-Judge introduces a constrained LLM evidence judge that reasons over score-blind relation evidence rather than base scores or predictions. It produces structured verdicts and explanations through contract verification, while fusion is gated and conservative so that LLM judgement acts as evidence-conditioned assistance rather than uncontrolled replacement of the relation reasoner. This gives interpretability without turning the method into score-following LLM rationalization.

## Method Overview

### 1. Stage1 Base BWGNN Structural Prior

The base model is a fresh BWGNN detector trained on YelpChi or Amazon. It provides a structural prior through node embeddings and base logits. In CoVER-REL, this prior is not a teacher to imitate and is not exposed to the LLM judge. Stage3 uses the base logit as a detached prior and learns bounded evidence-conditioned residual interventions.

### 2. Relation-Aware Evidence Extraction

For each dataset relation schema, CoVER-REL builds one evidence view per relation from anonymous handcrafted features and relation neighborhoods.

- YelpChi schema: `RUR`, `RSR`, `RTR`
- Amazon schema: `UPU`, `USU`, `UVU`

Each relation evidence view includes degree, feature deviation from relation neighbors, neighbor consistency, z-score outlier count, train-only fraud prototype distance, train-only benign prototype distance, and fraud-vs-benign prototype margin. These evidence views are score-blind and do not use val/test labels.

### 3. Schema-Aware Relation Expert Gate

CoVER-REL-Gate encodes each relation evidence view with a relation-specific expert. It then uses a schema-aware gate to preserve the strongest relation when relation utility is concentrated and conditionally include weaker relations when the dataset relation utility is distributed.

This is the main quantitative model:

```text
CoVER-REL-Gate = base BWGNN prior
               + relation-aware anonymous feature evidence
               + sparse schema-aware bounded residual reasoner
```

It should be presented as the primary performance model.

### 4. Optional Score-Blind LLM Evidence Judge

CoVER-REL-Judge runs after relation evidence has been made meaningful. It gives a local LLM only score-blind relation evidence packets and asks for structured JSON:

```json
{
  "verdict": "fake | real | uncertain",
  "evidence_strength": "weak | moderate | strong",
  "key_relation": "...",
  "supporting_evidence": ["..."],
  "counter_evidence": ["..."],
  "uncertainty_factors": ["..."],
  "short_explanation": "..."
}
```

The judge must not see base scores, probabilities, logits, confidence, base predictions, target labels, FN/FP status, split identity, or ground truth. Rejected judge outputs are excluded from fusion training. The short explanation is human-facing only and is not used in loss.

### 5. Gated Residual Fusion and Explanation

Accepted judge outputs are encoded as structured features and fused conservatively:

```text
final_logit = relation_logit + alpha_llm * delta_llm
```

If judge output is missing or rejected:

```text
alpha_llm = 0
final_logit = relation_logit
```

CoVER-REL-Judge is therefore an LLM-assisted research extension for structured interpretability, not the main metric gain.

## Final Results

Stage1 fresh BWGNN is the comparison baseline.

| Dataset | Main Model | Gate Delta AUPRC vs Fresh BWGNN | Judge Delta AUPRC vs Gate | Interpretation |
|---|---|---:|---:|---|
| YelpChi | CoVER-REL-Gate | +0.026585 | +0.000010 | RUR-concentrated relation utility |
| Amazon | CoVER-REL-Gate | +0.003508 | +0.000224 | UVU-centered but more distributed relation utility |

Relation findings:

| Dataset | Strongest Single Relation | Best-Single Delta AUPRC | CoVER-REL-Gate Delta AUPRC | Interpretation |
|---|---|---:|---:|---|
| YelpChi | RUR | +0.027099 | +0.026585 | Gate is slightly below RUR-only but reduces near-cap residual behavior |
| Amazon | UVU | +0.003159 | +0.003508 | Gate improves over UVU-only, indicating distributed relation utility |

These results support CoVER-REL as a schema-aware relation evidence framework rather than a YelpChi-specific RUR method. They show improvement over the fresh BWGNN baseline, not state-of-the-art claims.

## What Changed From Earlier CoVER Versions

### CoVER-DIR / CV-SCD

CoVER-DIR and CV-SCD were diagnostic routes. They did not yield stable metric gains because evidence-supported FN/FP correction was too sparse. FP negative correction was especially unreliable because most base false positives remained fraud-dominant under score-blind structural evidence.

### CoVER-LIFT

CoVER-LIFT tested canonical ERR hidden-state distillation. Student latent alignment could be high, but AUPRC and ranking did not improve. This showed that the bottleneck was not hidden-state alignment alone: canonical ERR hidden states were too thin and not sufficiently fraud-discriminative.

### CoVER-REL

CoVER-REL is the final effective evidence source. It shifts the method from generic structural tokens or ERR latents to schema-aware relation evidence extracted from anonymous features and multi-relation graph construction. The LLM judge is used only after this relation evidence has been made meaningful.

## Final Positioning

- **CoVER-REL-Gate**: main quantitative model and deployment-friendly relation-only detector.
- **CoVER-REL-Judge**: LLM-assisted research extension for score-blind structured judgement and explanation.
- **Do not overclaim**: Judge is not the main performance source, raw review text/metadata is not used, and the method should be described as improving over fresh BWGNN rather than as state of the art.
