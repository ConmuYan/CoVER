# CoVER-DIR: Directional Contrastive Evidence Teacher

> **Date**: 2026-05-13
> **Status**: Design Approved
> **Task**: 8.7

---

## Problem Statement

Task 8.6 results show CoVER does not improve over base BWGNN on YelpChi:

- Δ ROC-AUC ≈ 0
- Δ MF1@val ≈ 0
- Qwen 4B generates 95-96.5% `structural_discrepancy` ERRs
- Residual shift ≈ 0
- L_signed / L_corr cannot function (uniform evidence pattern)

**Root cause**: Stage2 teacher task is too weak — single-node risk_type classification produces degenerate output.

**Solution**: Replace with score-blind directional contrastive evidence reasoning.

---

## Goals

1. Keep score-blind: no base_score/logit/prob/confidence to LLM
2. No test labels in any pipeline stage
3. No base prediction exposed to LLM
4. LLM outputs direction-aware ERR (increase_risk / decrease_risk / uncertain)
5. CV-SCD uses evidence_direction for selective correction
6. Microbenchmark first — no t200 unless quality gates pass

---

## Architecture

```
Stage1 (BWGNN)
    ↓ base_logits, embeddings, extras
Stage2 (Directional Teacher)
    ├── prototypes.py     → fraud/benign prototype banks (train-only)
    ├── retrieval.py      → hybrid token retrieval per node
    ├── adapter.py        → evidence tokens (6 categories)
    ├── prompt.py         → contrastive_directional mode
    ├── schema.py         → directional ERR fields
    └── verifier.py       → direction-aware validation
    ↓ directional ERR cache
Stage3 (CV-SCD-DIR)
    └── losses.py         → L_det + L_err(dir) + L_signed(dir) + L_corr(dir)
```

---

## Step A: Directional ERR Schema

**File**: `evidence/schema.py`

### New Fields

| Field | Type | Default (backward compat) |
|-------|------|---------------------------|
| `evidence_direction` | `Literal["increase_risk", "decrease_risk", "uncertain"]` | `"uncertain"` |
| `evidence_strength` | `Literal["weak", "moderate", "strong"]` | `"weak"` |
| `uncertainty_factors` | `List[str]` | `[]` |
| `contrastive_basis` | `Optional[Dict]` | `None` |

### Backward Compatibility

Old ERR without new fields → defaults applied at parse time:
```python
err.evidence_direction = err.evidence_direction or "uncertain"
err.evidence_strength = err.evidence_strength or "weak"
```

### Hard Constraints

- `summary` remains human-inspection only, never enters model or loss
- All new fields must pass score-blind check
- Internal bank names containing `fn`/`fp`/`base_error` must never be serialized into `teacher_payloads.jsonl`, `prompts.jsonl`, or any LLM-visible JSON key

---

## Step B: Prototype and Retrieval

**New files**: `evidence/prototypes.py`, `evidence/retrieval.py`

### Prototype Banks (train-only)

| Bank | Source | Contents |
|------|--------|----------|
| `fraud_prototype_bank` | Train positive nodes | Evidence tokens, distinctive patterns |
| `benign_prototype_bank` | Train negative nodes | Evidence tokens, distinctive patterns |
| `train_fn_bank` | Train false negatives | Nodes where base model missed fraud |
| `train_fp_bank` | Train false positives | Nodes where base model flagged benign |

**Hard constraint**: All banks built from train nodes only. No test labels.

### Prototype Summaries

For each prototype bank, compute:
- `fraud_prototype_summary`: aggregate token frequencies, **distinctive_tokens**
- `benign_prototype_summary`: aggregate token frequencies, **distinctive_tokens**
- `normal_structure_summary`: median/mode of structural tokens across train

**Distinctive tokens**: computed from train-only token statistics using class contrast or IDF-weighted log odds. Not just frequent tokens.

### Retrieval Metric: Hybrid Jaccard + IDF-Cosine

1. **Node representation**:
   - `evidence_token_set: Set[str]`
   - `evidence_token_vector: sparse binary or IDF-weighted`

2. **Candidate recall**:
   - Jaccard similarity over `evidence_token_set`
   - Top `top_k_recall=50` from each bank

3. **Reranking**:
   - IDF-weighted cosine on `evidence_token_vector`
   - IDF computed from train nodes only
   - `score = 0.4 * jaccard + 0.6 * idf_cosine`

4. **Final output**:
   - `top_k_final=3` nearest fraud cases
   - `top_k_final=3` nearest benign cases
   - `top_k_final=3` nearest train FN cases (prompt-visible as "fraud_like_reference_cases")
   - `top_k_final=3` nearest train FP cases (prompt-visible as "benign_like_reference_cases")

### LLM-Facing Terminology

**Critical**: Do not expose FN/FP/base-error terminology to LLM.

| Internal | Prompt-visible |
|----------|----------------|
| `train_fn_bank` | `fraud_like_reference_cases` |
| `train_fp_bank` | `benign_like_reference_cases` |

LLM must not see: base prediction, base error status, target label.

### Output Categorical Summaries

- `closer_to_fraud_prototype`: low / medium / high / unknown
- `closer_to_benign_prototype`: low / medium / high / unknown
- `fraud_prototype_matching_tokens: List[str]`
- `benign_prototype_matching_tokens: List[str]`
- `prototype_conflict_level`: low / medium / high / unknown
- `similar_fraud_like_reference_count_bucket`
- `similar_benign_like_reference_count_bucket`

### Saved Stats

`prototype_stats.json` and `retrieval_stats.json` must include:
```json
{
  "metric": "hybrid_token_jaccard_idf_cosine",
  "jaccard_weight": 0.4,
  "cosine_weight": 0.6,
  "top_k_recall": 50,
  "top_k_final": 3,
  "train_only_banks": true,
  "test_label_used": false,
  "score_visible_to_teacher": false,
  "target_label_visible_to_teacher": false,
  "base_prediction_visible_to_teacher": false
}
```

---

## Step C: Graph Evidence Language Tokens

**Files**: `evidence/adapter.py`, `evidence/vocab.py`

### Minimal High-Value Token Set (required for microbenchmark)

| Category | Tokens |
|----------|--------|
| **Spectral / BWGNN** | `HF_RATIO_TOP10`, `HF_RATIO_HIGH`, `HF_RATIO_LOW`, `BAND_ENERGY_CONFLICT_HIGH`, `LOW_HIGH_BAND_MISMATCH` |
| **Feature-structure conflict** | `FEAT_NEIGH_COS_BOTTOM10`, `EMB_NEIGH_COS_BOTTOM10`, `FEATURE_EMBED_DISAGREE_HIGH` |
| **Prototype relation** | `PROTO_FRAUD_CLOSE`, `PROTO_BENIGN_CLOSE`, `PROTO_CONFLICT_HIGH` |
| **Normal-structure deviation** | `NORMAL_STRUCTURE_DIST_HIGH`, `NORMAL_PATTERN_DEVIATION_HIGH` |
| **Clean-view / interference** | `INTERFERING_EDGE_RATIO_HIGH`, `CLEAN_VIEW_SHIFT_HIGH`, `RAW_TO_CLEAN_CONFLICT` |

### Optional Tokens (do not block microbenchmark)

| Category | Tokens | Notes |
|----------|--------|-------|
| **Geometry / curvature** | `LOCAL_CURVATURE_OUTLIER_HIGH`, `EDGE_CURVATURE_VAR_HIGH` | Lightweight proxy OK, record `method="proxy"` |

### Token Generation

- All tokens are score-blind
- Generated from percentile / rank / z-score bucket
- No raw base score, probability, logit, or confidence exposed

---

## Step D: Contrastive Directional Prompt

**File**: `evidence/prompt.py`

### New Prompt Mode

`prompt_mode: "contrastive_directional"`

### LLM Input

1. Target score-blind graph evidence tokens
2. Target categorical reasoning fields
3. `fraud_prototype_summary` (with distinctive_tokens)
4. `benign_prototype_summary` (with distinctive_tokens)
5. `normal_structure_summary`
6. Retrieved `fraud_like_reference_cases` (top 3)
7. Retrieved `benign_like_reference_cases` (top 3)
8. `allowed_support_ids`, `allowed_counter_ids`
9. Valid risk types, evidence_direction values, evidence_strength values

### Prompt Task

> Compare target evidence against fraud-like and benign-like structural prototypes.
> Decide evidence_direction, evidence_strength, risk_type.
> Select supporting/counter evidence and uncertainty_factors.
> Return strict JSON.

### Mandatory Disclaimer

> "You are not given any base model score, probability, logit, confidence, or prediction."

### Forbidden in Prompt

- Actual base_score / prob / logit / confidence
- Target label
- Base prediction
- FN/FP terminology

### Output JSON Schema

```json
{
  "risk_type": "...",
  "evidence_direction": "increase_risk|decrease_risk|uncertain",
  "evidence_strength": "weak|moderate|strong",
  "supporting_evidence": [...],
  "counter_evidence": [...],
  "uncertainty_factors": [...],
  "contrastive_basis": {
    "closer_prototype": "fraud|benign|mixed|unknown",
    "matched_fraud_patterns": [...],
    "matched_benign_patterns": [...]
  },
  "summary": "..."
}
```

---

## Step E: Direction-Aware Verifier

**File**: `evidence/verifier.py`

### Hard Checks (rejection)

1. `evidence_direction` must be valid enum value
2. `evidence_strength` must be valid enum value
3. `uncertainty_factors` must reference available fields/tokens
4. `contrastive_basis` fields must be valid if present
5. All new fields pass score-blind check

### Semantic Checks (warning only, no rejection)

6. If `evidence_direction == "increase_risk"`: supporting_evidence should be non-empty (unless `risk_type == "weak_or_uncertain_evidence"`)
7. If `evidence_direction == "decrease_risk"`: counter_evidence or uncertainty_factors should be non-empty
8. If `evidence_direction == "uncertain"`: evidence_strength should be weak OR risk_type should be weak_or_uncertain_evidence

**Design decision**: Direction/risk semantic consistency is logged as warning first, not immediate rejection. This allows the microbenchmark to proceed without strict semantic gating.

### Hard Constraints

- Do not use test labels in verifier
- Do not relax existing checks

---

## Step F: Microbenchmark

**New file**: `scripts/run_stage2_microbenchmark.py`

### Sampling (30 nodes, YelpChi seed 123)

| Category | Count | Source |
|----------|-------|--------|
| Train false negatives | 10 | Base model missed these frauds |
| Train false positives | 10 | Base model falsely flagged |
| Train high-loss | 5 | High training loss nodes |
| Val boundary | 5 | Near decision boundary |

### Comparison

| Condition | Prompts |
|-----------|---------|
| Current | `enhanced` prompt mode |
| Treatment | `contrastive_directional` prompt mode |

### Metrics

- Parse success rate
- Verifier acceptance rate
- Risk type distribution
- Evidence direction distribution
- Evidence strength distribution
- Supporting evidence entropy
- Counter evidence entropy
- Uncertainty factors distribution
- Structural discrepancy ratio
- **Direction error alignment** (offline audit only):
  - Train FN nodes → `increase_risk` is aligned
  - Train FP nodes → `decrease_risk` is aligned
  - Report alignment rate vs random (0.33)
- Score-blind audit
- Qwen vs rule agreement (if rule outputs available)

### Saved Artifacts

```
artifacts/reports/stage2_directional_microbenchmark.md
artifacts/tables/stage2_directional_microbenchmark.csv
current_prompts.jsonl
directional_prompts.jsonl
current_outputs.jsonl
directional_outputs.jsonl
```

### Pass Criteria

contrastive_directional must satisfy **all 4**:

1. `structural_discrepancy` ratio < 80%
2. Evidence direction contains both `increase_risk` and `decrease_risk`
3. Direction error alignment ≥ 0.55 (meaningfully above random 0.33)
4. Verifier acceptance rate ≥ 80%

### Failure Mode

If microbenchmark fails:
- Do NOT run t200
- Output failure diagnosis
- Suggest whether larger LLM or textual review features needed

### Failure Diagnosis Guidance

- If diversity passes but direction_error_alignment fails → diagnose prompt reasoning quality; direction field may be unreliable
- If direction_alignment passes but structural_discrepancy remains high → inspect whether risk_type is too coarse but direction is useful; direction field may still have value for CV-SCD
- If acceptance rate fails → check verifier strictness; may need to relax semantic warnings to logging-only

---

## Step G: Qwen t200 (conditional on Step F passing)

### Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | YelpChi |
| Seeds | 123, 456, 789 |
| trace_size | 200 |
| teacher | qwen |
| prompt_mode | contrastive_directional |
| run_name | qwen_directional_t200 |

### Output Files

```
artifacts/err_cache/yelpchi/bwgnn/qwen_directional_t200/seed_{seed}/
├── evidence_cards.jsonl
├── teacher_payloads.jsonl
├── prompts.jsonl
├── raw_llm_outputs.jsonl
├── llm_err.jsonl
├── accepted_err.jsonl
├── rejected_err.jsonl
├── verifier_stats.json
├── stage2_stats.json
├── prototype_stats.json
├── retrieval_stats.json
└── trace_sampling_stats.json
```

### Reports

```
artifacts/reports/qwen_directional_t200_stage2_quality.md
artifacts/tables/qwen_directional_t200_evidence_quality.md
```

---

## Step H: CV-SCD with Evidence Direction

**File**: `training/losses.py`

### Loss Structure (unchanged)

```
L_CoVER = L_det + λ_err * L_err + λ_signed * L_signed + λ_corr * L_corr
```

### L_err Modification

```
L_err = risk_type_CE + direction_CE + pos_mask_BCE + neg_mask_BCE
```

- Direction CE labels: `increase_risk` / `decrease_risk` / `uncertain`
- **Backward compat**: Direction CE skipped or zero-weighted for old ERR (default "uncertain")

### L_signed Modification

- If `direction == "increase_risk"`: supporting should dominate counter
- If `direction == "decrease_risk"`: counter should dominate supporting
- If `direction == "uncertain"`: downweight or skip signed ranking

### L_corr Modification

- Strong correction only when `evidence_direction` agrees with train correction direction
- If disagreement or uncertain: use inheritance/anchor behavior
- `base_logits` must be `detach()`
- Keep max shift safety

### New Logs

- `direction_ce_loss`
- `direction_label_distribution`
- `direction_correction_agreement`
- `residual_shift_by_direction`
- `correction_nodes_by_direction`

### Backward Compatibility

- Old ERR defaults to `"uncertain"` / `"weak"`
- Direction CE skipped for old ERR
- L_signed/L_corr downweight uncertain direction

---

## Step I: Stage3 Directional Run (conditional on Step G)

### Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | YelpChi |
| Seeds | 123, 456, 789 |
| stage2_run_name | qwen_directional_t200 |
| stage3_run_name | qwen_directional_t200_cvscd |
| gate_mode | safe_residual |
| rho | 0.1 |
| delta_scale | 2.0 |
| threshold_mode | val_macro_f1 |

### Output Tables

```
artifacts/tables/yelpchi_fresh_bwgnn_vs_qwen_directional_t200_cvscd_3seed.csv
artifacts/tables/yelpchi_fresh_bwgnn_vs_qwen_directional_t200_cvscd_3seed.md
artifacts/reports/yelpchi_directional_cvscd_conclusion.md
```

### Metrics

- ROC-AUC, AUPRC, F1@val, Macro-F1@val
- Base FN correction rate, Base FP correction rate
- Residual shift direction accuracy
- Direction correction agreement
- Residual shift mean, max_abs

### Decision Criteria

3-seed mean must satisfy **any one**:
- Δ ROC-AUC ≥ +0.006
- Δ AUPRC ≥ +0.008
- Base FN correction rate明显提升且 MF1@val下降 ≤ 0.005

If met →补齐 seeds 42, 2026
If not met → output failure diagnosis, do NOT raise rho or run Amazon

---

## Hard Constraints

1. No base_score / score / logit / prob / probability / confidence to LLM
2. No base prediction to LLM
3. No target label to LLM
4. No test labels in prototypes / retrieval / trace selection / verifier / loss / threshold
5. No relaxation of verifier
6. Rejected ERR does not enter loss
7. Summary does not enter model or loss
8. Stage3 train/evaluate must be LLM-free
9. Backward compatible with old ERR
10. All new evidence fields score-blind
11. Internal bank names (containing fn/fp/base_error) must never appear in LLM-visible JSON keys

---

## Required Tests

| # | Test | File |
|---|------|------|
| 1 | Directional ERR schema validation | `tests/test_directional_schema.py` |
| 2 | Backward compatibility for old ERR | `tests/test_directional_schema.py` |
| 3 | Score-blind prompt test for contrastive_directional | `tests/test_directional_prompt.py` |
| 4 | Prototype builder does not use test labels | `tests/test_prototypes.py` |
| 5 | Retrieval summaries do not expose target label or base prediction | `tests/test_retrieval.py` |
| 6 | Teacher payload does not contain forbidden score fields | `tests/test_directional_prompt.py` |
| 7 | Verifier rejects invalid evidence_direction | `tests/test_directional_verifier.py` |
| 8 | Verifier rejects invalid uncertainty_factors | `tests/test_directional_verifier.py` |
| 9 | L_err direction CE only uses accepted ERR | `tests/test_cvscd_directional.py` |
| 10 | L_signed skips/downweights uncertain direction | `tests/test_cvscd_directional.py` |
| 11 | L_corr strongly corrects only when direction agrees | `tests/test_cvscd_directional.py` |
| 12 | rho=0 and aux_only still recover base | `tests/test_cvscd_directional.py` |
| 13 | train_stage3.py and evaluate.py do not import llm_teacher.py | `tests/test_no_llm_in_train.py` |

### Score- Blind Test Clarification

Score-blind tests should allow the fixed disclaimer sentence:
> "You are not given any base model score, probability, logit, confidence, or prediction."

But must reject:
- Actual `base_score`/`prob`/`logit`/`confidence` fields or numerical values in teacher payload
- Forbidden fields in prompt JSON, EvidenceCard reasoning channel, or LLM-visible examples
- Target label or base prediction anywhere in LLM-visible content

The test target is "no actual score/prob/logit/confidence values leak to LLM", not "these words never appear in any context".

---

## Implementation Order

1. **Step A-B**: Schema + prototypes/retrieval (foundation)
2. **Step C-D**: Evidence tokens + prompt (Stage2 interface)
3. **Step E**: Verifier (quality gate)
4. **Step F**: Microbenchmark (30 nodes, must pass before t200)
5. **Step G**: Qwen t200 (if microbenchmark passes)
6. **Step H-I**: CV-SCD update + Stage3 run

Each step must pass tests before proceeding to next.
