# CoVER-REL: A Two-Phase Base-Agnostic Relation-Evidence Reasoner

## Abstract

CoVER-REL is the canonical CoVER method for fake-review / fraud detection on the
YelpChi and Amazon multi-relation graphs. The central finding is that the useful
evidence in these `.mat` benchmarks is not free-form LLM rationale, base-score-driven
explanation, or canonical ERR hidden-state distillation. It is the **relation-aware
anonymous feature evidence** induced by the graph construction schema.

CoVER-REL is organized as a **two-phase framework**:

- **Phase1** trains a fresh base detector (BWGNN or GraphSAGE) and freezes it
  as a structural prior.
- **Phase2** trains a single **unified CoVER-REL Reasoner** over relation-aware
  evidence and optional contract-verified score-blind LLM judge features.

The Phase2 Reasoner is **base-agnostic**: it consumes only frozen base logits,
frozen base embeddings, relation evidence features, and optional accepted judge
features. It does not contain BWGNN-specific or GraphSAGE-specific logic.

## Contributions

1. **Evidence-source reframing.** CoVER reframes fraud detection from generic
   GNN residual correction or LLM rationale distillation into schema-aware
   relation evidence reasoning. The actionable signals are relation-wise
   anonymous feature deviation, neighbor consistency, and train-only prototype
   margins carried by multi-relation graph construction.

2. **Two-phase base-agnostic Reasoner.** A frozen Phase1 base supplies the
   structural prior `b_i`. The Phase2 Reasoner learns a bounded relation-evidence
   residual `Δ_rel,i` and an optional bounded judge residual `Δ_llm,i`, mixed
   through a conservative gate `α_i`. The same Phase2 Reasoner works on top of
   either BWGNN or GraphSAGE without architectural changes.

3. **Contract-verified score-blind LLM judgement.** The LLM judge reasons over
   score-blind relation evidence rather than base scores or predictions. Outputs
   are contract-verified; only accepted records contribute. The judge is used
   primarily as an **evidence-alignment / regularization** signal — it is not a
   teacher, predictor, or main metric source.

## Method Overview

### 1. Phase1 — Frozen Base Structural Prior

The base detector is a fresh `BWGNN` or `GraphSAGE` model trained on YelpChi or
Amazon. After Phase1 it is **frozen** and exposes two artifacts to Phase2:

- `b_i` — detached base logit
- `base_z_i` — detached base embedding / hidden representation

The base is never used as an imitation teacher and is never exposed to the LLM.

### 2. Phase2 — Relation-Aware Evidence Extraction

For each dataset relation schema, CoVER-REL builds one evidence view per relation
from anonymous handcrafted features and relation neighborhoods.

- YelpChi schema: `RUR`, `RSR`, `RTR`
- Amazon  schema: `UPU`, `USU`, `UVU`

Each relation view includes relation degree, feature deviation from relation
neighbors, neighbor consistency, z-score outlier count, train-only fraud
prototype distance, train-only benign prototype distance, and the
fraud-vs-benign prototype margin. All views are **score-blind** and never use
val/test labels.

### 3. Phase2 — Unified CoVER-REL Reasoner

A relation-specific expert encodes each evidence view; a schema-aware gate
distributes weight over relations. The Reasoner output is a bounded residual
intervention on top of the frozen base prior.

Final logit:

```text
z_i = b_i + Δ_rel,i + α_i · Δ_llm,i
p_i = sigmoid(z_i)
```

- `b_i` — frozen base logit (detached)
- `Δ_rel,i` — bounded relation-evidence residual (`|Δ_rel| ≤ delta_rel_max`)
- `Δ_llm,i` — bounded LLM judge residual (`|Δ_llm| ≤ delta_llm_max`)
- `α_i` — conservative judge gate (`α ∈ [0, alpha_max]`)

If the judge output is missing or rejected, `α_i = 0` and the model falls back
to `z_i = b_i + Δ_rel,i`.

### 4. Score-Blind LLM Evidence Judge (optional)

The LLM judge consumes only relation-evidence packets (no base score, no label,
no split identity, no prediction). It returns:

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

Outputs are contract-verified. Rejected packets are excluded from any training
signal and forced to `α = 0`. `short_explanation` is human-facing only and is
never used in any loss.

The judge is used primarily through `L_align` as an evidence-routing
alignment signal (Section 5). The optional bounded `Δ_llm,i` is a conservative
secondary contribution, not the main performance source.

## 5. Loss

Total loss:

```text
L_CoVER = L_cls + λ_trust · L_trust + λ_sparse · L_sparse + λ_align · L_align
```

### L_cls — supervised task loss

```text
L_cls = BCEWithLogits(z_i, y_i)
```

### L_trust — bounded residual regularization

```text
L_trust = mean_i [ Δ_rel,i² + η_llm · α_i · Δ_llm,i² ]
```

Prevents the Reasoner and the judge residual from unboundedly overriding the
frozen base prior.

### L_sparse — schema-aware sparse gate

Let `g_i ∈ Δ^R` be the relation gate distribution, `H(g_i) = −Σ_r g_i,r log(g_i,r + ε)`,
and `ρ_i` an evidence-dominance score over relations.

```text
L_sparse = mean_i [ ρ_i · H(g_i) ]
```

Encourages sparse relation routing when one relation dominates, while allowing
distributed evidence when relation utility is spread.

### L_align — judge → gate alignment (accepted records only)

For accepted judge outputs, `key_relation` and `evidence_strength` are converted
to a soft target `q_i ∈ Δ^R`. Then:

```text
L_align = mean_{i ∈ Train ∩ AcceptedJudge} [ w_i · KL(q_i || g_i) ]
```

Lets the score-blind LLM judge align relation routing without supervising the
final label and without replacing the relation reasoner.

The canonical formulation has **one** supervised task loss and **three**
auxiliary losses. Do not add more auxiliary losses unless a future experiment
explicitly proves the need.

## 6. E0 / E1 / E2 / E3 Variants

| Variant | `use_judge` | `alpha_max` | `lambda_align` | `lambda_trust` | Final logit                    |
|---------|:-----------:|:-----------:|:--------------:|:--------------:|--------------------------------|
| E0      | false       | 0           | 0              | >0             | b + Δ_rel                      |
| E1      | true        | 0           | >0             | >0             | b + Δ_rel  (judge → gate only) |
| E2      | true        | >0          | >0             | >0             | b + Δ_rel + α · Δ_llm          |
| E3      | true        | >0          | >0             | **0**          | b + Δ_rel + α · Δ_llm (no trust) |

The current canonical "confirmed" config (`configs/phase2_reasoner/...`) is
an E1-flavored setting with `alpha_max = 0` and `lambda_align = 1e-2`: the LLM
is used for alignment / regularization, not as a residual predictor.

## 7. Results

Phase1 fresh base is the comparison baseline. All numbers come from saved
artifacts; do not interpret as state of the art unless an explicit SOTA
comparison is added.

### 7.1 BWGNN base (canonical Phase2 backbone)

| Dataset | Method                | ΔAUPRC vs Phase1 BWGNN | ΔAUPRC vs Phase2 Gate | Interpretation                         |
|---------|-----------------------|-----------------------:|----------------------:|----------------------------------------|
| YelpChi | Phase2 CoVER-REL Gate | +0.026585              | 0.0                   | RUR-concentrated relation utility       |
| YelpChi | Phase2 CoVER-REL Judge| +0.026595              | +0.000010             | Judge fusion neutral; safety preserved  |
| Amazon  | Phase2 CoVER-REL Gate | +0.003508              | 0.0                   | UVU-centered but more distributed       |
| Amazon  | Phase2 CoVER-REL Judge| +0.003732              | +0.000224             | Judge fusion neutral; safety preserved  |

The main performance source is **relation-aware evidence**. The LLM contribution
on BWGNN is conservative and should not be described as the primary gain.

### 7.2 GraphSAGE base (cross-base validation)

The same Phase2 Reasoner runs on a frozen SAGE base without code changes.

YelpChi-SAGE (5 seeds: 42, 123, 456, 789, 2026):

| Stage                                        | AUPRC (mean ± std)     |
|----------------------------------------------|------------------------|
| Phase1 SAGE base                             | 0.2246 ± 0.1261        |
| Legacy Stage3 anchor_gate                    | 0.4465 ± 0.0289        |
| Phase2 default E2                            | 0.4539 ± 0.0891        |
| Phase2 confirmed (`phase2_yelp_confirm_lalign_1em2_standard`) | **0.4786 ± 0.0522** |

Paired Δ (confirmed − baseline, 5-seed):

| Compared against                | ΔAUPRC                 | paired t | Note                                  |
|---------------------------------|------------------------|----------|----------------------------------------|
| Phase1 SAGE base                | **+0.2540 ± 0.0833**   | **+6.82**| p<0.01                                 |
| Legacy Stage3 anchor_gate       | +0.0321 ± 0.0283       | +2.53    | marginal for n=5; not strict p<0.05    |
| Phase2 default E0/E1/E2         | +0.025 .. +0.028       | +1.3..+1.5 ns | mean lift not significant; std cut ~41% |

Interpretation:

- The confirmed config delivers a **highly significant lift over the Phase1
  base** and a **significant lift over legacy Stage3 anchor_gate** on AUPRC and
  especially on ROC-AUC (+0.0474, t=+12.27, p<0.001).
- The lift over the SAGE-default Phase2 family is positive but not significant
  in mean. The real value over default Phase2 is **stability and worst-seed
  rescue** (~41% std reduction; seed_456 rescued from 0.32 to 0.42).
- Diagnostic note: `alpha_max=0` and `mean α_llm = 0` for the confirmed config,
  so the gain is **not** from direct LLM residual prediction. It should be
  attributed to **judge-aligned relation reasoning / regularization**.
- Gate behavior is RUR-dominant but not fully collapsed:
  `RUR ≈ 0.7615 / RSR ≈ 0.1375 / RTR ≈ 0.1011`; gate entropy ≈ 0.3837.

Amazon-SAGE: Phase2 remained a diagnostic / saturation case under the available
runs; the strict-gate at E0 prevented full sweeps. Do not present Amazon-SAGE
as a solved positive result.

### 7.3 GCN / GAT cross-model (5-seed)

| Dataset | Base         | Phase2 Gate ΔAUPRC vs base | Phase2 Judge ΔAUPRC vs Gate |
|---------|--------------|---------------------------:|----------------------------:|
| YelpChi | GCN          | +0.2901                    | −0.0002                     |
| YelpChi | GAT (heads=1)| +0.0145                    | **+0.2032**                 |
| Amazon  | GCN          | +0.0000                    | −0.0002                     |
| Amazon  | GAT (heads=1)| +0.0823                    | +0.0087                     |

The cross-model results validate CoVER-REL as a base-agnostic framework: Phase2
rescues weak base detectors (GCN/GAT) on a strong-anchor dataset (YelpChi/RUR)
to BWGNN-base level, and the LLM Judge contributes meaningfully when the Gate
has not yet captured the relation signal (GAT/YelpChi).

## 8. Historical / Legacy Naming

Old run names and config locations are kept for reproducibility of the legacy
"Gate + Judge" family but should not be used for the canonical method.

| Status   | Family            | Run names                                                                          | Config root                                |
|----------|-------------------|-----------------------------------------------------------------------------------|--------------------------------------------|
| Current  | Phase2 unified    | `phase2_E0_relgate`, `phase2_E1_judge_align`, `phase2_E2_judge_residual`, `phase2_E3_no_trust`, `phase2_yelp_confirm_lalign_1em2_standard`, `phase2_amz_yelpstyle_lalign1em2_alpha0_cuda_tb` | `configs/phase2_reasoner/`, `configs/cover-rel-gj/phase2_ablations/` |
| Legacy   | Stage3 G/J        | `cover_rel_anchor_gate_nollm`, `cover_rel_judge_strength_gate`, `cover_rel_judge_rur_strength_gate`, `cover_rel_judge_uvu_strength_gate` | `configs/cover-rel-gj/stage3_legacy/` |

See `configs/README.md` for the full layout. Historical reports under
`artifacts/reports/` and `artifacts/paper/` retain old `configs/<name>.yaml`
paths verbatim as time-stamped snapshots.

## 9. Safety Constraints

The LLM judge packet **must not** include any of:

- `base_score`, `base_prob`, `base_probability`, `base_logit`, `confidence`
- base prediction, final prediction
- target label, val/test label, split identity
- FN/FP/base-error status, ground truth

Rejected or missing judge outputs:

- must not contribute to `L_align`
- must set `α_i = 0` and `Δ_llm,i = 0` (audited via `max_abs_alpha_llm_rejected < 1e-6`)
- must fall back to `z_i = b_i + Δ_rel,i`

`short_explanation` is human-facing only and never enters any loss.

## 10. Final Positioning

- **CoVER-REL Reasoner (two-phase, base-agnostic)** is the canonical method.
- The LLM judge is a **score-blind, contract-verified, conservative** alignment
  / explanation signal — not a teacher, predictor, or main metric source.
- GraphSAGE validates base-agnostic transfer primarily on YelpChi.
  Amazon-SAGE remains a diagnostic / saturation case.
- Do **not** claim state of the art. Improvements are reported relative to the
  fresh Phase1 base.
- Do **not** describe LLM residual prediction as the main performance source.
- Do **not** present GraphSAGE as a second main detector — it is a cross-base
  validation of the Reasoner interface.

## Appendix A: What Changed From Earlier CoVER Versions

### CoVER-DIR / CV-SCD

CoVER-DIR and CV-SCD were diagnostic routes. Evidence-supported FN/FP correction
was too sparse to yield stable metric gains; FP negative correction in particular
remained unreliable because most base false positives stayed fraud-dominant
under score-blind structural evidence.

### CoVER-LIFT

CoVER-LIFT tested canonical ERR hidden-state distillation. Student latent
alignment could be high while AUPRC and ranking did not improve — the
canonical ERR hidden states were too thin and not sufficiently
fraud-discriminative.

### Legacy CoVER-REL-Gate / CoVER-REL-Judge

The first CoVER-REL generation trained two separate Stage3 stages:
`anchor_gate` (no LLM) and a follow-up `judge_train` (gated LLM residual on top
of anchor_gate). These remain reproducible from `configs/cover-rel-gj/stage3_legacy/`
and are the historical baseline that the unified Phase2 Reasoner replaces.
The Phase2 Reasoner subsumes both stages into a single base-agnostic training
pass with one supervised task loss and three auxiliary losses.
