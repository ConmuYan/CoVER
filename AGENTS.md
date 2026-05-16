# AGENTS.md

## Current Project State

The canonical CoVER method is the **two-phase CoVER-REL Reasoner**.

```
Phase1  : train a fresh base detector (BWGNN or GraphSAGE)
          → frozen as structural prior
Phase2  : train one unified, base-agnostic CoVER-REL Reasoner over
          relation-aware evidence and contract-verified score-blind
          LLM judge features
          → z_i = b_i + Δ_rel,i + α_i · Δ_llm,i
```

- The Phase2 Reasoner consumes only frozen base logits/embeddings,
  relation evidence features, and optional accepted judge features. It
  contains no BWGNN- or SAGE-specific logic.
- The LLM judge is a **score-blind, contract-verified, conservative**
  alignment / explanation signal — not a teacher, not a predictor, not
  the main metric source.

Cross-base coverage:

- **BWGNN** is the primary backbone for canonical Phase2 results.
- **GraphSAGE** validates that the Phase2 Reasoner interface is base-agnostic;
  positive on YelpChi (confirmed config delivers significant lift over both
  Phase1 base and legacy Stage3 anchor_gate, with ~41% std reduction over
  default Phase2). Amazon-SAGE remains a diagnostic / saturation case.
- **GCN / GAT** are cross-model validation; Phase2 rescues weak base
  detectors on strong-anchor datasets (YelpChi/RUR).

Legacy (do not deprecate, do not promote): the older Stage3 `anchor_gate`
(CoVER-REL-Gate) and Stage3 `judge_train` (CoVER-REL-Judge) configs and
artifacts remain reproducible under `configs/cover-rel-gj/stage3_legacy/`
and the matching `cover_rel_anchor_gate_nollm` / `cover_rel_judge_*` run
names. They are the historical baseline that Phase2 replaces.

CoVER-DIR, CV-SCD, and CoVER-LIFT are exploratory or negative routes. Do
not restart them as the main method.

Do **not** claim state of the art unless an explicit SOTA comparison is
added. Improvements are reported relative to the fresh Phase1 base.

## Headline Numbers (artifact-backed; do not invent)

Deterministic BWGNN baseline (retrained 2026-05-15 with
`torch.use_deterministic_algorithms(True)`, see PROGRESS.md
"Stage 1 Re-training"):

| Dataset | Legacy Stage3 G/J ΔAUPRC vs BWGNN | Notes                                    |
|---------|----------------------------------:|------------------------------------------|
| YelpChi | +0.0325 (Gate) / +0.0007 (J−G)    | RUR-concentrated                          |
| Amazon  | +0.0017 (Gate) / +0.0003 (J−G)    | UVU-centered but more distributed         |

Cross-model 5-seed (`Phase2` Gate / Judge ablations under
`configs/cover-rel-gj/phase2_ablations/`):

| Dataset | Base         | Phase2 Gate ΔAUPRC vs Base | Phase2 Judge ΔAUPRC vs Gate |
|---------|--------------|---------------------------:|----------------------------:|
| YelpChi | GCN          | +0.2901                    | −0.0002                     |
| YelpChi | GAT (heads=1)| +0.0145                    | **+0.2032**                 |
| Amazon  | GCN          | +0.0000                    | −0.0002                     |
| Amazon  | GAT (heads=1)| +0.0823                    | +0.0087                     |

SAGE-YelpChi confirmed Phase2 reasoner (5-seed mean ± std, from
`configs/phase2_reasoner/phase2_yelpchi_sage_confirm_lalign_1em2_standard.yaml`):

| Compared against        | ΔAUPRC                | paired t    |
|-------------------------|-----------------------|-------------|
| Phase1 SAGE base        | **+0.2540 ± 0.0833**  | **+6.82** (p<0.01) |
| Legacy Stage3 anchor_gate | +0.0321 ± 0.0283    | +2.53 (marginal at n=5) |
| Phase2 default E0/E1/E2 | +0.025..+0.028        | ns; **std cut ~41%** |

`alpha_max=0` and mean `α_llm = 0`: the gain is **not** from direct LLM
residual prediction. Attribute it to judge-aligned relation reasoning /
regularization.

## Run-Label Taxonomy

When reporting any run, tag it with one of:

| Label                           | Meaning                                                                 | Config root                                       |
|---------------------------------|-------------------------------------------------------------------------|---------------------------------------------------|
| **legacy Stage3 anchor_gate**   | First-gen `cover_rel_anchor_gate_nollm`                                 | `configs/cover-rel-gj/stage3_legacy/`             |
| **legacy Stage3 judge_train**   | First-gen `cover_rel_judge_*_strength_gate`                             | `configs/cover-rel-gj/stage3_legacy/`             |
| **Phase2 E0** (relgate)         | unified Reasoner, judge off                                             | `configs/cover-rel-gj/phase2_ablations/`          |
| **Phase2 E1** (judge_align)     | unified Reasoner, judge on, `alpha=0`, `lambda_align>0`                 | `configs/cover-rel-gj/phase2_ablations/`          |
| **Phase2 E2** (judge_residual)  | unified Reasoner, judge on, `alpha>0`, `lambda_align>0`                 | `configs/cover-rel-gj/phase2_ablations/`          |
| **Phase2 E3** (no_trust)        | E2 with `lambda_trust=0` (ablation)                                     | `configs/cover-rel-gj/phase2_ablations/`          |
| **Phase2 confirmed**            | canonical unified Reasoner used in the manuscript                       | `configs/phase2_reasoner/`                        |

Do not introduce new Stage3 anchor_gate / judge_train docs as the final
method. New experiments should target `configs/phase2_reasoner/` or, for
ablation studies, `configs/cover-rel-gj/phase2_ablations/`.

## Required Metadata For Every Run Report

Every results table or per-run conclusion must record:

- `dataset`
- `base_model` (bwgnn / sage / gcn / gat)
- `seed list`
- `run_name` (matches the on-disk artifact directory)
- `config path` (full `configs/...` path, including new subdirs)
- `trainer script` (e.g., `scripts/train_phase2_reasoner.py`)
- `checkpoint path`
- whether the **base is frozen** (must be true for Phase2)
- `alpha_max`, `lambda_align`, `lambda_trust`, `lambda_sparse`
- accepted judge count, plus the rejected-α audit
  (`max_abs_alpha_llm_rejected < 1e-6`)

## Operational Rules

- Use the **two-phase CoVER-REL Reasoner** as the canonical method.
- Use the LLM judge only as a **score-blind, contract-verified** alignment
  / explanation signal; never as a teacher or a primary predictor.
- Preserve score-blind packet and prompt boundaries.
- Preserve train-only prototype construction.
- Never expose `base_score`, `base_prob`, `base_probability`, `base_logit`,
  `confidence`, base prediction, final prediction, target label, val/test
  label, split identity, FN/FP/base-error status, or ground truth to an LLM.
- Rejected judge outputs must not contribute to `L_align`; they must force
  `α = 0` and `Δ_llm = 0` (audited).
- `short_explanation` is human-facing only and must not enter any loss.
- Phase2 reasoner training consumes accepted judge features and must not
  call Qwen.

## Safety Constraints Summary

```
LLM packet forbidden fields:
  base_score, base_prob, base_probability, base_logit, confidence,
  base prediction, final prediction, target label, val/test label,
  split identity, FN/FP/base-error status, ground truth.

Rejected / missing judge output:
  α_i = 0 ; Δ_llm,i = 0 ; L_align contribution = 0
  audited via max_abs_alpha_llm_rejected < 1e-6
```

## Before Modifying Models

Check whether the change affects:

- relation feature extraction
- schema-aware gate behavior
- forbidden-field safety
- train/val/test leakage
- final result tables and paper artifacts
- base-agnostic interface contract (no base-specific logic in the Phase2 Reasoner)

If asked to improve metrics, inspect relation and gate diagnostics first
(`Δ_rel` magnitude, gate entropy, gate distribution, rejected-α audit, ρ
dominance). Do not start from LLM prompt tuning.

## Paper and Artifact Guidance

If asked to update paper text or results, use:

```
docs/cover_main.md                                 # canonical method
docs/sage_paper_alignment.md                       # cross-base study
artifacts/paper/
artifacts/tables/paper_*.md
artifacts/reports/cover_rel_judge_safety_audit.md
artifacts/reports/sage_safety_audit.md
artifacts/reports/sage_confirmed_vs_sage_baselines.md
```

Never invent numbers. Only use metrics from saved artifacts.

## Coding Style

- Keep code small, explicit, research-friendly.
- Prefer simple PyTorch / PyG code.
- Avoid unnecessary abstraction.
- Use deterministic seeds.
- Save config, seed, git hash, metrics, and checkpoint path for every run.
- Prefer concise progress updates and artifact-backed claims.

## Required Tests For Code Changes

For code changes, run:

```bash
pytest -q
python scripts/train_stage1.py --config configs/yelpchi_gcn.yaml --debug
python scripts/generate_stage2_err.py --config configs/yelpchi_gcn.yaml --teacher rule --debug
python scripts/train_stage3.py --config configs/yelpchi_gcn.yaml --debug
python scripts/evaluate.py --config configs/yelpchi_gcn.yaml --debug
```

For documentation-only changes, do not run experiments. Run lightweight
markdown / text sanity checks if available.

## Expected Deliverables

For every task, report:

1. Files changed.
2. What was implemented or edited.
3. How it was checked.
4. Remaining limitations.
5. Next recommended step.
