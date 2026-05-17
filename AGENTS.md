# AGENTS.md

## Current Project State

The canonical CoVER method is the **two-phase CoVER-REL Reasoner** with a
**cls-only loss**.

```
Phase1  : train a fresh base detector (BWGNN / GraphSAGE / GCN / GAT)
          → frozen as structural prior (SHA-256 verified)
Phase2  : train one unified, base-agnostic CoVER-REL Reasoner over
          relation-aware score-blind evidence
          → z_i = b_i + Δ_rel,i        (no LLM judge term)
          → L   = L_cls                (no L_int / L_sparse / L_align)
```

- The Phase2 Reasoner consumes only frozen base logits/embeddings and
  per-relation 9-dim anonymous evidence statistics. It contains no
  BWGNN- / SAGE- / GCN- / GAT-specific logic.
- **All LLM-judge code paths are removed from the canonical method** after
  5-seed paired t-tests falsified every variant; see "Killed Routes" below.
- The 4-term legacy loss (L_cls + λ_int·L_int + λ_sparse·L_sparse + λ_align·L_align)
  was collapsed to **L_cls only** after 5-seed paired t-test on
  YelpChi-BWGNN showed all four loss variants statistically
  indistinguishable from cls-only (see
  `artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md`).

Cross-base coverage (8/8 configurations method-direction positive,
6/8 5-seed paired-t significant; see `CROSS_DATASET_CROSS_BASE_FINDING.md`):

- **BWGNN** is the primary backbone for canonical Phase2 results.
- **GraphSAGE** validates that the Phase2 Reasoner interface is base-agnostic;
  YelpChi-SAGE confirmed +0.254 AUPRC paired t = +6.82 (p<0.01) vs Phase1.
  Amazon-SAGE remains a diagnostic / saturation case.
- **GCN / GAT** are cross-model validation; rel branch rescues weak base
  detectors on strong-anchor datasets (YelpChi-GCN: +0.292 AUPRC,
  YelpChi-GAT: +0.320 AUPRC).

Legacy (do not deprecate, do not promote): the older Stage3 `anchor_gate`
(CoVER-REL-Gate) and Stage3 `judge_train` (CoVER-REL-Judge) configs and
artifacts remain reproducible under `configs/cover-rel-gj/stage3_legacy/`
and the matching `cover_rel_anchor_gate_nollm` / `cover_rel_judge_*` run
names. They are the historical baseline that Phase2 replaces.

Killed Routes (all 5-seed paired-t falsified — **do not restart as the main
method**):

- `α · Δ_llm` additive LLM-judge residual (paired t = +0.03, p = 0.976)
- `L_align` judge-tilted KL (paired t = −1.14, p = 0.32)
- `L_intervention` base-anchored Δ_rel² penalty (paired t = +0.26, p = 0.81 alone;
  +1.02, p = 0.36 with L_sparse)
- `L_sparse` evidence→gate KL (paired t = +2.59, p = 0.061 — closest to bar, still fails)
- LEQA (LoRA-Qwen3 evidence-quality auditor + L_audit): null/negative at sanity gate
- B3 PRTAE (per-relation MLP-hidden auxiliary feature injection):
  1/8 conditions significant positive, 3/8 significant negative
- CoVER-DIR, CV-SCD, CoVER-LIFT: directional / counter-evidence / hidden-state
  distillation variants, all sub-+0.008 AUPRC or gate-degenerate
- LLM verbalization embedding (PCA32): paired t = −10.6, p < 0.01 worse
- Raw-text sigpool: paired t = −19.4, p < 0.01 catastrophically worse

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

SAGE-YelpChi confirmed Phase2 reasoner (5-seed mean ± std). Numbers reported
were obtained with the 4-term loss (`L_int + L_sparse + L_align`); the
4-term-vs-cls-only paired t is ns (p = 0.991 on YelpChi-BWGNN, see ablation
table), so these numbers carry over verbatim to the cls-only canonical
form:

| Compared against        | ΔAUPRC                | paired t    |
|-------------------------|-----------------------|-------------|
| Phase1 SAGE base        | **+0.2540 ± 0.0833**  | **+6.82** (p<0.01) |
| Legacy Stage3 anchor_gate | +0.0321 ± 0.0283    | +2.53 (marginal at n=5) |
| Phase2 default E0/E1/E2 | +0.025..+0.028        | ns; **std cut ~41%** (variance-reduction side-effect of L_int+L_sparse; observability footnote only) |

`alpha_max=0` and mean `α_llm = 0`: the gain is not from any LLM residual.
Attribute it to the rel-branch architecture (per-relation expert MLP +
schema softmax gate + bounded tanh residual), not to any loss-term magic.

## Loss-Term Ablation (the honest record)

5-seed paired t-test, YelpChi-BWGNN, all cells vs L7 (full 4-term) reference.
Source: `artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md`.

| Cell  | Variant       | AUPRC paired Δ vs L7 (t, p)        |
|-------|---------------|------------------------------------|
| A0    | base only     | Δ=−0.1042 (t=−30.4, p≈7e-6) ★★★    |
| **L0**| **cls only**  | Δ=+0.0000 (t=+0.01, p=0.991) ns ★ canonical |
| L1    | +L_int        | Δ=+0.0006 (t=+0.26, p=0.81) ns     |
| L2    | +L_sparse     | Δ=+0.0025 (t=+2.59, p=0.061) ns    |
| L3    | +L_align      | Δ=−0.0004 (t=−1.14, p=0.32) ns     |
| L4    | +L_int+L_sp   | Δ=+0.0015 (t=+1.02, p=0.36) ns     |
| L7    | full 4-term ★ | reference (now collapsed to L0)    |
| A1    | rel-only      | Δ=+0.0014 (t=+1.23, p=0.29) ns     |

→ The architecture (frozen base + rel branch) supplies the entire +0.1042
AUPRC lift; no loss term clears the 5-seed p<0.05 bar over cls-only.
Canonical loss is therefore **L_cls only**.

## Run-Label Taxonomy

When reporting any run, tag it with one of:

| Label                           | Meaning                                                                 | Config root                                       |
|---------------------------------|-------------------------------------------------------------------------|---------------------------------------------------|
| **legacy Stage3 anchor_gate**   | First-gen `cover_rel_anchor_gate_nollm`                                 | `configs/cover-rel-gj/stage3_legacy/`             |
| **legacy Stage3 judge_train**   | First-gen `cover_rel_judge_*_strength_gate`                             | `configs/cover-rel-gj/stage3_legacy/`             |
| **legacy Phase2 4-term**        | Pre-cls-only Phase2 runs with L_int+L_sparse[+L_align]                  | `configs/phase2_reasoner/*` (numbers still valid) |
| **legacy Phase2 E1/E2/E3** (judge cells) | Judge-on ablations; reasoner now raises NotImplementedError on judge_on path | `configs/cover-rel-gj/phase2_ablations/`  |
| **Phase2 E0** (rel-only)        | unified Reasoner, judge off                                             | `configs/cover-rel-gj/phase2_ablations/`          |
| **Phase2 cls-only (canonical)** | Current canonical: `z = b + Δ_rel`, `L = L_cls`                         | `configs/phase2_reasoner/`                        |

Do not introduce new Stage3 anchor_gate / judge_train docs as the final
method. New experiments should target `configs/phase2_reasoner/` with the
cls-only loss; legacy 4-term ablations stay under
`configs/cover-rel-gj/phase2_ablations/` for historical reproducibility.

## Required Metadata For Every Run Report

Every results table or per-run conclusion must record:

- `dataset`
- `base_model` (bwgnn / sage / gcn / gat)
- `seed list`
- `run_name` (matches the on-disk artifact directory)
- `config path` (full `configs/...` path, including new subdirs)
- `trainer script` (e.g., `scripts/train_phase2_reasoner.py`)
- `checkpoint path`
- whether the **base is frozen** (must be true for Phase2; SHA-256 verified)
- `delta_rel_max`, `tau_gate` (rel-branch hyperparameters)
- For legacy 4-term runs only: `lambda_int`, `lambda_sparse`, `lambda_align` (now deprecation no-ops in code)

## Operational Rules

- Use the **two-phase CoVER-REL Reasoner with cls-only loss** as the canonical method.
- The LLM-judge code path is removed; do not reintroduce `use_judge=True`,
  `α·Δ_llm`, or `L_align` as canonical components. All such configurations
  raise `NotImplementedError` from `models/cover_rel_reasoner.py`.
- Preserve score-blind packet and prompt boundaries on any LLM diagnostic / explainability tooling that may still call Qwen.
- Preserve train-only prototype construction.
- Never expose `base_score`, `base_prob`, `base_probability`, `base_logit`,
  `confidence`, base prediction, final prediction, target label, val/test
  label, split identity, FN/FP/base-error status, or ground truth to an LLM.
- Legacy: rejected judge outputs in any historical Stage3 run must still force
  `α = 0` and `Δ_llm = 0` (audited via `max_abs_alpha_llm_rejected < 1e-6`).
- `short_explanation` is human-facing only and must not enter any loss.

## Safety Constraints Summary

```
LLM packet forbidden fields (still applies to any diagnostic LLM call):
  base_score, base_prob, base_probability, base_logit, confidence,
  base prediction, final prediction, target label, val/test label,
  split identity, FN/FP/base-error status, ground truth.

Legacy Stage3 judge audit (cls-only canonical does NOT use judge):
  Rejected / missing judge output:
    α_i = 0 ; Δ_llm,i = 0 ; L_align contribution = 0
    audited via max_abs_alpha_llm_rejected < 1e-6

Phase 2 canonical (cls-only) audit:
  base_freeze SHA-256 pre/post hashes bit-identical (base.pt, base_logits, base_z)
  forbidden-field audit count == 0 on the 9R-dim relation evidence input
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

<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 75 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/data1/mq/codes/aris_repo`).
Update with: `bash /data1/mq/codes/aris_repo/tools/install_aris.sh /data1/mq/codes/awesome-graph-anomaly-detection/cover-fd --aris-repo /data1/mq/codes/aris_repo`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
