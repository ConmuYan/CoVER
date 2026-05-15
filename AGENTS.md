# AGENTS.md

## Current Project State

The final CoVER method is **CoVER-REL**.

- **CoVER-REL-Gate** is the main quantitative model and deployment-friendly relation-only detector.
- **CoVER-REL-Judge** is the LLM-assisted research extension for score-blind structured judgement and explanation.
- CoVER-DIR, CV-SCD, and CoVER-LIFT are exploratory or negative routes. Do not restart them as the main method.

Current final results are improvements over the **deterministic BWGNN baseline** (retrained 2026-05-15 with `torch.use_deterministic_algorithms(True)`, see PROGRESS.md "Stage 1 Re-training (Deterministic Baseline, 2026-05-15)"):

| Dataset | Main Model | Gate Delta AUPRC vs BWGNN | Judge Delta AUPRC vs Gate |
|---|---|---:|---:|
| YelpChi | CoVER-REL-Gate | +0.0325 | +0.0007 |
| Amazon | CoVER-REL-Gate | +0.0017 | +0.0003 |

Reference deltas before the deterministic re-baselining (kept for audit only): YelpChi +0.026585 / Amazon +0.003508 vs the previous non-deterministic BWGNN; Judge over Gate was +0.000010 / +0.000224. Sign and order match the new deterministic numbers.

Do not claim state of the art unless an explicit SOTA comparison is added.

## Operational Rules

- Use CoVER-REL-Gate as the main model for performance claims.
- Use CoVER-REL-Judge only as the explanation-oriented LLM-assisted research extension.
- Preserve score-blind packet and prompt boundaries.
- Preserve train-only prototype construction.
- Never expose base score, probability, logit, confidence, base prediction, target label, split identity, FN/FP/base-error status, or ground truth to an LLM.
- Rejected judge outputs must not enter fusion training.
- `short_explanation` is human-facing only and must not be used in loss.
- Stage3 training consumes accepted judge features and must not call Qwen.

## Before Modifying Models

Check whether the change affects:

- relation feature extraction;
- schema-aware gate behavior;
- forbidden-field safety;
- train/val/test leakage;
- final result tables and paper artifacts.

If asked to improve metrics, inspect relation and gate diagnostics first. Do not start from LLM prompt tuning.

## Paper and Artifact Guidance

If asked to update paper text or results, use:

```text
artifacts/paper/
artifacts/tables/paper_*.md
artifacts/reports/cover_rel_judge_safety_audit.md
```

Never invent numbers. Only use metrics from saved artifacts.

## Coding Style

- Keep code small, explicit, and research-friendly.
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

For documentation-only changes, do not run experiments. Run lightweight markdown or text sanity checks if available.

## Expected Deliverables

For every task, report:

1. Files changed.
2. What was implemented or edited.
3. How it was checked.
4. Remaining limitations.
5. Next recommended step.
