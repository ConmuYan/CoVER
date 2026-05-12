# AGENTS.md

## Project Goal

This repository implements CoVER-FD: Contract-Verified Evidence Distillation for LLM-Free Fake Review Detection.

The main task is binary graph fraud detection on YelpChi and Amazon. The target pipeline has three stages:

1. Train a base GNN detector.
2. Generate score-blind structural evidence cards, ask an offline LLM teacher to produce ERR records, and verify them with deterministic evidence contracts.
3. Train an evidence-conditioned student reasoner with supervised detection loss and accepted ERR distillation loss.

Do not build a large framework. Keep the code small, explicit, and research-friendly.

## Key Method Constraints

- LLM is offline only.
- No LLM call during model training except the explicit Stage 2 generation script.
- No LLM call during inference.
- The LLM prompt must not contain base_score, probability, logit, confidence, or raw prediction labels.
- ERR summary is for human inspection only and must not be used in loss.
- Rejected ERR records must not contribute to evidence distillation loss.
- If accepted ERR count is zero in a batch, fall back to supervised task loss.
- rho=0 in the reasoner must make final_logit equal base_logit.

## Coding Style

- Prefer simple PyTorch / PyG code.
- Avoid unnecessary abstraction.
- Each module should be under 300 lines unless unavoidable.
- Use type hints for public functions.
- Add docstrings only where they clarify non-obvious logic.
- Use deterministic seeds.
- Save config, seed, git hash, metrics, and checkpoint path for every run.

## Required Tests

Before claiming a task is complete, run:

```bash
pytest -q
python scripts/train_stage1.py --config configs/yelpchi_gcn.yaml --debug
python scripts/generate_stage2_err.py --config configs/yelpchi_gcn.yaml --teacher rule --debug
python scripts/train_stage3.py --config configs/yelpchi_gcn.yaml --debug
python scripts/evaluate.py --config configs/yelpchi_gcn.yaml --debug
```

If a command fails, report the exact error and the file/function likely responsible.

## External Repositories

Reference repositories may be cloned into `external/`, but do not directly copy large code blocks unless the license permits it.

Use external code only for:

* understanding data format,
* checking model architecture,
* reproducing baseline behavior,
* comparing training scripts.

Our own implementation should live in this repository.

## Expected Deliverables Per Task

For every coding task, provide:

1. Files changed.
2. What was implemented.
3. How it was tested.
4. Remaining limitations.
5. Next recommended step.
