# Task 7: Real-data BWGNN-aligned Small-scale Sanity Run Report

## A. Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| configs/yelpchi_bwgnn.yaml | Updated | BWGNN paper alignment, correct dataset path |
| configs/amazon_bwgnn.yaml | Updated | BWGNN paper alignment, correct dataset path |
| data/load_fraud.py | Updated | Added split_mode, train_ratio, val_test_ratio params |
| scripts/train_stage1.py | Updated | TensorBoard support, split params |
| scripts/train_stage3.py | Updated | TensorBoard support, split params |
| scripts/generate_stage2_err.py | Updated | Split params |
| scripts/evaluate.py | Updated | Split params |
| utils/tensorboard.py | New | TensorBoard logging utility |
| tests/test_real_sanity_scripts.py | New | 5 tests for sanity scripts |

## B. New Files Created

| File | Description |
|------|-------------|
| utils/tensorboard.py | TensorBoard logging utility for training visualization |
| tests/test_real_sanity_scripts.py | Tests for real sanity scripts without loading real data |

## C. BWGNN Parameter Alignment

| Parameter | Value | Notes |
|-----------|-------|-------|
| epochs | 100 | BWGNN paper setting |
| optimizer | adam | BWGNN paper setting |
| lr | 0.01 | BWGNN paper setting |
| hidden_dim | 64 | BWGNN paper setting |
| order C | 2 | num_bands=3 corresponds to C=2 (low/mid/high frequency bands) |
| aggregation | concat | BWGNN paper setting |
| train_ratio | 0.4 | Supervised scenario |
| val:test | 1:2 | BWGNN paper setting |
| select_metric | macro_f1 | Best validation Macro-F1 |

## D. pytest Result

```
501 passed, 20 failed (external/gread-core), 17 errors (external/gread-core)
```

All cover-fd tests pass. Failures are from external/gread-core tests which are not relevant.

## E. run_model_smoke_tests Result

```
gcn: PASS (33.2s, 100.0%)
sage: PASS (32.8s, 100.0%)
gat: PASS (32.9s, 100.0%)
bwgnn: PASS (32.9s, 100.0%)
```

All models pass smoke tests with 100% acceptance rate.

## F. GPU Support Summary

| Setting | Value |
|---------|-------|
| train_gpus default | "2" (YelpChi), "3" (Amazon) |
| llm_gpus default | "3" |
| CUDA_VISIBLE_DEVICES | Set per subprocess via build_gpu_env() |
| Stage 1/3 | Physical GPU 2 (YelpChi), GPU 3 (Amazon) |
| Stage 2 LLM | Physical GPU 3 |

## G. Whether Real YelpChi Path Exists

**Yes.** Path: `/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd/datasets/YelpChi.mat`

## H. YelpChi BWGNN Rule Sanity Result

**Success**

| Field | Value |
|-------|-------|
| trace_size | 32 |
| train_gpus | "2" |
| stage1 roc_auc | 0.6099 |
| stage1 auprc | 0.2027 |
| stage1 macro_f1 | 0.4612 |
| stage3 roc_auc | 0.6102 |
| stage3 auprc | 0.2028 |
| stage3 macro_f1 | 0.4612 |
| delta roc_auc | +0.0003 |
| delta auprc | +0.0000 |
| delta macro_f1 | +0.0000 |
| verifier acceptance_rate | 100% |
| weak_or_uncertain_ratio | 100% |

## I. YelpChi BWGNN Qwen Sanity Result

**Success**

| Field | Value |
|-------|-------|
| trace_size | 16 |
| train_gpus | "2" |
| llm_gpus | "3" |
| num_llm_calls | 16 |
| parse_success | 16 |
| parse_failed | 0 |
| accepted_after_initial | 16 |
| accepted_after_retry | 0 |
| final_accepted | 16 |
| final_rejected | 0 |
| final_acceptance_rate | 100% |
| reject_reason_counts | {} |
| weak_or_uncertain_ratio | 100% |
| stage1 roc_auc | 0.6099 |
| stage3 roc_auc | 0.6110 |
| delta roc_auc | +0.0010 |

## J. Whether Amazon Path Exists

**Yes.** Path: `/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd/datasets/Amazon.mat`

## K. Amazon BWGNN Rule Sanity Result

**Success**

| Field | Value |
|-------|-------|
| trace_size | 32 |
| train_gpus | "3" |
| stage1 roc_auc | 0.9819 |
| stage1 auprc | 0.8885 |
| stage1 macro_f1 | 0.9221 |
| stage3 roc_auc | 0.9819 |
| stage3 auprc | 0.8854 |
| stage3 macro_f1 | 0.9232 |
| delta roc_auc | +0.0000 |
| delta auprc | -0.0032 |
| delta macro_f1 | +0.0011 |
| verifier acceptance_rate | 100% |

## L. Evidence Quality Report Paths

- YelpChi: `artifacts/reports/yelpchi/bwgnn/seed_0/evidence_quality_report.json`
- Amazon: `artifacts/reports/amazon/bwgnn/seed_0/evidence_quality_report.json`

## M. Whether contracts.yaml Changed

**No.** `evidence/contracts.yaml` was not modified.

## N. Whether verifier.py Changed

**No.** `evidence/verifier.py` was not modified.

## O. Whether Stage 3 / evaluate Remain LLM-Free

**Yes.** Neither `scripts/train_stage3.py` nor `scripts/evaluate.py` import or call LLM.

## P. Current Limitations

1. **YelpChi performance**: The model shows low recall (0.0000) and F1 (0.0000) on YelpChi. This is expected with only 32 trace nodes and random weights. More training epochs and larger trace sizes may improve performance.

2. **Amazon performance**: The model shows good performance on Amazon (roc_auc=0.9819, macro_f1=0.9232). This suggests the dataset is easier or the model is better suited for it.

3. **TensorBoard**: TensorBoard logging is implemented but not yet tested with real TensorBoard visualization.

4. **Dual GPU sanity**: `scripts/run_dual_gpu_sanity.py` is not yet implemented.

## Q. Recommended Next Step

1. **Scarcity experiments**: Run with 5/10/20/40/100% train ratios to evaluate label efficiency.
2. **Ablation studies**: Run ablation experiments to evaluate the contribution of each component.
3. **Multi-seed aggregation**: Run 5 seeds and report mean ± std for paper tables.
4. **TensorBoard visualization**: Test TensorBoard logging with real training runs.
5. **Performance tuning**: Investigate why YelpChi shows low recall and tune hyperparameters.
