# Amazon Phase2 Focused Final Record

Date: 2026-05-16

This document records the Amazon-focused CoVER-REL Phase2 search, CUDA
execution checks, and safety/leakage audit.

Do not use this document to claim state of the art. It records only saved
project artifacts.

## 1. Purpose

After the YelpChi Phase2 confirmation, Amazon was handled separately because
the deterministic BWGNN prior is already strong and previous Amazon CoVER-REL
gains were small.

The goal was to answer four questions:

1. Can a focused Amazon Phase2 search find a stronger default than the first
   conservative candidates?
2. Does Judge alignment or Judge residual help Amazon?
3. Are all focused runs actually executed on CUDA?
4. Do the judge packets, relation features, and Phase2 fusion satisfy
   safety/leakage constraints?

## 2. Stage1 and Legacy References

Amazon deterministic Stage1 BWGNN baseline:

| Metric | Mean +/- Std |
|---|---:|
| ROC-AUC | 0.9747 +/- 0.0067 |
| AUPRC | 0.8643 +/- 0.0190 |
| Macro-F1 | 0.9168 +/- 0.0049 |
| G-Means | 0.8815 +/- 0.0130 |

Amazon legacy CoVER-REL-Gate:

| Metric | Mean +/- Std | Delta vs Stage1 |
|---|---:|---:|
| ROC-AUC | 0.9752 +/- 0.0064 | +0.0005 |
| AUPRC | 0.8661 +/- 0.0185 | +0.0018 |
| Macro-F1 | 0.9174 +/- 0.0049 | +0.0006 |
| G-Means | 0.8832 +/- 0.0110 | +0.0017 |

Amazon legacy CoVER-REL-Judge:

| Metric | Mean +/- Std | Delta vs Stage1 | Delta vs Legacy Gate |
|---|---:|---:|---:|
| ROC-AUC | 0.9751 +/- 0.0067 | +0.0004 | -0.0001 |
| AUPRC | 0.8663 +/- 0.0180 | +0.0020 | +0.0002 |
| Macro-F1 | 0.9168 +/- 0.0042 | +0.0000 | -0.0006 |
| G-Means | 0.8831 +/- 0.0116 | +0.0016 | -0.0001 |

Interpretation:

- Amazon is a near-saturated setting.
- Legacy Gate/Judge provide only small gains over Stage1.
- Any new Phase2 Amazon default must be judged conservatively.

## 3. Focused Search Design

The focused search used standard `scripts/train_phase2_reasoner.py` through a
CUDA launcher:

```text
scripts/run_phase2_amazon_focused_cuda.py
```

The launcher exists because shell-prefix environment assignments can hide CUDA
devices in this environment. It sets worker/thread variables inside Python and
then launches the standard trainer.

Execution settings:

```yaml
dataset: amazon
model: bwgnn
seeds: [42, 123, 456, 789, 2026]
device: cuda:0
workers: 4
trainer: scripts/train_phase2_reasoner.py
launcher: scripts/run_phase2_amazon_focused_cuda.py
```

Every run was checked from saved diagnostics:

```text
artifacts/logs/amazon/bwgnn/*_cuda/seed_*/phase2_diagnostics.json
```

The launcher fails if `device.requested_device != "cuda:0"` or
`device.cuda_available != true`.

## 4. Focused Search Candidates

The search used 16 candidates:

- 8 relation-only candidates varying `delta_rel_max`, `tau_gate`,
  `lambda_trust`, and learning rate.
- 4 Judge alignment-only candidates with `alpha_max=0`.
- 4 conservative Judge residual candidates with `alpha_max=0.05`.

All candidates kept the core Phase2 design:

```text
z_i = b_i + Delta_rel_i + alpha_i * Delta_llm_i
```

and used:

```text
L = L_cls + lambda_trust L_trust
          + lambda_sparse L_sparse
          + lambda_align L_align
```

## 5. Focused CUDA Search Results

Full table:

```text
artifacts/tables/amazon_phase2_focused_cuda_summary.md
artifacts/tables/amazon_phase2_focused_cuda_summary.csv
artifacts/tables/amazon_phase2_focused_cuda_per_seed.csv
```

Top focused candidates by AUPRC:

| Rank | Run | AUPRC | ROC-AUC | Macro-F1 | G-Means | mean_abs_delta_rel | alpha | gate_H |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `phase2_amz_f5_drel10_tau13_trust1em2_lsp0_cuda` | 0.8647 +/- 0.0186 | 0.9748 +/- 0.0066 | 0.9123 +/- 0.0036 | 0.8877 +/- 0.0169 | 0.6386 | 0.0000 | 0.5172 |
| 2 | `phase2_amz_f3_drel10_tau25_trust1em2_lsp0_cuda` | 0.8647 +/- 0.0187 | 0.9748 +/- 0.0066 | 0.9112 +/- 0.0040 | 0.8888 +/- 0.0170 | 0.6517 | 0.0000 | 0.7114 |
| 3 | `phase2_amz_f1_drel10_tau18_trust1em2_lsp0_cuda` | 0.8647 +/- 0.0187 | 0.9748 +/- 0.0066 | 0.9113 +/- 0.0039 | 0.8885 +/- 0.0170 | 0.6673 | 0.0000 | 0.6121 |
| 4 | `phase2_amz_f8_drel15_tau25_trust5em2_lr1em4_cuda` | 0.8647 +/- 0.0186 | 0.9748 +/- 0.0066 | 0.9112 +/- 0.0039 | 0.8881 +/- 0.0169 | 0.4137 | 0.0000 | 0.8890 |
| 5 | `phase2_amz_j6_drel10_align3em3_alpha005_cuda` | 0.8647 +/- 0.0187 | 0.9748 +/- 0.0066 | 0.9125 +/- 0.0039 | 0.8874 +/- 0.0172 | 0.5947 | 0.0001 | 0.6528 |

Best focused AUPRC:

```text
phase2_amz_f5_drel10_tau13_trust1em2_lsp0_cuda
AUPRC = 0.8647307639 +/- 0.0186413858
```

This is:

```text
+0.0004 AUPRC vs Stage1 BWGNN
-0.0014 AUPRC vs legacy CoVER-REL-Gate
-0.0016 AUPRC vs legacy CoVER-REL-Judge
```

Conclusion:

- Focused Phase2 does not recover the legacy Gate/Judge Amazon AUPRC.
- The difference is small, but the focused Phase2 search should not be claimed
  as the strongest Amazon result.
- Amazon should remain framed as a near-saturated dataset where Phase2 preserves
  base-level performance.

## 6. Recommended Amazon Defaults

For focused Phase2 reporting, the best AUPRC candidate is:

```yaml
run_name: phase2_amz_f5_drel10_tau13_trust1em2_lsp0_cuda
use_judge: false
alpha_max: 0.0
lambda_align: 0.0
delta_rel_max: 1.0
tau_gate: 1.3
lambda_trust: 1.0e-2
lambda_sparse: 0.0
lr: 1.0e-3
early_stop_metric: val_auprc
```

For more conservative residual reporting, use:

```yaml
run_name: phase2_amz_f8_drel15_tau25_trust5em2_lr1em4_cuda
use_judge: false
alpha_max: 0.0
lambda_align: 0.0
delta_rel_max: 1.5
tau_gate: 2.5
lambda_trust: 5.0e-2
lambda_sparse: 0.0
lr: 1.0e-4
early_stop_metric: val_auprc
```

Rationale:

- `f5` has the highest focused AUPRC.
- `f8` has lower mean residual shift and high gate entropy, but does not improve
  metrics enough to replace `f5` if optimizing AUPRC.
- Neither should replace legacy CoVER-REL-Gate as the strongest Amazon
  quantitative result.

## 7. Judge Findings on Amazon

Judge alignment and conservative residual were tested after relation-only
candidates.

Best judge-related focused candidate:

```text
phase2_amz_j6_drel10_align3em3_alpha005_cuda
AUPRC = 0.8647 +/- 0.0187
Macro-F1 = 0.9125 +/- 0.0039
mean alpha = 0.0001
rejected alpha max = 0.0000
```

Interpretation:

- Judge residual remains effectively closed.
- Judge alignment shifts gate mass toward UVU, but does not produce a stable
  ranking gain on Amazon.
- This supports the final method framing: Judge is an explanation/alignment
  extension, not the primary Amazon predictor.

## 8. CUDA Execution Verification

Focused CUDA diagnostics:

```text
diagnostic files: 80
bad CUDA diagnostics: 0
```

All focused runs reported:

```json
"requested_device": "cuda:0",
"cuda_available": true
```

Why this matters:

- Initial shell-launched attempts with environment-variable prefixes caused
  PyTorch to report `cuda_available=false` and were discarded.
- The final focused search used `scripts/run_phase2_amazon_focused_cuda.py`,
  which preserves CUDA visibility and sets worker/thread limits inside Python.
- The discarded CPU attempts used run names without the final focused `_cuda`
  aggregate and are not included in the reported table.

## 9. Safety and Leakage Audit

### Judge Packet Audit

Amazon judge forbidden-field audit passed on all five seeds.

| Seed | Passed | Prompt/Packet Passed | Accepted Outputs Passed | Accepted | Rejected | Acceptance |
|---:|---|---|---|---:|---:|---:|
| 42 | true | true | true | 73 | 47 | 0.6083 |
| 123 | true | true | true | 63 | 57 | 0.5250 |
| 456 | true | true | true | 57 | 63 | 0.4750 |
| 789 | true | true | true | 57 | 63 | 0.4750 |
| 2026 | true | true | true | 56 | 64 | 0.4667 |

Audit artifacts:

```text
artifacts/judge_packets/amazon/bwgnn/cover_rel_judge/seed_*/judge_forbidden_field_audit.json
artifacts/judge_packets/amazon/bwgnn/cover_rel_judge/seed_*/judge_forbidden_field_audit.jsonl
```

### Relation Feature Leakage Audit

All five Amazon relation feature metadata files report:

```text
score_blind = true
prototype_labels = train_only
target_label_used = false
val_label_used = false
test_label_used = false
```

Relation feature artifacts:

```text
artifacts/relation_features/amazon/bwgnn/seed_*/all/rel_feature_meta.json
artifacts/relation_features/amazon/bwgnn/seed_*/all/rel_stats.pt
```

### Phase2 Fusion Safety

Focused search aggregate:

```text
rows = 80
metrics_missing = 0
diagnostics_missing = 0
max_rejected_alpha = 0.0
max_mean_alpha = 0.000215
```

This verifies:

- rejected/missing judge records do not open the LLM gate;
- direct Judge residual is effectively closed on Amazon;
- accepted judge features only affect alignment/residual paths when configured;
- `short_explanation` remains human-facing and does not enter loss.

## 10. Figures

Focused search figures were generated under:

```text
artifacts/figures/phase2_sweeps/amazon_focused_cuda/
```

Files:

```text
01_relation_residual_regime_metrics.{png,pdf,svg}
01_relation_residual_regime_diagnostics.{png,pdf,svg}
02_judge_alignment_and_residual_metrics.{png,pdf,svg}
02_judge_alignment_and_residual_diagnostics.{png,pdf,svg}
```

## 10.1 TensorBoard and YelpChi-Style Transfer Check

After the focused Amazon search, Phase2 TensorBoard logging was added to the
standard trainer:

```text
scripts/train_phase2_reasoner.py
```

The old TensorBoard tree was removed before re-running:

```text
artifacts/tensorboard/
```

The new Phase2 TensorBoard layout is:

```text
artifacts/tensorboard/phase2/{dataset}/{model}/{run_name}/seed_{seed}/
```

TensorBoard was started in:

```text
tmux session: cover_phase2_tb
command: tensorboard --logdir artifacts/tensorboard/phase2 --host 0.0.0.0 --port 6006
url: http://0.0.0.0:6006/
```

Each Phase2 epoch now logs:

```text
train/loss
train/lr
loss/total
loss/l_cls
loss/l_trust
loss/l_sparse
loss/l_align
val/auprc
val/roc_auc
val/macro_f1
val/g_means
diagnostics/mean_abs_delta_rel
diagnostics/mean_alpha_llm
diagnostics/mean_gate_entropy
diagnostics/gate_weight_rel_*
gate/{relation_name}
```

To answer whether Amazon had been run with the YelpChi best-style Phase2
configuration, a dedicated 5-seed CUDA run was added and executed:

```text
config: configs/phase2_amazon_yelpstyle_judge_align.yaml
run: phase2_amz_yelpstyle_lalign1em2_alpha0_cuda_tb
device: cuda:2
workers: 4
seeds: [42, 123, 456, 789, 2026]
```

This keeps Amazon data and schema names but transfers the YelpChi-style
Phase2 hyperparameters:

```yaml
rel_hidden_dim: 64
tau_gate: 0.7
delta_rel_max: 2.0
alpha_max: 0.0
lambda_trust: 3.0e-3
lambda_sparse: 1.0e-3
lambda_align: 1.0e-2
```

Summary:

| Run | n | AUPRC | ROC-AUC | Macro-F1 | G-Means | mean_abs_delta_rel | gate_H | gate_0 | gate_1 | gate_2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `phase2_amz_yelpstyle_lalign1em2_alpha0_cuda_tb` | 5 | 0.8651 +/- 0.0192 | 0.9753 +/- 0.0069 | 0.9127 +/- 0.0035 | 0.8877 +/- 0.0172 | 1.4103 | 0.5443 | 0.1319 | 0.1838 | 0.6843 |

Compared with the deterministic Stage1 BWGNN baseline:

```text
Delta AUPRC   = +0.0008
Delta ROC-AUC = +0.0006
Delta Macro-F1 = -0.0041
Delta G-Means = +0.0062
```

Compared with legacy Amazon CoVER-REL-Gate:

```text
Delta AUPRC   = -0.0010
Delta ROC-AUC = +0.0001
Delta Macro-F1 = -0.0047
Delta G-Means = +0.0045
```

Training-curve artifacts:

```text
artifacts/figures/phase2_training_curves/amazon_yelpstyle_cuda_tb/phase2_train_losses.{png,pdf,svg}
artifacts/figures/phase2_training_curves/amazon_yelpstyle_cuda_tb/phase2_val_metrics.{png,pdf,svg}
```

Curve interpretation:

- Train loss decreases strongly for every seed.
- Validation AUPRC reaches its best value early and then mostly plateaus or
  declines.
- Validation Macro-F1 usually declines while BCE keeps improving.
- The transferred YelpChi-style gate is too aggressive for Amazon: mean
  `|delta_rel|` rises to 1.4103, much larger than the focused Amazon default
  candidate `f5` at 0.6386.
- `alpha_max=0`, so the Judge residual is closed by design; this run tests
  Judge alignment and relation routing only.

Interpretation:

Directly transferring the YelpChi-style sparse/sharp gate to Amazon is not the
best Amazon setting. It gives a tiny AUPRC gain over Stage1 but underperforms
legacy Gate and hurts Macro-F1. Amazon remains better served by more conservative
relation residuals and weaker/no sparsity.

## 11. Final Amazon Interpretation

The Amazon result should be written conservatively:

```text
Amazon is near saturated under the deterministic BWGNN prior. The legacy
CoVER-REL-Gate remains the strongest Amazon quantitative result, while the new
unified Phase2 Reasoner preserves base-level performance under bounded
residuals. Judge alignment and conservative residual paths pass safety checks
but do not provide a stable Amazon ranking gain.
```

Recommended paper stance:

- Use legacy CoVER-REL-Gate as the main Amazon quantitative model.
- Report focused Phase2 as a unified-reasoner diagnostic/ablation, not as the
  best Amazon result.
- Report Judge as safe and explanation-oriented, not as a major Amazon metric
  source.

## 12. Files Added or Updated

Added:

```text
scripts/run_phase2_amazon_focused_cuda.py
artifacts/reports/amazon_phase2_focused_final_record.md
```

Updated:

```text
scripts/plot_phase2_sweep_metrics.py
```

Generated artifacts:

```text
artifacts/tables/amazon_phase2_focused_cuda_summary.md
artifacts/tables/amazon_phase2_focused_cuda_summary.csv
artifacts/tables/amazon_phase2_focused_cuda_per_seed.csv
artifacts/figures/phase2_sweeps/amazon_focused_cuda/
artifacts/sweeps/_drivers/phase2_amazon_focused_cuda_full.log
```

## 13. Remaining Limitations

- Focused Phase2 did not exceed legacy CoVER-REL-Gate on Amazon.
- The initial CPU fallback attempts created partial run directories without the
  final focused aggregate meaning. They are intentionally excluded from the
  `_cuda` summary.
- Amazon judge acceptance is lower than YelpChi and ranges from 0.4667 to
  0.6083 across seeds.
- No SOTA claim is supported by these artifacts.

## 14. Next Recommended Step

For Amazon, stop broad Phase2 tuning unless there is a new modeling reason.

Recommended next work:

1. Use legacy CoVER-REL-Gate as Amazon main quantitative result.
2. Use the focused Phase2 table as a robustness/diagnostic appendix.
3. Include the Amazon safety/leakage audit in the Judge appendix.
4. If one more experiment is required, run only a direct legacy-Gate-compatible
   ablation inside the unified Phase2 framework rather than expanding Judge
   residual tuning.
