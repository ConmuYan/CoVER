# CoVER-REL Phase2 Final Record

Date: 2026-05-15

This document records the current final CoVER-REL Phase2 state after the
standard 5-seed confirmation run for the YelpChi best configuration.

Do not use this document to claim state of the art. It only records
artifact-backed results in the current project.

## 1. Final Method Position

The final method remains CoVER-REL.

- CoVER-REL-Gate is the main quantitative and deployment-friendly relation-only
  detector.
- CoVER-REL-Judge is the LLM-assisted research extension for score-blind
  structured judgement, evidence alignment, and explanation.
- The LLM judge is not treated as the primary predictor.
- The Phase2 reasoner learns bounded residual interventions on top of a frozen
  BWGNN structural prior.

The final training story is two-phase:

1. Phase1 trains a fresh deterministic BWGNN.
2. Phase2 trains a unified CoVER-REL Reasoner using relation-aware evidence and,
   optionally, accepted contract-verified judge features.

## 2. Frozen Stage1 Baseline

The Stage1 baseline is the deterministic BWGNN baseline recorded in
`PROGRESS.md` under "Stage 1 Re-training (Deterministic Baseline,
2026-05-15)".

YelpChi Stage1 deterministic BWGNN, 5 seeds:

| Metric | Mean +/- Std |
|---|---:|
| ROC-AUC | 0.8076 +/- 0.0081 |
| AUPRC | 0.4674 +/- 0.0151 |
| Macro-F1 | 0.6483 +/- 0.0161 |
| G-Means | 0.5105 +/- 0.0353 |

Per-seed Stage1 records are stored under:

```text
artifacts/checkpoints/yelpchi/bwgnn/base/seed_*/retraining_metrics.json
artifacts/results/yelpchi/bwgnn/base/seed_*/stage1_metrics.json
```

## 3. Legacy CoVER-REL Reference Results

These are the already confirmed legacy CoVER-REL Gate/Judge results against the
deterministic BWGNN baseline.

YelpChi legacy CoVER-REL-Gate:

| Metric | Mean +/- Std | Delta vs Stage1 |
|---|---:|---:|
| ROC-AUC | 0.8177 +/- 0.0091 | +0.0101 |
| AUPRC | 0.4998 +/- 0.0210 | +0.0324 |
| Macro-F1 | 0.6598 +/- 0.0229 | +0.0115 |
| G-Means | 0.5253 +/- 0.0479 | +0.0148 |

YelpChi legacy CoVER-REL-Judge:

| Metric | Mean +/- Std | Delta vs Stage1 | Delta vs Legacy Gate |
|---|---:|---:|---:|
| ROC-AUC | 0.8180 +/- 0.0102 | +0.0104 | +0.0003 |
| AUPRC | 0.5006 +/- 0.0248 | +0.0332 | +0.0008 |
| Macro-F1 | 0.6650 +/- 0.0178 | +0.0167 | +0.0052 |
| G-Means | 0.5359 +/- 0.0370 | +0.0254 | +0.0106 |

Interpretation:

- Relation-aware evidence is the main source of improvement.
- The legacy judge adds only a small quantitative gain, mostly visible in
  threshold-sensitive metrics.
- This supports the final paper framing that Judge is an auxiliary
  explanation/alignment extension rather than the primary detector.

## 4. Phase2 Reasoner Architecture

The Phase2 reasoner uses a frozen BWGNN prior:

```text
b_i = detached frozen BWGNN base logit
```

Relation experts produce relation-specific residual evidence:

```text
h_i,r = Expert_r(E_i,r)
g_i = softmax(a_i / tau_gate)
Delta_rel_i = delta_rel_max * tanh(sum_r g_i,r * Head_r(h_i,r))
z_rel_i = b_i + Delta_rel_i
```

Accepted judge features are optional:

```text
j_i = JudgeEncoder(J_i)
Delta_llm_i = delta_llm_max * tanh(Head_llm(j_i, h_i, g_i))
alpha_i = accepted_mask_i * alpha_max * sigmoid(Head_alpha(j_i, h_i, g_i))
z_i = b_i + Delta_rel_i + alpha_i * Delta_llm_i
```

If judge output is missing or rejected:

```text
alpha_i = 0
z_i = z_rel_i
```

The total loss is:

```text
L = L_cls + lambda_trust * L_trust
          + lambda_sparse * L_sparse
          + lambda_align * L_align
```

where:

- `L_cls` is BCEWithLogits on training labels.
- `L_trust` penalizes bounded residual drift from the frozen BWGNN prior.
- `L_sparse` regularizes relation gate entropy under relation-strength
  conditioning.
- `L_align` aligns the relation gate with accepted judge key-relation targets.

`short_explanation` is not used in training loss.

## 5. Best YelpChi Phase2 Configuration

The selected best configuration is a judge-alignment-only reasoner. It does not
use direct LLM residual fusion.

Run name:

```text
phase2_yelp_confirm_lalign_1em2_standard
```

Core configuration:

```yaml
dataset: yelpchi
model: bwgnn
seeds: [42, 123, 456, 789, 2026]
config: configs/phase2_yelpchi_E1_judge_align.yaml

phase2_reasoner:
  use_judge: true
  alpha_max: 0.0
  lambda_align: 1.0e-2
  lambda_trust: 3.0e-3
  lambda_sparse: 1.0e-3
  tau_gate: 0.7
  delta_rel_max: 2.0
  delta_llm_max: 0.75
  rel_hidden_dim: 64
  judge_hidden_dim: 32
  alpha_bias_init: -3.0
  optimizer: adamw
  lr: 1.0e-3
  weight_decay: 1.0e-4
  epochs: 300
  patience: 50
  early_stop_metric: val_auprc
  eval_interval: 1
```

Important: this confirmation run used the standard
`scripts/train_phase2_reasoner.py` path, not the GPU packed sweep trainer.

## 6. Standard 5-Seed Confirmation Result

YelpChi standard confirmation result:

| Model | ROC-AUC | AUPRC | Macro-F1 | G-Means |
|---|---:|---:|---:|---:|
| Stage1 BWGNN | 0.8076 +/- 0.0081 | 0.4674 +/- 0.0151 | 0.6483 +/- 0.0161 | 0.5105 +/- 0.0353 |
| Phase2 confirm | 0.8739 +/- 0.0039 | 0.5804 +/- 0.0103 | 0.7341 +/- 0.0033 | 0.7175 +/- 0.0100 |
| Delta vs Stage1 | +0.0663 | +0.1130 | +0.0858 | +0.2070 |

Per-seed standard confirmation metrics:

| Seed | ROC-AUC | AUPRC | Macro-F1 | G-Means | Threshold |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.8690 | 0.5647 | 0.7299 | 0.7201 | 0.6800 |
| 123 | 0.8778 | 0.5930 | 0.7383 | 0.7244 | 0.7200 |
| 456 | 0.8707 | 0.5843 | 0.7321 | 0.7121 | 0.7000 |
| 789 | 0.8773 | 0.5799 | 0.7349 | 0.7280 | 0.7100 |
| 2026 | 0.8749 | 0.5803 | 0.7355 | 0.7031 | 0.7300 |

Diagnostics:

| Diagnostic | Value |
|---|---:|
| mean_abs_delta_rel | 1.5167 |
| alpha | 0.0000 |
| gate entropy | 0.3843 |
| gate_0 / RUR | 0.7834 |
| gate_1 / RSR | 0.1039 |
| gate_2 / RTR | 0.1128 |
| rejected_alpha_max | 0.0000 |

Interpretation:

- The result is confirmed by the standard trainer and is not an artifact of the
  packed sweep implementation.
- The improvement is consistent across all five seeds; no single seed explains
  the average.
- The judge branch contributes through alignment only because `alpha_max=0`.
- The LLM does not directly change the final logit in this confirmed setting.
- The relation gate becomes less RUR-collapsed than relation-only Phase2, which
  suggests `lambda_align=1e-2` regularizes relation routing rather than acting
  as a direct label teacher.

Artifacts:

```text
artifacts/tables/yelpchi_phase2_confirm_lalign_1em2_standard_summary.md
artifacts/tables/yelpchi_phase2_confirm_lalign_1em2_standard_summary.csv
artifacts/results/yelpchi/bwgnn/phase2_yelp_confirm_lalign_1em2_standard/seed_*/stage3_metrics.json
artifacts/logs/yelpchi/bwgnn/phase2_yelp_confirm_lalign_1em2_standard/seed_*/phase2_train_log.jsonl
artifacts/logs/yelpchi/bwgnn/phase2_yelp_confirm_lalign_1em2_standard/seed_*/phase2_diagnostics.json
artifacts/checkpoints/yelpchi/bwgnn/phase2_yelp_confirm_lalign_1em2_standard/seed_*/reasoner.pt
```

## 7. Comparison With Prior Packed Sweep

The prior GPU packed sweep reported:

```text
phase2_yelp_s3_lalign_1em2
AUPRC    0.5737 +/- 0.0080
ROC-AUC  0.8706 +/- 0.0033
Macro-F1 0.7302 +/- 0.0052
G-Means  0.7180 +/- 0.0176
```

The standard confirmation run reported:

```text
phase2_yelp_confirm_lalign_1em2_standard
AUPRC    0.5804 +/- 0.0103
ROC-AUC  0.8739 +/- 0.0039
Macro-F1 0.7341 +/- 0.0033
G-Means  0.7175 +/- 0.0100
```

Conclusion:

- The packed sweep did not inflate the result.
- The standard trainer confirms the effect and is slightly stronger on AUPRC,
  ROC-AUC, and Macro-F1.
- The standard confirmation should be used for final YelpChi result reporting.

## 8. YelpChi Hyperparameter Findings

Targeted sensitivity summary:

- `lambda_trust` matters. Increasing trust loss to `3e-2` improved
  relation-only AUPRC to `0.5700 +/- 0.0175` and reduced mean absolute relation
  residual.
- `lambda_sparse` has little effect in the tested range because the YelpChi
  relation-only gate already strongly prefers RUR.
- `lambda_align` is the most useful judge-related knob. A larger alignment
  weight improved both AUPRC and gate diversity.
- `alpha_max` did not help. Even with `alpha_max=0.1` or `0.3`, the learned mean
  alpha stayed near zero.

Current best confirmed setting:

```text
lambda_align = 1e-2
alpha_max = 0
```

This supports the final interpretation:

```text
Use Judge as a score-blind relation evidence alignment signal, not as a direct
residual predictor.
```

## 9. Amazon Finding

Amazon remains a near-saturated dataset under the deterministic BWGNN prior.

Best Amazon candidate from the default-candidate sweep:

```text
phase2_amz_c2_drel075_ltrust1em2_tau18_lsp0
AUPRC = 0.8646 +/- 0.0187
ROC-AUC = 0.9748 +/- 0.0066
Macro-F1 = 0.9113 +/- 0.0040
G-Means = 0.8888 +/- 0.0170
```

More conservative Amazon default candidate:

```text
phase2_amz_c1_drel05_ltrust1em2_tau18_lsp0
AUPRC = 0.8645 +/- 0.0188
ROC-AUC = 0.9748 +/- 0.0066
Macro-F1 = 0.9129 +/- 0.0033
G-Means = 0.8871 +/- 0.0169
```

Interpretation:

- Amazon Phase2 candidates are approximately tied with the deterministic base
  and below the legacy CoVER-REL-Gate result.
- The final paper should not overstate Amazon Phase2 as a performance
  improvement.
- Amazon is better framed as a saturation case where conservative Phase2
  preserves performance and diagnostics rather than providing large metric
  gains.

Recommended Amazon default for conservative reporting:

```yaml
delta_rel_max: 0.5
lambda_trust: 1.0e-2
tau_gate: 1.8
lambda_sparse: 0.0
alpha_max: 0.0
```

Amazon artifacts:

```text
artifacts/tables/amazon_phase2_default_candidates_summary.md
artifacts/tables/amazon_phase2_default_candidates_summary.csv
artifacts/figures/phase2_sweeps/amazon_default_candidates/
```

## 10. Safety and Leakage Boundaries

The following boundaries remain mandatory:

- Do not expose base score, probability, logit, confidence, base prediction,
  target label, split identity, FN/FP/base-error status, or ground truth to the
  LLM.
- Do not expose raw review text through the judge packet.
- Use train-only labels for prototype construction.
- Rejected judge outputs must not enter fusion training.
- Missing or rejected judge records must force `alpha=0`.
- `short_explanation` is human-facing only and must not enter the loss.
- Stage3/Phase2 training must consume accepted judge features only and must not
  call Qwen.

The confirmed YelpChi run satisfies the most important fusion safety diagnostic:

```text
rejected_alpha_max = 0.0000
```

## 11. Recommended Paper Wording

Use the following high-level wording:

```text
We train CoVER(BWGNN) in two phases. First, we train a fresh BWGNN as a frozen
structural prior. Second, we train a unified CoVER-REL Reasoner over
relation-aware evidence and contract-verified score-blind judge features. The
reasoner learns bounded residual interventions from relation experts and uses
the LLM judge as a conservative relation-evidence alignment signal rather than
as a primary predictor.
```

For YelpChi:

```text
On YelpChi, the confirmed Phase2 judge-alignment reasoner improves over the
deterministic BWGNN baseline across all four metrics, with AUPRC increasing from
0.4674 to 0.5804 and G-Means increasing from 0.5105 to 0.7175.
```

For Amazon:

```text
On Amazon, the deterministic BWGNN prior is already near saturated. Phase2
reasoner variants preserve strong performance but do not consistently exceed
the legacy relation-gate result, so Amazon is reported as a conservative
saturation case rather than as a large-gain setting.
```

## 12. Remaining Limitations

- The best confirmed YelpChi result has not yet been crossed with
  `lambda_trust=3e-2`; that may improve or destabilize the current best setting.
- The current single-confirmation plotting command failed because
  `plot_phase2_sweep_metrics.py` only accepts the existing suite names
  `amazon_candidates` and `yelpchi_targeted`. This does not affect metrics or
  aggregation.
- Amazon Phase2 does not currently improve over the legacy Gate result.
- The current result should not be called SOTA without explicit SOTA baselines.

## 13. Next Recommended Step

Run one small cross-check on YelpChi:

```text
lambda_trust in {3e-3, 3e-2}
lambda_align in {3e-3, 1e-2}
alpha_max = 0
seeds = 42, 123, 456, 789, 2026
```

Purpose:

- confirm whether the relation-only trust improvement combines with the best
  judge-alignment setting;
- keep Judge as alignment-only;
- avoid broad hyperparameter search.

If this cross-check does not beat the confirmed result, use
`phase2_yelp_confirm_lalign_1em2_standard` as the final YelpChi Phase2 result.

## 14. CUDA TensorBoard Confirmation and Loss Diagnostics (2026-05-16)

TensorBoard logging was added to the standard Phase2 trainer and the old
TensorBoard tree was removed before the new runs. The active Phase2 log layout
is:

```text
artifacts/tensorboard/phase2/{dataset}/{model}/{run_name}/seed_{seed}/
```

A TensorBoard server was started in tmux:

```text
tmux session: cover_phase2_tb
command: tensorboard --logdir artifacts/tensorboard/phase2 --host 0.0.0.0 --port 6006
url: http://0.0.0.0:6006/
```

Only the two BWGNN Phase2 TB runs are kept in the active tree:

```text
artifacts/tensorboard/phase2/amazon/bwgnn/phase2_amz_yelpstyle_lalign1em2_alpha0_cuda_tb/seed_*/
artifacts/tensorboard/phase2/yelpchi/bwgnn/phase2_yelp_best_lalign1em2_alpha0_cuda_tb/seed_*/
```

### YelpChi Best CUDA+TB Run

Run:

```text
phase2_yelp_best_lalign1em2_alpha0_cuda_tb
```

Configuration:

```yaml
use_judge: true
alpha_max: 0.0
lambda_align: 1.0e-2
lambda_trust: 3.0e-3
lambda_sparse: 1.0e-3
delta_rel_max: 2.0
delta_llm_max: 0.75
tau_gate: 0.7
workers: 4
device: cuda:2
seeds: [42, 123, 456, 789, 2026]
```

Summary:

| Run | n | AUPRC | ROC-AUC | Macro-F1 | G-Means | mean_abs_delta_rel | alpha | gate_H | RUR | RSR | RTR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `phase2_yelp_best_lalign1em2_alpha0_cuda_tb` | 5 | 0.5776 +/- 0.0090 | 0.8720 +/- 0.0032 | 0.7323 +/- 0.0051 | 0.7206 +/- 0.0173 | 1.5281 | 0.0000 | 0.3849 | 0.7805 | 0.1170 | 0.1025 |

The CUDA+TB run is close to the earlier CPU confirmation but slightly lower in
AUPRC:

```text
earlier phase2_yelp_confirm_lalign_1em2_standard AUPRC = 0.5804 +/- 0.0103
new CUDA+TB phase2_yelp_best_lalign1em2_alpha0_cuda_tb AUPRC = 0.5776 +/- 0.0090
```

The ordering and interpretation are unchanged: YelpChi has a large Phase2 gain
over Stage1 and a strongly RUR-dominant gate.

### Static Training Curves

Static curves were regenerated with the mean line computed only over the common
epoch range across seeds, avoiding artificial jumps when early-stopped seeds
leave the average.

```text
artifacts/figures/phase2_training_curves/amazon_yelpstyle_cuda_tb/phase2_train_losses.{png,pdf,svg}
artifacts/figures/phase2_training_curves/amazon_yelpstyle_cuda_tb/phase2_val_metrics.{png,pdf,svg}
artifacts/figures/phase2_training_curves/yelpchi_best_cuda_tb/phase2_train_losses.{png,pdf,svg}
artifacts/figures/phase2_training_curves/yelpchi_best_cuda_tb/phase2_val_metrics.{png,pdf,svg}
```

### Loss Diagnostic Comparison

Amazon yelpstyle aggregate from epoch 1 to best epoch to final epoch:

| Quantity | epoch 1 | best val AUPRC epoch | final | best - epoch 1 | final - best |
|---|---:|---:|---:|---:|---:|
| Total loss | 0.6074 | 0.3869 | 0.3283 | -0.2205 | -0.0586 |
| BCE loss | 0.6004 | 0.3779 | 0.3167 | -0.2226 | -0.0611 |
| Trust loss | 0.0000 | 2.1101 | 2.9486 | +2.1101 | +0.8385 |
| Sparse loss | 0.0000 | 0.2188 | 0.2457 | +0.2188 | +0.0268 |
| Align loss | 0.6925 | 0.2493 | 0.2479 | -0.4431 | -0.0014 |
| Val AUPRC | 0.8606 | 0.8625 | 0.8595 | +0.0019 | -0.0030 |
| Val Macro-F1 | 0.9143 | 0.8916 | 0.8747 | -0.0227 | -0.0169 |

YelpChi best aggregate:

| Quantity | epoch 1 | best val AUPRC epoch | final | best - epoch 1 | final - best |
|---|---:|---:|---:|---:|---:|
| Total loss | 1.2356 | 0.7020 | 0.6883 | -0.5336 | -0.0137 |
| BCE loss | 1.2288 | 0.6931 | 0.6789 | -0.5357 | -0.0142 |
| Trust loss | 0.0000 | 2.6387 | 2.7670 | +2.6387 | +0.1283 |
| Sparse loss | 0.0000 | 0.0999 | 0.0975 | +0.0999 | -0.0024 |
| Align loss | 0.6846 | 0.0922 | 0.1008 | -0.5925 | +0.0086 |
| Val AUPRC | 0.4692 | 0.5813 | 0.5786 | +0.1121 | -0.0026 |
| Val Macro-F1 | 0.6556 | 0.7020 | 0.7015 | +0.0464 | -0.0004 |

Interpretation:

- `L_trust` increases in both datasets because useful relation residuals must
  move away from the frozen BWGNN prior. This is acceptable on YelpChi because
  validation AUPRC and Macro-F1 improve strongly while trust grows.
- On Amazon, `L_trust` growth is not buying ranking gain. This indicates the
  residual scale/trust weight is too permissive for a near-saturated prior.
- `L_sparse` is stable and low on YelpChi, consistent with a real RUR-dominant
  schema. On Amazon yelpstyle, sparse loss is larger and drifts upward, showing
  that the transferred sharp/sparse gate prior is mismatched to Amazon's more
  distributed relation utility.
- `L_align` becomes small and stable on YelpChi, but Amazon shows seed-level
  instability and little validation benefit. Judge alignment should remain an
  explanation/alignment regularizer, not a metric driver.
- The normal classification loss alone is not enough to decide whether Phase2
  is healthy. Amazon BCE keeps improving while validation AUPRC is flat and
  Macro-F1 degrades.

### Loss Design Implications

For YelpChi, keep the current best alignment-only design as the main Phase2
configuration. The auxiliary losses are not perfect, but their behavior is
consistent with a useful relation intervention.

For Amazon, do not use the YelpChi-style configuration as default. The next
algorithmic fix should be conservative and dataset-specific:

```yaml
alpha_max: 0.0
lambda_align: 0.0 or very small
lambda_sparse: 0.0
tau_gate: >= 1.3
delta_rel_max: <= 1.0
lambda_trust: >= 1.0e-2
```

If modifying the loss design, the highest-priority Amazon fix is not another
judge residual. It is to make trust/sparse adaptive to base confidence or
validation saturation, for example:

```text
stronger trust when |base_logit| is large;
weaker or zero sparse pressure when relation dominance is low;
optional residual shrinkage schedule or delta_rel cap warmup;
consider replacing pos-weighted BCE with a ranking-aware objective only as a
separate ablation, because current BCE improves while AUPRC does not.
```
