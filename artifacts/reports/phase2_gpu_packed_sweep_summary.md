# Phase2 GPU-Packed Sweep Summary

## Why the runner changed

The original Phase2 sweep launched one small MLP reasoner per process.  That made the bottleneck Python, CPU-side metric computation, and repeated full validation rather than GPU matrix multiplication.  Running many such processes increased CPU pressure without meaningfully improving GPU utilization.

The updated YelpChi judge sweep uses a GPU-packed trainer:

- one process trains multiple judge/alpha configurations for the same seed;
- relation experts, gates, judge heads, losses, and validation rank metrics are batched on GPU;
- CPU thread pools are limited with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `COVER_NUM_THREADS=1`;
- existing artifact layout is preserved under `artifacts/{checkpoints,logs,results}/yelpchi/bwgnn/{run_name}/seed_*`.

## Completed Artifacts

- YelpChi targeted sensitivity: `artifacts/tables/yelpchi_phase2_targeted_sensitivity_summary.md`
- Amazon default candidates: `artifacts/tables/amazon_phase2_default_candidates_summary.md`
- YelpChi figures: `artifacts/figures/phase2_sweeps/yelpchi_targeted_sensitivity/`
- Amazon figures: `artifacts/figures/phase2_sweeps/amazon_default_candidates/`

## Key Results

### YelpChi

Best AUPRC among targeted sweeps:

| run_name | AUPRC | AUROC | Macro-F1 | G-Means |
|---|---:|---:|---:|---:|
| `phase2_yelp_s3_lalign_1em2` | 0.5737 ± 0.0080 | 0.8706 ± 0.0033 | 0.7302 ± 0.0052 | 0.7180 ± 0.0176 |
| `phase2_yelp_s3_lalign_3em3` | 0.5722 ± 0.0111 | 0.8708 ± 0.0042 | 0.7300 ± 0.0039 | 0.7131 ± 0.0174 |
| `phase2_yelp_s1_ltrust_3em2` | 0.5700 ± 0.0175 | 0.8683 ± 0.0052 | 0.7301 ± 0.0055 | 0.7120 ± 0.0056 |

Interpretation: stronger judge-gate alignment improves YelpChi in this sweep, while increasing `alpha_max` does not outperform alignment-only.  This supports the paper framing that the LLM judge is better used as evidence alignment than as a primary residual predictor.

### Amazon

Best AUPRC among candidate defaults:

| run_name | AUPRC | AUROC | Macro-F1 | G-Means |
|---|---:|---:|---:|---:|
| `phase2_amz_c2_drel075_ltrust1em2_tau18_lsp0` | 0.8646 ± 0.0187 | 0.9748 ± 0.0066 | 0.9113 ± 0.0040 | 0.8888 ± 0.0170 |

Interpretation: smaller relation residuals and distributed gates keep Amazon near the frozen BWGNN prior, but still do not recover the legacy gate's small AUPRC edge.  Amazon should remain framed as a near-saturated case.

## Caveat

The packed trainer is an execution optimization for sweep throughput.  It preserves the Phase2 equations and artifact schema, but evaluates validation every 5 epochs to reduce CPU-side pressure.  Main claims should continue to cite the exact artifact-backed tables used in the paper.
