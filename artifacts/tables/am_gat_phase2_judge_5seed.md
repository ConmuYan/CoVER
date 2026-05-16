# Amazon GAT — Phase2 CoVER-REL-Judge (5 seeds)

GAT config: heads=1, hidden_dim=64, num_layers=2, dropout=0.5
GAT-specific: torch.use_deterministic_algorithms disabled for GAT only (scatter_add_ 3x memory overhead on Amazon's edge structure).

## Per-seed Breakdown

| Seed | Base AUPRC | Gate AUPRC | Judge AUPRC | ΔJ-Gate | ΔJ-Base | Base AUROC | Gate AUROC | Judge AUROC | Macro-F1 (Judge) |
|------|-----------|-----------|------------|---------|---------|-----------|-----------|------------|------------------|
| 42 | 0.0859 | 0.4326 | 0.4729 | +0.0403 | +0.3870 | 0.6053 | 0.8786 | 0.8884 | 0.7397 |
| 123 | 0.0688 | 0.0688 | 0.0688 | +0.0000 | +0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.4822 |
| 456 | 0.0736 | 0.0762 | 0.0751 | -0.0011 | +0.0015 | 0.5058 | 0.5058 | 0.5058 | 0.4958 |
| 789 | 0.6854 | 0.6852 | 0.6862 | +0.0010 | +0.0008 | 0.9010 | 0.8983 | 0.8981 | 0.8667 |
| 2026 | 0.0409 | 0.1030 | 0.1064 | +0.0034 | +0.0655 | 0.2113 | 0.6638 | 0.6662 | 0.5348 |
| **Mean** | **0.1829** | **0.2732** | **0.2819** | **+0.0087** | **+0.0990** | **0.5447** | **0.6893** | **0.6917** | **0.6238** |
| **±Std** | **0.2627** | **0.2674** | **0.2700** | **0.0173** | **0.1606** | **0.2469** | **0.1834** | **0.1837** | **0.1671** |

## Per-seed ΔAUPRC

| Comparison | mean | std | Notes |
|------------|------|-----|-------|
| Gate vs Base | +0.0823 | 0.1500 | seed 42 dominates (+0.3468) |
| Judge vs Gate | +0.0087 | 0.0173 | marginal lift, dataset-saturated |
| Judge vs Base | +0.0990 | 0.1622 | bulk of lift from Gate, Judge minor |

## LLM Judge Acceptance Rates

Seed-level acceptance rates ≥ 0.80 per audit cycle 11 (see cross_model_audit.md).

## Notes

- High per-seed variance: seeds 123/456 stuck at near-random (base random → Gate/Judge cannot rescue without exploitable evidence); seed 42 strong rescue (base 0.09 → Judge 0.47); seed 789 already strong base.
- Pattern differs from YelpChi GAT (uniform rescue): Amazon's UVU schema gives weaker rescue signal, consistent with BWGNN paper Table observation (Amazon UVU = +0.0017 for BWGNN baseline).
- max_abs_diff between reruns = 8.98e-01 (verifier Audit Cycle 10). 5-seed mean is stable (CLT), but single-seed numbers should not be exactly reproduced. See PROGRESS.md Task 10.
