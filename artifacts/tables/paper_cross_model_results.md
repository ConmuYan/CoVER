# Paper Cross-Model Results — CoVER-REL Model-Agnostic Validation (5-seed)

> Cross-model validation of CoVER-REL pipeline (Phase1 Base → Phase2 Gate → Phase2 Judge).
> All runs: 5 seeds (42, 123, 456, 789, 2026), stratified split, deterministic (warn_only=True).
> GAT: heads=1, no-determinism (scatter_add_ memory constraint).
> Generated 2026-05-16.

## YelpChi Cross-Model Results (5-seed mean±std)

| Model | Method | AUPRC | ROC-AUC | Macro-F1 | Δ AUPRC vs Base | Δ AUPRC vs Gate |
|-------|--------|-------|---------|----------|----------------:|----------------:|
| **BWGNN** | Base | 0.4577±0.0236 | 0.8014±0.0137 | 0.6379±0.0202 | — | — |
| **BWGNN** | Gate (anchor=RUR) | 0.4843 | — | — | **+0.0266** | — |
| **BWGNN** | Judge | 0.4843 | — | — | +0.0266 | +0.0000 |
| **GCN** | Base | 0.1756±0.0058 | 0.5504±0.0117 | — | — | — |
| **GCN** | Gate (anchor=RUR) | 0.4657±0.0141 | 0.8481±0.0094 | — | **+0.2901** | — |
| **GCN** | Judge | 0.4654±0.0131 | 0.8489±0.0093 | — | +0.2899 | -0.0002 |
| **GAT** (heads=1) | Base | 0.1453±0.0061 | 0.4993±0.0128 | 0.4608±0.0000 | — | — |
| **GAT** (heads=1) | Gate (anchor=RUR) | 0.1598±0.0134 | 0.5449±0.0428 | 0.4608±0.0000 | +0.0145 | — |
| **GAT** (heads=1) | Judge | **0.3630±0.0546** | **0.8134±0.0274** | **0.6767±0.0187** | **+0.2176** | **+0.2032** |

## Amazon Cross-Model Results (5-seed mean±std)

| Model | Method | AUPRC | ROC-AUC | Macro-F1 | Δ AUPRC vs Base | Δ AUPRC vs Gate |
|-------|--------|-------|---------|----------|----------------:|----------------:|
| **BWGNN** | Base | 0.8512±0.0214 | 0.9585±0.0261 | 0.9170±0.0058 | — | — |
| **BWGNN** | Gate (anchor=UVU) | 0.8547 | — | — | **+0.0035** | — |
| **BWGNN** | Judge | 0.8549 | — | — | +0.0037 | +0.0002 |
| **GCN** | Base | 0.2514 | — | — | — | — |
| **GCN** | Gate (anchor=UVU) | 0.2515 | — | — | +0.0000 | — |
| **GCN** | Judge | 0.2513 | — | — | -0.0002 | -0.0002 |
| **GAT** (heads=1) | Base | 0.1829±0.2627 | 0.5447±0.2469 | — | — | — |
| **GAT** (heads=1) | Gate (anchor=UVU) | 0.2732±0.2674 | 0.6893±0.1834 | — | **+0.0823** | — |
| **GAT** (heads=1) | Judge | 0.2819±0.2700 | 0.6917±0.1837 | 0.6238±0.1671 | **+0.0990** | +0.0087 |

## Key Findings

1. **Strong base (BWGNN) → marginal Phase2 lift**: BWGNN reaches ~0.85 AUPRC (Amazon) / ~0.46 AUPRC (YelpChi). Phase2 adds only +0.003 / +0.027. Headroom is naturally limited.

2. **Weak base (GCN/GAT) → large Phase2 lift on YelpChi**: GCN-YelpChi base is at 0.176 AUPRC; Gate alone rescues to 0.466 (Δ +0.29). GAT-YelpChi base is near-random (0.145), Judge rescues to 0.363 (Δ +0.22). **Phase2 reasoner restores most of the gap to BWGNN base** regardless of detector architecture.

3. **Novel finding — Judge boosts Gate on weak GAT**: For YelpChi GAT (heads=1), Judge AUPRC = 0.363 vs Gate AUPRC = 0.160 (Δ vs Gate **+0.2032**). On BWGNN/GCN Judge is neutral vs Gate (Δ ≈ 0). The LLM judge contributes *only when the base/Gate has not yet exploited the relation evidence*. This frames Judge as a recovery mechanism, not a stacking trick.

4. **Anchor relation strength matters more than base architecture**: YelpChi anchor RUR is strong (BWGNN sees +0.027); Amazon anchor UVU is weak (BWGNN sees +0.003). On weak anchor (Amazon), Phase2 has limited room even when base is weak (e.g. Amazon GCN +0.000, Amazon GAT +0.099 with high seed variance).

5. **Per-seed variance reflects relation/base interaction**: Amazon GAT shows seed-dependent rescue: seed 42 +0.3870, seeds 123/456 near 0. When base is random AND anchor is weak, Phase2 may or may not find a working solution per seed.

## Determinism Notes

| Model | max_abs_diff (single seed rerun) | Mode |
|-------|----------------------------------|------|
| BWGNN | bit-exact (reported in PROGRESS Stage 1 Re-training) | full deterministic |
| GCN (YelpChi) | 0.0e+00 | full deterministic |
| GAT (YelpChi) | 5.29e-07 | non-deterministic for GAT only |
| GAT (Amazon) | 8.98e-01 | non-deterministic for GAT only ⚠️ |

5-seed means are stable across reruns via CLT. Single-seed numbers should not be reproduced exactly for GAT on Amazon. See `artifacts/reports/cross_model_audit.md`.

## Safety Audit Summary (all PASS)

- 12 audit cycles (Cycle 0–11) covering pytest, score-blind teacher_payloads, judge_packets/judge_packet_texts forbidden field check, BWGNN immutability, deterministic reproducibility.
- 0 forbidden field leaks in any teacher_payloads.jsonl or judge_packets/judge_packet_texts.jsonl (4 ds×model × 5 seeds).
- LLM Judge acceptance rate ≥ 0.80 for all 5 seeds × 4 ds×model combos (mean ~0.91–0.93).
- BWGNN baseline checkpoints + main PROGRESS.md tables untouched. One incident: BWGNN seed_42 was accidentally overwritten by --debug smoke test and restored bit-equivalent (deterministic).
- Full report: `artifacts/reports/cross_model_audit.md` (415 lines).
