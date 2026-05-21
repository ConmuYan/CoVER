"""4-metric aggregator for Flash-RAER + CBR-Flash paper tables.

Reads per-seed ``stage3_metrics.json`` for ALL relevant C3 conditions and
produces a 4-metric (AUPRC / AUROC / Macro-F1 / G-Means) cross-cell table
with per-cell paired-t vs the deterministic top-K Flash-RAER baseline
(``g_opd_flash_det_mask``).

Condition naming schemes handled (historical run directories and v3 CBR
ablations live under different prefixes):

* ``artifacts/results/{ds}/{base}/g_opd_flash_{mode}/seed_{s}/stage3_metrics.json``
* ``artifacts/results/{ds}/{base}/v3_cbr_{variant}/seed_{s}/stage3_metrics.json``
* ``artifacts/results/{ds}/{base}/idea2c_distill_adapter/seed_{s}/stage3_metrics.json``
  (hand-crafted teacher distillation; written as ``hc_distill``)
* ``artifacts/results/{ds}/{base}/idea2c_distill_adapter_2b/seed_{s}/stage3_metrics.json``
  (LREE teacher distillation; written as ``lree_distill``)

Per-cell paired-t hypothesis (5-seed, df=4):
  H0: condition == det_mask (baseline)
  H1: condition > det_mask
Output: one-sided t + p value, per metric, per cell.

Outputs:

* ``artifacts/tables/c3_4metric_8cell_5seed.md``  (full per-cell, 4 metrics)
* ``artifacts/tables/c3_4metric_8cell_5seed.csv``
* ``artifacts/tables/c3_4metric_summary.json``    (cross-cell sig counts)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))


DATASETS = ("yelpchi", "amazon")
BASES = ("bwgnn", "sage", "gcn", "gat")
SEEDS = (42, 123, 456, 789, 2026)
METRICS = ("auprc", "roc_auc", "macro_f1", "g_means")
METRIC_LABEL = {
    "auprc": "AUPRC",
    "roc_auc": "AUROC",
    "macro_f1": "Macro-F1",
    "g_means": "G-Means",
}

# (display_name, on-disk dirname). dirname = "" means use display name directly.
CONDITIONS = [
    # Historical benchmark directories.
    ("off_policy",            "g_opd_flash_off_policy"),
    ("all_node_mh",           "g_opd_flash_all_node_mh"),
    ("det_mask",              "g_opd_flash_det_mask"),
    ("det_mask_mh",           "g_opd_flash_det_mask_mh"),
    ("det_mask_no_rel",       "g_opd_flash_det_mask_no_rel"),
    ("det_mask_fixed_bce",    "g_opd_flash_det_mask_fixed_bce"),
    ("det_mask_rev_only",     "g_opd_flash_det_mask_rev_only"),
    ("det_mask_single_denom", "g_opd_flash_det_mask_single_denom"),
    ("det_mask_cbr",          "g_opd_flash_det_mask_cbr"),  # CBR-K1 (λ=0.5 lin)
    ("g_opd_flash",           "g_opd_flash_g_opd_flash"),
    ("opd_action_strict",     "g_opd_flash_opd_action_strict"),
    ("opd_action_strict_mh",  "g_opd_flash_opd_action_strict_mh"),
    # 9 CBR ablation arms (v3)
    ("cbr_best",       "v3_cbr_best"),       # λ=1.0 exp (HEADLINE)
    ("cbr_l10",        "v3_cbr_l10"),        # λ=1.0 linear
    ("cbr_exp",        "v3_cbr_exp"),        # λ=0.5 exp
    ("cbr_sq",         "v3_cbr_sq"),         # square weight
    ("cbr_bin",        "v3_cbr_bin"),        # binary weight
    ("cbr_sym",        "v3_cbr_sym"),        # symmetric reward
    ("cbr_mask_HT",    "v3_cbr_mask_HT"),    # mask by H(p_T)
    ("cbr_mask_dis",   "v3_cbr_mask_disagree"),  # mask by base↔teacher flip
    ("cbr_mask_rand",  "v3_cbr_mask_rand"),  # random K
    ("cbr_mh",         "v3_cbr_mh"),         # CBR + multi-head
    ("cbr_hc_teacher", "v3_cbr_hc_teacher"), # CBR on hand-crafted teacher
    # Static distillation baselines
    ("hc_distill",     "idea2c_distill_adapter"),     # hand-crafted teacher
    ("lree_distill",   "idea2c_distill_adapter_2b"),  # LREE teacher
]
BASELINE_COND = "det_mask"


def _load_one(ds: str, base: str, dirname: str, seed: int) -> dict[str, float] | None:
    metrics_path = (
        Path("artifacts/results") / ds / base / dirname / f"seed_{seed}"
        / "stage3_metrics.json"
    )
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        m = json.load(f)
    out: dict[str, float] = {}
    for met in METRICS:
        # JSON uses "auprc" / "roc_auc" / "macro_f1" / "g_means" — match.
        v = m.get(met, float("nan"))
        try:
            out[met] = float(v)
        except (TypeError, ValueError):
            out[met] = float("nan")
    return out


def _collect_all() -> dict[tuple[str, str, str], dict[int, dict[str, float]]]:
    cells: dict[tuple[str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    for ds in DATASETS:
        for base in BASES:
            for cond_name, dirname in CONDITIONS:
                for seed in SEEDS:
                    m = _load_one(ds, base, dirname, seed)
                    if m is None:
                        continue
                    cells[(ds, base, cond_name)][seed] = m
    return cells


def _paired_t_one_sided(a: list[float], b: list[float]) -> tuple[float, float]:
    """One-sided paired t: H1: a > b. Returns (t-stat, p-value)."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    arr = np.array(a) - np.array(b)
    if np.allclose(arr, 0):
        return 0.0, 1.0
    t, p_two = stats.ttest_1samp(arr, 0.0)
    if np.isnan(t):
        return float("nan"), float("nan")
    p_one = (p_two / 2) if t > 0 else (1 - p_two / 2)
    return float(t), float(p_one)


def _star(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "★★★"
    if p < 0.01:
        return "★★"
    if p < 0.05:
        return "★"
    return ""


def aggregate(
    cells: dict[tuple[str, str, str], dict[int, dict[str, float]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """For every (cell × condition × metric): compute mean±std + paired-t vs det_mask.

    Returns flat rows (one per cell × cond × metric) + a summary dict
    (cross-cell sig counts per condition per metric).
    """
    rows: list[dict[str, Any]] = []
    # cross-cell tracker[(cond, metric)] -> {cells, dir_pos, sig05, sig01}
    tracker: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"cells": 0, "dir_pos": 0, "sig05": 0, "sig01": 0}
    )

    for ds in DATASETS:
        for base in BASES:
            base_runs = cells.get((ds, base, BASELINE_COND), {})
            for cond_name, _ in CONDITIONS:
                runs = cells.get((ds, base, cond_name), {})
                if not runs:
                    continue
                for met in METRICS:
                    vals = [runs[s][met] for s in SEEDS
                            if s in runs and not math.isnan(runs[s][met])]
                    if not vals:
                        continue
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    paired_seeds = [s for s in SEEDS if s in runs and s in base_runs
                                    and not math.isnan(runs[s][met])
                                    and not math.isnan(base_runs[s].get(met, float("nan")))]
                    paired_a = [runs[s][met] for s in paired_seeds]
                    paired_b = [base_runs[s][met] for s in paired_seeds]
                    if cond_name != BASELINE_COND and len(paired_seeds) >= 2:
                        t, p = _paired_t_one_sided(paired_a, paired_b)
                        delta = float(np.mean(paired_a)) - float(np.mean(paired_b))
                    else:
                        t, p, delta = float("nan"), float("nan"), float("nan")
                    sig = _star(p)
                    rows.append({
                        "dataset": ds, "base": base, "condition": cond_name,
                        "metric": met, "n_seeds": len(vals),
                        "mean": mean, "std": std,
                        "delta_vs_det_mask": delta,
                        "t_vs_det_mask": t,
                        "p_vs_det_mask": p,
                        "sig_vs_det_mask": sig,
                    })
                    if cond_name != BASELINE_COND:
                        tk = tracker[(cond_name, met)]
                        tk["cells"] += 1
                        if np.isfinite(delta) and delta > 0:
                            tk["dir_pos"] += 1
                        if np.isfinite(p) and p < 0.05:
                            tk["sig05"] += 1
                        if np.isfinite(p) and p < 0.01:
                            tk["sig01"] += 1

    summary: dict[str, Any] = {
        "baseline_condition": BASELINE_COND,
        "n_cells_total": len(DATASETS) * len(BASES),
        "statistical_convention": (
            "Per-cell paired-t (5-seed, one-sided H1: condition > det_mask). "
            "Cross-cell aggregation reports directional / sig counts; "
            "pooled-cell paired-t NOT reported (Critic round-4 fix)."
        ),
        "per_condition": {},
    }
    for cond_name, _ in CONDITIONS:
        if cond_name == BASELINE_COND:
            continue
        summary["per_condition"][cond_name] = {}
        for met in METRICS:
            tk = tracker[(cond_name, met)]
            summary["per_condition"][cond_name][met] = dict(tk)
    return rows, summary


def render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    out: list[str] = []
    out.append("# Flash-RAER + CBR family — 4-metric 8-cell × 5-seed paired-t\n")
    out.append("**Baseline for paired-t**: `det_mask` (deterministic top-K Flash-RAER, one-sided H1: condition > det_mask).\n")
    out.append("**Significance (df=4)**: `★` p<0.05 · `★★` p<0.01 · `★★★` p<0.001.\n")
    out.append("**Provenance**: per-seed `artifacts/results/{ds}/{base}/{cond_dirname}/seed_{s}/stage3_metrics.json` — see `CONDITIONS` map in `scripts/aggregate_cbr_flash_4metric.py`.\n")

    # Per-cell tables, one section per metric
    for met in METRICS:
        out.append(f"\n## {METRIC_LABEL[met]}\n")
        out.append("| Dataset | Base | Condition | n | Mean ± std | Δ vs det_mask | t | p | sig |")
        out.append("|---|---|---|---:|---:|---:|---:|---:|:---:|")
        for r in rows:
            if r["metric"] != met:
                continue
            mean = f"{r['mean']:.4f} ± {r['std']:.4f}"
            delta = f"{r['delta_vs_det_mask']:+.4f}" if np.isfinite(r['delta_vs_det_mask']) else ""
            t = f"{r['t_vs_det_mask']:+.2f}" if np.isfinite(r['t_vs_det_mask']) else ""
            p = f"{r['p_vs_det_mask']:.4f}" if np.isfinite(r['p_vs_det_mask']) else ""
            out.append(
                f"| {r['dataset']} | {r['base']} | `{r['condition']}` | {r['n_seeds']} | "
                f"{mean} | {delta} | {t} | {p} | {r['sig_vs_det_mask']} |"
            )

    out.append("\n## Cross-cell sig summary (per condition × per metric)\n")
    out.append("| Condition | Metric | Cells | Dir + | Sig p<0.05 | Sig p<0.01 |")
    out.append("|---|---|---:|---:|---:|---:|")
    for cond, mets in summary["per_condition"].items():
        for met in METRICS:
            tk = mets[met]
            out.append(
                f"| `{cond}` | {METRIC_LABEL[met]} | {tk['cells']} | "
                f"{tk['dir_pos']}/{tk['cells']} | "
                f"{tk['sig05']}/{tk['cells']} | "
                f"{tk['sig01']}/{tk['cells']} |"
            )
    return "\n".join(out) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_md", type=str, default="artifacts/tables/c3_4metric_8cell_5seed.md")
    p.add_argument("--out_csv", type=str, default="artifacts/tables/c3_4metric_8cell_5seed.csv")
    p.add_argument("--out_json", type=str, default="artifacts/tables/c3_4metric_summary.json")
    args = p.parse_args()

    cells = _collect_all()
    n_runs = sum(len(v) for v in cells.values())
    print(f"[c3-4metric] Loaded {n_runs} runs from {len(cells)} (cell, cond) combos.")

    rows, summary = aggregate(cells)

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(rows, summary))
    write_csv(rows, Path(args.out_csv))
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"[c3-4metric] md   → {args.out_md}")
    print(f"[c3-4metric] csv  → {args.out_csv}")
    print(f"[c3-4metric] json → {args.out_json}")


if __name__ == "__main__":
    main()
