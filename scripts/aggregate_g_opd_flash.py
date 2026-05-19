"""T5 — Aggregate G-OPD-Flash 8-cell × 5-seed results into paired-t table.

Reads ``artifacts/results/{ds}/{base}/g_opd_flash_{mode}/seed_{s}/stage3_metrics.json``
and ``..._diagnostics.json`` for all (ds, base, seed, mode) combos.

Outputs:

* ``artifacts/tables/g_opd_flash_8cell_5seed.md`` — markdown table with
  per-cell mean ± std AUPRC for each mode, plus paired-t p-values vs the
  baselines (base only / full teacher / off_policy / det_mask), plus
  AUPRC capture rate vs teacher.

* ``artifacts/tables/g_opd_flash_8cell_5seed.csv`` — same content, machine
  readable.

* ``artifacts/tables/g_opd_flash_8cell_5seed_summary.json`` — overall
  summary (paired-t aggregated across cells, capture rate, speed-up).

Per-mode paired-t hypothesis (per cell, 5-seed):
  H0: mode == baseline (AUPRC equal)
  H1: mode > baseline
  reported as one-sided t-statistic + p-value.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
MODES = ("off_policy", "all_node_mh", "det_mask", "g_opd_flash", "opd_action_strict", "opd_action_strict_mh")
DEFAULT_BASELINE_MODE = "off_policy"


def _load_one(ds: str, base: str, mode: str, seed: int) -> dict[str, float] | None:
    metrics_path = (
        Path("artifacts/results") / ds / base / f"g_opd_flash_{mode}" / f"seed_{seed}"
        / "stage3_metrics.json"
    )
    diag_path = (
        Path("artifacts/logs") / ds / base / f"g_opd_flash_{mode}" / f"seed_{seed}"
        / "phase2_diagnostics.json"
    )
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        m = json.load(f)
    out = {
        "auprc": float(m.get("auprc", float("nan"))),
        "roc_auc": float(m.get("roc_auc", float("nan"))),
        "macro_f1": float(m.get("macro_f1", float("nan"))),
    }
    if diag_path.exists():
        with open(diag_path) as f:
            d = json.load(f)
        out["teacher_auprc"] = float(d.get("test_metrics_teacher", {}).get("auprc", float("nan")))
        out["base_auprc"] = float(d.get("test_metrics_base_only", {}).get("auprc", float("nan")))
        out["elapsed"] = float(d.get("elapsed_seconds", float("nan")))
        out["student_params"] = int(d.get("student_params", 0))
        out["teacher_params"] = int(d.get("teacher_params", 0))
    return out


def _collect_all() -> dict[tuple[str, str, str], dict[int, dict[str, float]]]:
    cells: dict[tuple[str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    for ds in DATASETS:
        for base in BASES:
            for mode in MODES:
                for seed in SEEDS:
                    m = _load_one(ds, base, mode, seed)
                    if m is None:
                        continue
                    cells[(ds, base, mode)][seed] = m
    return cells


def _paired_t_one_sided(a: list[float], b: list[float]) -> tuple[float, float]:
    """One-sided paired t: H1: a > b.  Returns (t-stat, p-value)."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    arr = np.array(a) - np.array(b)
    if np.allclose(arr, 0):
        return 0.0, 1.0
    t, p_two = stats.ttest_1samp(arr, 0.0)
    # One-sided (H1: mean > 0): p_one = p_two / 2 if t > 0 else 1 - p_two/2
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
    baseline_mode: str = DEFAULT_BASELINE_MODE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-cell paired-t aggregation.

    **Statistical convention (Critic round-4 fix)**: paired-t is computed
    **per (ds, base) cell only**, pairing AUPRCs by ``(seed)`` tuple
    explicitly.  Cross-cell aggregation uses ``directional_positive`` and
    ``sig_cells (p<0.05)`` counts, NOT a pooled t-test — pooling cells
    treats them as IID and inflates apparent sample size.
    """
    rows: list[dict[str, Any]] = []
    # cross-cell directional / sig tracker (idea2b-style report)
    cross_cell_tracker: dict[str, dict[str, int]] = {}

    for ds in DATASETS:
        for base in BASES:
            base_runs = cells.get((ds, base, baseline_mode), {})

            # Discover the union of teacher/base AUPRCs from any run with a diag
            teacher_auprcs: list[float] = []
            base_only_auprcs: list[float] = []
            for mode in MODES:
                for s in SEEDS:
                    m = cells.get((ds, base, mode), {}).get(s)
                    if m and not math.isnan(m.get("teacher_auprc", float("nan"))):
                        teacher_auprcs.append(m["teacher_auprc"])
                        break
                if teacher_auprcs:
                    break
            for mode in MODES:
                for s in SEEDS:
                    m = cells.get((ds, base, mode), {}).get(s)
                    if m and not math.isnan(m.get("base_auprc", float("nan"))):
                        base_only_auprcs.append(m["base_auprc"])
                        break
                if base_only_auprcs:
                    break
            teacher_auprc = float(np.mean(teacher_auprcs)) if teacher_auprcs else float("nan")
            base_only_auprc = float(np.mean(base_only_auprcs)) if base_only_auprcs else float("nan")

            for mode in MODES:
                runs = cells.get((ds, base, mode), {})
                if not runs:
                    rows.append({
                        "dataset": ds, "base": base, "mode": mode,
                        "n_seeds": 0,
                        "auprc_mean": float("nan"), "auprc_std": float("nan"),
                        "capture_pct": float("nan"),
                        "t_vs_baseline": float("nan"), "p_vs_baseline": float("nan"),
                        "sig_vs_baseline": "",
                        "teacher_auprc": teacher_auprc,
                        "base_only_auprc": base_only_auprc,
                    })
                    continue

                # ── Tuple-key paired sample construction ────────────────────
                # Only pair AUPRCs for seeds present in BOTH `mode` and
                # `baseline_mode`.  This makes the paired-t well-defined
                # even when a seed is missing for one of the two modes.
                paired_seeds = [s for s in SEEDS if s in runs and s in base_runs]
                auprcs_mode = [runs[s]["auprc"] for s in paired_seeds]
                auprcs_base = [base_runs[s]["auprc"] for s in paired_seeds]
                # For mean/std, use ALL seeds present for this mode (even if
                # baseline is missing) — gives the most honest mean.
                all_auprcs_mode = [runs[s]["auprc"] for s in SEEDS if s in runs]

                row = {
                    "dataset": ds, "base": base, "mode": mode,
                    "n_seeds": len(all_auprcs_mode),
                    "n_paired_vs_baseline": len(paired_seeds),
                    "auprc_mean": float(np.mean(all_auprcs_mode)),
                    "auprc_std": (float(np.std(all_auprcs_mode, ddof=1))
                                  if len(all_auprcs_mode) > 1 else 0.0),
                    "capture_pct": (
                        (float(np.mean(all_auprcs_mode)) / teacher_auprc * 100)
                        if teacher_auprc > 0 else float("nan")
                    ),
                    "teacher_auprc": teacher_auprc,
                    "base_only_auprc": base_only_auprc,
                }
                if mode != baseline_mode and len(paired_seeds) >= 2:
                    t, p = _paired_t_one_sided(auprcs_mode, auprcs_base)
                    row["t_vs_baseline"] = t
                    row["p_vs_baseline"] = p
                    row["sig_vs_baseline"] = _star(p)
                else:
                    row["t_vs_baseline"] = float("nan")
                    row["p_vs_baseline"] = float("nan")
                    row["sig_vs_baseline"] = ""
                rows.append(row)

                # cross-cell counter (per mode, NOT per cell)
                ctr = cross_cell_tracker.setdefault(mode, {
                    "cells_evaluated": 0,
                    "directional_positive": 0,
                    "sig_cells_p05": 0,
                    "sig_cells_p01": 0,
                    "n_total_runs": 0,
                })
                ctr["cells_evaluated"] += 1
                ctr["n_total_runs"] += len(all_auprcs_mode)
                if mode != baseline_mode and len(paired_seeds) >= 2:
                    delta = float(np.mean(auprcs_mode)) - float(np.mean(auprcs_base))
                    if delta > 0:
                        ctr["directional_positive"] += 1
                    if np.isfinite(row["p_vs_baseline"]) and row["p_vs_baseline"] < 0.05:
                        ctr["sig_cells_p05"] += 1
                    if np.isfinite(row["p_vs_baseline"]) and row["p_vs_baseline"] < 0.01:
                        ctr["sig_cells_p01"] += 1

    # ── Cross-cell summary (NO pooled paired-t — Critic round-4 fix) ──────
    overall_summary: dict[str, Any] = {
        "baseline_mode": baseline_mode,
        "n_cells_total": len(DATASETS) * len(BASES),
        "statistical_convention": (
            "Per-cell paired-t (5-seed, one-sided H1: mode > baseline). "
            "Cross-cell aggregation reports directional / sig counts only; "
            "pooled paired-t across cells is NOT reported because cells are "
            "not IID (Critic round-4 fix)."
        ),
        "per_mode": {},
    }
    for mode, ctr in cross_cell_tracker.items():
        # Compute per-mode mean AUPRC across all cells (unweighted).
        mode_means = [r["auprc_mean"] for r in rows
                      if r["mode"] == mode and np.isfinite(r["auprc_mean"])]
        overall_summary["per_mode"][mode] = {
            "cells_evaluated": ctr["cells_evaluated"],
            "n_total_runs": ctr["n_total_runs"],
            "auprc_mean_across_cells": (
                float(np.mean(mode_means)) if mode_means else float("nan")
            ),
            "auprc_std_across_cells": (
                float(np.std(mode_means, ddof=1)) if len(mode_means) > 1 else 0.0
            ),
            "directional_positive_cells": ctr["directional_positive"],
            "sig_cells_p05": ctr["sig_cells_p05"],
            "sig_cells_p01": ctr["sig_cells_p01"],
        }

    return rows, overall_summary


def render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# G-OPD-Flash — 8-cell × 5-seed benchmark\n")
    lines.append(f"Baseline mode for paired-t: `{summary['baseline_mode']}`  (one-sided, H1: mode > baseline)\n")
    lines.append("Capture % = mean student AUPRC / mean teacher AUPRC × 100\n")

    # Per-cell table
    lines.append("## Per-cell results\n")
    lines.append(
        "| Dataset | Base | Mode | n | Student AUPRC | Capture % | t vs baseline | p | sig | Teacher AUPRC |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---:|")
    for r in rows:
        auprc = f"{r['auprc_mean']:.4f}±{r['auprc_std']:.4f}" if np.isfinite(r["auprc_mean"]) else "n/a"
        cap = f"{r['capture_pct']:.1f}%" if np.isfinite(r["capture_pct"]) else "n/a"
        t = f"{r['t_vs_baseline']:+.2f}" if np.isfinite(r["t_vs_baseline"]) else ""
        p = f"{r['p_vs_baseline']:.4f}" if np.isfinite(r["p_vs_baseline"]) else ""
        teacher = f"{r['teacher_auprc']:.4f}" if np.isfinite(r["teacher_auprc"]) else "n/a"
        lines.append(
            f"| {r['dataset']} | {r['base']} | `{r['mode']}` | {r['n_seeds']} | {auprc} | {cap} | {t} | {p} | {r['sig_vs_baseline']} | {teacher} |"
        )

    lines.append("")
    lines.append("## Cross-cell summary (per mode)\n")
    lines.append(
        "**Statistical convention**: per-cell paired-t only (5-seed, one-sided "
        "H1: mode > baseline).  Cells are NOT pooled — pooling treats cells as "
        "IID and inflates apparent sample size (Critic round-4 fix).\n"
    )
    lines.append(
        "| Mode | Cells | Mean AUPRC across cells | Directional + cells | Sig cells (p<0.05) | Sig cells (p<0.01) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode, s in summary["per_mode"].items():
        auprc = (
            f"{s['auprc_mean_across_cells']:.4f}±{s['auprc_std_across_cells']:.4f}"
            if np.isfinite(s["auprc_mean_across_cells"]) else "n/a"
        )
        dir_total = s["cells_evaluated"] if mode != summary["baseline_mode"] else s["cells_evaluated"]
        lines.append(
            f"| `{mode}` | {s['cells_evaluated']} | {auprc} | "
            f"{s['directional_positive_cells']}/{dir_total} | "
            f"{s['sig_cells_p05']}/{dir_total} | {s['sig_cells_p01']}/{dir_total} |"
        )

    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate G-OPD-Flash 8-cell × 5-seed (T5)")
    p.add_argument("--out_md", type=str, default="artifacts/tables/g_opd_flash_8cell_5seed.md")
    p.add_argument("--out_csv", type=str, default="artifacts/tables/g_opd_flash_8cell_5seed.csv")
    p.add_argument("--out_summary_json", type=str,
                   default="artifacts/tables/g_opd_flash_8cell_5seed_summary.json")
    p.add_argument("--baseline_mode", type=str, default=DEFAULT_BASELINE_MODE,
                   choices=list(MODES))
    args = p.parse_args()

    cells = _collect_all()
    n_runs = sum(len(v) for v in cells.values())
    print(f"[T5/agg] Loaded {n_runs} runs from {len(cells)} (cell, mode) combos.")

    rows, summary = aggregate(cells, baseline_mode=args.baseline_mode)

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(rows, summary))
    write_csv(rows, Path(args.out_csv))
    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"[T5/agg] Wrote markdown → {args.out_md}")
    print(f"[T5/agg] Wrote csv      → {args.out_csv}")
    print(f"[T5/agg] Wrote summary  → {args.out_summary_json}")

    # Console preview
    print()
    print(render_markdown(rows, summary)[:2000])


if __name__ == "__main__":
    main()
