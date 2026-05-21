"""Aggregate interp-A incremental loss-component ablation (4 conditions × 5 seeds × N bases).

YelpChi-{bwgnn, sage, gcn} (or any subset via --bases CLI).  Tests the
3-component C3 locked loss form
    L = L_bce + α_f·L_KL + λ_cbr·L_CBR
under all 2² = 4 add-on subsets:

    interp_a_bce         — BCE only (baseline)
    interp_a_bce_kl      — BCE + rKL
    interp_a_bce_cbr     — BCE + CBR-BEST
    interp_a_bce_kl_cbr  — BCE + rKL + CBR-BEST  (= locked canonical)

Per-seed reads ``artifacts/results/{dataset}/{base}/{cond}/seed_{s}/stage3_metrics.json``;
computes per-(base,condition) mean ± std for 4 metrics + paired-t one-sided
vs the ``interp_a_bce`` baseline (paired within the same base × seed).

Outputs:

* ``artifacts/tables/c3_interp_a_ablation_{dataset}_{bases}_5seed.md``
* ``artifacts/tables/c3_interp_a_ablation_{dataset}_{bases}_5seed.csv``
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


SEEDS = (42, 123, 456, 789, 2026)
METRICS = ("auprc", "roc_auc", "macro_f1", "g_means")
METRIC_LABEL = {
    "auprc": "AUPRC",
    "roc_auc": "AUROC",
    "macro_f1": "Macro-F1",
    "g_means": "G-Means",
}
CONDITIONS = [
    ("interp_a_bce",        "BCE only",                  "baseline"),
    ("interp_a_bce_kl",     "BCE + rKL",                 "+ reverse-KL"),
    ("interp_a_bce_cbr",    "BCE + CBR-BEST",            "+ Contract-Budgeted Residual"),
    ("interp_a_bce_kl_cbr", "BCE + rKL + CBR-BEST",      "= locked canonical (3-component)"),
]
BASELINE = "interp_a_bce"


def _load_one(dataset: str, base: str, cond: str, seed: int) -> dict[str, float] | None:
    p = Path(f"artifacts/results/{dataset}/{base}/{cond}/seed_{seed}/stage3_metrics.json")
    if not p.exists():
        return None
    with open(p) as f:
        m = json.load(f)
    out: dict[str, float] = {}
    for met in METRICS:
        v = m.get(met, float("nan"))
        try:
            out[met] = float(v)
        except (TypeError, ValueError):
            out[met] = float("nan")
    return out


def _paired_t_one_sided(a: list[float], b: list[float]) -> tuple[float, float]:
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


def aggregate_for_base(dataset: str, base: str) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, float]]]]:
    """Aggregate per-(condition × metric) rows for a single base."""
    data: dict[str, dict[int, dict[str, float]]] = {}
    for cond, _, _ in CONDITIONS:
        data[cond] = {}
        for s in SEEDS:
            m = _load_one(dataset, base, cond, s)
            if m is not None:
                data[cond][s] = m

    rows: list[dict[str, Any]] = []
    for cond, label, desc in CONDITIONS:
        for met in METRICS:
            vals = [data[cond][s][met] for s in SEEDS
                    if s in data[cond] and not math.isnan(data[cond][s][met])]
            if not vals:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if cond == BASELINE:
                t, p, delta, sig = float("nan"), float("nan"), float("nan"), ""
            else:
                paired_seeds = [s for s in SEEDS
                                if s in data[cond] and s in data[BASELINE]
                                and not math.isnan(data[cond][s][met])
                                and not math.isnan(data[BASELINE][s][met])]
                a = [data[cond][s][met] for s in paired_seeds]
                b = [data[BASELINE][s][met] for s in paired_seeds]
                if len(paired_seeds) >= 2:
                    t, p = _paired_t_one_sided(a, b)
                    delta = float(np.mean(a)) - float(np.mean(b))
                else:
                    t, p, delta = float("nan"), float("nan"), float("nan")
                sig = _star(p)
            rows.append({
                "dataset": dataset, "base": base,
                "condition": cond, "label": label, "desc": desc,
                "metric": met, "n_seeds": len(vals),
                "mean": mean, "std": std,
                "delta_vs_baseline": delta,
                "t_vs_baseline": t,
                "p_vs_baseline": p,
                "sig": sig,
            })
    return rows, data


def render_markdown(all_rows: list[dict[str, Any]], dataset: str, bases: list[str]) -> str:
    out: list[str] = []
    out.append(f"# Interp-A incremental loss-component ablation — {dataset.capitalize()} × {{{', '.join(bases)}}} × 5 seeds\n")
    out.append("**Question**: which of the 3 locked C3 components ({BCE_distill, rKL, CBR-BEST}) is empirically needed across multiple bases?\n")
    out.append("**Baseline**: `interp_a_bce` = BCE_distill alone on top-K H(p_S) mask, with `--ablate_reliability --ablate_adaptive_bce` to remove the Z1-falsified weighting.\n")
    out.append(f"**Conditions**: 4 conditions × 5 seeds × {len(bases)} bases = {4*5*len(bases)} runs; mode = `det_mask_cbr` throughout; multi-head heads (`alpha_r`/`alpha_g`) permanently 0.\n")
    out.append("**Paired-t**: per-(base × condition) 5-seed, df=4, one-sided H1: condition > BCE-only baseline. `★` p<0.05 · `★★` p<0.01 · `★★★` p<0.001.\n")
    out.append("**Provenance**: per-seed `artifacts/results/{dataset}/{base}/interp_a_*/seed_{s}/stage3_metrics.json`; aggregator `scripts/aggregate_interp_a.py`; sweep `BASE={base} bash scripts/run_interp_a_ablation.sh PARALLEL`.\n")

    # ── Per-base × per-metric tables ──
    for base in bases:
        out.append(f"\n# Base: **{base.upper()}**\n")
        for met in METRICS:
            out.append(f"\n## {met.upper().replace('_', '-')}\n" if met == "roc_auc" else f"\n## {METRIC_LABEL[met]}\n")
            out.append("| Condition | Spec | Mean ± std | Δ vs baseline | t | p | sig |")
            out.append("|---|---|---:|---:|---:|---:|:---:|")
            for r in all_rows:
                if r["base"] != base or r["metric"] != met:
                    continue
                mean = f"{r['mean']:.4f} ± {r['std']:.4f}"
                if r["condition"] == BASELINE:
                    delta, t, p, sig = "reference", "—", "—", ""
                else:
                    delta = f"{r['delta_vs_baseline']:+.4f}" if np.isfinite(r['delta_vs_baseline']) else ""
                    t = f"{r['t_vs_baseline']:+.2f}" if np.isfinite(r['t_vs_baseline']) else ""
                    p = f"{r['p_vs_baseline']:.4f}" if np.isfinite(r['p_vs_baseline']) else ""
                    sig = r["sig"]
                out.append(
                    f"| **{r['label']}** | {r['desc']} | {mean} | {delta} | {t} | {p} | {sig} |"
                )

    # ── Cross-base × cross-metric headline (Δ vs baseline + sig) ──
    out.append("\n# Cross-base × cross-metric Δ headline\n")
    out.append(f"Each cell = `Δ (sig)` from paired-$t$ vs **base-local BCE-only baseline** (5 seeds per cell, df=4).\n")
    out.append(f"`★` p<0.05 · `★★` p<0.01 · `★★★` p<0.001 · blank = ns. Empty cell = baseline reference.\n")
    out.append("\n| Base | Condition | AUPRC | AUROC | Macro-F1 | G-Means |")
    out.append("|---|---|---|---|---|---|")
    for base in bases:
        for cond, label, _ in CONDITIONS:
            if cond == BASELINE:
                row = f"| {base} | **{label}** | reference | reference | reference | reference |"
            else:
                cells = []
                for met in METRICS:
                    m = next((r for r in all_rows if r["base"] == base and r["condition"] == cond and r["metric"] == met), None)
                    if m and np.isfinite(m["delta_vs_baseline"]):
                        s = m['sig'] if m['sig'] else ''
                        cells.append(f"{m['delta_vs_baseline']:+.4f} {s}".strip())
                    else:
                        cells.append("n/a")
                row = f"| {base} | **{label}** | " + " | ".join(cells) + " |"
            out.append(row)

    # ── Per-metric: how many bases each condition is sig on ──
    out.append("\n# Per-condition cross-base sig count\n")
    out.append(f"Total cells = {len(bases)} bases × 1 metric.\n")
    out.append("\n| Condition | AUPRC sig | AUROC sig | Macro-F1 sig | G-Means sig |")
    out.append("|---|:---:|:---:|:---:|:---:|")
    for cond, label, _ in CONDITIONS:
        if cond == BASELINE:
            continue
        cells = []
        for met in METRICS:
            count = sum(1 for r in all_rows
                        if r["condition"] == cond and r["metric"] == met
                        and np.isfinite(r["p_vs_baseline"])
                        and r["p_vs_baseline"] < 0.05)
            cells.append(f"{count}/{len(bases)}")
        out.append(f"| **{label}** | " + " | ".join(cells) + " |")

    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="yelpchi")
    p.add_argument("--bases", type=str, nargs="+", default=["bwgnn", "sage", "gcn"],
                   help="List of bases to aggregate.")
    p.add_argument("--out_md", type=str, default=None)
    p.add_argument("--out_csv", type=str, default=None)
    args = p.parse_args()

    all_rows: list[dict[str, Any]] = []
    for base in args.bases:
        rows, _ = aggregate_for_base(args.dataset, base)
        all_rows.extend(rows)

    bases_str = "_".join(args.bases)
    out_md = args.out_md or f"artifacts/tables/c3_interp_a_ablation_{args.dataset}_{bases_str}_5seed.md"
    out_csv = args.out_csv or f"artifacts/tables/c3_interp_a_ablation_{args.dataset}_{bases_str}_5seed.csv"

    md_text = render_markdown(all_rows, args.dataset, args.bases)
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(md_text)

    with open(out_csv, "w", newline="") as f:
        fields = list(all_rows[0].keys()) if all_rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    n_runs = sum(1 for r in all_rows if r["metric"] == "auprc") * 5  # per (base,cond) → 5 seeds
    print(f"[interp-a] {len(all_rows)} (base,cond,metric) rows from {n_runs} run files across {len(args.bases)} bases.")
    print(f"[interp-a] md  → {out_md}")
    print(f"[interp-a] csv → {out_csv}")
    print()
    print(md_text)


if __name__ == "__main__":
    main()
