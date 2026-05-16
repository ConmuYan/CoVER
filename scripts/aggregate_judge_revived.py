"""Aggregate Phase 2 revived-judge ablation into a markdown comparison table.

Cells:
    L7_revived  : full champion with 2000-node vLLM judge (NEW pipeline)
    A1_revived  : relation-only baseline (judge off, sanity check)
    A2_revived  : judge-only (Δ_rel=0, key indicator of judge activation)
    L7_legacy   : full champion with original 120-node sparse judge
    A0_base     : reuse fixed_v1_100ep Phase 1 metrics

Outputs:
    artifacts/tables/yelpchi_bwgnn_judge_revived_5seed.md
    artifacts/tables/yelpchi_bwgnn_judge_revived_per_seed.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "artifacts/logs/yelpchi/bwgnn"
TABLES = REPO / "artifacts/tables"
TABLES.mkdir(parents=True, exist_ok=True)
OUT = TABLES / "yelpchi_bwgnn_judge_revived_5seed.md"

SEEDS = [42, 123, 456, 789, 2026]
METRICS = ["roc_auc", "auprc", "macro_f1", "g_means"]
METRIC_LABELS = {"roc_auc": "AUROC", "auprc": "AUPRC", "macro_f1": "Macro-F1", "g_means": "G-Means"}

CELLS = [
    ("A0_base",     "base only",   "Phase 1 fixed_v1_100ep",                "—"),
    ("L7_legacy",   "full (120 pkt)",  "z = b + Δ_rel + α·Δ_llm",            "old judge, ~50 accepted/seed"),
    ("L7_revived",  "full (2000 pkt)", "z = b + Δ_rel + α·Δ_llm",            "vLLM judge, ~800 accepted/seed"),
    ("A1_revived",  "rel-only",    "z = b + Δ_rel",                          "judge off"),
    ("A2_revived",  "judge-only",  "z = b + α·Δ_llm",                        "Δ_rel=0"),
]


def load_cell(cell_id: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {m: [] for m in METRICS}
    if cell_id == "A0_base":
        for seed in SEEDS:
            p = LOGS / "fixed_v1_100ep" / f"seed_{seed}" / "stage1.json"
            if not p.exists():
                return {m: [] for m in METRICS}
            d = json.load(open(p))
            tm = d.get("test_metrics", {})
            for m in METRICS:
                out[m].append(float(tm[m]))
        return out
    for seed in SEEDS:
        p = LOGS / f"ablation_{cell_id}" / f"seed_{seed}" / "phase2_diagnostics.json"
        if not p.exists():
            return {m: [] for m in METRICS}
        d = json.load(open(p))
        tm = d.get("test_metrics", {})
        for m in METRICS:
            out[m].append(float(tm[m]))
    return out


def load_diag(cell_id: str, key: str) -> list[float]:
    out: list[float] = []
    if cell_id == "A0_base":
        return out
    for seed in SEEDS:
        p = LOGS / f"ablation_{cell_id}" / f"seed_{seed}" / "phase2_diagnostics.json"
        if not p.exists():
            return []
        d = json.load(open(p))
        fd = d.get("final_diagnostics", {})
        v = fd.get(key)
        if v is not None:
            out.append(float(v))
    return out


def fmt_pm(vals: list[float], dec: int = 4) -> str:
    if not vals:
        return "—"
    return f"{np.mean(vals):.{dec}f} ± {np.std(vals, ddof=1):.{dec}f}"


def paired(cell: list[float], ref: list[float]) -> str:
    if len(cell) != 5 or len(ref) != 5:
        return "—"
    diff = np.array(cell) - np.array(ref)
    if np.allclose(diff, 0):
        return "Δ=0 (n.s.)"
    res = ttest_rel(cell, ref)
    t, p = res.statistic, res.pvalue
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    return f"Δ={np.mean(diff):+.4f} (t={t:+.2f}, p={p:.3g}) {sig}"


def main():
    all_m = {c[0]: load_cell(c[0]) for c in CELLS}
    legacy = all_m["L7_legacy"]
    revived = all_m["L7_revived"]
    a1 = all_m["A1_revived"]
    a2 = all_m["A2_revived"]
    base = all_m["A0_base"]

    lines: list[str] = []
    lines.append("# YelpChi BWGNN — Judge Revival Ablation (5 seeds)\n")
    lines.append("**Base**: `fixed_v1_100ep` (BWGNN, paper-faithful, frozen)  ")
    lines.append("**Judge revival**: 120 packets random → 2000 packets base-uncertain + vLLM (Qwen3-4B)  ")
    lines.append(f"**Seeds**: {SEEDS}\n")

    lines.append("## Cell legend\n")
    lines.append("| Cell | Variant | Formula | Note |")
    lines.append("|---|---|---|---|")
    for cid, var, formula, note in CELLS:
        lines.append(f"| **{cid}** | {var} | `{formula}` | {note} |")
    lines.append("")

    lines.append("## Headline metrics (mean ± std)\n")
    head = "| Cell | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |"
    sep = "|---|" + "---:|" * len(METRICS)
    lines.append(head)
    lines.append(sep)
    for cid, *_ in CELLS:
        cells = [fmt_pm(all_m[cid][m]) for m in METRICS]
        marker = ""
        if cid == "L7_revived":
            marker = " 🆕"
        elif cid == "L7_legacy":
            marker = " ★(old)"
        lines.append(f"| {cid}{marker} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Critical paired t-tests — AUPRC\n")
    lines.append("| Comparison | Δ (paired t-test) | Interpretation |")
    lines.append("|---|---|---|")
    lines.append(f"| L7_revived vs L7_legacy | {paired(revived['auprc'], legacy['auprc'])} | Did 2000-pkt judge help? |")
    lines.append(f"| L7_revived vs A1_revived | {paired(revived['auprc'], a1['auprc'])} | **Is judge alive in full?** |")
    lines.append(f"| A2_revived vs A0_base    | {paired(a2['auprc'], base['auprc'])} | **Judge-only beat base?** |")
    lines.append(f"| L7_revived vs A0_base    | {paired(revived['auprc'], base['auprc'])} | Total CoVER lift |")
    lines.append("")

    # Judge activation diagnostics
    lines.append("## Judge activation diagnostics (L7_revived vs L7_legacy)\n")
    diag_keys = [
        ("mean_abs_delta_rel",                "|Δ_rel| (relation residual)"),
        ("mean_abs_alpha_delta_llm",          "|α·Δ_llm| full-sample"),
        ("mean_abs_alpha_delta_llm_accepted", "|α·Δ_llm| on accepted"),
        ("mean_alpha_llm_accepted",           "α on accepted"),
        ("mean_intervention",                 "total intervention |z-b|"),
    ]
    lines.append("| Metric | L7_legacy | L7_revived | Δ |")
    lines.append("|---|---:|---:|---:|")
    for key, label in diag_keys:
        l = load_diag("L7_legacy", key)
        r = load_diag("L7_revived", key)
        delta = (np.mean(r) - np.mean(l)) if (l and r) else 0
        lines.append(f"| {label} | {fmt_pm(l, 5)} | {fmt_pm(r, 5)} | {delta:+.5f} |")
    lines.append("")

    # Per-seed CSV
    csv = TABLES / "yelpchi_bwgnn_judge_revived_per_seed.csv"
    with open(csv, "w") as f:
        f.write("cell,seed," + ",".join(METRICS) + "\n")
        for cid, *_ in CELLS:
            d = all_m[cid]
            for i in range(len(d[METRICS[0]])):
                row = [cid, str(SEEDS[i])] + [f"{d[m][i]:.6f}" for m in METRICS]
                f.write(",".join(row) + "\n")
    lines.append(f"## Per-seed CSV: `{csv.relative_to(REPO)}`\n")
    lines.append("---")
    lines.append("**Significance**: `***` p<0.001, `**` p<0.01, `*` p<0.05, `n.s.` p≥0.05")

    OUT.write_text("\n".join(lines))
    print(f"[ok] wrote {OUT.relative_to(REPO)} ({len(lines)} lines)")
    print(f"[ok] wrote {csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
