"""Aggregate Phase 2 ablation results (8 loss + 2 arch + base reuse) into markdown.

Cells:
    L0..L7  : loss ablation on full architecture
    A0      : base only (reuse fixed_v1_100ep Phase 1 metrics)
    A1, A2  : architecture switches (relation-only, judge-only)
    A3==L7  : champion (full)

Per cell: mean ± std over 5 seeds for AUROC / AUPRC / Macro-F1 / G-Means.
Plus paired t-test vs L7 (champion) for each non-champion cell.
Output → artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md
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
OUT = TABLES / "yelpchi_bwgnn_ablation_loss_arch_5seed.md"

SEEDS = [42, 123, 456, 789, 2026]
METRICS = ["roc_auc", "auprc", "macro_f1", "g_means"]
METRIC_LABELS = {"roc_auc": "AUROC", "auprc": "AUPRC", "macro_f1": "Macro-F1", "g_means": "G-Means"}

CELLS = [
    ("A0", "base only", "z = b",                       None,                                       "reuse fixed_v1_100ep"),
    ("L0", "cls only",  "z = b + Δ_rel + α·Δ_llm",     "λ_int=0, λ_sp=0, λ_al=0",                  ""),
    ("L1", "+int",      "z = b + Δ_rel + α·Δ_llm",     "λ_int=3e-3, λ_sp=0, λ_al=0",               ""),
    ("L2", "+sparse",   "z = b + Δ_rel + α·Δ_llm",     "λ_int=0, λ_sp=1e-3, λ_al=0",               ""),
    ("L3", "+align",    "z = b + Δ_rel + α·Δ_llm",     "λ_int=0, λ_sp=0, λ_al=3e-2",               ""),
    ("L4", "+int +sp",  "z = b + Δ_rel + α·Δ_llm",     "λ_int=3e-3, λ_sp=1e-3, λ_al=0",            ""),
    ("L5", "+int +al",  "z = b + Δ_rel + α·Δ_llm",     "λ_int=3e-3, λ_sp=0, λ_al=3e-2",            ""),
    ("L6", "+sp +al",   "z = b + Δ_rel + α·Δ_llm",     "λ_int=0, λ_sp=1e-3, λ_al=3e-2",            ""),
    ("L7", "full ★",    "z = b + Δ_rel + α·Δ_llm",     "λ_int=3e-3, λ_sp=1e-3, λ_al=3e-2",         "champion (= A3)"),
    ("A1", "rel-only",  "z = b + Δ_rel",               "use_judge=0, α_max=0, λ_al=0",             "λ_al degenerate w/o judge"),
    ("A2", "judge-only","z = b + α·Δ_llm",             "Δ_rel_max=0",                              ""),
]


def load_cell_metrics(cell_id: str) -> dict[str, list[float]]:
    """Return per-metric list of length 5 (one per seed), or [] if missing."""
    out: dict[str, list[float]] = {m: [] for m in METRICS}

    if cell_id == "A0":
        # Reuse Phase 1 base test metrics
        for seed in SEEDS:
            p = LOGS / "fixed_v1_100ep" / f"seed_{seed}" / "stage1.json"
            if not p.exists():
                return {m: [] for m in METRICS}
            with open(p) as f:
                data = json.load(f)
            tm = data.get("test_metrics", {})
            for m in METRICS:
                out[m].append(float(tm[m]))
        return out

    # CoVER cells: read phase2_diagnostics.json
    for seed in SEEDS:
        p = LOGS / f"ablation_{cell_id}" / f"seed_{seed}" / "phase2_diagnostics.json"
        if not p.exists():
            return {m: [] for m in METRICS}
        with open(p) as f:
            data = json.load(f)
        tm = data.get("test_metrics", {})
        for m in METRICS:
            out[m].append(float(tm[m]))
    return out


def fmt_pm(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"


def paired_test(cell_vals: list[float], champ_vals: list[float]) -> str:
    if len(cell_vals) != 5 or len(champ_vals) != 5:
        return "—"
    diff = np.array(cell_vals) - np.array(champ_vals)
    if np.allclose(diff, 0):
        return "0.000 (n.s.)"
    res = ttest_rel(cell_vals, champ_vals)
    t = res.statistic
    p = res.pvalue
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    return f"Δ={np.mean(diff):+.4f} (t={t:+.2f}, p={p:.3g}) {sig}"


def main():
    # Collect
    all_metrics: dict[str, dict[str, list[float]]] = {}
    for cell_id, *_ in CELLS:
        all_metrics[cell_id] = load_cell_metrics(cell_id)

    champ = all_metrics["L7"]

    # Header
    lines: list[str] = []
    lines.append("# YelpChi BWGNN — Phase 2 Loss × Architecture Ablation (5 seeds)\n")
    lines.append(f"**Base**: `fixed_v1_100ep` (BWGNN, paper-faithful, frozen)  ")
    lines.append(f"**Champion (L7=A3)**: full 4-loss + full arch, `α_bias_init=0.0, α_max=0.3, λ_align=3e-2`  ")
    lines.append(f"**Seeds**: {SEEDS}  ")
    lines.append(f"**Statistical test**: paired t-test (per-seed Δ vs L7), 5 seeds → df=4\n")

    lines.append("## Cell legend\n")
    lines.append("| Cell | Variant | Formula | HP override | Note |")
    lines.append("|---|---|---|---|---|")
    for cell_id, var, formula, hp, note in CELLS:
        hp_str = hp if hp else "—"
        lines.append(f"| **{cell_id}** | {var} | `{formula}` | {hp_str} | {note} |")
    lines.append("")

    # Main result table
    lines.append("## Results (mean ± std, 5 seeds)\n")
    head = "| Cell | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |"
    sep = "|---|" + "---:|" * len(METRICS)
    lines.append(head)
    lines.append(sep)
    for cell_id, *_ in CELLS:
        cells = [fmt_pm(all_metrics[cell_id][m]) for m in METRICS]
        marker = " **★**" if cell_id == "L7" else ""
        lines.append(f"| {cell_id}{marker} | " + " | ".join(cells) + " |")
    lines.append("")

    # Paired t vs L7
    lines.append("## Paired t-test vs L7 (champion) — AUPRC\n")
    lines.append("| Cell | Variant | AUPRC paired Δ vs L7 |")
    lines.append("|---|---|---|")
    for cell_id, var, *_ in CELLS:
        if cell_id == "L7":
            lines.append(f"| L7 | full ★ | reference |")
            continue
        sig = paired_test(all_metrics[cell_id]["auprc"], champ["auprc"])
        lines.append(f"| {cell_id} | {var} | {sig} |")
    lines.append("")

    # Save per-seed CSV as well
    csv_path = TABLES / "yelpchi_bwgnn_ablation_loss_arch_per_seed.csv"
    with open(csv_path, "w") as f:
        f.write("cell,seed," + ",".join(METRICS) + "\n")
        for cell_id, *_ in CELLS:
            data = all_metrics[cell_id]
            n = len(data[METRICS[0]])
            for i in range(n):
                row = [cell_id, str(SEEDS[i])] + [f"{data[m][i]:.6f}" for m in METRICS]
                f.write(",".join(row) + "\n")
    lines.append(f"## Per-seed CSV: `{csv_path.relative_to(REPO)}`\n")

    # Significance code legend
    lines.append("---\n")
    lines.append("**Significance**: `***` p<0.001, `**` p<0.01, `*` p<0.05, `n.s.` p≥0.05\n")

    OUT.write_text("\n".join(lines))
    print(f"[ok] wrote {OUT.relative_to(REPO)} ({len(lines)} lines)")
    print(f"[ok] wrote {csv_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
