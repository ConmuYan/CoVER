"""Aggregate the historical Phase 2 loss × architecture ablation (10 cells × 5
seeds) into ``artifacts/tables/yelpchi_bwgnn_ablation_loss_arch_5seed.md``.

This is the **falsification table** — it is the evidence that retired every
deleted loss term and the LLM-judge route. The runs themselves were produced
by an earlier 4-term-loss trainer; see AGENTS.md §9 and §7.1 for the
interpretation in the cls-only canonical narrative.

Cells (frozen historical record):

* ``A0``       — base only, ``z = b`` (Phase 1 ``fixed_v1_100ep``)
* ``L0..L7``   — 8 loss settings on the full architecture
                 (``z = b + Δ_rel + α·Δ_llm``, varying λ_int, λ_sparse, λ_align)
* ``A1``       — relation-only switch (``α_max = 0``)
* ``A2``       — judge-only switch (``Δ_rel_max = 0``)

Per cell: mean ± std over 5 seeds for AUROC / AUPRC / Macro-F1 / G-Means.
Plus paired t-test vs ``L7`` (the historical "champion" — equivalent to
``A3`` in the original 4-term schema). Headline result: L0 cls-only differs
from L7 4-term-full by Δ_AUPRC = +0.0000 (p = 0.991), and every L_i term
fails the paired-t bar individually. → Canonical loss collapsed to L_cls
only; see ``training/phase2_losses.py``.
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

# (cell_id, variant_name, historical_formula, historical_hp_override, note)
# Formulas describe what was *historically run* (with the 4-term loss and judge
# active in L-series; A1 = judge off; A2 = rel-residual off). The cls-only
# canonical retired α·Δ_llm post-hoc because L7 ≡ L0 at 5-seed paired-t.
CELLS = [
    ("A0", "base only",  "z = b",                    "—",                                    "reuse fixed_v1_100ep (Phase 1)"),
    ("L0", "cls only",   "z = b + Δ_rel + α·Δ_llm",  "λ_int=0, λ_sp=0, λ_al=0",              "★ canonical reference (= retained loss)"),
    ("L1", "+int",       "z = b + Δ_rel + α·Δ_llm",  "λ_int=3e-3, λ_sp=0, λ_al=0",           "falsified vs L0 (p=0.81)"),
    ("L2", "+sparse",    "z = b + Δ_rel + α·Δ_llm",  "λ_int=0, λ_sp=1e-3, λ_al=0",           "falsified vs L0 (p=0.061)"),
    ("L3", "+align",     "z = b + Δ_rel + α·Δ_llm",  "λ_int=0, λ_sp=0, λ_al=3e-2",           "falsified vs L0 (p=0.32)"),
    ("L4", "+int +sp",   "z = b + Δ_rel + α·Δ_llm",  "λ_int=3e-3, λ_sp=1e-3, λ_al=0",        "falsified vs L0 (p=0.36)"),
    ("L5", "+int +al",   "z = b + Δ_rel + α·Δ_llm",  "λ_int=3e-3, λ_sp=0, λ_al=3e-2",        "falsified vs L0 (p=0.51)"),
    ("L6", "+sp +al",    "z = b + Δ_rel + α·Δ_llm",  "λ_int=0, λ_sp=1e-3, λ_al=3e-2",        "falsified vs L0 (p=0.54)"),
    ("L7", "full 4-term","z = b + Δ_rel + α·Δ_llm",  "λ_int=3e-3, λ_sp=1e-3, λ_al=3e-2",     "historical champion (≡ L0 at p=0.991)"),
    ("A1", "rel-only",   "z = b + Δ_rel",            "use_judge=0, α_max=0, λ_al=0",         "judge-off arch switch"),
    ("A2", "judge-only", "z = b + α·Δ_llm",          "Δ_rel_max=0",                          "rel-off arch switch"),
]


def load_cell_metrics(cell_id: str) -> dict[str, list[float]]:
    """Return per-metric list of length 5 (one per seed), or empty if missing."""
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


def paired_test(cell_vals: list[float], ref_vals: list[float]) -> str:
    if len(cell_vals) != 5 or len(ref_vals) != 5:
        return "—"
    diff = np.array(cell_vals) - np.array(ref_vals)
    if np.allclose(diff, 0):
        return "0.000 (n.s.)"
    res = ttest_rel(cell_vals, ref_vals)
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
    all_metrics: dict[str, dict[str, list[float]]] = {}
    for cell_id, *_ in CELLS:
        all_metrics[cell_id] = load_cell_metrics(cell_id)

    # Reference for paired-t: L0 cls-only (the *retained* canonical setting).
    # L7 (historical champion) is statistically indistinguishable from L0 at
    # 5 seeds → cls-only became the canonical, and L7 is now reported as a
    # negative ablation cell.
    ref_label = "L0"
    ref = all_metrics[ref_label]

    lines: list[str] = []
    lines.append("# YelpChi BWGNN — Phase 2 Loss × Architecture Ablation (5 seeds)\n")
    lines.append("**Base**: `fixed_v1_100ep` (BWGNN, paper-faithful, frozen)  ")
    lines.append("**Reference cell**: `L0` (cls-only, *retained* canonical)  ")
    lines.append(f"**Seeds**: {SEEDS}  ")
    lines.append("**Statistical test**: paired t-test (per-seed Δ vs L0), 5 seeds → df=4  ")
    lines.append("**Falsification ledger**: every L_i / L7 / judge cell fails the 5-seed paired-t bar vs L0; see AGENTS.md §§7.1, 9.\n")

    lines.append("## Cell legend (historical 4-term schema)\n")
    lines.append("| Cell | Variant | Historical formula | HP override | Note |")
    lines.append("|---|---|---|---|---|")
    for cell_id, var, formula, hp, note in CELLS:
        lines.append(f"| **{cell_id}** | {var} | `{formula}` | {hp} | {note} |")
    lines.append("")

    lines.append("## Results (mean ± std, 5 seeds)\n")
    head = "| Cell | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |"
    sep = "|---|" + "---:|" * len(METRICS)
    lines.append(head)
    lines.append(sep)
    for cell_id, *_ in CELLS:
        cells = [fmt_pm(all_metrics[cell_id][m]) for m in METRICS]
        marker = " **★**" if cell_id == ref_label else ""
        lines.append(f"| {cell_id}{marker} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Paired t-test vs L0 (cls-only canonical) — AUPRC\n")
    lines.append("| Cell | Variant | AUPRC paired Δ vs L0 |")
    lines.append("|---|---|---|")
    for cell_id, var, *_ in CELLS:
        if cell_id == ref_label:
            lines.append(f"| {ref_label} | {var} | reference |")
            continue
        sig = paired_test(all_metrics[cell_id]["auprc"], ref["auprc"])
        lines.append(f"| {cell_id} | {var} | {sig} |")
    lines.append("")

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

    lines.append("---\n")
    lines.append("**Significance**: `***` p<0.001, `**` p<0.01, `*` p<0.05, `n.s.` p≥0.05\n")

    OUT.write_text("\n".join(lines))
    print(f"[ok] wrote {OUT.relative_to(REPO)} ({len(lines)} lines)")
    print(f"[ok] wrote {csv_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
