"""Aggregate Idea 2B (learned extractor) vs Idea 1 canonical (hand-crafted A/B/C)
into an 8-cell × 4-metric paired-t table.

Reads from:
  artifacts/results/{ds}/{base}/idea1_canonical_clsonly/seed_{s}/stage3_metrics.json
  artifacts/results/{ds}/{base}/idea2b_learned_extractor/seed_{s}/stage3_metrics.json

Writes:
  artifacts/tables/idea2b_learned_vs_canonical_8cell_5seed.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats

CELLS: list[tuple[str, str]] = [
    ("yelpchi", "bwgnn"),
    ("yelpchi", "sage"),
    ("yelpchi", "gcn"),
    ("yelpchi", "gat"),
    ("amazon", "bwgnn"),
    ("amazon", "sage"),
    ("amazon", "gcn"),
    ("amazon", "gat"),
]
SEEDS: list[int] = [42, 123, 456, 789, 2026]
METRICS: list[str] = ["auprc", "roc_auc", "macro_f1", "g_means"]
METRIC_LABEL: dict[str, str] = {
    "auprc": "AUPRC",
    "roc_auc": "AUROC",
    "macro_f1": "M-F1",
    "g_means": "G-Means",
}


def _sig(t: float) -> str:
    a = abs(t)
    if a > 8.61:
        return "★★★"
    if a > 4.60:
        return "★★"
    if a > 2.78:
        return "★"
    if a > 2.13:
        return "trend"
    return "ns"


def _fetch(run_name: str, ds: str, base: str, seed: int, metric: str) -> float | None:
    f = Path(f"artifacts/results/{ds}/{base}/{run_name}/seed_{seed}/stage3_metrics.json")
    if not f.exists():
        return None
    return float(json.load(f.open())[metric])


def _canonical_run_name(ds: str, base: str) -> str:
    """Canonical baseline run name varies: YelpChi-BWGNN uses bare name,
    others use the per-cell prefix from ablation campaign."""
    if ds == "yelpchi" and base == "bwgnn":
        return "idea1_canonical_clsonly"
    return f"idea1_{ds}_{base}_canonical_clsonly"


def _paired_t(canon: np.ndarray, learn: np.ndarray) -> tuple[float, float]:
    diff = learn - canon
    if diff.std(ddof=1) <= 1e-12:
        return float("inf"), 0.0
    t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
    p = float(2 * (1 - stats.t.cdf(abs(t), df=len(diff) - 1)))
    return t, p


def main() -> int:
    lines: list[str] = []
    lines.append("# Idea-2B Cross-Cell Verdict — 8 cells × 5 seeds (paired-t vs Idea-1 canonical)")
    lines.append("")
    lines.append("**Comparison**: ``idea2b_learned_extractor`` (online learned 9-dim per-relation evidence)")
    lines.append("vs ``idea1_canonical_clsonly`` (hand-crafted A/B/C 9-dim per-relation evidence).")
    lines.append("")
    lines.append("**Contracts preserved by 2B**: score-blind (no base logit access),")
    lines.append("train-only prototype (under ``torch.no_grad()`` over train_mask × train_labels==c),")
    lines.append("frozen base, bounded intervention (downstream tanh + δ_max).")
    lines.append("")
    lines.append("**Significance bar (df=4)**: |t|>2.78 → p<0.05 ★; |t|>4.60 → p<0.01 ★★; |t|>8.61 → p<0.001 ★★★.")
    lines.append("")

    # Per-metric tables
    sig_summary: dict[str, dict[str, int]] = {m: {"win_sig": 0, "win_ns": 0, "lose": 0, "missing": 0} for m in METRICS}
    for m in METRICS:
        lines.append(f"## {METRIC_LABEL[m]}")
        lines.append("")
        lines.append("| Cell             | canonical (mean ± sd) | 2B learned (mean ± sd) | Δ        |  t        | p       | sig |")
        lines.append("|------------------|----------------------:|-----------------------:|---------:|----------:|--------:|:---:|")
        for ds, base in CELLS:
            canon_run = _canonical_run_name(ds, base)
            canon_vals = [_fetch(canon_run, ds, base, s, m) for s in SEEDS]
            learn_vals = [_fetch("idea2b_learned_extractor", ds, base, s, m) for s in SEEDS]
            if any(v is None for v in canon_vals + learn_vals):
                missing = [
                    f"canon_seed_{s}" if c is None else (f"2b_seed_{s}" if l is None else None)
                    for s, c, l in zip(SEEDS, canon_vals, learn_vals)
                    if c is None or l is None
                ]
                lines.append(f"| {ds}-{base:<12s} | MISSING                | MISSING                |          |           |         | {','.join(filter(None, missing))} |")
                sig_summary[m]["missing"] += 1
                continue
            canon = np.array(canon_vals)
            learn = np.array(learn_vals)
            diff = learn.mean() - canon.mean()
            t, p = _paired_t(canon, learn)
            s_tag = _sig(t)
            if t > 0 and s_tag not in ("ns", "trend"):
                sig_summary[m]["win_sig"] += 1
            elif t > 0:
                sig_summary[m]["win_ns"] += 1
            else:
                sig_summary[m]["lose"] += 1
            cell_label = f"{ds}-{base}"
            lines.append(
                f"| {cell_label:<16s} | "
                f"{canon.mean():.4f} ± {canon.std(ddof=1):.4f}      | "
                f"{learn.mean():.4f} ± {learn.std(ddof=1):.4f}       | "
                f"{diff:+.4f}  | "
                f"{t:+.3f}   | "
                f"{p:.4f} | "
                f"{s_tag} |"
            )
        lines.append("")

    # Summary block
    lines.append("## Cross-cell sig summary")
    lines.append("")
    lines.append("| Metric     | wins (sig) | wins (ns) | losses | missing |")
    lines.append("|-----------|:----------:|:---------:|:------:|:-------:|")
    for m in METRICS:
        s = sig_summary[m]
        lines.append(f"| {METRIC_LABEL[m]:9s} |     {s['win_sig']:>2d}     |    {s['win_ns']:>2d}     |   {s['lose']:>2d}   |   {s['missing']:>2d}    |")
    lines.append("")

    # Total stat-sig wins across metric × cell
    total_runs = len(CELLS) * len(METRICS)
    total_sig_wins = sum(sig_summary[m]["win_sig"] for m in METRICS)
    total_ns_wins = sum(sig_summary[m]["win_ns"] for m in METRICS)
    total_losses = sum(sig_summary[m]["lose"] for m in METRICS)
    lines.append(f"**Headline**: 2B wins (stat-sig) {total_sig_wins} / {total_runs} (cell × metric) comparisons; ")
    lines.append(f"wins (ns positive Δ) {total_ns_wins}; loses {total_losses}.")
    lines.append("")

    out = Path("artifacts/tables/idea2b_learned_vs_canonical_8cell_5seed.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print()
    print("Headline:", f"2B wins (stat-sig) {total_sig_wins}/{total_runs}; wins (ns) {total_ns_wins}; loses {total_losses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
