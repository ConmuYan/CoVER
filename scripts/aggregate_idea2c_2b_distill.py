"""Aggregate Idea-2C distillation with Idea-2B learned-extractor teachers.

Reads:
  artifacts/logs/{ds}/{base}/idea2c_distill_adapter_2b/seed_{s}/phase2_diagnostics.json

Writes:
  artifacts/tables/idea2c_2b_distill_full_benchmark.md
"""
from __future__ import annotations

import json
from pathlib import Path

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
SEEDS = [42, 123, 456, 789, 2026]
METRICS = ["auprc", "roc_auc", "macro_f1", "g_means"]
METRIC_LABEL = {
    "auprc": "AUPRC",
    "roc_auc": "AUROC",
    "macro_f1": "M-F1",
    "g_means": "G-Means",
}
RUN_NAME = "idea2c_distill_adapter_2b"


def sig_from_t(t: float) -> str:
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


def paired_t(diff: np.ndarray) -> tuple[float, float]:
    if diff.std(ddof=1) <= 1e-12:
        if abs(float(diff.mean())) <= 1e-12:
            return 0.0, 1.0
        return float("inf") if diff.mean() > 0 else float("-inf"), 0.0
    t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
    p = float(2 * (1 - stats.t.cdf(abs(t), df=len(diff) - 1)))
    return t, p


def load_diag(ds: str, base: str, seed: int) -> dict:
    path = Path(f"artifacts/logs/{ds}/{base}/{RUN_NAME}/seed_{seed}/phase2_diagnostics.json")
    if not path.exists():
        raise FileNotFoundError(path)
    return json.load(path.open())


def values(diags: list[dict], bucket: str, metric: str) -> np.ndarray:
    return np.array([float(d[bucket][metric]) for d in diags], dtype=float)


def fmt_mean_sd(arr: np.ndarray) -> str:
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}"


def main() -> int:
    lines: list[str] = []
    lines.append("# Idea-2C Distillation with Idea-2B Learned-Extractor Teachers")
    lines.append("")
    lines.append("**Run**: `idea2c_distill_adapter_2b`, 8 cells × 5 seeds, same seeds `[42, 123, 456, 789, 2026]`.")
    lines.append("")
    lines.append("**Teacher**: Idea-2B learned evidence extractor + CoVER-REL reasoner.")
    lines.append("**Student**: RelDistillAdapter (4132 params, score-blind, bounded residual).")
    lines.append("")
    lines.append("**Significance (df=4)**: |t|>2.78 → ★; |t|>4.60 → ★★; |t|>8.61 → ★★★.")
    lines.append("")

    teacher_wins = {m: 0 for m in METRICS}
    adapter_wins = {m: 0 for m in METRICS}
    ns_cells = {m: 0 for m in METRICS}

    lines.append("## AUPRC (primary)")
    lines.append("")
    lines.append("| Cell | Base-only | Distill-2B | 2B teacher | % 2B REL gain | Distill vs Teacher |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for ds, base in CELLS:
        diags = [load_diag(ds, base, seed) for seed in SEEDS]
        base_arr = values(diags, "test_metrics_base_only", "auprc")
        adapter_arr = values(diags, "test_metrics_adapter", "auprc")
        teacher_arr = values(diags, "test_metrics_teacher", "auprc")
        denom = teacher_arr.mean() - base_arr.mean()
        gain = ((adapter_arr.mean() - base_arr.mean()) / denom * 100.0) if abs(denom) > 1e-12 else float("nan")
        # Positive t means teacher > adapter, matching the old 2C table.
        t, _p = paired_t(teacher_arr - adapter_arr)
        sig = sig_from_t(t)
        if sig not in ("ns", "trend"):
            if t > 0:
                teacher_wins["auprc"] += 1
            else:
                adapter_wins["auprc"] += 1
        else:
            ns_cells["auprc"] += 1
        lines.append(
            f"| {ds}-{base} | {base_arr.mean():.4f} | {fmt_mean_sd(adapter_arr)} | "
            f"{fmt_mean_sd(teacher_arr)} | {gain:.1f}% | "
            f"Δ={adapter_arr.mean() - teacher_arr.mean():+.4f} {sig} (t={t:+.2f}) |"
        )
    lines.append("")

    lines.append("## All Metrics")
    lines.append("")
    lines.append("| Metric | Cell | Distill-2B | 2B teacher | Δ adapter-teacher | t(teacher-adapter) | sig |")
    lines.append("|---|---|---:|---:|---:|---:|:---:|")
    for metric in METRICS:
        if metric == "auprc":
            continue
        for ds, base in CELLS:
            diags = [load_diag(ds, base, seed) for seed in SEEDS]
            adapter_arr = values(diags, "test_metrics_adapter", metric)
            teacher_arr = values(diags, "test_metrics_teacher", metric)
            t, _p = paired_t(teacher_arr - adapter_arr)
            sig = sig_from_t(t)
            if sig not in ("ns", "trend"):
                if t > 0:
                    teacher_wins[metric] += 1
                else:
                    adapter_wins[metric] += 1
            else:
                ns_cells[metric] += 1
            lines.append(
                f"| {METRIC_LABEL[metric]} | {ds}-{base} | {fmt_mean_sd(adapter_arr)} | "
                f"{fmt_mean_sd(teacher_arr)} | {adapter_arr.mean() - teacher_arr.mean():+.4f} | "
                f"{t:+.2f} | {sig} |"
            )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | teacher sig wins | adapter sig wins | non-sig / trend |")
    lines.append("|---|---:|---:|---:|")
    for metric in METRICS:
        lines.append(
            f"| {METRIC_LABEL[metric]} | {teacher_wins[metric]} | {adapter_wins[metric]} | {ns_cells[metric]} |"
        )
    total_teacher = sum(teacher_wins.values())
    total_adapter = sum(adapter_wins.values())
    total_ns = sum(ns_cells.values())
    lines.append("")
    lines.append(
        f"**Headline**: across 32 cell × metric comparisons, teacher has {total_teacher} significant wins, "
        f"adapter has {total_adapter} significant wins, and {total_ns} are non-significant/trend."
    )
    lines.append("")

    out = Path("artifacts/tables/idea2c_2b_distill_full_benchmark.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
