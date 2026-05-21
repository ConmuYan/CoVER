"""Aggregate Idea-2B module ablation paired-t table.

Compares each 2B ablation switch vs 2B canonical (learned extractor, no switches)
on YelpChi × {BWGNN, SAGE, GCN, GAT} × 5 seeds.

Reads:
  artifacts/results/yelpchi/{base}/idea2b_learned_extractor/seed_{s}/stage3_metrics.json   (canonical)
  artifacts/results/yelpchi/{base}/idea2b_ablate_{switch}/seed_{s}/stage3_metrics.json     (ablation)

Writes:
  artifacts/tables/idea2b_ablation_4base_5seed.md
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

CELLS = [("yelpchi", b) for b in ["bwgnn", "sage", "gcn", "gat"]]
SEEDS = [42, 123, 456, 789, 2026]
METRICS = ["auprc", "roc_auc", "macro_f1", "g_means"]
MLABEL = {"auprc": "AUPRC", "roc_auc": "AUROC", "macro_f1": "M-F1", "g_means": "G-Means"}
SWITCHES = [
    ("drop_gcn",       "drop GCN encoder"),
    ("drop_proto",     "drop proto features"),
    ("encoder_shared", "shared encoder + one-hot"),
]


def _sig(t: float) -> str:
    a = abs(t)
    if a > 8.61: return "★★★"
    if a > 4.60: return "★★"
    if a > 2.78: return "★"
    if a > 2.13: return "trend"
    return "ns"


def _fetch(run_name: str, ds: str, base: str, seed: int, metric: str) -> float | None:
    f = Path(f"artifacts/results/{ds}/{base}/{run_name}/seed_{seed}/stage3_metrics.json")
    if not f.exists():
        return None
    return float(json.load(f.open())[metric])


def _paired_t(canon: np.ndarray, abl: np.ndarray) -> tuple[float, float]:
    diff = canon - abl  # positive = ablation hurts (canon better)
    if diff.std(ddof=1) <= 1e-12:
        return 0.0, 1.0
    t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
    p = float(2 * (1 - stats.t.cdf(abs(t), df=len(diff) - 1)))
    return t, p


def main() -> int:
    lines: list[str] = []
    lines.append("# Idea-2B Module Ablation — YelpChi × 4 bases × 3 switches × 5 seeds\n")
    lines.append("**Reference**: `idea2b_learned_extractor` (canonical 2B, no switches).")
    lines.append("**Ablation**: each row flips ONE switch, keeping the other two at canonical values.")
    lines.append("**Direction**: positive t = canonical wins (ablation hurts). Negative t = ablation helps.\n")
    lines.append("**Significance (df=4)**: |t|>2.78 → ★ (p<0.05); |t|>4.60 → ★★ (p<0.01); |t|>8.61 → ★★★ (p<0.001).\n")

    # ---- AUPRC table (primary) ----
    lines.append("## AUPRC (primary)\n")
    header = "| Switch (vs canonical) | Cell | canonical | ablated | Δ | t | sig |"
    sep    = "|---|---|---:|---:|---:|---:|:---:|"
    lines.append(header)
    lines.append(sep)
    sig_count = {"total": 0, "sig": 0, "hurt": 0, "help": 0}
    for switch, desc in SWITCHES:
        run_name = f"idea2b_ablate_{switch}"
        for ds, base in CELLS:
            canon = [_fetch("idea2b_learned_extractor", ds, base, s, "auprc") for s in SEEDS]
            abl   = [_fetch(run_name, ds, base, s, "auprc") for s in SEEDS]
            if any(v is None for v in canon + abl):
                lines.append(f"| {desc} | {ds}-{base} | MISSING | | | | |")
                continue
            c, a = np.array(canon), np.array(abl)
            t, p = _paired_t(c, a)
            s_tag = _sig(t)
            sig_count["total"] += 1
            if s_tag not in ("ns", "trend"):
                sig_count["sig"] += 1
                if t > 0: sig_count["hurt"] += 1
                else: sig_count["help"] += 1
            lines.append(
                f"| {desc} | {ds}-{base} | "
                f"{c.mean():.4f} ± {c.std(ddof=1):.4f} | "
                f"{a.mean():.4f} ± {a.std(ddof=1):.4f} | "
                f"{(a.mean()-c.mean()):+.4f} | {t:+.3f} | {s_tag} |"
            )
    lines.append("")

    # ---- Cross-metric summary ----
    lines.append("## Cross-metric summary (4 metrics × 4 cells = 16 tests per switch)\n")
    lines.append("| Switch | metric | sig hurt | sig help | ns | total |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|")
    for switch, desc in SWITCHES:
        run_name = f"idea2b_ablate_{switch}"
        for m in MLABEL:
            hurt = help_ = ns = 0
            for ds, base in CELLS:
                canon = [_fetch("idea2b_learned_extractor", ds, base, s, m) for s in SEEDS]
                abl   = [_fetch(run_name, ds, base, s, m) for s in SEEDS]
                if any(v is None for v in canon + abl):
                    continue
                c, a = np.array(canon), np.array(abl)
                t, _ = _paired_t(c, a)
                s_tag = _sig(t)
                if s_tag in ("ns", "trend"):
                    ns += 1
                elif t > 0:
                    hurt += 1
                else:
                    help_ += 1
            lines.append(f"| {desc} | {MLABEL[m]} | {hurt} | {help_} | {ns} | 4 |")
    lines.append("")

    # ---- Verdict ----
    lines.append("## Verdict\n")
    lines.append(f"- Total AUPRC tests: {sig_count['total']}")
    lines.append(f"- AUPRC stat-sig: {sig_count['sig']} (hurt={sig_count['hurt']}, help={sig_count['help']})")
    lines.append("")

    out = Path("artifacts/tables/idea2b_ablation_4base_5seed.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
