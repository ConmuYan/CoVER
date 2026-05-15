"""Aggregate Phase2 SAGE reasoner experiments and write CSV/MD/markdown report.

Inputs (per experiment, per seed):
  artifacts/results/{ds}/sage/phase2_E{i}_{name}/seed_{seed}/stage3_metrics.json
  artifacts/logs/{ds}/sage/phase2_E{i}_{name}/seed_{seed}/phase2_diagnostics.json

Outputs:
  artifacts/tables/phase2_sage_5seed_summary.csv
  artifacts/tables/phase2_sage_5seed_summary.md
  artifacts/reports/phase2_sage_first_round_conclusion.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ["yelpchi", "amazon"]
EXPERIMENTS = [
    ("E0", "phase2_E0_relgate"),
    ("E1", "phase2_E1_judge_align"),
    ("E2", "phase2_E2_judge_residual"),
]
DEFAULT_SEEDS = [42, 123, 456, 789, 2026]

# BWGNN reference baselines (from PROGRESS.md / 5-seed reports).
BWGNN_BASELINES = {
    ("yelpchi", "BWGNN base"): {"auprc": 0.4674, "roc_auc": 0.8076, "macro_f1": 0.6483, "g_means": 0.5105},
    ("yelpchi", "anchor_gate"): {"auprc": 0.4998, "roc_auc": 0.8177, "macro_f1": 0.6598, "g_means": 0.5253},
    ("yelpchi", "judge_strength_gate"): {"auprc": 0.5006, "roc_auc": 0.8180, "macro_f1": 0.6650, "g_means": 0.5359},
    ("amazon", "BWGNN base"): {"auprc": 0.8643, "roc_auc": 0.9747, "macro_f1": 0.9168, "g_means": 0.8815},
    ("amazon", "anchor_gate"): {"auprc": 0.8661, "roc_auc": 0.9752, "macro_f1": 0.9174, "g_means": 0.8832},
    ("amazon", "judge_strength_gate"): {"auprc": 0.8663, "roc_auc": 0.9751, "macro_f1": 0.9168, "g_means": 0.8831},
}

METRICS = ["auprc", "roc_auc", "macro_f1", "g_means"]
DIAG_KEYS = [
    "mean_abs_delta_rel",
    "mean_alpha_llm",
    "mean_alpha_llm_accepted",
    "mean_alpha_llm_rejected",
    "mean_gate_entropy",
    "mean_dominance_rho",
    "judge_align_count",
]


def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[warn] could not parse {p}: {exc}", file=sys.stderr)
        return None


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(float(x))]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def _load_sage_base_baseline(seeds: list[int]) -> dict[str, dict[str, float]]:
    """Compute SAGE base baseline from 5-seed stage1_metrics.json files."""
    baselines: dict[str, dict[str, float]] = {}
    for ds in DATASETS:
        metric_lists: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed in seeds:
            p = PROJECT_ROOT / "artifacts" / "results" / ds / "sage" / "base" / f"seed_{seed}" / "stage1_metrics.json"
            data = _load_json(p)
            if data is None:
                continue
            for m in METRICS:
                v = data.get(m)
                if isinstance(v, (int, float)):
                    metric_lists[m].append(float(v))
        agg: dict[str, float] = {}
        for m in METRICS:
            mean, _ = _mean_std(metric_lists[m])
            agg[m] = mean
        baselines[ds] = agg
    return baselines


def collect(seeds: list[int]) -> tuple[list[dict], list[dict]]:
    """Return (per_seed_rows, summary_rows)."""
    per_seed: list[dict] = []
    summary: list[dict] = []
    for ds in DATASETS:
        for exp_id, run_name in EXPERIMENTS:
            metric_lists: dict[str, list[float]] = {m: [] for m in METRICS}
            diag_lists: dict[str, list[float]] = {k: [] for k in DIAG_KEYS}
            seed_rejected_alpha_max: list[float] = []
            for seed in seeds:
                metrics_path = PROJECT_ROOT / "artifacts" / "results" / ds / "sage" / run_name / f"seed_{seed}" / "stage3_metrics.json"
                diag_path = PROJECT_ROOT / "artifacts" / "logs" / ds / "sage" / run_name / f"seed_{seed}" / "phase2_diagnostics.json"
                metrics = _load_json(metrics_path) or {}
                diag_payload = _load_json(diag_path) or {}
                diag = diag_payload.get("final_diagnostics", diag_payload)
                row = {
                    "dataset": ds,
                    "experiment": exp_id,
                    "run_name": run_name,
                    "seed": seed,
                    "missing_metrics": metrics_path.exists() is False,
                    "missing_diagnostics": diag_path.exists() is False,
                }
                for m in METRICS:
                    val = metrics.get(m, "")
                    row[m] = val
                    if isinstance(val, (int, float)):
                        metric_lists[m].append(float(val))
                for k in DIAG_KEYS:
                    val = diag.get(k, "")
                    row[f"diag_{k}"] = val
                    if isinstance(val, (int, float)):
                        diag_lists[k].append(float(val))
                rej = diag.get("max_abs_alpha_llm_rejected")
                if isinstance(rej, (int, float)):
                    seed_rejected_alpha_max.append(float(rej))
                    row["max_abs_alpha_llm_rejected"] = rej
                per_seed.append(row)
            agg = {
                "dataset": ds,
                "experiment": exp_id,
                "run_name": run_name,
                "n_seeds_present": sum(1 for s in seeds if (PROJECT_ROOT / "artifacts" / "results" / ds / "sage" / run_name / f"seed_{s}" / "stage3_metrics.json").exists()),
            }
            for m in METRICS:
                mean, std = _mean_std(metric_lists[m])
                agg[f"{m}_mean"] = mean
                agg[f"{m}_std"] = std
            for k in DIAG_KEYS:
                mean, std = _mean_std(diag_lists[k])
                agg[f"diag_{k}_mean"] = mean
                agg[f"diag_{k}_std"] = std
            agg["max_rejected_alpha_violation"] = max(seed_rejected_alpha_max) if seed_rejected_alpha_max else float("nan")
            summary.append(agg)
    return per_seed, summary


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v, digits: int = 4) -> str:
    if isinstance(v, (int, float)):
        if math.isnan(float(v)):
            return "—"
        return f"{v:.{digits}f}"
    return str(v) if v not in ("", None) else "—"


def write_md_summary(summary: list[dict], sage_baselines: dict[str, dict[str, float]], path: Path) -> None:
    lines: list[str] = ["# Phase2 SAGE 5-seed Summary", ""]
    for ds in DATASETS:
        lines.append(f"## {ds}")
        lines.append("| Experiment | run_name | n | AUPRC | ROC-AUC | Macro-F1 | G-Means | mean|Δrel| | mean α | gate H | reject α (max) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for ag in summary:
            if ag["dataset"] != ds:
                continue
            lines.append(
                "| {exp} | `{run}` | {n} | {auprc} ± {auprc_s} | {roc} ± {roc_s} | {mf1} ± {mf1_s} | {gm} ± {gm_s} | {dr} | {al} | {ge} | {rej} |".format(
                    exp=ag["experiment"],
                    run=ag["run_name"],
                    n=ag["n_seeds_present"],
                    auprc=fmt(ag["auprc_mean"]), auprc_s=fmt(ag["auprc_std"]),
                    roc=fmt(ag["roc_auc_mean"]), roc_s=fmt(ag["roc_auc_std"]),
                    mf1=fmt(ag["macro_f1_mean"]), mf1_s=fmt(ag["macro_f1_std"]),
                    gm=fmt(ag["g_means_mean"]), gm_s=fmt(ag["g_means_std"]),
                    dr=fmt(ag.get("diag_mean_abs_delta_rel_mean")),
                    al=fmt(ag.get("diag_mean_alpha_llm_mean")),
                    ge=fmt(ag.get("diag_mean_gate_entropy_mean")),
                    rej=fmt(ag.get("max_rejected_alpha_violation"), digits=2),
                )
            )
        lines.append("")
        # Baselines
        lines.append("### Baselines")
        sage_base = sage_baselines.get(ds, {})
        lines.append(
            f"- **SAGE base**: AUPRC {fmt(sage_base.get('auprc'))} | ROC-AUC {fmt(sage_base.get('roc_auc'))} | Macro-F1 {fmt(sage_base.get('macro_f1'))} | G-Means {fmt(sage_base.get('g_means'))}"
        )
        for label in ("BWGNN base", "anchor_gate", "judge_strength_gate"):
            row = BWGNN_BASELINES.get((ds, label), {})
            lines.append(
                f"- **{label} (BWGNN)**: AUPRC {fmt(row.get('auprc'))} | ROC-AUC {fmt(row.get('roc_auc'))} | Macro-F1 {fmt(row.get('macro_f1'))} | G-Means {fmt(row.get('g_means'))}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_conclusion_report(summary: list[dict], sage_baselines: dict[str, dict[str, float]], path: Path) -> None:
    """Five-section first-round conclusion for SAGE."""
    lookup = {(s["dataset"], s["experiment"]): s for s in summary}

    def lookup_metric(ds: str, exp: str, key: str) -> float:
        entry = lookup.get((ds, exp), {})
        v = entry.get(f"{key}_mean")
        return v if isinstance(v, (int, float)) else float("nan")

    lines: list[str] = []
    lines.append("# Phase2 SAGE First-Round Conclusion")
    lines.append("")
    lines.append("## 1. E0 relation-gate improvement over SAGE base")
    lines.append("")
    lines.append("| Dataset | E0 AUPRC (SAGE + relation gate) | SAGE base | Δ |")
    lines.append("|---|---|---|---|")
    for ds in DATASETS:
        new = lookup_metric(ds, "E0", "auprc")
        base = sage_baselines.get(ds, {}).get("auprc", float("nan"))
        delta = new - base if isinstance(new, (int, float)) and not math.isnan(new) and not math.isnan(base) else float("nan")
        lines.append(f"| {ds} | {fmt(new)} | {fmt(base)} | {fmt(delta)} |")
    lines.append("")
    lines.append("Pass criterion for proceeding to E1/E2: Δ > 0.")
    lines.append("")

    lines.append("## 2. Main result table (5-seed mean ± std)")
    lines.append("")
    for ds in DATASETS:
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Experiment | AUPRC | ROC-AUC | Macro-F1 | G-Means |")
        lines.append("|---|---|---|---|---|")
        for exp_id, _ in EXPERIMENTS:
            ent = lookup.get((ds, exp_id), {})
            lines.append(
                f"| {exp_id} | {fmt(ent.get('auprc_mean'))} ± {fmt(ent.get('auprc_std'))} | "
                f"{fmt(ent.get('roc_auc_mean'))} ± {fmt(ent.get('roc_auc_std'))} | "
                f"{fmt(ent.get('macro_f1_mean'))} ± {fmt(ent.get('macro_f1_std'))} | "
                f"{fmt(ent.get('g_means_mean'))} ± {fmt(ent.get('g_means_std'))} |"
            )
        lines.append("")

    lines.append("## 3. Comparison with BWGNN baselines")
    lines.append("")
    for ds in DATASETS:
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Method | AUPRC | ROC-AUC | Macro-F1 | G-Means |")
        lines.append("|---|---|---|---|---|")
        sage_base = sage_baselines.get(ds, {})
        lines.append(f"| SAGE base | {fmt(sage_base.get('auprc'))} | {fmt(sage_base.get('roc_auc'))} | {fmt(sage_base.get('macro_f1'))} | {fmt(sage_base.get('g_means'))} |")
        for label in ("BWGNN base", "anchor_gate", "judge_strength_gate"):
            row = BWGNN_BASELINES[(ds, label)]
            lines.append(f"| {label} (BWGNN) | {fmt(row['auprc'])} | {fmt(row['roc_auc'])} | {fmt(row['macro_f1'])} | {fmt(row['g_means'])} |")
        for exp_id, _ in EXPERIMENTS:
            ent = lookup.get((ds, exp_id), {})
            lines.append(
                f"| SAGE Phase2 {exp_id} | {fmt(ent.get('auprc_mean'))} | "
                f"{fmt(ent.get('roc_auc_mean'))} | "
                f"{fmt(ent.get('macro_f1_mean'))} | "
                f"{fmt(ent.get('g_means_mean'))} |"
            )
        lines.append("")

    lines.append("## 4. Diagnostics & safety checks")
    lines.append("")
    lines.append("| Dataset | Exp | mean|Δ_rel| | mean α | accepted α | rejected/missing α (max) | gate entropy | dominance ρ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for exp_id, _ in EXPERIMENTS:
            ent = lookup.get((ds, exp_id), {})
            lines.append(
                f"| {ds} | {exp_id} | {fmt(ent.get('diag_mean_abs_delta_rel_mean'))} | "
                f"{fmt(ent.get('diag_mean_alpha_llm_mean'))} | "
                f"{fmt(ent.get('diag_mean_alpha_llm_accepted_mean'))} | "
                f"{fmt(ent.get('max_rejected_alpha_violation'))} | "
                f"{fmt(ent.get('diag_mean_gate_entropy_mean'))} | "
                f"{fmt(ent.get('diag_mean_dominance_rho_mean'))} |"
            )
    lines.append("")
    lines.append("Safety assertion: rejected/missing α (max) must be 0 (≤ 1e-6). Any non-zero value indicates a leak in the LLM gate.")
    lines.append("")

    lines.append("## 5. Conclusion (auto-template — fill in once numbers land)")
    lines.append("")
    lines.append("- **E0 (relation gate only)**: see Section 1 — does relation evidence improve SAGE base?")
    lines.append("- **E1 (judge alignment)**: compare to E0 — does L_align improve gate alignment?")
    lines.append("- **E2 (judge residual)**: compare to E1 — does small α residual buy further AUPRC?")
    lines.append("- **Cross-model comparison**: compare SAGE E0/E1/E2 with BWGNN E0/E1/E2 — does the method generalize?")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    sage_baselines = _load_sage_base_baseline(args.seeds)
    per_seed, summary = collect(args.seeds)
    write_csv(per_seed, PROJECT_ROOT / "artifacts" / "tables" / "phase2_sage_5seed_per_seed.csv")
    write_csv(summary, PROJECT_ROOT / "artifacts" / "tables" / "phase2_sage_5seed_summary.csv")
    write_md_summary(summary, sage_baselines, PROJECT_ROOT / "artifacts" / "tables" / "phase2_sage_5seed_summary.md")
    write_conclusion_report(summary, sage_baselines, PROJECT_ROOT / "artifacts" / "reports" / "phase2_sage_first_round_conclusion.md")
    print(f"Aggregated {len(per_seed)} per-seed rows into {len(summary)} (ds × exp) summary entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
