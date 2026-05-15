"""Aggregate arbitrary Phase2 run names.

Example:
    python scripts/aggregate_phase2_custom_runs.py \
        --dataset yelpchi \
        --runs phase2_yelp_s1_trust_0p0 phase2_yelp_s1_trust_1p0em3 \
        --out-prefix artifacts/tables/yelpchi_phase2_sensitivity
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
DEFAULT_SEEDS = [42, 123, 456, 789, 2026]
METRICS = ["auprc", "roc_auc", "macro_f1", "g_means"]
DIAG_KEYS = [
    "mean_abs_delta_rel",
    "mean_alpha_llm",
    "mean_alpha_llm_accepted",
    "mean_alpha_llm_rejected",
    "max_abs_alpha_llm_rejected",
    "mean_gate_entropy",
    "mean_dominance_rho",
    "gate_weight_rel_0",
    "gate_weight_rel_1",
    "gate_weight_rel_2",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def mean_std(values: list[float]) -> tuple[float, float]:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return "-"
        return f"{float(value):.{digits}f}"
    return str(value) if value not in ("", None) else "-"


def collect(dataset: str, model: str, runs: list[str], seeds: list[int]) -> tuple[list[dict], list[dict]]:
    per_seed: list[dict] = []
    summary: list[dict] = []
    for run in runs:
        buckets: dict[str, list[float]] = {k: [] for k in METRICS + DIAG_KEYS}
        present = 0
        for seed in seeds:
            metrics_path = (
                PROJECT_ROOT / "artifacts" / "results" / dataset / model
                / run / f"seed_{seed}" / "stage3_metrics.json"
            )
            diag_path = (
                PROJECT_ROOT / "artifacts" / "logs" / dataset / model
                / run / f"seed_{seed}" / "phase2_diagnostics.json"
            )
            metrics = load_json(metrics_path)
            diag_payload = load_json(diag_path)
            diag = diag_payload.get("final_diagnostics", diag_payload)
            row = {
                "dataset": dataset,
                "model": model,
                "run_name": run,
                "seed": seed,
                "metrics_present": bool(metrics),
                "diagnostics_present": bool(diag),
            }
            if metrics:
                present += 1
            for key in METRICS:
                value = metrics.get(key, float("nan"))
                row[key] = value
                if isinstance(value, (int, float)):
                    buckets[key].append(float(value))
            for key in DIAG_KEYS:
                value = diag.get(key, float("nan"))
                row[f"diag_{key}"] = value
                if isinstance(value, (int, float)):
                    buckets[key].append(float(value))
            per_seed.append(row)

        ag = {
            "dataset": dataset,
            "model": model,
            "run_name": run,
            "n_seeds_present": present,
        }
        for key, values in buckets.items():
            mu, sd = mean_std(values)
            ag[f"{key}_mean"] = mu
            ag[f"{key}_std"] = sd
        summary.append(ag)
    return per_seed, summary


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(summary: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase2 Custom Run Summary",
        "",
        "| run_name | n | AUPRC | ROC-AUC | Macro-F1 | G-Means | mean_abs_delta_rel | alpha | gate_H | gate_0 | gate_1 | gate_2 | rejected_alpha_max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {run} | {n} | {ap} +/- {ap_s} | {auc} +/- {auc_s} | {mf1} +/- {mf1_s} | "
            "{gm} +/- {gm_s} | {dr} | {alpha} | {gh} | {g0} | {g1} | {g2} | {rej} |".format(
                run=row["run_name"],
                n=row["n_seeds_present"],
                ap=fmt(row.get("auprc_mean")), ap_s=fmt(row.get("auprc_std")),
                auc=fmt(row.get("roc_auc_mean")), auc_s=fmt(row.get("roc_auc_std")),
                mf1=fmt(row.get("macro_f1_mean")), mf1_s=fmt(row.get("macro_f1_std")),
                gm=fmt(row.get("g_means_mean")), gm_s=fmt(row.get("g_means_std")),
                dr=fmt(row.get("mean_abs_delta_rel_mean")),
                alpha=fmt(row.get("mean_alpha_llm_mean")),
                gh=fmt(row.get("mean_gate_entropy_mean")),
                g0=fmt(row.get("gate_weight_rel_0_mean")),
                g1=fmt(row.get("gate_weight_rel_1_mean")),
                g2=fmt(row.get("gate_weight_rel_2_mean")),
                rej=fmt(row.get("max_abs_alpha_llm_rejected_mean")),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["yelpchi", "amazon"])
    parser.add_argument("--model", default="bwgnn")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    per_seed, summary = collect(args.dataset, args.model, args.runs, args.seeds)
    prefix = PROJECT_ROOT / args.out_prefix
    write_csv(per_seed, prefix.with_name(prefix.name + "_per_seed.csv"))
    write_csv(summary, prefix.with_name(prefix.name + "_summary.csv"))
    write_md(summary, prefix.with_name(prefix.name + "_summary.md"))
    print(f"Aggregated {len(summary)} runs for {args.dataset}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
