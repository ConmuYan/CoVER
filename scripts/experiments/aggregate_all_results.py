#!/usr/bin/env python3
"""Aggregate experiment results across seeds into mean±std tables.

Scans artifacts/results/ for test_metrics.json / base_metrics.json files,
groups by (experiment, dataset, model, stage), and outputs CSV with 5-seed
mean±std for key metrics (AUPRC, AUROC, Macro-F1).

Usage:
    python scripts/experiments/aggregate_all_results.py
    python scripts/experiments/aggregate_all_results.py --experiment E1
    python scripts/experiments/aggregate_all_results.py --output artifacts/tables/all_results.csv
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def find_result_files(root: Path):
    """Find all result JSON files under root."""
    results = []
    for p in root.rglob("*.json"):
        if p.name in ("base_metrics.json", "test_metrics.json",
                       "cbr_flash_summary.json", "raer_teacher_metrics.json"):
            results.append(p)
    return results


def parse_path(path: Path):
    """Extract dataset, model, run_name, seed from path.

    Expected patterns:
      artifacts/results/{dataset}/{model}/{run_name}/seed_{s}/{metrics}.json
    """
    parts = path.parts
    # Find 'results' index
    try:
        ri = parts.index("results")
    except ValueError:
        return None

    if len(parts) < ri + 5:
        return None

    dataset = parts[ri + 1]
    model = parts[ri + 2]
    run_name = parts[ri + 3]
    seed_dir = parts[ri + 4]
    seed_match = re.match(r"seed_(\d+)", seed_dir)
    seed = int(seed_match.group(1)) if seed_match else -1

    stage = "unknown"
    if path.name == "base_metrics.json":
        stage = "base"
    elif path.name == "test_metrics.json":
        if "cbr_flash" in run_name:
            stage = "student"
        else:
            stage = "teacher"
    elif path.name == "raer_teacher_metrics.json":
        stage = "teacher"
    elif path.name == "cbr_flash_summary.json":
        stage = "student"

    return {
        "dataset": dataset,
        "model": model,
        "run_name": run_name,
        "seed": seed,
        "stage": stage,
        "path": path,
    }


def determine_experiment(run_name: str, dataset: str) -> str:
    """Infer experiment ID from run_name."""
    if "scarcity" in run_name:
        return "E2"
    if "neighbor_mb" in run_name or "lree_scalable" in run_name:
        if dataset in ("yelpnyc", "yelpzip"):
            return "E3"
        if dataset in ("tfinance", "tsocial"):
            return "E4"
    if "bwgnn_424" in run_name:
        if dataset in ("tfinance", "tsocial"):
            return "E4"
        return "E1"
    if "semi" in run_name:
        return "E1"
    if "raer_hc" in run_name:
        return "E6"
    if run_name in ("base", "raer_lree", "cbr_flash"):
        return "E1"
    if "cbr_flash_v" in run_name:
        return "E7"
    if "cbr_l" in run_name and "_K" in run_name:
        return "E8"
    return "E1"


METRIC_KEYS = ["auprc", "roc_auc", "macro_f1", "f1", "g_means",
               "precision", "recall"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="artifacts/results")
    parser.add_argument("--output", default="artifacts/tables/all_results.csv")
    parser.add_argument("--experiment", default=None,
                        help="Filter to specific experiment (E1-E8)")
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        print(f"No results found at {root}")
        return

    files = find_result_files(root)
    print(f"Found {len(files)} result files")

    # Group results
    groups = defaultdict(list)
    for f in files:
        info = parse_path(f)
        if info is None or info["seed"] < 0:
            continue
        try:
            with open(f) as fh:
                metrics = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        exp = determine_experiment(info["run_name"], info["dataset"])
        if args.experiment and exp != args.experiment:
            continue

        info["metrics"] = metrics
        info["experiment"] = exp
        key = (exp, info["dataset"], info["model"], info["stage"], info["run_name"])
        groups[key].append(info)

    # Aggregate
    rows = []
    for (exp, ds, model, stage, run_name), entries in sorted(groups.items()):
        seeds = sorted(entries, key=lambda x: x["seed"])
        row = {
            "experiment": exp,
            "dataset": ds,
            "model": model,
            "stage": stage,
            "run_name": run_name,
            "n_seeds": len(seeds),
            "seeds": ",".join(str(s["seed"]) for s in seeds),
        }
        for mk in METRIC_KEYS:
            vals = [s["metrics"].get(mk, float("nan")) for s in seeds]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                row[f"{mk}_mean"] = np.mean(vals)
                row[f"{mk}_std"] = np.std(vals) if len(vals) > 1 else 0.0
                row[f"{mk}_str"] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"
        rows.append(row)

    if not rows:
        print("No results to aggregate.")
        return

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")

    # Print summary table
    print(f"\n{'Exp':<4} {'Dataset':<10} {'Model':<8} {'Stage':<8} {'Seeds':<5} "
          f"{'AUPRC':<16} {'AUROC':<16} {'Macro-F1':<16}")
    print("-" * 95)
    for r in rows:
        print(f"{r['experiment']:<4} {r['dataset']:<10} {r['model']:<8} "
              f"{r['stage']:<8} {r['n_seeds']:<5} "
              f"{r.get('auprc_str', 'N/A'):<16} "
              f"{r.get('roc_auc_str', 'N/A'):<16} "
              f"{r.get('macro_f1_str', 'N/A'):<16}")


if __name__ == "__main__":
    main()
