#!/usr/bin/env python3
"""Aggregate 5-seed experiment results into mean±std tables.

Scans artifacts/ for test_metrics.json / cbr_flash_summary.json /
base_metrics.json files and produces CSV + Markdown tables grouped by
experiment, dataset, split, and model.

Usage:
    python scripts/experiments/aggregate_results.py
    python scripts/experiments/aggregate_results.py --experiment E1
    python scripts/experiments/aggregate_results.py --output artifacts/tables
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "tables"

METRICS_KEYS = ["roc_auc", "auprc", "macro_f1", "f1", "g_means"]
SEEDS = [42, 123, 456, 789, 2026]

# Pattern to extract dataset/model/stage/seed from checkpoint paths
# artifacts/checkpoints/{ds}/{model}/{stage}/seed_{s}/...
# artifacts/results/{ds}/{model}/{stage}/seed_{s}/...
CKPT_RE = re.compile(
    r"artifacts/(?:results|logs|checkpoints)/"
    r"(?P<dataset>\w+)/(?P<model>\w+)/(?P<stage>[^/]+)/seed_(?P<seed>\d+)"
)


def find_result_files(root: Path) -> list[Path]:
    """Find all result JSON files under artifacts/."""
    patterns = ["**/test_metrics.json", "**/base_metrics.json",
                "**/cbr_flash_summary.json", "**/base_training.json"]
    files = []
    for pat in patterns:
        files.extend(root.glob(pat))
    return sorted(files)


def extract_info(path: Path) -> dict | None:
    """Extract dataset, model, stage, seed from path."""
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        rel = path
    m = CKPT_RE.search(str(rel))
    if m is None:
        return None
    return {
        "dataset": m.group("dataset"),
        "model": m.group("model"),
        "stage": m.group("stage"),
        "seed": int(m.group("seed")),
        "path": str(path),
    }


def load_metrics(path: Path) -> dict:
    """Load metrics from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    # Different files have different structures
    if "test_metrics" in data:
        return data["test_metrics"]
    if isinstance(data, dict) and "roc_auc" in data:
        return data
    return data


def group_results(files: list[Path]) -> dict:
    """Group results by (dataset, model, stage)."""
    groups = defaultdict(list)
    for f in files:
        info = extract_info(f)
        if info is None:
            continue
        metrics = load_metrics(f)
        key = (info["dataset"], info["model"], info["stage"])
        groups[key].append({
            "seed": info["seed"],
            "metrics": metrics,
        })
    return groups


def compute_stats(values: list[float]) -> dict:
    """Compute mean, std, and count."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "count": len(arr),
    }


def format_cell(stats: dict) -> str:
    """Format mean±std as string."""
    if stats["count"] == 0:
        return "—"
    return f"{stats['mean']:.4f}±{stats['std']:.4f}"


def aggregate(groups: dict) -> list[dict]:
    """Aggregate grouped results into table rows."""
    rows = []
    for (ds, model, stage), entries in sorted(groups.items()):
        row = {"dataset": ds, "model": model, "stage": stage, "n_seeds": len(entries)}
        for mk in METRICS_KEYS:
            values = []
            for e in entries:
                v = e["metrics"].get(mk)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass
            stats = compute_stats(values)
            row[f"{mk}_mean"] = stats["mean"]
            row[f"{mk}_std"] = stats["std"]
            row[f"{mk}"] = format_cell(stats)
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    import csv
    fields = ["dataset", "model", "stage", "n_seeds"]
    for mk in METRICS_KEYS:
        fields.extend([mk, f"{mk}_mean", f"{mk}_std"])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_markdown(rows: list[dict], path: Path, title: str = "Results") -> None:
    """Write rows as Markdown table."""
    if not rows:
        return
    col_display = ["dataset", "model", "stage", "n_seeds"] + METRICS_KEYS
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(col_display) + " |\n")
        f.write("| " + " | ".join(["---"] * len(col_display)) + " |\n")
        for r in rows:
            vals = []
            for c in col_display:
                v = r.get(c, "—")
                vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--experiment", "-e", type=str, default=None,
                        help="Filter by experiment prefix (E1, E2, etc.)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--artifacts-dir", type=str, default=str(ARTIFACTS))
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    output = Path(args.output) if args.output else DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {artifacts} for results...")
    files = find_result_files(artifacts)
    print(f"Found {len(files)} result files")

    groups = group_results(files)
    print(f"Grouped into {len(groups)} (dataset, model, stage) combinations")

    rows = aggregate(groups)

    if args.experiment:
        tag = args.experiment
    else:
        tag = "all"

    csv_path = output / f"aggregated_{tag}.csv"
    md_path = output / f"aggregated_{tag}.md"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path, title=f"CoVER-FD Results ({tag})")

    print(f"\nCSV:  {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"\n{len(rows)} rows written")

    # Print summary table
    if rows:
        print("\n=== Summary ===")
        datasets = sorted(set(r["dataset"] for r in rows))
        for ds in datasets:
            print(f"\n--- {ds} ---")
            ds_rows = [r for r in rows if r["dataset"] == ds]
            for r in ds_rows:
                print(f"  {r['model']:8s} | {r['stage']:12s} | "
                      f"AUPRC={r.get('auprc', '—'):16s} | "
                      f"AUROC={r.get('roc_auc', '—'):16s} | "
                      f"MaF1={r.get('macro_f1', '—'):16s}")


if __name__ == "__main__":
    main()
