"""Aggregate canonical CBR-Flash rerun results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("auprc", "roc_auc", "macro_f1", "g_means")


def load_metric(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {m: float(raw[m]) for m in METRICS if m in raw}


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate RAER-FD CBR-Flash results")
    p.add_argument("--datasets", nargs="+", default=["yelpchi", "amazon"])
    p.add_argument("--bases", nargs="+", default=["bwgnn", "sage", "gcn", "gat"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026])
    p.add_argument("--run_name", default="cbr_flash")
    p.add_argument("--out_dir", default="artifacts/tables")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for base in args.bases:
            per_seed: dict[str, list[float]] = {m: [] for m in METRICS}
            found = 0
            for seed in args.seeds:
                metric_path = (
                    Path("artifacts/results")
                    / dataset
                    / base
                    / args.run_name
                    / f"seed_{seed}"
                    / "test_metrics.json"
                )
                metrics = load_metric(metric_path)
                if metrics is None:
                    continue
                found += 1
                for metric, value in metrics.items():
                    per_seed[metric].append(value)
            row: dict[str, object] = {
                "dataset": dataset,
                "base": base,
                "run_name": args.run_name,
                "n_seeds": found,
            }
            for metric in METRICS:
                mean, std = mean_std(per_seed[metric])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cbr_flash_compact_fullgraph_5seed.csv"
    json_path = out_dir / "cbr_flash_compact_fullgraph_5seed.json"
    md_path = out_dir / "cbr_flash_compact_fullgraph_5seed.md"

    fields = ["dataset", "base", "run_name", "n_seeds"]
    for metric in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    lines = [
        "# CBR-Flash Results",
        "",
        "| Dataset | Base | Seeds | AUPRC | AUROC | Macro-F1 | G-Mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(metric: str) -> str:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            if not np.isfinite(mean):
                return "missing"
            return f"{mean:.4f} ± {std:.4f}"

        lines.append(
            f"| {row['dataset']} | {row['base']} | {row['n_seeds']} | "
            f"{fmt('auprc')} | {fmt('roc_auc')} | {fmt('macro_f1')} | {fmt('g_means')} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
