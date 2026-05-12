from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    seed = args.seed or config["train"]["seed"]
    run_name = args.run_name or config.get("run", {}).get("run_name", "base")

    results_dir = Path("artifacts") / "results" / dataset_name / model_name / run_name / f"seed_{seed}"
    base_results_dir = Path("artifacts") / "results" / dataset_name / model_name / "base" / f"seed_{seed}"

    stage1_path = base_results_dir / "stage1_metrics.json"
    stage3_path = results_dir / "stage3_metrics.json"

    stage1 = load_metrics(stage1_path)
    stage3 = load_metrics(stage3_path)

    if stage1 is None:
        print(f"Stage 1 metrics not found at {stage1_path}")
        print("Run: python scripts/evaluate.py --config <config> --stage stage1")
        sys.exit(1)

    if stage3 is None:
        print(f"Stage 3 metrics not found at {stage3_path}")
        print("Run: python scripts/evaluate.py --config <config> --stage stage3")
        sys.exit(1)

    metrics = ["roc_auc", "auprc", "f1", "precision", "recall"]

    print(f"\n{'='*60}")
    print(f"Stage 1 vs Stage 3 Comparison")
    print(f"Dataset: {dataset_name} | Model: {model_name} | Seed: {seed}")
    print(f"{'='*60}")

    header = f"{'Metric':<20} {'Stage 1':>10} {'Stage 3':>10} {'Delta':>10}"
    print(header)
    print("-" * 60)

    for metric in metrics:
        v1 = stage1.get(metric, 0)
        v3 = stage3.get(metric, 0)
        delta = v3 - v1
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<20} {v1:>10.4f} {v3:>10.4f} {sign}{delta:>9.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
