"""Aggregate reasoner sweep results across seeds and recommend best configurations.

Reads sweep artifacts from artifacts/sweeps/{dataset}/{model}/ and produces
a summary table with per-(rho, lambda) aggregated metrics and recommendations.

Usage:
    python scripts/aggregate_sweep_results.py \
        --dataset yelpchi --model bwgnn --run_name qwen --seeds 123 456 789
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import ensure_dir, get_results_dir


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def compute_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, variance ** 0.5


def fmt_ms(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def parse_rho_lambda(run_name: str, base_name: str) -> tuple[float, float] | None:
    # pattern: {base_name}_rho{value}_lambda{value}
    pattern = re.escape(base_name) + r"_rho([0-9.eE+-]+)_lambda([0-9.eE+-]+)$"
    m = re.match(pattern, run_name)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None


def discover_sweep_configs(
    sweep_base: Path, run_name: str
) -> list[tuple[str, float, float]]:
    configs: list[tuple[str, float, float]] = []
    if not sweep_base.exists():
        return configs
    for child in sorted(sweep_base.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_rho_lambda(child.name, run_name)
        if parsed is not None:
            configs.append((child.name, parsed[0], parsed[1]))
    return configs


def load_seed_data(
    sweep_dir: Path, seed: int
) -> dict | None:
    seed_dir = sweep_dir / f"seed_{seed}"
    stage3 = load_json(seed_dir / "stage3_metrics.json")
    calibrated = load_json(seed_dir / "calibrated_metrics.json")
    diagnosis = load_json(seed_dir / "reasoner_diagnosis.json")

    if stage3 is None:
        return None

    base_metrics = None
    if diagnosis and "base_metrics" in diagnosis:
        base_metrics = diagnosis["base_metrics"]

    return {
        "stage3": stage3,
        "calibrated": calibrated or {},
        "diagnosis": diagnosis or {},
        "base_metrics": base_metrics,
    }


def load_seed_data_from_results(
    dataset: str, model: str, run_name_dir: str, seed: int, rho: float, lambda_evi: float
) -> dict | None:
    results_dir = get_results_dir(dataset, model, run_name_dir, seed)
    stage3 = load_json(results_dir / "stage3_metrics.json")
    calibrated = load_json(results_dir / "stage3_calibrated_metrics.json")

    if stage3 is None:
        return None

    base_results = get_results_dir(dataset, model, "base", seed)
    base_metrics_raw = load_json(base_results / "stage1_metrics.json")

    return {
        "stage3": stage3,
        "calibrated": calibrated or {},
        "diagnosis": {},
        "base_metrics": base_metrics_raw,
    }


def extract_metrics(seed_data: dict) -> dict:
    stage3 = seed_data["stage3"]
    calibrated = seed_data["calibrated"]
    base_metrics = seed_data.get("base_metrics") or {}

    # Handle nested format from run_reasoner_sweep.py (test_metrics key)
    test_metrics = stage3.get("test_metrics", stage3)

    result: dict = {
        "roc_auc": test_metrics.get("roc_auc", 0.0),
        "auprc": test_metrics.get("auprc", 0.0),
        "f1_fixed": test_metrics.get("f1", 0.0),
        "macro_f1_fixed": test_metrics.get("macro_f1", 0.0),
    }

    fixed_ppr = test_metrics.get("positive_prediction_rate", 0.0)
    result["positive_prediction_rate_fixed"] = fixed_ppr

    val_f1_data = calibrated.get("val_f1", {})
    if isinstance(val_f1_data, dict):
        test_f1 = val_f1_data.get("test", val_f1_data)
        result["f1_val_f1_threshold"] = test_f1.get("f1", 0.0)
        result["positive_prediction_rate_val_f1"] = test_f1.get(
            "positive_prediction_rate", 0.0
        )
    else:
        result["f1_val_f1_threshold"] = 0.0
        result["positive_prediction_rate_val_f1"] = 0.0

    val_macro_data = calibrated.get("val_macro_f1", {})
    if isinstance(val_macro_data, dict):
        test_macro = val_macro_data.get("test", val_macro_data)
        result["macro_f1_val_macro_threshold"] = test_macro.get("macro_f1", 0.0)
        result["positive_prediction_rate_val_macro"] = test_macro.get(
            "positive_prediction_rate", 0.0
        )
    else:
        result["macro_f1_val_macro_threshold"] = 0.0
        result["positive_prediction_rate_val_macro"] = 0.0

    # Handle nested format from run_reasoner_sweep.py (fixed.test key)
    if not result["f1_fixed"] and calibrated.get("fixed", {}).get("test"):
        fixed_test = calibrated["fixed"]["test"]
        result["f1_fixed"] = fixed_test.get("f1", 0.0)
        result["macro_f1_fixed"] = fixed_test.get("macro_f1", 0.0)
        result["roc_auc"] = fixed_test.get("roc_auc", result["roc_auc"])
        result["auprc"] = fixed_test.get("auprc", result["auprc"])
        result["positive_prediction_rate_fixed"] = fixed_test.get(
            "positive_prediction_rate", result["positive_prediction_rate_fixed"]
        )

    # evaluate.py saves metrics at top-level (val_f1_threshold_metrics), while
    # run_reasoner_sweep.py nests them under mode keys — handle both formats
    if not calibrated.get("val_f1") and calibrated.get("val_f1_threshold_metrics"):
        vtm = calibrated["val_f1_threshold_metrics"]
        result["f1_val_f1_threshold"] = vtm.get("f1", result["f1_val_f1_threshold"])
        result["positive_prediction_rate_val_f1"] = vtm.get(
            "positive_prediction_rate", result["positive_prediction_rate_val_f1"]
        )
    if not calibrated.get("val_macro_f1") and calibrated.get("val_macro_f1_threshold_metrics"):
        vmm = calibrated["val_macro_f1_threshold_metrics"]
        result["macro_f1_val_macro_threshold"] = vmm.get(
            "macro_f1", result["macro_f1_val_macro_threshold"]
        )
        result["positive_prediction_rate_val_macro"] = vmm.get(
            "positive_prediction_rate", result["positive_prediction_rate_val_macro"]
        )

    result["base_roc_auc"] = base_metrics.get("roc_auc", 0.0)
    result["base_auprc"] = base_metrics.get("auprc", 0.0)
    result["base_f1"] = base_metrics.get("f1", 0.0)
    result["base_macro_f1"] = base_metrics.get("macro_f1", 0.0)

    return result


METRIC_KEYS = [
    "roc_auc", "auprc", "f1_fixed", "macro_f1_fixed",
    "f1_val_f1_threshold", "macro_f1_val_macro_threshold",
    "positive_prediction_rate_fixed", "positive_prediction_rate_val_f1",
    "positive_prediction_rate_val_macro",
    "base_roc_auc", "base_auprc", "base_f1", "base_macro_f1",
]


def aggregate_across_seeds(seed_metrics: list[dict]) -> dict:
    aggregated: dict = {}
    for key in METRIC_KEYS:
        values = [s.get(key, 0.0) for s in seed_metrics]
        mean, std = compute_mean_std(values)
        aggregated[f"{key}_mean"] = mean
        aggregated[f"{key}_std"] = std
    return aggregated


def compute_recommendation(agg: dict) -> tuple[str, str]:
    roc_delta = agg["roc_auc_mean"] - agg["base_roc_auc_mean"]
    auprc_delta = agg["auprc_mean"] - agg["base_auprc_mean"]
    macro_f1_cal_delta = agg["macro_f1_val_macro_threshold_mean"] - agg["base_macro_f1_mean"]
    f1_fixed_delta = agg["f1_fixed_mean"] - agg["base_f1_mean"]
    macro_f1_fixed_delta = agg["macro_f1_fixed_mean"] - agg["base_macro_f1_mean"]

    delta_str = (
        f"ROC-AUC:{roc_delta:+.4f} AUPRC:{auprc_delta:+.4f} "
        f"F1_fixed:{f1_fixed_delta:+.4f} MacroF1_cal:{macro_f1_cal_delta:+.4f}"
    )

    # noise tolerance: 0.5*std or 0.002 minimum
    def tol(key: str) -> float:
        return max(agg.get(f"{key}_std", 0.0) * 0.5, 0.002)

    ranking_degraded = (
        roc_delta < -tol("roc_auc") or auprc_delta < -tol("auprc")
    )

    calibrated_good = macro_f1_cal_delta >= -tol("macro_f1_val_macro_threshold")
    fixed_bad = macro_f1_fixed_delta < -tol("macro_f1_fixed")

    if ranking_degraded:
        return "ranking_degradation", delta_str
    elif fixed_bad and calibrated_good:
        return "calibration_issue", delta_str
    elif calibrated_good and not ranking_degraded:
        return "yes", delta_str
    else:
        return "no", delta_str


def build_table_row(
    run_name: str,
    rho: float,
    lambda_evi: float,
    num_seeds: int,
    agg: dict,
) -> dict:
    recommended, delta_vs_base = compute_recommendation(agg)

    return {
        "run_name": run_name,
        "rho": rho,
        "lambda_evi": lambda_evi,
        "num_seeds": num_seeds,
        "roc_auc": fmt_ms(agg["roc_auc_mean"], agg["roc_auc_std"]),
        "auprc": fmt_ms(agg["auprc_mean"], agg["auprc_std"]),
        "f1_fixed": fmt_ms(agg["f1_fixed_mean"], agg["f1_fixed_std"]),
        "macro_f1_fixed": fmt_ms(agg["macro_f1_fixed_mean"], agg["macro_f1_fixed_std"]),
        "f1_val_f1_threshold": fmt_ms(
            agg["f1_val_f1_threshold_mean"], agg["f1_val_f1_threshold_std"]
        ),
        "macro_f1_val_macro_threshold": fmt_ms(
            agg["macro_f1_val_macro_threshold_mean"],
            agg["macro_f1_val_macro_threshold_std"],
        ),
        "positive_prediction_rate_fixed": f"{agg['positive_prediction_rate_fixed_mean']:.4f}",
        "positive_prediction_rate_calibrated": f"{agg['positive_prediction_rate_val_macro_mean']:.4f}",
        "delta_vs_base": delta_vs_base,
        "recommended": recommended,
    }


def build_csv_row(md_row: dict, agg: dict) -> dict:
    row = dict(md_row)
    for key in METRIC_KEYS:
        row[f"{key}_mean"] = agg[f"{key}_mean"]
        row[f"{key}_std"] = agg[f"{key}_std"]
    return row


CSV_COLUMNS = [
    "run_name", "rho", "lambda_evi", "num_seeds",
    "roc_auc_mean", "roc_auc_std",
    "auprc_mean", "auprc_std",
    "f1_fixed_mean", "f1_fixed_std",
    "macro_f1_fixed_mean", "macro_f1_fixed_std",
    "f1_val_f1_threshold_mean", "f1_val_f1_threshold_std",
    "macro_f1_val_macro_threshold_mean", "macro_f1_val_macro_threshold_std",
    "positive_prediction_rate_fixed_mean",
    "positive_prediction_rate_val_macro_mean",
    "delta_vs_base", "recommended",
]


def save_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(rows: list[dict], path: Path, dataset: str) -> None:
    ensure_dir(path.parent)
    lines = [
        f"# Reasoner Sweep Results — {dataset}",
        "",
        "| run_name | rho | lambda_evi | seeds | ROC-AUC | AUPRC | "
        "F1 (fixed) | Macro-F1 (fixed) | F1 (val_f1) | Macro-F1 (val_macro) | "
        "PPR (fixed) | PPR (calibrated) | delta_vs_base | recommended |",
        "|----------|-----|------------|-------|---------|-------|"
        "-----------|-----------------|-------------|---------------------|"
        "------------|------------------|---------------|-------------|",
    ]

    for r in rows:
        lines.append(
            f"| {r['run_name']} | {r['rho']} | {r['lambda_evi']} | "
            f"{r['num_seeds']} | {r['roc_auc']} | {r['auprc']} | "
            f"{r['f1_fixed']} | {r['macro_f1_fixed']} | "
            f"{r['f1_val_f1_threshold']} | {r['macro_f1_val_macro_threshold']} | "
            f"{r['positive_prediction_rate_fixed']} | "
            f"{r['positive_prediction_rate_calibrated']} | "
            f"{r['delta_vs_base']} | {r['recommended']} |"
        )

    rec_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        rec_counts[r["recommended"]] += 1

    lines.append("")
    lines.append("## Recommendation Summary")
    lines.append("")
    for rec, count in sorted(rec_counts.items()):
        lines.append(f"- **{rec}**: {count} configuration(s)")
    lines.append("")

    yes_rows = [r for r in rows if r["recommended"] == "yes"]
    if yes_rows:
        lines.append("## Recommended Configurations")
        lines.append("")
        for r in yes_rows:
            lines.append(
                f"- rho={r['rho']}, lambda={r['lambda_evi']}: "
                f"ROC-AUC={r['roc_auc']}, AUPRC={r['auprc']}, "
                f"Macro-F1(cal)={r['macro_f1_val_macro_threshold']}"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate reasoner sweep results and recommend best configs."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. yelpchi)")
    parser.add_argument("--model", required=True, help="Model name (e.g. bwgnn)")
    parser.add_argument("--run_name", required=True, help="Base run name prefix (e.g. qwen)")
    parser.add_argument(
        "--seeds", nargs="+", type=int, required=True, help="Seeds to aggregate"
    )
    args = parser.parse_args()

    sweep_base = Path("artifacts") / "sweeps" / args.dataset / args.model

    configs = discover_sweep_configs(sweep_base, args.run_name)

    if not configs:
        print(f"No sweep results found under {sweep_base}")
        print("Attempting to read from artifacts/results/ as fallback...")
        results_base = Path("artifacts") / "results" / args.dataset / args.model
        if results_base.exists():
            for child in sorted(results_base.iterdir()):
                if not child.is_dir():
                    continue
                parsed = parse_rho_lambda(child.name, args.run_name)
                if parsed is not None:
                    configs.append((child.name, parsed[0], parsed[1]))

    if not configs:
        print(f"ERROR: No sweep configurations found for {args.run_name} under {sweep_base}")
        print("Run run_reasoner_sweep.py first to generate sweep results.")
        sys.exit(1)

    print(f"Found {len(configs)} configuration(s) for {args.run_name}:")
    for dirname, rho, lam in configs:
        print(f"  {dirname}  (rho={rho}, lambda={lam})")

    md_rows: list[dict] = []
    csv_rows: list[dict] = []

    for dirname, rho, lambda_evi in configs:
        seed_metrics: list[dict] = []

        for seed in args.seeds:
            sweep_dir = sweep_base / dirname
            data = load_seed_data(sweep_dir, seed)

            if data is None:
                data = load_seed_data_from_results(
                    args.dataset, args.model, dirname, seed, rho, lambda_evi
                )

            if data is not None:
                seed_metrics.append(extract_metrics(data))

        if not seed_metrics:
            print(f"  WARNING: No seed data for {dirname}")
            continue

        agg = aggregate_across_seeds(seed_metrics)
        md_row = build_table_row(args.run_name, rho, lambda_evi, len(seed_metrics), agg)
        csv_row = build_csv_row(md_row, agg)
        md_rows.append(md_row)
        csv_rows.append(csv_row)

        rec_label = md_row["recommended"]
        print(
            f"  {dirname}: {len(seed_metrics)} seeds, "
            f"ROC-AUC={agg['roc_auc_mean']:.4f} "
            f"recommended={rec_label}"
        )

    if not md_rows:
        print("ERROR: No data aggregated. Check your seeds and run_name.")
        sys.exit(1)

    table_dir = ensure_dir(Path("artifacts") / "tables")
    csv_path = table_dir / f"reasoner_sweep_{args.dataset}.csv"
    md_path = table_dir / f"reasoner_sweep_{args.dataset}.md"

    save_csv(csv_rows, csv_path)
    save_markdown(md_rows, md_path, args.dataset)

    print(f"\n{'=' * 60}")
    print(f"Sweep aggregation complete for {args.dataset}/{args.model}")
    print(f"{'=' * 60}")
    print(f"\nCSV: {csv_path}")
    print(f"Markdown: {md_path}")

    rec_counts: dict[str, int] = defaultdict(int)
    for r in md_rows:
        rec_counts[r["recommended"]] += 1
    print(f"\nRecommendations:")
    for rec, count in sorted(rec_counts.items()):
        print(f"  {rec}: {count}")


if __name__ == "__main__":
    main()
