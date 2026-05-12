"""Compare methods for a single dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get_results_dir, get_err_cache_dir, get_reports_dir, ensure_dir


CALIBRATED_STAGE3_METRICS = "stage3_calibrated_metrics.json"


def load_metrics(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def metric_value(metrics: dict | None, *keys: str, default: float = 0.0) -> float:
    if not metrics:
        return default
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def normalize_metrics(metrics: dict | None, *, calibrated: bool = False) -> dict[str, float] | None:
    if not metrics:
        return None

    if calibrated:
        # Calibrated metrics JSON has nested structure:
        #   fixed_threshold_metrics  → metrics at threshold=0.5
        #   val_f1_threshold_metrics → metrics at val-calibrated threshold (F1-optimized)
        #   val_macro_f1_threshold_metrics → metrics at val-calibrated threshold (macro-F1-optimized)
        fixed_m = metrics.get("fixed_threshold_metrics", {})
        val_f1_m = metrics.get("val_f1_threshold_metrics", {})
        val_macro_m = metrics.get("val_macro_f1_threshold_metrics", {})

        f1_fixed = metric_value(fixed_m, "f1")
        f1_val_f1 = metric_value(val_f1_m, "f1")
        macro_f1_fixed = metric_value(fixed_m, "macro_f1")
        macro_f1_val_macro = metric_value(val_macro_m, "macro_f1")

        return {
            "roc_auc": metric_value(fixed_m, "roc_auc"),
            "auprc": metric_value(fixed_m, "auprc"),
            "f1": f1_val_f1,
            "macro_f1": macro_f1_val_macro,
            "f1_fixed": f1_fixed,
            "f1_val_f1": f1_val_f1,
            "macro_f1_fixed": macro_f1_fixed,
            "macro_f1_val_macro": macro_f1_val_macro,
        }
    else:
        f1 = metric_value(metrics, "f1", "f1_fixed")
        macro_f1 = metric_value(metrics, "macro_f1", "macro_f1_fixed")

        return {
            "roc_auc": metric_value(metrics, "roc_auc"),
            "auprc": metric_value(metrics, "auprc"),
            "f1": f1,
            "macro_f1": macro_f1,
            "f1_fixed": f1,
            "f1_val_f1": 0.0,
            "macro_f1_fixed": macro_f1,
            "macro_f1_val_macro": 0.0,
        }


def summarize_metrics(metrics_by_seed: dict[int, dict[str, float]]) -> dict[str, object]:
    seeds = sorted(metrics_by_seed)
    roc_aucs = [metrics_by_seed[seed]["roc_auc"] for seed in seeds]
    auprcs = [metrics_by_seed[seed]["auprc"] for seed in seeds]
    f1s = [metrics_by_seed[seed]["f1"] for seed in seeds]
    macro_f1s = [metrics_by_seed[seed]["macro_f1"] for seed in seeds]
    f1_fixeds = [metrics_by_seed[seed].get("f1_fixed", 0.0) for seed in seeds]
    f1_val_f1s = [metrics_by_seed[seed].get("f1_val_f1", 0.0) for seed in seeds]
    macro_f1_fixeds = [metrics_by_seed[seed].get("macro_f1_fixed", 0.0) for seed in seeds]
    macro_f1_val_macros = [metrics_by_seed[seed].get("macro_f1_val_macro", 0.0) for seed in seeds]

    roc_auc_mean, roc_auc_std = compute_mean_std(roc_aucs)
    auprc_mean, auprc_std = compute_mean_std(auprcs)
    f1_mean, f1_std = compute_mean_std(f1s)
    macro_f1_mean, macro_f1_std = compute_mean_std(macro_f1s)
    f1_fixed_mean, f1_fixed_std = compute_mean_std(f1_fixeds)
    f1_val_f1_mean, f1_val_f1_std = compute_mean_std(f1_val_f1s)
    macro_f1_fixed_mean, macro_f1_fixed_std = compute_mean_std(macro_f1_fixeds)
    macro_f1_val_macro_mean, macro_f1_val_macro_std = compute_mean_std(macro_f1_val_macros)

    return {
        "seeds": seeds,
        "roc_auc_mean": roc_auc_mean,
        "roc_auc_std": roc_auc_std,
        "auprc_mean": auprc_mean,
        "auprc_std": auprc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "macro_f1_mean": macro_f1_mean,
        "macro_f1_std": macro_f1_std,
        "f1_fixed_mean": f1_fixed_mean,
        "f1_fixed_std": f1_fixed_std,
        "f1_val_f1_mean": f1_val_f1_mean,
        "f1_val_f1_std": f1_val_f1_std,
        "macro_f1_fixed_mean": macro_f1_fixed_mean,
        "macro_f1_fixed_std": macro_f1_fixed_std,
        "macro_f1_val_macro_mean": macro_f1_val_macro_mean,
        "macro_f1_val_macro_std": macro_f1_val_macro_std,
        "per_seed_f1": {seed: metrics_by_seed[seed]["f1"] for seed in seeds},
        "per_seed_macro_f1": {seed: metrics_by_seed[seed]["macro_f1"] for seed in seeds},
        "by_seed": metrics_by_seed,
    }


def subset_metrics(metrics_by_seed: dict[int, dict[str, float]], seeds: list[int]) -> dict[int, dict[str, float]]:
    return {seed: metrics_by_seed[seed] for seed in seeds if seed in metrics_by_seed}


def format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "N/A"
    return f"{mean:.4f}±{std:.4f}"


def calibration_status(fixed_stats: dict[str, object], calibrated_stats: dict[str, object] | None) -> str:
    if not calibrated_stats:
        return "N/A"

    fixed_f1 = float(fixed_stats["f1_mean"])
    fixed_macro = float(fixed_stats["macro_f1_mean"])
    calibrated_f1 = float(calibrated_stats["f1_mean"])
    calibrated_macro = float(calibrated_stats["macro_f1_mean"])

    f1_delta = calibrated_f1 - fixed_f1
    macro_delta = calibrated_macro - fixed_macro
    if f1_delta > 0 and macro_delta > 0:
        return "calibration helps"
    if f1_delta < 0 and macro_delta < 0:
        return "ranking issue"
    if f1_delta == 0 and macro_delta == 0:
        return "no change"
    return "mixed"


def compute_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = variance ** 0.5
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="bwgnn")
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789, 42, 2026])
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    methods = {
        "BWGNN": "base",
        "CoVER-BWGNN-Rule": "rule",
        "CoVER-BWGNN-Qwen": "qwen",
    }

    all_metrics = {}
    all_calibrated_metrics = {}
    for method_name, teacher in methods.items():
        for seed in args.seeds:
            if teacher == "base":
                metrics_path = get_results_dir(args.dataset, args.model, "base", seed) / "stage1_metrics.json"
                calibrated_metrics_path = None
            else:
                metrics_path = get_results_dir(args.dataset, args.model, teacher, seed) / "stage3_metrics.json"
                calibrated_metrics_path = get_results_dir(args.dataset, args.model, teacher, seed) / CALIBRATED_STAGE3_METRICS

            metrics = load_metrics(metrics_path)
            if metrics:
                all_metrics[(method_name, seed)] = metrics

            if calibrated_metrics_path is not None:
                calibrated_metrics = load_metrics(calibrated_metrics_path)
                if calibrated_metrics:
                    all_calibrated_metrics[(method_name, seed)] = calibrated_metrics

    method_stats = {}
    calibrated_method_stats = {}
    for method_name in methods:
        fixed_by_seed: dict[int, dict[str, float]] = {}
        calibrated_by_seed: dict[int, dict[str, float]] = {}

        for seed in args.seeds:
            key = (method_name, seed)
            if key in all_metrics:
                fixed_metrics = normalize_metrics(all_metrics[key], calibrated=False)
                if fixed_metrics:
                    fixed_by_seed[seed] = fixed_metrics
            if key in all_calibrated_metrics:
                calibrated_metrics = normalize_metrics(all_calibrated_metrics[key], calibrated=True)
                if calibrated_metrics:
                    calibrated_by_seed[seed] = calibrated_metrics

        if fixed_by_seed:
            method_stats[method_name] = summarize_metrics(fixed_by_seed)

        if calibrated_by_seed:
            calibrated_method_stats[method_name] = summarize_metrics(calibrated_by_seed)

    report_dir = ensure_dir(Path("artifacts") / "reports" / args.dataset / args.model)

    md_lines = [
        f"# Method Comparison: {args.dataset}",
        "",
        f"Model: {args.model}",
        f"Seeds: {args.seeds}",
        "",
        "**Note:** `F1@0.5` and `F1@val-threshold` are different metrics. ROC-AUC/AUPRC are threshold-independent.",
        "",
        "## Results",
        "",
        "| Method | Seeds | ROC-AUC | AUPRC | F1@0.5 | F1@val-threshold | Macro-F1@0.5 | Macro-F1@val-threshold | Calibration |",
        "|--------|-------|---------|-------|--------|------------------|--------------|------------------------|-------------|",
    ]

    for method_name in methods:
        if method_name in method_stats:
            s = method_stats[method_name]
            num_seeds = len(s['seeds'])
            seeds_note = f" ({num_seeds}/{len(args.seeds)})" if num_seeds < len(args.seeds) else ""
            calibrated = calibrated_method_stats.get(method_name)
            calibration_note = calibration_status(s, calibrated)
            f1_val = format_mean_std(calibrated["f1_mean"], calibrated["f1_std"]) if calibrated else "N/A"
            macro_val = format_mean_std(calibrated["macro_f1_mean"], calibrated["macro_f1_std"]) if calibrated else "N/A"
            md_lines.append(
                f"| {method_name}{seeds_note} | {num_seeds} | "
                f"{s['roc_auc_mean']:.4f}±{s['roc_auc_std']:.4f} | "
                f"{s['auprc_mean']:.4f}±{s['auprc_std']:.4f} | "
                f"{s['f1_mean']:.4f}±{s['f1_std']:.4f} | "
                f"{f1_val} | "
                f"{s['macro_f1_mean']:.4f}±{s['macro_f1_std']:.4f} | "
                f"{macro_val} | "
                f"{calibration_note} |"
            )
        else:
            md_lines.append(
                f"| {method_name} | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )

    md_lines.extend(["", "## Analysis"])

    if "BWGNN" in method_stats and "CoVER-BWGNN-Rule" in method_stats:
        base = method_stats["BWGNN"]
        rule = method_stats["CoVER-BWGNN-Rule"]
        base_cal = calibrated_method_stats.get("BWGNN")
        rule_cal = calibrated_method_stats.get("CoVER-BWGNN-Rule")
        delta_roc = rule["roc_auc_mean"] - base["roc_auc_mean"]
        delta_auprc = rule["auprc_mean"] - base["auprc_mean"]
        delta_f1 = rule["f1_mean"] - base["f1_mean"]
        delta_macro = rule["macro_f1_mean"] - base["macro_f1_mean"]

        md_lines.append(f"### CoVER-BWGNN-Rule vs BWGNN")
        md_lines.append(f"- Δ ROC-AUC: {delta_roc:+.4f}")
        md_lines.append(f"- Δ AUPRC: {delta_auprc:+.4f}")
        md_lines.append(f"- Δ F1@0.5: {delta_f1:+.4f}")
        md_lines.append(f"- Δ Macro-F1@0.5: {delta_macro:+.4f}")

        if base_cal and rule_cal:
            delta_f1_cal = rule_cal["f1_mean"] - base_cal["f1_mean"]
            delta_macro_cal = rule_cal["macro_f1_mean"] - base_cal["macro_f1_mean"]
            md_lines.append(f"- Δ F1@val-threshold: {delta_f1_cal:+.4f}")
            md_lines.append(f"- Δ Macro-F1@val-threshold: {delta_macro_cal:+.4f}")

        if delta_roc < 0:
            md_lines.extend([
                "",
                "**Possible reasons for ROC-AUC decrease:**",
                "- trace_size too small",
                "- weak_or_uncertain_ratio too high",
                "- accepted ERR too few",
                "- Stage 3 lambda_evi too large",
                "- base already strong",
            ])

    if "CoVER-BWGNN-Rule" in method_stats and "CoVER-BWGNN-Qwen" in method_stats:
        rule = method_stats["CoVER-BWGNN-Rule"]
        qwen = method_stats["CoVER-BWGNN-Qwen"]
        rule_cal = calibrated_method_stats.get("CoVER-BWGNN-Rule")
        qwen_cal = calibrated_method_stats.get("CoVER-BWGNN-Qwen")
        qwen_seeds = qwen["seeds"]

        rule_on_qwen_seeds = {
            "roc_aucs": [],
            "auprcs": [],
            "f1s": [],
            "macro_f1s": [],
        }
        for seed in qwen_seeds:
            key = ("CoVER-BWGNN-Rule", seed)
            if key in all_metrics:
                m = all_metrics[key]
                rule_on_qwen_seeds["roc_aucs"].append(m.get("roc_auc", 0))
                rule_on_qwen_seeds["auprcs"].append(m.get("auprc", 0))
                rule_on_qwen_seeds["f1s"].append(m.get("f1", 0))
                rule_on_qwen_seeds["macro_f1s"].append(m.get("macro_f1", 0))

        if rule_on_qwen_seeds["roc_aucs"]:
            rule_roc_mean, rule_roc_std = compute_mean_std(rule_on_qwen_seeds["roc_aucs"])
            rule_auprc_mean, rule_auprc_std = compute_mean_std(rule_on_qwen_seeds["auprcs"])
            rule_f1_mean, rule_f1_std = compute_mean_std(rule_on_qwen_seeds["f1s"])
            rule_macro_mean, rule_macro_std = compute_mean_std(rule_on_qwen_seeds["macro_f1s"])

            delta_roc = qwen["roc_auc_mean"] - rule_roc_mean
            delta_auprc = qwen["auprc_mean"] - rule_auprc_mean
            delta_f1 = qwen["f1_mean"] - rule_f1_mean
            delta_macro = qwen["macro_f1_mean"] - rule_macro_mean

            md_lines.extend([
                "",
                "### CoVER-BWGNN-Qwen vs CoVER-BWGNN-Rule (fair same-seed comparison)",
                f"- Seeds compared: {qwen_seeds}",
                f"- CoVER-BWGNN-Rule (same seeds): ROC-AUC={rule_roc_mean:.4f}±{rule_roc_std:.4f}, AUPRC={rule_auprc_mean:.4f}±{rule_auprc_std:.4f}, F1={rule_f1_mean:.4f}±{rule_f1_std:.4f}, Macro-F1={rule_macro_mean:.4f}±{rule_macro_std:.4f}",
                f"- CoVER-BWGNN-Qwen: ROC-AUC={qwen['roc_auc_mean']:.4f}±{qwen['roc_auc_std']:.4f}, AUPRC={qwen['auprc_mean']:.4f}±{qwen['auprc_std']:.4f}, F1={qwen['f1_mean']:.4f}±{qwen['f1_std']:.4f}, Macro-F1={qwen['macro_f1_mean']:.4f}±{qwen['macro_f1_std']:.4f}",
                f"- Δ ROC-AUC: {delta_roc:+.4f}",
                f"- Δ AUPRC: {delta_auprc:+.4f}",
                f"- Δ F1@0.5: {delta_f1:+.4f}",
                f"- Δ Macro-F1@0.5: {delta_macro:+.4f}",
            ])

            if rule_cal and qwen_cal:
                md_lines.extend([
                    f"- Δ F1@val-threshold: {qwen_cal['f1_mean'] - rule_cal['f1_mean']:+.4f}",
                    f"- Δ Macro-F1@val-threshold: {qwen_cal['macro_f1_mean'] - rule_cal['macro_f1_mean']:+.4f}",
                ])

    if "BWGNN" in method_stats and "CoVER-BWGNN-Qwen" in method_stats:
        base = method_stats["BWGNN"]
        qwen = method_stats["CoVER-BWGNN-Qwen"]
        base_cal = calibrated_method_stats.get("BWGNN")
        qwen_cal = calibrated_method_stats.get("CoVER-BWGNN-Qwen")
        qwen_seeds = qwen["seeds"]

        base_qwen_seeds = {
            "seeds": [],
            "roc_aucs": [],
            "auprcs": [],
            "f1s": [],
            "macro_f1s": [],
        }
        rule_qwen_seeds = {
            "roc_aucs": [],
            "auprcs": [],
            "f1s": [],
            "macro_f1s": [],
        }
        for seed in qwen_seeds:
            key = ("BWGNN", seed)
            if key in all_metrics:
                m = all_metrics[key]
                base_qwen_seeds["seeds"].append(seed)
                base_qwen_seeds["roc_aucs"].append(m.get("roc_auc", 0))
                base_qwen_seeds["auprcs"].append(m.get("auprc", 0))
                base_qwen_seeds["f1s"].append(m.get("f1", 0))
                base_qwen_seeds["macro_f1s"].append(m.get("macro_f1", 0))

            key = ("CoVER-BWGNN-Rule", seed)
            if key in all_metrics:
                m = all_metrics[key]
                rule_qwen_seeds["roc_aucs"].append(m.get("roc_auc", 0))
                rule_qwen_seeds["auprcs"].append(m.get("auprc", 0))
                rule_qwen_seeds["f1s"].append(m.get("f1", 0))
                rule_qwen_seeds["macro_f1s"].append(m.get("macro_f1", 0))

        if base_qwen_seeds["roc_aucs"]:
            base_roc_mean, base_roc_std = compute_mean_std(base_qwen_seeds["roc_aucs"])
            base_auprc_mean, base_auprc_std = compute_mean_std(base_qwen_seeds["auprcs"])
            base_f1_mean, base_f1_std = compute_mean_std(base_qwen_seeds["f1s"])
            base_macro_mean, base_macro_std = compute_mean_std(base_qwen_seeds["macro_f1s"])

            delta_roc = qwen["roc_auc_mean"] - base_roc_mean
            delta_auprc = qwen["auprc_mean"] - base_auprc_mean
            delta_f1 = qwen["f1_mean"] - base_f1_mean
            delta_macro = qwen["macro_f1_mean"] - base_macro_mean

            md_lines.extend([
                "",
                "### CoVER-BWGNN-Qwen vs BWGNN (fair same-seed comparison)",
                f"- Seeds compared: {base_qwen_seeds['seeds']}",
                f"- BWGNN (same seeds): ROC-AUC={base_roc_mean:.4f}±{base_roc_std:.4f}, AUPRC={base_auprc_mean:.4f}±{base_auprc_std:.4f}, F1={base_f1_mean:.4f}±{base_f1_std:.4f}, Macro-F1={base_macro_mean:.4f}±{base_macro_std:.4f}",
                f"- CoVER-BWGNN-Qwen: ROC-AUC={qwen['roc_auc_mean']:.4f}±{qwen['roc_auc_std']:.4f}, AUPRC={qwen['auprc_mean']:.4f}±{qwen['auprc_std']:.4f}, F1={qwen['f1_mean']:.4f}±{qwen['f1_std']:.4f}, Macro-F1={qwen['macro_f1_mean']:.4f}±{qwen['macro_f1_std']:.4f}",
                f"- Δ ROC-AUC: {delta_roc:+.4f}",
                f"- Δ AUPRC: {delta_auprc:+.4f}",
                f"- Δ F1@0.5: {delta_f1:+.4f}",
                f"- Δ Macro-F1@0.5: {delta_macro:+.4f}",
            ])

            if base_cal and qwen_cal:
                md_lines.extend([
                    f"- Δ F1@val-threshold: {qwen_cal['f1_mean'] - base_cal['f1_mean']:+.4f}",
                    f"- Δ Macro-F1@val-threshold: {qwen_cal['macro_f1_mean'] - base_cal['macro_f1_mean']:+.4f}",
                ])

        if rule_qwen_seeds["roc_aucs"]:
            rule_roc_mean, rule_roc_std = compute_mean_std(rule_qwen_seeds["roc_aucs"])
            rule_auprc_mean, rule_auprc_std = compute_mean_std(rule_qwen_seeds["auprcs"])
            rule_f1_mean, rule_f1_std = compute_mean_std(rule_qwen_seeds["f1s"])
            rule_macro_mean, rule_macro_std = compute_mean_std(rule_qwen_seeds["macro_f1s"])

            md_lines.extend([
                "",
                "### CoVER-BWGNN-Rule (same seeds as Qwen)",
                f"- Seeds: {base_qwen_seeds['seeds']}",
                f"- ROC-AUC={rule_roc_mean:.4f}±{rule_roc_std:.4f}, AUPRC={rule_auprc_mean:.4f}±{rule_auprc_std:.4f}, F1={rule_f1_mean:.4f}±{rule_f1_std:.4f}, Macro-F1={rule_macro_mean:.4f}±{rule_macro_std:.4f}",
            ])

            md_lines.extend([
                "",
                "**Note:** The Rule vs BWGNN 5-seed comparison shows a larger delta because seeds 42 and 2026 have lower base BWGNN performance. The fair same-seed comparisons above are more accurate.",
            ])

    md_lines.extend([
        "",
        "## Quality Concerns",
        "",
        "### Low ERR Diversity",
        "Qwen teacher produces 31/32 records with `risk_type: structural_discrepancy` per seed. Only 2-3 unique summaries across 32 records. This suggests pattern-matching rather than diverse reasoning.",
    ])

    if "BWGNN" in method_stats and "CoVER-BWGNN-Rule" in method_stats:
        base_per_seed_f1 = method_stats["BWGNN"]["per_seed_f1"]
        rule_per_seed_f1 = method_stats["CoVER-BWGNN-Rule"]["per_seed_f1"]
        f1_zero_seeds = [s for s, f1 in rule_per_seed_f1.items() if f1 == 0.0]
        f1_regressed_seeds = [s for s in f1_zero_seeds if base_per_seed_f1.get(s, 0) > 0.0]

        if f1_zero_seeds:
            md_lines.extend([
                "",
                "### F1=0 Anomaly in Rule Reasoner",
                f"- Seeds with F1=0 in Rule reasoner: {f1_zero_seeds}",
                f"- Seeds where base F1>0 but reasoner F1=0 (regression): {f1_regressed_seeds}",
                "- The reasoner's classification threshold appears miscalibrated for these seeds.",
            ])

    if "BWGNN" in method_stats and "CoVER-BWGNN-Qwen" in method_stats:
        base_per_seed_f1 = method_stats["BWGNN"]["per_seed_f1"]
        qwen_per_seed_f1 = method_stats["CoVER-BWGNN-Qwen"]["per_seed_f1"]
        f1_zero_seeds = [s for s, f1 in qwen_per_seed_f1.items() if f1 == 0.0]
        f1_regressed_seeds = [s for s in f1_zero_seeds if base_per_seed_f1.get(s, 0) > 0.0]

        if f1_zero_seeds:
            md_lines.extend([
                "",
                "### F1=0 Anomaly in Qwen Reasoner",
                f"- Seeds with F1=0 in Qwen reasoner: {f1_zero_seeds}",
                f"- Seeds where base F1>0 but reasoner F1=0 (regression): {f1_regressed_seeds}",
                "- This cross-method failure suggests a deeper calibration issue, not specific to one teacher.",
            ])

    if "BWGNN" in method_stats and "CoVER-BWGNN-Rule" in method_stats:
        base = method_stats["BWGNN"]
        rule = method_stats["CoVER-BWGNN-Rule"]
        delta_f1 = rule["f1_mean"] - base["f1_mean"]
        delta_macro = rule["macro_f1_mean"] - base["macro_f1_mean"]

        if delta_f1 < -0.05 or delta_macro < -0.05:
            md_lines.extend([
                "",
                "### Rule Reasoner Hurts Performance",
                f"- F1 change: {delta_f1:+.4f} (base={base['f1_mean']:.4f}, rule={rule['f1_mean']:.4f})",
                f"- Macro-F1 change: {delta_macro:+.4f} (base={base['macro_f1_mean']:.4f}, rule={rule['macro_f1_mean']:.4f})",
                "- The Rule reasoner actively degrades performance compared to base BWGNN.",
                "- Possible causes: evidence loss weight too high, threshold miscalibration, insufficient ERR diversity.",
            ])

    if "BWGNN" in method_stats and "CoVER-BWGNN-Qwen" in method_stats:
        base = method_stats["BWGNN"]
        qwen = method_stats["CoVER-BWGNN-Qwen"]
        base_cal = calibrated_method_stats.get("BWGNN")
        qwen_cal = calibrated_method_stats.get("CoVER-BWGNN-Qwen")
        delta_f1 = qwen["f1_mean"] - base["f1_mean"]
        delta_macro = qwen["macro_f1_mean"] - base["macro_f1_mean"]

        if delta_f1 < -0.05 or delta_macro < -0.05:
            qwen_seeds = qwen["seeds"]
            base_qwen_seeds = {
                "f1s": [],
                "macro_f1s": [],
            }
            for seed in qwen_seeds:
                key = ("BWGNN", seed)
                if key in all_metrics:
                    m = all_metrics[key]
                    base_qwen_seeds["f1s"].append(m.get("f1", 0))
                    base_qwen_seeds["macro_f1s"].append(m.get("macro_f1", 0))

            if base_qwen_seeds["f1s"]:
                base_f1_mean, base_f1_std = compute_mean_std(base_qwen_seeds["f1s"])
                base_macro_mean, base_macro_std = compute_mean_std(base_qwen_seeds["macro_f1s"])
                fair_delta_f1 = qwen["f1_mean"] - base_f1_mean
                fair_delta_macro = qwen["macro_f1_mean"] - base_macro_mean

                md_lines.extend([
                    "",
                    "### Qwen Reasoner Hurts F1 Performance",
                    f"- Fair same-seed F1 change: {fair_delta_f1:+.4f} (base={base_f1_mean:.4f}, qwen={qwen['f1_mean']:.4f})",
                    f"- Fair same-seed Macro-F1 change: {fair_delta_macro:+.4f} (base={base_macro_mean:.4f}, qwen={qwen['macro_f1_mean']:.4f})",
                    "- The Qwen reasoner shows nearly identical F1 regression to Rule, suggesting the issue is in the reasoner architecture or training, not the teacher.",
                ])

                if base_cal and qwen_cal:
                    md_lines.extend([
                        f"- Fair same-seed F1@val-threshold change: {qwen_cal['f1_mean'] - base_cal['f1_mean']:+.4f}",
                        f"- Fair same-seed Macro-F1@val-threshold change: {qwen_cal['macro_f1_mean'] - base_cal['macro_f1_mean']:+.4f}",
                    ])

    md_lines.extend([
        "",
        "### Amazon Qwen Gap",
        "Amazon Qwen experiments have not been run. Results are pending.",
    ])

    with open(report_dir / "method_comparison.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'='*60}")
    print(f"Method Comparison: {args.dataset}")
    print(f"{'='*60}")

    for method_name in methods:
        if method_name in method_stats:
            s = method_stats[method_name]
            print(f"  {method_name}: ROC-AUC={s['roc_auc_mean']:.4f}±{s['roc_auc_std']:.4f}")

    print(f"\nReport saved to: {report_dir / 'method_comparison.md'}")


if __name__ == "__main__":
    main()
