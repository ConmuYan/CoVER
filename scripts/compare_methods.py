"""Compare methods for a single dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get_results_dir, get_err_cache_dir, get_reports_dir, ensure_dir


CALIBRATED_STAGE1_METRICS = "stage1_calibrated_metrics.json"
CALIBRATED_STAGE3_METRICS = "stage3_calibrated_metrics.json"

METHOD_DISPLAY: dict[str, str] = {
    "base_strat": "BWGNN",
    "rule_safe": "CoVER-Rule-Safe",
    "qwen_safe": "CoVER-Qwen-Safe",
    "base": "BWGNN",
    "rule": "CoVER-Rule",
    "qwen": "CoVER-Qwen",
}

METHOD_METRICS_FILE: dict[str, str] = {
    "base_strat": "stage1_metrics.json",
    "rule_safe": "stage3_metrics.json",
    "qwen_safe": "stage3_metrics.json",
    "base": "stage1_metrics.json",
    "rule": "stage3_metrics.json",
    "qwen": "stage3_metrics.json",
}

METHOD_CALIBRATED_METRICS_FILE: dict[str, str] = {
    "base_strat": CALIBRATED_STAGE1_METRICS,
    "rule_safe": CALIBRATED_STAGE3_METRICS,
    "qwen_safe": CALIBRATED_STAGE3_METRICS,
    "base": CALIBRATED_STAGE1_METRICS,
    "rule": CALIBRATED_STAGE3_METRICS,
    "qwen": CALIBRATED_STAGE3_METRICS,
}

METHOD_RUN_NAME: dict[str, str] = {
    "base_strat": "base_strat",
    "rule_safe": "rule_safe",
    "qwen_safe": "qwen_safe",
    "base": "base",
    "rule": "rule",
    "qwen": "qwen",
}


def method_display(method: str) -> str:
    """Return display name for a method key."""
    return METHOD_DISPLAY.get(method, method)


def method_metrics_file(method: str) -> str:
    """Return the metrics filename for a method key."""
    return METHOD_METRICS_FILE.get(method, "stage3_metrics.json")


def method_calibrated_metrics_file(method: str) -> str:
    """Return the calibrated metrics filename for a method key."""
    return METHOD_CALIBRATED_METRICS_FILE.get(method, CALIBRATED_STAGE3_METRICS)


def method_is_base(method: str) -> bool:
    return method_metrics_file(method) == "stage1_metrics.json"


def method_report_dir(dataset: str, model: str, method: str, seed: int) -> Path:
    return Path("artifacts") / "reports" / dataset / model / method_run_name(method) / f"seed_{seed}"


def method_run_name(method: str) -> str:
    """Return the run_name used in results directory for a method key."""
    return METHOD_RUN_NAME.get(method, method)


# ── Existing helper functions (unchanged) ──────────────────────────────


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


def _collect_metrics_for_methods(
    methods: list[str],
    dataset: str,
    model: str,
    seeds: list[int],
) -> tuple[
    dict[tuple[str, int], dict],
    dict[tuple[str, int], dict],
]:
    """Load raw + calibrated metrics for all (method, seed) combos."""
    all_metrics: dict[tuple[str, int], dict] = {}
    all_calibrated: dict[tuple[str, int], dict] = {}

    for method in methods:
        run = method_run_name(method)
        metrics_file = method_metrics_file(method)
        cal_file = method_calibrated_metrics_file(method)

        for seed in seeds:
            results_dir = get_results_dir(dataset, model, run, seed)
            metrics = load_metrics(results_dir / metrics_file)
            if metrics is not None:
                all_metrics[(method, seed)] = metrics

            cal_path = results_dir / cal_file
            cal = load_metrics(cal_path)
            if cal is not None:
                all_calibrated[(method, seed)] = cal

    return all_metrics, all_calibrated


def _build_method_stats(
    methods: list[str],
    seeds: list[int],
    all_metrics: dict[tuple[str, int], dict],
    all_calibrated: dict[tuple[str, int], dict],
    *,
    seed_filter: set[int] | None = None,
) -> tuple[
    dict[str, dict],
    dict[str, dict],
]:
    """Build summarised stats per method, optionally restricted to *seed_filter*."""
    method_stats: dict[str, dict] = {}
    calibrated_method_stats: dict[str, dict] = {}

    for method in methods:
        fixed_by_seed: dict[int, dict[str, float]] = {}
        calibrated_by_seed: dict[int, dict[str, float]] = {}

        for seed in seeds:
            if seed_filter is not None and seed not in seed_filter:
                continue
            key = (method, seed)
            if key in all_metrics:
                nm = normalize_metrics(all_metrics[key], calibrated=False)
                if nm is not None:
                    fixed_by_seed[seed] = nm
            if key in all_calibrated:
                nm = normalize_metrics(all_calibrated[key], calibrated=True)
                if nm is not None:
                    calibrated_by_seed[seed] = nm

        if fixed_by_seed:
            method_stats[method] = summarize_metrics(fixed_by_seed)
        if calibrated_by_seed:
            calibrated_method_stats[method] = summarize_metrics(calibrated_by_seed)

    return method_stats, calibrated_method_stats


def _seed_sets_by_method(all_metrics: dict[tuple[str, int], dict], methods: list[str]) -> dict[str, set[int]]:
    seed_sets: dict[str, set[int]] = {}
    for method in methods:
        seed_sets[method] = {seed for (m, seed) in all_metrics if m == method}
    return seed_sets


def _find_common_seeds(methods: list[str], all_metrics: dict[tuple[str, int], dict]) -> set[int] | None:
    seed_sets = [_seed_sets_by_method(all_metrics, [method])[method] for method in methods]
    if not seed_sets or any(not seeds for seeds in seed_sets):
        return None
    common = set(seed_sets[0])
    for seeds in seed_sets[1:]:
        common &= seeds
    return common


def _load_json_fallback(paths: list[Path]) -> dict | None:
    for path in paths:
        data = load_metrics(path)
        if data is not None:
            return data
    return None


def load_evidence_quality_report(dataset: str, model: str, method: str, seed: int) -> dict | None:
    run = method_run_name(method)
    candidates = [
        get_err_cache_dir(dataset, model, run, seed) / "evidence_quality_report.json",
        get_reports_dir(dataset, model, seed) / "evidence_quality_report.json",
        method_report_dir(dataset, model, method, seed) / "evidence_quality_report.json",
    ]
    return _load_json_fallback(candidates)


def load_reasoner_diagnosis(dataset: str, model: str, method: str, seed: int) -> dict | None:
    run = method_run_name(method)
    candidates = [
        get_err_cache_dir(dataset, model, run, seed) / "reasoner_diagnosis.json",
        get_results_dir(dataset, model, run, seed) / "reasoner_diagnosis.json",
        method_report_dir(dataset, model, method, seed) / "reasoner_diagnosis.json",
    ]
    return _load_json_fallback(candidates)


def _format_seed_list(seeds: list[int] | set[int]) -> str:
    ordered = sorted(seeds)
    return ", ".join(str(seed) for seed in ordered) if ordered else "none"


def _available_seed_summary(methods: list[str], seed_sets: dict[str, set[int]]) -> list[str]:
    lines = ["### Available Seed Coverage", "", "| Method | Available seeds |", "|---|---|"]
    for method in methods:
        lines.append(f"| {method_display(method)} | {_format_seed_list(seed_sets.get(method, set()))} |")
    return lines


def _render_results_table(
    methods: list[str],
    method_stats: dict[str, dict],
    calibrated_method_stats: dict[str, dict],
    total_seeds: int,
) -> list[str]:
    lines = [
        "| Method | Seeds | ROC-AUC | AUPRC | F1@0.5 | F1@val-threshold | Macro-F1@0.5 | Macro-F1@val-threshold | Calibration |",
        "|--------|-------|---------|-------|--------|------------------|--------------|------------------------|-------------|",
    ]

    for method in methods:
        display = method_display(method)
        if method in method_stats:
            s = method_stats[method]
            num_seeds = len(s["seeds"])
            seeds_note = f" ({num_seeds}/{total_seeds})" if num_seeds < total_seeds else ""
            calibrated = calibrated_method_stats.get(method)
            cal_note = calibration_status(s, calibrated)
            f1_val = format_mean_std(calibrated["f1_mean"], calibrated["f1_std"]) if calibrated else "N/A"
            macro_val = format_mean_std(calibrated["macro_f1_mean"], calibrated["macro_f1_std"]) if calibrated else "N/A"
            lines.append(
                f"| {display}{seeds_note} | {num_seeds} | "
                f"{s['roc_auc_mean']:.4f}±{s['roc_auc_std']:.4f} | "
                f"{s['auprc_mean']:.4f}±{s['auprc_std']:.4f} | "
                f"{s['f1_mean']:.4f}±{s['f1_std']:.4f} | "
                f"{f1_val} | "
                f"{s['macro_f1_mean']:.4f}±{s['macro_f1_std']:.4f} | "
                f"{macro_val} | "
                f"{cal_note} |"
            )
        else:
            lines.append(f"| {display} | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    return lines


def _metrics_for_seeds(metrics_by_seed: dict[int, dict[str, float]], seeds: set[int] | None) -> dict[int, dict[str, float]]:
    if seeds is None:
        return metrics_by_seed
    return subset_metrics(metrics_by_seed, sorted(seeds))


def _pairwise_deltas(
    method_a: str,
    method_b: str,
    method_stats: dict[str, dict],
    calibrated_method_stats: dict[str, dict],
) -> dict[str, float] | None:
    if method_a not in method_stats or method_b not in method_stats:
        return None
    a = method_stats[method_a]
    b = method_stats[method_b]
    delta = {
        "roc_auc": a["roc_auc_mean"] - b["roc_auc_mean"],
        "auprc": a["auprc_mean"] - b["auprc_mean"],
        "f1": a["f1_mean"] - b["f1_mean"],
        "macro_f1": a["macro_f1_mean"] - b["macro_f1_mean"],
    }
    a_cal = calibrated_method_stats.get(method_a)
    b_cal = calibrated_method_stats.get(method_b)
    if a_cal and b_cal:
        delta["f1_val"] = a_cal["f1_mean"] - b_cal["f1_mean"]
        delta["macro_f1_val"] = a_cal["macro_f1_mean"] - b_cal["macro_f1_mean"]
    return delta


def _evidence_interpretation(dataset: str, model: str, methods: list[str], seed_filter: set[int] | None) -> list[str]:
    lines = ["### Evidence Quality Interpretation", ""]
    found = False
    for method in methods:
        if method_is_base(method):
            continue
        method_lines: list[str] = []
        for seed in sorted(seed_filter) if seed_filter is not None else []:
            report = load_evidence_quality_report(dataset, model, method, seed)
            if not report:
                continue
            found = True
            weak_ratio = report.get("weak_or_uncertain_ratio", report.get("weak_or_uncertain_count", 0))
            method_lines.append(
                f"- seed {seed}: acceptance={report.get('acceptance_rate', report.get('final_acceptance_rate', 'N/A'))}, weak/uncertain={weak_ratio}, cards={report.get('num_cards', 'N/A')}"
            )
        if method_lines:
            lines.append(f"- **{method_display(method)}**")
            lines.extend(method_lines)
    if not found:
        lines.append("- No evidence quality reports found.")
    return lines


def _residual_interpretation(dataset: str, model: str, methods: list[str], seed_filter: set[int] | None) -> list[str]:
    lines = ["### Residual Safety Interpretation", ""]
    found = False
    for method in methods:
        if method_is_base(method):
            continue
        method_lines: list[str] = []
        for seed in sorted(seed_filter) if seed_filter is not None else []:
            diag = load_reasoner_diagnosis(dataset, model, method, seed)
            if not diag:
                continue
            found = True
            residual = diag.get("residual_shift", {})
            rho_check = diag.get("rho_zero_check", {})
            delta = diag.get("delta_vs_base", {})
            gate = diag.get("gate", {})
            method_lines.append(
                f"- seed {seed}: rho={diag.get('rho', 'N/A')}, gate_mean={gate.get('mean', 'N/A')}, residual_mean={residual.get('mean', 'N/A')}, rho0_pass={rho_check.get('passes', 'N/A')}, ΔROC-AUC={delta.get('roc_auc', 'N/A')}"
            )
        if method_lines:
            lines.append(f"- **{method_display(method)}**")
            lines.extend(method_lines)
    if not found:
        lines.append("- No reasoner diagnosis reports found.")
    return lines


def _final_recommendation(
    methods: list[str],
    fair_stats: dict[str, dict],
) -> list[str]:
    lines = ["### Final Recommendation", ""]
    base_method = next((method for method in methods if method_is_base(method)), None)
    if base_method is None or base_method not in fair_stats:
        lines.append("- Insufficient fair same-seed data to make a recommendation.")
        return lines

    base_roc = fair_stats[base_method]["roc_auc_mean"]
    best_method = base_method
    best_roc = base_roc
    for method in methods:
        if method == base_method or method not in fair_stats:
            continue
        roc = fair_stats[method]["roc_auc_mean"]
        if roc > best_roc:
            best_method = method
            best_roc = roc

    if best_method == base_method:
        lines.append(f"- Recommend reporting **{method_display(base_method)}** as best on fair same-seed ROC-AUC.")
    else:
        lines.append(
            f"- Recommend reporting **{method_display(best_method)}**; it beats {method_display(base_method)} by ΔROC-AUC={best_roc - base_roc:+.4f} on fair same-seed data."
        )

    for other in methods:
        if other == base_method or other not in fair_stats:
            continue
        delta = fair_stats[other]["roc_auc_mean"] - base_roc
        lines.append(
            f"- {method_display(other)} vs {method_display(base_method)}: {('beats' if delta > 0 else 'does not beat')} base (ΔROC-AUC={delta:+.4f})."
        )
    if len([m for m in methods if not method_is_base(m)]) >= 2:
        student_methods = [m for m in methods if not method_is_base(m)]
        if student_methods[0] in fair_stats and student_methods[1] in fair_stats:
            delta = fair_stats[student_methods[1]]["roc_auc_mean"] - fair_stats[student_methods[0]]["roc_auc_mean"]
            lines.append(
                f"- {method_display(student_methods[1])} vs {method_display(student_methods[0])}: {('beats' if delta > 0 else 'does not beat')} (ΔROC-AUC={delta:+.4f})."
            )
    return lines


def _pairwise_section(
    methods: list[str],
    fair_stats: dict[str, dict],
    fair_cal_stats: dict[str, dict],
) -> list[str]:
    lines: list[str] = []
    base_method = next((method for method in methods if method_is_base(method)), None)
    if base_method:
        for method in methods:
            if method == base_method:
                continue
            delta = _pairwise_deltas(method, base_method, fair_stats, fair_cal_stats)
            if delta is None:
                continue
            lines.extend([
                f"### Fair same-seed: {method_display(method)} vs {method_display(base_method)}",
                f"- Δ ROC-AUC: {delta['roc_auc']:+.4f}",
                f"- Δ AUPRC: {delta['auprc']:+.4f}",
                f"- Δ F1@0.5: {delta['f1']:+.4f}",
                f"- Δ Macro-F1@0.5: {delta['macro_f1']:+.4f}",
            ])
            if "f1_val" in delta:
                lines.append(f"- Δ F1@val-threshold: {delta['f1_val']:+.4f}")
                lines.append(f"- Δ Macro-F1@val-threshold: {delta['macro_f1_val']:+.4f}")
            lines.append("")

    student_methods = [m for m in methods if not method_is_base(m)]
    if len(student_methods) >= 2:
        for i in range(len(student_methods)):
            for j in range(i + 1, len(student_methods)):
                delta = _pairwise_deltas(student_methods[i], student_methods[j], fair_stats, fair_cal_stats)
                if delta is None:
                    continue
                lines.extend([
                    f"### Fair same-seed: {method_display(student_methods[i])} vs {method_display(student_methods[j])}",
                    f"- Δ ROC-AUC: {delta['roc_auc']:+.4f}",
                    f"- Δ AUPRC: {delta['auprc']:+.4f}",
                    f"- Δ F1@0.5: {delta['f1']:+.4f}",
                    f"- Δ Macro-F1@0.5: {delta['macro_f1']:+.4f}",
                ])
                if "f1_val" in delta:
                    lines.append(f"- Δ F1@val-threshold: {delta['f1_val']:+.4f}")
                    lines.append(f"- Δ Macro-F1@val-threshold: {delta['macro_f1_val']:+.4f}")
                lines.append("")
    return lines


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Compare methods for a single dataset.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="bwgnn")
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789, 42, 2026])
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["base_strat", "rule_safe", "qwen_safe"],
        help="Method names matching result dirs (e.g. base_strat rule_safe qwen_safe).",
    )
    parser.add_argument(
        "--fair_same_seed",
        action="store_true",
        default=False,
        help="Restrict all comparisons to seeds present in ALL requested methods.",
    )
    args = parser.parse_args()

    methods: list[str] = args.methods
    all_metrics, all_calibrated = _collect_metrics_for_methods(methods, args.dataset, args.model, args.seeds)
    seed_sets = _seed_sets_by_method(all_metrics, methods)
    common_seeds = _find_common_seeds(methods, all_metrics) if args.fair_same_seed else None

    full_stats, full_cal_stats = _build_method_stats(methods, args.seeds, all_metrics, all_calibrated)
    fair_stats: dict[str, dict] = {}
    fair_cal_stats: dict[str, dict] = {}
    if common_seeds is not None:
        fair_stats, fair_cal_stats = _build_method_stats(
            methods,
            args.seeds,
            all_metrics,
            all_calibrated,
            seed_filter=common_seeds,
        )

    report_dir = ensure_dir(Path("artifacts") / "reports" / args.dataset / args.model)
    output_path = report_dir / "final_method_comparison.md"

    md_lines: list[str] = [
        f"# Method Comparison: {args.dataset}",
        "",
        f"Model: {args.model}",
        f"Requested seeds: {args.seeds}",
        f"Methods: {[method_display(m) for m in methods]}",
        f"Fair same-seed mode: {args.fair_same_seed}",
        "",
        "## Full Available Seed Results",
        "",
    ]
    md_lines.extend(_render_results_table(methods, full_stats, full_cal_stats, len(args.seeds)))
    md_lines.extend(["", "## Available Seed Coverage", "", "| Method | Available seeds |", "|---|---|"])
    for method in methods:
        md_lines.append(f"| {method_display(method)} | {_format_seed_list(seed_sets.get(method, set()))} |")

    md_lines.extend(["", "## Fair Same-Seed Results", ""])
    if common_seeds is None:
        md_lines.extend([
            "**Could not compute:** no common seeds across all methods.",
            "",
        ])
    else:
        mismatch = any(seed_sets.get(method, set()) != common_seeds for method in methods)
        if mismatch:
            md_lines.append(
                f"**Seed alignment warning:** methods have different seed sets; common seeds are {_format_seed_list(common_seeds)}."
            )
            for method in methods:
                md_lines.append(f"- {method_display(method)}: {_format_seed_list(seed_sets.get(method, set()))}")
            md_lines.append("")
        md_lines.append(f"Seeds used for fair comparison: {_format_seed_list(common_seeds)}")
        md_lines.append("")
        md_lines.extend(_render_results_table(methods, fair_stats, fair_cal_stats, len(common_seeds)))
        md_lines.extend(["", "### Pairwise Deltas (Fair Same-Seed)", ""])
        md_lines.extend(_pairwise_section(methods, fair_stats, fair_cal_stats))
        md_lines.extend(_evidence_interpretation(args.dataset, args.model, methods, common_seeds))
        md_lines.append("")
        md_lines.extend(_residual_interpretation(args.dataset, args.model, methods, common_seeds))
        md_lines.append("")
        md_lines.extend(_final_recommendation(methods, fair_stats))

        base_method = next((method for method in methods if method_is_base(method)), None)
        rule_method = next((method for method in methods if "rule" in method_display(method).lower() and not method_is_base(method)), None)
        qwen_method = next((method for method in methods if "qwen" in method_display(method).lower()), None)
        if base_method and rule_method and qwen_method and base_method in fair_stats and rule_method in fair_stats and qwen_method in fair_stats:
            md_lines.extend([
                "",
                "## Verdicts",
                f"- Rule-Safe beats base: {'Yes' if fair_stats[rule_method]['roc_auc_mean'] > fair_stats[base_method]['roc_auc_mean'] else 'No'} (ΔROC-AUC={fair_stats[rule_method]['roc_auc_mean'] - fair_stats[base_method]['roc_auc_mean']:+.4f})",
                f"- Qwen-Safe beats base: {'Yes' if fair_stats[qwen_method]['roc_auc_mean'] > fair_stats[base_method]['roc_auc_mean'] else 'No'} (ΔROC-AUC={fair_stats[qwen_method]['roc_auc_mean'] - fair_stats[base_method]['roc_auc_mean']:+.4f})",
                f"- Qwen-Safe beats Rule-Safe: {'Yes' if fair_stats[qwen_method]['roc_auc_mean'] > fair_stats[rule_method]['roc_auc_mean'] else 'No'} (ΔROC-AUC={fair_stats[qwen_method]['roc_auc_mean'] - fair_stats[rule_method]['roc_auc_mean']:+.4f})",
            ])

    if full_stats:
        md_lines.extend(["", "## Analysis", ""])
        base_method = next((method for method in methods if method_is_base(method)), None)
        student_methods = [m for m in methods if not method_is_base(m)]
        if base_method:
            for method in student_methods:
                if method in full_stats:
                    delta = _pairwise_deltas(method, base_method, full_stats, full_cal_stats)
                    if delta is not None:
                        md_lines.extend([
                            f"### {method_display(method)} vs {method_display(base_method)}",
                            f"- Δ ROC-AUC: {delta['roc_auc']:+.4f}",
                            f"- Δ AUPRC: {delta['auprc']:+.4f}",
                            f"- Δ F1@0.5: {delta['f1']:+.4f}",
                            f"- Δ Macro-F1@0.5: {delta['macro_f1']:+.4f}",
                        ])
                        if "f1_val" in delta:
                            md_lines.append(f"- Δ F1@val-threshold: {delta['f1_val']:+.4f}")
                            md_lines.append(f"- Δ Macro-F1@val-threshold: {delta['macro_f1_val']:+.4f}")
                        md_lines.append("")

        if base_method and any(m for m in student_methods if m in full_stats):
            base_per_seed_f1 = full_stats[base_method]["per_seed_f1"]
            for method in student_methods:
                if method not in full_stats:
                    continue
                per_seed_f1 = full_stats[method]["per_seed_f1"]
                zero_seeds = [seed for seed, f1 in per_seed_f1.items() if f1 == 0.0]
                regressed = [seed for seed in zero_seeds if base_per_seed_f1.get(seed, 0) > 0.0]
                if zero_seeds:
                    md_lines.extend([
                        f"### F1=0 Anomaly in {method_display(method)} Reasoner",
                        f"- Seeds with F1=0: {zero_seeds}",
                        f"- Seeds where base F1>0 but reasoner F1=0 (regression): {regressed}",
                    ])
                    md_lines.append("")

        rule_method = next((m for m in methods if "rule" in method_display(m).lower() and not method_is_base(m)), None)
        if base_method and rule_method:
            if base_method in full_stats and rule_method in full_stats:
                base_s = full_stats[base_method]
                rule_s = full_stats[rule_method]
                delta_f1 = rule_s["f1_mean"] - base_s["f1_mean"]
                delta_macro = rule_s["macro_f1_mean"] - base_s["macro_f1_mean"]
                if delta_f1 < -0.05 or delta_macro < -0.05:
                    md_lines.extend([
                        f"### {method_display(rule_method)} Hurts Performance",
                        f"- F1 change: {delta_f1:+.4f} (base={base_s['f1_mean']:.4f}, rule={rule_s['f1_mean']:.4f})",
                        f"- Macro-F1 change: {delta_macro:+.4f} (base={base_s['macro_f1_mean']:.4f}, rule={rule_s['macro_f1_mean']:.4f})",
                    ])

        qwen_method = next((m for m in methods if method_display(m) == "CoVER-Qwen-Safe"), None)
        if base_method and qwen_method and base_method in full_stats and qwen_method in full_stats:
            base_s = full_stats[base_method]
            qwen_s = full_stats[qwen_method]
            delta_f1 = qwen_s["f1_mean"] - base_s["f1_mean"]
            delta_macro = qwen_s["macro_f1_mean"] - base_s["macro_f1_mean"]
            if delta_f1 < -0.05 or delta_macro < -0.05:
                md_lines.extend([
                    f"### {method_display(qwen_method)} Hurts F1 Performance",
                    f"- F1 change: {delta_f1:+.4f} (base={base_s['f1_mean']:.4f}, qwen={qwen_s['f1_mean']:.4f})",
                    f"- Macro-F1 change: {delta_macro:+.4f} (base={base_s['macro_f1_mean']:.4f}, qwen={qwen_s['macro_f1_mean']:.4f})",
                ])

    md_lines.extend(["", "### Amazon Qwen Gap", "Amazon Qwen experiments have not been run. Results are pending."])

    with open(output_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'='*60}")
    print(f"Method Comparison: {args.dataset}")
    print(f"{'='*60}")
    for method in methods:
        display = method_display(method)
        if method in full_stats:
            s = full_stats[method]
            print(f"  {display}: ROC-AUC={s['roc_auc_mean']:.4f}±{s['roc_auc_std']:.4f}")
        else:
            print(f"  {display}: no data")
    if args.fair_same_seed and common_seeds is not None:
        print(f"\n  Fair same-seed (common seeds: {sorted(common_seeds)}):")
        for method in methods:
            if method in fair_stats:
                s = fair_stats[method]
                print(f"    {method_display(method)}: ROC-AUC={s['roc_auc_mean']:.4f}±{s['roc_auc_std']:.4f}")

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
