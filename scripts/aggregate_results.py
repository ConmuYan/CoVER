"""Aggregate results from controlled experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get_results_dir, get_err_cache_dir, get_reports_dir, get_split_path, ensure_dir


def load_metrics(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


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
    std = variance ** 0.5
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["yelpchi", "amazon"])
    parser.add_argument("--model", type=str, default="bwgnn")
    parser.add_argument("--teachers", nargs="+", default=["base", "rule", "qwen"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789, 42, 2026])
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    all_metrics = {}
    all_calibrated = {}
    all_split_meta = {}
    all_evidence = {}
    missing = []

    for dataset in args.datasets:
        for teacher in args.teachers:
            for seed in args.seeds:
                if teacher == "base":
                    results_dir = get_results_dir(dataset, args.model, "base", seed)
                    metrics_path = results_dir / "stage1_metrics.json"
                    cal_path = results_dir / "stage1_calibrated_metrics.json"
                else:
                    results_dir = get_results_dir(dataset, args.model, teacher, seed)
                    metrics_path = results_dir / "stage3_metrics.json"
                    cal_path = results_dir / "stage3_calibrated_metrics.json"

                metrics = load_metrics(metrics_path)
                if metrics:
                    key = (dataset, teacher, seed)
                    all_metrics[key] = metrics
                else:
                    missing.append((dataset, teacher, seed))

                calibrated = load_json(cal_path)
                if calibrated:
                    key = (dataset, teacher, seed)
                    all_calibrated[key] = calibrated

                split_meta_path = get_split_path(dataset, seed).parent / "split_meta.json"
                split_meta = load_json(split_meta_path)
                if split_meta:
                    all_split_meta[(dataset, seed)] = split_meta

                if teacher != "base":
                    err_stats_path = get_err_cache_dir(dataset, args.model, teacher, seed) / "stage2_stats.json"
                    err_stats = load_json(err_stats_path)
                    if err_stats:
                        key = (dataset, teacher, seed)
                        all_evidence[key] = err_stats

                    verifier_stats_path = get_err_cache_dir(dataset, args.model, teacher, seed) / "verifier_stats.json"
                    verifier_stats = load_json(verifier_stats_path)
                    if verifier_stats and key in all_evidence:
                        all_evidence[key]["reject_reason_counts"] = verifier_stats.get("reject_reason_counts", {})

                    evidence_report_path = get_err_cache_dir(dataset, args.model, teacher, seed) / "evidence_quality_report.json"
                    evidence_report = load_json(evidence_report_path)
                    if not evidence_report:
                        reports_dir = Path("artifacts") / "reports" / dataset / args.model / teacher / f"seed_{seed}"
                        evidence_report_path = reports_dir / "evidence_quality_report.json"
                        evidence_report = load_json(evidence_report_path)
                    if evidence_report and key in all_evidence:
                        all_evidence[key]["risk_type_distribution"] = evidence_report.get("risk_type_distribution", {})
                        all_evidence[key]["supporting_evidence_distribution"] = evidence_report.get("supporting_evidence_distribution", {})
                        all_evidence[key]["counter_evidence_distribution"] = evidence_report.get("counter_evidence_distribution", {})
                        all_evidence[key]["weak_or_uncertain_ratio"] = evidence_report.get("weak_or_uncertain_ratio", 0)

    if missing and not args.allow_missing:
        print(f"Error: Missing metrics for {len(missing)} runs:")
        for d, t, s in missing:
            print(f"  {d}/{t}/seed_{s}")
        sys.exit(1)

    if missing:
        print(f"Warning: Missing metrics for {len(missing)} runs (allow_missing=True)")

    metrics_rows = []
    for dataset in args.datasets:
        for teacher in args.teachers:
            method_name = {
                "base": "BWGNN",
                "rule": "CoVER-BWGNN-Rule",
                "qwen": "CoVER-BWGNN-Qwen",
            }.get(teacher, teacher)

            seeds_found = []
            roc_aucs = []
            auprcs = []
            f1s = []
            macro_f1s = []
            f1_fixeds = []
            f1_val_f1s = []
            macro_f1_fixeds = []
            macro_f1_val_macros = []
            thresholds_used = []
            has_calibrated = False
            has_stratified_info = True
            any_non_stratified = False

            for seed in args.seeds:
                key = (dataset, teacher, seed)
                if key in all_metrics:
                    m = all_metrics[key]
                    seeds_found.append(seed)
                    roc_aucs.append(m.get("roc_auc", 0))
                    auprcs.append(m.get("auprc", 0))
                    f1s.append(m.get("f1", 0))
                    macro_f1s.append(m.get("macro_f1", 0))

                    cal = all_calibrated.get(key)
                    if cal:
                        has_calibrated = True
                        fixed = cal.get("fixed_threshold_metrics", {})
                        val_f1 = cal.get("val_f1_threshold_metrics", {})
                        val_macro = cal.get("val_macro_f1_threshold_metrics", {})
                        f1_fixeds.append(fixed.get("f1", 0))
                        f1_val_f1s.append(val_f1.get("f1", 0))
                        macro_f1_fixeds.append(fixed.get("macro_f1", 0))
                        macro_f1_val_macros.append(val_macro.get("macro_f1", 0))
                        thresholds_used.append(val_f1.get("threshold_used", 0.5))

                    split_meta = all_split_meta.get((dataset, seed))
                    if split_meta is not None:
                        if not split_meta.get("stratified", True):
                            any_non_stratified = True
                    else:
                        has_stratified_info = False

            if seeds_found:
                roc_auc_mean, roc_auc_std = compute_mean_std(roc_aucs)
                auprc_mean, auprc_std = compute_mean_std(auprcs)
                f1_mean, f1_std = compute_mean_std(f1s)
                macro_f1_mean, macro_f1_std = compute_mean_std(macro_f1s)

                row = {
                    "dataset": dataset,
                    "model": args.model,
                    "method": method_name,
                    "seeds": len(seeds_found),
                    "roc_auc_mean": roc_auc_mean,
                    "roc_auc_std": roc_auc_std,
                    "auprc_mean": auprc_mean,
                    "auprc_std": auprc_std,
                    "f1_mean": f1_mean,
                    "f1_std": f1_std,
                    "macro_f1_mean": macro_f1_mean,
                    "macro_f1_std": macro_f1_std,
                }

                if has_calibrated:
                    f1_fixed_mean, f1_fixed_std = compute_mean_std(f1_fixeds)
                    f1_val_f1_mean, f1_val_f1_std = compute_mean_std(f1_val_f1s)
                    macro_f1_fixed_mean, macro_f1_fixed_std = compute_mean_std(macro_f1_fixeds)
                    macro_f1_val_macro_mean, macro_f1_val_macro_std = compute_mean_std(macro_f1_val_macros)
                    threshold_mean, _ = compute_mean_std(thresholds_used)
                    row.update({
                        "f1_fixed_mean": f1_fixed_mean,
                        "f1_fixed_std": f1_fixed_std,
                        "f1_val_f1_mean": f1_val_f1_mean,
                        "f1_val_f1_std": f1_val_f1_std,
                        "macro_f1_fixed_mean": macro_f1_fixed_mean,
                        "macro_f1_fixed_std": macro_f1_fixed_std,
                        "macro_f1_val_macro_mean": macro_f1_val_macro_mean,
                        "macro_f1_val_macro_std": macro_f1_val_macro_std,
                        "threshold_used_mean": threshold_mean,
                    })

                if has_stratified_info and any_non_stratified:
                    row["split_warning"] = "non-stratified"
                elif not has_stratified_info:
                    row["split_warning"] = "unknown"
                else:
                    row["split_warning"] = ""

                row["has_calibrated"] = has_calibrated
                metrics_rows.append(row)

    for row in metrics_rows:
        if row["method"] == "CoVER-BWGNN-Qwen":
            row["delta_roc_auc_vs_base"] = None
            row["delta_auprc_vs_base"] = None
            row["delta_f1_vs_base"] = None
            row["delta_macro_f1_vs_base"] = None
        elif row["method"] != "BWGNN":
            base_row = next((r for r in metrics_rows if r["dataset"] == row["dataset"] and r["method"] == "BWGNN"), None)
            if base_row:
                row["delta_roc_auc_vs_base"] = row["roc_auc_mean"] - base_row["roc_auc_mean"]
                row["delta_auprc_vs_base"] = row["auprc_mean"] - base_row["auprc_mean"]
                row["delta_f1_vs_base"] = row["f1_mean"] - base_row["f1_mean"]
                row["delta_macro_f1_vs_base"] = row["macro_f1_mean"] - base_row["macro_f1_mean"]
            else:
                row["delta_roc_auc_vs_base"] = 0
                row["delta_auprc_vs_base"] = 0
                row["delta_f1_vs_base"] = 0
                row["delta_macro_f1_vs_base"] = 0
        else:
            row["delta_roc_auc_vs_base"] = 0
            row["delta_auprc_vs_base"] = 0
            row["delta_f1_vs_base"] = 0
            row["delta_macro_f1_vs_base"] = 0

    evidence_rows = []
    for dataset in args.datasets:
        for teacher in ["rule", "qwen"]:
            parse_successes = []
            acceptances = []
            retry_rates = []
            weak_ratios = []
            risk_type_counts = {}
            supporting_counts = {}
            counter_counts = {}
            reject_reason_counts = {}

            for seed in args.seeds:
                key = (dataset, teacher, seed)
                if key in all_evidence:
                    e = all_evidence[key]
                    if teacher == "rule":
                        parse_successes.append(1.0)
                    else:
                        parse_successes.append(e.get("num_parse_success", 0) / max(e.get("num_initial_calls", 1), 1))
                    acceptances.append(e.get("final_acceptance_rate", e.get("acceptance_rate", 0)))
                    retry_rates.append(e.get("num_accepted_after_retry", 0) / max(e.get("num_final_accepted", 1), 1))
                    weak_ratios.append(e.get("weak_or_uncertain_ratio", 0))

                    for rt, count in e.get("risk_type_distribution", {}).items():
                        risk_type_counts[rt] = risk_type_counts.get(rt, 0) + count
                    for s, count in e.get("supporting_evidence_distribution", {}).items():
                        supporting_counts[s] = supporting_counts.get(s, 0) + count
                    for c, count in e.get("counter_evidence_distribution", {}).items():
                        counter_counts[c] = counter_counts.get(c, 0) + count
                    for r, count in e.get("reject_reason_counts", {}).items():
                        reject_reason_counts[r] = reject_reason_counts.get(r, 0) + count

            if parse_successes:
                parse_mean, parse_std = compute_mean_std(parse_successes)
                acc_mean, acc_std = compute_mean_std(acceptances)
                retry_mean, retry_std = compute_mean_std(retry_rates)
                weak_mean, weak_std = compute_mean_std(weak_ratios)

                top_risk_types = sorted(risk_type_counts.items(), key=lambda x: -x[1])[:5]
                top_supporting = sorted(supporting_counts.items(), key=lambda x: -x[1])[:5]
                top_counter = sorted(counter_counts.items(), key=lambda x: -x[1])[:5]
                top_reject_reasons = sorted(reject_reason_counts.items(), key=lambda x: -x[1])[:5]

                evidence_rows.append({
                    "dataset": dataset,
                    "teacher": teacher,
                    "seeds": len(parse_successes),
                    "parse_success_rate_mean": parse_mean,
                    "parse_success_rate_std": parse_std,
                    "acceptance_rate_mean": acc_mean,
                    "acceptance_rate_std": acc_std,
                    "accepted_after_retry_rate_mean": retry_mean,
                    "accepted_after_retry_rate_std": retry_std,
                    "weak_or_uncertain_ratio_mean": weak_mean,
                    "weak_or_uncertain_ratio_std": weak_std,
                    "top_risk_types": "; ".join(f"{rt}({c})" for rt, c in top_risk_types),
                    "top_supporting_evidence": "; ".join(f"{s}({c})" for s, c in top_supporting),
                    "top_counter_evidence": "; ".join(f"{c}({cnt})" for c, cnt in top_counter),
                    "main_reject_reasons": "; ".join(f"{r}({c})" for r, c in top_reject_reasons),
                })

    fair_comparison_rows = []
    for dataset in args.datasets:
        qwen_row = next((r for r in metrics_rows if r["dataset"] == dataset and r["method"] == "CoVER-BWGNN-Qwen"), None)
        if not qwen_row:
            continue

        qwen_seeds = []
        for seed in args.seeds:
            key = (dataset, "qwen", seed)
            if key in all_metrics:
                qwen_seeds.append(seed)

        if not qwen_seeds:
            continue

        roc_aucs = []
        auprcs = []
        f1s = []
        macro_f1s = []
        for seed in qwen_seeds:
            key = (dataset, "base", seed)
            if key in all_metrics:
                m = all_metrics[key]
                roc_aucs.append(m.get("roc_auc", 0))
                auprcs.append(m.get("auprc", 0))
                f1s.append(m.get("f1", 0))
                macro_f1s.append(m.get("macro_f1", 0))

        if roc_aucs:
            roc_auc_mean, roc_auc_std = compute_mean_std(roc_aucs)
            auprc_mean, auprc_std = compute_mean_std(auprcs)
            f1_mean, f1_std = compute_mean_std(f1s)
            macro_f1_mean, macro_f1_std = compute_mean_std(macro_f1s)

            fair_comparison_rows.append({
                "dataset": dataset,
                "model": args.model,
                "method": "BWGNN (same seeds)",
                "seeds": len(qwen_seeds),
                "roc_auc_mean": roc_auc_mean,
                "roc_auc_std": roc_auc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "macro_f1_mean": macro_f1_mean,
                "macro_f1_std": macro_f1_std,
                "delta_roc_auc_vs_base": 0,
                "delta_auprc_vs_base": 0,
                "delta_f1_vs_base": 0,
                "delta_macro_f1_vs_base": 0,
                "seed_list": qwen_seeds,
            })

    table_dir = ensure_dir(Path("artifacts") / "tables")

    if metrics_rows:
        fieldnames = list(metrics_rows[0].keys())
    else:
        fieldnames = []

    with open(table_dir / "controlled_experiments_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    any_calibrated = any(r.get("has_calibrated") for r in metrics_rows)

    md_header = "| Dataset | Method | Seeds | ROC-AUC | AUPRC | F1 | Macro-F1"
    md_sep = "|---------|--------|-------|---------|-------|-----|----------"
    if any_calibrated:
        md_header += " | F1@0.5 | F1@val-F1 | Macro-F1@0.5 | Macro-F1@val | Threshold"
        md_sep += " |--------|-----------|--------------|-------------|---------"
    md_header += " | Δ ROC-AUC | Δ AUPRC | Δ F1 | Δ Macro-F1 |"
    md_sep += " |-----------|---------|------|------------|"

    md_lines = [
        "# Controlled Experiments Metrics",
        "",
        "## Full Comparison (all available seeds)",
        "",
        md_header,
        md_sep,
    ]

    for row in metrics_rows:
        delta_roc = row.get("delta_roc_auc_vs_base", "")
        delta_auprc = row.get("delta_auprc_vs_base", "")
        delta_f1 = row.get("delta_f1_vs_base", "")
        delta_macro = row.get("delta_macro_f1_vs_base", "")

        delta_roc_str = f"{delta_roc:+.4f}" if isinstance(delta_roc, (int, float)) and delta_roc is not None else "N/A*"
        delta_auprc_str = f"{delta_auprc:+.4f}" if isinstance(delta_auprc, (int, float)) and delta_auprc is not None else "N/A*"
        delta_f1_str = f"{delta_f1:+.4f}" if isinstance(delta_f1, (int, float)) and delta_f1 is not None else "N/A*"
        delta_macro_str = f"{delta_macro:+.4f}" if isinstance(delta_macro, (int, float)) and delta_macro is not None else "N/A*"

        split_warn = ""
        if row.get("split_warning") == "non-stratified":
            split_warn = " ⚠️non-strat"
        elif row.get("split_warning") == "unknown":
            split_warn = " ⚠️split?"

        line = (
            f"| {row['dataset']} | {row['method']}{split_warn} | {row['seeds']} | "
            f"{row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f} | "
            f"{row['auprc_mean']:.4f}±{row['auprc_std']:.4f} | "
            f"{row['f1_mean']:.4f}±{row['f1_std']:.4f} | "
            f"{row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} |"
        )

        if any_calibrated:
            if row.get("has_calibrated"):
                line += (
                    f" {row['f1_fixed_mean']:.4f}±{row['f1_fixed_std']:.4f} | "
                    f"{row['f1_val_f1_mean']:.4f}±{row['f1_val_f1_std']:.4f} | "
                    f"{row['macro_f1_fixed_mean']:.4f}±{row['macro_f1_fixed_std']:.4f} | "
                    f"{row['macro_f1_val_macro_mean']:.4f}±{row['macro_f1_val_macro_std']:.4f} | "
                    f"{row['threshold_used_mean']:.4f} |"
                )
            else:
                line += " — | — | — | — | — |"

        line += f" {delta_roc_str} | {delta_auprc_str} | {delta_f1_str} | {delta_macro_str} |"
        md_lines.append(line)

    if fair_comparison_rows:
        fair_header = "| Dataset | Method | Seeds | ROC-AUC | AUPRC | F1 | Macro-F1"
        fair_sep = "|---------|--------|-------|---------|-------|-----|----------"
        if any_calibrated:
            fair_header += " | F1@0.5 | F1@val-F1 | Macro-F1@0.5 | Macro-F1@val"
            fair_sep += " |--------|-----------|--------------|-------------"
        fair_header += " | Δ ROC-AUC | Δ AUPRC | Δ F1 | Δ Macro-F1 |"
        fair_sep += " |-----------|---------|------|------------|"

        md_lines.extend([
            "",
            "## Fair Same-Seed Comparison (BWGNN vs CoVER-BWGNN-Qwen on identical seeds)",
            "",
            fair_header,
            fair_sep,
        ])

        for row in fair_comparison_rows:
            line = (
                f"| {row['dataset']} | {row['method']} | {row['seeds']} | "
                f"{row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f} | "
                f"{row['auprc_mean']:.4f}±{row['auprc_std']:.4f} | "
                f"{row['f1_mean']:.4f}±{row['f1_std']:.4f} | "
                f"{row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} |"
            )
            if any_calibrated:
                line += " — | — | — | — |"
            line += " |  |  |  |  |"
            md_lines.append(line)

        qwen_row = next((r for r in metrics_rows if r["method"] == "CoVER-BWGNN-Qwen"), None)
        base_same_seeds_row = next((r for r in fair_comparison_rows if r["method"] == "BWGNN (same seeds)"), None)
        if qwen_row and base_same_seeds_row:
            delta_roc = qwen_row["roc_auc_mean"] - base_same_seeds_row["roc_auc_mean"]
            delta_auprc = qwen_row["auprc_mean"] - base_same_seeds_row["auprc_mean"]
            delta_f1 = qwen_row["f1_mean"] - base_same_seeds_row["f1_mean"]
            delta_macro = qwen_row["macro_f1_mean"] - base_same_seeds_row["macro_f1_mean"]

            line = (
                f"| {qwen_row['dataset']} | {qwen_row['method']} | {qwen_row['seeds']} | "
                f"{qwen_row['roc_auc_mean']:.4f}±{qwen_row['roc_auc_std']:.4f} | "
                f"{qwen_row['auprc_mean']:.4f}±{qwen_row['auprc_std']:.4f} | "
                f"{qwen_row['f1_mean']:.4f}±{qwen_row['f1_std']:.4f} | "
                f"{qwen_row['macro_f1_mean']:.4f}±{qwen_row['macro_f1_std']:.4f} |"
            )
            if any_calibrated and qwen_row.get("has_calibrated"):
                line += (
                    f" {qwen_row['f1_fixed_mean']:.4f}±{qwen_row['f1_fixed_std']:.4f} | "
                    f"{qwen_row['f1_val_f1_mean']:.4f}±{qwen_row['f1_val_f1_std']:.4f} | "
                    f"{qwen_row['macro_f1_fixed_mean']:.4f}±{qwen_row['macro_f1_fixed_std']:.4f} | "
                    f"{qwen_row['macro_f1_val_macro_mean']:.4f}±{qwen_row['macro_f1_val_macro_std']:.4f} |"
                )
            elif any_calibrated:
                line += " — | — | — | — |"
            line += (
                f" {delta_roc:+.4f} | {delta_auprc:+.4f} | {delta_f1:+.4f} | {delta_macro:+.4f} |"
            )
            md_lines.append(line)

    notes = [
        "",
        "**Notes:**",
        "- Qwen row (3 seeds) excludes the two lowest-performing base seeds (42, 2026). See Fair Same-Seed Comparison for controlled delta.",
    ]
    if any_calibrated:
        notes.extend([
            "- **F1@0.5** and **F1@val-threshold** are different metrics. F1@0.5 uses a fixed 0.5 threshold; F1@val-F1 uses the threshold that maximized F1 on the validation set.",
            "- ROC-AUC and AUPRC are threshold-independent and do not change with threshold selection.",
        ])
    any_non_strat = any(r.get("split_warning") == "non-stratified" for r in metrics_rows)
    if any_non_strat:
        notes.append("- ⚠️ **non-stratified**: The train/val/test split was not stratified. Class distribution may vary across splits, which can affect F1/Macro-F1 comparability.")
    md_lines.extend(notes)

    with open(table_dir / "controlled_experiments_metrics.md", "w") as f:
        f.write("\n".join(md_lines))

    with open(table_dir / "evidence_quality_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=evidence_rows[0].keys() if evidence_rows else [])
        writer.writeheader()
        writer.writerows(evidence_rows)

    md_lines = [
        "# Evidence Quality Summary",
        "",
        "| Dataset | Teacher | Seeds | Parse Rate | Acceptance Rate | Retry Rate | Weak/Uncertain Ratio | Top Risk Types | Top Supporting | Main Reject Reasons |",
        "|---------|---------|-------|------------|-----------------|------------|---------------------|----------------|----------------|---------------------|",
    ]

    for row in evidence_rows:
        top_risk_types = row.get('top_risk_types', '') or "N/A (rule-based)" if row['teacher'] == 'rule' else row.get('top_risk_types', '')
        top_supporting = row.get('top_supporting_evidence', '') or "N/A (rule-based)" if row['teacher'] == 'rule' else row.get('top_supporting_evidence', '')
        main_reject = row.get('main_reject_reasons', '') or "N/A (rule-based)" if row['teacher'] == 'rule' else row.get('main_reject_reasons', '')

        md_lines.append(
            f"| {row['dataset']} | {row['teacher']} | {row['seeds']} | "
            f"{row['parse_success_rate_mean']:.2%}±{row['parse_success_rate_std']:.2%} | "
            f"{row['acceptance_rate_mean']:.2%}±{row['acceptance_rate_std']:.2%} | "
            f"{row['accepted_after_retry_rate_mean']:.2%}±{row['accepted_after_retry_rate_std']:.2%} | "
            f"{row['weak_or_uncertain_ratio_mean']:.2%}±{row['weak_or_uncertain_ratio_std']:.2%} | "
            f"{top_risk_types} | "
            f"{top_supporting} | "
            f"{main_reject} |"
        )

    with open(table_dir / "evidence_quality_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'='*60}")
    print("Aggregation Complete")
    print(f"{'='*60}")
    print(f"\nMetrics table: {table_dir / 'controlled_experiments_metrics.md'}")
    print(f"Evidence table: {table_dir / 'evidence_quality_summary.md'}")

    print(f"\nMetrics Summary:")
    for row in metrics_rows:
        print(f"  {row['dataset']}/{row['method']}: ROC-AUC={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}")


if __name__ == "__main__":
    main()
