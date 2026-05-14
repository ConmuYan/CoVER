"""Aggregate results from controlled experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get_results_dir, get_err_cache_dir, get_reports_dir, get_split_path, ensure_dir


# ---------------------------------------------------------------------------
# Teacher / method configuration
# ---------------------------------------------------------------------------
# Each entry:  CLI name → (results_dir_name, metrics_file, display_name, is_base)
TEACHER_CONFIG: dict[str, tuple[str, str, str, bool]] = {
    # Backward-compatible short names
    "base":   ("base",       "stage1_metrics.json", "BWGNN",             True),
    "rule":   ("rule",       "stage3_metrics.json", "CoVER-Rule-Safe",   False),
    "qwen":   ("qwen",       "stage3_metrics.json", "CoVER-Qwen-Safe",   False),
    # New explicit names
    "base_strat": ("base_strat", "stage1_metrics.json", "BWGNN",                True),
    "rule_safe":  ("rule_safe",  "stage3_metrics.json", "CoVER-Rule-Safe",      False),
    "qwen_safe":  ("qwen_safe",  "stage3_metrics.json", "CoVER-Qwen-Safe",     False),
}

# Which calibrated metrics file corresponds to each entry
TEACHER_CALIBRATED_FILE: dict[str, str] = {
    "base":        "stage1_calibrated_metrics.json",
    "rule":        "stage3_calibrated_metrics.json",
    "qwen":        "stage3_calibrated_metrics.json",
    "base_strat":  "stage1_calibrated_metrics.json",
    "rule_safe":   "stage3_calibrated_metrics.json",
    "qwen_safe":   "stage3_calibrated_metrics.json",
}


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


def _collect_per_seed_metrics(
    dataset: str,
    teacher: str,
    seeds: list[int],
    model: str,
    all_metrics: dict,
    all_calibrated: dict,
    all_split_meta: dict,
) -> dict:
    """Collect per-seed metric values for a single (dataset, teacher) pair.

    Returns dict with lists: roc_aucs, auprcs, f1s, macro_f1s,
    f1_fixeds, f1_val_f1s, macro_f1_fixeds, macro_f1_val_macros,
    thresholds_used, seeds_found, has_calibrated, has_stratified_info,
    any_non_stratified, per_seed (dict seed→{metric: value}).
    """
    dir_name, _, _, _is_base = TEACHER_CONFIG[teacher]

    seeds_found: list[int] = []
    roc_aucs: list[float] = []
    auprcs: list[float] = []
    f1s: list[float] = []
    macro_f1s: list[float] = []
    f1_fixeds: list[float] = []
    f1_val_f1s: list[float] = []
    macro_f1_fixeds: list[float] = []
    macro_f1_val_macros: list[float] = []
    thresholds_used: list[float] = []
    has_calibrated = False
    has_stratified_info = True
    any_non_stratified = False
    per_seed: dict[int, dict[str, float]] = {}

    for seed in seeds:
        key = (dataset, teacher, seed)
        if key not in all_metrics:
            continue

        m = all_metrics[key]
        seeds_found.append(seed)
        roc_aucs.append(m.get("roc_auc", 0))
        auprcs.append(m.get("auprc", 0))
        f1s.append(m.get("f1", 0))
        macro_f1s.append(m.get("macro_f1", 0))

        per_seed[seed] = {
            "roc_auc": m.get("roc_auc", 0),
            "auprc": m.get("auprc", 0),
            "f1": m.get("f1", 0),
            "macro_f1": m.get("macro_f1", 0),
        }

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

            per_seed[seed]["f1_fixed"] = fixed.get("f1", 0)
            per_seed[seed]["f1_val_f1"] = val_f1.get("f1", 0)
            per_seed[seed]["macro_f1_fixed"] = fixed.get("macro_f1", 0)
            per_seed[seed]["macro_f1_val_macro"] = val_macro.get("macro_f1", 0)
            per_seed[seed]["threshold_used"] = val_f1.get("threshold_used", 0.5)

        split_meta = all_split_meta.get((dataset, seed))
        if split_meta is not None:
            if not split_meta.get("stratified", True):
                any_non_stratified = True
        else:
            has_stratified_info = False

    return {
        "seeds_found": seeds_found,
        "roc_aucs": roc_aucs,
        "auprcs": auprcs,
        "f1s": f1s,
        "macro_f1s": macro_f1s,
        "f1_fixeds": f1_fixeds,
        "f1_val_f1s": f1_val_f1s,
        "macro_f1_fixeds": macro_f1_fixeds,
        "macro_f1_val_macros": macro_f1_val_macros,
        "thresholds_used": thresholds_used,
        "has_calibrated": has_calibrated,
        "has_stratified_info": has_stratified_info,
        "any_non_stratified": any_non_stratified,
        "per_seed": per_seed,
    }


def _build_metric_row(
    dataset: str,
    model: str,
    method_name: str,
    collected: dict,
) -> dict:
    """Build a single metrics row dict from collected per-seed data."""
    roc_auc_mean, roc_auc_std = compute_mean_std(collected["roc_aucs"])
    auprc_mean, auprc_std = compute_mean_std(collected["auprcs"])
    f1_mean, f1_std = compute_mean_std(collected["f1s"])
    macro_f1_mean, macro_f1_std = compute_mean_std(collected["macro_f1s"])

    row: dict = {
        "dataset": dataset,
        "model": model,
        "method": method_name,
        "seeds": len(collected["seeds_found"]),
        "roc_auc_mean": roc_auc_mean,
        "roc_auc_std": roc_auc_std,
        "auprc_mean": auprc_mean,
        "auprc_std": auprc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "macro_f1_mean": macro_f1_mean,
        "macro_f1_std": macro_f1_std,
    }

    if collected["has_calibrated"]:
        f1_fixed_mean, f1_fixed_std = compute_mean_std(collected["f1_fixeds"])
        f1_val_f1_mean, f1_val_f1_std = compute_mean_std(collected["f1_val_f1s"])
        macro_f1_fixed_mean, macro_f1_fixed_std = compute_mean_std(collected["macro_f1_fixeds"])
        macro_f1_val_macro_mean, macro_f1_val_macro_std = compute_mean_std(collected["macro_f1_val_macros"])
        threshold_mean, _ = compute_mean_std(collected["thresholds_used"])
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

    if collected["has_stratified_info"] and collected["any_non_stratified"]:
        row["split_warning"] = "non-stratified"
    elif not collected["has_stratified_info"]:
        row["split_warning"] = "unknown"
    else:
        row["split_warning"] = ""

    row["has_calibrated"] = collected["has_calibrated"]
    return row


def _add_deltas_vs_base(
    metrics_rows: list[dict],
    base_display_name: str = "BWGNN",
) -> None:
    """Add delta columns comparing each row to the base row (in-place)."""
    for row in metrics_rows:
        if row["method"] == base_display_name:
            row["delta_roc_auc_vs_base"] = 0
            row["delta_auprc_vs_base"] = 0
            row["delta_f1_vs_base"] = 0
            row["delta_macro_f1_vs_base"] = 0
        else:
            base_row = next(
                (r for r in metrics_rows
                 if r["dataset"] == row["dataset"] and r["method"] == base_display_name),
                None,
            )
            if base_row:
                row["delta_roc_auc_vs_base"] = row["roc_auc_mean"] - base_row["roc_auc_mean"]
                row["delta_auprc_vs_base"] = row["auprc_mean"] - base_row["auprc_mean"]
                row["delta_f1_vs_base"] = row["f1_mean"] - base_row["f1_mean"]
                row["delta_macro_f1_vs_base"] = row["macro_f1_mean"] - base_row["macro_f1_mean"]
            else:
                row["delta_roc_auc_vs_base"] = None
                row["delta_auprc_vs_base"] = None
                row["delta_f1_vs_base"] = None
                row["delta_macro_f1_vs_base"] = None


def _write_metrics_table_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        with open(path, "w"):
            pass
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_table_md(
    rows: list[dict],
    path: Path,
    title: str,
    extra_notes: list[str] | None = None,
    force_calibrated_columns: bool = False,
    force_val_delta_columns: bool = False,
) -> None:
    if not rows:
        md_lines = [f"# {title}", "", "_No rows available._"]
        with open(path, "w") as f:
            f.write("\n".join(md_lines))
        return

    any_calibrated = force_calibrated_columns or any(r.get("has_calibrated") for r in rows)
    has_seed_list = any("seed_list" in r for r in rows)
    use_num_seeds = any("num_seeds" in r for r in rows)
    use_val_deltas = force_val_delta_columns or any("delta_f1_val_vs_base" in r or "delta_macro_f1_val_vs_base" in r for r in rows)

    seed_label = "Num Seeds" if use_num_seeds else "Seeds"
    header_cols = ["Dataset", "Method", seed_label]
    if has_seed_list:
        header_cols.append("Seed List")
    header_cols.extend(["ROC-AUC", "AUPRC", "F1", "Macro-F1"])
    if any_calibrated:
        header_cols.extend(["F1@0.5", "F1@val-threshold", "Macro-F1@0.5", "Macro-F1@val-threshold", "Threshold"])
    if use_val_deltas:
        header_cols.extend(["Δ ROC-AUC", "Δ AUPRC", "Δ F1@val vs base", "Δ Macro-F1@val vs base"])
        delta_keys = ["delta_roc_auc_vs_base", "delta_auprc_vs_base", "delta_f1_val_vs_base", "delta_macro_f1_val_vs_base"]
    else:
        header_cols.extend(["Δ ROC-AUC", "Δ AUPRC", "Δ F1", "Δ Macro-F1"])
        delta_keys = ["delta_roc_auc_vs_base", "delta_auprc_vs_base", "delta_f1_vs_base", "delta_macro_f1_vs_base"]

    md_header = "| " + " | ".join(header_cols) + " |"
    md_sep = "| " + " | ".join("-" * max(len(col), 3) for col in header_cols) + " |"

    md_lines = [f"# {title}", "", md_header, md_sep]

    for row in rows:
        delta_vals = [row.get(k, "") for k in delta_keys]

        delta_strs = [
            f"{value:+.4f}" if isinstance(value, (int, float)) and value is not None else "N/A*"
            for value in delta_vals
        ]

        split_warn = ""
        if row.get("split_warning") == "non-stratified":
            split_warn = " ⚠️non-strat"
        elif row.get("split_warning") == "unknown":
            split_warn = " ⚠️split?"

        seed_value = row.get("num_seeds", row.get("seeds", 0))
        cells = [
            str(row["dataset"]),
            f"{row['method']}{split_warn}",
            str(seed_value),
        ]
        if has_seed_list:
            cells.append(str(row.get("seed_list", "")))
        cells.extend([
            f"{row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}",
            f"{row['auprc_mean']:.4f}±{row['auprc_std']:.4f}",
            f"{row['f1_mean']:.4f}±{row['f1_std']:.4f}",
            f"{row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}",
        ])

        if any_calibrated:
            if row.get("has_calibrated"):
                cells.extend([
                    f"{row['f1_fixed_mean']:.4f}±{row['f1_fixed_std']:.4f}",
                    f"{row['f1_val_f1_mean']:.4f}±{row['f1_val_f1_std']:.4f}",
                    f"{row['macro_f1_fixed_mean']:.4f}±{row['macro_f1_fixed_std']:.4f}",
                    f"{row['macro_f1_val_macro_mean']:.4f}±{row['macro_f1_val_macro_std']:.4f}",
                    f"{row['threshold_used_mean']:.4f}",
                ])
            else:
                cells.extend(["—", "—", "—", "—", "—"])

        cells.extend(delta_strs)
        line = "| " + " | ".join(cells) + " |"
        md_lines.append(line)

    if extra_notes:
        md_lines.append("")
        for note in extra_notes:
            md_lines.append(note)

    with open(path, "w") as f:
        f.write("\n".join(md_lines))


def _write_evidence_quality(
    evidence_rows: list[dict],
    table_dir: Path,
) -> None:
    """Write evidence quality summary CSV and MD."""
    _write_evidence_quality_csv(evidence_rows, table_dir / "final_evidence_quality_summary.csv")
    _write_evidence_quality_md(evidence_rows, table_dir / "final_evidence_quality_summary.md")


def _write_evidence_quality_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        with open(path, "w"):
            pass
        return
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_evidence_quality_md(rows: list[dict], path: Path) -> None:
    if not rows:
        with open(path, "w") as f:
            f.write("# Evidence Quality Summary\n\n_No rows available._\n")
        return
    md_lines = [
        "# Evidence Quality Summary",
        "",
        "| Dataset | Teacher | Seeds | Parse Rate | Acceptance Rate | Retry Rate | Weak/Uncertain Ratio | Top Risk Types | Top Supporting | Main Reject Reasons |",
        "|---------|---------|-------|------------|-----------------|------------|---------------------|----------------|----------------|---------------------|",
    ]

    for row in rows:
        is_rule_based = bool(row.get("is_rule_based", False))
        top_risk_types = row.get("top_risk_types", "") or ("N/A (rule-based)" if is_rule_based else "")
        top_supporting = row.get("top_supporting_evidence", "") or ("N/A (rule-based)" if is_rule_based else "")
        main_reject = row.get("main_reject_reasons", "") or ("N/A (rule-based)" if is_rule_based else "")

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

    with open(path, "w") as f:
        f.write("\n".join(md_lines))


def _build_same_seed_rows(
    dataset: str,
    model: str,
    base_teacher: str,
    method_teachers: list[str],
    all_metrics: dict,
    all_calibrated: dict,
    all_split_meta: dict,
    seeds: list[int],
) -> list[dict]:
    """Build same-seed comparison rows with fair delta computation.

    For each method, find common seeds with base, compute metrics on those
        common seeds only, and compute per-seed deltas from calibrated val-threshold
        metrics.
    """
    # Identify base seeds that have data
    base_collected = _collect_per_seed_metrics(
        dataset, base_teacher, seeds, model, all_metrics, all_calibrated, all_split_meta,
    )
    base_seeds_set = set(base_collected["seeds_found"])

    if not base_seeds_set:
        return []

    rows: list[dict] = []

    for method_teacher in method_teachers:
        _, _, method_display, _ = TEACHER_CONFIG[method_teacher]
        method_collected = _collect_per_seed_metrics(
            dataset, method_teacher, seeds, model, all_metrics, all_calibrated, all_split_meta,
        )
        method_seeds_set = set(method_collected["seeds_found"])

        if not method_seeds_set:
            continue

        # Find common seeds
        common_seeds = sorted(base_seeds_set & method_seeds_set)

        if not common_seeds:
            continue

        # Compute base metrics on common seeds only
        base_common_collected = _collect_per_seed_metrics(
            dataset, base_teacher, common_seeds, model,
            all_metrics, all_calibrated, all_split_meta,
        )

        # Compute method metrics on common seeds only
        method_common_collected = _collect_per_seed_metrics(
            dataset, method_teacher, common_seeds, model,
            all_metrics, all_calibrated, all_split_meta,
        )

        # Per-seed deltas for calibrated metrics
        base_per = base_common_collected["per_seed"]
        method_per = method_common_collected["per_seed"]

        calibrated_common_seeds = [
            s for s in common_seeds
            if all(
                k in base_per.get(s, {}) and k in method_per.get(s, {})
                for k in ("f1_val_f1", "macro_f1_val_macro")
            )
        ]

        # Compute per-seed deltas for standard metrics
        delta_roc_aucs = [
            method_per[s]["roc_auc"] - base_per[s]["roc_auc"]
            for s in common_seeds if s in method_per and s in base_per
        ]
        delta_auprcs = [
            method_per[s]["auprc"] - base_per[s]["auprc"]
            for s in common_seeds if s in method_per and s in base_per
        ]
        delta_f1s = [
            method_per[s]["f1_val_f1"] - base_per[s]["f1_val_f1"]
            for s in calibrated_common_seeds
        ]
        delta_macro_f1s = [
            method_per[s]["macro_f1_val_macro"] - base_per[s]["macro_f1_val_macro"]
            for s in calibrated_common_seeds
        ]

        method_row = _build_metric_row(dataset, model, method_display, method_common_collected)
        method_row["num_seeds"] = len(common_seeds)
        method_row["seed_list"] = ", ".join(str(seed) for seed in common_seeds)

        delta_roc_mean, _ = compute_mean_std(delta_roc_aucs)
        delta_auprc_mean, _ = compute_mean_std(delta_auprcs)
        delta_f1_mean, _ = compute_mean_std(delta_f1s)
        delta_macro_f1_mean, _ = compute_mean_std(delta_macro_f1s)

        method_row["delta_roc_auc_vs_base"] = delta_roc_mean
        method_row["delta_auprc_vs_base"] = delta_auprc_mean
        method_row["delta_f1_val_vs_base"] = delta_f1_mean if delta_f1s else None
        method_row["delta_macro_f1_val_vs_base"] = delta_macro_f1_mean if delta_macro_f1s else None

        rows.append(method_row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["yelpchi", "amazon"])
    parser.add_argument("--model", type=str, default="bwgnn")
    parser.add_argument(
        "--teachers", nargs="+",
        default=["base", "rule", "qwen"],
        help="Teacher/method names. Supports backward-compat 'base rule qwen' and explicit names 'base_strat rule_safe qwen_safe'.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789, 42, 2026])
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument(
        "--fair_same_seed",
        action="store_true",
        default=False,
        help="Compute fair same-seed deltas and generate separate comparison tables.",
    )
    args = parser.parse_args()

    # Validate teacher names
    for t in args.teachers:
        if t not in TEACHER_CONFIG:
            print(f"Error: Unknown teacher/method name '{t}'. "
                  f"Valid: {sorted(TEACHER_CONFIG.keys())}")
            sys.exit(1)

    # Identify which teacher is "base" for delta computation
    base_teacher = None
    for t in args.teachers:
        _, _, _, is_base = TEACHER_CONFIG[t]
        if is_base:
            base_teacher = t
            break

    method_teachers = [t for t in args.teachers if t != base_teacher]

    # ------------------------------------------------------------------
    # Phase 1: Load all data
    # ------------------------------------------------------------------
    all_metrics: dict[tuple, dict] = {}
    all_calibrated: dict[tuple, dict] = {}
    all_split_meta: dict[tuple, dict] = {}
    all_evidence: dict[tuple, dict] = {}
    missing: list[tuple] = []

    for dataset in args.datasets:
        for teacher in args.teachers:
            dir_name, metrics_file, _, is_base = TEACHER_CONFIG[teacher]
            cal_file = TEACHER_CALIBRATED_FILE[teacher]

            for seed in args.seeds:
                results_dir = get_results_dir(dataset, args.model, dir_name, seed)
                metrics_path = results_dir / metrics_file
                cal_path = results_dir / cal_file

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

                if not is_base:
                    err_stats_path = get_err_cache_dir(dataset, args.model, dir_name, seed) / "stage2_stats.json"
                    err_stats = load_json(err_stats_path)
                    if err_stats:
                        key = (dataset, teacher, seed)
                        all_evidence[key] = err_stats

                    verifier_stats_path = get_err_cache_dir(dataset, args.model, dir_name, seed) / "verifier_stats.json"
                    verifier_stats = load_json(verifier_stats_path)
                    if verifier_stats and key in all_evidence:
                        all_evidence[key]["reject_reason_counts"] = verifier_stats.get("reject_reason_counts", {})

                    evidence_report_path = get_err_cache_dir(dataset, args.model, dir_name, seed) / "evidence_quality_report.json"
                    evidence_report = load_json(evidence_report_path)
                    if not evidence_report:
                        evidence_report_path = get_reports_dir(dataset, args.model, seed).parent / dir_name / f"seed_{seed}" / "evidence_quality_report.json"
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

    # ------------------------------------------------------------------
    # Phase 2: Build "full available" metrics rows
    # ------------------------------------------------------------------
    metrics_rows: list[dict] = []
    collected_per_method: dict[tuple[str, str], dict] = {}

    for dataset in args.datasets:
        for teacher in args.teachers:
            _, _, display_name, _ = TEACHER_CONFIG[teacher]
            collected = _collect_per_seed_metrics(
                dataset, teacher, args.seeds, args.model,
                all_metrics, all_calibrated, all_split_meta,
            )
            collected_per_method[(dataset, teacher)] = collected

            if collected["seeds_found"]:
                row = _build_metric_row(dataset, args.model, display_name, collected)
                metrics_rows.append(row)

    _add_deltas_vs_base(metrics_rows)

    # ------------------------------------------------------------------
    # Phase 3: Build evidence quality rows
    # ------------------------------------------------------------------
    evidence_rows: list[dict] = []
    for dataset in args.datasets:
        for teacher in method_teachers:
            parse_successes: list[float] = []
            acceptances: list[float] = []
            retry_rates: list[float] = []
            weak_ratios: list[float] = []
            risk_type_counts: dict[str, int] = {}
            supporting_counts: dict[str, int] = {}
            counter_counts: dict[str, int] = {}
            reject_reason_counts: dict[str, int] = {}

            for seed in args.seeds:
                key = (dataset, teacher, seed)
                if key not in all_evidence:
                    continue
                e = all_evidence[key]

                _, _, _, _is_base = TEACHER_CONFIG[teacher]
                # rule-based teachers always parse successfully
                dir_name = TEACHER_CONFIG[teacher][0]
                if "rule" in dir_name:
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
                    "teacher": display_name,
                    "teacher_key": teacher,
                    "is_rule_based": "rule" in dir_name,
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

    # ------------------------------------------------------------------
    # Phase 4: Write output files
    # ------------------------------------------------------------------
    table_dir = ensure_dir(Path("artifacts") / "tables")

    # --- Full available metrics table ---
    full_csv_path = table_dir / "final_controlled_metrics_full_available.csv"
    full_md_path = table_dir / "final_controlled_metrics_full_available.md"

    _write_metrics_table_csv(metrics_rows, full_csv_path)

    any_calibrated = any(r.get("has_calibrated") for r in metrics_rows)
    full_notes: list[str] = []
    full_notes.append("")
    full_notes.append("**Notes:**")
    if any_calibrated:
        full_notes.extend([
            "- **F1@0.5** and **F1@val-threshold** are different metrics. F1@0.5 uses a fixed 0.5 threshold; F1@val-threshold uses the threshold that maximized F1 on the validation set.",
            "- ROC-AUC and AUPRC are threshold-independent and do not change with threshold selection.",
        ])
    any_non_strat = any(r.get("split_warning") == "non-stratified" for r in metrics_rows)
    if any_non_strat:
        full_notes.append("- ⚠️ **non-stratified**: The train/val/test split was not stratified. Class distribution may vary across splits, which can affect F1/Macro-F1 comparability.")

    _write_metrics_table_md(metrics_rows, full_md_path, "Controlled Experiments Metrics (Full Available Seeds)", full_notes)

    # --- Same-seed table (only when --fair_same_seed) ---
    if args.fair_same_seed and base_teacher is not None:
        same_seed_csv_path = table_dir / "final_controlled_metrics_same_seed.csv"
        same_seed_md_path = table_dir / "final_controlled_metrics_same_seed.md"

        same_seed_rows: list[dict] = []
        for dataset in args.datasets:
            ds_rows = _build_same_seed_rows(
                dataset, args.model, base_teacher, method_teachers,
                all_metrics, all_calibrated, all_split_meta, args.seeds,
            )
            same_seed_rows.extend(ds_rows)

        if same_seed_rows:
            _write_metrics_table_csv(same_seed_rows, same_seed_csv_path)

            ss_notes: list[str] = []
            ss_notes.append("")
            ss_notes.append("**Notes:**")
            ss_notes.append("- This table uses **only common seeds** between each method and the base for fair delta computation.")
            ss_notes.append("- `seed_list` and `num_seeds` show exactly which seeds were used for that row.")
            if any_calibrated:
                ss_notes.extend([
                    "- **F1@0.5** uses a fixed 0.5 threshold; **F1@val-threshold** uses the threshold that maximized F1 on the validation set.",
                    "- ROC-AUC and AUPRC are threshold-independent.",
                    "- Δ F1 and Δ Macro-F1 columns are computed from the calibrated val-threshold metrics on the shared seed set.",
                ])
            any_non_strat_ss = any(r.get("split_warning") == "non-stratified" for r in same_seed_rows)
            if any_non_strat_ss:
                ss_notes.append("- ⚠️ **non-stratified**: The train/val/test split was not stratified.")

            _write_metrics_table_md(
                same_seed_rows,
                same_seed_md_path,
                "Fair Same-Seed Comparison",
                ss_notes,
                force_calibrated_columns=True,
                force_val_delta_columns=True,
            )

    # --- Evidence quality summary ---
    _write_evidence_quality(evidence_rows, table_dir)

    # ------------------------------------------------------------------
    # Phase 5: Summary printout
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Aggregation Complete")
    print(f"{'='*60}")
    print(f"\nFull metrics table: {full_md_path}")
    if args.fair_same_seed and base_teacher is not None:
        same_seed_md_path_check = table_dir / "final_controlled_metrics_same_seed.md"
        if same_seed_md_path_check.exists():
            print(f"Same-seed table:    {same_seed_md_path_check}")
    print(f"Evidence table:     {table_dir / 'final_evidence_quality_summary.md'}")

    print("\nMetrics Summary:")
    for row in metrics_rows:
        print(f"  {row['dataset']}/{row['method']}: ROC-AUC={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}")


if __name__ == "__main__":
    main()
