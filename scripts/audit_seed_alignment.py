from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import ensure_dir, get_err_cache_dir, get_reports_dir, get_results_dir, get_stratified_split_meta_path


BASE_METHOD = "base_strat"
RULE_METHOD = "rule_safe"
QWEN_METHOD = "qwen_safe"
PAIR_METHODS = {
    "base_rule": (BASE_METHOD, RULE_METHOD),
    "base_qwen": (BASE_METHOD, QWEN_METHOD),
    "rule_qwen": (RULE_METHOD, QWEN_METHOD),
}


def stage_metrics_name(method: str) -> str:
    return "stage1_metrics.json" if method == BASE_METHOD else "stage3_metrics.json"


def format_seed_list(seeds: list[int]) -> str:
    return ", ".join(str(seed) for seed in seeds) if seeds else "none"


def file_check(path: Path) -> bool:
    return path.is_file()


def evidence_report_candidates(dataset: str, model: str, method: str, seed: int) -> list[Path]:
    return [
        Path("artifacts") / "reports" / dataset / model / method / f"seed_{seed}" / "evidence_quality_report.json",
        get_reports_dir(dataset, model, seed) / "evidence_quality_report.json",
        Path("artifacts") / "reports" / dataset / model / f"seed_{seed}" / "evidence_quality_report.json",
    ]


def first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if file_check(candidate):
            return candidate
    return None


def check_run(dataset: str, model: str, method: str, seed: int) -> dict[str, Any]:
    missing_files: list[str] = []

    split_meta_path = get_stratified_split_meta_path(dataset, True, seed)
    if not file_check(split_meta_path):
        missing_files.append("split_meta.json")

    results_path = get_results_dir(dataset, model, method, seed) / stage_metrics_name(method)
    if not file_check(results_path):
        missing_files.append(results_path.name)

    evidence_path = None
    verifier_stats_path = None
    if method in {RULE_METHOD, QWEN_METHOD}:
        err_cache_dir = get_err_cache_dir(dataset, model, method, seed)
        verifier_stats_path = err_cache_dir / "verifier_stats.json"
        if not file_check(verifier_stats_path):
            missing_files.append("verifier_stats.json")

        primary_report = err_cache_dir / "evidence_quality_report.json"
        if file_check(primary_report):
            evidence_path = primary_report
        else:
            evidence_path = first_existing_path(evidence_report_candidates(dataset, model, method, seed))
            if evidence_path is None:
                missing_files.append("evidence_quality_report.json")

    return {
        "dataset": dataset,
        "model": model,
        "method": method,
        "seed": seed,
        "complete": not missing_files,
        "missing_files": missing_files,
        "split_meta_path": str(split_meta_path),
        "results_path": str(results_path),
        "evidence_quality_report_path": str(evidence_path) if evidence_path else None,
        "verifier_stats_path": str(verifier_stats_path) if verifier_stats_path else None,
    }


def summarize_scope(label: str, datasets: list[str], methods: list[str], seeds: list[int], model: str) -> dict[str, Any]:
    run_checks: dict[tuple[str, str, int], dict[str, Any]] = {}
    warnings: list[str] = []

    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                result = check_run(dataset, model, method, seed)
                run_checks[(dataset, method, seed)] = result
                if result["missing_files"]:
                    missing = ", ".join(result["missing_files"])
                    warnings.append(f"{dataset}/{method}/seed_{seed}: missing {missing}")

    available_seeds: dict[str, list[int]] = {}
    missing_seeds: dict[str, list[int]] = {}
    for method in methods:
        complete_by_dataset: list[set[int]] = []
        for dataset in datasets:
            complete_by_dataset.append(
                {
                    seed
                    for seed in seeds
                    if run_checks[(dataset, method, seed)]["complete"]
                }
            )

        if complete_by_dataset:
            complete = set(seeds)
            for seed_set in complete_by_dataset:
                complete &= seed_set
        else:
            complete = set()

        available_seeds[method] = sorted(complete)
        missing_seeds[method] = [seed for seed in seeds if seed not in complete]

    common_seeds: dict[str, list[int]] = {}
    can_compute_fair_delta: dict[str, bool] = {}
    for pair_name, (left, right) in PAIR_METHODS.items():
        if left in available_seeds and right in available_seeds:
            common = sorted(set(available_seeds[left]) & set(available_seeds[right]))
        else:
            common = []
        common_seeds[pair_name] = common
        can_compute_fair_delta[pair_name] = bool(common)
        if not common:
            warnings.append(f"{label}: cannot compute fair delta for {pair_name}")

    common_seeds_all = sorted(
        set(available_seeds.get(BASE_METHOD, []))
        & set(available_seeds.get(RULE_METHOD, []))
        & set(available_seeds.get(QWEN_METHOD, []))
    )

    if not common_seeds_all:
        warnings.append(f"{label}: no common seeds across all three methods")

    if any(m not in available_seeds for m in (BASE_METHOD, RULE_METHOD, QWEN_METHOD)):
        warnings.append(f"{label}: expected methods base_strat, rule_safe, qwen_safe were not all provided")

    return {
        "scope": label,
        "datasets": datasets,
        "methods": methods,
        "seeds": seeds,
        "available_seeds": available_seeds,
        "missing_seeds": missing_seeds,
        "common_seeds_base_rule": common_seeds["base_rule"],
        "common_seeds_base_qwen": common_seeds["base_qwen"],
        "common_seeds_all": common_seeds_all,
        "can_compute_fair_delta": can_compute_fair_delta,
        "warnings": warnings,
    }


def render_scope_md(title: str, summary: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append(f"- Datasets: {', '.join(summary['datasets'])}")
    lines.append(f"- Methods: {', '.join(summary['methods'])}")
    lines.append(f"- Seeds checked: {format_seed_list(summary['seeds'])}")
    lines.append("")
    lines.append("| Method | Available seeds | Missing seeds |")
    lines.append("|---|---|---|")
    for method in summary["methods"]:
        lines.append(
            f"| {method} | {format_seed_list(summary['available_seeds'].get(method, []))} | {format_seed_list(summary['missing_seeds'].get(method, []))} |"
        )
    lines.append("")
    lines.append(f"- common_seeds_base_rule: {format_seed_list(summary['common_seeds_base_rule'])}")
    lines.append(f"- common_seeds_base_qwen: {format_seed_list(summary['common_seeds_base_qwen'])}")
    lines.append(f"- common_seeds_all: {format_seed_list(summary['common_seeds_all'])}")
    lines.append("")
    lines.append("| Pair | Can compute fair delta |")
    lines.append("|---|---|")
    for pair_name, can_compute in summary["can_compute_fair_delta"].items():
        lines.append(f"| {pair_name} | {'Yes' if can_compute else 'No'} |")
    lines.append("")
    lines.append("### Warnings")
    if summary["warnings"]:
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit seed alignment across CoVER-FD experiment artifacts.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Datasets to audit")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--methods", nargs="+", required=True, help="Methods/run names to audit")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Seeds to audit")
    args = parser.parse_args()

    by_dataset = {
        dataset: summarize_scope(dataset, [dataset], args.methods, args.seeds, args.model)
        for dataset in args.datasets
    }
    overall = summarize_scope("overall", args.datasets, args.methods, args.seeds, args.model)

    warnings = list(dict.fromkeys(overall["warnings"] + [w for summary in by_dataset.values() for w in summary["warnings"]]))

    report = {
        "datasets": args.datasets,
        "model": args.model,
        "methods": args.methods,
        "seeds": args.seeds,
        "available_seeds": overall["available_seeds"],
        "missing_seeds": overall["missing_seeds"],
        "common_seeds_base_rule": overall["common_seeds_base_rule"],
        "common_seeds_base_qwen": overall["common_seeds_base_qwen"],
        "common_seeds_all": overall["common_seeds_all"],
        "can_compute_fair_delta": overall["can_compute_fair_delta"],
        "warnings": warnings,
        "by_dataset": by_dataset,
    }

    report_dir = ensure_dir(Path("artifacts") / "reports")
    json_path = report_dir / "seed_alignment_audit.json"
    md_path = report_dir / "seed_alignment_audit.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_lines = ["# Seed Alignment Audit", ""]
    md_lines.append("**Overall summary counts a seed only when it is complete for that method in every requested dataset.**")
    md_lines.append("")
    md_lines.append(f"- Datasets: {', '.join(args.datasets)}")
    md_lines.append(f"- Model: {args.model}")
    md_lines.append(f"- Methods: {', '.join(args.methods)}")
    md_lines.append(f"- Seeds checked: {format_seed_list(args.seeds)}")
    md_lines.append("")
    md_lines.append("## Overall")
    md_lines.append("")
    md_lines.append("| Method | Available seeds | Missing seeds |")
    md_lines.append("|---|---|---|")
    for method in args.methods:
        md_lines.append(
            f"| {method} | {format_seed_list(overall['available_seeds'].get(method, []))} | {format_seed_list(overall['missing_seeds'].get(method, []))} |"
        )
    md_lines.append("")
    md_lines.append(f"- common_seeds_base_rule: {format_seed_list(overall['common_seeds_base_rule'])}")
    md_lines.append(f"- common_seeds_base_qwen: {format_seed_list(overall['common_seeds_base_qwen'])}")
    md_lines.append(f"- common_seeds_all: {format_seed_list(overall['common_seeds_all'])}")
    md_lines.append("")
    md_lines.append("| Pair | Can compute fair delta |")
    md_lines.append("|---|---|")
    for pair_name, can_compute in overall["can_compute_fair_delta"].items():
        md_lines.append(f"| {pair_name} | {'Yes' if can_compute else 'No'} |")
    md_lines.append("")
    md_lines.append("### Warnings")
    if warnings:
        for warning in warnings:
            md_lines.append(f"- {warning}")
    else:
        md_lines.append("- none")
    md_lines.append("")

    for dataset, summary in by_dataset.items():
        md_lines.extend(render_scope_md(f"Dataset: {dataset}", summary))

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Audit complete for {len(args.datasets)} dataset(s).")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
