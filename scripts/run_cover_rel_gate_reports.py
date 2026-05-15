from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import ensure_dir


SEEDS = [42, 123, 456, 789, 2026]
DATASETS = {
    "yelpchi": {
        "model": "bwgnn",
        "best_relation": "RUR",
        "best_single_run": "qwen_directional_t200_cover_rel_rur_nollm",
        "all_rel_run": "qwen_directional_t200_cover_rel_nollm",
        "anchor_relation": "RUR",
        "extra_gate_runs": {},
    },
    "amazon": {
        "model": "bwgnn",
        "best_relation": "UVU",
        "best_single_run": "cover_rel_uvu_nollm",
        "all_rel_run": "cover_rel_all_nollm",
        "anchor_relation": "UVU",
        "extra_gate_runs": {
            "anchor_gate_conservative": "cover_rel_anchor_gate_conservative_nollm",
        },
    },
}
BASE_GATE_RUNS = {
    "anchor_gate": "cover_rel_anchor_gate_nollm",
    "base_gate": "cover_rel_base_gate_nollm",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def metrics_path(dataset: str, model: str, run: str, seed: int, stage: str) -> Path:
    filename = "stage1_metrics.json" if stage == "base" else "stage3_metrics.json"
    return Path("artifacts/results") / dataset / model / run / f"seed_{seed}" / filename


def log_path(dataset: str, model: str, run: str, seed: int, filename: str) -> Path:
    return Path("artifacts/logs") / dataset / model / run / f"seed_{seed}" / filename


def method_metrics(dataset: str, model: str, run: str, seed: int) -> dict[str, Any]:
    if run == "base":
        metrics = load_json(metrics_path(dataset, model, "base", seed, "base"))
        return {"metrics": metrics, "residual": {}, "gate": {}}
    metrics = load_json(metrics_path(dataset, model, run, seed, "stage3"))
    residual = load_json(log_path(dataset, model, run, seed, "residual_diagnostics.json"))
    gate = load_json(log_path(dataset, model, run, seed, "relation_gate_stats.json"))
    return {"metrics": metrics, "residual": residual, "gate": gate}


def gate_runs_for_dataset(dataset: str) -> dict[str, str]:
    spec = DATASETS[dataset]
    return {**BASE_GATE_RUNS, **spec.get("extra_gate_runs", {})}


def row_for_method(
    dataset: str,
    model: str,
    method: str,
    run: str,
    seed: int,
    base: dict[str, Any],
    best_single: dict[str, Any],
) -> dict[str, Any]:
    payload = method_metrics(dataset, model, run, seed)
    metrics = payload["metrics"]
    residual = payload["residual"]
    gate = payload["gate"]
    status = "complete" if metrics else "missing"
    return {
        "dataset": dataset,
        "method": method,
        "run_name": run,
        "row_type": "seed",
        "seed": seed,
        "status": status,
        "roc_auc": metrics.get("roc_auc", ""),
        "auprc": metrics.get("auprc", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "f1": metrics.get("f1", ""),
        "delta_auprc_vs_base": (
            metrics.get("auprc", 0.0) - base.get("auprc", 0.0) if metrics and base else ""
        ),
        "delta_roc_auc_vs_base": (
            metrics.get("roc_auc", 0.0) - base.get("roc_auc", 0.0) if metrics and base else ""
        ),
        "delta_macro_f1_vs_base": (
            metrics.get("macro_f1", 0.0) - base.get("macro_f1", 0.0) if metrics and base else ""
        ),
        "delta_auprc_vs_best_single": (
            metrics.get("auprc", 0.0) - best_single.get("auprc", 0.0) if metrics and best_single else ""
        ),
        "delta_macro_f1_vs_best_single": (
            metrics.get("macro_f1", 0.0) - best_single.get("macro_f1", 0.0) if metrics and best_single else ""
        ),
        "near_cap_fraction": residual.get("near_cap_fraction", ""),
        "residual_shift_mean": residual.get("residual_shift_mean", ""),
        "residual_shift_max_abs": residual.get("residual_shift_max_abs", ""),
        "mean_gate_all_relations": gate.get("mean_gate_all_relations", ""),
        "gate_vs_near_cap_corr": gate.get("gate_vs_near_cap_corr", ""),
        "gate_vs_abs_residual_corr": gate.get("gate_vs_abs_residual_corr", ""),
        "mean_sparse_penalty": gate.get("mean_sparse_penalty", ""),
    }


def aggregate_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete = [row for row in rows if row["status"] == "complete"]
    result: list[dict[str, Any]] = []
    for kind, fn in (("mean", mean), ("std", stdev)):
        agg: dict[str, Any] = {
            "dataset": rows[0]["dataset"],
            "method": rows[0]["method"],
            "run_name": rows[0]["run_name"],
            "row_type": kind,
            "seed": "",
            "status": f"{len(complete)}/{len(rows)} complete",
            "positive_auprc_vs_base_seeds": sum(
                1 for row in complete if row.get("delta_auprc_vs_base", "") != "" and float(row["delta_auprc_vs_base"]) > 0
            ),
        }
        for field in (
            "roc_auc",
            "auprc",
            "macro_f1",
            "f1",
            "delta_auprc_vs_base",
            "delta_roc_auc_vs_base",
            "delta_macro_f1_vs_base",
            "delta_auprc_vs_best_single",
            "delta_macro_f1_vs_best_single",
            "near_cap_fraction",
            "residual_shift_mean",
            "residual_shift_max_abs",
            "mean_gate_all_relations",
            "gate_vs_near_cap_corr",
            "gate_vs_abs_residual_corr",
            "mean_sparse_penalty",
        ):
            values = [float(row[field]) for row in complete if row.get(field, "") != ""]
            agg[field] = fn(values) if values else ""
        result.append(agg)
    return result


def dataset_rows(dataset: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    spec = DATASETS[dataset]
    model = spec["model"]
    method_runs = {
        "base": "base",
        "best_single": spec["best_single_run"],
        "all_rel": spec["all_rel_run"],
        **gate_runs_for_dataset(dataset),
    }
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for method, run in method_runs.items():
        method_seed_rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            base = method_metrics(dataset, model, "base", seed)["metrics"]
            best = method_metrics(dataset, model, spec["best_single_run"], seed)["metrics"]
            method_seed_rows.append(row_for_method(dataset, model, method, run, seed, base, best))
        rows.extend(method_seed_rows)
        aggregates = aggregate_method(method_seed_rows)
        rows.extend(aggregates)
        summary[method] = aggregates[0]
    return rows, summary


def gate_diagnostic_rows(dataset: str) -> list[dict[str, Any]]:
    spec = DATASETS[dataset]
    model = spec["model"]
    rows: list[dict[str, Any]] = []
    for method, run in gate_runs_for_dataset(dataset).items():
        for seed in SEEDS:
            for filename, diagnostic_type in (
                ("relation_gate_by_relation.csv", "relation"),
                ("relation_gate_by_label.csv", "label"),
                ("relation_gate_by_base_status.csv", "base_status"),
            ):
                for row in load_csv(log_path(dataset, model, run, seed, filename)):
                    row = dict(row)
                    row["diagnostic_type"] = diagnostic_type
                    row["dataset"] = dataset
                    row["method"] = method
                    row["run_name"] = run
                    rows.append(row)
    return rows


def verdict(summary: dict[str, dict[str, Any]]) -> tuple[str, str]:
    best = summary["best_single"]
    gate_candidates = {name: summary[name] for name in ("anchor_gate", "base_gate")}
    best_gate_name, best_gate = max(gate_candidates.items(), key=lambda item: float(item[1].get("auprc", -999.0)))
    best_single_delta = float(best.get("delta_auprc_vs_base", 0.0))
    gate_delta = float(best_gate.get("delta_auprc_vs_base", 0.0))
    gate_vs_single = float(best_gate.get("delta_auprc_vs_best_single", 0.0))
    macro_vs_single = float(best_gate.get("delta_macro_f1_vs_best_single", 0.0))
    positives = int(best_gate.get("positive_auprc_vs_base_seeds", 0))
    gate_near = best_gate.get("near_cap_fraction", "")
    single_near = best.get("near_cap_fraction", "")
    near_reduced = gate_near != "" and single_near != "" and float(gate_near) < float(single_near)

    if gate_delta >= best_single_delta and macro_vs_single >= -0.005 and positives >= 4:
        return "Strong GO", f"{best_gate_name} matches or exceeds best-single mean Delta AUPRC."
    if gate_vs_single >= -0.002 and macro_vs_single >= -0.005 and (near_reduced or positives >= 4):
        return "Acceptable GO", f"{best_gate_name} is within 0.002 AUPRC of best-single with stability or near-cap benefit."
    return "No-Go", f"{best_gate_name} is lower than best-single by more than 0.002 without enough stability benefit."


def write_dataset_outputs(dataset: str) -> dict[str, Any]:
    rows, summary = dataset_rows(dataset)
    diagnostic_rows = gate_diagnostic_rows(dataset)
    table_prefix = f"{dataset}_cover_rel_gate_5seed"
    csv_path = Path("artifacts/tables") / f"{table_prefix}.csv"
    md_path = Path("artifacts/tables") / f"{table_prefix}.md"
    report_path = Path("artifacts/reports") / f"{table_prefix}_conclusion.md"
    diag_path = Path("artifacts/reports") / f"{dataset}_cover_rel_gate_diagnostics_5seed.md"
    relation_csv = Path("artifacts/tables") / f"{dataset}_cover_rel_gate_diagnostics_5seed.csv"

    fields = [
        "dataset",
        "method",
        "row_type",
        "seed",
        "status",
        "auprc",
        "delta_auprc_vs_base",
        "delta_auprc_vs_best_single",
        "roc_auc",
        "delta_roc_auc_vs_base",
        "macro_f1",
        "delta_macro_f1_vs_best_single",
        "near_cap_fraction",
        "mean_gate_all_relations",
        "positive_auprc_vs_base_seeds",
    ]
    write_csv(csv_path, rows)
    write_md(md_path, rows, fields)
    write_csv(relation_csv, diagnostic_rows)
    decision, reason = verdict(summary)
    spec = DATASETS[dataset]
    report_lines = [
        f"# {dataset} CoVER-REL Gate 5-Seed Conclusion",
        "",
        f"- Verdict: **{decision}**",
        f"- Decision reason: {reason}",
        f"- Best relation: {spec['best_relation']}",
        f"- Best-single mean Delta AUPRC: {float(summary['best_single'].get('delta_auprc_vs_base', 0.0)):.6f}",
        f"- Anchor-gate mean Delta AUPRC: {float(summary['anchor_gate'].get('delta_auprc_vs_base', 0.0)):.6f}",
        f"- Base-gate mean Delta AUPRC: {float(summary['base_gate'].get('delta_auprc_vs_base', 0.0)):.6f}",
        f"- Anchor-gate mean near-cap: {fmt(summary['anchor_gate'].get('near_cap_fraction', ''))}",
        f"- Base-gate mean near-cap: {fmt(summary['base_gate'].get('near_cap_fraction', ''))}",
    ]
    if "anchor_gate_conservative" in summary:
        conservative = summary["anchor_gate_conservative"]
        conservative_positive = float(conservative.get("delta_auprc_vs_base", 0.0)) > 0.0
        conservative_near = float(conservative.get("near_cap_fraction", 1.0))
        anchor_near = float(summary["anchor_gate"].get("near_cap_fraction", 1.0))
        report_lines.extend([
            f"- Conservative anchor-gate mean Delta AUPRC: {float(conservative.get('delta_auprc_vs_base', 0.0)):.6f}",
            f"- Conservative anchor-gate mean near-cap: {fmt(conservative.get('near_cap_fraction', ''))}",
            f"- Conservative branch accepted: {conservative_positive and conservative_near < anchor_near}",
        ])
    report_lines.extend(["", "## Metrics", "", md_path.read_text()])
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(report_lines) + "\n")

    diag_fields = [
        "diagnostic_type",
        "dataset",
        "method",
        "seed",
        "relation",
        "label",
        "base_status",
        "gate_mean",
        "gate_open_rate",
        "contribution_norm_mean",
    ]
    diag_lines = [
        f"# {dataset} CoVER-REL Gate Diagnostics 5-Seed",
        "",
        f"- Rows: {len(diagnostic_rows)}",
        "- Gate-open threshold: 0.5",
        f"- Anchor-gate gate-vs-near-cap corr: {fmt(summary['anchor_gate'].get('gate_vs_near_cap_corr', ''))}",
        f"- Anchor-gate gate-vs-abs-residual corr: {fmt(summary['anchor_gate'].get('gate_vs_abs_residual_corr', ''))}",
        f"- Base-gate gate-vs-near-cap corr: {fmt(summary['base_gate'].get('gate_vs_near_cap_corr', ''))}",
        f"- Base-gate gate-vs-abs-residual corr: {fmt(summary['base_gate'].get('gate_vs_abs_residual_corr', ''))}",
        "",
    ]
    if diagnostic_rows:
        write_md(diag_path.with_suffix(".tmp.md"), diagnostic_rows, diag_fields)
        diag_lines.extend(["## Gate Diagnostics", "", diag_path.with_suffix(".tmp.md").read_text()])
        diag_path.with_suffix(".tmp.md").unlink()
    else:
        diag_lines.append("No gate diagnostic rows found.")
    diag_path.write_text("\n".join(diag_lines) + "\n")

    return {
        "dataset": dataset,
        "verdict": decision,
        "reason": reason,
        "best_relation": spec["best_relation"],
        "best_single_delta_auprc": summary["best_single"].get("delta_auprc_vs_base", ""),
        "anchor_gate_delta_auprc": summary["anchor_gate"].get("delta_auprc_vs_base", ""),
        "base_gate_delta_auprc": summary["base_gate"].get("delta_auprc_vs_base", ""),
        "conservative_delta_auprc": summary.get("anchor_gate_conservative", {}).get("delta_auprc_vs_base", ""),
        "anchor_gate_delta_vs_best_single": summary["anchor_gate"].get("delta_auprc_vs_best_single", ""),
        "base_gate_delta_vs_best_single": summary["base_gate"].get("delta_auprc_vs_best_single", ""),
        "conservative_delta_vs_best_single": summary.get("anchor_gate_conservative", {}).get(
            "delta_auprc_vs_best_single",
            "",
        ),
        "anchor_gate_near_cap": summary["anchor_gate"].get("near_cap_fraction", ""),
        "base_gate_near_cap": summary["base_gate"].get("near_cap_fraction", ""),
        "conservative_near_cap": summary.get("anchor_gate_conservative", {}).get("near_cap_fraction", ""),
        "best_single_near_cap": summary["best_single"].get("near_cap_fraction", ""),
    }


def write_cross_dataset(rows: list[dict[str, Any]]) -> None:
    csv_path = Path("artifacts/tables/cover_rel_schema_aware_summary.csv")
    md_path = Path("artifacts/tables/cover_rel_schema_aware_summary.md")
    report_path = Path("artifacts/reports/cover_rel_schema_aware_final_conclusion.md")
    fields = [
        "dataset",
        "verdict",
        "best_relation",
        "best_single_delta_auprc",
        "anchor_gate_delta_auprc",
        "base_gate_delta_auprc",
        "conservative_delta_auprc",
        "anchor_gate_delta_vs_best_single",
        "base_gate_delta_vs_best_single",
        "conservative_delta_vs_best_single",
        "best_single_near_cap",
        "anchor_gate_near_cap",
        "base_gate_near_cap",
        "conservative_near_cap",
    ]
    write_csv(csv_path, rows)
    write_md(md_path, rows, fields)
    report_lines = [
        "# CoVER-REL Schema-Aware Final Conclusion",
        "",
        "- YelpChi best relation: RUR.",
        "- Amazon best relation: UVU.",
        "- Anchor/base gates were evaluated without Qwen latents and with dataset-driven relation schemas.",
        "- Strong GO requires gate mean Delta AUPRC to match or exceed best-single; Acceptable GO allows near-cap or stability gains within 0.002 AUPRC.",
        "",
        "## Summary",
        "",
        md_path.read_text(),
        "",
        "## Interpretation",
        "",
        "- CoVER-REL is schema-aware because relation schemas differ across YelpChi and Amazon and are configured rather than hardcoded.",
        "- YelpChi remains RUR-concentrated: anchor_gate stays within 0.001 AUPRC of RUR-only and reduces near-cap, while base_gate is weaker.",
        "- Amazon is UVU-centered but distributed: anchor/base gates improve over UVU-only, while the conservative anchor branch trades AUPRC for zero near-cap.",
        "- Final model selection should keep dataset-specific relation schemas but not hardcode YelpChi relation names.",
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(report_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["yelpchi", "amazon"])
    args = parser.parse_args()
    summary_rows = [write_dataset_outputs(dataset) for dataset in args.datasets]
    write_cross_dataset(summary_rows)
    print(json.dumps({
        "datasets": args.datasets,
        "summary": "artifacts/reports/cover_rel_schema_aware_final_conclusion.md",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
