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


DEFAULT_SEEDS = [42, 123, 456, 789, 2026]
MODEL = "bwgnn"
DATASETS = {
    "yelpchi": {
        "judge": "cover_rel_judge_rur_strength_gate",
        "original_judge": "cover_rel_judge_rur",
        "anchor": "cover_rel_anchor_gate_nollm",
        "single": "qwen_directional_t200_cover_rel_rur_nollm",
        "single_label": "RUR-only",
        "primary_relation": "RUR",
    },
    "amazon": {
        "judge": "cover_rel_judge_uvu_strength_gate",
        "original_judge": "cover_rel_judge_uvu",
        "anchor": "cover_rel_anchor_gate_nollm",
        "single": "cover_rel_uvu_nollm",
        "single_label": "UVU-only",
        "primary_relation": "UVU",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def metric_path(dataset: str, run: str, seed: int, base: bool = False) -> Path:
    run_name = "base" if base else run
    filename = "stage1_metrics.json" if base else "stage3_metrics.json"
    return Path("artifacts/results") / dataset / MODEL / run_name / f"seed_{seed}" / filename


def log_path(dataset: str, run: str, seed: int, name: str) -> Path:
    return Path("artifacts/logs") / dataset / MODEL / run / f"seed_{seed}" / name


def judge_dir(dataset: str, seed: int) -> Path:
    return Path("artifacts/judge_packets") / dataset / MODEL / "cover_rel_judge" / f"seed_{seed}"


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


def ranking_gap_delta(dataset: str, run: str, seed: int) -> float | str:
    ranking = load_json(log_path(dataset, run, seed, "ranking_gap_diagnostics.json"))
    if not ranking:
        return ""
    gap = ranking.get("ranking_gap_pos_vs_hard_neg")
    base_gap = ranking.get("base_ranking_gap_pos_vs_hard_neg")
    if gap is None or base_gap is None:
        return ""
    return float(gap) - float(base_gap)


def alpha_by_strength(dataset: str, run: str, seed: int) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in load_csv_rows(log_path(dataset, run, seed, "llm_judge_by_strength.csv")):
        group = row.get("group", "")
        alpha = row.get("alpha_mean", "")
        if group.startswith("strength_") and alpha != "":
            values[f"alpha_{group.removeprefix('strength_')}"] = float(alpha)
    return values


def seed_row(dataset: str, seed: int) -> dict[str, Any]:
    cfg = DATASETS[dataset]
    base = load_json(metric_path(dataset, "base", seed, base=True))
    judge = load_json(metric_path(dataset, cfg["judge"], seed))
    anchor = load_json(metric_path(dataset, cfg["anchor"], seed))
    single = load_json(metric_path(dataset, cfg["single"], seed))
    original = load_json(metric_path(dataset, cfg["original_judge"], seed))
    diag = load_json(log_path(dataset, cfg["judge"], seed, "llm_judge_diagnostics.json"))
    residual = load_json(log_path(dataset, cfg["judge"], seed, "residual_diagnostics.json"))
    alpha_strength = alpha_by_strength(dataset, cfg["judge"], seed)
    stats = load_json(judge_dir(dataset, seed) / "judge_stats.json")
    audit = load_json(judge_dir(dataset, seed) / "judge_forbidden_field_audit.json")
    audit_passed = bool(audit.get("passed", False))
    return {
        "dataset": dataset,
        "seed": seed,
        "status": "complete" if judge else "missing",
        "base_auprc": base.get("auprc", ""),
        "single_auprc": single.get("auprc", ""),
        "anchor_auprc": anchor.get("auprc", ""),
        "judge_auprc": judge.get("auprc", ""),
        "delta_auprc_vs_anchor": judge.get("auprc", 0.0) - anchor.get("auprc", 0.0) if judge and anchor else "",
        "delta_auprc_vs_single": judge.get("auprc", 0.0) - single.get("auprc", 0.0) if judge and single else "",
        "delta_auprc_vs_base": judge.get("auprc", 0.0) - base.get("auprc", 0.0) if judge and base else "",
        "delta_auprc_vs_original_judge": judge.get("auprc", 0.0) - original.get("auprc", 0.0) if judge and original else "",
        "roc_auc": judge.get("roc_auc", ""),
        "macro_f1": judge.get("macro_f1", ""),
        "f1": judge.get("f1", ""),
        "judge_acceptance_rate": stats.get("acceptance_rate", ""),
        "judge_num_accepted": stats.get("num_accepted", ""),
        "judge_num_rejected": stats.get("num_rejected", ""),
        "forbidden_audit_passed": audit_passed,
        "accepted_outputs_passed": bool(audit.get("accepted_outputs_passed", audit_passed)),
        "prompt_packet_passed": bool(audit.get("prompt_packet_passed", audit_passed)),
        "verdict_distribution": json.dumps(stats.get("verdict_distribution", {}), sort_keys=True),
        "strength_distribution": json.dumps(stats.get("strength_distribution", {}), sort_keys=True),
        "alpha_llm_mean": diag.get("alpha_llm_mean", ""),
        "alpha_llm_max": diag.get("alpha_llm_max", ""),
        "alpha_weak": alpha_strength.get("alpha_weak", ""),
        "alpha_moderate": alpha_strength.get("alpha_moderate", ""),
        "alpha_strong": alpha_strength.get("alpha_strong", ""),
        "delta_llm_mean": diag.get("delta_llm_mean", ""),
        "delta_llm_max_abs": diag.get("delta_llm_max_abs", ""),
        "final_vs_rel_delta_abs_mean": diag.get("final_vs_rel_delta_abs_mean", ""),
        "llm_near_cap_fraction": diag.get("llm_near_cap_fraction", ""),
        "overall_near_cap_fraction": residual.get("near_cap_fraction", ""),
        "ranking_gap_delta": ranking_gap_delta(dataset, cfg["judge"], seed),
    }


def aggregate(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "complete"]
    fn = mean if kind == "mean" else stdev
    result = {"dataset": rows[0]["dataset"], "seed": kind, "status": f"{len(complete)}/{len(rows)} complete"}
    for field in rows[0]:
        if field in {"dataset", "seed", "status", "verdict_distribution", "strength_distribution"}:
            continue
        if field in {"forbidden_audit_passed", "accepted_outputs_passed", "prompt_packet_passed"}:
            result[field] = all(bool(row.get(field, False)) for row in complete)
            continue
        values = [float(row[field]) for row in complete if row.get(field, "") != ""]
        result[field] = fn(values) if values else ""
    return result


def dataset_verdict(mean_row: dict[str, Any]) -> str:
    delta = float(mean_row.get("delta_auprc_vs_anchor", 0.0))
    acceptance = float(mean_row.get("judge_acceptance_rate", 0.0))
    alpha = float(mean_row.get("alpha_llm_mean", 0.0))
    audit_ok = bool(mean_row.get("forbidden_audit_passed", False))
    if delta >= 0.001 and acceptance >= 0.8 and alpha <= 0.9 and audit_ok:
        return "Strong 5-seed GO"
    if delta >= 0.0 and acceptance >= 0.8 and alpha <= 0.9 and audit_ok:
        return "Acceptable 5-seed GO"
    return "No-Go"


def write_examples(path: Path, seeds: list[int]) -> None:
    lines = ["# CoVER-REL-Judge Explanation Examples", ""]
    for dataset, cfg in DATASETS.items():
        lines.extend([f"## {dataset}", ""])
        count = 0
        for seed in seeds:
            for row in load_jsonl(judge_dir(dataset, seed) / "accepted_judge.jsonl"):
                lines.append(
                    f"- seed {seed} node {row['node_id']}: verdict={row['verdict']} "
                    f"strength={row['evidence_strength']} key_relation={row['key_relation']} "
                    f"explanation={row['short_explanation']}"
                )
                count += 1
                if count >= 6:
                    break
            if count >= 6:
                break
        lines.append("")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")


def write_safety_report(path: Path, rows: list[dict[str, Any]], seeds: list[int]) -> None:
    lines = ["# CoVER-REL-Judge Safety Audit", ""]
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset and isinstance(row["seed"], int)]
        mean_accept = mean([float(row["judge_acceptance_rate"]) for row in dataset_rows if row.get("judge_acceptance_rate", "") != ""])
        lines.extend([
            f"## {dataset}",
            "",
            f"- Judge acceptance mean: {fmt(mean_accept)}",
            f"- Forbidden audit passed all seeds: {all(bool(row.get('forbidden_audit_passed', False)) for row in dataset_rows)}",
            f"- Accepted outputs passed all seeds: {all(bool(row.get('accepted_outputs_passed', False)) for row in dataset_rows)}",
            f"- Prompt/packet audit passed all seeds: {all(bool(row.get('prompt_packet_passed', False)) for row in dataset_rows)}",
            "- Rejected judge outputs are excluded from judge features and fusion training.",
            "- `short_explanation` is human-facing only and is not used in loss.",
            "- Stage3 training consumes accepted judge features only and does not call Qwen.",
            "",
        ])
        for seed in seeds:
            audit = load_json(judge_dir(dataset, seed) / "judge_forbidden_field_audit.json")
            stats = load_json(judge_dir(dataset, seed) / "judge_stats.json")
            lines.append(
                f"- seed {seed}: accepted={stats.get('num_accepted', '')}, rejected={stats.get('num_rejected', '')}, "
                f"acceptance={fmt(stats.get('acceptance_rate', ''))}, audit_passed={audit.get('passed', False)}"
            )
        lines.append("")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        dataset_rows = [seed_row(dataset, seed) for seed in args.seeds]
        rows.extend(dataset_rows)
        mean_row = aggregate(dataset_rows, "mean")
        std_row = aggregate(dataset_rows, "std")
        rows.extend([mean_row, std_row])
        summaries[dataset] = mean_row

    table_csv = Path("artifacts/tables/cover_rel_judge_final_5seed_summary.csv")
    table_md = Path("artifacts/tables/cover_rel_judge_final_5seed_summary.md")
    fields = [
        "dataset",
        "seed",
        "judge_auprc",
        "delta_auprc_vs_anchor",
        "delta_auprc_vs_single",
        "roc_auc",
        "macro_f1",
        "judge_acceptance_rate",
        "alpha_llm_mean",
        "alpha_llm_max",
        "alpha_weak",
        "alpha_moderate",
        "alpha_strong",
        "llm_near_cap_fraction",
        "overall_near_cap_fraction",
        "ranking_gap_delta",
        "forbidden_audit_passed",
    ]
    write_csv(table_csv, rows)
    write_md(table_md, rows, fields)

    conclusion = Path("artifacts/reports/cover_rel_judge_final_conclusion.md")
    ensure_dir(conclusion.parent)
    conclusion.write_text(
        "\n".join([
            "# CoVER-REL-Judge Final 5-Seed Conclusion",
            "",
            "## Verdicts",
            "",
            *[
                f"- {dataset}: **{dataset_verdict(summary)}** "
                f"(Delta AUPRC vs anchor={fmt(summary.get('delta_auprc_vs_anchor', 0.0))}, "
                f"Delta vs best-single={fmt(summary.get('delta_auprc_vs_single', 0.0))}, "
                f"acceptance={fmt(summary.get('judge_acceptance_rate', 0.0))}, "
                f"alpha={fmt(summary.get('alpha_llm_mean', 0.0))})"
                for dataset, summary in summaries.items()
            ],
            "",
            "## Recommendation",
            "",
            "- Use CoVER-REL-Gate as the relation-only baseline.",
            "- Use CoVER-REL-Judge as the LLM-assisted research model.",
            "- Report LLM near-cap separately from overall near-cap because overall residual saturation can be dominated by the relation branch.",
            "",
            "## Summary Table",
            "",
            table_md.read_text(),
        ])
        + "\n"
    )
    write_safety_report(Path("artifacts/reports/cover_rel_judge_safety_audit.md"), rows, args.seeds)
    write_examples(Path("artifacts/reports/cover_rel_judge_explanation_examples.md"), args.seeds)
    print(json.dumps({
        dataset: {
            "verdict": dataset_verdict(summary),
            "delta_auprc_vs_anchor": summary.get("delta_auprc_vs_anchor", ""),
            "judge_acceptance_rate": summary.get("judge_acceptance_rate", ""),
        }
        for dataset, summary in summaries.items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
