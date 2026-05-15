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


DEFAULT_SEEDS = [123, 456, 789]
DATASET = "amazon"
MODEL = "bwgnn"
RUN_JUDGE = "cover_rel_judge_uvu"
RUN_ANCHOR = "cover_rel_anchor_gate_nollm"
RUN_UVU = "cover_rel_uvu_nollm"
RUN_BASE_GATE = "cover_rel_base_gate_nollm"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_path(run: str, seed: int, base: bool = False) -> Path:
    run_name = "base" if base else run
    filename = "stage1_metrics.json" if base else "stage3_metrics.json"
    return Path("artifacts/results") / DATASET / MODEL / run_name / f"seed_{seed}" / filename


def log_path(run: str, seed: int, name: str) -> Path:
    return Path("artifacts/logs") / DATASET / MODEL / run / f"seed_{seed}" / name


def judge_dir(seed: int) -> Path:
    return Path("artifacts/judge_packets") / DATASET / MODEL / "cover_rel_judge" / f"seed_{seed}"


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


def seed_row(seed: int) -> dict[str, Any]:
    base = load_json(metric_path("base", seed, base=True))
    uvu = load_json(metric_path(RUN_UVU, seed))
    anchor = load_json(metric_path(RUN_ANCHOR, seed))
    base_gate = load_json(metric_path(RUN_BASE_GATE, seed))
    judge = load_json(metric_path(RUN_JUDGE, seed))
    stats = load_json(judge_dir(seed) / "judge_stats.json")
    diag = load_json(log_path(RUN_JUDGE, seed, "llm_judge_diagnostics.json"))
    residual = load_json(log_path(RUN_JUDGE, seed, "residual_diagnostics.json"))
    ranking = load_json(log_path(RUN_JUDGE, seed, "ranking_gap_diagnostics.json"))
    return {
        "seed": seed,
        "status": "complete" if judge else "missing",
        "base_auprc": base.get("auprc", ""),
        "uvu_auprc": uvu.get("auprc", ""),
        "anchor_auprc": anchor.get("auprc", ""),
        "base_gate_auprc": base_gate.get("auprc", ""),
        "judge_auprc": judge.get("auprc", ""),
        "delta_auprc_vs_anchor": judge.get("auprc", 0.0) - anchor.get("auprc", 0.0) if judge and anchor else "",
        "delta_auprc_vs_uvu": judge.get("auprc", 0.0) - uvu.get("auprc", 0.0) if judge and uvu else "",
        "delta_auprc_vs_base_gate": judge.get("auprc", 0.0) - base_gate.get("auprc", 0.0) if judge and base_gate else "",
        "roc_auc": judge.get("roc_auc", ""),
        "macro_f1": judge.get("macro_f1", ""),
        "f1": judge.get("f1", ""),
        "judge_acceptance_rate": stats.get("acceptance_rate", ""),
        "judge_num_accepted": stats.get("num_accepted", ""),
        "judge_num_rejected": stats.get("num_rejected", ""),
        "judge_verdict_distribution": json.dumps(stats.get("verdict_distribution", {}), sort_keys=True),
        "judge_strength_distribution": json.dumps(stats.get("strength_distribution", {}), sort_keys=True),
        "alpha_llm_mean": diag.get("alpha_llm_mean", ""),
        "delta_llm_mean": diag.get("delta_llm_mean", ""),
        "delta_llm_max_abs": diag.get("delta_llm_max_abs", ""),
        "final_vs_rel_delta_abs_mean": diag.get("final_vs_rel_delta_abs_mean", ""),
        "near_cap_fraction": residual.get("near_cap_fraction", ""),
        "ranking_gap_delta": ranking.get("ranking_gap_delta", ""),
    }


def aggregate(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "complete"]
    fn = mean if kind == "mean" else stdev
    result = {"seed": kind, "status": f"{len(complete)}/{len(rows)} complete"}
    for field in rows[0]:
        if field in {"seed", "status", "judge_verdict_distribution", "judge_strength_distribution"}:
            continue
        values = [float(row[field]) for row in complete if row.get(field, "") != ""]
        result[field] = fn(values) if values else ""
    return result


def verdict(mean_row: dict[str, Any]) -> tuple[str, str]:
    delta_anchor = float(mean_row.get("delta_auprc_vs_anchor", 0.0))
    macro = float(mean_row.get("macro_f1", 0.0))
    acceptance = float(mean_row.get("judge_acceptance_rate", 0.0))
    alpha = float(mean_row.get("alpha_llm_mean", 0.0))
    if delta_anchor >= 0.002 and acceptance >= 0.8 and macro > 0.0 and 0.02 < alpha < 0.98:
        return "Strong GO", "Amazon judge improves anchor gate by at least 0.002 AUPRC with acceptable verifier and alpha behavior."
    if delta_anchor >= 0.0 and acceptance >= 0.8 and macro > 0.0 and 0.02 < alpha < 0.98:
        return "Acceptable GO", "Amazon judge is weakly non-negative over anchor gate with valid verifier; inspect per-seed alpha saturation before 5-seed expansion."
    return "No-Go", "Amazon judge branch does not meet AUPRC, verifier, or alpha stability criteria."


def write_examples(path: Path, seeds: list[int]) -> None:
    lines = ["# Amazon CoVER-REL-Judge Examples", ""]
    for seed in seeds:
        rows = load_jsonl(judge_dir(seed) / "accepted_judge.jsonl")[:5]
        lines.append(f"## Seed {seed}")
        lines.append("")
        for row in rows:
            lines.append(
                f"- node {row['node_id']}: verdict={row['verdict']} strength={row['evidence_strength']} "
                f"key_relation={row['key_relation']} explanation={row['short_explanation']}"
            )
        lines.append("")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    seeds = args.seeds
    rows = [seed_row(seed) for seed in seeds]
    rows.extend([aggregate(rows, "mean"), aggregate(rows, "std")])
    table_csv = Path("artifacts/tables/amazon_cover_rel_judge_3seed.csv")
    table_md = Path("artifacts/tables/amazon_cover_rel_judge_3seed.md")
    write_csv(table_csv, rows)
    fields = [
        "seed",
        "status",
        "judge_auprc",
        "delta_auprc_vs_anchor",
        "delta_auprc_vs_uvu",
        "roc_auc",
        "macro_f1",
        "judge_acceptance_rate",
        "alpha_llm_mean",
        "delta_llm_max_abs",
        "near_cap_fraction",
    ]
    write_md(table_md, rows, fields)
    mean_row = rows[-2]
    decision, reason = verdict(mean_row)
    saturated_alpha_seeds = [
        row["seed"] for row in rows[: len(seeds)]
        if row.get("alpha_llm_mean", "") != "" and float(row["alpha_llm_mean"]) >= 0.98
    ]

    conclusion = Path("artifacts/reports/amazon_cover_rel_judge_3seed_conclusion.md")
    diag = Path("artifacts/reports/amazon_cover_rel_judge_diagnostics_3seed.md")
    ensure_dir(conclusion.parent)
    conclusion.write_text(
        "\n".join([
            "# Amazon CoVER-REL-Judge 3-Seed Conclusion",
            "",
            f"- Verdict: **{decision}**",
            f"- Reason: {reason}",
            f"- Mean Delta AUPRC vs anchor_gate: {fmt(mean_row.get('delta_auprc_vs_anchor', 0.0))}",
            f"- Mean Delta AUPRC vs UVU-only: {fmt(mean_row.get('delta_auprc_vs_uvu', 0.0))}",
            f"- Judge acceptance rate: {fmt(mean_row.get('judge_acceptance_rate', 0.0))}",
            f"- Alpha LLM mean: {fmt(mean_row.get('alpha_llm_mean', 0.0))}",
            f"- Alpha saturation risk seeds: {saturated_alpha_seeds}",
            "",
            "## Metrics",
            "",
            table_md.read_text(),
        ])
        + "\n"
    )
    diag.write_text(
        "\n".join([
            "# Amazon CoVER-REL-Judge Diagnostics 3-Seed",
            "",
            f"- Mean alpha_llm: {fmt(mean_row.get('alpha_llm_mean', 0.0))}",
            f"- Mean delta_llm: {fmt(mean_row.get('delta_llm_mean', 0.0))}",
            f"- Mean delta_llm max abs: {fmt(mean_row.get('delta_llm_max_abs', 0.0))}",
            f"- Mean final-vs-rel abs delta: {fmt(mean_row.get('final_vs_rel_delta_abs_mean', 0.0))}",
            f"- Mean near-cap fraction: {fmt(mean_row.get('near_cap_fraction', 0.0))}",
            f"- Mean ranking gap delta: {fmt(mean_row.get('ranking_gap_delta', 0.0))}",
        ])
        + "\n"
    )
    write_examples(Path("artifacts/reports/amazon_cover_rel_judge_examples.md"), seeds)
    print(json.dumps({"verdict": decision, "table": str(table_csv), "report": str(conclusion)}, indent=2))


if __name__ == "__main__":
    main()
