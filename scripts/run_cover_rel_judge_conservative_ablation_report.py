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


DATASET = "amazon"
MODEL = "bwgnn"
DEFAULT_SEEDS = [123, 456, 789]
RUN_ANCHOR = "cover_rel_anchor_gate_nollm"
RUN_ORIGINAL = "cover_rel_judge_uvu"
RUNS = {
    "original_judge": RUN_ORIGINAL,
    "delta05": "cover_rel_judge_uvu_delta05",
    "alpha05": "cover_rel_judge_uvu_alpha05",
    "conservative": "cover_rel_judge_uvu_conservative",
    "strength_gate": "cover_rel_judge_uvu_strength_gate",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def metric_path(run: str, seed: int) -> Path:
    return Path("artifacts/results") / DATASET / MODEL / run / f"seed_{seed}" / "stage3_metrics.json"


def log_path(run: str, seed: int, name: str) -> Path:
    return Path("artifacts/logs") / DATASET / MODEL / run / f"seed_{seed}" / name


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


def ranking_gap_delta(run: str, seed: int) -> float | str:
    ranking = load_json(log_path(run, seed, "ranking_gap_diagnostics.json"))
    if not ranking:
        return ""
    if "ranking_gap_delta" in ranking:
        return float(ranking["ranking_gap_delta"])
    gap = ranking.get("ranking_gap_pos_vs_hard_neg")
    base_gap = ranking.get("base_ranking_gap_pos_vs_hard_neg")
    if gap is None or base_gap is None:
        return ""
    return float(gap) - float(base_gap)


def group_alpha(run: str, seed: int, group_file: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in load_csv_rows(log_path(run, seed, group_file)):
        group = row.get("group", "")
        alpha = row.get("alpha_mean", "")
        if group and alpha != "":
            result[group] = float(alpha)
    return result


def seed_row(label: str, run: str, seed: int) -> dict[str, Any]:
    metrics = load_json(metric_path(run, seed))
    anchor = load_json(metric_path(RUN_ANCHOR, seed))
    original = load_json(metric_path(RUN_ORIGINAL, seed))
    diag = load_json(log_path(run, seed, "llm_judge_diagnostics.json"))
    residual = load_json(log_path(run, seed, "residual_diagnostics.json"))
    original_diag = load_json(log_path(RUN_ORIGINAL, seed, "llm_judge_diagnostics.json"))
    original_residual = load_json(log_path(RUN_ORIGINAL, seed, "residual_diagnostics.json"))
    by_strength = group_alpha(run, seed, "llm_judge_by_strength.csv")
    by_verdict = group_alpha(run, seed, "llm_judge_by_verdict.csv")
    return {
        "variant": label,
        "run_name": run,
        "seed": seed,
        "status": "complete" if metrics else "missing",
        "auprc": metrics.get("auprc", ""),
        "roc_auc": metrics.get("roc_auc", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "f1": metrics.get("f1", ""),
        "delta_auprc_vs_anchor": metrics.get("auprc", 0.0) - anchor.get("auprc", 0.0) if metrics and anchor else "",
        "delta_auprc_vs_original_judge": metrics.get("auprc", 0.0) - original.get("auprc", 0.0) if metrics and original else "",
        "delta_macro_f1_vs_anchor": metrics.get("macro_f1", 0.0) - anchor.get("macro_f1", 0.0) if metrics and anchor else "",
        "delta_macro_f1_vs_original_judge": metrics.get("macro_f1", 0.0) - original.get("macro_f1", 0.0) if metrics and original else "",
        "alpha_llm_mean": diag.get("alpha_llm_mean", ""),
        "alpha_llm_max": diag.get("alpha_llm_max", ""),
        "alpha_fake": by_verdict.get("verdict_fake", ""),
        "alpha_real": by_verdict.get("verdict_real", ""),
        "alpha_uncertain": by_verdict.get("verdict_uncertain", ""),
        "alpha_weak": by_strength.get("strength_weak", ""),
        "alpha_moderate": by_strength.get("strength_moderate", ""),
        "alpha_strong": by_strength.get("strength_strong", ""),
        "delta_llm_mean": diag.get("delta_llm_mean", ""),
        "delta_llm_max_abs": diag.get("delta_llm_max_abs", ""),
        "final_vs_rel_delta_abs_mean": diag.get("final_vs_rel_delta_abs_mean", ""),
        "llm_near_cap_fraction": diag.get("llm_near_cap_fraction", ""),
        "llm_near_cap_delta_vs_original": diag.get("llm_near_cap_fraction", 0.0) - original_diag.get("llm_near_cap_fraction", 0.0)
        if diag and original_diag else "",
        "near_cap_fraction": residual.get("near_cap_fraction", ""),
        "near_cap_delta_vs_original": residual.get("near_cap_fraction", 0.0) - original_residual.get("near_cap_fraction", 0.0)
        if residual and original_residual else "",
        "ranking_gap_delta": ranking_gap_delta(run, seed),
    }


def aggregate(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "complete"]
    fn = mean if kind == "mean" else stdev
    result = {"variant": rows[0]["variant"], "run_name": rows[0]["run_name"], "seed": kind, "status": f"{len(complete)}/{len(rows)} complete"}
    for field in rows[0]:
        if field in {"variant", "run_name", "seed", "status"}:
            continue
        values = [float(row[field]) for row in complete if row.get(field, "") != ""]
        result[field] = fn(values) if values else ""
    return result


def variant_verdict(mean_row: dict[str, Any], seed_rows: list[dict[str, Any]], original_mean: dict[str, Any]) -> str:
    delta_anchor = float(mean_row.get("delta_auprc_vs_anchor", 0.0))
    macro_drop = -float(mean_row.get("delta_macro_f1_vs_anchor", 0.0))
    alpha_means = [float(row.get("alpha_llm_mean", 1.0)) for row in seed_rows if row.get("alpha_llm_mean", "") != ""]
    near_delta = float(mean_row.get("near_cap_delta_vs_original", 0.0))
    llm_near_delta = float(mean_row.get("llm_near_cap_delta_vs_original", 0.0))
    alpha_fixed = alpha_means and max(alpha_means) <= 0.9
    alpha_strong = alpha_means and max(alpha_means) <= 0.7
    near_reduced = near_delta < 0.0 or llm_near_delta < 0.0
    original_auprc = float(original_mean.get("auprc", 0.0))
    auprc = float(mean_row.get("auprc", 0.0))
    if delta_anchor >= 0.0005 and macro_drop <= 0.005 and alpha_strong and near_reduced:
        return "Strong conservative GO"
    if auprc >= original_auprc - 0.0005 and macro_drop <= 0.005 and alpha_fixed and near_reduced:
        return "Acceptable conservative GO"
    return "No-Go"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    seed_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for label, run in RUNS.items():
        variant_rows = [seed_row(label, run, seed) for seed in args.seeds]
        seed_rows_by_variant[label] = variant_rows
        rows.extend(variant_rows)
        mean_row = aggregate(variant_rows, "mean")
        std_row = aggregate(variant_rows, "std")
        rows.extend([mean_row, std_row])
        summaries[label] = mean_row

    original_mean = summaries["original_judge"]
    verdicts = {
        label: variant_verdict(mean_row, seed_rows_by_variant[label], original_mean)
        for label, mean_row in summaries.items()
        if label != "original_judge"
    }
    priority = {"Strong conservative GO": 2, "Acceptable conservative GO": 1, "No-Go": 0}
    best_label = max(
        (label for label in verdicts),
        key=lambda label: (
            priority[verdicts[label]],
            float(summaries[label].get("delta_auprc_vs_anchor", -999.0)),
            -float(summaries[label].get("alpha_llm_mean", 999.0)),
        ),
    )

    table_csv = Path("artifacts/tables/amazon_cover_rel_judge_conservative_ablation_3seed.csv")
    table_md = Path("artifacts/tables/amazon_cover_rel_judge_conservative_ablation_3seed.md")
    write_csv(table_csv, rows)
    fields = [
        "variant",
        "seed",
        "auprc",
        "delta_auprc_vs_anchor",
        "delta_auprc_vs_original_judge",
        "macro_f1",
        "alpha_llm_mean",
        "alpha_llm_max",
        "llm_near_cap_fraction",
        "near_cap_fraction",
        "ranking_gap_delta",
    ]
    write_md(table_md, rows, fields)

    report = Path("artifacts/reports/amazon_cover_rel_judge_conservative_ablation_3seed.md")
    alpha_report = Path("artifacts/reports/amazon_cover_rel_judge_alpha_diagnostics_3seed.md")
    ensure_dir(report.parent)
    best = summaries[best_label]
    report.write_text(
        "\n".join([
            "# Amazon CoVER-REL-Judge Conservative Ablation 3-Seed",
            "",
            f"- Best conservative variant: **{best_label}**",
            f"- Verdict: **{verdicts[best_label]}**",
            f"- Mean Delta AUPRC vs anchor_gate: {fmt(best.get('delta_auprc_vs_anchor', 0.0))}",
            f"- Mean Delta AUPRC vs original judge: {fmt(best.get('delta_auprc_vs_original_judge', 0.0))}",
            f"- Mean Macro-F1 delta vs anchor_gate: {fmt(best.get('delta_macro_f1_vs_anchor', 0.0))}",
            f"- Mean alpha_llm: {fmt(best.get('alpha_llm_mean', 0.0))}",
            f"- Max-seed alpha_llm: {fmt(max(float(row.get('alpha_llm_mean', 0.0)) for row in seed_rows_by_variant[best_label]))}",
            f"- Mean LLM near-cap delta vs original judge: {fmt(best.get('llm_near_cap_delta_vs_original', 0.0))}",
            f"- Mean overall near-cap delta vs original judge: {fmt(best.get('near_cap_delta_vs_original', 0.0))}",
            "",
            "## Variant Verdicts",
            "",
            *[
                f"- {label}: {verdicts[label]} "
                f"(Delta AUPRC vs anchor={fmt(summaries[label].get('delta_auprc_vs_anchor', 0.0))}, "
                f"alpha={fmt(summaries[label].get('alpha_llm_mean', 0.0))}, "
                f"LLM near-cap={fmt(summaries[label].get('llm_near_cap_fraction', 0.0))})"
                for label in verdicts
            ],
            "",
            "## Metrics",
            "",
            table_md.read_text(),
        ])
        + "\n"
    )
    alpha_report.write_text(
        "\n".join([
            "# Amazon CoVER-REL-Judge Alpha Diagnostics 3-Seed",
            "",
            "Overall near-cap is measured against the base-detector residual and can remain high because the anchor relation branch already uses a large residual. "
            "LLM near-cap isolates the judge branch effect relative to the relation-only logit.",
            "",
            "| variant | alpha_mean | alpha_max | alpha_fake | alpha_real | alpha_uncertain | alpha_weak | alpha_moderate | alpha_strong | llm_near_cap | overall_near_cap |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *[
                "| "
                + " | ".join([
                    label,
                    fmt(summaries[label].get("alpha_llm_mean", "")),
                    fmt(summaries[label].get("alpha_llm_max", "")),
                    fmt(summaries[label].get("alpha_fake", "")),
                    fmt(summaries[label].get("alpha_real", "")),
                    fmt(summaries[label].get("alpha_uncertain", "")),
                    fmt(summaries[label].get("alpha_weak", "")),
                    fmt(summaries[label].get("alpha_moderate", "")),
                    fmt(summaries[label].get("alpha_strong", "")),
                    fmt(summaries[label].get("llm_near_cap_fraction", "")),
                    fmt(summaries[label].get("near_cap_fraction", "")),
                ])
                + " |"
                for label in RUNS
            ],
        ])
        + "\n"
    )
    print(json.dumps({"best_variant": best_label, "verdict": verdicts[best_label], "table": str(table_csv)}, indent=2))


if __name__ == "__main__":
    main()
