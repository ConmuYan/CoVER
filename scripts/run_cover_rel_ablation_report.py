from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import ensure_dir


DEFAULT_SEEDS = [123, 456, 789]
DEFAULT_RUNS = {
    "upu": "cover_rel_upu_nollm",
    "usu": "cover_rel_usu_nollm",
    "uvu": "cover_rel_uvu_nollm",
    "all": "cover_rel_all_nollm",
}
SINGLE_RELATIONS = ("upu", "usu", "uvu")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


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


def write_md_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def train_trend(dataset: str, model: str, run_name: str, seed: int) -> dict[str, Any]:
    rows = load_jsonl(
        Path("artifacts")
        / "logs"
        / dataset
        / model
        / run_name
        / f"seed_{seed}"
        / "stage3_train_log.jsonl"
    )
    if not rows:
        return {}
    first, last = rows[0], rows[-1]
    first_gap = first.get("ranking_gap_pos_vs_hard_neg")
    last_gap = last.get("ranking_gap_pos_vs_hard_neg")
    return {
        "ranking_gap_train_first": first_gap if first_gap is not None else "",
        "ranking_gap_train_last": last_gap if last_gap is not None else "",
        "ranking_gap_train_delta": (
            float(last_gap) - float(first_gap)
            if first_gap is not None and last_gap is not None
            else ""
        ),
        "val_auprc_first": first.get("val_auprc", ""),
        "val_auprc_last": last.get("val_auprc", ""),
    }


def row_for_seed(dataset: str, model: str, relation: str, run_name: str, seed: int) -> dict[str, Any]:
    base = load_json(
        Path("artifacts") / "results" / dataset / model / "base" / f"seed_{seed}" / "stage1_metrics.json"
    )
    cover = load_json(
        Path("artifacts") / "results" / dataset / model / run_name / f"seed_{seed}" / "stage3_metrics.json"
    )
    log_dir = Path("artifacts") / "logs" / dataset / model / run_name / f"seed_{seed}"
    residual = load_json(log_dir / "residual_diagnostics.json")
    ranking = load_json(log_dir / "ranking_gap_diagnostics.json")
    stage3 = load_json(log_dir / "stage3.json")
    relation_meta = stage3.get("relation_feature_meta", {}) if stage3 else {}

    row: dict[str, Any] = {
        "relation": relation,
        "run_name": run_name,
        "row_type": "seed",
        "seed": seed,
        "status": "complete" if base and cover else "missing",
        "base_roc_auc": base.get("roc_auc", ""),
        "cover_roc_auc": cover.get("roc_auc", ""),
        "delta_roc_auc": cover.get("roc_auc", 0.0) - base.get("roc_auc", 0.0) if base and cover else "",
        "base_auprc": base.get("auprc", ""),
        "cover_auprc": cover.get("auprc", ""),
        "delta_auprc": cover.get("auprc", 0.0) - base.get("auprc", 0.0) if base and cover else "",
        "base_f1_at_val": base.get("f1", ""),
        "cover_f1_at_val": cover.get("f1", ""),
        "delta_f1_at_val": cover.get("f1", 0.0) - base.get("f1", 0.0) if base and cover else "",
        "base_macro_f1_at_val": base.get("macro_f1", ""),
        "cover_macro_f1_at_val": cover.get("macro_f1", ""),
        "delta_macro_f1_at_val": (
            cover.get("macro_f1", 0.0) - base.get("macro_f1", 0.0) if base and cover else ""
        ),
        "base_recall@50": base.get("recall@50", ""),
        "cover_recall@50": cover.get("recall@50", ""),
        "delta_recall@50": (
            cover.get("recall@50", 0.0) - base.get("recall@50", 0.0) if base and cover else ""
        ),
        "ranking_gap_pos_vs_hard_neg": ranking.get("ranking_gap_pos_vs_hard_neg", ""),
        "base_ranking_gap_pos_vs_hard_neg": ranking.get("base_ranking_gap_pos_vs_hard_neg", ""),
        "ranking_gap_delta": (
            ranking.get("ranking_gap_pos_vs_hard_neg", 0.0)
            - ranking.get("base_ranking_gap_pos_vs_hard_neg", 0.0)
            if ranking
            and "ranking_gap_pos_vs_hard_neg" in ranking
            and "base_ranking_gap_pos_vs_hard_neg" in ranking
            else ""
        ),
        "residual_shift_mean": residual.get("residual_shift_mean", ""),
        "residual_shift_max_abs": residual.get("residual_shift_max_abs", ""),
        "near_cap_fraction": residual.get("near_cap_fraction", ""),
        "fn_correction_rate": stage3.get("fn_correction_rate", "") if stage3 else "",
        "fp_correction_rate": stage3.get("fp_correction_rate", "") if stage3 else "",
        "num_accepted_err": stage3.get("num_accepted_err", "") if stage3 else "",
        "relation_dim": relation_meta.get("rel_dim", stage3.get("relation_dim", "") if stage3 else ""),
        "relations": ",".join(relation_meta.get("relations", [])) if relation_meta else "",
        "prototype_labels": relation_meta.get("prototype_labels", ""),
        "test_label_used": relation_meta.get("test_label_used", ""),
        "relation_features_path": stage3.get("relation_features_path", "") if stage3 else "",
    }
    row.update(train_trend(dataset, model, run_name, seed))
    return row


def aggregate_rows(relation: str, run_name: str, seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete = [row for row in seed_rows if row["status"] == "complete"]
    result: list[dict[str, Any]] = []
    for kind, fn in (("mean", mean), ("std", stdev)):
        agg: dict[str, Any] = {
            "relation": relation,
            "run_name": run_name,
            "row_type": kind,
            "seed": "",
            "status": f"{len(complete)}/{len(seed_rows)} complete",
            "positive_auprc_seeds": sum(
                1 for row in complete if row.get("delta_auprc", "") != "" and float(row["delta_auprc"]) > 0
            ),
        }
        for field in (
            "base_roc_auc",
            "cover_roc_auc",
            "delta_roc_auc",
            "base_auprc",
            "cover_auprc",
            "delta_auprc",
            "base_f1_at_val",
            "cover_f1_at_val",
            "delta_f1_at_val",
            "base_macro_f1_at_val",
            "cover_macro_f1_at_val",
            "delta_macro_f1_at_val",
            "base_recall@50",
            "cover_recall@50",
            "delta_recall@50",
            "ranking_gap_pos_vs_hard_neg",
            "base_ranking_gap_pos_vs_hard_neg",
            "ranking_gap_delta",
            "residual_shift_mean",
            "residual_shift_max_abs",
            "near_cap_fraction",
            "fn_correction_rate",
            "fp_correction_rate",
            "ranking_gap_train_delta",
        ):
            values = [float(row[field]) for row in complete if row.get(field, "") != ""]
            agg[field] = fn(values) if values else ""
        result.append(agg)
    return result


def verdict(summary: dict[str, dict[str, Any]]) -> tuple[str, str]:
    single = {name: row for name, row in summary.items() if name in SINGLE_RELATIONS}
    best_name, best_row = max(single.items(), key=lambda item: float(item[1].get("delta_auprc", -999.0)))
    all_row = summary.get("all", {})
    best_delta = float(best_row.get("delta_auprc", 0.0))
    best_macro = float(best_row.get("delta_macro_f1_at_val", 0.0))
    all_delta = float(all_row.get("delta_auprc", 0.0)) if all_row else 0.0
    best_gap_delta = float(best_row.get("ranking_gap_delta", 0.0)) if best_row.get("ranking_gap_delta", "") != "" else 0.0
    best_near_cap = float(best_row.get("near_cap_fraction", 0.0)) if best_row.get("near_cap_fraction", "") != "" else 0.0

    unstable = best_near_cap >= 0.90
    if best_delta >= 0.006 and best_macro >= -0.005 and not unstable:
        return "Strong GO", f"{best_name.upper()} mean Delta AUPRC >= +0.006 with bounded Macro-F1 change."
    if (best_delta >= 0.003 or all_delta >= 0.003) and not unstable:
        return "GO", "A single relation or all-relation setting improves mean AUPRC by at least +0.003."
    if abs(best_delta) < 0.003 and best_gap_delta > 0.0 and best_macro >= -0.005 and not unstable:
        return "Conditional GO", "AUPRC is flat, but ranking gap improves and Macro-F1 is preserved."
    if unstable:
        return "NO-GO", f"Residual near-cap fraction is high for best relation ({best_near_cap:.3f})."
    return "NO-GO", "No Amazon relation expert or all-relation setting improves AUPRC enough."


def write_report(
    dataset: str,
    model: str,
    rows: list[dict[str, Any]],
    summary: dict[str, dict[str, Any]],
    csv_path: Path,
    md_path: Path,
    report_path: Path,
) -> None:
    single_summary = {name: row for name, row in summary.items() if name in SINGLE_RELATIONS}
    best_relation, best_row = max(
        single_summary.items(),
        key=lambda item: float(item[1].get("delta_auprc", -999.0)),
    )
    all_row = summary.get("all", {})
    decision, reason = verdict(summary)
    relation_deltas = {
        name: float(row.get("delta_auprc", 0.0))
        for name, row in single_summary.items()
    }
    high_near_cap = [
        name.upper()
        for name, row in summary.items()
        if row.get("near_cap_fraction", "") != "" and float(row["near_cap_fraction"]) >= 0.90
    ]
    spread = max(relation_deltas.values()) - min(relation_deltas.values())
    concentration = (
        "concentrated"
        if spread >= 0.003 and float(best_row.get("delta_auprc", 0.0)) > 0
        else "distributed_or_weak"
    )

    lines = [
        "# Amazon CoVER-REL Relation Ablation 3-Seed",
        "",
        f"- Dataset/model: `{dataset}/{model}`",
        f"- Generated git hash: `{git_hash()}`",
        f"- Verdict: **{decision}**",
        f"- Decision reason: {reason}",
        f"- Best single relation: `{best_relation.upper()}`",
        f"- Best single relation mean Delta AUPRC: {fmt(best_row.get('delta_auprc', ''))}",
        f"- Best single relation mean Delta Macro-F1: {fmt(best_row.get('delta_macro_f1_at_val', ''))}",
        f"- All-relation mean Delta AUPRC: {fmt(all_row.get('delta_auprc', '')) if all_row else 'missing'}",
        f"- Relation utility pattern: {concentration}",
        (
            f"- Residual near-cap caution: {', '.join(high_near_cap)} mean near-cap >= 0.90"
            if high_near_cap
            else "- Residual near-cap caution: none above 0.90 mean near-cap"
        ),
        "",
        "## Interpretation",
        "",
        (
            "- YelpChi utility was concentrated in R-U-R; Amazon is evaluated schema-first over "
            "UPU/USU/UVU without YelpChi relation-name assumptions."
        ),
        (
            f"- Amazon best relation is `{best_relation.upper()}`; compare its sign and magnitude "
            "against YelpChi RUR-only mean Delta AUPRC +0.027099."
        ),
        "- Qwen latents were not used in this task; Stage3 remains LLM-free.",
        "",
        "## Metrics",
        "",
        md_path.read_text(),
    ]
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_cover_rel_amazon_nollm.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[f"{relation}={run}" for relation, run in DEFAULT_RUNS.items()],
        help="Relation/run mappings such as upu=cover_rel_upu_nollm",
    )
    parser.add_argument("--output_prefix", default="amazon_cover_rel_relation_ablation_3seed")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    dataset = config["dataset"]["name"]
    model = config["model"]["name"]

    run_map: dict[str, str] = {}
    for item in args.runs:
        relation, run_name = item.split("=", 1)
        run_map[relation.lower()] = run_name

    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for relation, run_name in run_map.items():
        seed_rows = [row_for_seed(dataset, model, relation, run_name, seed) for seed in args.seeds]
        rows.extend(seed_rows)
        aggregates = aggregate_rows(relation, run_name, seed_rows)
        rows.extend(aggregates)
        summary[relation] = aggregates[0]

    csv_path = Path("artifacts") / "tables" / f"{args.output_prefix}.csv"
    md_path = Path("artifacts") / "tables" / f"{args.output_prefix}.md"
    report_path = Path("artifacts") / "reports" / f"{args.output_prefix}.md"
    write_csv(csv_path, rows)
    md_fields = [
        "relation",
        "row_type",
        "seed",
        "status",
        "delta_auprc",
        "delta_roc_auc",
        "delta_macro_f1_at_val",
        "delta_f1_at_val",
        "ranking_gap_delta",
        "residual_shift_mean",
        "residual_shift_max_abs",
        "near_cap_fraction",
        "positive_auprc_seeds",
        "relations",
        "prototype_labels",
        "test_label_used",
    ]
    write_md_table(md_path, rows, md_fields)
    ensure_dir(report_path.parent)
    write_report(dataset, model, rows, summary, csv_path, md_path, report_path)
    print(json.dumps({
        "csv": str(csv_path),
        "md": str(md_path),
        "report": str(report_path),
        "verdict": verdict(summary)[0],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
