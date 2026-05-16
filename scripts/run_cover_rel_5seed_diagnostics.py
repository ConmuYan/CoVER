from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.relation_features import RELATION_STAT_NAMES, load_relation_stats
from evidence.vocab import get_evidence_slots
from models.gnn import build_detector
from models.reasoner import EvidenceReasoner
from scripts.run_cover_rel_ablation_report import (
    aggregate_rows,
    fmt,
    load_json,
    mean,
    row_for_seed,
    write_csv,
    write_md_table,
)
from utils.paths import ensure_dir, get_base_checkpoint_path, get_checkpoint_dir


DEFAULT_SEEDS = [42, 123, 456, 789, 2026]
RUNS = {
    "uvu": "cover_rel_uvu_nollm",
    "all": "cover_rel_all_nollm",
}


def _relation_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).upper() for item in value if str(item).strip()]
    return [item.strip().upper() for item in str(value).split(",") if item.strip()]


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key, "") != ""]


def _correlation(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    return float(num / (den_x * den_y))


def write_stability_outputs(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    dataset = config["dataset"]["name"]
    model = config["model"]["name"]
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for relation, run_name in RUNS.items():
        seed_rows = [row_for_seed(dataset, model, relation, run_name, seed) for seed in seeds]
        rows.extend(seed_rows)
        aggregates = aggregate_rows(relation, run_name, seed_rows)
        rows.extend(aggregates)
        summary[relation] = aggregates[0]

    csv_path = Path("artifacts/tables/amazon_fresh_bwgnn_vs_cover_rel_uvu_all_5seed.csv")
    md_path = Path("artifacts/tables/amazon_fresh_bwgnn_vs_cover_rel_uvu_all_5seed.md")
    report_path = Path("artifacts/reports/amazon_cover_rel_uvu_all_5seed_conclusion.md")
    fields = [
        "relation",
        "row_type",
        "seed",
        "status",
        "base_auprc",
        "cover_auprc",
        "delta_auprc",
        "delta_roc_auc",
        "delta_macro_f1_at_val",
        "ranking_gap_delta",
        "residual_shift_mean",
        "residual_shift_max_abs",
        "near_cap_fraction",
        "positive_auprc_seeds",
        "relations",
        "prototype_labels",
        "test_label_used",
    ]
    write_csv(csv_path, rows)
    write_md_table(md_path, rows, fields)

    best_name, best_row = max(summary.items(), key=lambda item: float(item[1].get("delta_auprc", -999.0)))
    best_delta = float(best_row.get("delta_auprc", 0.0))
    best_macro = float(best_row.get("delta_macro_f1_at_val", 0.0))
    best_positive = int(best_row.get("positive_auprc_seeds", 0))
    best_near_cap = float(best_row.get("near_cap_fraction", 0.0))
    high_near_cap = [
        relation
        for relation, row in summary.items()
        if row.get("near_cap_fraction", "") != "" and float(row["near_cap_fraction"]) >= 0.90
    ]
    if best_delta >= 0.003 and best_macro >= -0.005 and best_positive >= 4 and best_near_cap < 0.90:
        verdict = "Strong GO for UVU"
        reason = (
            "best setting reaches mean Delta AUPRC >= +0.003 with Macro-F1 preserved; "
            "residual diagnostics still need conservative handling before a gate."
        )
    elif best_delta > 0.0 and best_macro >= -0.005 and best_positive >= 4:
        verdict = "Conditional GO"
        reason = "AUPRC is positive and stable, but near-cap or magnitude requires conservative follow-up."
    else:
        verdict = "NO-GO"
        reason = "5-seed gains are not stable enough for gate development."

    report_lines = [
        "# Amazon CoVER-REL UVU/All 5-Seed Conclusion",
        "",
        f"- Dataset/model: `{dataset}/{model}`",
        f"- Verdict: **{verdict}**",
        f"- Decision reason: {reason}",
        f"- Best setting: `{best_name}`",
        f"- Best mean Delta AUPRC: {best_delta:.6f}",
        f"- Best mean Delta Macro-F1: {best_macro:.6f}",
        f"- Best AUPRC positive seeds: {best_positive}/{len(seeds)}",
        f"- Best mean near-cap fraction: {best_near_cap:.6f}",
        f"- UVU mean Delta AUPRC: {float(summary['uvu'].get('delta_auprc', 0.0)):.6f}",
        f"- UVU mean near-cap fraction: {float(summary['uvu'].get('near_cap_fraction', 0.0)):.6f}",
        f"- All-rel mean Delta AUPRC: {float(summary['all'].get('delta_auprc', 0.0)):.6f}",
        f"- All-rel mean near-cap fraction: {float(summary['all'].get('near_cap_fraction', 0.0)):.6f}",
        (
            f"- Near-cap caution: {', '.join(high_near_cap)} mean near-cap >= 0.90"
            if high_near_cap
            else "- Near-cap caution: no setting has mean near-cap >= 0.90"
        ),
        "",
        "## Metrics",
        "",
        md_path.read_text(),
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(report_lines) + "\n")
    return [row for row in rows if row.get("row_type") == "seed"]


def load_stage3_info(dataset: str, model: str, run_name: str, seed: int) -> dict[str, Any]:
    return load_json(Path("artifacts/logs") / dataset / model / run_name / f"seed_{seed}" / "stage3.json")


def relation_feature_path(dataset: str, model: str, seed: int, relation: str) -> Path:
    return Path("artifacts/relation_features") / dataset / model / f"seed_{seed}" / relation / "rel_stats.pt"


def compute_full_outputs(
    config: dict[str, Any],
    run_name: str,
    relation: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    dataset = config["dataset"]["name"]
    model_name = config["model"]["name"]
    data = load_fraud_dataset(
        dataset,
        path=config["dataset"]["path"],
        seed=seed,
        split_mode=config["dataset"].get("split_mode", "supervised"),
        train_ratio=config["dataset"].get("train_ratio", 0.7),
        val_test_ratio=config["dataset"].get("val_test_ratio", [1, 2]),
        stratified=config["dataset"].get("stratified", False),
    )
    device_name = config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    base_model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    ).to(device)
    base_model.load_state_dict(torch.load(get_base_checkpoint_path(dataset, model_name, seed), weights_only=True))
    base_model.eval()

    x, edge_index = data.x.to(device), data.edge_index.to(device)
    with torch.no_grad():
        base_out = base_model(x, edge_index, return_output=True)
    base_logits = base_out.logits.detach()
    z = base_out.embeddings.detach()

    info = load_stage3_info(dataset, model_name, run_name, seed)
    rc = info.get("config", config).get("reasoner", config.get("reasoner", {}))
    reasoner = EvidenceReasoner(
        z_dim=z.shape[1],
        hidden_dim=rc.get("hidden_dim", 128),
        rho=rc.get("rho", 0.1),
        gate_mode=rc.get("gate_mode", "safe_residual"),
        delta_scale=rc.get("delta_scale", 2.0),
        gate_bias_init=rc.get("gate_bias_init", -2.0),
        residual_init_zero=rc.get("residual_init_zero", True),
        latent_dim=rc.get("latent_dim", 256),
        relation_dim=rc.get("relation_dim", 0),
        relation_hidden_dim=rc.get("relation_hidden_dim", 32),
        relation_fusion_mode=rc.get("relation_fusion_mode", "concat"),
        relation_names=_relation_list(rc.get("relation_names")),
        relation_stat_dim=rc.get("relation_stat_dim"),
        anchor_relation=rc.get("anchor_relation"),
        optional_relations=_relation_list(rc.get("optional_relations")),
        relation_dropout=0.0,
        gate_hidden_dim=rc.get("gate_hidden_dim", 64),
        gate_temperature=rc.get("gate_temperature", 1.0),
    ).to(device)
    checkpoint = get_checkpoint_dir(dataset, model_name, run_name, seed) / "reasoner.pt"
    reasoner.load_state_dict(torch.load(checkpoint, weights_only=True), strict=False)
    reasoner.eval()

    rel_path = Path(str(rc.get("relation_features_path") or relation_feature_path(dataset, model_name, seed, relation)))
    relation_features, _ = load_relation_stats(rel_path, num_nodes=data.x.shape[0])
    relation_features = relation_features.to(device)
    evidence_token_ids = torch.zeros(data.x.shape[0], len(get_evidence_slots()), dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = reasoner(z, base_logits, evidence_token_ids, relation_features=relation_features)
    return (
        base_logits.detach().cpu(),
        outputs["final_logit"].detach().cpu(),
        data.y.detach().cpu(),
        data.test_mask.detach().cpu().bool(),
        data.train_mask.detach().cpu().bool(),
        float(rc.get("max_allowed_shift", 0.2)),
    )


def _mask_stats(residual: torch.Tensor, mask: torch.Tensor, prefix: str) -> dict[str, Any]:
    if not bool(mask.any()):
        return {
            f"{prefix}_count": 0,
            f"{prefix}_shift_mean": "",
            f"{prefix}_shift_abs_mean": "",
            f"{prefix}_near_cap_fraction": "",
        }
    vals = residual[mask]
    return {
        f"{prefix}_count": int(mask.sum().item()),
        f"{prefix}_shift_mean": float(vals.mean().item()),
        f"{prefix}_shift_abs_mean": float(vals.abs().mean().item()),
    }


def write_residual_diagnostics(config: dict[str, Any], seed_rows: list[dict[str, Any]], seeds: list[int]) -> None:
    rows: list[dict[str, Any]] = []
    for relation, run_name in RUNS.items():
        for seed in seeds:
            base_logits, final_logits, y, test_mask, _, max_shift = compute_full_outputs(config, run_name, relation, seed)
            residual = final_logits - base_logits
            near_cap = residual.abs() >= 0.9 * max_shift
            base_pred = (torch.sigmoid(base_logits) >= 0.5).long()
            correct = base_pred == y
            masks = {
                "test_label0": test_mask & (y == 0),
                "test_label1": test_mask & (y == 1),
                "test_base_tn": test_mask & correct & (y == 0),
                "test_base_tp": test_mask & correct & (y == 1),
                "test_base_fp": test_mask & (~correct) & (y == 0),
                "test_base_fn": test_mask & (~correct) & (y == 1),
            }
            row = {
                "relation": relation,
                "run_name": run_name,
                "seed": seed,
                "split": "test",
                "delta_auprc": next(
                    float(r["delta_auprc"])
                    for r in seed_rows
                    if r["relation"] == relation and int(r["seed"]) == seed
                ),
                "delta_macro_f1_at_val": next(
                    float(r["delta_macro_f1_at_val"])
                    for r in seed_rows
                    if r["relation"] == relation and int(r["seed"]) == seed
                ),
                "near_cap_nodes_all": int(near_cap.sum().item()),
                "near_cap_fraction_all": float(near_cap.float().mean().item()),
                "residual_shift_mean_all": float(residual.mean().item()),
                "residual_shift_max_abs_all": float(residual.abs().max().item()),
            }
            for name, mask in masks.items():
                row.update(_mask_stats(residual, mask, name))
                row[f"{name}_near_cap_fraction"] = (
                    float(near_cap[mask].float().mean().item()) if bool(mask.any()) else ""
                )
            rows.append(row)

    csv_path = Path("artifacts/tables/amazon_cover_rel_residual_diagnostics_5seed.csv")
    report_path = Path("artifacts/reports/amazon_cover_rel_residual_diagnostics_5seed.md")
    write_csv(csv_path, rows)
    fields = [
        "relation",
        "seed",
        "delta_auprc",
        "delta_macro_f1_at_val",
        "near_cap_fraction_all",
        "near_cap_nodes_all",
        "residual_shift_mean_all",
        "test_label1_shift_mean",
        "test_base_fn_shift_mean",
        "test_base_fp_shift_mean",
    ]
    md_lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        md_lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    near_cap_vals = _float_values(rows, "near_cap_fraction_all")
    deltas = _float_values(rows, "delta_auprc")
    high_near = [row for row in rows if float(row["near_cap_fraction_all"]) >= 0.90]
    high_macro = _float_values(high_near, "delta_macro_f1_at_val")
    report = [
        "# Amazon CoVER-REL Residual Diagnostics 5-Seed",
        "",
        f"- Mean near-cap fraction: {mean(near_cap_vals):.6f}",
        f"- Near-cap / Delta AUPRC correlation: {_correlation(near_cap_vals, deltas):.6f}",
        f"- High near-cap rows: {len(high_near)}/{len(rows)}",
        f"- Mean Delta Macro-F1 on high near-cap rows: {mean(high_macro):.6f}" if high_macro else "- Mean Delta Macro-F1 on high near-cap rows: unavailable",
        "",
        "## Table",
        "",
        "\n".join(md_lines),
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(report) + "\n")


def write_prototype_quality(config: dict[str, Any], seeds: list[int]) -> None:
    dataset = config["dataset"]["name"]
    model = config["model"]["name"]
    rows: list[dict[str, Any]] = []
    for relation_set in RUNS:
        for seed in seeds:
            rel_path = relation_feature_path(dataset, model, seed, relation_set)
            rel_stats, meta = load_relation_stats(rel_path)
            relations = meta["relations"]
            for rel_idx, relation_name in enumerate(relations):
                rel_meta = meta["relation_meta"][relation_name]
                start = rel_idx * len(RELATION_STAT_NAMES)
                for stat_idx, stat_name in enumerate(RELATION_STAT_NAMES):
                    vals = rel_stats[:, start + stat_idx].float()
                    quantiles = torch.quantile(vals, torch.tensor([0.05, 0.50, 0.95]))
                    rows.append({
                        "relation_set": relation_set,
                        "seed": seed,
                        "relation": relation_name,
                        "stat": stat_name,
                        "mean": float(vals.mean().item()),
                        "std": float(vals.std(unbiased=False).item()),
                        "min": float(vals.min().item()),
                        "q05": float(quantiles[0].item()),
                        "q50": float(quantiles[1].item()),
                        "q95": float(quantiles[2].item()),
                        "max": float(vals.max().item()),
                        "degree_q10": rel_meta.get("degree_q10", ""),
                        "degree_q90": rel_meta.get("degree_q90", ""),
                        "feature_deviation_q90": rel_meta.get("feature_deviation_q90", ""),
                        "neighbor_cosine_q10": rel_meta.get("neighbor_cosine_q10", ""),
                        "fraud_proto_train_nodes": rel_meta.get("fraud_proto_train_nodes", ""),
                        "benign_proto_train_nodes": rel_meta.get("benign_proto_train_nodes", ""),
                        "prototype_labels": meta.get("prototype_labels", ""),
                        "test_label_used": meta.get("test_label_used", ""),
                    })
    csv_path = Path("artifacts/tables/amazon_cover_rel_prototype_quality_5seed.csv")
    report_path = Path("artifacts/reports/amazon_cover_rel_prototype_quality_5seed.md")
    write_csv(csv_path, rows)
    scale_rows = [
        row for row in rows
        if row["stat"] in {
            "feature_l2_deviation_z",
            "neighbor_cosine",
            "fraud_proto_dist_z",
            "benign_proto_dist_z",
            "fraud_minus_benign_margin_z",
        }
    ]
    fields = [
        "relation_set",
        "seed",
        "relation",
        "stat",
        "mean",
        "std",
        "q05",
        "q50",
        "q95",
        "degree_q90",
        "fraud_proto_train_nodes",
        "benign_proto_train_nodes",
        "prototype_labels",
        "test_label_used",
    ]
    md_lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in scale_rows[:80]:
        md_lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    report = [
        "# Amazon CoVER-REL Prototype / Relation-Feature Quality 5-Seed",
        "",
        "- Relation feature stats are normalized or z-scored where applicable.",
        "- Prototype rows are generated from train labels only; `test_label_used` remains false.",
        f"- CSV rows: {len(rows)}",
        "",
        "## Key Distributions",
        "",
        "\n".join(md_lines),
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cover-rel-gj/stage3_legacy/stage3_cover_rel_amazon_nollm.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    seed_rows = write_stability_outputs(config, args.seeds)
    write_residual_diagnostics(config, seed_rows, args.seeds)
    write_prototype_quality(config, args.seeds)
    print(json.dumps({
        "stability_csv": "artifacts/tables/amazon_fresh_bwgnn_vs_cover_rel_uvu_all_5seed.csv",
        "residual_csv": "artifacts/tables/amazon_cover_rel_residual_diagnostics_5seed.csv",
        "prototype_csv": "artifacts/tables/amazon_cover_rel_prototype_quality_5seed.csv",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
