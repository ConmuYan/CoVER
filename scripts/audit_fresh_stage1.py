"""Audit fresh Stage1 baseline artifacts for BWGNN across YelpChi and Amazon.

Verifies checkpoints, splits, and metrics for 5 seeds. Computes calibrated
threshold metrics (val_macro_f1) by running inference on existing models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_from_mat
from models.gnn import build_detector
from training.metrics import compute_metrics, compute_metrics_with_threshold
from utils.paths import (
    ensure_dir,
    get_base_checkpoint_path,
    get_results_dir,
    get_stratified_split_meta_path,
    get_stratified_split_path,
)
from utils.threshold import find_best_threshold, evaluate_with_threshold

DATASETS = {
    "yelpchi": "/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd/datasets/YelpChi.mat",
    "amazon": "/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd/datasets/Amazon.mat",
}
SEEDS = [42, 123, 456, 789, 2026]
MODEL = "bwgnn"
CONFIG_PATHS = {
    "yelpchi": "/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd/configs/yelpchi_bwgnn.yaml",
    "amazon": "/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd/configs/amazon_bwgnn.yaml",
}

ARTIFACTS_ROOT = Path("/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd/artifacts")
REPORT_DIR = ARTIFACTS_ROOT / "reports"
TABLE_DIR = ARTIFACTS_ROOT / "tables"


def _check_artifacts(dataset: str, seed: int) -> dict:
    """Check existence of checkpoint, split, and metrics for one seed."""
    ckpt_path = get_base_checkpoint_path(dataset, MODEL, seed)
    split_path = get_stratified_split_path(dataset, True, seed)
    meta_path = get_stratified_split_meta_path(dataset, True, seed)
    metrics_path = get_results_dir(dataset, MODEL, "base", seed) / "stage1_metrics.json"

    return {
        "seed": seed,
        "checkpoint_exists": ckpt_path.exists(),
        "checkpoint_path": str(ckpt_path),
        "split_exists": split_path.exists(),
        "split_path": str(split_path),
        "meta_exists": meta_path.exists(),
        "meta_path": str(meta_path),
        "metrics_exists": metrics_path.exists(),
        "metrics_path": str(metrics_path),
    }


def _verify_split(meta: dict) -> dict:
    """Verify split integrity from split_meta.json."""
    n = meta["num_nodes"]
    expected_train = int(0.4 * n)
    expected_val = int(0.2 * n)
    expected_test = n - expected_train - expected_val
    actual_ratios = (
        meta["num_train"] / n,
        meta["num_val"] / n,
        meta["num_test"] / n,
    )

    overlaps = meta["mask_overlap_counts"]
    pos_rates = (meta["pos_rate_train"], meta["pos_rate_val"], meta["pos_rate_test"])

    return {
        "num_nodes": n,
        "ratios_approx": {k: round(v, 3) for k, v in zip(["train", "val", "test"], actual_ratios)},
        "ratio_ok": all(abs(r - t) < 0.02 for r, t in zip(actual_ratios, (0.4, 0.2, 0.4))),
        "overlaps_zero": all(v == 0 for v in overlaps.values()),
        "overlap_detail": overlaps,
        "pos_rates": {k: round(v, 4) for k, v in zip(["train", "val", "test"], pos_rates)},
        "max_pos_rate_gap": meta["max_pos_rate_gap"],
        "stratified": meta.get("stratified", False),
    }


def _load_and_evaluate(dataset: str, seed: int, device: str) -> dict | None:
    """Load model + data, run inference, compute fixed and calibrated metrics."""
    ckpt_path = get_base_checkpoint_path(dataset, MODEL, seed)
    if not ckpt_path.exists():
        return None

    data_path = DATASETS[dataset]
    data = load_from_mat(data_path)
    split = torch.load(
        get_stratified_split_path(dataset, True, seed), weights_only=True
    )
    data.train_mask = split["train_mask"]
    data.val_mask = split["val_mask"]
    data.test_mask = split["test_mask"]

    with open(CONFIG_PATHS[dataset]) as f:
        config = yaml.safe_load(f)

    device_obj = torch.device(device)
    model = build_detector(
        name=MODEL,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.3),
    ).to(device_obj)

    state = torch.load(ckpt_path, weights_only=True, map_location=device_obj)
    model.load_state_dict(state)
    model.eval()

    x = data.x.to(device_obj)
    edge_index = data.edge_index.to(device_obj)
    y = data.y

    with torch.no_grad():
        output = model(x, edge_index, return_output=True)
        logits = output.logits

    prob_test = torch.sigmoid(logits[data.test_mask]).cpu().numpy()
    y_test = y[data.test_mask].numpy()
    prob_val = torch.sigmoid(logits[data.val_mask]).cpu().numpy()
    y_val = y[data.val_mask].numpy()

    fixed_metrics = compute_metrics(y_test, prob_test)

    best_thresh, best_val_score, _ = find_best_threshold(y_val, prob_val, metric="macro_f1")
    cal_val = evaluate_with_threshold(y_val, prob_val, best_thresh)
    cal_test = evaluate_with_threshold(y_test, prob_test, best_thresh)

    return {
        "fixed": fixed_metrics,
        "calibrated": {
            "best_threshold": best_thresh,
            "best_val_macro_f1": best_val_score,
            "val": cal_val,
            "test": cal_test,
        },
    }


def _build_table_rows(all_results: dict) -> list[dict]:
    """Flatten results into CSV-friendly rows."""
    rows = []
    for dataset in ["yelpchi", "amazon"]:
        for seed in SEEDS:
            r = all_results.get(dataset, {}).get(seed)
            if r is None:
                rows.append({"dataset": dataset, "seed": seed, "status": "MISSING"})
                continue
            fixed = r["fixed"]
            cal = r["calibrated"]
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "status": "OK",
                "roc_auc": round(fixed["roc_auc"], 4),
                "auprc": round(fixed["auprc"], 4),
                "f1_0.5": round(fixed["f1"], 4),
                "macro_f1_0.5": round(fixed["macro_f1"], 4),
                "cal_threshold": round(cal["best_threshold"], 4),
                "cal_test_roc_auc": round(cal["test"]["roc_auc"], 4),
                "cal_test_auprc": round(cal["test"]["auprc"], 4),
                "cal_test_f1": round(cal["test"]["f1"], 4),
                "cal_test_macro_f1": round(cal["test"]["macro_f1"], 4),
            })
    return rows


def _save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row.get(h, "")) for h in headers))
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")


def _save_md_table(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")


def _build_report(
    artifact_checks: dict,
    split_audits: dict,
    eval_results: dict,
) -> str:
    """Build the full markdown audit report."""
    sections = ["# Fresh Stage1 Baseline Audit\n"]
    sections.append(f"Model: {MODEL}  \nSeeds: {SEEDS}\n")

    for dataset in ["yelpchi", "amazon"]:
        sections.append(f"\n## {dataset.upper()}\n")

        # Artifact checks
        sections.append("### Artifact Presence\n")
        sections.append("| Seed | Checkpoint | Split | Meta | Metrics |")
        sections.append("|------|-----------|-------|------|---------|")
        for seed in SEEDS:
            c = artifact_checks[dataset][seed]
            mark = lambda b: "OK" if b else "MISSING"
            sections.append(
                f"| {seed} | {mark(c['checkpoint_exists'])} | {mark(c['split_exists'])} "
                f"| {mark(c['meta_exists'])} | {mark(c['metrics_exists'])} |"
            )

        # Split audit
        sections.append("\n### Split Integrity\n")
        for seed in SEEDS:
            s = split_audits[dataset][seed]
            sections.append(f"\n**Seed {seed}**")
            sections.append(f"- Ratios: {s['ratios_approx']} (4:2:4 OK: {s['ratio_ok']})")
            sections.append(f"- Mask overlaps zero: {s['overlaps_zero']} ({s['overlap_detail']})")
            sections.append(f"- Pos rates: {s['pos_rates']}, max gap: {s['max_pos_rate_gap']:.6f}")
            sections.append(f"- Stratified: {s['stratified']}")

        # Evaluation results
        sections.append("\n### Baseline Metrics\n")
        sections.append(
            "| Seed | ROC-AUC | AUPRC | F1@0.5 | Macro-F1@0.5 |"
            " Cal-Thresh | Cal-F1 | Cal-Macro-F1 |"
        )
        sections.append(
            "|------|---------|-------|--------|-------------|"
            "------------|--------|-------------|"
        )
        for seed in SEEDS:
            r = eval_results.get(dataset, {}).get(seed)
            if r is None:
                sections.append(f"| {seed} | - | - | - | - | - | - | - |")
                continue
            f = r["fixed"]
            c = r["calibrated"]
            sections.append(
                f"| {seed} | {f['roc_auc']:.4f} | {f['auprc']:.4f} | {f['f1']:.4f} "
                f"| {f['macro_f1']:.4f} | {c['best_threshold']:.4f} "
                f"| {c['test']['f1']:.4f} | {c['test']['macro_f1']:.4f} |"
            )

        # Mean/std
        sections.append("\n### Summary (mean ± std)\n")
        valid = [(seed, eval_results[dataset][seed]) for seed in SEEDS if eval_results.get(dataset, {}).get(seed)]
        if valid:
            for label, mode in [("Fixed@0.5", "fixed"), ("Calibrated", "calibrated")]:
                roc = [v[mode]["roc_auc"] for _, v in valid] if mode == "fixed" else [v[mode]["test"]["roc_auc"] for _, v in valid]
                auprc = [v[mode]["auprc"] for _, v in valid] if mode == "fixed" else [v[mode]["test"]["auprc"] for _, v in valid]
                f1 = [v[mode]["f1"] for _, v in valid] if mode == "fixed" else [v[mode]["test"]["f1"] for _, v in valid]
                mf1 = [v[mode]["macro_f1"] for _, v in valid] if mode == "fixed" else [v[mode]["test"]["macro_f1"] for _, v in valid]
                sections.append(f"**{label}** ({len(valid)} seeds):")
                sections.append(
                    f"- ROC-AUC: {np.mean(roc):.4f} ± {np.std(roc):.4f}"
                    f"  AUPRC: {np.mean(auprc):.4f} ± {np.std(auprc):.4f}"
                )
                sections.append(
                    f"- F1: {np.mean(f1):.4f} ± {np.std(f1):.4f}"
                    f"  Macro-F1: {np.mean(mf1):.4f} ± {np.std(mf1):.4f}"
                )

    return "\n".join(sections)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    artifact_checks = {}
    split_audits = {}
    eval_results = {}

    for dataset in ["yelpchi", "amazon"]:
        print(f"\n{'='*60}")
        print(f"  Auditing {dataset.upper()}")
        print(f"{'='*60}")

        artifact_checks[dataset] = {}
        split_audits[dataset] = {}
        eval_results[dataset] = {}

        for seed in SEEDS:
            # Step 1: check artifacts
            ac = _check_artifacts(dataset, seed)
            artifact_checks[dataset][seed] = ac
            all_ok = ac["checkpoint_exists"] and ac["split_exists"] and ac["meta_exists"]
            print(f"\n  Seed {seed}: ckpt={ac['checkpoint_exists']} split={ac['split_exists']} meta={ac['meta_exists']} metrics={ac['metrics_exists']}")

            # Step 2: verify split
            if ac["meta_exists"]:
                with open(ac["meta_path"]) as f:
                    meta = json.load(f)
                sa = _verify_split(meta)
                split_audits[dataset][seed] = sa
                print(f"    Split: ratios_ok={sa['ratio_ok']} overlaps_zero={sa['overlaps_zero']} stratified={sa['stratified']}")
            else:
                split_audits[dataset][seed] = {"error": "no meta"}
                print(f"    Split: MISSING meta")

            # Step 3-4: evaluate
            if all_ok:
                print(f"    Running inference...")
                ev = _load_and_evaluate(dataset, seed, args.device)
                if ev is not None:
                    eval_results[dataset][seed] = ev
                    f = ev["fixed"]
                    c = ev["calibrated"]
                    print(f"    Fixed:  ROC-AUC={f['roc_auc']:.4f} AUPRC={f['auprc']:.4f} F1={f['f1']:.4f} Macro-F1={f['macro_f1']:.4f}")
                    print(f"    Cal@{c['best_threshold']:.4f}: F1={c['test']['f1']:.4f} Macro-F1={c['test']['macro_f1']:.4f}")
                else:
                    print(f"    Evaluation FAILED")
            else:
                print(f"    Skipping evaluation (missing artifacts)")

    # Save outputs
    rows = _build_table_rows(eval_results)
    csv_path = TABLE_DIR / "fresh_bwgnn_stage1_5seed.csv"
    md_table_path = TABLE_DIR / "fresh_bwgnn_stage1_5seed.md"
    report_path = REPORT_DIR / "fresh_stage1_audit.md"

    _save_csv(rows, csv_path)
    _save_md_table(rows, md_table_path)

    report = _build_report(artifact_checks, split_audits, eval_results)
    ensure_dir(REPORT_DIR)
    report_path.write_text(report)

    print(f"\n\nOutputs saved:")
    print(f"  Report:  {report_path}")
    print(f"  CSV:     {csv_path}")
    print(f"  MD table: {md_table_path}")


if __name__ == "__main__":
    main()
