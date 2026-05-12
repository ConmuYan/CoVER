from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.vocab import encode_reasoning, get_evidence_slots
from models.gnn import build_detector
from models.reasoner import EvidenceReasoner
from training.metrics import compute_metrics_with_threshold
from utils.paths import get_base_checkpoint_path, get_err_cache_dir, get_reasoner_checkpoint_path, ensure_dir
from utils.threshold import evaluate_with_threshold, find_best_threshold


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_evidence_cards(path: Path) -> dict[int, dict]:
    cards: dict[int, dict] = {}
    for card in load_jsonl(path):
        node_id = card.get("node_id")
        if isinstance(node_id, int):
            cards[node_id] = card
    return cards


def compute_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def compute_split_pos_rate_gap(data) -> dict[str, float]:
    y = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    pos_rate_train = float(y[train_mask].float().mean().item()) if int(train_mask.sum()) > 0 else 0.0
    pos_rate_val = float(y[val_mask].float().mean().item()) if int(val_mask.sum()) > 0 else 0.0
    pos_rate_test = float(y[test_mask].float().mean().item()) if int(test_mask.sum()) > 0 else 0.0
    split_pos_rates = [pos_rate_train, pos_rate_val, pos_rate_test]
    max_gap = max(split_pos_rates) - min(split_pos_rates)
    global_pos_rate = float(y.float().mean().item()) if y.numel() > 0 else 0.0
    relative_gap = max_gap / global_pos_rate if global_pos_rate > 0 else 0.0

    return {
        "pos_rate_train": pos_rate_train,
        "pos_rate_val": pos_rate_val,
        "pos_rate_test": pos_rate_test,
        "global_pos_rate": global_pos_rate,
        "max_pos_rate_gap": max_gap,
        "relative_pos_rate_gap": relative_gap,
    }


def compute_risk_type_diversity(errs: list[dict]) -> dict[str, float]:
    counts = Counter(err.get("risk_type", "unknown") for err in errs if err.get("risk_type"))
    total = sum(counts.values())
    if total == 0:
        return {"risk_type_diversity": 0.0, "risk_type_unique_count": 0.0, "risk_type_entropy": 0.0}

    probs = [count / total for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    normalized_entropy = entropy / math.log2(len(counts)) if len(counts) > 1 else 0.0
    return {
        "risk_type_diversity": normalized_entropy,
        "risk_type_unique_count": float(len(counts)),
        "risk_type_entropy": entropy,
    }


def load_reasoner_inputs(dataset_name: str, model_name: str, run_name: str, seed: int, config_path: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_path = config["dataset"].get("path")
    split_mode = config["dataset"].get("split_mode", "supervised")
    train_ratio = config["dataset"].get("train_ratio", 0.7)
    val_test_ratio = config["dataset"].get("val_test_ratio", [1, 2])
    device = torch.device(config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    data = load_fraud_dataset(
        dataset_name,
        path=dataset_path,
        seed=seed,
        split_mode=split_mode,
        train_ratio=train_ratio,
        val_test_ratio=val_test_ratio,
    )

    base_model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    ).to(device)

    return config, data, device, base_model


@torch.no_grad()
def diagnose_reasoner_outputs(
    dataset_name: str,
    model_name: str,
    run_name: str,
    seed: int,
    config_path: Path,
    threshold_mode: str,
) -> dict:
    config, data, device, base_model = load_reasoner_inputs(dataset_name, model_name, run_name, seed, config_path)

    base_path = get_base_checkpoint_path(dataset_name, model_name, seed)
    reasoner_path = get_reasoner_checkpoint_path(dataset_name, model_name, run_name, seed)
    err_cache_dir = get_err_cache_dir(dataset_name, model_name, run_name, seed)

    missing: list[str] = []
    if not base_path.exists():
        missing.append(f"missing base checkpoint: {base_path}")
    if not reasoner_path.exists():
        missing.append(f"missing reasoner checkpoint: {reasoner_path}")

    cards_path = err_cache_dir / "evidence_cards.jsonl"
    accepted_path = err_cache_dir / "accepted_err.jsonl"
    if not cards_path.exists():
        missing.append(f"missing evidence cards: {cards_path}")
    if not accepted_path.exists():
        missing.append(f"missing accepted ERR: {accepted_path}")

    if missing:
        return {
            "dataset": dataset_name,
            "model": model_name,
            "run_name": run_name,
            "seed": seed,
            "status": "missing",
            "missing": missing,
        }

    base_state = torch.load(base_path, map_location=device, weights_only=True)
    reasoner_state = torch.load(reasoner_path, map_location=device, weights_only=True)
    base_model.load_state_dict(base_state)
    base_model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    with torch.no_grad():
        base_output = base_model(x, edge_index, return_output=True)
        base_logits = base_output.logits
        z = base_output.embeddings

    reasoner = EvidenceReasoner(
        z_dim=z.shape[1],
        hidden_dim=config.get("reasoner", {}).get("hidden_dim", 128),
        rho=config.get("reasoner", {}).get("rho", 0.3),
    ).to(device)
    reasoner.load_state_dict(reasoner_state)
    reasoner.eval()

    evidence_cards = load_evidence_cards(cards_path)
    accepted_errs = load_jsonl(accepted_path)
    rejected_errs = load_jsonl(err_cache_dir / "rejected_err.jsonl")

    num_nodes = data.x.shape[0]
    num_slots = len(get_evidence_slots())
    evidence_token_ids = torch.zeros(num_nodes, num_slots, dtype=torch.long)
    for node_id, card in evidence_cards.items():
        if node_id < num_nodes:
            evidence_token_ids[node_id] = encode_reasoning(card.get("reasoning", {}))
    evidence_token_ids = evidence_token_ids.to(device)

    with torch.no_grad():
        val_outputs = reasoner(z[val_mask], base_logits[val_mask], evidence_token_ids[val_mask])
        test_outputs = reasoner(z[test_mask], base_logits[test_mask], evidence_token_ids[test_mask])

    y_val = y[val_mask].cpu().numpy()
    y_test = y[test_mask].cpu().numpy()
    prob_val = torch.sigmoid(val_outputs["final_logit"]).cpu().numpy()
    prob_test = torch.sigmoid(test_outputs["final_logit"]).cpu().numpy()
    base_prob_test = torch.sigmoid(base_logits[test_mask]).cpu().numpy()

    fixed_metrics = compute_metrics_with_threshold(y_test, prob_test, 0.5)
    threshold_metric = "macro_f1" if threshold_mode == "val_macro_f1" else "f1"
    calibrated_threshold, best_val_score, _ = find_best_threshold(y_val, prob_val, metric=threshold_metric)
    calibrated_metrics = evaluate_with_threshold(y_test, prob_test, calibrated_threshold)

    split_stats = compute_split_pos_rate_gap(data)
    risk_stats = compute_risk_type_diversity(accepted_errs + rejected_errs)
    logit_shift = float((test_outputs["final_logit"] - base_logits[test_mask]).mean().item()) if test_mask.sum() > 0 else 0.0

    diagnosis = {
        "dataset": dataset_name,
        "model": model_name,
        "run_name": run_name,
        "seed": seed,
        "status": "ok",
        "num_val": int(val_mask.sum().item()),
        "num_test": int(test_mask.sum().item()),
        "split_pos_rate_gap": split_stats,
        "logit_shift_mean": logit_shift,
        "positive_prediction_rate_0_5": float(fixed_metrics["positive_prediction_rate"]),
        "calibrated_threshold": float(calibrated_threshold),
        "calibration_metric": threshold_metric,
        "best_val_score": float(best_val_score),
        "fixed_metrics": {
            "f1": float(fixed_metrics["f1"]),
            "macro_f1": float(fixed_metrics["macro_f1"]),
            "positive_prediction_rate": float(fixed_metrics["positive_prediction_rate"]),
        },
        "calibrated_metrics": {
            "f1": float(calibrated_metrics["f1"]),
            "macro_f1": float(calibrated_metrics["macro_f1"]),
            "positive_prediction_rate": float(calibrated_metrics["positive_prediction_rate"]),
        },
        "risk_type_diversity": risk_stats,
        "num_accepted_err": len(accepted_errs),
        "num_rejected_err": len(rejected_errs),
        "base_test_probability_mean": float(base_prob_test.mean()) if base_prob_test.size > 0 else 0.0,
    }

    # Keep the output explicit enough for later analysis.
    diagnosis["delta_f1"] = diagnosis["calibrated_metrics"]["f1"] - diagnosis["fixed_metrics"]["f1"]
    diagnosis["delta_macro_f1"] = diagnosis["calibrated_metrics"]["macro_f1"] - diagnosis["fixed_metrics"]["macro_f1"]
    return diagnosis


def build_aggregate_report(records: list[dict], requested_total: int) -> dict:
    ok_records = [r for r in records if r.get("status") == "ok"]

    def collect(path: list[str]) -> list[float]:
        values: list[float] = []
        for record in ok_records:
            item: object = record
            for key in path:
                if not isinstance(item, dict) or key not in item:
                    item = None
                    break
                item = item[key]
            if isinstance(item, (int, float)):
                values.append(float(item))
        return values

    aggregate = {
        "requested_runs": requested_total,
        "available_runs": len(ok_records),
        "missing_runs": requested_total - len(ok_records),
        "split_pos_rate_gap_mean": {},
        "metrics": {},
        "risk_type_diversity": {},
    }

    for key in ["pos_rate_train", "pos_rate_val", "pos_rate_test", "global_pos_rate", "max_pos_rate_gap", "relative_pos_rate_gap"]:
        mean, std = compute_mean_std(collect(["split_pos_rate_gap", key]))
        aggregate["split_pos_rate_gap_mean"][key] = {"mean": mean, "std": std}

    for key in ["logit_shift_mean", "positive_prediction_rate_0_5", "calibrated_threshold", "best_val_score", "delta_f1", "delta_macro_f1"]:
        mean, std = compute_mean_std(collect([key]))
        aggregate["metrics"][key] = {"mean": mean, "std": std}

    for metric_key in ["f1", "macro_f1", "positive_prediction_rate"]:
        fixed_mean, fixed_std = compute_mean_std(collect(["fixed_metrics", metric_key]))
        calib_mean, calib_std = compute_mean_std(collect(["calibrated_metrics", metric_key]))
        aggregate["metrics"][f"fixed_{metric_key}"] = {"mean": fixed_mean, "std": fixed_std}
        aggregate["metrics"][f"calibrated_{metric_key}"] = {"mean": calib_mean, "std": calib_std}

    for key in ["risk_type_diversity", "risk_type_unique_count", "risk_type_entropy"]:
        mean, std = compute_mean_std(collect(["risk_type_diversity", key]))
        aggregate["risk_type_diversity"][key] = {"mean": mean, "std": std}

    return aggregate


def render_markdown(records: list[dict], aggregate: dict, threshold_mode: str) -> str:
    lines = [
        "# Reasoner Diagnosis Summary",
        "",
        f"Threshold calibration mode: `{threshold_mode}`",
        f"Requested runs: {aggregate['requested_runs']} | Available: {aggregate['available_runs']} | Missing: {aggregate['missing_runs']}",
        "",
        "## Per-run results",
        "",
        "| run_name | seed | status | split gap | logit shift | p(pred=1)@0.5 | cal. thr | F1 fixed | F1 cal. | Macro-F1 fixed | Macro-F1 cal. | risk diversity |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for record in records:
        if record.get("status") != "ok":
            lines.append(
                f"| {record['run_name']} | {record['seed']} | missing | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        lines.append(
            "| {run_name} | {seed} | ok | {gap:.4f} | {shift:.4f} | {ppr:.4f} | {thr:.4f} | {f1f:.4f} | {f1c:.4f} | {mf1f:.4f} | {mf1c:.4f} | {div:.4f} |".format(
                run_name=record["run_name"],
                seed=record["seed"],
                gap=record["split_pos_rate_gap"]["max_pos_rate_gap"],
                shift=record["logit_shift_mean"],
                ppr=record["positive_prediction_rate_0_5"],
                thr=record["calibrated_threshold"],
                f1f=record["fixed_metrics"]["f1"],
                f1c=record["calibrated_metrics"]["f1"],
                mf1f=record["fixed_metrics"]["macro_f1"],
                mf1c=record["calibrated_metrics"]["macro_f1"],
                div=record["risk_type_diversity"]["risk_type_diversity"],
            )
        )

    lines.extend([
        "",
        "## Aggregated metrics",
        "",
        f"- Avg split max pos rate gap: {aggregate['split_pos_rate_gap_mean']['max_pos_rate_gap']['mean']:.4f} ± {aggregate['split_pos_rate_gap_mean']['max_pos_rate_gap']['std']:.4f}",
        f"- Avg logit shift: {aggregate['metrics']['logit_shift_mean']['mean']:.4f} ± {aggregate['metrics']['logit_shift_mean']['std']:.4f}",
        f"- Avg positive_prediction_rate @ 0.5: {aggregate['metrics']['positive_prediction_rate_0_5']['mean']:.4f} ± {aggregate['metrics']['positive_prediction_rate_0_5']['std']:.4f}",
        f"- Avg calibrated threshold: {aggregate['metrics']['calibrated_threshold']['mean']:.4f} ± {aggregate['metrics']['calibrated_threshold']['std']:.4f}",
        f"- Avg F1 fixed vs calibrated: {aggregate['metrics']['fixed_f1']['mean']:.4f} → {aggregate['metrics']['calibrated_f1']['mean']:.4f}",
        f"- Avg Macro-F1 fixed vs calibrated: {aggregate['metrics']['fixed_macro_f1']['mean']:.4f} → {aggregate['metrics']['calibrated_macro_f1']['mean']:.4f}",
        f"- Avg risk_type diversity: {aggregate['risk_type_diversity']['risk_type_diversity']['mean']:.4f} ± {aggregate['risk_type_diversity']['risk_type_diversity']['std']:.4f}",
    ])
    return "\n".join(lines)


def resolve_config_path(dataset: str, model: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return Path("configs") / f"{dataset}_{model}.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="bwgnn")
    parser.add_argument("--run_names", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--threshold_mode",
        type=str,
        default="val_macro_f1",
        choices=["val_f1", "val_macro_f1"],
        help="Validation metric used to calibrate the threshold.",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.dataset, args.model, args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    records: list[dict] = []
    for run_name in args.run_names:
        for seed in args.seeds:
            print(f"Diagnosing {args.dataset}/{args.model}/{run_name}/seed_{seed} ...")
            try:
                record = diagnose_reasoner_outputs(
                    args.dataset,
                    args.model,
                    run_name,
                    seed,
                    config_path,
                    args.threshold_mode,
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    "dataset": args.dataset,
                    "model": args.model,
                    "run_name": run_name,
                    "seed": seed,
                    "status": "error",
                    "error": str(exc),
                }
            records.append(record)

    aggregate = build_aggregate_report(records, requested_total=len(args.run_names) * len(args.seeds))
    report = {
        "dataset": args.dataset,
        "model": args.model,
        "config_path": str(config_path),
        "threshold_mode": args.threshold_mode,
        "records": records,
        "aggregate": aggregate,
    }

    report_dir = ensure_dir(Path("artifacts") / "reports" / args.dataset / args.model)
    json_path = report_dir / "reasoner_diagnosis_summary.json"
    md_path = report_dir / "reasoner_diagnosis_summary.md"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w") as f:
        f.write(render_markdown(records, aggregate, args.threshold_mode))

    print("\nReasoner diagnosis summary")
    print(f"  Requested runs: {aggregate['requested_runs']}")
    print(f"  Available runs: {aggregate['available_runs']}")
    print(f"  Missing runs: {aggregate['missing_runs']}")
    print(f"  Avg split max pos rate gap: {aggregate['split_pos_rate_gap_mean']['max_pos_rate_gap']['mean']:.4f}")
    print(f"  Avg logit shift: {aggregate['metrics']['logit_shift_mean']['mean']:.4f}")
    print(f"  Avg positive_prediction_rate @ 0.5: {aggregate['metrics']['positive_prediction_rate_0_5']['mean']:.4f}")
    print(f"  Avg calibrated threshold: {aggregate['metrics']['calibrated_threshold']['mean']:.4f}")
    print(f"  Avg F1 fixed -> calibrated: {aggregate['metrics']['fixed_f1']['mean']:.4f} -> {aggregate['metrics']['calibrated_f1']['mean']:.4f}")
    print(f"  Avg Macro-F1 fixed -> calibrated: {aggregate['metrics']['fixed_macro_f1']['mean']:.4f} -> {aggregate['metrics']['calibrated_macro_f1']['mean']:.4f}")
    print(f"  Avg risk_type diversity: {aggregate['risk_type_diversity']['risk_type_diversity']['mean']:.4f}")
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
