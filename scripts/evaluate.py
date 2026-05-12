from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.vocab import encode_reasoning, get_evidence_slots, get_reason_types
from models.gnn import build_detector
from models.reasoner import EvidenceReasoner, VALID_GATE_MODES
from training.metrics import compute_metrics, compute_metrics_with_threshold
from utils.paths import get_checkpoint_dir, get_results_dir, get_err_cache_dir, get_base_checkpoint_path, get_reasoner_checkpoint_path, ensure_dir
from utils.threshold import find_best_threshold, evaluate_with_threshold


def load_evidence_cards(path: Path) -> dict[int, dict]:
    cards = {}
    if not path.exists():
        return cards
    with open(path) as f:
        for line in f:
            card = json.loads(line)
            cards[card["node_id"]] = card
    return cards


def _run_threshold_calibration(
    y_val_np,
    prob_val_np,
    y_test_np,
    prob_test_np,
    threshold_mode: str,
) -> dict:
    """Run threshold calibration on validation set and evaluate on both splits.

    Returns dict with best_threshold, val_metrics, test_metrics.
    """
    metric_key = "f1" if threshold_mode == "val_f1" else "macro_f1"
    best_threshold, best_val_score, _ = find_best_threshold(
        y_val_np, prob_val_np, metric=metric_key,
    )
    val_metrics = evaluate_with_threshold(y_val_np, prob_val_np, best_threshold)
    test_metrics = evaluate_with_threshold(y_test_np, prob_test_np, best_threshold)

    return {
        "best_threshold": best_threshold,
        "best_val_metric_score": best_val_score,
        "calibration_metric": metric_key,
        "val": val_metrics,
        "test": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage3"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_name", type=str, default="base")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stratified", action="store_true", help="Use stratified split")
    parser.add_argument(
        "--threshold_mode",
        type=str,
        default="fixed",
        choices=["fixed", "val_f1", "val_macro_f1"],
        help="Threshold mode: fixed (0.5), val_f1 (best F1 on val), val_macro_f1 (best Macro-F1 on val)",
    )
    parser.add_argument("--gate_mode", type=str, default=None, choices=VALID_GATE_MODES)
    parser.add_argument("--delta_scale", type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    seed = args.seed or config["train"]["seed"]
    model_name = config["model"]["name"]
    run_name = args.run_name

    torch.manual_seed(seed)
    device = torch.device(config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    if args.debug:
        print("[DEBUG] Using tiny synthetic graph")
        data = load_fraud_dataset("tiny", seed=seed)
    else:
        split_mode = config["dataset"].get("split_mode", "supervised")
        train_ratio = config["dataset"].get("train_ratio", 0.7)
        val_test_ratio = config["dataset"].get("val_test_ratio", [1, 2])
        stratified = config["dataset"].get("stratified", False)

        data = load_fraud_dataset(
            dataset_name, path=dataset_path, seed=seed,
            split_mode=split_mode, train_ratio=train_ratio, val_test_ratio=val_test_ratio,
            stratified=stratified,
        )

    base_model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    ).to(device)

    base_path = get_base_checkpoint_path(dataset_name, model_name, seed)

    if base_path.exists():
        state = torch.load(base_path, weights_only=True)
        base_model.load_state_dict(state)
        print(f"Loaded base checkpoint from {base_path}")

    base_model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    test_mask = data.test_mask.to(device)
    val_mask = data.val_mask.to(device) if hasattr(data, "val_mask") and data.val_mask is not None else None

    with torch.no_grad():
        output = base_model(x, edge_index, return_output=True)
        base_logits = output.logits
        z = output.embeddings

    if args.stage == "stage1":
        prob_test = torch.sigmoid(base_logits[test_mask]).cpu().numpy()
        y_test_np = y[test_mask].cpu().numpy()
        metrics = compute_metrics(y_test_np, prob_test)

        print("\n=== Stage 1 Evaluation ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        results_dir = ensure_dir(get_results_dir(dataset_name, model_name, "base", seed))
        with open(results_dir / "stage1_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {results_dir / 'stage1_metrics.json'}")

        if args.threshold_mode != "fixed" and val_mask is not None:
            prob_val = torch.sigmoid(base_logits[val_mask]).cpu().numpy()
            y_val_np = y[val_mask].cpu().numpy()

            calibration = _run_threshold_calibration(
                y_val_np, prob_val, y_test_np, prob_test, args.threshold_mode,
            )

            print(f"\n=== Stage 1 Threshold Calibration ({args.threshold_mode}) ===")
            print(f"  Best threshold: {calibration['best_threshold']:.4f}")
            print(f"  Best val {calibration['calibration_metric']}: {calibration['best_val_metric_score']:.4f}")
            print(f"\n  Validation metrics:")
            for k, v in calibration["val"].items():
                print(f"    {k}: {v:.4f}")
            print(f"\n  Test metrics:")
            for k, v in calibration["test"].items():
                print(f"    {k}: {v:.4f}")

            fixed_metrics = compute_metrics_with_threshold(y_test_np, prob_test, 0.5)

            calibrated_result = {
                "threshold_mode": args.threshold_mode,
                "fixed_threshold_metrics": fixed_metrics,
                f"{args.threshold_mode}_threshold_metrics": calibration["test"],
                "calibration_info": {
                    "best_threshold": calibration["best_threshold"],
                    "calibration_metric": calibration["calibration_metric"],
                    "best_val_metric_score": calibration["best_val_metric_score"],
                    "val_metrics_at_best_threshold": calibration["val"],
                },
            }

            calibrated_path = results_dir / "stage1_calibrated_metrics.json"
            with open(calibrated_path, "w") as f:
                json.dump(calibrated_result, f, indent=2)
            print(f"\nCalibrated metrics saved to: {calibrated_path}")
        elif args.threshold_mode != "fixed" and val_mask is None:
            print("\n[WARNING] threshold_mode requires val_mask but none found. Falling back to fixed threshold.")

    elif args.stage == "stage3":
        reasoner_path = get_reasoner_checkpoint_path(dataset_name, model_name, run_name, seed)
        if not reasoner_path.exists():
            print(f"Error: Reasoner checkpoint not found at {reasoner_path}")
            sys.exit(1)

        rc = config.get("reasoner", {})
        gate_mode = args.gate_mode or rc.get("gate_mode", "safe_residual")
        delta_scale = args.delta_scale or rc.get("delta_scale", 2.0)

        reasoner = EvidenceReasoner(
            z_dim=z.shape[1],
            hidden_dim=rc.get("hidden_dim", 128),
            rho=rc.get("rho", 0.3),
            gate_mode=gate_mode,
            delta_scale=delta_scale,
            gate_bias_init=rc.get("gate_bias_init", -2.0),
            residual_init_zero=rc.get("residual_init_zero", True),
        ).to(device)

        state = torch.load(reasoner_path, weights_only=True)
        reasoner.load_state_dict(state)
        reasoner.eval()

        err_cache_dir = get_err_cache_dir(dataset_name, model_name, run_name, seed)
        cards = load_evidence_cards(err_cache_dir / "evidence_cards.jsonl")

        num_slots = len(get_evidence_slots())
        num_nodes = data.x.shape[0]
        evidence_token_ids = torch.zeros(num_nodes, num_slots, dtype=torch.long)
        for node_id, card in cards.items():
            if node_id < num_nodes:
                reasoning = card.get("reasoning", {})
                evidence_token_ids[node_id] = encode_reasoning(reasoning)

        evidence_token_ids = evidence_token_ids.to(device)

        with torch.no_grad():
            outputs_test = reasoner(
                z[test_mask],
                base_logits[test_mask],
                evidence_token_ids[test_mask],
            )

        prob_test = torch.sigmoid(outputs_test["final_logit"]).cpu().numpy()
        y_test_np = y[test_mask].cpu().numpy()
        metrics = compute_metrics(y_test_np, prob_test)

        print("\n=== Stage 3 Evaluation ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        results_dir = ensure_dir(get_results_dir(dataset_name, model_name, run_name, seed))
        with open(results_dir / "stage3_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        reason_types = get_reason_types()
        type_probs = torch.softmax(outputs_test["type_logits"], dim=-1)
        pred_types = type_probs.argmax(dim=-1)

        print("\n=== Explanation Preview (first 5 test nodes) ===")
        test_indices = test_mask.nonzero(as_tuple=True)[0][:5]
        for i, idx in enumerate(test_indices):
            node_id = idx.item()
            pred_score = torch.sigmoid(outputs_test["final_logit"][i]).item()
            pred_type = reason_types[pred_types[i]]
            print(f"\nNode {node_id}:")
            print(f"  Predicted score: {pred_score:.4f}")
            print(f"  Predicted type: {pred_type}")

        print(f"\nMetrics saved to: {results_dir / 'stage3_metrics.json'}")

        if args.threshold_mode != "fixed" and val_mask is not None:
            with torch.no_grad():
                outputs_val = reasoner(
                    z[val_mask],
                    base_logits[val_mask],
                    evidence_token_ids[val_mask],
                )

            prob_val = torch.sigmoid(outputs_val["final_logit"]).cpu().numpy()
            y_val_np = y[val_mask].cpu().numpy()

            calibration = _run_threshold_calibration(
                y_val_np, prob_val, y_test_np, prob_test, args.threshold_mode,
            )

            print(f"\n=== Stage 3 Threshold Calibration ({args.threshold_mode}) ===")
            print(f"  Best threshold: {calibration['best_threshold']:.4f}")
            print(f"  Best val {calibration['calibration_metric']}: {calibration['best_val_metric_score']:.4f}")
            print(f"\n  Validation metrics:")
            for k, v in calibration["val"].items():
                print(f"    {k}: {v:.4f}")
            print(f"\n  Test metrics:")
            for k, v in calibration["test"].items():
                print(f"    {k}: {v:.4f}")

            fixed_test_metrics = compute_metrics_with_threshold(y_test_np, prob_test, 0.5)

            calibrated_result = {
                "threshold_mode": args.threshold_mode,
                "fixed_threshold_metrics": fixed_test_metrics,
                f"{args.threshold_mode}_threshold_metrics": calibration["test"],
                "calibration_info": {
                    "best_threshold": calibration["best_threshold"],
                    "calibration_metric": calibration["calibration_metric"],
                    "best_val_metric_score": calibration["best_val_metric_score"],
                    "val_metrics_at_best_threshold": calibration["val"],
                },
            }

            other_mode = "val_macro_f1" if args.threshold_mode == "val_f1" else "val_f1"
            other_metric_key = "macro_f1" if args.threshold_mode == "val_f1" else "f1"
            other_threshold, other_score, _ = find_best_threshold(
                y_val_np, prob_val, metric=other_metric_key,
            )
            other_test_metrics = evaluate_with_threshold(y_test_np, prob_test, other_threshold)
            other_val_metrics = evaluate_with_threshold(y_val_np, prob_val, other_threshold)
            calibrated_result[f"{other_mode}_threshold_metrics"] = other_test_metrics

            calibrated_path = results_dir / "stage3_calibrated_metrics.json"
            with open(calibrated_path, "w") as f:
                json.dump(calibrated_result, f, indent=2)
            print(f"\nCalibrated metrics saved to: {calibrated_path}")
        elif args.threshold_mode != "fixed" and val_mask is None:
            print("\n[WARNING] threshold_mode requires val_mask but none found. Falling back to fixed threshold.")


if __name__ == "__main__":
    main()
