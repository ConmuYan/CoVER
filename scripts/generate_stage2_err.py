from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.adapter import EvidenceAdapter
from evidence.prompt import assert_score_blind_payload, build_teacher_payload
from evidence.rule_teacher import RuleTeacher
from models.gnn import build_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--teacher", type=str, default="rule", choices=["rule"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    seed = config["train"]["seed"]
    model_name = config["model"]["name"]

    torch.manual_seed(seed)

    if args.debug:
        print("[DEBUG] Using tiny synthetic graph, 64 trace nodes")
        data = load_fraud_dataset("tiny", seed=seed)
        trace_size = 64
    else:
        data = load_fraud_dataset(dataset_name, path=dataset_path, seed=seed)
        trace_size = config["evidence"].get("trace_size", 2000)

    model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    )

    checkpoint_path = Path("artifacts") / "checkpoints" / dataset_name / model_name / f"seed_{seed}" / "base.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}, using random weights")

    model.eval()

    with torch.no_grad():
        base_logits, embeddings = model(data.x, data.edge_index)

    train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    if len(train_indices) > trace_size:
        import random
        random.seed(seed)
        trace_nodes = random.sample(train_indices, trace_size)
    else:
        trace_nodes = train_indices

    print(f"Selected {len(trace_nodes)} trace nodes")

    adapter = EvidenceAdapter(
        detector_name=model_name,
        x=data.x,
        edge_index=data.edge_index,
    )

    start_time = time.time()

    cards = adapter.extract(
        node_ids=trace_nodes,
        base_logits=base_logits,
        embeddings=embeddings,
    )

    teacher = RuleTeacher()
    errs = []
    score_blind_passed = True

    for card in cards:
        payload = build_teacher_payload(card)
        try:
            assert_score_blind_payload(payload)
        except ValueError as e:
            print(f"Score-blind check failed: {e}")
            score_blind_passed = False
            continue

        err = teacher.generate(card)
        errs.append(err)

    elapsed = time.time() - start_time

    out_dir = Path("artifacts") / "err_cache" / dataset_name / model_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "evidence_cards.jsonl", "w") as f:
        for card in cards:
            f.write(json.dumps({
                "node_id": card.node_id,
                "detector_name": card.detector_name,
                "calibration": {
                    "base_score": card.calibration.base_score,
                    "uncertainty": card.calibration.uncertainty,
                },
                "reasoning": {
                    "degree_level": card.reasoning.degree_level,
                    "neighbor_consistency": card.reasoning.neighbor_consistency,
                    "feature_neighbor_discrepancy": card.reasoning.feature_neighbor_discrepancy,
                    "detector_signal": card.reasoning.detector_signal,
                    "detector_signal_strength": card.reasoning.detector_signal_strength,
                    "counter_signal": card.reasoning.counter_signal,
                    "allowed_support_ids": card.reasoning.allowed_support_ids,
                    "allowed_counter_ids": card.reasoning.allowed_counter_ids,
                },
            }) + "\n")

    with open(out_dir / "teacher_payloads.jsonl", "w") as f:
        for card in cards:
            payload = build_teacher_payload(card)
            f.write(json.dumps(payload) + "\n")

    with open(out_dir / "rule_err.jsonl", "w") as f:
        for err in errs:
            f.write(json.dumps({
                "node_id": err.node_id,
                "risk_type": err.risk_type,
                "supporting_evidence": err.supporting_evidence,
                "counter_evidence": err.counter_evidence,
                "summary": err.summary,
            }) + "\n")

    stats = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "teacher": args.teacher,
        "num_trace_nodes": len(trace_nodes),
        "num_cards": len(cards),
        "num_err": len(errs),
        "score_blind_check_passed": score_blind_passed,
        "elapsed_seconds": elapsed,
    }

    with open(out_dir / "stage2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Stage 2 Results ===")
    print(f"  Trace nodes: {len(trace_nodes)}")
    print(f"  Evidence cards: {len(cards)}")
    print(f"  ERR generated: {len(errs)}")
    print(f"  Score-blind check: {'PASSED' if score_blind_passed else 'FAILED'}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
