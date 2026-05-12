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
from evidence.prompt import assert_score_blind_payload, build_llm_messages, build_teacher_payload
from evidence.rule_teacher import RuleTeacher
from evidence.schema import ERR
from evidence.verifier import EvidenceContractVerifier, load_contracts
from models.gnn import build_detector
from utils.paths import get_err_cache_dir, get_base_checkpoint_path, teacher_to_run_name, ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--teacher", type=str, default="rule", choices=["rule", "llm"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable_retry", action="store_true")
    parser.add_argument("--max_verifier_retries", type=int, default=None)
    parser.add_argument("--trace_size", type=int, default=None)
    parser.add_argument("--confirm_large_llm_run", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    seed = args.seed or config["train"]["seed"]
    model_name = config["model"]["name"]
    run_name = args.run_name or teacher_to_run_name(args.teacher)
    llm_config = config.get("llm", {})

    enable_retry = args.enable_retry or llm_config.get("enable_verifier_retry", False)
    max_verifier_retries = args.max_verifier_retries or llm_config.get("max_verifier_retries", 1)
    max_parse_retries = llm_config.get("max_parse_retries", 1)

    torch.manual_seed(seed)

    if args.debug:
        print("[DEBUG] Using tiny synthetic graph, 64 trace nodes")
        data = load_fraud_dataset("tiny", seed=seed)
        trace_size = 64
        if args.teacher == "llm":
            max_trace = llm_config.get("max_trace_nodes_debug", 8)
            trace_size = min(trace_size, max_trace)
    else:
        split_mode = config["dataset"].get("split_mode", "supervised")
        train_ratio = config["dataset"].get("train_ratio", 0.7)
        val_test_ratio = config["dataset"].get("val_test_ratio", [1, 2])

        data = load_fraud_dataset(
            dataset_name, path=dataset_path, seed=seed,
            split_mode=split_mode, train_ratio=train_ratio, val_test_ratio=val_test_ratio
        )
        if args.trace_size is not None:
            trace_size = args.trace_size
        else:
            trace_size = config["evidence"].get("trace_size", 2000)
            if args.teacher == "llm":
                trace_size = min(trace_size, llm_config.get("max_trace_nodes", 1000))

    if args.teacher == "llm" and trace_size > 64 and not args.confirm_large_llm_run:
        print(f"Error: trace_size={trace_size} > 64 for LLM teacher.")
        print(f"Use --confirm_large_llm_run to proceed.")
        sys.exit(1)

    model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    )

    checkpoint_path = get_base_checkpoint_path(dataset_name, model_name, seed)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}, using random weights")

    model.eval()

    with torch.no_grad():
        output = model(data.x, data.edge_index, return_output=True)
        base_logits = output.logits
        embeddings = output.embeddings
        extras = output.extras

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
        extras=extras,
    )

    if args.teacher == "llm":
        from evidence.llm_teacher import OfflineLLMTeacher

        llm_teacher = OfflineLLMTeacher(
            backend=llm_config.get("backend", "mock"),
            model_name_or_path=llm_config.get("model_name_or_path"),
            temperature=llm_config.get("temperature", 0.0),
            max_retries=max_parse_retries,
            max_new_tokens=llm_config.get("max_new_tokens", 256),
            device_map=llm_config.get("device_map", "auto"),
            torch_dtype=llm_config.get("torch_dtype", "auto"),
            trust_remote_code=llm_config.get("trust_remote_code", True),
            enable_verifier_retry=enable_retry,
            max_verifier_retries=max_verifier_retries,
            max_parse_retries=max_parse_retries,
        )
        teacher_name = f"llm_{llm_config.get('backend', 'mock')}"
    else:
        llm_teacher = None
        teacher_name = "rule"

    contracts = load_contracts()
    verifier = EvidenceContractVerifier(contracts, enable_label_compatibility=False)

    accepted_errs = []
    rejected_errs = []
    all_errs = []
    card_map = {}
    score_blind_passed = True
    all_prompts = []
    all_raw_outputs = []
    num_accepted_after_initial = 0
    num_accepted_after_retry = 0
    num_verifier_retried = 0

    rule_teacher = RuleTeacher()

    for card in cards:
        payload = build_teacher_payload(card)
        try:
            assert_score_blind_payload(payload)
        except ValueError as e:
            print(f"Score-blind check failed: {e}")
            score_blind_passed = False
            continue

        if args.teacher == "llm":
            messages = build_llm_messages(payload)
            all_prompts.append({
                "node_id": card.node_id,
                "attempt_id": 0,
                "attempt_type": "initial",
                "messages": messages,
            })

            if enable_retry:
                err, metadata = llm_teacher.generate_with_verifier_retry(payload, verifier, card)
                attempts = metadata.get("attempts", [])
                for i, attempt in enumerate(attempts):
                    all_raw_outputs.append({
                        "node_id": card.node_id,
                        "attempt_id": i,
                        "attempt_type": "initial" if i == 0 else "verifier_retry",
                        "raw_output": attempt.get("raw_output"),
                        "parsed_ok": attempt.get("parsed_ok", False),
                        "parse_error": attempt.get("parse_error"),
                        "verifier_accepted": metadata.get("final_status") == "accepted" or metadata.get("final_status") == "accepted_after_retry",
                        "reject_reasons": attempt.get("reject_reasons", []),
                    })

                if metadata.get("final_status") in ("accepted", "accepted_after_retry"):
                    num_accepted_after_initial += 1 if metadata.get("accepted_after_retry") is False else 0
                    num_accepted_after_retry += 1 if metadata.get("accepted_after_retry") is True else 0
                    if metadata.get("verifier_retries", 0) > 0:
                        num_verifier_retried += 1
                    accepted_errs.append(err)
                else:
                    rejected_errs.append({
                        "err": err,
                        "reasons": metadata.get("reject_reasons", []),
                        "num_attempts": len(attempts),
                    })
            else:
                err, metadata = llm_teacher.generate(payload)
                all_raw_outputs.append({
                    "node_id": card.node_id,
                    "attempt_id": 0,
                    "attempt_type": "initial",
                    "raw_output": metadata.get("raw_output"),
                    "parsed_ok": metadata.get("parsed_ok", False),
                    "parse_error": metadata.get("parse_error"),
                })

                if err is None:
                    err = ERR(
                        node_id=card.node_id,
                        risk_type="weak_or_uncertain_evidence",
                        supporting_evidence=[],
                        counter_evidence=[],
                        summary="LLM parse failed",
                    )

                accepted, reasons = verifier.verify(err, card)
                if accepted:
                    accepted_errs.append(err)
                    num_accepted_after_initial += 1
                else:
                    rejected_errs.append({"err": err, "reasons": reasons, "num_attempts": 1})

            all_errs.append(err)
            card_map[err.node_id] = card
        else:
            err = rule_teacher.generate(card)
            all_errs.append(err)
            card_map[err.node_id] = card

            accepted, reasons = verifier.verify(err, card)
            if accepted:
                accepted_errs.append(err)
            else:
                rejected_errs.append({"err": err, "reasons": reasons})

    elapsed = time.time() - start_time

    reject_reason_counts: dict[str, int] = {}
    for item in rejected_errs:
        for r in item.get("reasons", []):
            reject_reason_counts[r] = reject_reason_counts.get(r, 0) + 1

    out_dir = ensure_dir(get_err_cache_dir(dataset_name, model_name, run_name, seed))

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
        for err in all_errs:
            f.write(json.dumps({
                "node_id": err.node_id,
                "risk_type": err.risk_type,
                "supporting_evidence": err.supporting_evidence,
                "counter_evidence": err.counter_evidence,
                "summary": err.summary,
            }) + "\n")

    if args.teacher == "llm":
        with open(out_dir / "prompts.jsonl", "w") as f:
            for item in all_prompts:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(out_dir / "raw_llm_outputs.jsonl", "w") as f:
            for item in all_raw_outputs:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

        with open(out_dir / "llm_err.jsonl", "w") as f:
            for err in all_errs:
                f.write(json.dumps({
                    "node_id": err.node_id,
                    "risk_type": err.risk_type,
                    "supporting_evidence": err.supporting_evidence,
                    "counter_evidence": err.counter_evidence,
                    "summary": err.summary,
                }) + "\n")

    with open(out_dir / "accepted_err.jsonl", "w") as f:
        for err in accepted_errs:
            f.write(json.dumps({
                "node_id": err.node_id,
                "risk_type": err.risk_type,
                "supporting_evidence": err.supporting_evidence,
                "counter_evidence": err.counter_evidence,
                "summary": err.summary,
            }) + "\n")

    with open(out_dir / "rejected_err.jsonl", "w") as f:
        for item in rejected_errs:
            err = item["err"]
            f.write(json.dumps({
                "node_id": err.node_id,
                "risk_type": err.risk_type,
                "supporting_evidence": err.supporting_evidence,
                "counter_evidence": err.counter_evidence,
                "summary": err.summary,
                "reject_reasons": item.get("reasons", []),
                "num_attempts": item.get("num_attempts", 1),
            }) + "\n")

    num_total = len(all_errs)
    num_accepted = len(accepted_errs)
    num_rejected = len(rejected_errs)
    acceptance_rate = num_accepted / num_total if num_total > 0 else 0.0

    verifier_stats = {
        "num_total": num_total,
        "num_accepted": num_accepted,
        "num_rejected": num_rejected,
        "acceptance_rate": acceptance_rate,
        "reject_reason_counts": reject_reason_counts,
        "label_compatibility_enabled": False,
        "contract_file": str(Path(__file__).parent.parent / "evidence" / "contracts.yaml"),
    }

    with open(out_dir / "verifier_stats.json", "w") as f:
        json.dump(verifier_stats, f, indent=2)

    stats = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "teacher": args.teacher,
        "num_trace_nodes": len(trace_nodes),
        "num_cards": len(cards),
        "num_err": len(all_errs),
        "score_blind_check_passed": score_blind_passed,
        "verifier_enabled": True,
        "num_accepted": num_accepted,
        "num_rejected": num_rejected,
        "acceptance_rate": acceptance_rate,
        "elapsed_seconds": elapsed,
    }

    if args.teacher == "llm":
        num_parse_success = sum(1 for m in all_raw_outputs if m.get("parsed_ok", False))
        num_parse_failed = len(all_raw_outputs) - num_parse_success
        stats.update({
            "llm_backend": llm_config.get("backend", "mock"),
            "llm_model_name_or_path": llm_config.get("model_name_or_path"),
            "enable_verifier_retry": enable_retry,
            "max_verifier_retries": max_verifier_retries,
            "max_parse_retries": max_parse_retries,
            "num_initial_calls": len(cards),
            "num_total_llm_calls": len(all_raw_outputs),
            "num_parse_success": num_parse_success,
            "num_parse_failed": num_parse_failed,
            "num_verifier_retried": num_verifier_retried,
            "num_accepted_after_initial": num_accepted_after_initial,
            "num_accepted_after_retry": num_accepted_after_retry,
            "num_final_accepted": num_accepted,
            "num_final_rejected": num_rejected,
            "final_acceptance_rate": acceptance_rate,
            "max_new_tokens": llm_config.get("max_new_tokens", 256),
            "temperature": llm_config.get("temperature", 0.0),
        })

    with open(out_dir / "stage2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Stage 2 Results ===")
    print(f"  Teacher: {args.teacher}")
    if args.teacher == "llm":
        print(f"  LLM Backend: {llm_config.get('backend', 'mock')}")
        print(f"  Verifier retry: {'enabled' if enable_retry else 'disabled'}")
    print(f"  Trace nodes: {len(trace_nodes)}")
    print(f"  Evidence cards: {len(cards)}")
    print(f"  ERR generated: {len(all_errs)}")
    print(f"  Score-blind check: {'PASSED' if score_blind_passed else 'FAILED'}")
    print(f"  Verifier: {num_accepted} accepted, {num_rejected} rejected ({acceptance_rate:.1%})")
    if args.teacher == "llm" and enable_retry:
        print(f"  Accepted after initial: {num_accepted_after_initial}")
        print(f"  Accepted after retry: {num_accepted_after_retry}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
