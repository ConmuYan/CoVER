from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.adapter import EvidenceAdapter
from evidence.prompt import (
    assert_score_blind_payload, build_llm_messages, build_teacher_payload,
    build_contrastive_directional_messages,
)
from evidence.rule_teacher import RuleTeacher
from evidence.schema import ERR
from evidence.verifier import EvidenceContractVerifier, load_contracts
from models.gnn import build_detector
from utils.paths import get_err_cache_dir, get_base_checkpoint_path, teacher_to_run_name, ensure_dir


TOKEN_CACHE_VERSION = "cover_dir_v3"


def _git_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _set_cpu_threads(num_threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)


def _err_to_record(err: ERR) -> dict:
    return {
        "node_id": err.node_id,
        "risk_type": err.risk_type,
        "supporting_evidence": err.supporting_evidence,
        "counter_evidence": err.counter_evidence,
        "summary": err.summary,
        "evidence_direction": getattr(err, "evidence_direction", "uncertain"),
        "evidence_strength": getattr(err, "evidence_strength", "weak"),
        "uncertainty_factors": getattr(err, "uncertainty_factors", []),
    }


def _card_to_record(card) -> dict:
    return {
        "node_id": card.node_id,
        "detector_name": card.detector_name,
        "calibration": dataclasses.asdict(card.calibration),
        "reasoning": dataclasses.asdict(card.reasoning),
    }


def _write_evidence_cards(path: Path, cards: list) -> None:
    with open(path, "w") as f:
        for card in cards:
            f.write(json.dumps(_card_to_record(card), ensure_ascii=False) + "\n")


def _record_to_err(rec: dict) -> ERR:
    return ERR(
        node_id=rec["node_id"],
        risk_type=rec["risk_type"],
        supporting_evidence=rec.get("supporting_evidence", []),
        counter_evidence=rec.get("counter_evidence", []),
        summary=rec.get("summary", ""),
        evidence_direction=rec.get("evidence_direction", "uncertain"),
        evidence_strength=rec.get("evidence_strength", "weak"),
        uncertainty_factors=rec.get("uncertainty_factors", []),
    )


def _append_jsonl(handle, rec: dict) -> None:
    handle.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _completed_node_ids(out_dir: Path) -> set[int]:
    completed = set()
    for name in ("accepted_err.partial.jsonl", "rejected_err.partial.jsonl", "llm_err.partial.jsonl"):
        for rec in _load_jsonl(out_dir / name):
            node_id = rec.get("node_id")
            if node_id is None and isinstance(rec.get("err"), dict):
                node_id = rec["err"].get("node_id")
            if node_id is not None:
                completed.add(int(node_id))
    return completed


def _dedupe_by_node(rows: list[dict]) -> list[dict]:
    by_node = {}
    for rec in rows:
        node_id = rec.get("node_id")
        if node_id is None and isinstance(rec.get("err"), dict):
            node_id = rec["err"].get("node_id")
        if node_id is not None:
            by_node[int(node_id)] = rec
    return [by_node[k] for k in sorted(by_node)]


def _reference_cases(bank: dict, limit: int = 5) -> list[dict]:
    return [{"node_id": int(node_id), "tokens": tokens} for node_id, tokens in list(bank.items())[:limit]]


def _augment_directional_payload(payload: dict, prototypes: dict) -> dict:
    fraud_summary = prototypes.get("fraud_prototype_summary", {})
    benign_summary = prototypes.get("benign_prototype_summary", {})
    payload["fraud_prototype"] = fraud_summary.get("field_modes", {})
    payload["benign_prototype"] = benign_summary.get("field_modes", {})
    payload["fraud_prototype_summary"] = {
        "distinctive_tokens": fraud_summary.get("distinctive_tokens", []),
        "token_frequencies": fraud_summary.get("token_frequencies", {}),
        "num_nodes": fraud_summary.get("num_nodes", 0),
    }
    payload["benign_prototype_summary"] = {
        "distinctive_tokens": benign_summary.get("distinctive_tokens", []),
        "token_frequencies": benign_summary.get("token_frequencies", {}),
        "num_nodes": benign_summary.get("num_nodes", 0),
    }
    payload["fraud_like_reference_cases"] = _reference_cases(prototypes.get("fraud_prototype_bank", {}))
    payload["benign_like_reference_cases"] = _reference_cases(prototypes.get("benign_prototype_bank", {}))
    payload["normal_structure_summary"] = prototypes.get("normal_structure_summary", {})
    reasoning = payload.get("reasoning", {})
    payload["prototype_stats"] = {
        "fraud_nodes": fraud_summary.get("num_nodes", 0),
        "benign_nodes": benign_summary.get("num_nodes", 0),
    }
    payload["retrieval_stats"] = {
        "fraud_like_reference_cases": len(payload["fraud_like_reference_cases"]),
        "benign_like_reference_cases": len(payload["benign_like_reference_cases"]),
        "evidence_polarity": reasoning.get("evidence_polarity", "unknown"),
        "fraud_token_count": reasoning.get("fraud_token_count", 0),
        "benign_token_count": reasoning.get("benign_token_count", 0),
        "neutral_token_count": reasoning.get("neutral_token_count", 0),
    }
    return payload


class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.step_idx = 0
        self.start = time.time()

    def step(self, name: str) -> None:
        self.step_idx += 1
        elapsed = time.time() - self.start
        eta = 0.0
        if self.step_idx > 1:
            eta = elapsed / (self.step_idx - 1) * (self.total - self.step_idx + 1)
        print(f"[{self.step_idx}/{self.total}] {name} (elapsed={elapsed:.1f}s eta={eta:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--teacher", type=str, default="rule", choices=["rule", "llm"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable_retry", action="store_true")
    parser.add_argument("--max_verifier_retries", type=int, default=None)
    parser.add_argument("--trace_size", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=None, help="Alias for --trace_size")
    parser.add_argument("--confirm_large_llm_run", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stratified", action="store_true", help="Use stratified split")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for LLM inference")
    parser.add_argument("--prompt_mode", type=str, default="enhanced",
                        choices=["enhanced", "contrastive_directional"],
                        help="Prompt mode for LLM teacher")
    parser.add_argument("--resume", action="store_true", help="Resume LLM generation from partial JSONL files")
    parser.add_argument("--refresh_token_cache", action="store_true", help="Recompute directional token cache")
    parser.add_argument("--build_payloads_only", action="store_true", help="Build payloads and exit before LLM calls")
    parser.add_argument("--cpu_threads", type=int, default=4, help="CPU threads for torch/BLAS/tokenizers")
    parser.add_argument("--trace_sampler", type=str, default=None, choices=["random", "error_aware", "counter_focused"])
    args = parser.parse_args()
    _set_cpu_threads(args.cpu_threads)
    progress = ProgressTracker(7)
    progress.step("Load config and data")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    seed = args.seed or config["train"]["seed"]
    model_name = config["model"]["name"]
    run_name = args.run_name or teacher_to_run_name(args.teacher)
    llm_config = config.get("llm", {})
    out_dir = ensure_dir(get_err_cache_dir(dataset_name, model_name, run_name, seed))

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
            split_mode=split_mode, train_ratio=train_ratio, val_test_ratio=val_test_ratio,
            stratified=args.stratified
        )
        if args.num_nodes is not None:
            trace_size = args.num_nodes
        elif args.trace_size is not None:
            trace_size = args.trace_size
        else:
            trace_size = config["evidence"].get("trace_size", 2000)
            if args.teacher == "llm":
                trace_size = min(trace_size, llm_config.get("max_trace_nodes", 1000))

    if args.teacher == "llm" and trace_size > 64 and not args.confirm_large_llm_run:
        print(f"Error: trace_size={trace_size} > 64 for LLM teacher.")
        print("Use --confirm_large_llm_run to proceed.")
        sys.exit(1)

    progress.step("Load detector and run base inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    ).to(device)

    checkpoint_path = get_base_checkpoint_path(dataset_name, model_name, seed)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}, using random weights")

    model.eval()

    with torch.no_grad():
        output = model(data.x.to(device), data.edge_index.to(device), return_output=True)
        base_logits = output.logits.cpu()
        embeddings = output.embeddings.cpu()
        extras_cpu = {}
        if output.extras:
            for k, v in output.extras.items():
                extras_cpu[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        extras = extras_cpu

    stage2_cfg = config.get("stage2", {})
    trace_sampler_mode = args.trace_sampler or stage2_cfg.get("trace_sampler", "random")
    trace_nodes_path = out_dir / "trace_nodes.json"
    counter_candidate_stats = None

    if args.resume and trace_nodes_path.exists():
        with open(trace_nodes_path) as f:
            trace_rec = json.load(f)
        trace_nodes = [int(n) for n in trace_rec.get("trace_nodes", trace_rec)]
        print(f"Resuming with saved trace nodes from {trace_nodes_path}: {len(trace_nodes)} nodes", flush=True)
    elif trace_sampler_mode == "error_aware":
        from evidence.trace_sampler import sample_traces, DEFAULT_RATIOS
        trace_nodes, sampling_stats = sample_traces(
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask if hasattr(data, "val_mask") and data.val_mask is not None else torch.zeros(data.y.shape[0], dtype=torch.bool),
            base_logits=base_logits,
            edge_index=data.edge_index,
            trace_size=trace_size,
            seed=seed,
            ratios=DEFAULT_RATIOS,
            extras=extras,
            output_dir=out_dir,
        )
        print(f"Error-aware sampling: {len(trace_nodes)} nodes from {len(sampling_stats['pool_sizes'])} pools")
        for pool_name, drawn in sampling_stats["pool_drawn"].items():
            print(f"  {pool_name}: {drawn}")
        with open(trace_nodes_path, "w") as f:
            json.dump({
                "trace_nodes": trace_nodes,
                "trace_size": trace_size,
                "sampler": trace_sampler_mode,
                "sampling_stats": sampling_stats,
            }, f, indent=2)
    elif trace_sampler_mode == "counter_focused":
        from evidence.trace_sampler import build_counter_focused_candidates
        trace_nodes, counter_candidate_stats = build_counter_focused_candidates(
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask if hasattr(data, "val_mask") and data.val_mask is not None else torch.zeros(data.y.shape[0], dtype=torch.bool),
            base_logits=base_logits,
            trace_size=trace_size,
            seed=seed,
        )
        print(
            f"Counter-focused candidate sampling: {len(trace_nodes)} candidates "
            f"from {len(counter_candidate_stats['pool_sizes'])} pools",
            flush=True,
        )
        for pool_name, drawn in counter_candidate_stats["candidate_pool_drawn"].items():
            print(f"  {pool_name}: {drawn}", flush=True)
    else:
        train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
        if len(train_indices) > trace_size:
            import random
            random.seed(seed)
            trace_nodes = random.sample(train_indices, trace_size)
        else:
            trace_nodes = train_indices
        with open(trace_nodes_path, "w") as f:
            json.dump({
                "trace_nodes": trace_nodes,
                "trace_size": trace_size,
                "sampler": trace_sampler_mode,
            }, f, indent=2)

    print(f"Selected {len(trace_nodes)} trace nodes")

    # Build prototypes for directional mode (must happen before card extraction)
    progress.step("Build directional prototypes")
    prototypes = {}
    old_prototypes = None
    if args.teacher == "llm" and args.prompt_mode == "contrastive_directional":
        from evidence.prototypes import PrototypeBuilder
        train_node_ids = data.train_mask.nonzero(as_tuple=True)[0].tolist()
        print(f"Building prototypes from {len(train_node_ids)} train nodes...")
        _adapter_tmp = EvidenceAdapter(detector_name=model_name, x=data.x, edge_index=data.edge_index)
        token_cache_dir = ensure_dir(Path("artifacts") / "token_cache")
        cache_file = token_cache_dir / f"{dataset_name}_{model_name}_seed{seed}_{TOKEN_CACHE_VERSION}_train_tokens.json"
        if cache_file.exists() and not args.refresh_token_cache:
            print(f"Loading token cache from {cache_file}", flush=True)
            with open(cache_file) as f:
                cached = json.load(f)
            if isinstance(cached, dict) and "train_tokens" in cached:
                train_tokens_dict = {int(k): v for k, v in cached["train_tokens"].items()}
            else:
                train_tokens_dict = {int(k): v for k, v in cached.items()}
        else:
            print(f"Extracting train graph tokens for cache {cache_file}", flush=True)
            train_tokens_dict = _adapter_tmp.generate_graph_evidence_tokens_vectorized(
                node_ids=train_node_ids, base_logits=base_logits, embeddings=embeddings,
                extras=extras or None, prototypes=None,
            )
            with open(cache_file, "w") as f:
                json.dump({
                    "metadata": {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "cache_version": TOKEN_CACHE_VERSION,
                        "git_hash": _git_hash(),
                    },
                    "train_tokens": {str(k): v for k, v in train_tokens_dict.items()},
                }, f, indent=2)
            print(f"Saved token cache to {cache_file}", flush=True)
        evidence_tokens = {nid: {t: "active" for t in toks} for nid, toks in train_tokens_dict.items()}
        base_preds = (torch.sigmoid(base_logits) >= 0.5).long()
        builder = PrototypeBuilder()
        prototypes = builder.build(
            train_mask=data.train_mask, y=data.y,
            evidence_tokens=evidence_tokens, base_preds=base_preds,
            graph_tokens=train_tokens_dict,
        )
        old_prototypes = {
            "fraud_prototype": prototypes.get("normal_structure_summary", {}).get("field_modes", {}),
            "benign_prototype": prototypes.get("normal_structure_summary", {}).get("field_modes", {}),
            "fraud_prototype_summary": prototypes.get("fraud_prototype_summary", {}),
            "benign_prototype_summary": prototypes.get("benign_prototype_summary", {}),
        }
        print(
            "Prototypes built: "
            f"fraud={prototypes.get('fraud_prototype_summary', {}).get('num_nodes', 0)} nodes, "
            f"benign={prototypes.get('benign_prototype_summary', {}).get('num_nodes', 0)} nodes",
            flush=True,
        )

    adapter = EvidenceAdapter(
        detector_name=model_name,
        x=data.x,
        edge_index=data.edge_index,
    )

    start_time = time.time()

    if trace_sampler_mode == "counter_focused" and not (args.resume and trace_nodes_path.exists()):
        candidate_cards = adapter.extract_batch(
            node_ids=trace_nodes,
            base_logits=base_logits,
            embeddings=embeddings,
            extras=extras,
            prototypes=old_prototypes,
            show_progress=True,
            progress_desc="Counter candidates",
        )
        polarity_by_node = {
            card.node_id: card.reasoning.evidence_polarity
            for card in candidate_cards
        }
        from evidence.trace_sampler import sample_counter_focused_traces
        trace_nodes, sampling_stats = sample_counter_focused_traces(
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask if hasattr(data, "val_mask") and data.val_mask is not None else torch.zeros(data.y.shape[0], dtype=torch.bool),
            base_logits=base_logits,
            trace_size=trace_size,
            seed=seed,
            polarity_by_node=polarity_by_node,
            candidate_nodes=[card.node_id for card in candidate_cards],
            output_dir=out_dir,
            candidate_stats=counter_candidate_stats,
        )
        selected_nodes = set(trace_nodes)
        cards = [card for card in candidate_cards if card.node_id in selected_nodes]
        with open(trace_nodes_path, "w") as f:
            json.dump({
                "trace_nodes": trace_nodes,
                "trace_size": trace_size,
                "sampler": trace_sampler_mode,
                "sampling_stats": sampling_stats,
            }, f, indent=2)
        print(f"Counter-focused final sampling selected {len(trace_nodes)} trace nodes", flush=True)
        for pool_name, drawn in sampling_stats["pool_drawn"].items():
            print(f"  {pool_name}: {drawn}", flush=True)
    else:
        cards = adapter.extract_batch(
            node_ids=trace_nodes,
            base_logits=base_logits,
            embeddings=embeddings,
            extras=extras,
            prototypes=old_prototypes,
            show_progress=True,
            progress_desc="Stage2 cards",
        )
    print(f"Extracted {len(cards)} evidence cards")

    progress.step("Build and dump teacher payloads")
    payload_by_node: dict[int, dict] = {}
    with open(out_dir / "teacher_payloads.jsonl", "w") as f:
        for card in cards:
            payload = build_teacher_payload(card)
            if args.prompt_mode == "contrastive_directional":
                payload = _augment_directional_payload(payload, prototypes)
            assert_score_blind_payload(payload)
            payload_by_node[card.node_id] = payload
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if args.build_payloads_only:
        _write_evidence_cards(out_dir / "evidence_cards.jsonl", cards)
        print(f"Built {len(payload_by_node)} payloads only; no LLM calls made.", flush=True)
        print(f"Artifacts saved to: {out_dir}", flush=True)
        return

    progress.step("Initialize teacher and verifier")
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
            batch_size=args.batch_size,
            prompt_mode=args.prompt_mode,
        )
    else:
        llm_teacher = None

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

    if args.teacher == "llm" and llm_teacher is not None:
        valid_cards = []
        valid_payloads = []
        
        msgs_fn = build_contrastive_directional_messages if args.prompt_mode == "contrastive_directional" else build_llm_messages

        for card in tqdm(cards, desc="Score-blind check", ncols=80):
            payload = payload_by_node[card.node_id]
            try:
                assert_score_blind_payload(payload)
                valid_cards.append(card)
                valid_payloads.append(payload)
                all_prompts.append({
                    "node_id": card.node_id,
                    "attempt_id": 0,
                    "attempt_type": "initial",
                    "messages": msgs_fn(payload),
                })
            except ValueError as e:
                print(f"Score-blind check failed for node {card.node_id}: {e}")
                score_blind_passed = False

        if args.resume:
            for rec in _dedupe_by_node(_load_jsonl(out_dir / "accepted_err.partial.jsonl")):
                accepted_errs.append(_record_to_err(rec))
            for rec in _dedupe_by_node(_load_jsonl(out_dir / "rejected_err.partial.jsonl")):
                err = _record_to_err(rec)
                rejected_errs.append({
                    "err": err,
                    "reasons": rec.get("reject_reasons", []),
                    "num_attempts": rec.get("num_attempts", 1),
                })
            for rec in _dedupe_by_node(_load_jsonl(out_dir / "llm_err.partial.jsonl")):
                err = _record_to_err(rec)
                all_errs.append(err)
                if err.node_id in payload_by_node:
                    card_map[err.node_id] = next(c for c in cards if c.node_id == err.node_id)

        completed = _completed_node_ids(out_dir) if args.resume else set()
        if completed:
            print(f"Resuming: skipping {len(completed)} completed node_ids", flush=True)
        pairs = [(c, p) for c, p in zip(valid_cards, valid_payloads) if c.node_id not in completed]
        print(f"Processing {len(pairs)} remaining valid cards with batch size {args.batch_size}...", flush=True)

        raw_partial = open(out_dir / "raw_llm_outputs.partial.jsonl", "a")
        llm_partial = open(out_dir / "llm_err.partial.jsonl", "a")
        accepted_partial = open(out_dir / "accepted_err.partial.jsonl", "a")
        rejected_partial = open(out_dir / "rejected_err.partial.jsonl", "a")
        try:
            for start in range(0, len(pairs), max(args.batch_size, 1)):
                batch = pairs[start:start + max(args.batch_size, 1)]
                batch_cards = [x[0] for x in batch]
                batch_payloads = [x[1] for x in batch]
                print(
                    f"[LLM] batch {start // max(args.batch_size, 1) + 1}: "
                    f"{start + 1}-{start + len(batch)}/{len(pairs)}",
                    flush=True,
                )
                if enable_retry:
                    batch_results = llm_teacher.generate_batch_with_verifier_retry(
                        batch_payloads, verifier, batch_cards
                    )
                else:
                    batch_results = llm_teacher.generate_batch(batch_payloads)

                for i, (err, metadata) in enumerate(batch_results):
                    card = batch_cards[i]
                    raw_rec = {
                        "node_id": card.node_id,
                        "attempt_id": 0,
                        "attempt_type": "initial",
                        "raw_output": metadata.get("raw_output"),
                        "parsed_ok": metadata.get("parsed_ok", False),
                        "parse_error": metadata.get("parse_error"),
                    }
                    all_raw_outputs.append(raw_rec)
                    _append_jsonl(raw_partial, raw_rec)

                    if err is None:
                        err = ERR(
                            node_id=card.node_id,
                            risk_type="weak_or_uncertain_evidence",
                            supporting_evidence=[],
                            counter_evidence=[],
                            summary="LLM parse failed",
                        )

                    if enable_retry:
                        attempts = metadata.get("attempts", [])
                        for j, attempt in enumerate(attempts[1:], 1):
                            retry_rec = {
                                "node_id": card.node_id,
                                "attempt_id": j,
                                "attempt_type": "verifier_retry",
                                "raw_output": attempt.get("raw_output"),
                                "parsed_ok": attempt.get("parsed_ok", False),
                                "parse_error": attempt.get("parse_error"),
                                "verifier_accepted": metadata.get("final_status") in ("accepted", "accepted_after_retry"),
                                "reject_reasons": attempt.get("reject_reasons", []),
                            }
                            all_raw_outputs.append(retry_rec)
                            _append_jsonl(raw_partial, retry_rec)

                        accepted_now = metadata.get("final_status") in ("accepted", "accepted_after_retry")
                        reasons = metadata.get("reject_reasons", [])
                        if accepted_now:
                            if metadata.get("accepted_after_retry"):
                                num_accepted_after_retry += 1
                            else:
                                num_accepted_after_initial += 1
                            if metadata.get("verifier_retries", 0) > 0:
                                num_verifier_retried += 1
                        else:
                            rejected_errs.append({"err": err, "reasons": reasons, "num_attempts": len(attempts)})
                    else:
                        accepted_now = False
                        reasons = ["parse_failed"]
                        if metadata.get("parsed_ok", False):
                            accepted_now, reasons = verifier.verify(err, card)
                        if accepted_now:
                            num_accepted_after_initial += 1
                        else:
                            rejected_errs.append({"err": err, "reasons": reasons, "num_attempts": 1})

                    if accepted_now:
                        accepted_errs.append(err)
                        _append_jsonl(accepted_partial, _err_to_record(err))
                    else:
                        rej_rec = _err_to_record(err)
                        rej_rec["reject_reasons"] = reasons
                        rej_rec["num_attempts"] = len(metadata.get("attempts", [])) if enable_retry else 1
                        _append_jsonl(rejected_partial, rej_rec)

                    all_errs.append(err)
                    card_map[err.node_id] = card
                    _append_jsonl(llm_partial, _err_to_record(err))
        finally:
            raw_partial.close()
            llm_partial.close()
            accepted_partial.close()
            rejected_partial.close()

    else:
        for card in tqdm(cards, desc="Rule teacher", ncols=80):
            payload = build_teacher_payload(card)
            try:
                assert_score_blind_payload(payload)
            except ValueError as e:
                print(f"Score-blind check failed: {e}")
                score_blind_passed = False
                continue

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

    _write_evidence_cards(out_dir / "evidence_cards.jsonl", cards)

    with open(out_dir / "teacher_payloads.jsonl", "w") as f:
        for card in cards:
            f.write(json.dumps(payload_by_node[card.node_id], ensure_ascii=False) + "\n")

    with open(out_dir / "rule_err.jsonl", "w") as f:
        for err in all_errs:
            rec = {
                "node_id": err.node_id,
                "risk_type": err.risk_type,
                "supporting_evidence": err.supporting_evidence,
                "counter_evidence": err.counter_evidence,
                "summary": err.summary,
            }
            if args.prompt_mode == "contrastive_directional":
                rec["evidence_direction"] = getattr(err, "evidence_direction", "uncertain")
                rec["evidence_strength"] = getattr(err, "evidence_strength", "weak")
                rec["uncertainty_factors"] = getattr(err, "uncertainty_factors", [])
            f.write(json.dumps(rec) + "\n")

    if args.teacher == "llm":
        with open(out_dir / "prompts.jsonl", "w") as f:
            for item in all_prompts:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(out_dir / "raw_llm_outputs.jsonl", "w") as f:
            for item in all_raw_outputs:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

        with open(out_dir / "llm_err.jsonl", "w") as f:
            for err in all_errs:
                rec = {
                    "node_id": err.node_id,
                    "risk_type": err.risk_type,
                    "supporting_evidence": err.supporting_evidence,
                    "counter_evidence": err.counter_evidence,
                    "summary": err.summary,
                }
                if args.prompt_mode == "contrastive_directional":
                    rec["evidence_direction"] = getattr(err, "evidence_direction", "uncertain")
                    rec["evidence_strength"] = getattr(err, "evidence_strength", "weak")
                    rec["uncertainty_factors"] = getattr(err, "uncertainty_factors", [])
                f.write(json.dumps(rec) + "\n")

    with open(out_dir / "accepted_err.jsonl", "w") as f:
        for err in accepted_errs:
            rec = {
                "node_id": err.node_id,
                "risk_type": err.risk_type,
                "supporting_evidence": err.supporting_evidence,
                "counter_evidence": err.counter_evidence,
                "summary": err.summary,
            }
            if args.prompt_mode == "contrastive_directional":
                rec["evidence_direction"] = getattr(err, "evidence_direction", "uncertain")
                rec["evidence_strength"] = getattr(err, "evidence_strength", "weak")
                rec["uncertainty_factors"] = getattr(err, "uncertainty_factors", [])
            f.write(json.dumps(rec) + "\n")

    with open(out_dir / "rejected_err.jsonl", "w") as f:
        for item in rejected_errs:
            err = item["err"]
            if err is None:
                continue
            rec = _err_to_record(err)
            rec["reject_reasons"] = item.get("reasons", [])
            rec["num_attempts"] = item.get("num_attempts", 1)
            f.write(json.dumps(rec) + "\n")

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

    rejected_report = {
        "rejected_reason_counts": reject_reason_counts,
        "num_rejected": num_rejected,
        "examples": [
            {
                **_err_to_record(item["err"]),
                "reject_reasons": item.get("reasons", []),
                "num_attempts": item.get("num_attempts", 1),
            }
            for item in rejected_errs[:5]
            if item.get("err") is not None
        ],
    }
    with open(out_dir / "rejected_reason_report.json", "w") as f:
        json.dump(rejected_report, f, indent=2)

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
            "num_initial_calls": len(valid_payloads) if args.teacher == "llm" else len(cards),
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
            "batch_size": args.batch_size,
            "prompt_mode": args.prompt_mode,
        })

    with open(out_dir / "stage2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Stage 2 Results ===")
    print(f"  Teacher: {args.teacher}")
    if args.teacher == "llm":
        print(f"  LLM Backend: {llm_config.get('backend', 'mock')}")
        print(f"  Batch size: {args.batch_size}")
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
    print(f"  Speed: {len(trace_nodes)/elapsed:.1f} nodes/sec")
    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
