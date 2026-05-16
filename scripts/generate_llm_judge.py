from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.judge_encoder import encode_judge_records, save_judge_features
from evidence.judge_verifier import audit_forbidden_fields, verify_judge_output
from utils.paths import ensure_dir


MODEL_PATH = "/data1/mq/models/Qwen3-4B-Instruct-2507"


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_judge_messages(packet: dict[str, Any]) -> list[dict[str, str]]:
    allowed = packet.get("allowed_support_fields", [])
    system = (
        "You are a score-blind fraud-evidence judge. Return JSON only. "
        "Use only evidence field names shown in the packet. "
        "Do not infer from any model score. Do not assume any hidden class annotation."
    )
    user = {
        "task": "Judge whether this node is more likely fake, real, or uncertain.",
        "rules": [
            "Use only fields in the packet.",
            "Copy evidence field names exactly into supporting_evidence, counter_evidence, and uncertainty_factors.",
            "Use at most 3 supporting_evidence fields, at most 3 counter_evidence fields, and at most 3 uncertainty_factors.",
            "Always include all JSON keys. Use an empty array when counter_evidence or uncertainty_factors is absent.",
            "Do not provide long reasoning. short_explanation must be concise and evidence-grounded.",
        ],
        "output_schema": {
            "node_id": "int",
            "verdict": "fake|real|uncertain",
            "evidence_strength": "weak|moderate|strong",
            "key_relation": "one relation from relation_schema",
            "supporting_evidence": ["field name from allowed_evidence_fields"],
            "counter_evidence": ["field name from allowed_evidence_fields"],
            "uncertainty_factors": ["field name from allowed_evidence_fields"],
            "short_explanation": "short human-facing sentence",
        },
        "allowed_evidence_fields": allowed,
        "packet": packet,
        "required_empty_array_example": {
            "counter_evidence": [],
            "uncertainty_factors": [],
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no_json_object")
    return json.loads(cleaned[start : end + 1])


def mock_judge(packet: dict[str, Any]) -> dict[str, Any]:
    primary = str(packet["primary_relation"])
    rel = packet["relation_evidence"][primary]
    support = [f"relation_evidence.{primary}.prototype_margin_bucket"]
    counter: list[str] = []
    verdict = "uncertain"
    strength = "weak"
    if "fraud_like" in rel.get("prototype_margin_bucket", ""):
        verdict, strength = "fake", "moderate"
        support.append(f"relation_evidence.{primary}.feature_deviation_bucket")
    elif rel.get("neighbor_consistency_bucket") == "high":
        verdict, strength = "real", "weak"
        counter.append(f"relation_evidence.{primary}.neighbor_consistency_bucket")
    return {
        "node_id": packet["node_id"],
        "verdict": verdict,
        "evidence_strength": strength,
        "key_relation": primary,
        "supporting_evidence": support,
        "counter_evidence": counter,
        "uncertainty_factors": [] if verdict != "uncertain" else support,
        "short_explanation": f"{primary} relation evidence gives a {strength} {verdict} judgement.",
    }


def load_qwen(model_path: str, device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype, "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def vllm_generate(packets: list[dict[str, Any]], args) -> list[tuple[dict[str, Any], str, dict[str, Any] | None, str | None]]:
    """High-throughput vLLM batched generation.

    Builds chat-formatted prompts via the model's tokenizer, then dispatches
    them to vLLM's continuous batching engine. ~10-30x faster than HF
    transformers ``model.generate`` on a single 3090 for Qwen-class 4B models.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts: list[str] = []
    for packet in packets:
        messages = build_judge_messages(packet)
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=False,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    rows: list[tuple[dict[str, Any], str, dict[str, Any] | None, str | None]] = []
    for packet, output in zip(packets, outputs):
        raw = output.outputs[0].text.strip()
        try:
            rows.append((packet, raw, extract_json_object(raw), None))
        except Exception as exc:
            rows.append((packet, raw, None, str(exc)))
    return rows


@torch.inference_mode()
def qwen_generate(packets: list[dict[str, Any]], args) -> list[tuple[dict[str, Any], str, dict[str, Any] | None, str | None]]:
    tokenizer, model = load_qwen(args.model_path, args.device, args.dtype)
    rows: list[tuple[dict[str, Any], str, dict[str, Any] | None, str | None]] = []
    for start in range(0, len(packets), args.batch_size):
        batch = packets[start : start + args.batch_size]
        texts: list[str] = []
        for packet in batch:
            messages = build_judge_messages(packet)
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        outputs = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        prompt_len = encoded["input_ids"].shape[1]
        for packet, output in zip(batch, outputs):
            raw = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
            try:
                rows.append((packet, raw, extract_json_object(raw), None))
            except Exception as exc:
                rows.append((packet, raw, None, str(exc)))
    return rows


def completed_nodes(output_dir: Path) -> set[int]:
    done: set[int] = set()
    for name in ("accepted_judge.jsonl", "rejected_judge.jsonl"):
        for row in load_jsonl(output_dir / name):
            if "node_id" in row:
                done.add(int(row["node_id"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--backend", choices=["qwen", "vllm", "mock"], default="qwen")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--num_graph_nodes", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="vLLM GPU memory utilization (default 0.85).")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="vLLM max model context length (default 4096).")
    parser.add_argument("--kv_cache_dtype", type=str, default="auto",
                        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
                        help="vLLM KV cache dtype (use fp8 for big models on tight memory).")
    parser.add_argument("--reverify_existing", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    packets_path = Path(args.packets_path)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else packets_path.parent)
    packets = load_jsonl(packets_path)
    packet_by_node = {int(packet["node_id"]): packet for packet in packets}
    if args.num_nodes is not None:
        packets = packets[: args.num_nodes]
    if args.resume:
        done = completed_nodes(output_dir)
        packets = [packet for packet in packets if int(packet["node_id"]) not in done]

    if args.reverify_existing:
        generated = []
        for row in load_jsonl(output_dir / "llm_judge_raw.jsonl"):
            node_id = int(row["node_id"])
            raw = row.get("raw_output", "")
            try:
                parsed = extract_json_object(raw)
                generated.append((packet_by_node[node_id], raw, parsed, None))
            except Exception as exc:
                generated.append((packet_by_node[node_id], raw, None, str(exc)))
        args.resume = False
    else:
        if args.backend == "mock":
            generated = [
                (packet, json.dumps(mock_judge(packet)), mock_judge(packet), None)
                for packet in packets
            ]
        elif args.backend == "vllm":
            generated = vllm_generate(packets, args)
        else:
            generated = qwen_generate(packets, args)

    raw_rows = load_jsonl(output_dir / "llm_judge_raw.jsonl") if args.resume else []
    structured_rows = load_jsonl(output_dir / "llm_judge_structured.jsonl") if args.resume else []
    accepted = load_jsonl(output_dir / "accepted_judge.jsonl") if args.resume else []
    rejected = load_jsonl(output_dir / "rejected_judge.jsonl") if args.resume else []
    audits: list[dict[str, Any]] = []

    for packet, raw, parsed, parse_error in generated:
        node_id = int(packet["node_id"])
        prompt = json.dumps(build_judge_messages(packet), ensure_ascii=False)
        raw_rows.append({"node_id": node_id, "raw_output": raw, "parse_error": parse_error})
        if parsed is None:
            rejected.append({"node_id": node_id, "raw_output": raw, "reasons": [parse_error or "parse_failed"]})
            continue
        check = verify_judge_output(packet, parsed)
        audit = audit_forbidden_fields(packet, prompt, parsed)
        enters_fusion_training = check.accepted and audit["passed"]
        audits.append({
            "node_id": node_id,
            "verifier_accepted": check.accepted,
            "entered_fusion_training": enters_fusion_training,
            **audit,
        })
        structured_rows.append({"node_id": node_id, "judge": parsed, "accepted": check.accepted, "reasons": check.reasons})
        if enters_fusion_training:
            assert check.record is not None
            accepted.append(check.record)
        else:
            rejected.append({"node_id": node_id, "judge": parsed, "raw_output": raw, "reasons": check.reasons + audit["violations"]})

    accepted = _dedupe(accepted)
    rejected = _dedupe(rejected)
    write_jsonl(output_dir / "llm_judge_raw.jsonl", raw_rows)
    write_jsonl(output_dir / "llm_judge_structured.jsonl", structured_rows)
    write_jsonl(output_dir / "accepted_judge.jsonl", accepted)
    write_jsonl(output_dir / "rejected_judge.jsonl", rejected)
    write_jsonl(output_dir / "judge_forbidden_field_audit.jsonl", audits)

    meta_path = packets_path.parent / "judge_packet_meta.json"
    packet_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rel_meta_path = Path(packet_meta.get("relation_feature_meta_path", ""))
    rel_meta = json.loads(rel_meta_path.read_text()) if rel_meta_path.exists() else {}
    num_graph_nodes = args.num_graph_nodes or int(rel_meta.get("num_nodes", max([p["node_id"] for p in packets], default=0) + 1))
    evidence_fields = sorted({field for packet in load_jsonl(packets_path) for field in packet.get("allowed_support_fields", [])})
    features, mask, feature_meta = encode_judge_records(
        accepted,
        num_nodes=num_graph_nodes,
        relation_names=packet_meta.get("relation_schema", []),
        evidence_fields=evidence_fields,
        primary_relation=packet_meta.get("primary_relation", ""),
    )
    feature_meta["num_rejected"] = len(rejected)
    save_judge_features(output_dir, features, mask, feature_meta)

    stats = {
        "num_packets_processed": len(generated),
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "acceptance_rate": len(accepted) / max(1, len(accepted) + len(rejected)),
        "verdict_distribution": dict(Counter(row["verdict"] for row in accepted)),
        "strength_distribution": dict(Counter(row["evidence_strength"] for row in accepted)),
        "backend": args.backend,
        "model_path": args.model_path if args.backend in ("qwen", "vllm") else "",
    }
    (output_dir / "judge_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    prompt_packet_violations = [
        row for row in audits
        if not row.get("packet_forbidden_field_free", False)
        or not row.get("prompt_forbidden_field_free", False)
    ]
    accepted_violations = [
        row for row in audits
        if row.get("entered_fusion_training", False)
        and not row.get("output_forbidden_field_free", False)
    ]
    generated_violations = [row for row in audits if not row.get("passed", False)]
    rejected_forbidden_violations = [
        row for row in generated_violations
        if not row.get("entered_fusion_training", False)
        and not row.get("output_forbidden_field_free", False)
    ]
    audit_summary = {
        "passed": not prompt_packet_violations and not accepted_violations,
        "accepted_outputs_passed": not accepted_violations,
        "prompt_packet_passed": not prompt_packet_violations,
        "num_audited": len(audits),
        "num_generated_violations": len(generated_violations),
        "num_prompt_packet_violations": len(prompt_packet_violations),
        "num_accepted_violations": len(accepted_violations),
        "num_rejected_forbidden_violations": len(rejected_forbidden_violations),
    }
    (output_dir / "judge_forbidden_field_audit.json").write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), **stats}, indent=2, sort_keys=True))


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_node = {int(row["node_id"]): row for row in rows if "node_id" in row}
    return [by_node[node_id] for node_id in sorted(by_node)]


if __name__ == "__main__":
    main()
