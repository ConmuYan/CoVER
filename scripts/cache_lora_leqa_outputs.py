"""vLLM-based batch inference for LEQA LoRA adapter.

Reads evidence packets (judge_packets.jsonl or teacher_payloads.jsonl),
runs the LoRA-augmented Qwen3-4B to produce per-token quality scores,
and writes a parquet with columns: node_id, token, q, reason.

Forbidden-field audit runs on every generated JSON before write.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEQA LoRA batch inference via vLLM")
    p.add_argument("--adapter", type=str, required=True,
                   help="Path to LoRA adapter directory")
    p.add_argument("--packets", type=str, required=True,
                   help="Path to judge_packets.jsonl or teacher_payloads.jsonl")
    p.add_argument("--sample", type=int, default=-1,
                   help="Number of packets to sample (-1 = all, 100 for sanity)")
    p.add_argument("--out", type=str, required=True,
                   help="Output parquet path")
    p.add_argument("--base_model", type=str,
                   default="/data1/mq/models/Qwen3-4B-Instruct-2507")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_new_tokens", type=int, default=320)
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--relation_names", type=str, nargs="+",
                   default=["RUR", "RSR", "RTR"])
    return p.parse_args()


def load_packets(path: str, sample: int) -> list[dict]:
    """Load packets from JSONL, optionally sampling."""
    packets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            packets.append(json.loads(line))
    if 0 < sample < len(packets):
        import random
        random.seed(42)
        packets = random.sample(packets, sample)
    logger.info("Loaded %d packets from %s", len(packets), path)
    return packets


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from models.leqa_lora import (
        build_leqa_token_names,
        format_leqa_prompt,
        parse_leqa_json,
        audit_leqa_output,
    )
    from models.lora_qwen_loader import create_vllm_engine

    token_names = build_leqa_token_names(args.relation_names)

    # Load packets
    packets = load_packets(args.packets, args.sample)

    # Build prompts
    prompts = []
    node_ids = []
    for pkt in packets:
        node_id = pkt.get("node_id", -1)
        node_ids.append(node_id)
        # Remove calibration (score-blind: calibration contains base_score)
        pkt_clean = {k: v for k, v in pkt.items()
                     if k != "calibration"}
        prompts.append(format_leqa_prompt(json.dumps(pkt_clean, indent=None)))

    # Create vLLM engine
    logger.info("Creating vLLM engine with adapter: %s", args.adapter)
    engine, lora_request = create_vllm_engine(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    # Run inference
    logger.info("Running inference on %d packets...", len(prompts))
    outputs = engine.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    # Parse outputs and audit
    rows = []
    audit_fail_count = 0
    for i, output in enumerate(outputs):
        raw_text = output.outputs[0].text
        nid = node_ids[i]

        # Forbidden-field audit
        if not audit_leqa_output(raw_text):
            audit_fail_count += 1
            logger.warning("Forbidden field in output for node %d, using defaults", nid)
            q_list = [1.0] * len(token_names)
            reason_list = ["none"] * len(token_names)
        else:
            q_list, reason_ids = parse_leqa_json(raw_text, token_names)
            from models.leqa_lora import LEQA_REASON_LABELS
            reason_list = [LEQA_REASON_LABELS[r] for r in reason_ids]

        for t_idx, (tname, q, reason) in enumerate(
            zip(token_names, q_list, reason_list)
        ):
            rows.append({
                "node_id": nid,
                "token": tname,
                "q": float(q),
                "reason": reason,
            })

    if audit_fail_count > 0:
        logger.warning(
            "Forbidden-field audit: %d / %d packets failed",
            audit_fail_count, len(outputs),
        )

    # Write parquet
    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path), index=False)
    logger.info(
        "Wrote %d rows (%d packets x %d tokens) to %s",
        len(rows), len(outputs), len(token_names), out_path,
    )

    # Summary stats
    q_vals = df["q"].values
    logger.info("q stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                q_vals.mean(), q_vals.std(), q_vals.min(), q_vals.max())
    reason_dist = df["reason"].value_counts().to_dict()
    logger.info("reason distribution: %s", reason_dist)


if __name__ == "__main__":
    main()
