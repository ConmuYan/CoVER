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

# Make `models/` importable when invoked as a top-level script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEQA LoRA batch inference via vLLM")
    p.add_argument("--adapter", type=str, required=True,
                   help="Path to LoRA adapter directory")
    p.add_argument("--packets", type=str, required=True,
                   help="Path to judge_packets.jsonl or teacher_payloads.jsonl")
    p.add_argument("--sample", type=int, default=-1,
                   help="Number of packets to sample (-1 = all, 100 for sanity)")
    p.add_argument("--balance_strategy", type=str, default="random",
                   choices=["random", "balanced"],
                   help="random: uniform sample; balanced: 50/50 fraud/benign "
                        "(uses --labels_path or auto-loads via data.load_fraud)")
    p.add_argument("--labels_path", type=str, default=None,
                   help="Optional JSONL with {node_id, label} per line. "
                        "If omitted in balanced mode, auto-loads YelpChi "
                        "labels via data.load_fraud (CPU, ~30s).")
    p.add_argument("--dataset", type=str, default="yelpchi",
                   choices=["yelpchi", "amazon"],
                   help="Dataset name for auto label loading (only used when "
                        "balance_strategy=balanced and labels_path is None).")
    p.add_argument("--out", type=str, required=True,
                   help="Output parquet path")
    p.add_argument("--base_model", type=str,
                   default="/data1/mq/models/Qwen3-4B-Instruct-2507")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_new_tokens", type=int, default=2048,
                   help="Output token budget. Empirical: 28-entry LEQA JSON "
                        "+ hallucinated extras ≈ 3500 chars ≈ 1100-1200 BPE "
                        "tokens. Pilot v5 with default 1024 truncated mid-"
                        "JSON. 2048 gives safe headroom.")
    p.add_argument("--max_model_len", type=int, default=4096,
                   help="vLLM context window. Packet+prompt can exceed 1024; "
                        "bumped to 4096 to be safe after pilot v3 1025-token "
                        "VLLMValidationError.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--relation_names", type=str, nargs="+",
                   default=["RUR", "RSR", "RTR"])
    return p.parse_args()


def _load_labels(labels_path: str | None, dataset: str) -> dict[int, int]:
    """Load {node_id: label} dict.  Score-blind constraint applies to the
    LoRA input/output stream; the sampler is offline data engineering and
    is allowed to use labels for stratification."""
    if labels_path is not None:
        labels = {}
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                labels[int(rec["node_id"])] = int(rec["label"])
        logger.info("Loaded %d labels from %s", len(labels), labels_path)
        return labels
    # Auto-load via scipy.io.loadmat (avoids torch_geometric import
    # which would CUDA-init the parent process and break vLLM subprocess fork).
    import scipy.io as sio
    logger.info("Auto-loading labels for dataset=%s ...", dataset)
    data_root = (
        Path(__file__).resolve().parent.parent / "datasets"
        / ("YelpChi.mat" if dataset == "yelpchi" else "Amazon.mat")
    )
    mat = sio.loadmat(str(data_root))
    y = mat["label"].reshape(-1).tolist()
    labels = {int(i): int(l) for i, l in enumerate(y)}
    logger.info("Auto-loaded %d labels (fraud=%d, %.1f%%)",
                len(labels), sum(labels.values()),
                100.0 * sum(labels.values()) / max(len(labels), 1))
    return labels


def load_packets(
    path: str,
    sample: int,
    balance_strategy: str = "random",
    labels: dict[int, int] | None = None,
) -> list[dict]:
    """Load packets from JSONL, optionally sampling.

    balance_strategy:
      - "random": uniform random sample
      - "balanced": 50% fraud + 50% benign (requires labels dict)
    """
    packets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            packets.append(json.loads(line))
    if 0 < sample < len(packets):
        import random
        rng = random.Random(42)
        if balance_strategy == "balanced":
            if labels is None:
                raise ValueError("balanced sampling requires labels")
            fraud = [p for p in packets if labels.get(p.get("node_id"), 0) == 1]
            benign = [p for p in packets if labels.get(p.get("node_id"), 0) == 0]
            half = max(sample // 2, 1)
            f_take = min(half, len(fraud))
            b_take = min(sample - f_take, len(benign))
            f_take = min(f_take, sample - b_take)  # in case benign short
            sampled = rng.sample(fraud, f_take) + rng.sample(benign, b_take)
            rng.shuffle(sampled)
            logger.info(
                "Balanced sample: %d fraud + %d benign (of %d fraud, %d benign available)",
                f_take, b_take, len(fraud), len(benign),
            )
            packets = sampled
        else:
            packets = rng.sample(packets, sample)
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

    # Load packets (with optional balanced sampling)
    labels = None
    if args.balance_strategy == "balanced" and args.sample > 0:
        labels = _load_labels(args.labels_path, args.dataset)
    packets = load_packets(
        args.packets, args.sample,
        balance_strategy=args.balance_strategy, labels=labels,
    )

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
