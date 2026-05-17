"""Generate synthetic per-token quality labels for LEQA LoRA warm-up.

Pipeline:
  1. Load existing judge_packets.jsonl (score-blind evidence packets)
  2. For each packet, randomly perturb some evidence tokens:
     - drop: remove a token entirely -> q=0, reason="insufficient"
     - swap: replace value with random other value -> q=0, reason="noisy"
     - noise: inject contradictory information -> q=0, reason="contradictory"
  3. Label clean tokens with q=1, reason="none"
  4. Serialize as JSONL with training_text field
  5. Assert no forbidden field at write time

Never writes any forbidden field (base_score, label, split, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from copy import deepcopy
from pathlib import Path

logger = logging.getLogger(__name__)

# Forbidden fields that must never appear in output
FORBIDDEN_FIELDS = frozenset({
    "base_score", "base_prob", "base_logit", "confidence",
    "label", "split", "FN", "FP", "ground_truth", "final_prediction",
    "base_probability", "base_prediction", "target_label",
    "train_label", "val_label", "test_label", "split_name",
    "split_identity",
})

# Perturbation types
PERTURBATION_TYPES = ["drop", "swap", "noise"]
PERTURBATION_TO_REASON = {
    "drop": "insufficient",
    "swap": "noisy",
    "noise": "contradictory",
}

# Possible replacement values for swap perturbation
SWAP_VALUES = ["low", "medium", "high", "unknown", "none", "weak", "strong"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic token quality labels for LEQA LoRA warm-up"
    )
    p.add_argument("--dataset", type=str, default="yelpchi",
                   choices=["yelpchi", "amazon"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=2000,
                   help="Number of training examples to generate")
    p.add_argument("--out", type=str, required=True,
                   help="Output JSONL path")
    p.add_argument("--packets_path", type=str, default=None,
                   help="Path to source packets JSONL (auto-detected if None)")
    p.add_argument("--perturb_rate", type=float, default=0.3,
                   help="Fraction of tokens to perturb per packet")
    p.add_argument("--relation_names", type=str, nargs="+", default=None)
    return p.parse_args()


def _assert_no_forbidden(obj, path="root"):
    """Recursively assert no forbidden field in a dict/list."""
    if isinstance(obj, dict):
        for key in obj:
            if key.lower() in FORBIDDEN_FIELDS:
                raise ValueError(f"Forbidden field '{key}' at {path}")
            _assert_no_forbidden(obj[key], f"{path}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden(item, f"{path}[{i}]")
    elif isinstance(obj, str):
        lower = obj.lower()
        for ff in FORBIDDEN_FIELDS:
            if ff in lower:
                raise ValueError(f"Forbidden text '{ff}' at {path}")


def get_perturbable_fields(packet: dict) -> list[tuple[str, str, str]]:
    """Extract perturbable (path, key, value) tuples from a packet.

    Returns list of (dotted_path, field_name, current_value) for fields
    that can be meaningfully perturbed.
    """
    fields = []

    # Relation evidence fields
    rel_ev = packet.get("relation_evidence", {})
    for rel_name, rel_data in rel_ev.items():
        if not isinstance(rel_data, dict):
            continue
        for key, val in rel_data.items():
            if key == "relation_token_list":
                continue  # handled separately
            if isinstance(val, str):
                fields.append((f"relation_evidence.{rel_name}", key, val))

    # Graph diagnostic fields
    gde = packet.get("graph_diagnostic_evidence", {})
    if isinstance(gde, dict):
        for key, val in gde.items():
            if isinstance(val, str):
                fields.append(("graph_diagnostic_evidence", key, val))

    # Gate evidence optional fields
    ge = packet.get("gate_evidence", {})
    if isinstance(ge, dict):
        for sub_key, sub_val in ge.items():
            if isinstance(sub_val, dict):
                for key, val in sub_val.items():
                    if isinstance(val, str):
                        fields.append(
                            (f"gate_evidence.{sub_key}", key, val)
                        )

    return fields


def perturb_packet(
    packet: dict,
    perturb_rate: float,
    rng: random.Random,
) -> tuple[dict, list[dict]]:
    """Perturb a packet and return (perturbed_packet, labels).

    Labels is a list of {"token": name, "q": float, "reason": str}.
    """
    perturbed = deepcopy(packet)
    fields = get_perturbable_fields(perturbed)

    if not fields:
        # No perturbable fields -> all clean
        return perturbed, []

    n_perturb = max(1, int(len(fields) * perturb_rate))
    perturb_indices = set(rng.sample(range(len(fields)), min(n_perturb, len(fields))))

    labels = []
    for i, (path, key, original_val) in enumerate(fields):
        token_name = f"{path}.{key}"
        if i in perturb_indices:
            ptype = rng.choice(PERTURBATION_TYPES)
            reason = PERTURBATION_TO_REASON[ptype]

            # Apply perturbation to the packet
            parts = path.split(".")
            obj = perturbed
            for part in parts:
                if isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                else:
                    break

            if isinstance(obj, dict) and key in obj:
                if ptype == "drop":
                    obj[key] = "unknown"
                elif ptype == "swap":
                    candidates = [v for v in SWAP_VALUES if v != original_val]
                    obj[key] = rng.choice(candidates) if candidates else "unknown"
                elif ptype == "noise":
                    # Inject contradictory: flip high<->low
                    flip_map = {"high": "low", "low": "high",
                                "strong": "weak", "weak": "strong"}
                    obj[key] = flip_map.get(original_val, "unknown")

            labels.append({"token": token_name, "q": 0.0, "reason": reason})
        else:
            labels.append({"token": token_name, "q": 1.0, "reason": "none"})

    return perturbed, labels


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = random.Random(args.seed)

    # Auto-detect packets path
    if args.packets_path:
        packets_path = args.packets_path
    else:
        base = Path("/data1/mq/codes/awesome-graph-anomaly-detection/cover-fd")
        packets_path = str(
            base / "artifacts" / "judge_packets" / args.dataset / "bwgnn"
            / "cover_rel_judge_revived" / "seed_42" / "judge_packets.jsonl"
        )

    # Load source packets
    source_packets = []
    with open(packets_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pkt = json.loads(line)
            # Remove calibration (contains base_score)
            pkt.pop("calibration", None)
            source_packets.append(pkt)
    logger.info("Loaded %d source packets from %s", len(source_packets), packets_path)

    if not source_packets:
        logger.error("No source packets found")
        sys.exit(1)

    # Determine relation names from data
    if args.relation_names:
        relation_names = args.relation_names
    else:
        sample_pkt = source_packets[0]
        relation_names = sample_pkt.get("relation_schema", ["RUR", "RSR", "RTR"])

    # Import LEQA formatting
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.leqa_lora import build_leqa_token_names, build_training_example

    token_names = build_leqa_token_names(relation_names)

    # Generate training examples
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    for i in range(args.n):
        pkt = rng.choice(source_packets)
        perturbed_pkt, perturb_labels = perturb_packet(
            pkt, args.perturb_rate, rng,
        )

        # Map perturbation labels to the canonical token list
        perturb_map = {lab["token"]: lab for lab in perturb_labels}
        q_labels = []
        reason_labels = []
        for tname in token_names:
            if tname in perturb_map:
                q_labels.append(perturb_map[tname]["q"])
                reason_labels.append(perturb_map[tname]["reason"])
            else:
                q_labels.append(1.0)
                reason_labels.append("none")

        # Build training text
        pkt_json = json.dumps(perturbed_pkt, indent=None)
        training_text = build_training_example(
            pkt_json, token_names, q_labels, reason_labels,
        )

        record = {
            "packet": pkt,
            "perturbed_packet": perturbed_pkt,
            "labels": [
                {"token": t, "q": q, "reason": r}
                for t, q, r in zip(token_names, q_labels, reason_labels)
            ],
            "training_text": training_text,
        }

        # Assert no forbidden field at write time
        _assert_no_forbidden(record["perturbed_packet"])
        _assert_no_forbidden(record["packet"])

        examples.append(record)

    # Write output
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    logger.info("Wrote %d training examples to %s", len(examples), out_path)

    # Summary stats
    n_perturbed = sum(
        1 for ex in examples
        for lab in ex["labels"]
        if lab["q"] == 0.0
    )
    n_total = sum(len(ex["labels"]) for ex in examples)
    logger.info(
        "Perturbation stats: %d / %d tokens perturbed (%.1f%%)",
        n_perturbed, n_total, 100.0 * n_perturbed / max(n_total, 1),
    )
    reason_counts = {}
    for ex in examples:
        for lab in ex["labels"]:
            r = lab["reason"]
            reason_counts[r] = reason_counts.get(r, 0) + 1
    logger.info("Reason distribution: %s", reason_counts)


if __name__ == "__main__":
    main()
