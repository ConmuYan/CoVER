from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.evidence_packet import build_evidence_packet, make_packet_context, serialize_evidence_packet
from utils.paths import ensure_dir


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


def by_node(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        node_id = row.get("node_id")
        if isinstance(node_id, int):
            result[node_id] = row
    return result


def default_relation_tokens_path(dataset: str, model: str, seed: int) -> Path:
    return Path("artifacts") / "relation_features" / dataset / model / f"seed_{seed}" / "all" / "rel_tokens.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        for row in rows:
            f.write(serialize_evidence_packet(row) + "\n")


def default_err_dir(dataset: str, model: str, stage2_run_name: str, seed: int) -> Path:
    return Path("artifacts") / "err_cache" / dataset / model / stage2_run_name / f"seed_{seed}"


def default_output_path(dataset: str, model: str, stage2_run_name: str, seed: int) -> Path:
    return (
        Path("artifacts")
        / "evidence_packets"
        / dataset
        / model
        / stage2_run_name
        / f"seed_{seed}"
        / "evidence_packets.jsonl"
    )


def build_packets(
    dataset_path: Path,
    accepted_err_path: Path,
    evidence_cards_path: Path,
    teacher_payloads_path: Path | None,
    relation_tokens_path: Path | None = None,
    max_packets: int | None = None,
) -> list[dict[str, Any]]:
    from scipy.io import loadmat

    mat = loadmat(dataset_path)
    features = mat["features"]
    relation_matrices = {
        key: mat[key]
        for key in ("net_rur", "net_rsr", "net_rtr")
        if key in mat
    }
    context = make_packet_context(features, relation_matrices)

    accepted = load_jsonl(accepted_err_path)
    cards = by_node(load_jsonl(evidence_cards_path))
    payloads = by_node(load_jsonl(teacher_payloads_path)) if teacher_payloads_path is not None else {}
    relation_rows = by_node(load_jsonl(relation_tokens_path)) if relation_tokens_path is not None else {}
    node_ids = [int(row["node_id"]) for row in accepted if isinstance(row.get("node_id"), int)]
    node_ids = sorted(dict.fromkeys(node_ids))
    if max_packets is not None:
        node_ids = node_ids[:max_packets]

    packets: list[dict[str, Any]] = []
    for node_id in node_ids:
        packets.append(
            build_evidence_packet(
                node_id=node_id,
                context=context,
                evidence_card=cards.get(node_id),
                teacher_payload=payloads.get(node_id),
                relation_tokens=relation_rows.get(node_id, {}).get("rel_tokens", []),
                relation_top_dims=relation_rows.get(node_id, {}).get("top_anonymous_feature_deviation_dims", {}),
            )
        )
    return packets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yelpchi")
    parser.add_argument("--model", type=str, default="bwgnn")
    parser.add_argument("--stage2_run_name", type=str, default="qwen_directional_t200")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dataset_path", type=str, default="datasets/YelpChi.mat")
    parser.add_argument("--accepted_err_path", type=str, default=None)
    parser.add_argument("--evidence_cards_path", type=str, default=None)
    parser.add_argument("--teacher_payloads_path", type=str, default=None)
    parser.add_argument("--relation_tokens_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_packets", type=int, default=None)
    args = parser.parse_args()

    err_dir = default_err_dir(args.dataset, args.model, args.stage2_run_name, args.seed)
    accepted_path = Path(args.accepted_err_path) if args.accepted_err_path else err_dir / "accepted_err.jsonl"
    cards_path = Path(args.evidence_cards_path) if args.evidence_cards_path else err_dir / "evidence_cards.jsonl"
    payloads_path = Path(args.teacher_payloads_path) if args.teacher_payloads_path else err_dir / "teacher_payloads.jsonl"
    if not payloads_path.exists():
        payloads_path = None
    rel_tokens_path = (
        Path(args.relation_tokens_path)
        if args.relation_tokens_path
        else default_relation_tokens_path(args.dataset, args.model, args.seed)
    )
    if not rel_tokens_path.exists():
        rel_tokens_path = None
    output_path = Path(args.output_path) if args.output_path else default_output_path(
        args.dataset, args.model, args.stage2_run_name, args.seed
    )

    for path in (Path(args.dataset_path), accepted_path, cards_path):
        if not path.exists():
            raise FileNotFoundError(f"required input not found: {path}")

    packets = build_packets(
        dataset_path=Path(args.dataset_path),
        accepted_err_path=accepted_path,
        evidence_cards_path=cards_path,
        teacher_payloads_path=payloads_path,
        relation_tokens_path=rel_tokens_path,
        max_packets=args.max_packets,
    )
    write_jsonl(output_path, packets)
    meta = {
        "dataset": args.dataset,
        "model": args.model,
        "stage2_run_name": args.stage2_run_name,
        "seed": args.seed,
        "score_blind": True,
        "latent_source_ready": "evidence_packet",
        "num_packets": len(packets),
        "accepted_err_path": str(accepted_path),
        "evidence_cards_path": str(cards_path),
        "teacher_payloads_path": str(payloads_path) if payloads_path is not None else None,
        "relation_tokens_path": str(rel_tokens_path) if rel_tokens_path is not None else None,
        "output_path": str(output_path),
    }
    (output_path.parent / "evidence_packet_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
