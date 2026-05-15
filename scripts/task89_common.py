from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.vocab import encode_reasoning, get_evidence_slots
from models.gnn import build_detector
from models.reasoner import EvidenceReasoner
from utils.paths import ensure_dir, get_base_checkpoint_path, get_err_cache_dir, get_reasoner_checkpoint_path


DEFAULT_SEEDS = [42, 123, 456, 789, 2026]
DIRECTIONS = ("increase_risk", "decrease_risk", "uncertain")
BASE_STATUS = ("FN", "FP", "TP", "TN")


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return float(-sum((v / total) * math.log2(v / total) for v in counter.values() if v > 0))


def load_data(config: dict[str, Any], seed: int):
    ds = config["dataset"]
    return load_fraud_dataset(
        ds["name"],
        path=ds.get("path"),
        seed=seed,
        split_mode=ds.get("split_mode", "supervised"),
        train_ratio=ds.get("train_ratio", 0.7),
        val_test_ratio=ds.get("val_test_ratio", [1, 2]),
        stratified=ds.get("stratified", False),
    )


def load_base_outputs(config: dict[str, Any], seed: int):
    dataset = config["dataset"]["name"]
    model_name = config["model"]["name"]
    requested_device = str(config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    data = load_data(config, seed)
    model_cfg = config["model"]
    model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.5),
    ).to(device)
    base_path = get_base_checkpoint_path(dataset, model_name, seed)
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    model.load_state_dict(torch.load(base_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        output = model(data.x.to(device), data.edge_index.to(device), return_output=True)
    return data, output.logits.detach().cpu(), output.embeddings.detach().cpu(), device


def split_name_for_node(data, node_id: int) -> str:
    if bool(data.train_mask[node_id]):
        return "train"
    if bool(data.val_mask[node_id]):
        return "val"
    if bool(data.test_mask[node_id]):
        return "test"
    return "unknown"


def base_status_for_node(y: torch.Tensor, base_prob: torch.Tensor, node_id: int) -> str:
    label = int(y[node_id].item())
    pred = int(base_prob[node_id].item() >= 0.5)
    if label == 1 and pred == 0:
        return "FN"
    if label == 0 and pred == 1:
        return "FP"
    if label == 1 and pred == 1:
        return "TP"
    return "TN"


def payload_polarity_by_node(err_dir: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for payload in load_jsonl(err_dir / "teacher_payloads.jsonl"):
        node_id = payload.get("node_id")
        polarity = payload.get("reasoning", {}).get("evidence_polarity")
        if isinstance(node_id, int) and isinstance(polarity, str):
            result[node_id] = polarity
    return result


def accepted_err_by_node(err_dir: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for err in load_jsonl(err_dir / "accepted_err.jsonl"):
        node_id = err.get("node_id")
        if isinstance(node_id, int):
            result[node_id] = err
    return result


def err_preference_key(err: dict[str, Any]) -> tuple[int, int]:
    direction = err.get("evidence_direction", "uncertain")
    strength = err.get("evidence_strength", "weak")
    direction_score = 0 if direction == "uncertain" else 1
    strength_score = {"weak": 0, "moderate": 1, "strong": 2}.get(strength, 0)
    return direction_score, strength_score


def load_stage2_sources_by_node(
    dataset: str,
    model: str,
    run_names: list[str],
    seed: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[int, str], dict[str, int]]:
    accepted_by_node: dict[int, dict[str, Any]] = {}
    cards_by_node: dict[int, dict[str, Any]] = {}
    polarity_by_node: dict[int, str] = {}
    stats: dict[str, int] = {}

    for run_name in run_names:
        err_dir = get_err_cache_dir(dataset, model, run_name, seed)
        source_cards = {
            card["node_id"]: card
            for card in load_jsonl(err_dir / "evidence_cards.jsonl")
            if isinstance(card.get("node_id"), int)
        }
        source_accepted = accepted_err_by_node(err_dir)
        source_polarity = payload_polarity_by_node(err_dir)
        stats[f"{run_name}_accepted_err"] = len(source_accepted)
        stats[f"{run_name}_evidence_cards"] = len(source_cards)

        for node_id, err in source_accepted.items():
            current = accepted_by_node.get(node_id)
            if current is None or err_preference_key(err) > err_preference_key(current):
                accepted_by_node[node_id] = err
                if node_id in source_cards:
                    cards_by_node[node_id] = source_cards[node_id]
                if node_id in source_polarity:
                    polarity_by_node[node_id] = source_polarity[node_id]

    stats["merged_accepted_err"] = len(accepted_by_node)
    stats["merged_evidence_cards"] = len(cards_by_node)
    return accepted_by_node, cards_by_node, polarity_by_node, stats


def load_stage3_reasoner(config: dict[str, Any], run_name: str, seed: int, z_dim: int, device: torch.device):
    dataset = config["dataset"]["name"]
    model_name = config["model"]["name"]
    stage3_log = Path("artifacts") / "logs" / dataset / model_name / run_name / f"seed_{seed}" / "stage3.json"
    run_info = load_json(stage3_log) if stage3_log.exists() else {}
    rc = run_info.get("config", config).get("reasoner", config.get("reasoner", {}))
    reasoner = EvidenceReasoner(
        z_dim=z_dim,
        hidden_dim=rc.get("hidden_dim", 128),
        rho=rc.get("rho", 0.3),
        gate_mode=rc.get("gate_mode", "safe_residual"),
        delta_scale=rc.get("delta_scale", 2.0),
        gate_bias_init=rc.get("gate_bias_init", -2.0),
        residual_init_zero=rc.get("residual_init_zero", True),
        latent_dim=rc.get("latent_dim", 256),
    ).to(device)
    reasoner_path = get_reasoner_checkpoint_path(dataset, model_name, run_name, seed)
    if not reasoner_path.exists():
        raise FileNotFoundError(reasoner_path)
    reasoner.load_state_dict(torch.load(reasoner_path, map_location=device, weights_only=True), strict=False)
    reasoner.eval()
    return reasoner, rc, run_info


def evidence_token_ids_from_cards(num_nodes: int, err_dir: Path) -> torch.Tensor:
    evidence_token_ids = torch.zeros(num_nodes, len(get_evidence_slots()), dtype=torch.long)
    for card in load_jsonl(err_dir / "evidence_cards.jsonl"):
        node_id = card.get("node_id")
        if isinstance(node_id, int) and node_id < num_nodes:
            evidence_token_ids[node_id] = encode_reasoning(card.get("reasoning", {}))
    return evidence_token_ids


def evidence_token_ids_from_card_map(num_nodes: int, cards_by_node: dict[int, dict[str, Any]]) -> torch.Tensor:
    evidence_token_ids = torch.zeros(num_nodes, len(get_evidence_slots()), dtype=torch.long)
    for node_id, card in cards_by_node.items():
        if node_id < num_nodes:
            evidence_token_ids[node_id] = encode_reasoning(card.get("reasoning", {}))
    return evidence_token_ids


def counter_for(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    c: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        if isinstance(value, str):
            c[value] += 1
    return c


def stat_dict(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
