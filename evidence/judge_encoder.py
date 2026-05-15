from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from utils.paths import ensure_dir


VERDICT_ORDER = ("fake", "real", "uncertain")
STRENGTH_ORDER = ("weak", "moderate", "strong")


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


def encode_judge_records(
    records: list[dict[str, Any]],
    num_nodes: int,
    relation_names: list[str],
    evidence_fields: list[str],
    primary_relation: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    relation_names = [name.upper() for name in relation_names]
    evidence_fields = sorted(dict.fromkeys(evidence_fields))
    field_index = {field: idx for idx, field in enumerate(evidence_fields)}
    rel_index = {name: idx for idx, name in enumerate(relation_names)}

    feature_dim = 10 + len(relation_names) + 3 * len(evidence_fields)
    offset = 10 + len(relation_names)
    features = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    verdict_counter: Counter[str] = Counter()
    strength_counter: Counter[str] = Counter()

    for record in records:
        node_id = int(record["node_id"])
        if node_id < 0 or node_id >= num_nodes:
            continue
        row = features[node_id]
        verdict = str(record.get("verdict", "uncertain"))
        strength = str(record.get("evidence_strength", "weak"))
        key_relation = str(record.get("key_relation", "")).upper()

        row[_index(VERDICT_ORDER, verdict)] = 1.0
        row[3 + _index(STRENGTH_ORDER, strength)] = 1.0
        row[6] = 1.0 if key_relation == primary_relation.upper() else 0.0
        support = _string_list(record.get("supporting_evidence"))
        counter = _string_list(record.get("counter_evidence"))
        uncertainty = _string_list(record.get("uncertainty_factors"))
        row[7] = min(len(support), 10) / 10.0
        row[8] = min(len(counter), 10) / 10.0
        row[9] = 1.0 if verdict == "uncertain" else 0.0

        if key_relation in rel_index:
            row[10 + rel_index[key_relation]] = 1.0

        _set_mask(row, offset, support, field_index)
        _set_mask(row, offset + len(evidence_fields), counter, field_index)
        _set_mask(row, offset + 2 * len(evidence_fields), uncertainty, field_index)
        mask[node_id] = True
        verdict_counter[verdict] += 1
        strength_counter[strength] += 1

    meta = {
        "accepted_only": True,
        "summary_used": False,
        "short_explanation_used_for_loss": False,
        "base_score_used": False,
        "target_label_used": False,
        "score_blind": True,
        "verifier_active": True,
        "num_accepted": int(mask.sum().item()),
        "relation_names": relation_names,
        "primary_relation": primary_relation.upper(),
        "evidence_fields": evidence_fields,
        "feature_dim": feature_dim,
        "verdict_distribution": dict(verdict_counter),
        "strength_distribution": dict(strength_counter),
        "layout": {
            "verdict_one_hot": [0, 3],
            "strength_one_hot": [3, 6],
            "key_relation_is_primary": 6,
            "support_count": 7,
            "counter_count": 8,
            "uncertain_flag": 9,
            "key_relation_one_hot": [10, 10 + len(relation_names)],
            "support_mask": [offset, offset + len(evidence_fields)],
            "counter_mask": [offset + len(evidence_fields), offset + 2 * len(evidence_fields)],
            "uncertainty_mask": [offset + 2 * len(evidence_fields), offset + 3 * len(evidence_fields)],
        },
    }
    return features, mask, meta


def save_judge_features(
    output_dir: Path,
    features: torch.Tensor,
    mask: torch.Tensor,
    meta: dict[str, Any],
) -> None:
    ensure_dir(output_dir)
    torch.save({"features": features, "mask": mask}, output_dir / "judge_features.pt")
    (output_dir / "judge_feature_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def _index(values: tuple[str, ...], value: str) -> int:
    try:
        return values.index(value)
    except ValueError:
        return values.index("uncertain") if "uncertain" in values else 0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _set_mask(row: torch.Tensor, offset: int, values: list[str], field_index: dict[str, int]) -> None:
    for value in values:
        idx = field_index.get(value)
        if idx is not None:
            row[offset + idx] = 1.0


__all__ = ["encode_judge_records", "load_jsonl", "save_judge_features"]
