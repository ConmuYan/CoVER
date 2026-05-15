from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse


FORBIDDEN_PACKET_KEYS = frozenset({
    "base_score",
    "base_probability",
    "base_prob",
    "base_logit",
    "base_logits",
    "base_confidence",
    "confidence",
    "base_prediction",
    "prediction",
    "target_label",
    "train_label",
    "val_label",
    "test_label",
    "ground_truth",
    "label",
    "split",
    "split_name",
    "split_identity",
    "prompt",
    "raw_prompt",
    "raw_llm_output",
    "raw_output",
    "llm_summary",
    "raw_summary",
    "summary",
})

RELATION_NAMES = {
    "net_rur": "same_user",
    "net_rsr": "same_product_same_rating",
    "net_rtr": "same_product_same_month",
    "net_upu": "same_product_reviewer",
    "net_usu": "same_star_one_week",
    "net_uvu": "review_text_similarity_top5",
}

GRAPH_REASONING_FIELDS = (
    "degree_level",
    "neighbor_consistency",
    "feature_neighbor_discrepancy",
    "detector_signal",
    "detector_signal_strength",
    "counter_signal",
    "degree_percentile_bucket",
    "neighbor_degree_skew_bucket",
    "two_hop_consistency_bucket",
    "feature_neighbor_cosine_bucket",
    "embedding_neighbor_cosine_bucket",
    "feature_embedding_disagreement_bucket",
    "bwgnn_low_band_energy_bucket",
    "bwgnn_mid_band_energy_bucket",
    "bwgnn_high_band_energy_bucket",
    "bwgnn_high_low_energy_ratio_bucket",
    "message_residual_bucket",
    "closer_to_fraud_prototype",
    "closer_to_benign_prototype",
    "prototype_conflict_level",
    "fraud_token_count",
    "benign_token_count",
    "neutral_token_count",
    "evidence_polarity",
)

JUDGE_RELATION_FIELDS = (
    "degree_bucket",
    "feature_deviation_bucket",
    "neighbor_consistency_bucket",
    "zscore_outlier_bucket",
    "fraud_prototype_distance_bucket",
    "benign_prototype_distance_bucket",
    "prototype_margin_bucket",
)


@dataclass
class PacketBuildContext:
    features: np.ndarray
    relation_matrices: dict[str, sparse.csr_matrix]
    feature_q33: np.ndarray
    feature_q66: np.ndarray
    relation_degree_q33: dict[str, float]
    relation_degree_q66: dict[str, float]
    decision: str = "use_feature_buckets_relation_summaries"


def make_packet_context(
    features: Any,
    relation_matrices: dict[str, Any] | None = None,
    decision: str = "use_feature_buckets_relation_summaries",
) -> PacketBuildContext:
    dense = features.toarray() if hasattr(features, "toarray") else np.asarray(features)
    dense = np.asarray(dense, dtype=np.float32)
    rels: dict[str, sparse.csr_matrix] = {}
    degree_q33: dict[str, float] = {}
    degree_q66: dict[str, float] = {}
    for key, matrix in (relation_matrices or {}).items():
        csr = matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
        rels[key] = csr
        degrees = np.asarray(csr.getnnz(axis=1), dtype=np.float32)
        degree_q33[key] = float(np.quantile(degrees, 0.33)) if degrees.size else 0.0
        degree_q66[key] = float(np.quantile(degrees, 0.66)) if degrees.size else 0.0
    return PacketBuildContext(
        features=dense,
        relation_matrices=rels,
        feature_q33=np.quantile(dense, 0.33, axis=0),
        feature_q66=np.quantile(dense, 0.66, axis=0),
        relation_degree_q33=degree_q33,
        relation_degree_q66=degree_q66,
        decision=decision,
    )


def build_evidence_packet(
    node_id: int,
    context: PacketBuildContext,
    evidence_card: dict[str, Any] | None = None,
    teacher_payload: dict[str, Any] | None = None,
    relation_tokens: list[str] | None = None,
    relation_top_dims: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    packet = {
        "node_id": int(node_id),
        "packet_version": "cover_meta_lite_v1",
        "score_blind": True,
        "data_source_decision": context.decision,
        "target_feature_summary": _target_feature_summary(node_id, context),
        "relation_context_summary": _relation_context_summary(node_id, context),
        "relation_evidence": _relation_evidence(relation_tokens, relation_top_dims),
        "graph_structural_summary": _graph_structural_summary(evidence_card, teacher_payload),
        "retrieved_context": _retrieved_context(teacher_payload),
        "contract_allowed_evidence": _contract_allowed_evidence(evidence_card, teacher_payload),
    }
    assert_packet_score_blind(packet)
    return packet


def serialize_evidence_packet(packet: dict[str, Any]) -> str:
    assert_packet_score_blind(packet)
    return json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_judge_packet(
    node_id: int,
    dataset: str,
    relation_schema: list[str],
    primary_relation: str,
    relation_fusion_source: str,
    relation_stats: np.ndarray,
    relation_stat_names: list[str],
    relation_tokens: list[str] | None = None,
    gate_values: dict[str, float] | None = None,
    graph_reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a score-blind packet for the inference-time LLM judge."""
    rel_schema = [str(name).upper() for name in relation_schema]
    stat_dim = len(relation_stat_names)
    if stat_dim <= 0 or relation_stats.shape[0] != stat_dim * len(rel_schema):
        raise ValueError("relation_stats length must match relation_schema * relation_stat_names")

    tokens_by_relation = _group_relation_tokens(relation_tokens or [], rel_schema)
    relation_evidence: dict[str, dict[str, Any]] = {}
    for rel_idx, rel_name in enumerate(rel_schema):
        chunk = relation_stats[rel_idx * stat_dim : (rel_idx + 1) * stat_dim]
        stats = {name: float(value) for name, value in zip(relation_stat_names, chunk)}
        relation_evidence[rel_name] = {
            "degree_bucket": _degree_bucket(stats),
            "feature_deviation_bucket": _z_bucket(stats.get("feature_l2_deviation_z", 0.0)),
            "neighbor_consistency_bucket": _bucket_cosine(stats.get("neighbor_cosine", 0.0)),
            "zscore_outlier_bucket": _fraction_bucket(stats.get("zscore_high_fraction", 0.0)),
            "fraud_prototype_distance_bucket": _z_bucket(stats.get("fraud_proto_dist_z", 0.0)),
            "benign_prototype_distance_bucket": _z_bucket(stats.get("benign_proto_dist_z", 0.0)),
            "prototype_margin_bucket": _margin_bucket(stats.get("fraud_minus_benign_margin_z", 0.0)),
            "relation_token_list": sorted(tokens_by_relation.get(rel_name, [])),
        }

    gate_values = {str(k).upper(): float(v) for k, v in (gate_values or {}).items()}
    gate_relations = list(rel_schema)
    packet = {
        "node_id": int(node_id),
        "dataset": str(dataset).lower(),
        "relation_schema": rel_schema,
        "primary_relation": str(primary_relation).upper(),
        "relation_fusion_source": str(relation_fusion_source),
        "relation_evidence": relation_evidence,
        "gate_evidence": {
            "anchor_relation": str(primary_relation).upper(),
            "optional_relation_gate_buckets": {
                name: _gate_bucket(gate_values.get(name, 0.0))
                for name in gate_relations
            },
            "optional_relation_open_flags": {
                name: bool(gate_values.get(name, 0.0) >= 0.5)
                for name in gate_relations
            },
        },
        "graph_diagnostic_evidence": _judge_graph_diagnostics(graph_reasoning or {}),
    }
    fields = judge_available_fields(packet)
    packet["allowed_support_fields"] = fields
    packet["allowed_counter_fields"] = fields
    assert_packet_score_blind(packet)
    return packet


def judge_available_fields(packet: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    relation_evidence = packet.get("relation_evidence", {})
    if isinstance(relation_evidence, dict):
        for relation, values in relation_evidence.items():
            if not isinstance(values, dict):
                continue
            for field in JUDGE_RELATION_FIELDS:
                if field in values:
                    fields.append(f"relation_evidence.{relation}.{field}")
            for token in values.get("relation_token_list", []) or []:
                fields.append(f"relation_evidence.{relation}.relation_token_list.{token}")

    gate_evidence = packet.get("gate_evidence", {})
    if isinstance(gate_evidence, dict):
        for group in ("optional_relation_gate_buckets", "optional_relation_open_flags"):
            values = gate_evidence.get(group, {})
            if isinstance(values, dict):
                for relation in values:
                    fields.append(f"gate_evidence.{group}.{relation}")

    graph = packet.get("graph_diagnostic_evidence", {})
    if isinstance(graph, dict):
        for key in graph:
            fields.append(f"graph_diagnostic_evidence.{key}")
    return sorted(dict.fromkeys(fields))


def assert_packet_score_blind(obj: Any, path: str = "root") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            if key_lower in FORBIDDEN_PACKET_KEYS:
                raise ValueError(f"Forbidden evidence packet field at {path}.{key}")
            _assert_forbidden_text(str(key), f"{path}.{key}")
            assert_packet_score_blind(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            assert_packet_score_blind(value, f"{path}[{idx}]")
    elif isinstance(obj, str):
        _assert_forbidden_text(obj, path)


def _assert_forbidden_text(text: str, path: str) -> None:
    lower = text.lower()
    for forbidden in FORBIDDEN_PACKET_KEYS:
        if lower == forbidden:
            raise ValueError(f"Forbidden evidence packet token at {path}: {text}")


def _target_feature_summary(node_id: int, context: PacketBuildContext) -> dict[str, Any]:
    values = context.features[node_id]
    buckets: dict[str, str] = {}
    counts = {"low": 0, "medium": 0, "high": 0}
    for idx, value in enumerate(values):
        bucket = _bucket_value(float(value), float(context.feature_q33[idx]), float(context.feature_q66[idx]))
        key = f"feature_{idx:02d}"
        buckets[key] = bucket
        counts[bucket] += 1

    high_features = [key for key, val in buckets.items() if val == "high"]
    low_features = [key for key, val in buckets.items() if val == "low"]
    return {
        "raw_review_text_available": False,
        "feature_source": f"anonymous_{context.features.shape[1]}d_features",
        "feature_buckets": buckets,
        "bucket_counts": counts,
        "high_bucket_features": high_features[:12],
        "low_bucket_features": low_features[:12],
    }


def _relation_context_summary(node_id: int, context: PacketBuildContext) -> dict[str, Any]:
    result: dict[str, Any] = {}
    target = context.features[node_id]
    for key, public_name in RELATION_NAMES.items():
        matrix = context.relation_matrices.get(key)
        if matrix is None:
            result[public_name] = {"available": False}
            continue
        start, end = matrix.indptr[node_id], matrix.indptr[node_id + 1]
        neighbors = matrix.indices[start:end]
        degree = int(neighbors.shape[0])
        degree_bucket = _bucket_value(
            float(degree),
            context.relation_degree_q33.get(key, 0.0),
            context.relation_degree_q66.get(key, 0.0),
        )
        if degree == 0:
            result[public_name] = {
                "available": True,
                "relation_key": key,
                "neighbor_count_bucket": degree_bucket,
                "neighbor_count": 0,
                "neighbor_feature_cosine_bucket": "unknown",
                "target_neighbor_feature_contrast": "unknown",
            }
            continue
        neigh_mean = context.features[neighbors].mean(axis=0)
        cosine = _cosine(target, neigh_mean)
        contrast = float(np.mean(np.abs(target - neigh_mean)))
        result[public_name] = {
            "available": True,
            "relation_key": key,
            "neighbor_count_bucket": degree_bucket,
            "neighbor_count": degree,
            "neighbor_feature_cosine_bucket": _bucket_cosine(cosine),
            "target_neighbor_feature_contrast": _bucket_threshold(contrast, 0.25, 0.75),
        }
    return result


def _graph_structural_summary(
    evidence_card: dict[str, Any] | None,
    teacher_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    reasoning: dict[str, Any] = {}
    for source in (teacher_payload, evidence_card):
        if isinstance(source, dict) and isinstance(source.get("reasoning"), dict):
            reasoning.update(source["reasoning"])
    safe_reasoning = {
        field: reasoning[field]
        for field in GRAPH_REASONING_FIELDS
        if field in reasoning
    }
    tokens = _active_tokens_from_payload(teacher_payload)
    bwgnn_tokens = [
        token for token in tokens
        if token.startswith(("HF_", "BAND_", "LOW_HIGH_", "NORMAL_", "CLEAN_", "RAW_TO_CLEAN"))
    ]
    return {
        "reasoning_buckets": safe_reasoning,
        "graph_evidence_tokens": tokens,
        "bwgnn_band_tokens": bwgnn_tokens,
        "neighbor_consistency": safe_reasoning.get("neighbor_consistency", "unknown"),
        "feature_structure_conflict": safe_reasoning.get("feature_embedding_disagreement_bucket", "unknown"),
        "evidence_polarity": safe_reasoning.get("evidence_polarity", "unknown"),
    }


def _relation_evidence(
    relation_tokens: list[str] | None,
    relation_top_dims: dict[str, list[int]] | None,
) -> dict[str, Any]:
    relation_keys = {
        str(key).upper()
        for key in (relation_top_dims or {})
        if str(key).upper() != "GLOBAL"
    }
    grouped: dict[str, list[str]] = {key: [] for key in sorted(relation_keys)}
    grouped["GLOBAL"] = []
    for token in relation_tokens or []:
        token = str(token)
        prefix = token.split("_", 1)[0]
        if prefix in grouped:
            grouped[prefix].append(token)
        elif prefix.isupper() and len(prefix) == 3:
            grouped[prefix] = [token]
        else:
            grouped["GLOBAL"].append(token)
    return {
        "tokens": {key: sorted(values) for key, values in grouped.items()},
        "top_anonymous_feature_deviation_dims": relation_top_dims or {},
    }


def _retrieved_context(teacher_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(teacher_payload, dict):
        return {
            "fraud_like_reference_cases": [],
            "benign_like_reference_cases": [],
            "hard_positive_patterns": [],
            "hard_negative_patterns": [],
        }
    return {
        "fraud_like_reference_cases": _case_token_summaries(teacher_payload.get("fraud_like_reference_cases", [])),
        "benign_like_reference_cases": _case_token_summaries(teacher_payload.get("benign_like_reference_cases", [])),
        "hard_positive_patterns": _distinctive_patterns(teacher_payload.get("fraud_prototype_summary", {})),
        "hard_negative_patterns": _distinctive_patterns(teacher_payload.get("benign_prototype_summary", {})),
    }


def _contract_allowed_evidence(
    evidence_card: dict[str, Any] | None,
    teacher_payload: dict[str, Any] | None,
) -> dict[str, list[str]]:
    reasoning: dict[str, Any] = {}
    for source in (teacher_payload, evidence_card):
        if isinstance(source, dict) and isinstance(source.get("reasoning"), dict):
            reasoning.update(source["reasoning"])
    return {
        "allowed_support_ids": _string_list(reasoning.get("allowed_support_ids", [])),
        "allowed_counter_ids": _string_list(reasoning.get("allowed_counter_ids", [])),
    }


def _active_tokens_from_payload(teacher_payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(teacher_payload, dict):
        return []
    token_set: set[str] = set()
    for case_key in ("fraud_like_reference_cases", "benign_like_reference_cases"):
        for case in teacher_payload.get(case_key, []) or []:
            tokens = case.get("tokens", {}) if isinstance(case, dict) else {}
            for token, value in tokens.items():
                if value == "active":
                    token_set.add(str(token))
    reasoning = teacher_payload.get("reasoning", {})
    if isinstance(reasoning, dict):
        for field in (
            "degree_level",
            "neighbor_consistency",
            "feature_neighbor_discrepancy",
            "detector_signal",
            "evidence_polarity",
        ):
            value = reasoning.get(field)
            if value not in (None, "unknown"):
                token_set.add(f"{field}:{value}")
    return sorted(token_set)


def _case_token_summaries(cases: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for case in cases if isinstance(cases, list) else []:
        if not isinstance(case, dict):
            continue
        tokens = case.get("tokens", {})
        if not isinstance(tokens, dict):
            continue
        active = sorted(str(k) for k, v in tokens.items() if v == "active")
        result.append({"active_tokens": active})
    return result[:5]


def _distinctive_patterns(summary: Any) -> list[dict[str, str]]:
    if not isinstance(summary, dict):
        return []
    result: list[dict[str, str]] = []
    for item in summary.get("distinctive_tokens", []) or []:
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        value = item.get("value")
        if field is None:
            continue
        result.append({"field": str(field), "value": str(value or "active")})
    return result[:8]


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v) for v in values]


def _bucket_value(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def _bucket_threshold(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value < high:
        return "medium"
    return "high"


def _bucket_cosine(value: float) -> str:
    if value < 0.3:
        return "low"
    if value < 0.7:
        return "medium"
    return "high"


def _degree_bucket(stats: dict[str, float]) -> str:
    if stats.get("degree_top10", 0.0) > 0.5:
        return "top10"
    if stats.get("degree_low", 0.0) > 0.5:
        return "low"
    value = stats.get("log_degree_norm", 0.0)
    if value < -0.5:
        return "low"
    if value > 0.5:
        return "high"
    return "medium"


def _z_bucket(value: float) -> str:
    if value <= -1.0:
        return "low"
    if value >= 1.0:
        return "high"
    return "medium"


def _fraction_bucket(value: float) -> str:
    if value <= 0.0:
        return "none"
    if value < 0.1:
        return "low"
    if value < 0.25:
        return "medium"
    return "high"


def _margin_bucket(value: float) -> str:
    if value >= 1.0:
        return "fraud_like_high"
    if value > 0.0:
        return "fraud_like"
    if value <= -1.0:
        return "benign_like_high"
    return "benign_like"


def _gate_bucket(value: float) -> str:
    if value >= 0.75:
        return "open_high"
    if value >= 0.5:
        return "open"
    if value >= 0.25:
        return "partly_open"
    return "closed"


def _group_relation_tokens(tokens: list[str], relations: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {relation: [] for relation in relations}
    for token in tokens:
        token = str(token)
        prefix = token.split("_", 1)[0].upper()
        if prefix in grouped:
            grouped[prefix].append(token)
        elif token == "RELATION_PROTO_CONFLICT_HIGH":
            for relation in relations:
                grouped[relation].append(token)
        elif token == "ANON_FEATURE_RELATION_ZSCORE_HIGH":
            for relation in relations:
                grouped[relation].append(token)
    return grouped


def _judge_graph_diagnostics(reasoning: dict[str, Any]) -> dict[str, str]:
    band_values = [
        reasoning.get("bwgnn_low_band_energy_bucket"),
        reasoning.get("bwgnn_mid_band_energy_bucket"),
        reasoning.get("bwgnn_high_band_energy_bucket"),
        reasoning.get("bwgnn_high_low_energy_ratio_bucket"),
    ]
    band_response = next((str(value) for value in band_values if value not in (None, "unknown")), "unknown")
    return {
        "band_response_bucket": band_response,
        "feature_structure_conflict_bucket": str(
            reasoning.get("feature_embedding_disagreement_bucket", "unknown")
        ),
        "embedding_neighbor_discrepancy_bucket": str(
            reasoning.get("embedding_neighbor_cosine_bucket", "unknown")
        ),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


__all__ = [
    "FORBIDDEN_PACKET_KEYS",
    "JUDGE_RELATION_FIELDS",
    "PacketBuildContext",
    "assert_packet_score_blind",
    "build_evidence_packet",
    "build_judge_packet",
    "judge_available_fields",
    "make_packet_context",
    "serialize_evidence_packet",
]
