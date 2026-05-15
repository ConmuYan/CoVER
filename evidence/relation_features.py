from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse

from utils.paths import ensure_dir


RELATION_SCHEMAS = {
    "yelpchi": {
        "RUR": {
            "mat_key": "net_rur",
            "description": "same user reviews",
        },
        "RSR": {
            "mat_key": "net_rsr",
            "description": "same product same star rating reviews",
        },
        "RTR": {
            "mat_key": "net_rtr",
            "description": "same product same month reviews",
        },
    },
    "amazon": {
        "UPU": {
            "mat_key": "net_upu",
            "description": "users reviewing at least one same product",
        },
        "USU": {
            "mat_key": "net_usu",
            "description": "users with at least one same star rating within one week",
        },
        "UVU": {
            "mat_key": "net_uvu",
            "description": "users with top-5% TF-IDF review text similarity",
        },
    },
}

# Backward-compatible alias used by older YelpChi tests/scripts.
RELATIONS = RELATION_SCHEMAS["yelpchi"]

RELATION_STAT_NAMES = [
    "log_degree_norm",
    "degree_top10",
    "degree_low",
    "feature_l2_deviation_z",
    "neighbor_cosine",
    "zscore_high_fraction",
    "fraud_proto_dist_z",
    "benign_proto_dist_z",
    "fraud_minus_benign_margin_z",
]


@dataclass
class RelationFeatureResult:
    rel_stats: torch.Tensor
    rel_tokens: dict[int, list[str]]
    top_deviation_dims: dict[int, dict[str, list[int]]]
    meta: dict[str, Any]


def compute_relation_features(
    features: Any,
    relation_matrices: dict[str, Any],
    y: torch.Tensor | np.ndarray,
    train_mask: torch.Tensor | np.ndarray,
    enabled_relations: list[str] | None = None,
    relation_schema: dict[str, dict[str, str]] | None = None,
    z_threshold: float = 2.0,
) -> RelationFeatureResult:
    """Build score-blind relation statistics with train-only label prototypes."""
    x = _to_numpy(features).astype(np.float32)
    labels = _to_numpy(y).reshape(-1).astype(np.int64)
    train = _to_numpy(train_mask).reshape(-1).astype(bool)
    schema = _normalize_relation_schema(relation_schema or RELATIONS)
    rel_names = _resolve_enabled_relations(enabled_relations, schema)
    if not rel_names:
        raise ValueError(
            "enabled_relations must contain at least one available relation: "
            f"{', '.join(schema)}"
        )

    all_stats: list[np.ndarray] = []
    tokens_by_node: dict[int, list[str]] = {i: [] for i in range(x.shape[0])}
    top_dims_by_node: dict[int, dict[str, list[int]]] = {i: {} for i in range(x.shape[0])}
    relation_meta: dict[str, Any] = {}

    for rel_name in rel_names:
        rel_spec = schema[rel_name]
        mat_key = rel_spec["mat_key"]
        if mat_key not in relation_matrices:
            raise KeyError(f"relation matrix missing: {mat_key}")
        matrix = _to_csr(relation_matrices[mat_key])
        stats, tokens, top_dims, meta = _compute_single_relation(
            rel_name=rel_name,
            rel_spec=rel_spec,
            x=x,
            matrix=matrix,
            labels=labels,
            train=train,
            z_threshold=z_threshold,
        )
        all_stats.append(stats)
        relation_meta[rel_name] = meta
        for node_id, node_tokens in tokens.items():
            tokens_by_node[node_id].extend(node_tokens)
        for node_id, dims in top_dims.items():
            top_dims_by_node[node_id][rel_name] = dims

    margin_columns = [idx * len(RELATION_STAT_NAMES) + 8 for idx in range(len(rel_names))]
    if margin_columns:
        margins = np.stack([all_stats[idx][:, 8] for idx in range(len(rel_names))], axis=1)
        conflict = np.min(np.abs(margins), axis=1) <= np.quantile(np.abs(margins), 0.10)
        for node_id in np.where(conflict)[0].tolist():
            tokens_by_node[int(node_id)].append("RELATION_PROTO_CONFLICT_HIGH")

    rel_stats_np = np.concatenate(all_stats, axis=1).astype(np.float32)
    rel_stats = torch.from_numpy(rel_stats_np)
    meta = {
        "relations": rel_names,
        "relation_descriptions": {name: schema[name]["description"] for name in rel_names},
        "stat_names_per_relation": RELATION_STAT_NAMES,
        "rel_dim": int(rel_stats.shape[1]),
        "num_nodes": int(rel_stats.shape[0]),
        "score_blind": True,
        "prototype_labels": "train_only",
        "target_label_used": False,
        "val_label_used": False,
        "test_label_used": False,
        "relation_meta": relation_meta,
    }
    return RelationFeatureResult(
        rel_stats=rel_stats,
        rel_tokens={node: sorted(set(tokens)) for node, tokens in tokens_by_node.items()},
        top_deviation_dims=top_dims_by_node,
        meta=meta,
    )


def save_relation_feature_artifacts(result: RelationFeatureResult, output_dir: Path) -> None:
    ensure_dir(output_dir)
    torch.save(
        {
            "rel_stats": result.rel_stats.float(),
            "rel_dim": int(result.rel_stats.shape[1]),
            "num_nodes": int(result.rel_stats.shape[0]),
            "meta": result.meta,
        },
        output_dir / "rel_stats.pt",
    )
    with open(output_dir / "rel_tokens.jsonl", "w") as f:
        for node_id in range(result.rel_stats.shape[0]):
            f.write(json.dumps({
                "node_id": int(node_id),
                "rel_tokens": result.rel_tokens.get(node_id, []),
                "top_anonymous_feature_deviation_dims": result.top_deviation_dims.get(node_id, {}),
            }, sort_keys=True) + "\n")
    (output_dir / "rel_feature_meta.json").write_text(json.dumps(result.meta, indent=2, sort_keys=True) + "\n")


def load_relation_stats(path: Path, num_nodes: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rel_stats = torch.as_tensor(payload["rel_stats"], dtype=torch.float32)
    if rel_stats.ndim != 2:
        raise ValueError(f"rel_stats must be rank-2, got {tuple(rel_stats.shape)}")
    if num_nodes is not None and rel_stats.shape[0] != num_nodes:
        raise ValueError(f"rel_stats node count mismatch: {rel_stats.shape[0]} != {num_nodes}")
    meta = dict(payload.get("meta", {}))
    meta.setdefault("rel_dim", int(rel_stats.shape[1]))
    return rel_stats, meta


def _compute_single_relation(
    rel_name: str,
    rel_spec: dict[str, str],
    x: np.ndarray,
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    train: np.ndarray,
    z_threshold: float,
) -> tuple[np.ndarray, dict[int, list[str]], dict[int, list[int]], dict[str, Any]]:
    n, num_features = x.shape
    degree = np.asarray(matrix.getnnz(axis=1), dtype=np.float32)
    degree_safe = np.maximum(degree, 1.0)
    has_neighbors = degree > 0

    neigh_mean = matrix @ x
    neigh_mean = neigh_mean / degree_safe[:, None]
    neigh_sq_mean = matrix @ (x * x)
    neigh_sq_mean = neigh_sq_mean / degree_safe[:, None]
    neigh_var = np.maximum(neigh_sq_mean - neigh_mean * neigh_mean, 0.0)
    neigh_std = np.sqrt(neigh_var + 1e-6)

    diff = x - neigh_mean
    l2_dev = np.linalg.norm(diff, axis=1)
    cosine = _row_cosine(x, neigh_mean)
    abs_z = np.abs(diff / neigh_std)
    abs_z[~has_neighbors] = 0.0
    z_count = (abs_z > float(z_threshold)).sum(axis=1).astype(np.float32)
    z_fraction = z_count / float(num_features)

    top_dims = {
        int(i): [int(d) for d in np.argsort(-abs_z[i])[:3].tolist()]
        for i in range(n)
    }

    fraud_proto, benign_proto, proto_meta = _relation_prototypes(
        x=x,
        neigh_mean=neigh_mean,
        labels=labels,
        train=train,
        has_neighbors=has_neighbors,
    )
    fraud_dist = np.linalg.norm(x - fraud_proto[None, :], axis=1)
    benign_dist = np.linalg.norm(x - benign_proto[None, :], axis=1)
    margin = fraud_dist - benign_dist

    degree_q10 = float(np.quantile(degree, 0.10))
    degree_q90 = float(np.quantile(degree, 0.90))
    dev_q90 = float(np.quantile(l2_dev[has_neighbors], 0.90)) if has_neighbors.any() else 0.0
    cos_q10 = float(np.quantile(cosine[has_neighbors], 0.10)) if has_neighbors.any() else 0.0
    fraud_q33 = float(np.quantile(fraud_dist, 0.33))
    benign_q33 = float(np.quantile(benign_dist, 0.33))

    log_degree = np.log1p(degree)
    log_degree_norm = log_degree / max(float(log_degree.max()), 1.0)
    stats = np.stack(
        [
            log_degree_norm,
            (degree >= degree_q90).astype(np.float32),
            (degree <= degree_q10).astype(np.float32),
            _zscore(l2_dev),
            cosine.astype(np.float32),
            z_fraction,
            _zscore(fraud_dist),
            _zscore(benign_dist),
            _zscore(margin),
        ],
        axis=1,
    ).astype(np.float32)

    tokens: dict[int, list[str]] = {}
    for node_id in range(n):
        node_tokens: list[str] = []
        if degree[node_id] >= degree_q90:
            node_tokens.append(f"{rel_name}_DEGREE_TOP10")
        if degree[node_id] <= degree_q10:
            node_tokens.append(f"{rel_name}_DEGREE_LOW")
        if has_neighbors[node_id] and l2_dev[node_id] >= dev_q90:
            node_tokens.append(f"{rel_name}_FEATURE_DEVIATION_HIGH")
        if has_neighbors[node_id] and cosine[node_id] <= cos_q10:
            node_tokens.append(f"{rel_name}_NEIGHBOR_CONSISTENCY_LOW")
        if z_count[node_id] > 0:
            node_tokens.append("ANON_FEATURE_RELATION_ZSCORE_HIGH")
            node_tokens.append(f"{rel_name}_ZSCORE_HIGH_COUNT")
        if fraud_dist[node_id] <= fraud_q33 and fraud_dist[node_id] < benign_dist[node_id]:
            node_tokens.append(f"{rel_name}_FRAUD_PROTO_CLOSE")
        if benign_dist[node_id] <= benign_q33 and benign_dist[node_id] < fraud_dist[node_id]:
            node_tokens.append(f"{rel_name}_BENIGN_PROTO_CLOSE")
        tokens[node_id] = node_tokens

    meta = {
        "mat_key": rel_spec["mat_key"],
        "description": rel_spec["description"],
        "degree_q10": degree_q10,
        "degree_q90": degree_q90,
        "feature_deviation_q90": dev_q90,
        "neighbor_cosine_q10": cos_q10,
        "fraud_proto_dist_q33": fraud_q33,
        "benign_proto_dist_q33": benign_q33,
        "z_threshold": float(z_threshold),
        "nodes_with_neighbors": int(has_neighbors.sum()),
        **proto_meta,
    }
    return stats, tokens, top_dims, meta


def _relation_prototypes(
    x: np.ndarray,
    neigh_mean: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    has_neighbors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_with_rel = train & has_neighbors
    fraud = train_with_rel & (labels == 1)
    benign = train_with_rel & (labels == 0)
    fallback_fraud = train & (labels == 1)
    fallback_benign = train & (labels == 0)

    if fraud.any():
        fraud_proto = neigh_mean[fraud].mean(axis=0)
        fraud_source = "train_relation_neighbor_mean"
    else:
        fraud_proto = x[fallback_fraud].mean(axis=0)
        fraud_source = "train_feature_mean_fallback"

    if benign.any():
        benign_proto = neigh_mean[benign].mean(axis=0)
        benign_source = "train_relation_neighbor_mean"
    else:
        benign_proto = x[fallback_benign].mean(axis=0)
        benign_source = "train_feature_mean_fallback"

    return fraud_proto.astype(np.float32), benign_proto.astype(np.float32), {
        "fraud_proto_train_nodes": int(fraud.sum()),
        "benign_proto_train_nodes": int(benign.sum()),
        "fraud_proto_source": fraud_source,
        "benign_proto_source": benign_source,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "toarray"):
        return value.toarray()
    return np.asarray(value)


def _to_csr(matrix: Any) -> sparse.csr_matrix:
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.maximum(denom, 1e-12)
    return (a * b).sum(axis=1) / denom


def _zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    std = float(values.std())
    if std <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


def get_relation_schema(dataset_name: str) -> dict[str, dict[str, str]]:
    key = dataset_name.lower()
    if key not in RELATION_SCHEMAS:
        raise ValueError(
            f"No relation schema registered for dataset '{dataset_name}'. "
            f"Available schemas: {', '.join(sorted(RELATION_SCHEMAS))}"
        )
    return _normalize_relation_schema(RELATION_SCHEMAS[key])


def _normalize_relation_schema(schema: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for name, spec in schema.items():
        rel_name = str(name).upper()
        if "mat_key" not in spec or "description" not in spec:
            raise ValueError(f"Relation schema for {rel_name} must include mat_key and description")
        normalized[rel_name] = {
            "mat_key": str(spec["mat_key"]),
            "description": str(spec["description"]),
        }
    return normalized


def _resolve_enabled_relations(
    enabled_relations: list[str] | None,
    schema: dict[str, dict[str, str]],
) -> list[str]:
    if enabled_relations is None:
        return list(schema)
    mapping = {name.lower(): name for name in schema}
    rel_names: list[str] = []
    unknown: list[str] = []
    for relation in enabled_relations:
        key = str(relation).lower()
        if key in mapping:
            rel_names.append(mapping[key])
        else:
            unknown.append(str(relation))
    if unknown:
        raise ValueError(
            f"Unknown relation(s): {', '.join(unknown)}. "
            f"Available relations: {', '.join(schema)}"
        )
    return rel_names


__all__ = [
    "RELATIONS",
    "RELATION_SCHEMAS",
    "RELATION_STAT_NAMES",
    "RelationFeatureResult",
    "compute_relation_features",
    "get_relation_schema",
    "load_relation_stats",
    "save_relation_feature_artifacts",
]
