from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.relation_features import (
    compute_relation_features,
    get_relation_schema,
    load_relation_stats,
    save_relation_feature_artifacts,
)


def test_relation_features_build_dense_stats_and_score_blind_tokens(tmp_path):
    x = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [2.0, 2.0, 0.0],
            [2.1, 2.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    y = torch.tensor([0, 0, 1, 1, 0])
    train_mask = torch.tensor([True, True, True, True, False])
    rel = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )

    result = compute_relation_features(
        x,
        {"net_rur": rel, "net_rsr": rel, "net_rtr": rel},
        y=y,
        train_mask=train_mask,
    )

    assert result.rel_stats.shape == (5, 27)
    assert result.meta["prototype_labels"] == "train_only"
    assert result.meta["target_label_used"] is False
    assert "RUR" in result.top_deviation_dims[4]
    assert any(token.startswith("RUR_") for token in result.rel_tokens[4])

    save_relation_feature_artifacts(result, tmp_path)
    loaded, meta = load_relation_stats(tmp_path / "rel_stats.pt", num_nodes=5)
    assert tuple(loaded.shape) == (5, 27)
    assert meta["score_blind"] is True
    text = (tmp_path / "rel_tokens.jsonl").read_text()
    assert "target_label" not in text
    assert "base_score" not in text


def test_relation_feature_subset_changes_dimension():
    x = np.eye(4, dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    train_mask = np.array([True, True, True, True])
    rel = sparse.csr_matrix(np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32))

    result = compute_relation_features(
        x,
        {"net_rur": rel},
        y=y,
        train_mask=train_mask,
        enabled_relations=["RUR"],
    )

    assert result.rel_stats.shape == (4, 9)
    assert result.meta["relations"] == ["RUR"]


def test_relation_features_support_amazon_schema():
    x = np.eye(5, dtype=np.float32)
    y = np.array([0, 1, 0, 1, 0])
    train_mask = np.array([True, True, True, True, False])
    rel = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 1, 0],
            ],
            dtype=np.float32,
        )
    )

    schema = get_relation_schema("amazon")
    result = compute_relation_features(
        x,
        {"net_upu": rel},
        y=y,
        train_mask=train_mask,
        enabled_relations=["UPU"],
        relation_schema=schema,
    )

    assert result.rel_stats.shape == (5, 9)
    assert result.meta["relations"] == ["UPU"]
    assert result.meta["relation_meta"]["UPU"]["mat_key"] == "net_upu"
    assert any(token.startswith("UPU_") for token in result.rel_tokens[4])
    assert result.meta["test_label_used"] is False


def test_yelp_style_new_dataset_schemas_match_yelpchi():
    for dataset in ("yelpnyc", "yelpzip"):
        schema = get_relation_schema(dataset)
        assert list(schema) == ["RUR", "RSR", "RTR"]
        assert schema["RUR"]["mat_key"] == "net_rur"
        assert schema["RSR"]["mat_key"] == "net_rsr"
        assert schema["RTR"]["mat_key"] == "net_rtr"


def test_single_relation_dgl_dataset_schemas_use_edge_index():
    for dataset in ("tfinance", "tsocial"):
        schema = get_relation_schema(dataset)
        assert list(schema) == ["EDGE"]
        assert schema["EDGE"]["mat_key"] == "edge_index"
