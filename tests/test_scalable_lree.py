from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.scalable_lree import build_scalable_lree_extractor
from models.raer_teacher import RAERTeacher
from scripts.train_raer_teacher import materialize_scalable_lree_features
from scripts.train_cbr_flash import generate_teacher_cache


def test_scalable_lree_forward_shape_and_score_blind_interface():
    extractor = build_scalable_lree_extractor(
        num_relations=3,
        in_dim_per_rel=9,
        cfg={"hidden_dim": 8, "out_dim_per_rel": 9, "dropout": 0.0},
    )
    basis = torch.randn(11, 27)

    out = extractor(basis)

    assert out.shape == (11, 27)


def test_scalable_lree_materialized_features_feed_raer_teacher():
    extractor = build_scalable_lree_extractor(
        num_relations=2,
        in_dim_per_rel=9,
        cfg={"hidden_dim": 8, "out_dim_per_rel": 9, "dropout": 0.0},
    )
    basis = torch.randn(13, 18)
    rel_features = materialize_scalable_lree_features(
        extractor,
        basis,
        torch.device("cpu"),
        batch_size=5,
        output_device="cpu",
    )
    teacher = RAERTeacher(
        base_z_dim=6,
        relation_names=["R1", "R2"],
        anchor_relation="R1",
        rel_stat_dim=9,
        rel_hidden_dim=8,
        rel_dropout=0.0,
    )
    out = teacher(
        torch.randn(13, 6),
        torch.randn(13),
        rel_features,
        return_heads=True,
    )

    assert out["final_logit"].shape == (13,)
    assert out["c_per_r"].shape == (13, 2)
    assert out["gate"].shape == (13, 2)


def test_generate_teacher_cache_is_chunked_and_complete():
    teacher = RAERTeacher(
        base_z_dim=5,
        relation_names=["R1", "R2"],
        anchor_relation="R1",
        rel_stat_dim=9,
        rel_hidden_dim=8,
        rel_dropout=0.0,
    )
    cache = generate_teacher_cache(
        teacher,
        torch.randn(17, 5),
        torch.randn(17),
        torch.randn(17, 18),
        torch.device("cpu"),
        batch_size=4,
    )

    assert cache["p"].shape == (17,)
    assert cache["logit"].shape == (17,)
    assert cache["delta_rel"].shape == (17,)
