from __future__ import annotations

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.relation_features import compute_relation_features_compact
from scripts.train_base_detector_minibatch import CpuNeighborSampler


def test_cpu_neighbor_sampler_keeps_roots_first_and_samples_in_neighbors():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    sampler = CpuNeighborSampler(
        edge_index=edge_index,
        num_nodes=5,
        fanouts=[2],
        seed=7,
        sampling_direction="in",
    )

    node_ids, sub_edge_index = sampler.sample(torch.tensor([3], dtype=torch.long))

    assert node_ids[0].item() == 3
    assert 2 in node_ids.tolist()
    local = {int(v): i for i, v in enumerate(node_ids.tolist())}
    assert [local[2], local[3]] in sub_edge_index.t().tolist()


def test_compact_relation_features_accept_edge_index_backend():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    train_mask = torch.tensor([True, True, False, False])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    schema = {
        "EDGE": {
            "mat_key": "edge_index",
            "description": "single relation",
        }
    }

    result = compute_relation_features_compact(
        features=x,
        relation_matrices={"edge_index": edge_index},
        y=y,
        train_mask=train_mask,
        enabled_relations=["EDGE"],
        relation_schema=schema,
    )

    assert result.rel_stats.shape == (4, 9)
    assert result.meta["compact"] is True
    assert result.meta["relation_meta"]["EDGE"]["edge_index_backend"] is True
