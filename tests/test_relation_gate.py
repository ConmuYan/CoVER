from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.relation_features import compute_relation_features, get_relation_schema
from models.reasoner import EvidenceReasoner
from scripts.train_stage3 import write_relation_gate_diagnostics


def _inputs(num_nodes: int = 6):
    z = torch.randn(num_nodes, 8)
    base = torch.randn(num_nodes)
    evi = torch.zeros(num_nodes, 20, dtype=torch.long)
    rel = torch.randn(num_nodes, 27)
    return z, base, evi, rel


def _anchor_reasoner(dropout: float = 0.0) -> EvidenceReasoner:
    return EvidenceReasoner(
        z_dim=8,
        num_slots=20,
        hidden_dim=16,
        relation_dim=27,
        relation_hidden_dim=4,
        relation_fusion_mode="anchor_gate",
        relation_names=["RUR", "RSR", "RTR"],
        relation_stat_dim=9,
        anchor_relation="RUR",
        optional_relations=["RSR", "RTR"],
        relation_dropout=dropout,
    )


def _force_gate(reasoner: EvidenceReasoner, value: float) -> None:
    for head in reasoner.relation_gate_heads.values():
        final = head[-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.constant_(final.bias, value)


def test_anchor_gate_reduces_to_best_single_when_optional_gates_zero():
    reasoner = _anchor_reasoner()
    _force_gate(reasoner, -100.0)
    z, base, evi, rel = _inputs()
    out = reasoner(z, base, evi, relation_features=rel, return_debug=True)

    assert out["relation_gate_values"].shape == (6, 3)
    assert torch.allclose(out["relation_gate_values"][:, 0], torch.ones(6))
    assert torch.all(out["relation_gate_values"][:, 1:] < 1e-4)

    anchor_chunk = rel[:, :9]
    anchor_h = reasoner.relation_experts["RUR"](anchor_chunk)
    assert torch.allclose(out["relation_fused"], anchor_h, atol=1e-4)


def test_base_additive_gate_reduces_to_base_branch_when_all_gates_zero():
    reasoner = EvidenceReasoner(
        z_dim=8,
        num_slots=20,
        hidden_dim=16,
        relation_dim=27,
        relation_hidden_dim=4,
        relation_fusion_mode="base_additive_gate",
        relation_names=["UPU", "USU", "UVU"],
        relation_stat_dim=9,
    )
    _force_gate(reasoner, -100.0)
    z, base, evi, rel = _inputs()
    out = reasoner(z, base, evi, relation_features=rel, return_debug=True)

    assert torch.all(out["relation_gate_values"] < 1e-4)
    assert torch.allclose(out["relation_fused"], torch.zeros_like(out["relation_fused"]), atol=1e-4)


def test_sparse_gate_penalty_is_finite_and_no_softmax_simplex():
    reasoner = _anchor_reasoner()
    z, base, evi, rel = _inputs()
    out = reasoner(z, base, evi, relation_features=rel)

    assert torch.isfinite(out["relation_gate_sparse_loss"])
    gate_sum = out["relation_gate_values"].sum(dim=1)
    assert not torch.allclose(gate_sum, torch.ones_like(gate_sum))


def test_relation_dropout_never_drops_anchor_relation():
    reasoner = _anchor_reasoner(dropout=1.0)
    reasoner.train()
    z, base, evi, rel = _inputs()
    out = reasoner(z, base, evi, relation_features=rel)

    assert torch.allclose(out["relation_gate_values"][:, 0], torch.ones(6))
    assert torch.allclose(out["relation_gate_values"][:, 1:], torch.zeros(6, 2))


def test_relation_names_are_schema_driven_and_train_only():
    schema = get_relation_schema("amazon")
    assert list(schema) == ["UPU", "USU", "UVU"]
    x = np.eye(4, dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    train_mask = np.array([True, True, False, False])
    rel = sparse.csr_matrix(np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32))

    result = compute_relation_features(
        x,
        {"net_uvu": rel},
        y=y,
        train_mask=train_mask,
        enabled_relations=["UVU"],
        relation_schema=schema,
    )

    assert result.meta["relations"] == ["UVU"]
    assert result.meta["prototype_labels"] == "train_only"
    assert result.meta["test_label_used"] is False


def test_gate_stats_are_saved(tmp_path):
    reasoner = _anchor_reasoner()
    z, base, evi, rel = _inputs()
    out = reasoner(z, base, evi, relation_features=rel)
    write_relation_gate_diagnostics(
        log_dir=tmp_path,
        outputs=out,
        relation_names=["RUR", "RSR", "RTR"],
        final_logit=out["final_logit"].detach(),
        base_logits=base,
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        seed=123,
        max_allowed_shift=0.2,
    )

    assert (tmp_path / "relation_gate_stats.json").exists()
    assert (tmp_path / "relation_gate_by_label.csv").exists()
    payload = json.loads((tmp_path / "relation_gate_stats.json").read_text())
    assert payload["relations"] == ["RUR", "RSR", "RTR"]


def test_gate_config_keeps_stage3_llm_free():
    config = yaml.safe_load(Path("configs/cover-rel-gj/stage3_legacy/stage3_cover_rel_gate_nollm.yaml").read_text())
    reasoner = config["reasoner"]

    assert reasoner["use_teacher_latents"] is False
    assert reasoner["lambda_latent"] == 0.0
    assert reasoner["relation_fusion_mode"] in {"anchor_gate", "base_additive_gate"}
