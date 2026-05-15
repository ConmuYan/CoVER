import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.reasoner import EvidenceReasoner
from evidence.vocab import get_evidence_slots, get_reason_types, get_num_values


def test_reasoner_shapes():
    z_dim = 32
    num_nodes = 10
    reasoner = EvidenceReasoner(z_dim=z_dim, rho=0.3)

    z = torch.randn(num_nodes, z_dim)
    base_logit = torch.randn(num_nodes)
    evi_ids = torch.randint(0, get_num_values(), (num_nodes, len(get_evidence_slots())))

    outputs = reasoner(z, base_logit, evi_ids)

    assert outputs["final_logit"].shape == (num_nodes,)
    assert outputs["type_logits"].shape == (num_nodes, len(get_reason_types()))
    assert outputs["pos_logits"].shape == (num_nodes, len(get_evidence_slots()))
    assert outputs["neg_logits"].shape == (num_nodes, len(get_evidence_slots()))


def test_rho_zero():
    z_dim = 32
    num_nodes = 10
    reasoner = EvidenceReasoner(z_dim=z_dim, rho=0.0)
    reasoner.eval()

    z = torch.randn(num_nodes, z_dim)
    base_logit = torch.randn(num_nodes)
    evi_ids = torch.randint(0, get_num_values(), (num_nodes, len(get_evidence_slots())))

    outputs = reasoner(z, base_logit, evi_ids)

    assert torch.allclose(outputs["final_logit"], base_logit, atol=1e-6)


def test_extras_not_in_input():
    z_dim = 32
    reasoner = EvidenceReasoner(z_dim=z_dim)

    z = torch.randn(5, z_dim)
    base_logit = torch.randn(5)
    evi_ids = torch.randint(0, get_num_values(), (5, len(get_evidence_slots())))

    outputs = reasoner(z, base_logit, evi_ids)
    assert "final_logit" in outputs
    assert "type_logits" in outputs


def test_return_debug_outputs():
    z_dim = 32
    num_nodes = 7
    reasoner = EvidenceReasoner(z_dim=z_dim, rho=0.3)

    z = torch.randn(num_nodes, z_dim)
    base_logit = torch.randn(num_nodes)
    evi_ids = torch.randint(0, get_num_values(), (num_nodes, len(get_evidence_slots())))

    outputs = reasoner(z, base_logit, evi_ids, return_debug=True)

    assert outputs["gate"].shape == (num_nodes, 1)
    assert outputs["delta_raw"].shape == (num_nodes, 1)
    assert outputs["delta"].shape == (num_nodes, 1)
    assert outputs["rho"].shape == ()
    assert outputs["gate_mode"].shape == ()
    assert outputs["delta_scale"].shape == ()
    assert outputs["residual_shift"].shape == (num_nodes,)
    assert torch.isclose(outputs["rho"], torch.tensor(0.3, device=outputs["rho"].device))


def test_reasoner_accepts_relation_features():
    z_dim = 32
    relation_dim = 9
    num_nodes = 6
    reasoner = EvidenceReasoner(z_dim=z_dim, rho=0.2, relation_dim=relation_dim)

    z = torch.randn(num_nodes, z_dim)
    base_logit = torch.randn(num_nodes)
    evi_ids = torch.randint(0, get_num_values(), (num_nodes, len(get_evidence_slots())))
    rel = torch.randn(num_nodes, relation_dim)

    outputs = reasoner(z, base_logit, evi_ids, relation_features=rel)

    assert outputs["final_logit"].shape == (num_nodes,)
    assert outputs["z_student"].shape[0] == num_nodes
