import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.reasoner import EvidenceReasoner


def _make_inputs(z_dim=64, num_nodes=10):
    from evidence.vocab import get_evidence_slots, get_num_values
    num_slots = len(get_evidence_slots())
    num_values = get_num_values()
    z = torch.randn(num_nodes, z_dim)
    base_logit = torch.randn(num_nodes)
    evidence_token_ids = torch.randint(0, num_values, (num_nodes, num_slots))
    return z, base_logit, evidence_token_ids


def test_safe_residual_extreme_residual_bounded():
    z, base_logit, evi_ids = _make_inputs()
    rho = 0.3
    delta_scale = 2.0
    reasoner = EvidenceReasoner(z_dim=64, gate_mode="safe_residual", rho=rho, delta_scale=delta_scale)

    with torch.no_grad():
        reasoner.residual_head.weight.fill_(100.0)
        reasoner.residual_head.bias.fill_(100.0)

    outputs = reasoner(z, base_logit, evi_ids, return_debug=True)

    max_possible = rho * delta_scale
    actual = outputs["residual_shift"].abs().max().item()
    assert actual <= max_possible + 0.01


def test_safe_residual_gate_non_negative():
    z, base_logit, evi_ids = _make_inputs()
    reasoner = EvidenceReasoner(z_dim=64, gate_mode="safe_residual")
    outputs = reasoner(z, base_logit, evi_ids, return_debug=True)

    assert (outputs["gate"] >= 0).all()
    assert (outputs["gate"] <= 1).all()


def test_legacy_gate_can_be_negative():
    z, base_logit, evi_ids = _make_inputs()
    reasoner = EvidenceReasoner(z_dim=64, gate_mode="signed_diff_legacy")
    outputs = reasoner(z, base_logit, evi_ids, return_debug=True)

    assert outputs["gate"].min().item() < 0


def test_aux_only_never_changes_base():
    z, base_logit, evi_ids = _make_inputs()
    reasoner = EvidenceReasoner(z_dim=64, gate_mode="aux_only")

    with torch.no_grad():
        reasoner.residual_head.weight.fill_(100.0) if hasattr(reasoner, 'residual_head') else None

    outputs = reasoner(z, base_logit, evi_ids)
    assert torch.allclose(outputs["final_logit"], base_logit, atol=1e-6)


def test_direct_tanh_bounded():
    z, base_logit, evi_ids = _make_inputs()
    rho = 0.3
    delta_scale = 2.0
    reasoner = EvidenceReasoner(z_dim=64, gate_mode="direct_tanh", rho=rho, delta_scale=delta_scale)

    with torch.no_grad():
        reasoner.residual_head.weight.fill_(100.0)
        reasoner.residual_head.bias.fill_(100.0)

    outputs = reasoner(z, base_logit, evi_ids, return_debug=True)

    max_possible = rho * delta_scale
    actual = outputs["residual_shift"].abs().max().item()
    assert actual <= max_possible + 0.01
