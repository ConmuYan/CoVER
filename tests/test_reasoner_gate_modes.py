import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.reasoner import EvidenceReasoner, VALID_GATE_MODES
from evidence.vocab import get_evidence_slots, get_reason_types, get_num_values


def _make_inputs(z_dim=64, num_nodes=10):
    num_slots = len(get_evidence_slots())
    num_values = get_num_values()
    z = torch.randn(num_nodes, z_dim)
    base_logit = torch.randn(num_nodes)
    evidence_token_ids = torch.randint(0, num_values, (num_nodes, num_slots))
    return z, base_logit, evidence_token_ids


@pytest.mark.parametrize("gate_mode", VALID_GATE_MODES)
def test_forward_shape(gate_mode):
    z, base_logit, evi_ids = _make_inputs()
    reasoner = EvidenceReasoner(z_dim=64, gate_mode=gate_mode)
    outputs = reasoner(z, base_logit, evi_ids)

    assert outputs["final_logit"].shape == (10,)
    assert outputs["type_logits"].shape == (10, len(get_reason_types()))
    assert outputs["pos_logits"].shape == (10, len(get_evidence_slots()))
    assert outputs["neg_logits"].shape == (10, len(get_evidence_slots()))


@pytest.mark.parametrize("gate_mode", VALID_GATE_MODES)
def test_return_debug_shape(gate_mode):
    z, base_logit, evi_ids = _make_inputs()
    reasoner = EvidenceReasoner(z_dim=64, gate_mode=gate_mode)
    outputs = reasoner(z, base_logit, evi_ids, return_debug=True)

    assert "residual_shift" in outputs
    assert "gate_mode" in outputs
    assert "rho" in outputs
    assert "delta_scale" in outputs

    if gate_mode == "safe_residual":
        assert "gate" in outputs
        assert "delta" in outputs
        assert "delta_raw" in outputs
    elif gate_mode == "direct_tanh":
        assert "delta" in outputs
        assert "delta_raw" in outputs
    elif gate_mode == "signed_diff_legacy":
        assert "gate" in outputs
        assert "delta_raw" in outputs


def test_aux_only_final_equals_base():
    z, base_logit, evi_ids = _make_inputs()
    reasoner = EvidenceReasoner(z_dim=64, gate_mode="aux_only")
    outputs = reasoner(z, base_logit, evi_ids)

    assert torch.allclose(outputs["final_logit"], base_logit, atol=1e-6)


def test_rho_zero_final_equals_base():
    z, base_logit, evi_ids = _make_inputs()
    reasoner = EvidenceReasoner(z_dim=64, gate_mode="safe_residual", rho=0.0)
    outputs = reasoner(z, base_logit, evi_ids)

    assert torch.allclose(outputs["final_logit"], base_logit, atol=1e-6)


def test_safe_residual_initialization_close_to_base():
    z, base_logit, evi_ids = _make_inputs()
    reasoner = EvidenceReasoner(z_dim=64, gate_mode="safe_residual", rho=0.3)
    outputs = reasoner(z, base_logit, evi_ids)

    shift = (outputs["final_logit"] - base_logit).abs().max().item()
    assert shift < 1.0, f"Initial shift too large: {shift}"


def test_safe_residual_bounded_shift():
    z, base_logit, evi_ids = _make_inputs()
    rho = 0.3
    delta_scale = 2.0
    reasoner = EvidenceReasoner(z_dim=64, gate_mode="safe_residual", rho=rho, delta_scale=delta_scale)
    outputs = reasoner(z, base_logit, evi_ids, return_debug=True)

    max_possible_shift = rho * delta_scale
    actual_shift = outputs["residual_shift"].abs().max().item()
    assert actual_shift <= max_possible_shift + 0.01, f"Shift {actual_shift} exceeds bound {max_possible_shift}"


def test_invalid_gate_mode():
    with pytest.raises(ValueError, match="Unknown gate_mode"):
        EvidenceReasoner(z_dim=64, gate_mode="invalid_mode")
