"""Unit tests for compute_audit_loss.

Verifies:
  (a) loss shape is scalar
  (b) zero loss when q=1 everywhere
  (c) zero loss when delta_rel=0 everywhere
  (d) larger loss when attention concentrates on q=0 tokens
  (e) gradient flows to all 3 inputs
  (f) clip bound is respected
"""

from __future__ import annotations

import pytest
import torch

from training.losses_audit import compute_audit_loss


@pytest.fixture
def standard_inputs():
    """Standard test inputs: N=8 nodes, T=37 tokens."""
    N, T = 8, 37
    torch.manual_seed(42)
    attention = torch.randn(N, T, requires_grad=True)
    lora_q = torch.rand(N, T, requires_grad=True)
    delta_rel = torch.randn(N, requires_grad=True)
    return attention, lora_q, delta_rel


class TestAuditLossShape:
    def test_scalar_output(self, standard_inputs):
        attention, lora_q, delta_rel = standard_inputs
        loss = compute_audit_loss(attention, lora_q, delta_rel)
        assert loss.dim() == 0  # scalar

    def test_positive_loss(self, standard_inputs):
        attention, lora_q, delta_rel = standard_inputs
        loss = compute_audit_loss(attention, lora_q, delta_rel)
        assert loss.item() >= 0.0


class TestAuditLossZeroCases:
    def test_zero_when_q_is_one(self):
        """When q=1 everywhere, (1-q)=0, so loss should be zero."""
        N, T = 4, 37
        attention = torch.randn(N, T)
        lora_q = torch.ones(N, T)
        delta_rel = torch.randn(N)
        loss = compute_audit_loss(attention, lora_q, delta_rel)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_zero_when_delta_rel_is_zero(self):
        """When delta_rel=0, clip(|0|, 0, c)=0, so loss should be zero."""
        N, T = 4, 37
        attention = torch.randn(N, T)
        lora_q = torch.rand(N, T)
        delta_rel = torch.zeros(N)
        loss = compute_audit_loss(attention, lora_q, delta_rel)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)


class TestAuditLossMonotonicity:
    def test_larger_loss_for_low_quality_attended_tokens(self):
        """Loss is larger when attention concentrates on q=0 tokens."""
        N, T = 4, 37
        delta_rel = torch.ones(N) * 1.0

        # Case 1: attention on high-quality tokens (q=1)
        attention_good = torch.zeros(N, T)
        attention_good[:, 0] = 10.0  # concentrate on token 0
        lora_q_good = torch.ones(N, T)
        lora_q_good[:, 0] = 1.0  # token 0 is high quality
        loss_good = compute_audit_loss(attention_good, lora_q_good, delta_rel)

        # Case 2: attention on low-quality tokens (q=0)
        attention_bad = torch.zeros(N, T)
        attention_bad[:, 0] = 10.0  # concentrate on token 0
        lora_q_bad = torch.ones(N, T)
        lora_q_bad[:, 0] = 0.0  # token 0 is LOW quality
        loss_bad = compute_audit_loss(attention_bad, lora_q_bad, delta_rel)

        assert loss_bad.item() > loss_good.item()


class TestAuditLossGradientFlow:
    def test_gradient_flows_to_attention(self, standard_inputs):
        attention, lora_q, delta_rel = standard_inputs
        loss = compute_audit_loss(attention, lora_q, delta_rel)
        loss.backward()
        assert attention.grad is not None
        assert attention.grad.abs().sum().item() > 0

    def test_gradient_flows_to_lora_q(self):
        N, T = 4, 37
        attention = torch.randn(N, T)
        lora_q = torch.rand(N, T, requires_grad=True)
        delta_rel = torch.ones(N) * 1.5
        loss = compute_audit_loss(attention, lora_q, delta_rel)
        loss.backward()
        assert lora_q.grad is not None
        assert lora_q.grad.abs().sum().item() > 0

    def test_gradient_flows_to_delta_rel(self):
        N, T = 4, 37
        attention = torch.randn(N, T)
        lora_q = torch.rand(N, T)
        # Need delta_rel to require grad and be non-zero
        delta_rel = torch.tensor([1.0, -0.5, 0.8, -1.2], requires_grad=True)
        # lora_q needs non-uniform values for non-zero loss
        lora_q_fixed = torch.ones(N, T) * 0.5
        loss = compute_audit_loss(attention, lora_q_fixed, delta_rel)
        loss.backward()
        assert delta_rel.grad is not None
        assert delta_rel.grad.abs().sum().item() > 0


class TestAuditLossClip:
    def test_clip_bound_respected(self):
        """Verify that clip bound limits the contribution of large delta_rel."""
        N, T = 4, 37
        attention = torch.ones(N, T)
        lora_q = torch.zeros(N, T)  # all low quality

        # delta_rel within clip
        delta_small = torch.ones(N) * 1.0
        loss_small = compute_audit_loss(attention, lora_q, delta_small, clip=2.0)

        # delta_rel exceeding clip
        delta_large = torch.ones(N) * 100.0
        loss_large = compute_audit_loss(attention, lora_q, delta_large, clip=2.0)

        # delta_rel at clip boundary
        delta_clip = torch.ones(N) * 2.0
        loss_clip = compute_audit_loss(attention, lora_q, delta_clip, clip=2.0)

        # loss_large should equal loss_clip (both clipped to 2.0)
        assert loss_large.item() == pytest.approx(loss_clip.item(), abs=1e-6)
        # loss_small < loss_clip
        assert loss_small.item() < loss_clip.item()


class TestAuditLossInputValidation:
    def test_wrong_attention_dim(self):
        with pytest.raises(ValueError, match="must be 2D"):
            compute_audit_loss(
                torch.randn(4), torch.rand(4, 37), torch.randn(4),
            )

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            compute_audit_loss(
                torch.randn(4, 37), torch.rand(4, 10), torch.randn(4),
            )
