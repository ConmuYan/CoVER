"""Assert that RAERTeacher (post Commit 1 v2) never produces
non-zero alpha_llm/delta_llm and raises NotImplementedError when
use_judge=True or judge_features is not None.
"""

from __future__ import annotations

import pytest
import torch

from models.raer_teacher import RAERTeacher


@pytest.fixture
def reasoner():
    """Minimal RAERTeacher for testing."""
    return RAERTeacher(
        base_z_dim=64,
        relation_names=["RUR", "RSR", "RTR"],
        anchor_relation="RUR",
        rel_stat_dim=9,
        rel_hidden_dim=64,
    )


@pytest.fixture
def dummy_inputs():
    """Dummy inputs for forward pass."""
    N = 8
    return {
        "base_z": torch.randn(N, 64),
        "base_logit": torch.randn(N),
        "relation_features": torch.randn(N, 3 * 9),
    }


class TestNoJudgePath:
    def test_alpha_llm_always_zero(self, reasoner, dummy_inputs):
        """alpha_llm must be identically zero in all forward passes."""
        out = reasoner(**dummy_inputs)
        assert (out["alpha_llm"] == 0.0).all(), (
            f"alpha_llm has non-zero values: max_abs = "
            f"{out['alpha_llm'].abs().max().item()}"
        )

    def test_delta_llm_always_zero(self, reasoner, dummy_inputs):
        """delta_llm must be identically zero in all forward passes."""
        out = reasoner(**dummy_inputs)
        assert (out["delta_llm"] == 0.0).all(), (
            f"delta_llm has non-zero values: max_abs = "
            f"{out['delta_llm'].abs().max().item()}"
        )

    def test_use_judge_true_raises(self):
        """Constructing with use_judge=True must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Judge fusion path"):
            RAERTeacher(
                base_z_dim=64,
                relation_names=["RUR", "RSR", "RTR"],
                anchor_relation="RUR",
                use_judge=True,
            )

    def test_judge_features_not_none_raises(self, reasoner, dummy_inputs):
        """Passing judge_features != None to forward must raise."""
        with pytest.raises(NotImplementedError, match="Judge fusion path"):
            reasoner(
                **dummy_inputs,
                judge_features=torch.randn(8, 32),
            )
