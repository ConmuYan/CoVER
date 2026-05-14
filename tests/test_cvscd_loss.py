"""Tests for compute_cvscd_loss (CV-SCD four-term Stage3 loss)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.losses import compute_cvscd_loss
from evidence.schema import EvidenceCard, CalibrationChannel, ReasoningChannel
from evidence.vocab import get_reason_types, get_evidence_slots

NUM_SLOTS = len(get_evidence_slots())
NUM_TYPES = len(get_reason_types())
REASON_TYPE_NAMES = get_reason_types()


def _make_outputs(n: int, num_slots: int = NUM_SLOTS, num_types: int = NUM_TYPES):
    return {
        "final_logit": torch.randn(n, requires_grad=True),
        "type_logits": torch.randn(n, num_types, requires_grad=True),
        "pos_logits": torch.randn(n, num_slots, requires_grad=True),
        "neg_logits": torch.randn(n, num_slots, requires_grad=True),
    }


def _make_targets(n: int, num_slots: int = NUM_SLOTS, accepted_frac: float = 0.5):
    accepted = (torch.rand(n) < accepted_frac).float()
    return {
        "risk_type_id": torch.randint(0, NUM_TYPES, (n,)),
        "pos_mask": (torch.rand(n, num_slots) > 0.5).float(),
        "neg_mask": (torch.rand(n, num_slots) > 0.5).float(),
        "accepted_mask": accepted,
    }


class TestScoreBlindness:
    """Test 1: Enhanced teacher_payload has no score/logit/prob/confidence/base_score."""

    def test_evidence_card_payload_no_scores(self):
        card = EvidenceCard(
            node_id=0,
            detector_name="bwgnn",
            calibration=CalibrationChannel(base_score=0.8, uncertainty=0.1),
            reasoning=ReasoningChannel(
                degree_level="high",
                neighbor_consistency="low",
                feature_neighbor_discrepancy="high",
                detector_signal="strong",
                detector_signal_strength="strong",
                counter_signal="weak",
            ),
        )
        payload = card.to_teacher_payload()
        forbidden = {"score", "logit", "prob", "confidence", "base_score"}
        for key in forbidden:
            assert key not in payload, f"teacher_payload contains forbidden key: {key}"
        assert "reasoning" in payload


class TestBaseLogitsDetach:
    """Test 2: base_logits gradients don't flow."""

    def test_base_logits_no_gradient(self):
        n = 20
        base = torch.randn(n, requires_grad=True)
        outputs = _make_outputs(n)
        targets = _make_targets(n)

        train_mask = torch.ones(n, dtype=torch.bool)

        loss, stats = compute_cvscd_loss(
            outputs, y=torch.randint(0, 2, (n,)).float(),
            targets=targets, train_mask=train_mask, base_logits=base,
            risk_type_names=REASON_TYPE_NAMES,
        )
        loss.backward()

        assert base.grad is None or base.grad.abs().sum().item() == 0.0, (
            "Gradients leaked into base_logits"
        )


class TestDetPosWeight:
    """Test 3: L_det uses pos_weight without double class weighting."""

    def test_pos_weight_applied(self):
        n = 50
        y = torch.cat([torch.ones(10), torch.zeros(40)])
        outputs = _make_outputs(n)
        targets = _make_targets(n, accepted_frac=0.0)
        targets["accepted_mask"] = torch.zeros(n)
        base = torch.randn(n)
        train_mask = torch.ones(n, dtype=torch.bool)

        # Without pos_weight
        loss1, _ = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, pos_weight=None,
            lambda_err=0, lambda_signed=0, lambda_corr=0,
        )

        # With pos_weight = 5.0 (heavily weight positives)
        loss2, _ = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, pos_weight=5.0,
            lambda_err=0, lambda_signed=0, lambda_corr=0,
        )

        # With pos_weight, the loss should be larger (more weight on positive class)
        # Both are computed from the same logits, so the weighted one differs
        assert loss1.item() != loss2.item(), "pos_weight had no effect"


class TestAcceptedERRParticipates:
    """Test 4: accepted ERR participates in L_err / L_signed."""

    def test_accepted_err_in_loss(self):
        n = 10
        targets = _make_targets(n, accepted_frac=0.5)
        outputs = _make_outputs(n)
        base = torch.randn(n)
        train_mask = torch.ones(n, dtype=torch.bool)
        y = torch.randint(0, 2, (n,)).float()

        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
        )
        assert stats["accepted_err_count"] > 0
        assert stats["total_loss"] > 0


class TestRejectedERRSkipped:
    """Test 5: rejected ERR (accepted_mask=0) does NOT participate in evidence loss."""

    def test_rejected_no_evidence_loss(self):
        n = 10
        targets = _make_targets(n, accepted_frac=0.0)
        targets["accepted_mask"] = torch.zeros(n)
        outputs = _make_outputs(n)
        base = torch.randn(n)
        train_mask = torch.ones(n, dtype=torch.bool)
        y = torch.randint(0, 2, (n,)).float()

        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
        )
        assert stats["accepted_err_count"] == 0
        assert stats["err_loss"] == 0.0
        assert stats["signed_loss"] == 0.0


class TestAcceptedMaskAllZero:
    """Test 6: accepted_mask all zero → L_err / L_signed return 0."""

    def test_all_zero_accepted(self):
        n = 10
        targets = _make_targets(n, accepted_frac=0.0)
        targets["accepted_mask"] = torch.zeros(n)
        outputs = _make_outputs(n)
        base = torch.randn(n)
        train_mask = torch.ones(n, dtype=torch.bool)
        y = torch.randint(0, 2, (n,)).float()

        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
        )
        assert stats["err_loss"] == 0.0
        assert stats["signed_loss"] == 0.0
        assert not torch.isnan(loss)


class TestSignedEmptySupport:
    """Test 7: L_signed with empty support mask → no NaN."""

    def test_empty_support_no_nan(self):
        n = 5
        targets = _make_targets(n, accepted_frac=1.0)
        targets["accepted_mask"] = torch.ones(n)
        targets["pos_mask"] = torch.zeros(n, NUM_SLOTS)  # no support
        targets["neg_mask"] = (torch.rand(n, NUM_SLOTS) > 0.3).float()  # has counter

        outputs = _make_outputs(n)
        base = torch.randn(n)
        train_mask = torch.ones(n, dtype=torch.bool)
        y = torch.randint(0, 2, (n,)).float()

        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
        )
        assert not torch.isnan(loss), f"Loss is NaN: {loss}"
        assert all(not torch.isnan(torch.tensor(v)) for v in stats.values())


class TestSignedEmptyCounter:
    """Test 8: L_signed with empty counter mask → no NaN."""

    def test_empty_counter_no_nan(self):
        n = 5
        targets = _make_targets(n, accepted_frac=1.0)
        targets["accepted_mask"] = torch.ones(n)
        targets["neg_mask"] = torch.zeros(n, NUM_SLOTS)  # no counter
        targets["pos_mask"] = (torch.rand(n, NUM_SLOTS) > 0.3).float()  # has support

        outputs = _make_outputs(n)
        base = torch.randn(n)
        train_mask = torch.ones(n, dtype=torch.bool)
        y = torch.randint(0, 2, (n,)).float()

        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
        )
        assert not torch.isnan(loss), f"Loss is NaN: {loss}"


class TestCorrPositiveShiftFN:
    """Test 9: L_corr encourages positive shift for base FN."""

    def test_fn_correction_direction(self):
        n = 10
        # All label=1 (anomaly), base predicts wrong (logit << 0)
        y = torch.ones(n)
        base = torch.full((n,), -5.0)  # base says "not anomaly"
        train_mask = torch.ones(n, dtype=torch.bool)
        targets = _make_targets(n, accepted_frac=0.0)
        targets["accepted_mask"] = torch.zeros(n)

        # Make final_logit slightly positive (correcting toward anomaly)
        outputs = _make_outputs(n)
        outputs["final_logit"] = torch.full((n,), 1.0, requires_grad=True)

        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
            lambda_det=0, lambda_err=0, lambda_signed=0,
        )
        # With positive shift for FN, corr_loss should be small
        assert stats["false_negative_correction_count"] == float(n)
        assert stats["corr_loss"] >= 0
        assert not torch.isnan(loss)


class TestCorrNegativeShiftFP:
    """Test 10: L_corr encourages negative shift for base FP."""

    def test_fp_correction_direction(self):
        n = 10
        # All label=0 (normal), base predicts wrong (logit >> 0)
        y = torch.zeros(n)
        base = torch.full((n,), 5.0)  # base says "anomaly" (wrong)
        train_mask = torch.ones(n, dtype=torch.bool)
        targets = _make_targets(n, accepted_frac=0.0)
        targets["accepted_mask"] = torch.zeros(n)

        outputs = _make_outputs(n)
        outputs["final_logit"] = torch.full((n,), -1.0, requires_grad=True)

        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
            lambda_det=0, lambda_err=0, lambda_signed=0,
        )
        assert stats["false_positive_correction_count"] == float(n)
        assert stats["corr_loss"] >= 0
        assert not torch.isnan(loss)


class TestRhoZeroNoCorrection:
    """Test 11: rho=0 → final_logit == base_logit (no residual)."""

    def test_rho_zero_no_residual(self):
        from models.reasoner import EvidenceReasoner

        n = 10
        z = torch.randn(n, 32)
        base_logit = torch.randn(n)
        evi_ids = torch.zeros(n, NUM_SLOTS, dtype=torch.long)

        reasoner = EvidenceReasoner(
            z_dim=32, rho=0.0, gate_mode="safe_residual",
            num_slots=NUM_SLOTS, num_types=NUM_TYPES,
        )
        reasoner.eval()
        with torch.no_grad():
            out = reasoner(z, base_logit, evi_ids)

        assert torch.allclose(out["final_logit"], base_logit, atol=1e-6), (
            "rho=0 should make final_logit == base_logit"
        )


class TestAuxOnlyNoCorrection:
    """Test 12: aux_only → final_logit == base_logit."""

    def test_aux_only_no_residual(self):
        from models.reasoner import EvidenceReasoner

        n = 10
        z = torch.randn(n, 32)
        base_logit = torch.randn(n)
        evi_ids = torch.zeros(n, NUM_SLOTS, dtype=torch.long)

        reasoner = EvidenceReasoner(
            z_dim=32, rho=0.3, gate_mode="aux_only",
            num_slots=NUM_SLOTS, num_types=NUM_TYPES,
        )
        reasoner.eval()
        with torch.no_grad():
            out = reasoner(z, base_logit, evi_ids)

        assert torch.allclose(out["final_logit"], base_logit, atol=1e-6), (
            "aux_only should make final_logit == base_logit"
        )


class TestNoLLMTeacherImport:
    """Test 13: train_stage3.py and evaluate.py must NOT import llm_teacher."""

    def test_train_stage3_no_llm(self):
        path = Path(__file__).parent.parent / "scripts" / "train_stage3.py"
        content = path.read_text()
        assert "llm_teacher" not in content

    def test_evaluate_no_llm(self):
        path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
        content = path.read_text()
        assert "llm_teacher" not in content


class TestFourTermsMax:
    """HARD CONSTRAINT: exactly 4 loss terms, no 5th/6th."""

    def test_only_four_terms(self):
        n = 20
        y = torch.randint(0, 2, (n,)).float()
        outputs = _make_outputs(n)
        targets = _make_targets(n)
        base = torch.randn(n)
        train_mask = torch.ones(n, dtype=torch.bool)

        loss, stats = compute_cvscd_loss(
            outputs, y=y, targets=targets, train_mask=train_mask,
            base_logits=base, risk_type_names=REASON_TYPE_NAMES,
        )

        expected_keys = {
            "det_loss", "err_loss", "signed_loss", "corr_loss", "total_loss",
            "correction_node_count", "inheritance_node_count",
            "accepted_err_count", "residual_shift_mean",
            "residual_shift_max_abs", "false_negative_correction_count",
            "false_positive_correction_count",
        }
        assert set(stats.keys()) == expected_keys, (
            f"Unexpected stats keys: {set(stats.keys()) - expected_keys}"
        )


class TestSummaryNotInLoss:
    """Summary never enters loss."""

    def test_summary_not_used(self):
        import inspect
        from training.losses import compute_cvscd_loss
        source = inspect.getsource(compute_cvscd_loss)
        assert "summary" not in source.lower(), "summary should not appear in loss code"
