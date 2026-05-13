"""End-to-end integration test for the directional pipeline."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.prototypes import PrototypeBuilder
from evidence.retrieval import HybridRetriever
from evidence.schema import CalibrationChannel, ERR, EvidenceCard, ReasoningChannel
from evidence.verifier import EvidenceContractVerifier, load_contracts
from evidence.vocab import get_evidence_slots, get_reason_types

training_losses = importlib.import_module("training.losses")


class TestEndToEnd:
    """Test end-to-end directional pipeline."""

    def test_full_pipeline(self):
        """Test schema → prototype → retrieval → verifier → loss."""
        err = ERR(
            node_id=0,
            risk_type="structural_discrepancy",
            supporting_evidence=["degree_level"],
            counter_evidence=["counter_signal"],
            summary="test",
            evidence_direction="increase_risk",
            evidence_strength="moderate",
            uncertainty_factors=["counter_signal"],
        )

        builder = PrototypeBuilder()
        train_mask = torch.tensor([True, True, False, False])
        y = torch.tensor([1, 0, 1, 0])
        evidence_tokens = {
            0: {"degree_level": "high"},
            1: {"degree_level": "low"},
            2: {"degree_level": "high"},
            3: {"degree_level": "low"},
        }
        base_preds = torch.tensor([1, 0, 1, 0])

        prototypes = builder.build(
            train_mask=train_mask,
            y=y,
            evidence_tokens=evidence_tokens,
            base_preds=base_preds,
        )

        retriever = HybridRetriever()
        retriever.fit_idf({i: evidence_tokens[i] for i in [0, 1]})

        target = {"degree_level": "high"}
        results = retriever.retrieve(target, prototypes["fraud_prototype_bank"])
        assert len(results) > 0

        contracts = load_contracts()
        verifier = EvidenceContractVerifier(contracts)

        card = EvidenceCard(
            node_id=0,
            detector_name="bwgnn",
            calibration=CalibrationChannel(base_score=0.5, uncertainty=0.1),
            reasoning=ReasoningChannel(
                degree_level="high",
                neighbor_consistency="low",
                feature_neighbor_discrepancy="high",
                detector_signal="strong",
                detector_signal_strength="strong",
                counter_signal="weak",
                allowed_support_ids=["degree_level"],
                allowed_counter_ids=["counter_signal"],
            ),
        )

        accepted, reasons = verifier.verify(err, card)
        assert accepted, f"Verifier rejected: {reasons}"

        num_slots = len(get_evidence_slots())
        num_types = len(get_reason_types())

        outputs = {
            "final_logit": torch.zeros(1, requires_grad=True),
            "type_logits": torch.zeros(1, num_types, requires_grad=True),
            "pos_logits": torch.zeros(1, num_slots, requires_grad=True),
            "neg_logits": torch.zeros(1, num_slots, requires_grad=True),
            "direction_logits": torch.zeros(1, 3, requires_grad=True),
        }
        targets = {
            "risk_type_id": torch.tensor([0]),
            "pos_mask": torch.zeros(1, num_slots),
            "neg_mask": torch.zeros(1, num_slots),
            "accepted_mask": torch.ones(1),
        }
        targets["pos_mask"][0, 0] = 1.0
        targets["neg_mask"][0, 5] = 1.0

        loss, stats = training_losses.compute_cvscd_loss(
            outputs,
            y=torch.tensor([1.0]),
            targets=targets,
            train_mask=torch.ones(1, dtype=torch.bool),
            base_logits=torch.zeros(1),
            risk_type_names=get_reason_types(),
            direction_ids=torch.tensor([0]),
            lambda_direction=1.0,
        )

        assert not torch.isnan(loss)
        assert stats["direction_ce_loss"] > 0
