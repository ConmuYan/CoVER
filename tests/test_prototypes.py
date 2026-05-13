import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.prototypes import PrototypeBuilder, FORBIDDEN_FIELDS, SCORE_BLIND_FIELDS


def _make_test_data():
    train_mask = torch.tensor([True, True, False, False, True])
    y = torch.tensor([1, 0, 1, 0, 1])
    base_preds = torch.tensor([0, 0, 1, 1, 1])

    evidence_tokens = {
        0: {
            "degree_level": "low",
            "detector_signal": "high_frequency_response_high",
            "feature_neighbor_discrepancy": "high",
            "detector_signal_strength": "strong",
        },
        1: {
            "degree_level": "high",
            "detector_signal": "normal",
            "feature_neighbor_discrepancy": "low",
            "detector_signal_strength": "weak",
        },
        2: {
            "degree_level": "medium",
            "detector_signal": "high_frequency_response_medium",
        },
        3: {
            "degree_level": "high",
            "detector_signal": "normal",
        },
        4: {
            "degree_level": "low",
            "detector_signal": "high_frequency_response_high",
            "feature_neighbor_discrepancy": "high",
        },
    }

    builder = PrototypeBuilder()
    return builder.build(train_mask, y, evidence_tokens, base_preds)


class TestPrototypeBuilderTrainOnly:
    def test_no_test_labels_used(self):
        """Builder should only use train nodes (0, 1), not test nodes (2, 3)."""
        result = _make_test_data()

        fraud_bank = result["fraud_prototype_bank"]
        benign_bank = result["benign_prototype_bank"]

        # train nodes are 0, 1, 4 — test nodes are 2, 3
        assert 0 in fraud_bank
        assert 4 in fraud_bank
        assert 1 in benign_bank
        assert 2 not in fraud_bank
        assert 2 not in benign_bank
        assert 3 not in fraud_bank
        assert 3 not in benign_bank

    def test_fn_fp_banks_created(self):
        """FN/FP banks should be created from train errors."""
        # Node 0: label=1, pred=0 → FN
        # Node 3: label=0, pred=1 → FP
        train_mask = torch.tensor([True, True, True, True, False])
        y = torch.tensor([1, 0, 1, 0, 1])
        base_preds = torch.tensor([0, 0, 1, 1, 0])

        evidence_tokens = {
            0: {"degree_level": "low"},
            1: {"degree_level": "high"},
            2: {"degree_level": "low"},
            3: {"degree_level": "high"},
            4: {"degree_level": "medium"},
        }

        builder = PrototypeBuilder()
        result = builder.build(train_mask, y, evidence_tokens, base_preds)

        fn_bank = result["train_fn_bank"]
        fp_bank = result["train_fp_bank"]

        assert 0 in fn_bank
        # node 2: label=1, pred=1 → correct, not FN
        assert 2 not in fn_bank
        # node 3: label=0, pred=1 → FP
        assert 3 in fp_bank
        # node 1: label=0, pred=0 → correct, not FP
        assert 1 not in fp_bank

    def test_all_expected_keys_present(self):
        result = _make_test_data()

        expected_keys = [
            "fraud_prototype_bank",
            "benign_prototype_bank",
            "train_fn_bank",
            "train_fp_bank",
            "fraud_prototype_summary",
            "benign_prototype_summary",
            "normal_structure_summary",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestDistinctiveTokens:
    def test_distinctive_tokens_computed(self):
        """Prototype summaries should include distinctive_tokens list."""
        result = _make_test_data()

        fraud_summary = result["fraud_prototype_summary"]
        benign_summary = result["benign_prototype_summary"]

        assert "distinctive_tokens" in fraud_summary
        assert "distinctive_tokens" in benign_summary
        assert isinstance(fraud_summary["distinctive_tokens"], list)
        assert isinstance(benign_summary["distinctive_tokens"], list)

    def test_distinctive_tokens_have_expected_fields(self):
        result = _make_test_data()

        fraud_distinctive = result["fraud_prototype_summary"]["distinctive_tokens"]
        if fraud_distinctive:
            entry = fraud_distinctive[0]
            assert "field" in entry
            assert "value" in entry
            assert "log_odds" in entry
            assert entry["log_odds"] > 0.5

    def test_token_frequencies_in_summary(self):
        result = _make_test_data()

        fraud_summary = result["fraud_prototype_summary"]
        assert "token_frequencies" in fraud_summary
        assert "num_nodes" in fraud_summary
        assert fraud_summary["num_nodes"] == len(result["fraud_prototype_bank"])


class TestNoScoreInPrototypes:
    def test_prototypes_score_blind(self):
        """No base_score, score, logit, prob, confidence, prediction in output."""
        result = _make_test_data()

        all_banks = [
            result["fraud_prototype_bank"],
            result["benign_prototype_bank"],
            result["train_fn_bank"],
            result["train_fp_bank"],
        ]

        for bank in all_banks:
            for node_id, tokens in bank.items():
                for field in tokens:
                    assert field not in FORBIDDEN_FIELDS, (
                        f"Node {node_id} has forbidden field '{field}'"
                    )

    def test_filter_score_blind_strips_forbidden(self):
        tokens = {
            "degree_level": "high",
            "base_score": "0.95",
            "score": "high_risk",
            "logit": "2.1",
            "prob": "0.95",
            "confidence": "0.9",
            "prediction": "1",
        }
        filtered = PrototypeBuilder._filter_score_blind(tokens)
        assert "degree_level" in filtered
        for forbidden in ["base_score", "score", "logit", "prob", "confidence", "prediction"]:
            assert forbidden not in filtered

    def test_output_serializable(self):
        result = _make_test_data()
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert "fraud_prototype_bank" in deserialized


class TestNormalStructureSummary:
    def test_summary_fields(self):
        result = _make_test_data()

        summary = result["normal_structure_summary"]
        assert "field_modes" in summary
        assert "num_train_nodes" in summary
        # train nodes: 0, 1, 4
        assert summary["num_train_nodes"] == 3

    def test_field_modes_cover_score_blind_fields(self):
        result = _make_test_data()

        modes = result["normal_structure_summary"]["field_modes"]
        for field in SCORE_BLIND_FIELDS:
            assert field in modes
