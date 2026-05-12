import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence.schema import CalibrationChannel, ReasoningChannel, EvidenceCard, ERR
from evidence.adapter import EvidenceAdapter
from evidence.rule_teacher import RuleTeacher
from evidence.verifier import EvidenceContractVerifier, load_contracts
import torch


def _make_small_graph():
    x = torch.randn(20, 8)
    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3],
    ])
    return x, edge_index


def test_stage2_with_verifier():
    x, edge_index = _make_small_graph()
    adapter = EvidenceAdapter("gcn", x, edge_index)

    node_ids = list(range(10))
    base_logits = torch.randn(20)
    embeddings = torch.randn(20, 16)

    cards = adapter.extract(node_ids, base_logits, embeddings)

    teacher = RuleTeacher()
    contracts = load_contracts()
    verifier = EvidenceContractVerifier(contracts, enable_label_compatibility=False)

    accepted_errs = []
    rejected_errs = []

    for card in cards:
        err = teacher.generate(card)
        accepted, reasons = verifier.verify(err, card)
        if accepted:
            accepted_errs.append(err)
        else:
            rejected_errs.append({"err": err, "reasons": reasons})

    total = len(accepted_errs) + len(rejected_errs)
    assert total == len(cards)
    assert len(accepted_errs) > 0


def test_verifier_stats_structure():
    x, edge_index = _make_small_graph()
    adapter = EvidenceAdapter("gcn", x, edge_index)

    node_ids = list(range(5))
    cards = adapter.extract(node_ids, torch.randn(20), torch.randn(20, 16))

    teacher = RuleTeacher()
    contracts = load_contracts()
    verifier = EvidenceContractVerifier(contracts)

    num_accepted = 0
    num_rejected = 0

    for card in cards:
        err = teacher.generate(card)
        accepted, _ = verifier.verify(err, card)
        if accepted:
            num_accepted += 1
        else:
            num_rejected += 1

    assert num_accepted + num_rejected == 5
