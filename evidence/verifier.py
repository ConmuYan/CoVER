from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from evidence.schema import ERR, EvidenceCard

SCORE_LEAKAGE_KEYS = frozenset({
    "base_score", "score", "logit", "logits",
    "prob", "probs", "probability", "probabilities",
    "confidence", "prediction", "pred", "label", "y", "target",
})

VALID_RISK_TYPES = frozenset({
    "structural_discrepancy",
    "camouflage_neighbor",
    "spectral_anomaly",
    "feature_structure_conflict",
    "relation_or_burst_anomaly",
    "weak_or_uncertain_evidence",
})

FRAUD_SPECIFIC_TYPES = frozenset({
    "spectral_anomaly",
    "camouflage_neighbor",
    "relation_or_burst_anomaly",
})


def load_contracts(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).parent / "contracts.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class EvidenceContractVerifier:
    def __init__(
        self,
        contracts: dict[str, Any],
        enable_label_compatibility: bool = False,
    ):
        self.contracts = contracts
        self.enable_label_compatibility = enable_label_compatibility

    def verify(
        self,
        err: ERR,
        card: EvidenceCard,
        label: int | None = None,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []

        reasons += self._check_schema(err)
        reasons += self._check_availability(err, card)
        reasons += self._check_role_consistency(err, card)
        reasons += self._check_contract(err, card)
        reasons += self._check_score_blindness(err)

        if self.enable_label_compatibility and label is not None:
            reasons += self._check_label_compatibility(err, label)

        accepted = len(reasons) == 0
        return accepted, reasons

    def _check_schema(self, err: ERR) -> list[str]:
        reasons = []
        if err.risk_type not in VALID_RISK_TYPES:
            reasons.append("invalid_risk_type")
        if not isinstance(err.supporting_evidence, list):
            reasons.append("invalid_evidence_type")
        if not isinstance(err.counter_evidence, list):
            reasons.append("invalid_evidence_type")
        return reasons

    def _check_availability(self, err: ERR, card: EvidenceCard) -> list[str]:
        reasons = []
        rea = card.reasoning

        available_fields = {
            "degree_level", "neighbor_consistency", "feature_neighbor_discrepancy",
            "detector_signal", "detector_signal_strength", "counter_signal",
        }
        available_ids = set(rea.allowed_support_ids) | set(rea.allowed_counter_ids)
        available = available_fields | available_ids

        cited = set(err.supporting_evidence) | set(err.counter_evidence)
        unavailable = cited - available
        if unavailable:
            reasons.append("unavailable_evidence")

        return reasons

    def _check_role_consistency(self, err: ERR, card: EvidenceCard) -> list[str]:
        reasons = []
        rea = card.reasoning

        support_set = set(err.supporting_evidence)
        counter_set = set(err.counter_evidence)

        allowed_support = set(rea.allowed_support_ids)
        allowed_counter = set(rea.allowed_counter_ids)

        if not support_set.issubset(allowed_support):
            reasons.append("invalid_support_role")
        if not counter_set.issubset(allowed_counter):
            reasons.append("invalid_counter_role")
        if support_set & counter_set:
            reasons.append("support_counter_overlap")

        return reasons

    def _check_contract(self, err: ERR, card: EvidenceCard) -> list[str]:
        reasons = []
        rea = card.reasoning

        spec = self.contracts.get(err.risk_type)
        if spec is None:
            return reasons

        required_any = spec.get("required_any", [])
        forbidden = spec.get("forbidden", [])

        if required_any:
            satisfied = False
            for cond in required_any:
                field = cond["field"]
                values = cond["values"]
                rea_value = getattr(rea, field, None)
                if rea_value in values and field in err.supporting_evidence:
                    satisfied = True
                    break
            if not satisfied:
                reasons.append("contract_required_not_satisfied")

        for cond in forbidden:
            field = cond["field"]
            values = cond["values"]
            rea_value = getattr(rea, field, None)
            if rea_value in values:
                reasons.append("contract_forbidden_hit")

        return reasons

    def _check_score_blindness(self, err: ERR) -> list[str]:
        reasons = []
        cited = set(err.supporting_evidence) | set(err.counter_evidence)
        if cited & SCORE_LEAKAGE_KEYS:
            reasons.append("score_leakage")
        return reasons

    def _check_label_compatibility(self, err: ERR, label: int) -> list[str]:
        reasons = []
        if label == 0 and err.risk_type in FRAUD_SPECIFIC_TYPES:
            reasons.append("label_incompatible")
        return reasons
