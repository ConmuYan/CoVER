from __future__ import annotations

from evidence.schema import ERR, EvidenceCard


class RuleTeacher:
    def generate(self, card: EvidenceCard) -> ERR:
        risk_type = self._determine_risk_type(card)
        supporting = self._select_supporting(card, risk_type)
        counter = self._select_counter(card)
        summary = self._build_summary(card, risk_type)

        return ERR(
            node_id=card.node_id,
            risk_type=risk_type,
            supporting_evidence=supporting,
            counter_evidence=counter,
            summary=summary,
        )

    def _determine_risk_type(self, card: EvidenceCard) -> str:
        r = card.reasoning

        if r.detector_signal_strength == "strong" and r.detector_signal == "embedding_neighbor_discrepancy_high":
            return "structural_discrepancy"

        if r.feature_neighbor_discrepancy == "high":
            return "feature_structure_conflict"

        if r.neighbor_consistency == "low":
            return "camouflage_neighbor"

        if r.detector_signal_strength == "strong" and "high_frequency_response" in r.detector_signal:
            return "spectral_anomaly"

        return "weak_or_uncertain_evidence"

    def _select_supporting(self, card: EvidenceCard, risk_type: str) -> list[str]:
        r = card.reasoning
        base = list(r.allowed_support_ids[:3])

        if risk_type == "spectral_anomaly":
            if "detector_signal" not in base:
                base = ["detector_signal"] + base
            if "detector_signal_strength" not in base:
                base = ["detector_signal_strength"] + base
        elif risk_type == "feature_structure_conflict":
            if "feature_neighbor_discrepancy" not in base:
                base = ["feature_neighbor_discrepancy"] + base
        elif risk_type == "structural_discrepancy":
            if r.detector_signal == "embedding_neighbor_discrepancy_high":
                if "detector_signal" not in base:
                    base = ["detector_signal"] + base
            elif r.degree_level == "high":
                if "degree_level" not in base:
                    base = ["degree_level"] + base
            elif r.neighbor_consistency == "low":
                if "neighbor_consistency" not in base:
                    base = ["neighbor_consistency"] + base

        return base[:5]

    def _select_counter(self, card: EvidenceCard) -> list[str]:
        return card.reasoning.allowed_counter_ids[:2]

    def _build_summary(self, card: EvidenceCard, risk_type: str) -> str:
        r = card.reasoning
        return (
            f"Node {card.node_id}: {risk_type}. "
            f"Degree={r.degree_level}, consistency={r.neighbor_consistency}, "
            f"discrepancy={r.feature_neighbor_discrepancy}, "
            f"signal={r.detector_signal}({r.detector_signal_strength})"
        )
