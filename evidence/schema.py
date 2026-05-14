from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalibrationChannel:
    base_score: float
    uncertainty: float


@dataclass
class ReasoningChannel:
    degree_level: str
    neighbor_consistency: str
    feature_neighbor_discrepancy: str
    detector_signal: str
    detector_signal_strength: str
    counter_signal: str
    allowed_support_ids: list[str] = field(default_factory=list)
    allowed_counter_ids: list[str] = field(default_factory=list)

    # structural fields
    degree_percentile_bucket: str = "unknown"
    neighbor_degree_skew_bucket: str = "unknown"
    two_hop_consistency_bucket: str = "unknown"

    # feature-structure conflict fields
    feature_neighbor_cosine_bucket: str = "unknown"
    embedding_neighbor_cosine_bucket: str = "unknown"
    feature_embedding_disagreement_bucket: str = "unknown"

    # BWGNN / spectral fields
    bwgnn_low_band_energy_bucket: str = "unknown"
    bwgnn_mid_band_energy_bucket: str = "unknown"
    bwgnn_high_band_energy_bucket: str = "unknown"
    bwgnn_high_low_energy_ratio_bucket: str = "unknown"
    message_residual_bucket: str = "unknown"

    # prototype-relative fields
    closer_to_fraud_prototype: str = "unknown"
    closer_to_benign_prototype: str = "unknown"
    fraud_prototype_matching_fields: list = field(default_factory=list)
    benign_prototype_matching_fields: list = field(default_factory=list)
    prototype_conflict_level: str = "unknown"

    # evidence polarity fields
    fraud_token_count: int = 0
    benign_token_count: int = 0
    neutral_token_count: int = 0
    evidence_polarity: str = "unknown"


@dataclass
class EvidenceCard:
    node_id: int
    detector_name: str
    calibration: CalibrationChannel
    reasoning: ReasoningChannel

    def to_teacher_payload(self) -> dict:
        return {
            "node_id": self.node_id,
            "detector_name": self.detector_name,
            "reasoning": {
                "degree_level": self.reasoning.degree_level,
                "neighbor_consistency": self.reasoning.neighbor_consistency,
                "feature_neighbor_discrepancy": self.reasoning.feature_neighbor_discrepancy,
                "detector_signal": self.reasoning.detector_signal,
                "detector_signal_strength": self.reasoning.detector_signal_strength,
                "counter_signal": self.reasoning.counter_signal,
                "allowed_support_ids": self.reasoning.allowed_support_ids,
                "allowed_counter_ids": self.reasoning.allowed_counter_ids,
                "degree_percentile_bucket": self.reasoning.degree_percentile_bucket,
                "neighbor_degree_skew_bucket": self.reasoning.neighbor_degree_skew_bucket,
                "two_hop_consistency_bucket": self.reasoning.two_hop_consistency_bucket,
                "feature_neighbor_cosine_bucket": self.reasoning.feature_neighbor_cosine_bucket,
                "embedding_neighbor_cosine_bucket": self.reasoning.embedding_neighbor_cosine_bucket,
                "feature_embedding_disagreement_bucket": self.reasoning.feature_embedding_disagreement_bucket,
                "bwgnn_low_band_energy_bucket": self.reasoning.bwgnn_low_band_energy_bucket,
                "bwgnn_mid_band_energy_bucket": self.reasoning.bwgnn_mid_band_energy_bucket,
                "bwgnn_high_band_energy_bucket": self.reasoning.bwgnn_high_band_energy_bucket,
                "bwgnn_high_low_energy_ratio_bucket": self.reasoning.bwgnn_high_low_energy_ratio_bucket,
                "message_residual_bucket": self.reasoning.message_residual_bucket,
                "closer_to_fraud_prototype": self.reasoning.closer_to_fraud_prototype,
                "closer_to_benign_prototype": self.reasoning.closer_to_benign_prototype,
                "fraud_prototype_matching_fields": self.reasoning.fraud_prototype_matching_fields,
                "benign_prototype_matching_fields": self.reasoning.benign_prototype_matching_fields,
                "prototype_conflict_level": self.reasoning.prototype_conflict_level,
                "fraud_token_count": self.reasoning.fraud_token_count,
                "benign_token_count": self.reasoning.benign_token_count,
                "neutral_token_count": self.reasoning.neutral_token_count,
                "evidence_polarity": self.reasoning.evidence_polarity,
            },
        }


@dataclass
class ERR:
    node_id: int
    risk_type: str
    supporting_evidence: list[str]
    counter_evidence: list[str]
    summary: str
    evidence_direction: str = "uncertain"
    evidence_strength: str = "weak"
    uncertainty_factors: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> ERR:
        return cls(
            node_id=d["node_id"],
            risk_type=d["risk_type"],
            supporting_evidence=d.get("supporting_evidence", []),
            counter_evidence=d.get("counter_evidence", []),
            summary=d.get("summary", ""),
            evidence_direction=d.get("evidence_direction", "uncertain"),
            evidence_strength=d.get("evidence_strength", "weak"),
            uncertainty_factors=d.get("uncertainty_factors", []),
        )
