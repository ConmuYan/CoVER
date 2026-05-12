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
            },
        }


@dataclass
class ERR:
    node_id: int
    risk_type: str
    supporting_evidence: list[str]
    counter_evidence: list[str]
    summary: str
