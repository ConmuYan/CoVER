from __future__ import annotations

import torch
from torch import Tensor

from evidence.schema import CalibrationChannel, EvidenceCard, ReasoningChannel


class EvidenceAdapter:
    def __init__(self, detector_name: str, x: Tensor, edge_index: Tensor):
        self.detector_name = detector_name
        self.x = x
        self.edge_index = edge_index
        self.num_nodes = x.shape[0]

        self.degrees = self._compute_degrees()

    def _compute_degrees(self) -> Tensor:
        row, _ = self.edge_index
        degree = torch.zeros(self.num_nodes, dtype=torch.long)
        ones = torch.ones(row.shape[0], dtype=torch.long)
        degree.scatter_add_(0, row, ones)
        return degree

    def _compute_neighbor_mean(self, embeddings: Tensor) -> Tensor:
        row, col = self.edge_index
        neighbor_mean = torch.zeros_like(embeddings)
        count = torch.zeros(self.num_nodes, 1, dtype=torch.float)
        neighbor_mean.index_add_(0, row, embeddings[col])
        count.index_add_(0, row, torch.ones(row.shape[0], 1))
        count = count.clamp(min=1)
        return neighbor_mean / count

    def extract(
        self,
        node_ids: list[int],
        base_logits: Tensor,
        embeddings: Tensor,
        extras: dict[str, Tensor] | None = None,
    ) -> list[EvidenceCard]:
        cards = []
        for node_id in node_ids:
            card = self._extract_single(node_id, base_logits, embeddings, extras)
            cards.append(card)
        return cards

    def _extract_single(
        self,
        node_id: int,
        base_logits: Tensor,
        embeddings: Tensor,
        extras: dict[str, Tensor] | None = None,
    ) -> EvidenceCard:
        degree = self.degrees[node_id].item()
        degree_level = self._level(degree=degree)

        neighbor_emb_mean = self._compute_neighbor_mean(embeddings)
        node_emb = embeddings[node_id]
        discrepancy = torch.norm(node_emb - neighbor_emb_mean[node_id]).item()
        feature_neighbor_discrepancy = self._level_threshold(discrepancy, low=0.5, high=2.0)

        row, col = self.edge_index
        neighbor_mask = row == node_id
        neighbor_ids = col[neighbor_mask]

        if len(neighbor_ids) > 0:
            neighbor_logits = base_logits[neighbor_ids]
            logit_std = neighbor_logits.std().item()
            neighbor_consistency = self._level_threshold(1.0 - logit_std, low=0.3, high=0.7)
        else:
            neighbor_consistency = "low"

        logit = base_logits[node_id].item()

        if extras and "high_freq_response" in extras:
            hf_response = extras["high_freq_response"]
            hf_value = hf_response[node_id].item()
            q_low = hf_response.quantile(0.33).item()
            q_high = hf_response.quantile(0.66).item()

            if hf_value > q_high:
                detector_signal = "high_frequency_response_high"
                detector_signal_strength = "strong"
            elif hf_value > q_low:
                detector_signal = "high_frequency_response_medium"
                detector_signal_strength = "moderate"
            else:
                detector_signal = "high_frequency_response_low"
                detector_signal_strength = "weak"
        else:
            detector_signal = "embedding_neighbor_discrepancy_high" if discrepancy > 2.0 else "normal"
            detector_signal_strength = "strong" if abs(logit) > 1.0 else "weak"

        counter_signal = "benign_neighbor_signal_low" if neighbor_consistency == "low" else "benign_neighbor_signal_high"

        allowed_support_ids = [
            "degree_level", "neighbor_consistency", "feature_neighbor_discrepancy",
            "detector_signal", "detector_signal_strength",
        ] + [f"neighbor_{nid.item()}" for nid in neighbor_ids[:5]]
        allowed_counter_ids = [
            "counter_signal",
        ] + [f"counter_{nid.item()}" for nid in neighbor_ids[:3]]

        calibration = CalibrationChannel(
            base_score=torch.sigmoid(base_logits[node_id]).item(),
            uncertainty=abs(logit),
        )

        reasoning = ReasoningChannel(
            degree_level=degree_level,
            neighbor_consistency=neighbor_consistency,
            feature_neighbor_discrepancy=feature_neighbor_discrepancy,
            detector_signal=detector_signal,
            detector_signal_strength=detector_signal_strength,
            counter_signal=counter_signal,
            allowed_support_ids=allowed_support_ids,
            allowed_counter_ids=allowed_counter_ids,
        )

        return EvidenceCard(
            node_id=node_id,
            detector_name=self.detector_name,
            calibration=calibration,
            reasoning=reasoning,
        )

    def _level(self, degree: int) -> str:
        if degree <= 5:
            return "low"
        elif degree <= 20:
            return "medium"
        else:
            return "high"

    def _level_threshold(self, value: float, low: float, high: float) -> str:
        if value < low:
            return "low"
        elif value < high:
            return "medium"
        else:
            return "high"
