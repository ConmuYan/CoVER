from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from evidence.schema import CalibrationChannel, EvidenceCard, ReasoningChannel

# Fields used for prototype construction — all are score-blind (no base_score/prob/logit/confidence)
SCORE_BLIND_FIELDS = [
    "degree_level",
    "feature_neighbor_discrepancy",
    "detector_signal",
    "detector_signal_strength",
    "degree_percentile_bucket",
    "neighbor_degree_skew_bucket",
    "two_hop_consistency_bucket",
    "feature_neighbor_cosine_bucket",
    "embedding_neighbor_cosine_bucket",
    "feature_embedding_disagreement_bucket",
    "bwgnn_low_band_energy_bucket",
    "bwgnn_mid_band_energy_bucket",
    "bwgnn_high_band_energy_bucket",
    "bwgnn_high_low_energy_ratio_bucket",
    "message_residual_bucket",
]


def build_prototypes(
    train_mask: Tensor,
    y: Tensor,
    reasoning_dict: dict[int, ReasoningChannel | dict],
) -> dict:
    """Build per-class prototypes from train set using only structural fields.

    FORBIDDEN inputs: test labels, base_score, prob, logit, confidence, target label.
    """
    train_indices = train_mask.nonzero(as_tuple=True)[0]

    fraud_mask = y[train_indices] == 1
    benign_mask = y[train_indices] == 0

    fraud_indices = train_indices[fraud_mask]
    benign_indices = train_indices[benign_mask]

    def _build_class_prototype(indices: Tensor) -> dict[str, str]:
        field_values: dict[str, list[str]] = {f: [] for f in SCORE_BLIND_FIELDS}
        for idx in indices:
            nid = idx.item()
            if nid not in reasoning_dict:
                continue
            r = reasoning_dict[nid]
            for field in SCORE_BLIND_FIELDS:
                val = r.get(field) if isinstance(r, dict) else getattr(r, field, None)
                if val is not None and val != "unknown":
                    field_values[field].append(val)

        prototype: dict[str, str] = {}
        for field in SCORE_BLIND_FIELDS:
            values = field_values[field]
            if values:
                prototype[field] = Counter(values).most_common(1)[0][0]
            else:
                prototype[field] = "unknown"
        return prototype

    return {
        "fraud_prototype": _build_class_prototype(fraud_indices),
        "benign_prototype": _build_class_prototype(benign_indices),
    }


def compute_prototype_similarity(
    reasoning: ReasoningChannel | dict,
    prototype: dict,
) -> tuple[int, int, list[str]]:
    """Compare a single node's reasoning fields with a prototype.

    Returns: (match_count, total_fields, matching_field_names)
    """
    match_count = 0
    total_fields = 0
    matching_field_names: list[str] = []

    for field in SCORE_BLIND_FIELDS:
        proto_val = prototype.get(field, "unknown")
        if proto_val == "unknown":
            continue

        node_val = reasoning.get(field) if isinstance(reasoning, dict) else getattr(reasoning, field, None)
        if node_val is None or node_val == "unknown":
            continue

        total_fields += 1
        if node_val == proto_val:
            match_count += 1
            matching_field_names.append(field)

    return match_count, total_fields, matching_field_names


def save_prototypes(prototypes: dict, path: str) -> None:
    """Save prototypes to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prototypes, f, indent=2)


def load_prototypes(path: str) -> dict:
    """Load prototypes from JSON."""
    with open(path) as f:
        return json.load(f)


class EvidenceAdapter:
    def __init__(self, detector_name: str, x: Tensor, edge_index: Tensor):
        self.detector_name = detector_name
        self.x = x
        self.edge_index = edge_index
        self.num_nodes = x.shape[0]

        self.degrees = self._compute_degrees()
        self._precomputed = False
        self._neighbor_emb_mean: Tensor | None = None
        self._neighbor_feat_mean: Tensor | None = None
        self._neighbor_count: Tensor | None = None

    def _compute_degrees(self) -> Tensor:
        row, _ = self.edge_index
        degree = torch.zeros(self.num_nodes, dtype=torch.long)
        ones = torch.ones(row.shape[0], dtype=torch.long)
        degree.scatter_add_(0, row, ones)
        return degree

    def _precompute_global(self, embeddings: Tensor, base_logits: Tensor) -> None:
        """Pre-compute all global statistics once (vectorized)."""
        if self._precomputed:
            return

        row, col = self.edge_index

        # Neighbor count per node
        self._neighbor_count = torch.zeros(self.num_nodes, dtype=torch.long)
        ones = torch.ones(row.shape[0], dtype=torch.long)
        self._neighbor_count.scatter_add_(0, row, ones)

        # Neighbor embedding mean
        emb_sum = torch.zeros_like(embeddings)
        emb_sum.index_add_(0, row, embeddings[col])
        count_f = self._neighbor_count.float().clamp(min=1).unsqueeze(1)
        self._neighbor_emb_mean = emb_sum / count_f

        # Neighbor feature mean
        feat_sum = torch.zeros_like(self.x)
        feat_sum.index_add_(0, row, self.x[col])
        self._neighbor_feat_mean = feat_sum / count_f

        # Degree quantiles (once)
        deg_f = self.degrees.float()
        self._deg_q33 = torch.quantile(deg_f, 0.33).item()
        self._deg_q66 = torch.quantile(deg_f, 0.66).item()

        # Neighbor logit std per node (vectorized)
        logit_sum = torch.zeros(self.num_nodes, dtype=torch.float)
        logit_sq_sum = torch.zeros(self.num_nodes, dtype=torch.float)
        neighbor_logits = base_logits[col]
        logit_sum.index_add_(0, row, neighbor_logits)
        logit_sq_sum.index_add_(0, row, neighbor_logits ** 2)
        n_count = self._neighbor_count.float().clamp(min=1)
        logit_mean = logit_sum / n_count
        logit_var = logit_sq_sum / n_count - logit_mean ** 2
        self._neighbor_logit_std = logit_var.clamp(min=0).sqrt()

        # Cosine similarity: node vs neighbor mean (vectorized)
        self._feat_cos = F.cosine_similarity(self.x, self._neighbor_feat_mean)
        self._emb_cos = F.cosine_similarity(embeddings, self._neighbor_emb_mean)

        # Embedding discrepancy (vectorized)
        self._emb_discrepancy = (embeddings - self._neighbor_emb_mean).norm(dim=1)

        # Neighbor degree stats per node (vectorized)
        neighbor_deg = self.degrees[col].float()
        nd_sum = torch.zeros(self.num_nodes, dtype=torch.float)
        nd_sq_sum = torch.zeros(self.num_nodes, dtype=torch.float)
        nd_sum.index_add_(0, row, neighbor_deg)
        nd_sq_sum.index_add_(0, row, neighbor_deg ** 2)
        self._nd_mean = nd_sum / n_count
        nd_var = nd_sq_sum / n_count - self._nd_mean ** 2
        self._nd_std = nd_var.clamp(min=0).sqrt()
        self._nd_skew = torch.where(self._nd_mean > 0, self._nd_std / self._nd_mean, torch.zeros(self.num_nodes))

        self._precomputed = True

    def extract_batch(
        self,
        node_ids: list[int],
        base_logits: Tensor,
        embeddings: Tensor,
        extras: dict[str, Tensor] | None = None,
        prototypes: dict | None = None,
    ) -> list[EvidenceCard]:
        """Vectorized batch extraction — pre-computes all global stats once."""
        base_logits = self._normalize_base_logits(base_logits)
        self._precompute_global(embeddings, base_logits)
        row, col = self.edge_index

        # Pre-compute extras quantiles once
        extras_q = {}
        if extras:
            for key, tensor in extras.items():
                try:
                    extras_q[key] = {
                        "q33": tensor.quantile(0.33).item(),
                        "q66": tensor.quantile(0.66).item(),
                    }
                except Exception:
                    pass

        cards = []
        for node_id in node_ids:
            cards.append(self._extract_from_precomputed(
                node_id, base_logits, embeddings, extras, extras_q, row, col,
                prototypes=prototypes,
            ))
        return cards

    def extract(
        self,
        node_ids: list[int],
        base_logits: Tensor,
        embeddings: Tensor,
        extras: dict[str, Tensor] | None = None,
        prototypes: dict | None = None,
    ) -> list[EvidenceCard]:
        return self.extract_batch(node_ids, base_logits, embeddings, extras, prototypes)

    def _extract_single(
        self,
        node_id: int,
        base_logits: Tensor,
        embeddings: Tensor,
        extras: dict[str, Tensor] | None = None,
        prototypes: dict | None = None,
    ) -> EvidenceCard:
        """Compatibility helper for legacy callers and tests."""
        base_logits = self._normalize_base_logits(base_logits)
        self._precompute_global(embeddings, base_logits)
        row, col = self.edge_index

        extras_q: dict[str, dict] = {}
        if extras:
            for key, tensor in extras.items():
                try:
                    extras_q[key] = {
                        "q33": tensor.quantile(0.33).item(),
                        "q66": tensor.quantile(0.66).item(),
                    }
                except Exception:
                    pass

        return self._extract_from_precomputed(
            node_id,
            base_logits,
            embeddings,
            extras,
            extras_q,
            row,
            col,
            prototypes=prototypes,
        )

    def _extract_from_precomputed(
        self,
        node_id: int,
        base_logits: Tensor,
        embeddings: Tensor,
        extras: dict[str, Tensor] | None,
        extras_q: dict[str, dict],
        row: Tensor,
        col: Tensor,
        prototypes: dict | None = None,
    ) -> EvidenceCard:
        reasoning_fields, allowed_support_ids, allowed_counter_ids = self._build_reasoning_fields(
            node_id=node_id,
            extras=extras,
            extras_q=extras_q,
            row=row,
            col=col,
        )

        calibration = CalibrationChannel(
            base_score=torch.sigmoid(base_logits[node_id]).item(),
            uncertainty=abs(base_logits[node_id].item()),
        )

        reasoning = ReasoningChannel(
            **reasoning_fields,
            allowed_support_ids=allowed_support_ids,
            allowed_counter_ids=allowed_counter_ids,
        )

        # --- Prototype-relative fields ---
        if prototypes:
            proto_fields = self._compute_prototype_relative_fields(reasoning, prototypes)
            for key, val in proto_fields.items():
                setattr(reasoning, key, val)

        return EvidenceCard(
            node_id=node_id,
            detector_name=self.detector_name,
            calibration=calibration,
            reasoning=reasoning,
        )

    @staticmethod
    def _level(degree: int) -> str:
        if degree <= 5:
            return "low"
        elif degree <= 20:
            return "medium"
        else:
            return "high"

    @staticmethod
    def _level_threshold(value: float, low: float, high: float) -> str:
        if value < low:
            return "low"
        elif value < high:
            return "medium"
        else:
            return "high"

    @staticmethod
    def _bucket_cosine(cos_val: float) -> str:
        if cos_val < 0.3:
            return "low"
        elif cos_val < 0.7:
            return "medium"
        else:
            return "high"

    @staticmethod
    def _bucket_from_precomputed(
        extras: dict[str, Tensor] | None,
        extras_q: dict[str, dict],
        key: str,
        node_id: int,
    ) -> str:
        if extras is None or key not in extras:
            return "unknown"
        try:
            val = extras[key][node_id].item()
            q = extras_q.get(key, {})
            q33 = q.get("q33", 0)
            q66 = q.get("q66", 0)
            if val <= q33:
                return "low"
            elif val <= q66:
                return "medium"
            else:
                return "high"
        except Exception:
            return "unknown"

    @staticmethod
    def _normalize_base_logits(base_logits: Tensor) -> Tensor:
        if base_logits.dim() == 2 and base_logits.shape[1] == 1:
            return base_logits[:, 0]
        return base_logits.view(-1)

    def _compute_prototype_relative_fields(
        self,
        reasoning: ReasoningChannel,
        prototypes: dict,
    ) -> dict:
        """Compute prototype-relative fields for a single node's reasoning."""
        fraud_proto = prototypes.get("fraud_prototype", {})
        benign_proto = prototypes.get("benign_prototype", {})

        fraud_match, fraud_total, fraud_fields = compute_prototype_similarity(reasoning, fraud_proto)
        benign_match, benign_total, benign_fields = compute_prototype_similarity(reasoning, benign_proto)

        fraud_ratio = fraud_match / fraud_total if fraud_total > 0 else 0.0
        benign_ratio = benign_match / benign_total if benign_total > 0 else 0.0

        closer_to_fraud = self._ratio_to_level(fraud_ratio) if fraud_total > 0 else "unknown"
        closer_to_benign = self._ratio_to_level(benign_ratio) if benign_total > 0 else "unknown"

        if fraud_total > 0 and benign_total > 0:
            diff = abs(fraud_ratio - benign_ratio)
            if diff < 0.1:
                conflict_level = "high"
            elif diff < 0.3:
                conflict_level = "medium"
            else:
                conflict_level = "low"
        else:
            conflict_level = "unknown"

        return {
            "closer_to_fraud_prototype": closer_to_fraud,
            "closer_to_benign_prototype": closer_to_benign,
            "fraud_prototype_matching_fields": fraud_fields,
            "benign_prototype_matching_fields": benign_fields,
            "prototype_conflict_level": conflict_level,
        }

    def _build_reasoning_fields(
        self,
        node_id: int,
        extras: dict[str, Tensor] | None,
        extras_q: dict[str, dict],
        row: Tensor,
        col: Tensor,
    ) -> tuple[dict[str, str], list[str], list[str]]:
        """Build the categorical reasoning fields used for cards and tokens."""
        degree = self.degrees[node_id].item()
        degree_level = self._level(degree)

        discrepancy = self._emb_discrepancy[node_id].item()
        feature_neighbor_discrepancy = self._level_threshold(discrepancy, 0.5, 2.0)

        logit_std = self._neighbor_logit_std[node_id].item()
        neighbor_consistency = self._level_threshold(1.0 - logit_std, 0.3, 0.7)

        if extras and "high_freq_response" in extras:
            hf_response = extras["high_freq_response"]
            hf_value = hf_response[node_id].item()
            q33 = extras_q.get("high_freq_response", {}).get("q33", 0)
            q66 = extras_q.get("high_freq_response", {}).get("q66", 0)
            if hf_value > q66:
                detector_signal = "high_frequency_response_high"
                detector_signal_strength = "strong"
            elif hf_value > q33:
                detector_signal = "high_frequency_response_medium"
                detector_signal_strength = "moderate"
            else:
                detector_signal = "high_frequency_response_low"
                detector_signal_strength = "weak"
        else:
            detector_signal = "embedding_neighbor_discrepancy_high" if discrepancy > 2.0 else "normal"
            detector_signal_strength = "strong" if discrepancy > 2.0 else "weak"

        counter_signal = "benign_neighbor_signal_low" if neighbor_consistency == "low" else "benign_neighbor_signal_high"

        deg_val = self.degrees[node_id].float().item()
        if deg_val <= self._deg_q33:
            degree_percentile_bucket = "low"
        elif deg_val <= self._deg_q66:
            degree_percentile_bucket = "medium"
        else:
            degree_percentile_bucket = "high"

        neighbor_mask = row == node_id
        neighbor_ids = col[neighbor_mask]

        skew = self._nd_skew[node_id].item()
        if len(neighbor_ids) > 0:
            neighbor_degree_skew_bucket = self._level_threshold(skew, 0.3, 0.7)
        else:
            neighbor_degree_skew_bucket = "unknown"

        if len(neighbor_ids) > 0:
            one_hop_set = set(neighbor_ids.tolist()) | {node_id}
            two_hop_set: set[int] = set()
            for nid in neighbor_ids.tolist():
                m = row == nid
                two_hop_set.update(col[m].tolist())
            overlap = two_hop_set & one_hop_set
            frac = len(overlap) / len(two_hop_set) if len(two_hop_set) > 0 else 0.0
            two_hop_consistency_bucket = self._level_threshold(frac, 0.2, 0.5)
        else:
            two_hop_consistency_bucket = "unknown"

        feat_cos_val = self._feat_cos[node_id].item()
        feature_neighbor_cosine_bucket = self._bucket_cosine(feat_cos_val)
        emb_cos_val = self._emb_cos[node_id].item()
        embedding_neighbor_cosine_bucket = self._bucket_cosine(emb_cos_val)
        disagreement = abs(feat_cos_val - emb_cos_val)
        feature_embedding_disagreement_bucket = self._level_threshold(disagreement, 0.1, 0.3)

        bwgnn_low = self._bucket_from_precomputed(extras, extras_q, "bwgnn_low_band", node_id)
        bwgnn_mid = self._bucket_from_precomputed(extras, extras_q, "bwgnn_mid_band", node_id)
        bwgnn_high = self._bucket_from_precomputed(extras, extras_q, "bwgnn_high_band", node_id)

        bwgnn_high_low_ratio = "unknown"
        if extras and "bwgnn_high_band" in extras and "bwgnn_low_band" in extras:
            low_val = extras["bwgnn_low_band"][node_id].item()
            high_val = extras["bwgnn_high_band"][node_id].item()
            ratio = high_val / low_val if low_val > 1e-9 else 0.0
            bwgnn_high_low_ratio = self._level_threshold(ratio, 0.5, 1.5)

        message_residual_bucket = "unknown"
        if extras and "message_residual" in extras:
            res_val = extras["message_residual"][node_id].item()
            message_residual_bucket = self._level_threshold(res_val, 0.3, 1.0)

        allowed_support_ids = [
            "degree_level", "neighbor_consistency", "feature_neighbor_discrepancy",
            "detector_signal", "detector_signal_strength",
        ] + [f"neighbor_{nid.item()}" for nid in neighbor_ids[:5]]
        allowed_counter_ids = [
            "counter_signal",
        ] + [f"counter_{nid.item()}" for nid in neighbor_ids[:3]]

        return (
            {
                "degree_level": degree_level,
                "neighbor_consistency": neighbor_consistency,
                "feature_neighbor_discrepancy": feature_neighbor_discrepancy,
                "detector_signal": detector_signal,
                "detector_signal_strength": detector_signal_strength,
                "counter_signal": counter_signal,
                "degree_percentile_bucket": degree_percentile_bucket,
                "neighbor_degree_skew_bucket": neighbor_degree_skew_bucket,
                "two_hop_consistency_bucket": two_hop_consistency_bucket,
                "feature_neighbor_cosine_bucket": feature_neighbor_cosine_bucket,
                "embedding_neighbor_cosine_bucket": embedding_neighbor_cosine_bucket,
                "feature_embedding_disagreement_bucket": feature_embedding_disagreement_bucket,
                "bwgnn_low_band_energy_bucket": bwgnn_low,
                "bwgnn_mid_band_energy_bucket": bwgnn_mid,
                "bwgnn_high_band_energy_bucket": bwgnn_high,
                "bwgnn_high_low_energy_ratio_bucket": bwgnn_high_low_ratio,
                "message_residual_bucket": message_residual_bucket,
            },
            allowed_support_ids,
            allowed_counter_ids,
        )

    def generate_graph_evidence_tokens(
        self,
        node_id: int,
        base_logits: Tensor,
        embeddings: Tensor,
        extras: dict[str, Tensor] | None,
        prototypes: dict | None,
    ) -> list[str]:
        """Generate score-blind graph evidence tokens for a node.

        Returns list of active tokens (e.g., ["HF_RATIO_HIGH", "FEAT_NEIGH_COS_BOTTOM10"]).
        All tokens are score-blind - no raw scores/probs/logits exposed.
        """
        base_logits = self._normalize_base_logits(base_logits)
        self._precompute_global(embeddings, base_logits)
        row, col = self.edge_index
        extras_q: dict[str, dict] = {}
        if extras:
            for key, tensor in extras.items():
                try:
                    extras_q[key] = {
                        "q10": tensor.quantile(0.10).item(),
                        "q33": tensor.quantile(0.33).item(),
                        "q50": tensor.quantile(0.50).item(),
                        "q66": tensor.quantile(0.66).item(),
                        "q90": tensor.quantile(0.90).item(),
                    }
                except Exception:
                    continue

        reasoning_fields, _, _ = self._build_reasoning_fields(node_id, extras, extras_q, row, col)
        tokens: list[str] = []

        if extras and "high_freq_response" in extras:
            hf = extras["high_freq_response"]
            hf_val = hf[node_id].item()
            hf_q90 = extras_q["high_freq_response"]["q90"]
            hf_q50 = extras_q["high_freq_response"]["q50"]

            if hf_val > hf_q90:
                tokens.append("HF_RATIO_TOP10")
            if hf_val > hf_q50:
                tokens.append("HF_RATIO_HIGH")
            else:
                tokens.append("HF_RATIO_LOW")

        if extras and "bwgnn_high_band" in extras and "bwgnn_low_band" in extras:
            high = extras["bwgnn_high_band"][node_id].item()
            low = extras["bwgnn_low_band"][node_id].item()
            if high > 0 and low > 0:
                ratio = high / low
                if ratio > 2.0:
                    tokens.append("BAND_ENERGY_CONFLICT_HIGH")

        if extras and "low_high_band_mismatch" in extras:
            if extras["low_high_band_mismatch"][node_id].item() > extras_q.get("low_high_band_mismatch", {}).get("q90", float("inf")):
                tokens.append("LOW_HIGH_BAND_MISMATCH")

        if extras and "normal_structure_dist" in extras:
            if extras["normal_structure_dist"][node_id].item() > extras_q.get("normal_structure_dist", {}).get("q90", float("inf")):
                tokens.append("NORMAL_STRUCTURE_DIST_HIGH")

        if extras and "pattern_deviation" in extras:
            if extras["pattern_deviation"][node_id].item() > extras_q.get("pattern_deviation", {}).get("q90", float("inf")):
                tokens.append("NORMAL_PATTERN_DEVIATION_HIGH")

        if extras and "interfering_edge_ratio" in extras:
            if extras["interfering_edge_ratio"][node_id].item() > extras_q.get("interfering_edge_ratio", {}).get("q90", float("inf")):
                tokens.append("INTERFERING_EDGE_RATIO_HIGH")

        if extras and "clean_view_shift" in extras:
            if extras["clean_view_shift"][node_id].item() > extras_q.get("clean_view_shift", {}).get("q90", float("inf")):
                tokens.append("CLEAN_VIEW_SHIFT_HIGH")

        if extras and "raw_to_clean_conflict" in extras:
            if extras["raw_to_clean_conflict"][node_id].item() > extras_q.get("raw_to_clean_conflict", {}).get("q90", float("inf")):
                tokens.append("RAW_TO_CLEAN_CONFLICT")

        if extras and "local_curvature_outlier" in extras:
            if extras["local_curvature_outlier"][node_id].item() > extras_q.get("local_curvature_outlier", {}).get("q90", float("inf")):
                tokens.append("LOCAL_CURVATURE_OUTLIER_HIGH")

        if extras and "edge_curvature_var" in extras:
            if extras["edge_curvature_var"][node_id].item() > extras_q.get("edge_curvature_var", {}).get("q90", float("inf")):
                tokens.append("EDGE_CURVATURE_VAR_HIGH")

        if reasoning_fields["feature_neighbor_cosine_bucket"] != "unknown":
            feat_cos = self._feat_cos[node_id].item()
            feat_q10 = torch.quantile(self._feat_cos, 0.10).item()
            if feat_cos < feat_q10:
                tokens.append("FEAT_NEIGH_COS_BOTTOM10")

        if reasoning_fields["embedding_neighbor_cosine_bucket"] != "unknown":
            emb_cos = self._emb_cos[node_id].item()
            emb_q10 = torch.quantile(self._emb_cos, 0.10).item()
            if emb_cos < emb_q10:
                tokens.append("EMB_NEIGH_COS_BOTTOM10")

        if reasoning_fields["feature_embedding_disagreement_bucket"] != "unknown":
            feat_cos = self._feat_cos[node_id].item()
            emb_cos = self._emb_cos[node_id].item()
            if abs(feat_cos - emb_cos) > 0.3:
                tokens.append("FEATURE_EMBED_DISAGREE_HIGH")

        if prototypes:
            reasoning_dict = reasoning_fields
            fraud_match, _, _ = compute_prototype_similarity(
                reasoning_dict, prototypes.get("fraud_prototype", {})
            )
            benign_match, _, _ = compute_prototype_similarity(
                reasoning_dict, prototypes.get("benign_prototype", {})
            )

            if fraud_match > 5:
                tokens.append("PROTO_FRAUD_CLOSE")
            if benign_match > 5:
                tokens.append("PROTO_BENIGN_CLOSE")
            if abs(fraud_match - benign_match) < 2:
                tokens.append("PROTO_CONFLICT_HIGH")

        return tokens

    @staticmethod
    def _ratio_to_level(ratio: float) -> str:
        if ratio < 0.3:
            return "low"
        elif ratio <= 0.6:
            return "medium"
        else:
            return "high"
