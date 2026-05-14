"""Stage 2 Microbenchmark: enhanced vs contrastive_directional prompt comparison.

Samples 30 nodes from YelpChi seed 123 using stratified error-aware strategy:
  - 10 train false negatives (label=1, pred<0.5)
  - 10 train false positives (label=0, pred>=0.5)
  - 5  train high-loss (top 25% BCE loss)
  - 5  val boundary (pred in [0.35, 0.65])

Compares two prompt modes and computes 10 metrics including direction error
alignment.  Reports pass/fail on 4 criteria for the contrastive_directional mode.

Usage:
    python scripts/run_stage2_microbenchmark.py --config configs/yelpchi_bwgnn.yaml --seed 123
    python scripts/run_stage2_microbenchmark.py --config configs/yelpchi_bwgnn.yaml --seed 123 --debug
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.adapter import EvidenceAdapter
from evidence.llm_teacher import OfflineLLMTeacher
from evidence.prompt import (
    build_contrastive_directional_messages,
    build_llm_messages,
    build_teacher_payload,
)
from evidence.prototypes import PrototypeBuilder
from evidence.schema import ERR
from evidence.trace_sampler import _bce_loss_per_node
from evidence.verifier import EvidenceContractVerifier, load_contracts
from models.gnn import build_detector
from utils.paths import ensure_dir, get_base_checkpoint_path


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_microbenchmark_nodes(
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    base_logits: torch.Tensor,
    seed: int,
) -> tuple[list[int], dict[str, list[int]]]:
    """Sample 30 nodes: 10 FN, 10 FP, 5 high-loss, 5 val-boundary."""
    rng = np.random.RandomState(seed)
    base_probs = torch.sigmoid(base_logits)

    train_idx = train_mask.nonzero(as_tuple=True)[0]
    val_idx = val_mask.nonzero(as_tuple=True)[0]

    y_train = y[train_idx]
    probs_train = base_probs[train_idx]

    # Pool 1: train FN (y=1, prob<0.5)
    fn_mask = (y_train == 1) & (probs_train < 0.5)
    fn_pool = train_idx[fn_mask].tolist()
    rng.shuffle(fn_pool)

    # Pool 2: train FP (y=0, prob>=0.5)
    fp_mask = (y_train == 0) & (probs_train >= 0.5)
    fp_pool = train_idx[fp_mask].tolist()
    rng.shuffle(fp_pool)

    # Pool 3: train high-loss (top 25% BCE)
    train_losses = _bce_loss_per_node(base_logits[train_idx], y[train_idx].float())
    loss_thresh = torch.quantile(train_losses, 0.75).item()
    hl_mask = train_losses >= loss_thresh
    hl_pool = train_idx[hl_mask].tolist()
    rng.shuffle(hl_pool)

    # Pool 4: val boundary (pred in [0.35, 0.65])
    val_probs = base_probs[val_idx]
    boundary_mask = (val_probs >= 0.35) & (val_probs <= 0.65)
    vb_pool = val_idx[boundary_mask].tolist()
    rng.shuffle(vb_pool)

    drawn: dict[str, list[int]] = {
        "train_fn": fn_pool[:10],
        "train_fp": fp_pool[:10],
        "train_high_loss": hl_pool[:5],
        "val_boundary": vb_pool[:5],
    }

    all_nodes: list[int] = []
    for nodes in drawn.values():
        all_nodes.extend(nodes)

    return all_nodes, drawn


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _entropy(values: list[str]) -> float:
    """Shannon entropy of a categorical distribution."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def _node_category(node_id: int, drawn: dict[str, list[int]]) -> str:
    """Return which sampling pool a node belongs to."""
    for cat, nodes in drawn.items():
        if node_id in nodes:
            return cat
    return "unknown"


def analyze_payload_evidence(payload: dict) -> dict:
    """Analyze a teacher payload for fraud/benign evidence availability."""
    reasoning = payload.get("reasoning", {})
    closer_to_fraud = reasoning.get("closer_to_fraud_prototype", "unknown")
    closer_to_benign = reasoning.get("closer_to_benign_prototype", "unknown")
    prototype_conflict = reasoning.get("prototype_conflict_level", "unknown")
    fraud_matching = reasoning.get("fraud_prototype_matching_fields", [])
    benign_matching = reasoning.get("benign_prototype_matching_fields", [])

    fraud_like_tokens = []
    benign_like_tokens = []

    degree = reasoning.get("degree_level", "unknown")
    if degree in ("high", "very_high"):
        fraud_like_tokens.append("degree_level=high")
    elif degree in ("low", "very_low"):
        benign_like_tokens.append("degree_level=low")

    neighbor = reasoning.get("neighbor_consistency", "unknown")
    if neighbor in ("low", "very_low"):
        fraud_like_tokens.append("neighbor_consistency=low")
    elif neighbor in ("high", "very_high"):
        benign_like_tokens.append("neighbor_consistency=high")

    feat_disc = reasoning.get("feature_neighbor_discrepancy", "unknown")
    if feat_disc in ("high", "very_high"):
        fraud_like_tokens.append("feature_neighbor_discrepancy=high")
    elif feat_disc in ("low", "very_low"):
        benign_like_tokens.append("feature_neighbor_discrepancy=low")

    detector = reasoning.get("detector_signal", "unknown")
    if "high" in detector.lower():
        fraud_like_tokens.append("detector_signal=high")
    elif "low" in detector.lower() or "normal" in detector.lower():
        benign_like_tokens.append("detector_signal=low/normal")

    if fraud_matching:
        fraud_like_tokens.extend([f"fraud_match={f}" for f in fraud_matching[:3]])
    if benign_matching:
        benign_like_tokens.extend([f"benign_match={f}" for f in benign_matching[:3]])

    if closer_to_fraud == "high":
        fraud_like_tokens.append("closer_to_fraud=high")
    if closer_to_benign == "high":
        benign_like_tokens.append("closer_to_benign=high")

    n_fraud = len(fraud_like_tokens)
    n_benign = len(benign_like_tokens)

    if n_fraud > 0 and n_benign == 0:
        payload_type = "fraud_dominant"
    elif n_benign > 0 and n_fraud == 0:
        payload_type = "benign_dominant"
    elif n_fraud > 0 and n_benign > 0:
        payload_type = "mixed"
    else:
        payload_type = "neutral"

    return {
        "has_fraud_signal": n_fraud > 0,
        "has_benign_signal": n_benign > 0,
        "has_prototype_conflict": prototype_conflict in ("medium", "high"),
        "closer_to_fraud_high": closer_to_fraud == "high",
        "closer_to_benign_high": closer_to_benign == "high",
        "fraud_matching_nonempty": len(fraud_matching) > 0,
        "benign_matching_nonempty": len(benign_matching) > 0,
        "n_fraud_tokens": n_fraud,
        "n_benign_tokens": n_benign,
        "fraud_benign_balance": n_fraud - n_benign,
        "payload_type": payload_type,
        "fraud_like_tokens": fraud_like_tokens,
        "benign_like_tokens": benign_like_tokens,
    }


def compute_payload_diagnostics(payloads: list[dict]) -> dict:
    """Compute aggregate payload evidence diagnostics."""
    analyses = [analyze_payload_evidence(p) for p in payloads]
    n = len(analyses)
    if n == 0:
        return {}

    benign_available = sum(1 for a in analyses if a["has_benign_signal"])
    fraud_available = sum(1 for a in analyses if a["has_fraud_signal"])
    conflict = sum(1 for a in analyses if a["has_prototype_conflict"])
    benign_dominant = sum(1 for a in analyses if a["payload_type"] == "benign_dominant")
    fraud_dominant = sum(1 for a in analyses if a["payload_type"] == "fraud_dominant")
    mixed = sum(1 for a in analyses if a["payload_type"] == "mixed")
    neutral = sum(1 for a in analyses if a["payload_type"] == "neutral")

    return {
        "benign_signal_available_rate": benign_available / n,
        "fraud_signal_available_rate": fraud_available / n,
        "prototype_conflict_rate": conflict / n,
        "benign_dominant_payload_count": benign_dominant,
        "fraud_dominant_payload_count": fraud_dominant,
        "mixed_payload_count": mixed,
        "neutral_payload_count": neutral,
        "payload_types": [a["payload_type"] for a in analyses],
    }


def compute_metrics(
    errs: list[ERR | None],
    metadata_list: list[dict],
    cards: list,
    drawn: dict[str, list[int]],
    mode: str,
    payload_types: list[str] | None = None,
) -> dict:
    """Compute the 10 metrics for one prompt mode."""
    n = len(errs)
    parsed_ok = [m.get("parsed_ok", False) for m in metadata_list]
    parse_success_rate = sum(parsed_ok) / n if n else 0.0

    # Verifier acceptance (from metadata final_status)
    accepted_count = sum(
        1 for m in metadata_list
        if m.get("final_status") in ("accepted", "accepted_after_retry")
    )
    verifier_acceptance_rate = accepted_count / n if n else 0.0

    valid_errs = [e for e in errs if e is not None]

    # Risk type distribution
    risk_types = [e.risk_type for e in valid_errs]
    risk_dist = dict(Counter(risk_types))

    # Evidence direction distribution (directional mode)
    direction_dist: dict[str, int] = {}
    if mode == "contrastive_directional":
        directions = [getattr(e, "evidence_direction", "unknown") for e in valid_errs]
        direction_dist = dict(Counter(directions))

    # Evidence strength distribution (directional mode)
    strength_dist: dict[str, int] = {}
    if mode == "contrastive_directional":
        strengths = [getattr(e, "evidence_strength", "unknown") for e in valid_errs]
        strength_dist = dict(Counter(strengths))

    # Supporting / counter evidence entropy
    all_supporting: list[str] = []
    all_counter: list[str] = []
    for e in valid_errs:
        all_supporting.extend(e.supporting_evidence)
        all_counter.extend(e.counter_evidence)
    supporting_entropy = _entropy(all_supporting)
    counter_entropy = _entropy(all_counter)

    # Uncertainty factors distribution
    uncertainty_all: list[str] = []
    for e in valid_errs:
        uf = getattr(e, "uncertainty_factors", [])
        uncertainty_all.extend(uf)
    uncertainty_dist = dict(Counter(uncertainty_all))

    # Structural discrepancy ratio
    struct_disc_count = sum(1 for rt in risk_types if rt == "structural_discrepancy")
    structural_discrepancy_ratio = struct_disc_count / len(risk_types) if risk_types else 0.0

    # Direction error alignment (offline audit)
    # FN nodes: increase_risk is aligned; FP nodes: decrease_risk is aligned
    alignment_total = 0
    alignment_correct = 0
    if mode == "contrastive_directional":
        for e in valid_errs:
            cat = _node_category(e.node_id, drawn)
            direction = getattr(e, "evidence_direction", "uncertain")
            if cat == "train_fn":
                alignment_total += 1
                if direction == "increase_risk":
                    alignment_correct += 1
            elif cat == "train_fp":
                alignment_total += 1
                if direction == "decrease_risk":
                    alignment_correct += 1
    direction_error_alignment = alignment_correct / alignment_total if alignment_total > 0 else 0.0

    # Conditioned direction alignment (payload-aware)
    conditioned_alignment_total = 0
    conditioned_alignment_correct = 0
    benign_direction_recall_total = 0
    benign_direction_recall_correct = 0
    fraud_direction_recall_total = 0
    fraud_direction_recall_correct = 0
    uncertain_on_mixed_total = 0
    uncertain_on_mixed_correct = 0

    if mode == "contrastive_directional" and payload_types:
        for e, ptype in zip(valid_errs, payload_types[:len(valid_errs)]):
            cat = _node_category(e.node_id, drawn)
            direction = getattr(e, "evidence_direction", "uncertain")

            # FN with fraud-dominant payload → expect increase_risk
            if cat == "train_fn" and ptype == "fraud_dominant":
                conditioned_alignment_total += 1
                if direction == "increase_risk":
                    conditioned_alignment_correct += 1

            # FP with benign-dominant payload → expect decrease_risk
            if cat == "train_fp" and ptype == "benign_dominant":
                conditioned_alignment_total += 1
                if direction == "decrease_risk":
                    conditioned_alignment_correct += 1

            # Benign-dominant payload → expect decrease_risk
            if ptype == "benign_dominant":
                benign_direction_recall_total += 1
                if direction == "decrease_risk":
                    benign_direction_recall_correct += 1

            # Fraud-dominant payload → expect increase_risk
            if ptype == "fraud_dominant":
                fraud_direction_recall_total += 1
                if direction == "increase_risk":
                    fraud_direction_recall_correct += 1

            # Mixed payload → expect uncertain
            if ptype == "mixed":
                uncertain_on_mixed_total += 1
                if direction == "uncertain":
                    uncertain_on_mixed_correct += 1

    conditioned_direction_alignment = (
        conditioned_alignment_correct / conditioned_alignment_total
        if conditioned_alignment_total > 0 else 0.0
    )
    benign_direction_recall = (
        benign_direction_recall_correct / benign_direction_recall_total
        if benign_direction_recall_total > 0 else 0.0
    )
    fraud_direction_recall = (
        fraud_direction_recall_correct / fraud_direction_recall_total
        if fraud_direction_recall_total > 0 else 0.0
    )
    uncertain_on_mixed_rate = (
        uncertain_on_mixed_correct / uncertain_on_mixed_total
        if uncertain_on_mixed_total > 0 else 0.0
    )

    return {
        "mode": mode,
        "n_total": n,
        "parse_success_rate": parse_success_rate,
        "verifier_acceptance_rate": verifier_acceptance_rate,
        "risk_type_distribution": risk_dist,
        "direction_distribution": direction_dist,
        "strength_distribution": strength_dist,
        "supporting_entropy": supporting_entropy,
        "counter_entropy": counter_entropy,
        "uncertainty_factors_distribution": uncertainty_dist,
        "structural_discrepancy_ratio": structural_discrepancy_ratio,
        "direction_error_alignment": direction_error_alignment,
        "direction_alignment_n": alignment_total,
        "conditioned_direction_alignment": conditioned_direction_alignment,
        "conditioned_alignment_n": conditioned_alignment_total,
        "benign_direction_recall": benign_direction_recall,
        "benign_direction_recall_n": benign_direction_recall_total,
        "fraud_direction_recall": fraud_direction_recall,
        "fraud_direction_recall_n": fraud_direction_recall_total,
        "uncertain_on_mixed_rate": uncertain_on_mixed_rate,
        "uncertain_on_mixed_n": uncertain_on_mixed_total,
    }


def check_pass_criteria(metrics: dict) -> dict[str, bool]:
    """Check the 4 pass criteria for contrastive_directional mode."""
    direction_dist = metrics.get("direction_distribution", {})
    has_increase = "increase_risk" in direction_dist and direction_dist["increase_risk"] > 0
    has_decrease = "decrease_risk" in direction_dist and direction_dist["decrease_risk"] > 0

    return {
        "structural_discrepancy_ratio_lt_80": metrics["structural_discrepancy_ratio"] < 0.80,
        "both_directions_present": has_increase and has_decrease,
        "direction_error_alignment_gte_55": metrics["direction_error_alignment"] >= 0.55,
        "verifier_acceptance_gte_80": metrics["verifier_acceptance_rate"] >= 0.80,
    }


# ---------------------------------------------------------------------------
# Run one mode
# ---------------------------------------------------------------------------

def run_mode(
    mode: str,
    payloads: list[dict],
    cards: list,
    verifier: EvidenceContractVerifier,
    llm_config: dict,
    teacher_type: str = "mock",
) -> tuple[list[ERR | None], list[dict]]:
    """Run one prompt mode and return (errs, metadata_list)."""
    if teacher_type == "qwen":
        teacher = OfflineLLMTeacher(
            backend="transformers_local",
            model_name_or_path=llm_config.get("model_name_or_path"),
            temperature=llm_config.get("temperature", 0.0),
            max_retries=llm_config.get("max_parse_retries", 1),
            max_new_tokens=llm_config.get("max_new_tokens", 256),
            device_map=llm_config.get("device_map", "auto"),
            torch_dtype=llm_config.get("torch_dtype", "auto"),
            trust_remote_code=llm_config.get("trust_remote_code", True),
            enable_verifier_retry=llm_config.get("enable_verifier_retry", True),
            max_verifier_retries=llm_config.get("max_verifier_retries", 1),
            max_parse_retries=llm_config.get("max_parse_retries", 1),
            prompt_mode=mode,
        )
    else:
        teacher = OfflineLLMTeacher(
            backend="mock",
            temperature=0.0,
            max_retries=1,
            max_new_tokens=256,
            enable_verifier_retry=True,
            max_verifier_retries=1,
            prompt_mode=mode,
        )

    errs: list[ERR | None] = []
    metadata_list: list[dict] = []

    for payload, card in zip(payloads, cards):
        err, meta = teacher.generate(payload)
        if err is not None:
            accepted, reasons = verifier.verify(err, card)
            if accepted:
                meta["final_status"] = "accepted"
            else:
                meta["final_status"] = "rejected"
                meta["reject_reasons"] = reasons
        else:
            meta["final_status"] = "parse_failed"
        errs.append(err)
        metadata_list.append(meta)

    return errs, metadata_list


# ---------------------------------------------------------------------------
# Prompt logging
# ---------------------------------------------------------------------------

def log_prompts(
    payloads: list[dict],
    mode: str,
    prompts_path: Path,
    outputs_path: Path,
    errs: list[ERR | None],
    metadata_list: list[dict],
) -> None:
    """Write prompts and outputs to JSONL files."""
    msgs_fn = (
        build_contrastive_directional_messages
        if mode == "contrastive_directional"
        else build_llm_messages
    )

    with open(prompts_path, "w") as fp, open(outputs_path, "w") as fo:
        for payload, err, meta in zip(payloads, errs, metadata_list):
            messages = msgs_fn(payload)
            fp.write(json.dumps({
                "node_id": payload.get("node_id"),
                "mode": mode,
                "messages": messages,
            }, ensure_ascii=False) + "\n")

            out_record: dict = {
                "node_id": payload.get("node_id"),
                "mode": mode,
                "parsed_ok": meta.get("parsed_ok", False),
                "final_status": meta.get("final_status"),
                "raw_output": meta.get("raw_output"),
            }
            if err is not None:
                out_record["err"] = {
                    "risk_type": err.risk_type,
                    "supporting_evidence": err.supporting_evidence,
                    "counter_evidence": err.counter_evidence,
                    "evidence_direction": getattr(err, "evidence_direction", ""),
                    "evidence_strength": getattr(err, "evidence_strength", ""),
                    "uncertainty_factors": getattr(err, "uncertainty_factors", []),
                    "summary": err.summary,
                }
            fo.write(json.dumps(out_record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_markdown_report(
    current_metrics: dict,
    directional_metrics: dict,
    pass_criteria: dict[str, bool],
    drawn: dict[str, list[int]],
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    seed: int,
    elapsed: float,
    teacher_type: str = "mock",
    payload_diag: dict | None = None,
) -> Path:
    """Write the markdown report."""
    report_path = output_dir / "artifacts" / "reports" / "stage2_directional_microbenchmark.md"
    ensure_dir(report_path.parent)

    all_pass = all(pass_criteria.values())

    lines = [
        f"# Stage 2 Directional Microbenchmark",
        f"",
        f"**Dataset:** {dataset_name}  ",
        f"**Model:** {model_name}  ",
        f"**Seed:** {seed}  ",
        f"**Teacher:** {teacher_type}  ",
        f"**Elapsed:** {elapsed:.2f}s  ",
        f"",
        f"## Sampling Summary",
        f"",
    ]
    for cat, nodes in drawn.items():
        lines.append(f"- **{cat}**: {len(nodes)} nodes")

    if payload_diag:
        lines += [
            f"",
            f"## Payload Evidence Diagnostics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Benign signal available rate | {payload_diag.get('benign_signal_available_rate', 0):.2%} |",
            f"| Fraud signal available rate | {payload_diag.get('fraud_signal_available_rate', 0):.2%} |",
            f"| Prototype conflict rate | {payload_diag.get('prototype_conflict_rate', 0):.2%} |",
            f"| Benign dominant payloads | {payload_diag.get('benign_dominant_payload_count', 0)} |",
            f"| Fraud dominant payloads | {payload_diag.get('fraud_dominant_payload_count', 0)} |",
            f"| Mixed payloads | {payload_diag.get('mixed_payload_count', 0)} |",
            f"| Neutral payloads | {payload_diag.get('neutral_payload_count', 0)} |",
        ]

    lines += [
        f"",
        f"## Metrics Comparison",
        f"",
        f"| Metric | enhanced | contrastive_directional |",
        f"|--------|----------|------------------------|",
        f"| Parse success rate | {current_metrics['parse_success_rate']:.2%} | {directional_metrics['parse_success_rate']:.2%} |",
        f"| Verifier acceptance | {current_metrics['verifier_acceptance_rate']:.2%} | {directional_metrics['verifier_acceptance_rate']:.2%} |",
        f"| Supporting entropy | {current_metrics['supporting_entropy']:.3f} | {directional_metrics['supporting_entropy']:.3f} |",
        f"| Counter entropy | {current_metrics['counter_entropy']:.3f} | {directional_metrics['counter_entropy']:.3f} |",
        f"| Structural discrepancy ratio | {current_metrics['structural_discrepancy_ratio']:.2%} | {directional_metrics['structural_discrepancy_ratio']:.2%} |",
        f"| Direction error alignment | N/A | {directional_metrics['direction_error_alignment']:.2%} (n={directional_metrics['direction_alignment_n']}) |",
        f"| Conditioned direction alignment | N/A | {directional_metrics['conditioned_direction_alignment']:.2%} (n={directional_metrics['conditioned_alignment_n']}) |",
        f"| Benign direction recall | N/A | {directional_metrics['benign_direction_recall']:.2%} (n={directional_metrics['benign_direction_recall_n']}) |",
        f"| Fraud direction recall | N/A | {directional_metrics['fraud_direction_recall']:.2%} (n={directional_metrics['fraud_direction_recall_n']}) |",
        f"| Uncertain on mixed rate | N/A | {directional_metrics['uncertain_on_mixed_rate']:.2%} (n={directional_metrics['uncertain_on_mixed_n']}) |",
        f"",
        f"### Risk Type Distribution",
        f"",
        f"| risk_type | enhanced | contrastive_directional |",
        f"|-----------|----------|------------------------|",
    ]
    all_risk_types = sorted(
        set(current_metrics["risk_type_distribution"].keys())
        | set(directional_metrics["risk_type_distribution"].keys())
    )
    for rt in all_risk_types:
        c_count = current_metrics["risk_type_distribution"].get(rt, 0)
        d_count = directional_metrics["risk_type_distribution"].get(rt, 0)
        lines.append(f"| {rt} | {c_count} | {d_count} |")

    if directional_metrics["direction_distribution"]:
        lines += [
            f"",
            f"### Direction Distribution (contrastive_directional)",
            f"",
        ]
        for d, cnt in sorted(directional_metrics["direction_distribution"].items()):
            lines.append(f"- **{d}**: {cnt}")

    if directional_metrics["strength_distribution"]:
        lines += [
            f"",
            f"### Strength Distribution (contrastive_directional)",
            f"",
        ]
        for s, cnt in sorted(directional_metrics["strength_distribution"].items()):
            lines.append(f"- **{s}**: {cnt}")

    if directional_metrics["uncertainty_factors_distribution"]:
        lines += [
            f"",
            f"### Uncertainty Factors Distribution (contrastive_directional)",
            f"",
        ]
        for uf, cnt in sorted(directional_metrics["uncertainty_factors_distribution"].items()):
            lines.append(f"- **{uf}**: {cnt}")

    lines += [
        f"",
        f"## Pass Criteria",
        f"",
        f"| Criterion | Value | Threshold | Pass |",
        f"|-----------|-------|-----------|------|",
        f"| structural_discrepancy ratio | {directional_metrics['structural_discrepancy_ratio']:.2%} | < 80% | {'PASS' if pass_criteria['structural_discrepancy_ratio_lt_80'] else 'FAIL'} |",
        f"| both increase_risk and decrease_risk present | {bool(directional_metrics['direction_distribution'])} | both > 0 | {'PASS' if pass_criteria['both_directions_present'] else 'FAIL'} |",
        f"| direction error alignment | {directional_metrics['direction_error_alignment']:.2%} | >= 55% | {'PASS' if pass_criteria['direction_error_alignment_gte_55'] else 'FAIL'} |",
        f"| verifier acceptance rate | {directional_metrics['verifier_acceptance_rate']:.2%} | >= 80% | {'PASS' if pass_criteria['verifier_acceptance_gte_80'] else 'FAIL'} |",
        f"",
        f"**Overall: {'ALL PASS' if all_pass else 'NOT ALL PASS'}**",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_csv(
    current_metrics: dict,
    directional_metrics: dict,
    csv_path: Path,
) -> None:
    """Write comparison CSV."""
    ensure_dir(csv_path.parent)
    rows = [
        {"metric": "parse_success_rate",
         "enhanced": current_metrics["parse_success_rate"],
         "contrastive_directional": directional_metrics["parse_success_rate"]},
        {"metric": "verifier_acceptance_rate",
         "enhanced": current_metrics["verifier_acceptance_rate"],
         "contrastive_directional": directional_metrics["verifier_acceptance_rate"]},
        {"metric": "supporting_entropy",
         "enhanced": current_metrics["supporting_entropy"],
         "contrastive_directional": directional_metrics["supporting_entropy"]},
        {"metric": "counter_entropy",
         "enhanced": current_metrics["counter_entropy"],
         "contrastive_directional": directional_metrics["counter_entropy"]},
        {"metric": "structural_discrepancy_ratio",
         "enhanced": current_metrics["structural_discrepancy_ratio"],
         "contrastive_directional": directional_metrics["structural_discrepancy_ratio"]},
        {"metric": "direction_error_alignment",
         "enhanced": "N/A",
         "contrastive_directional": directional_metrics["direction_error_alignment"]},
    ]

    all_risk_types = sorted(
        set(current_metrics["risk_type_distribution"].keys())
        | set(directional_metrics["risk_type_distribution"].keys())
    )
    for rt in all_risk_types:
        rows.append({
            "metric": f"risk_type_{rt}",
            "enhanced": current_metrics["risk_type_distribution"].get(rt, 0),
            "contrastive_directional": directional_metrics["risk_type_distribution"].get(rt, 0),
        })

    for d in ("increase_risk", "decrease_risk", "uncertain"):
        rows.append({
            "metric": f"direction_{d}",
            "enhanced": "N/A",
            "contrastive_directional": directional_metrics["direction_distribution"].get(d, 0),
        })

    for s in ("weak", "moderate", "strong"):
        rows.append({
            "metric": f"strength_{s}",
            "enhanced": "N/A",
            "contrastive_directional": directional_metrics["strength_distribution"].get(s, 0),
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "enhanced", "contrastive_directional"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2 microbenchmark: enhanced vs contrastive_directional")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--teacher", type=str, default="mock", choices=["mock", "qwen"],
                        help="Teacher backend: mock (fast) or qwen (real LLM)")
    parser.add_argument("--debug", action="store_true", help="Use tiny synthetic graph for quick test")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    model_name = config["model"]["name"]
    seed = args.seed
    llm_config = config.get("llm", {})

    torch.manual_seed(seed)

    # ---- Load data ----
    if args.debug:
        print("[DEBUG] Using tiny synthetic graph")
        data = load_fraud_dataset("tiny", seed=seed)
    else:
        split_mode = config["dataset"].get("split_mode", "supervised")
        train_ratio = config["dataset"].get("train_ratio", 0.7)
        val_test_ratio = config["dataset"].get("val_test_ratio", [1, 2])
        data = load_fraud_dataset(
            dataset_name, path=dataset_path, seed=seed,
            split_mode=split_mode, train_ratio=train_ratio, val_test_ratio=val_test_ratio,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    ).to(device)

    checkpoint_path = get_base_checkpoint_path(dataset_name, model_name, seed)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True, map_location=device)
        try:
            model.load_state_dict(state)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except RuntimeError:
            print(f"Warning: Checkpoint shape mismatch (debug mode), using random weights")
    else:
        print(f"Warning: No checkpoint at {checkpoint_path}, using random weights")

    model.eval()

    with torch.no_grad():
        output = model(data.x.to(device), data.edge_index.to(device), return_output=True)
        base_logits = output.logits.cpu()
        embeddings = output.embeddings.cpu()
        extras = {}
        if output.extras:
            for k, v in output.extras.items():
                extras[k] = v.cpu() if isinstance(v, torch.Tensor) else v

    # ---- Sample nodes ----
    val_mask = getattr(data, "val_mask", None)
    if val_mask is None:
        val_mask = torch.zeros(data.y.shape[0], dtype=torch.bool)

    nodes, drawn = sample_microbenchmark_nodes(
        y=data.y,
        train_mask=data.train_mask,
        val_mask=val_mask,
        base_logits=base_logits,
        seed=seed,
    )
    print(f"Sampled {len(nodes)} nodes: " + ", ".join(f"{k}={len(v)}" for k, v in drawn.items()))

    # ---- Build evidence cards + payloads ----
    adapter = EvidenceAdapter(
        detector_name=model_name,
        x=data.x,
        edge_index=data.edge_index,
    )

    # Extract evidence tokens for ALL train nodes using vectorized method
    train_node_ids = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    
    # Check for cached tokens
    cache_dir = Path("artifacts/token_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset_name}_{model_name}_seed{seed}_train_tokens.json"
    
    if cache_file.exists():
        print(f"Loading cached train tokens from {cache_file}...")
        with open(cache_file) as f:
            train_tokens_dict = json.load(f)
        train_tokens_dict = {int(k): v for k, v in train_tokens_dict.items()}
    else:
        print(f"Extracting evidence tokens for {len(train_node_ids)} train nodes (vectorized)...")
        start_tok = time.time()
        
        # Use fully vectorized method - no per-node loop
        train_tokens_dict = adapter.generate_graph_evidence_tokens_vectorized(
            node_ids=train_node_ids,
            base_logits=base_logits,
            embeddings=embeddings,
            extras=extras if extras else None,
            prototypes=None,
        )
        
        elapsed_tok = time.time() - start_tok
        print(f"Token extraction completed in {elapsed_tok:.1f}s ({len(train_node_ids)/elapsed_tok:.0f} nodes/s)")
        
        # Cache the tokens
        with open(cache_file, "w") as f:
            json.dump({str(k): v for k, v in train_tokens_dict.items()}, f)
        print(f"Cached train tokens to {cache_file}")
    
    # Convert to format expected by PrototypeBuilder
    evidence_tokens = {}
    for node_id, token_list in train_tokens_dict.items():
        # Convert token list to dict format (token_name -> "active")
        evidence_tokens[node_id] = {t: "active" for t in token_list}
    
    # Build reasoning dict for prototypes (needed for field_modes)
    train_cards = adapter.extract_batch(
        node_ids=train_node_ids,
        base_logits=base_logits,
        embeddings=embeddings,
        extras=extras if extras else None,
    )
    
    base_preds = (torch.sigmoid(base_logits) >= 0.5).long()
    
    # Use new PrototypeBuilder with distinctive tokens
    builder = PrototypeBuilder()
    prototypes = builder.build(
        train_mask=data.train_mask,
        y=data.y,
        evidence_tokens=evidence_tokens,
        base_preds=base_preds,
    )

    # Now extract cards for sampled nodes with prototypes
    old_prototypes = {
        "fraud_prototype": prototypes.get("fraud_prototype_summary", {}).get("field_modes", {}),
        "benign_prototype": prototypes.get("benign_prototype_summary", {}).get("field_modes", {}),
    }
    cards = adapter.extract_batch(
        node_ids=nodes,
        base_logits=base_logits,
        embeddings=embeddings,
        extras=extras if extras else None,
        prototypes=old_prototypes,
    )

    payloads = []
    for card in cards:
        payload = card.to_teacher_payload()
        # Inject prototype field modes for contrastive_directional prompt
        fraud_summary = prototypes.get("fraud_prototype_summary", {})
        benign_summary = prototypes.get("benign_prototype_summary", {})
        payload["fraud_prototype"] = fraud_summary.get("field_modes", {})
        payload["benign_prototype"] = benign_summary.get("field_modes", {})
        payloads.append(payload)

    # ---- Load verifier ----
    contracts = load_contracts()
    verifier = EvidenceContractVerifier(contracts, enable_label_compatibility=False)

    start_time = time.time()

    # ---- Run both modes ----
    teacher_type = args.teacher
    print(f"Running enhanced mode (teacher={teacher_type})...")
    current_errs, current_meta = run_mode("enhanced", payloads, cards, verifier, llm_config, teacher_type)

    print(f"Running contrastive_directional mode (teacher={teacher_type})...")
    dir_errs, dir_meta = run_mode("contrastive_directional", payloads, cards, verifier, llm_config, teacher_type)

    elapsed = time.time() - start_time

    # ---- Compute payload diagnostics ----
    payload_diag = compute_payload_diagnostics(payloads)
    payload_types = payload_diag.get("payload_types", [])

    # ---- Compute metrics ----
    current_metrics = compute_metrics(current_errs, current_meta, cards, drawn, "enhanced")
    dir_metrics = compute_metrics(dir_errs, dir_meta, cards, drawn, "contrastive_directional", payload_types)

    pass_criteria = check_pass_criteria(dir_metrics)

    # ---- Output dir ----
    output_dir = Path(".")
    reports_dir = output_dir / "artifacts" / "reports"
    tables_dir = output_dir / "artifacts" / "tables"
    ensure_dir(reports_dir)
    ensure_dir(tables_dir)

    # ---- Write reports ----
    report_path = write_markdown_report(
        current_metrics, dir_metrics, pass_criteria, drawn,
        output_dir, dataset_name, model_name, seed, elapsed, teacher_type,
        payload_diag,
    )
    csv_path = tables_dir / "stage2_directional_microbenchmark.csv"
    write_csv(current_metrics, dir_metrics, csv_path)

    # ---- Write prompt / output JSONL ----
    log_prompts(payloads, "enhanced",
                output_dir / "current_prompts.jsonl",
                output_dir / "current_outputs.jsonl",
                current_errs, current_meta)
    log_prompts(payloads, "contrastive_directional",
                output_dir / "directional_prompts.jsonl",
                output_dir / "directional_outputs.jsonl",
                dir_errs, dir_meta)

    # ---- Print summary ----
    all_pass = all(pass_criteria.values())

    print(f"\n{'='*60}")
    print(f"Stage 2 Directional Microbenchmark Results")
    print(f"{'='*60}")
    print(f"  Dataset: {dataset_name}, Model: {model_name}, Seed: {seed}, Teacher: {teacher_type}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"")
    print(f"  Payload Diagnostics:")
    print(f"    Benign signal available: {payload_diag.get('benign_signal_available_rate', 0):.2%}")
    print(f"    Fraud signal available: {payload_diag.get('fraud_signal_available_rate', 0):.2%}")
    print(f"    Benign dominant: {payload_diag.get('benign_dominant_payload_count', 0)}")
    print(f"    Fraud dominant: {payload_diag.get('fraud_dominant_payload_count', 0)}")
    print(f"    Mixed: {payload_diag.get('mixed_payload_count', 0)}")
    print(f"    Neutral: {payload_diag.get('neutral_payload_count', 0)}")
    print(f"")
    print(f"  Enhanced mode:")
    print(f"    Parse success: {current_metrics['parse_success_rate']:.2%}")
    print(f"    Verifier accept: {current_metrics['verifier_acceptance_rate']:.2%}")
    print(f"    Struct discrepancy: {current_metrics['structural_discrepancy_ratio']:.2%}")
    print(f"")
    print(f"  Contrastive directional mode:")
    print(f"    Parse success: {dir_metrics['parse_success_rate']:.2%}")
    print(f"    Verifier accept: {dir_metrics['verifier_acceptance_rate']:.2%}")
    print(f"    Struct discrepancy: {dir_metrics['structural_discrepancy_ratio']:.2%}")
    print(f"    Direction alignment: {dir_metrics['direction_error_alignment']:.2%} (n={dir_metrics['direction_alignment_n']})")
    print(f"    Conditioned alignment: {dir_metrics['conditioned_direction_alignment']:.2%} (n={dir_metrics['conditioned_alignment_n']})")
    print(f"    Benign direction recall: {dir_metrics['benign_direction_recall']:.2%} (n={dir_metrics['benign_direction_recall_n']})")
    print(f"    Fraud direction recall: {dir_metrics['fraud_direction_recall']:.2%} (n={dir_metrics['fraud_direction_recall_n']})")
    print(f"    Uncertain on mixed: {dir_metrics['uncertain_on_mixed_rate']:.2%} (n={dir_metrics['uncertain_on_mixed_n']})")
    print(f"")
    print(f"  Pass Criteria:")
    for name, ok in pass_criteria.items():
        print(f"    {name}: {'PASS' if ok else 'FAIL'}")
    print(f"")
    print(f"  Overall: {'ALL PASS' if all_pass else 'NOT ALL PASS'}")
    print(f"")
    print(f"  Files:")
    print(f"    Report: {report_path}")
    print(f"    CSV:    {csv_path}")
    print(f"    Prompts: current_prompts.jsonl, directional_prompts.jsonl")
    print(f"    Outputs: current_outputs.jsonl, directional_outputs.jsonl")


if __name__ == "__main__":
    main()
