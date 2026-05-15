from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.relation_features import load_relation_stats
from evidence.schema import ERR
from evidence.vocab import encode_direction_target, encode_err_targets, encode_reasoning, get_evidence_slots, get_reason_types
from models.gnn import build_detector
from models.reasoner import EvidenceReasoner, VALID_GATE_MODES, VALID_RELATION_FUSION_MODES
from training.losses import compute_cover_judge_loss, compute_cover_lift_loss, compute_cvscd_loss
from training.metrics import compute_metrics
from utils.tensorboard import create_logger
from utils.paths import get_checkpoint_dir, get_logs_dir, get_results_dir, get_err_cache_dir, get_base_checkpoint_path, ensure_dir


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def runtime_device_info(device: torch.device) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    info: dict[str, object] = {
        "requested_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": visible,
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        return info

    visible_index = device.index
    if visible_index is None:
        visible_index = torch.cuda.current_device()
    physical_ids = [item.strip() for item in visible.split(",")] if visible else []
    physical_id = physical_ids[visible_index] if visible_index < len(physical_ids) else str(visible_index)
    info.update({
        "visible_device_index": visible_index,
        "physical_device_id": physical_id,
        "device_name": torch.cuda.get_device_name(visible_index),
    })
    return info


def load_evidence_cards(path: Path) -> dict[int, dict]:
    cards = {}
    with open(path) as f:
        for line in f:
            card = json.loads(line)
            cards[card["node_id"]] = card
    return cards


def load_accepted_errs(path: Path) -> list[dict]:
    errs = []
    with open(path) as f:
        for line in f:
            errs.append(json.loads(line))
    return errs


def _err_preference_key(err: dict) -> tuple[int, int]:
    direction = err.get("evidence_direction", "uncertain")
    strength = err.get("evidence_strength", "weak")
    direction_score = 0 if direction == "uncertain" else 1
    strength_score = {"weak": 0, "moderate": 1, "strong": 2}.get(strength, 0)
    return direction_score, strength_score


def _strength_id(value: str) -> int:
    return {"weak": 0, "moderate": 1, "strong": 2}.get(value, -1)


def _polarity_id(value: str) -> int:
    return {
        "fraud_dominant": 0,
        "benign_dominant": 1,
        "mixed": 2,
        "weak": 3,
    }.get(value, -1)


def load_stage2_sources(dataset_name: str, model_name: str, run_names: list[str], seed: int) -> tuple[dict[int, dict], list[dict], dict[str, int]]:
    evidence_cards: dict[int, dict] = {}
    accepted_by_node: dict[int, dict] = {}
    stats: dict[str, int] = {}

    for source_run in run_names:
        err_cache_dir = get_err_cache_dir(dataset_name, model_name, source_run, seed)
        cards_path = err_cache_dir / "evidence_cards.jsonl"
        accepted_path = err_cache_dir / "accepted_err.jsonl"
        if not cards_path.exists() or not accepted_path.exists():
            print(f"Warning: ERR cache source incomplete for {source_run}: {err_cache_dir}")
            continue

        source_cards = load_evidence_cards(cards_path)
        source_errs = load_accepted_errs(accepted_path)
        stats[f"{source_run}_accepted_err"] = len(source_errs)

        for err in source_errs:
            node_id = err.get("node_id")
            if not isinstance(node_id, int):
                continue
            current = accepted_by_node.get(node_id)
            if current is None or _err_preference_key(err) > _err_preference_key(current):
                accepted_by_node[node_id] = err
                if node_id in source_cards:
                    evidence_cards[node_id] = source_cards[node_id]

    stats["merged_accepted_err"] = len(accepted_by_node)
    stats["merged_evidence_cards"] = len(evidence_cards)
    return evidence_cards, list(accepted_by_node.values()), stats


def prepare_targets(
    num_nodes: int,
    accepted_errs: list[dict],
    evidence_cards: dict[int, dict],
    train_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    num_slots = len(get_evidence_slots())

    evidence_token_ids = torch.zeros(num_nodes, num_slots, dtype=torch.long)
    risk_type_ids = torch.zeros(num_nodes, dtype=torch.long)
    pos_masks = torch.zeros(num_nodes, num_slots, dtype=torch.float)
    neg_masks = torch.zeros(num_nodes, num_slots, dtype=torch.float)
    accepted_mask = torch.zeros(num_nodes, dtype=torch.float)
    direction_ids = torch.full((num_nodes,), -1, dtype=torch.long)
    strength_ids = torch.full((num_nodes,), -1, dtype=torch.long)
    polarity_ids = torch.full((num_nodes,), -1, dtype=torch.long)

    for err_data in accepted_errs:
        node_id = err_data["node_id"]
        if node_id >= num_nodes:
            continue

        err = ERR(
            node_id=node_id,
            risk_type=err_data["risk_type"],
            supporting_evidence=err_data["supporting_evidence"],
            counter_evidence=err_data["counter_evidence"],
            summary="",
            evidence_direction=err_data.get("evidence_direction", "uncertain"),
            evidence_strength=err_data.get("evidence_strength", "weak"),
            uncertainty_factors=err_data.get("uncertainty_factors", []),
        )

        targets = encode_err_targets(err)
        direction_target = encode_direction_target(err)
        risk_type_ids[node_id] = targets["risk_type_id"]
        pos_masks[node_id] = targets["pos_mask"]
        neg_masks[node_id] = targets["neg_mask"]
        accepted_mask[node_id] = 1.0
        direction_ids[node_id] = direction_target["direction_id"]
        strength_ids[node_id] = _strength_id(err.evidence_strength)

    for node_id, card in evidence_cards.items():
        if node_id >= num_nodes:
            continue
        reasoning = card.get("reasoning", {})
        evidence_token_ids[node_id] = encode_reasoning(reasoning)
        polarity_ids[node_id] = _polarity_id(reasoning.get("evidence_polarity", "unknown"))

    return {
        "evidence_token_ids": evidence_token_ids,
        "risk_type_id": risk_type_ids,
        "pos_mask": pos_masks,
        "neg_mask": neg_masks,
        "accepted_mask": accepted_mask,
        "direction_id": direction_ids,
        "strength_id": strength_ids,
        "polarity_id": polarity_ids,
    }


def compute_train_pos_weight(y: torch.Tensor, train_mask: torch.Tensor) -> float:
    labels = y[train_mask].float()
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives.item() <= 0:
        return 1.0
    return float((negatives / positives).item())


def default_teacher_latent_dir(dataset_name: str, model_name: str, stage2_run_name: str, seed: int) -> Path:
    return Path("artifacts") / "teacher_latents" / dataset_name / model_name / stage2_run_name / f"seed_{seed}"


def default_relation_feature_path(dataset_name: str, model_name: str, seed: int, relation_set: str = "all") -> Path:
    return Path("artifacts") / "relation_features" / dataset_name / model_name / f"seed_{seed}" / relation_set / "rel_stats.pt"


def _relation_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).upper() for item in value if str(item).strip()]
    return [item.strip().upper() for item in str(value).split(",") if item.strip()]


def load_teacher_latent_targets(
    path: Path,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    node_ids = payload.get("node_ids")
    latents = payload.get("latents")
    if node_ids is None or latents is None:
        raise ValueError(f"Invalid teacher latent file: {path}")

    node_ids = torch.as_tensor(node_ids, dtype=torch.long)
    latents = torch.as_tensor(latents, dtype=torch.float)
    latent_dim = int(payload.get("latent_dim", latents.shape[1] if latents.ndim == 2 else 0))
    if latents.ndim != 2:
        raise ValueError(f"teacher latents must be rank-2, got {tuple(latents.shape)}")

    target = torch.zeros(num_nodes, latents.shape[1], dtype=torch.float)
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid = (node_ids >= 0) & (node_ids < num_nodes)
    target[node_ids[valid]] = latents[valid]
    mask[node_ids[valid]] = True
    meta = {
        "teacher_latent_path": str(path),
        "teacher_latent_dim": latent_dim,
        "teacher_latent_count": int(valid.sum().item()),
    }
    return target, mask, meta


def add_teacher_latents_to_targets(
    targets: dict[str, torch.Tensor],
    teacher_latents: torch.Tensor | None,
    teacher_mask: torch.Tensor | None,
) -> None:
    if teacher_latents is None or teacher_mask is None:
        num_nodes = targets["accepted_mask"].shape[0]
        targets["teacher_latents"] = torch.empty(num_nodes, 0)
        targets["teacher_latent_mask"] = torch.zeros(num_nodes, dtype=torch.bool)
        return
    accepted = targets["accepted_mask"].bool()
    targets["teacher_latents"] = teacher_latents
    targets["teacher_latent_mask"] = teacher_mask.bool() & accepted


def load_judge_features(path: Path, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = payload.get("features")
    mask = payload.get("mask")
    if not isinstance(features, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError(f"Invalid judge feature payload: {path}")
    if features.shape[0] != num_nodes or mask.shape[0] != num_nodes:
        raise ValueError(
            f"judge feature shape mismatch: features={tuple(features.shape)}, "
            f"mask={tuple(mask.shape)}, num_nodes={num_nodes}"
        )
    return features.float(), mask.bool()


def write_epoch_logs(log_dir: Path, rows: list[dict[str, float]]) -> None:
    jsonl_path = log_dir / "stage3_train_log.jsonl"
    csv_path = log_dir / "stage3_train_log.csv"
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    if rows:
        fields = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")


@torch.no_grad()
def run_reasoner_full(reasoner, z, base_logits, targets, device):
    reasoner.eval()
    relation_features = targets.get("relation_features")
    if relation_features is not None:
        relation_features = relation_features.to(device)
    judge_features = targets.get("judge_features")
    judge_mask = targets.get("judge_mask")
    if judge_features is not None:
        judge_features = judge_features.to(device)
    if judge_mask is not None:
        judge_mask = judge_mask.to(device)
    return reasoner(
        z,
        base_logits,
        targets["evidence_token_ids"].to(device),
        relation_features=relation_features,
        judge_features=judge_features,
        judge_mask=judge_mask,
        return_debug=True,
    )


def ranking_gap_diagnostics(final_logit: torch.Tensor, base_logits: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor) -> dict[str, float]:
    train = train_mask.bool()
    positives = train & (y.long() == 1)
    negatives = train & (y.long() == 0)
    if not bool(positives.any()) or not bool(negatives.any()):
        return {
            "ranking_gap_pos_vs_hard_neg": 0.0,
            "base_ranking_gap_pos_vs_hard_neg": 0.0,
            "hard_negative_count": 0.0,
        }
    neg_idx = negatives.nonzero(as_tuple=True)[0]
    hard_score = torch.maximum(torch.sigmoid(base_logits.detach()), torch.sigmoid(final_logit.detach()))
    k = min(int(positives.sum().item()) * 4, int(neg_idx.numel()), 512)
    hard_neg = neg_idx[torch.topk(hard_score[neg_idx], k=k).indices]
    final_gap = final_logit[positives].mean() - final_logit[hard_neg].mean()
    base_gap = base_logits[positives].mean() - base_logits[hard_neg].mean()
    return {
        "ranking_gap_pos_vs_hard_neg": float(final_gap.item()),
        "base_ranking_gap_pos_vs_hard_neg": float(base_gap.item()),
        "hard_negative_count": float(hard_neg.numel()),
    }


def residual_diagnostics(final_logit: torch.Tensor, base_logits: torch.Tensor, max_allowed_shift: float) -> dict[str, float]:
    residual = final_logit - base_logits.detach()
    if residual.numel() == 0:
        return {
            "residual_shift_mean": 0.0,
            "residual_shift_abs_mean": 0.0,
            "residual_shift_max_abs": 0.0,
            "near_cap_fraction": 0.0,
        }
    return {
        "residual_shift_mean": float(residual.mean().item()),
        "residual_shift_abs_mean": float(residual.abs().mean().item()),
        "residual_shift_max_abs": float(residual.abs().max().item()),
        "near_cap_fraction": float((residual.abs() >= 0.9 * max_allowed_shift).float().mean().item()),
    }


def _tensor_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2 or b.numel() < 2:
        return 0.0
    a = a.float().view(-1)
    b = b.float().view(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float(((a * b).sum() / denom).item())


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_relation_gate_diagnostics(
    log_dir: Path,
    outputs: dict[str, torch.Tensor],
    relation_names: list[str],
    final_logit: torch.Tensor,
    base_logits: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    max_allowed_shift: float,
) -> dict[str, object]:
    gates = outputs.get("relation_gate_values")
    if gates is None or not relation_names:
        return {}
    gates = gates.detach().cpu().float()
    contributions = outputs.get("relation_contribution_norms")
    if contributions is None:
        contributions = torch.zeros_like(gates)
    else:
        contributions = contributions.detach().cpu().float()

    residual = (final_logit.detach().cpu() - base_logits.detach().cpu()).float()
    near_cap = (residual.abs() >= 0.9 * float(max_allowed_shift)).float()
    labels = y.detach().cpu().long()
    base_pred = (torch.sigmoid(base_logits.detach().cpu()) >= 0.5).long()

    overall_rows: list[dict[str, object]] = []
    by_label_rows: list[dict[str, object]] = []
    by_status_rows: list[dict[str, object]] = []
    gate_mean = gates.mean(dim=1)
    sparse = outputs.get("relation_gate_sparse_loss", torch.tensor(0.0))
    summary: dict[str, object] = {
        "seed": seed,
        "relations": relation_names,
        "gate_vs_near_cap_corr": _tensor_corr(gate_mean, near_cap),
        "gate_vs_abs_residual_corr": _tensor_corr(gate_mean, residual.abs()),
        "mean_gate_all_relations": float(gate_mean.mean().item()),
        "mean_sparse_penalty": float(sparse.detach().cpu().item()),
    }

    for idx, relation in enumerate(relation_names):
        gate = gates[:, idx]
        contrib = contributions[:, idx]
        row = {
            "seed": seed,
            "relation": relation,
            "gate_mean": float(gate.mean().item()),
            "gate_std": float(gate.std(unbiased=False).item()),
            "gate_open_rate": float((gate >= 0.5).float().mean().item()),
            "contribution_norm_mean": float(contrib.mean().item()),
            "gate_vs_near_cap_corr": _tensor_corr(gate, near_cap),
            "gate_vs_abs_residual_corr": _tensor_corr(gate, residual.abs()),
        }
        overall_rows.append(row)
        summary[f"{relation}_gate_mean"] = row["gate_mean"]
        summary[f"{relation}_gate_open_rate"] = row["gate_open_rate"]
        summary[f"{relation}_contribution_norm_mean"] = row["contribution_norm_mean"]

        for label in (0, 1):
            mask = labels == label
            if bool(mask.any()):
                by_label_rows.append({
                    "seed": seed,
                    "relation": relation,
                    "label": label,
                    "count": int(mask.sum().item()),
                    "gate_mean": float(gate[mask].mean().item()),
                    "gate_open_rate": float((gate[mask] >= 0.5).float().mean().item()),
                    "contribution_norm_mean": float(contrib[mask].mean().item()),
                })

        status_masks = {
            "TP": (labels == 1) & (base_pred == 1),
            "TN": (labels == 0) & (base_pred == 0),
            "FP": (labels == 0) & (base_pred == 1),
            "FN": (labels == 1) & (base_pred == 0),
        }
        for status, mask in status_masks.items():
            if bool(mask.any()):
                by_status_rows.append({
                    "seed": seed,
                    "relation": relation,
                    "base_status": status,
                    "count": int(mask.sum().item()),
                    "gate_mean": float(gate[mask].mean().item()),
                    "gate_open_rate": float((gate[mask] >= 0.5).float().mean().item()),
                    "contribution_norm_mean": float(contrib[mask].mean().item()),
                })

    _write_rows(log_dir / "relation_gate_stats.csv", overall_rows)
    _write_rows(log_dir / "relation_gate_by_seed.csv", overall_rows)
    _write_rows(log_dir / "relation_gate_by_relation.csv", overall_rows)
    _write_rows(log_dir / "relation_gate_by_label.csv", by_label_rows)
    _write_rows(log_dir / "relation_gate_by_base_status.csv", by_status_rows)
    (log_dir / "relation_gate_stats.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_llm_judge_diagnostics(
    log_dir: Path,
    outputs: dict[str, torch.Tensor],
    judge_mask: torch.Tensor,
    judge_features: torch.Tensor,
    max_allowed_shift: float,
) -> dict[str, float]:
    alpha = outputs.get("alpha_llm")
    delta = outputs.get("delta_llm")
    llm_residual = outputs.get("llm_residual")
    rel_only = outputs.get("rel_only_logit")
    final = outputs.get("final_logit")
    if alpha is None or delta is None or llm_residual is None or rel_only is None or final is None:
        summary = {"judge_enabled": 0.0}
        (log_dir / "llm_judge_diagnostics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary

    alpha_cpu = alpha.detach().cpu().float()
    delta_cpu = delta.detach().cpu().float()
    residual_cpu = llm_residual.detach().cpu().float()
    rel_delta = (final.detach().cpu().float() - rel_only.detach().cpu().float())
    mask_cpu = judge_mask.detach().cpu().bool()
    features = judge_features.detach().cpu().float()
    near_cap = rel_delta.abs() >= 0.9 * float(max_allowed_shift)

    def _mean(values: torch.Tensor) -> float:
        return float(values.mean().item()) if values.numel() else 0.0

    rows: list[dict[str, float | str]] = []
    verdict_names = ("fake", "real", "uncertain")
    for idx, name in enumerate(verdict_names):
        group = mask_cpu & (features[:, idx] > 0.5)
        rows.append({
            "group": f"verdict_{name}",
            "count": float(group.sum().item()),
            "alpha_mean": _mean(alpha_cpu[group]),
            "delta_mean": _mean(delta_cpu[group]),
            "residual_abs_mean": _mean(residual_cpu[group].abs()),
        })
    _write_rows(log_dir / "llm_judge_by_verdict.csv", rows)

    rows = []
    strength_names = ("weak", "moderate", "strong")
    for idx, name in enumerate(strength_names):
        group = mask_cpu & (features[:, 3 + idx] > 0.5)
        rows.append({
            "group": f"strength_{name}",
            "count": float(group.sum().item()),
            "alpha_mean": _mean(alpha_cpu[group]),
            "delta_mean": _mean(delta_cpu[group]),
            "residual_abs_mean": _mean(residual_cpu[group].abs()),
        })
    rows.append({
        "group": "accepted_judge",
        "count": float(mask_cpu.sum().item()),
        "alpha_mean": _mean(alpha_cpu[mask_cpu]),
        "delta_mean": _mean(delta_cpu[mask_cpu]),
        "residual_abs_mean": _mean(residual_cpu[mask_cpu].abs()),
    })
    _write_rows(log_dir / "llm_judge_by_strength.csv", rows)
    summary = {
        "judge_enabled": 1.0,
        "judge_count": float(mask_cpu.sum().item()),
        "alpha_llm_mean": _mean(alpha_cpu[mask_cpu]),
        "alpha_llm_all_mean": _mean(alpha_cpu),
        "alpha_llm_min": float(alpha_cpu[mask_cpu].min().item()) if bool(mask_cpu.any()) else 0.0,
        "alpha_llm_max": float(alpha_cpu[mask_cpu].max().item()) if bool(mask_cpu.any()) else 0.0,
        "delta_llm_mean": _mean(delta_cpu[mask_cpu]),
        "delta_llm_max_abs": float(delta_cpu[mask_cpu].abs().max().item()) if bool(mask_cpu.any()) else 0.0,
        "final_vs_rel_delta_mean": _mean(rel_delta[mask_cpu]),
        "final_vs_rel_delta_abs_mean": _mean(rel_delta[mask_cpu].abs()),
        "llm_near_cap_fraction": _mean(near_cap[mask_cpu].float()),
    }
    (log_dir / "llm_judge_diagnostics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def latent_alignment_diagnostics(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, float]:
    mask = targets.get("teacher_latent_mask")
    teacher = targets.get("teacher_latents")
    if mask is None or teacher is None or not bool(mask.bool().any()):
        return {"latent_cosine_similarity": 0.0, "latent_teacher_count": 0.0}
    z_student = outputs["z_student"].detach().cpu()
    teacher = teacher.float()
    if teacher.shape[1] > z_student.shape[1]:
        teacher = teacher[:, : z_student.shape[1]]
    elif teacher.shape[1] < z_student.shape[1]:
        teacher = torch.nn.functional.pad(teacher, (0, z_student.shape[1] - teacher.shape[1]))
    valid = mask.bool().cpu()
    cosine = torch.nn.functional.cosine_similarity(z_student[valid], teacher[valid], dim=-1)
    return {
        "latent_cosine_similarity": float(cosine.mean().item()),
        "latent_teacher_count": float(valid.sum().item()),
    }


def correction_rate_diagnostics(
    final_logit: torch.Tensor,
    base_logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    final_prob = torch.sigmoid(final_logit[mask])
    base_prob = torch.sigmoid(base_logits[mask])
    labels = y[mask].long()
    base_pred = base_prob >= 0.5
    final_pred = final_prob >= 0.5

    base_fn = (labels == 1) & ~base_pred
    base_fp = (labels == 0) & base_pred
    fn_corrected = base_fn & final_pred
    fp_corrected = base_fp & ~final_pred

    return {
        "base_fn_count": float(base_fn.sum().item()),
        "base_fp_count": float(base_fp.sum().item()),
        "fn_correction_count": float(fn_corrected.sum().item()),
        "fp_correction_count": float(fp_corrected.sum().item()),
        "fn_correction_rate": float(fn_corrected.sum().item() / base_fn.sum().item()) if bool(base_fn.any()) else 0.0,
        "fp_correction_rate": float(fp_corrected.sum().item() / base_fp.sum().item()) if bool(base_fp.any()) else 0.0,
    }


def train_one_epoch(reasoner, z, base_logits, targets, y, train_mask, optimizer, config, epoch: int = 0):
    reasoner.train()
    device = z.device

    evi_ids = targets["evidence_token_ids"].to(device)
    risk_type_id = targets["risk_type_id"].to(device)
    pos_mask = targets["pos_mask"].to(device)
    neg_mask = targets["neg_mask"].to(device)
    accepted_mask = targets["accepted_mask"].to(device)
    direction_id = targets["direction_id"].to(device)
    strength_id = targets["strength_id"].to(device)
    polarity_id = targets["polarity_id"].to(device)
    teacher_latents = targets.get("teacher_latents")
    teacher_latent_mask = targets.get("teacher_latent_mask")
    contrast_class_id = targets.get("contrast_class_id")
    relation_features = targets.get("relation_features")
    judge_features = targets.get("judge_features")
    judge_mask = targets.get("judge_mask")
    if teacher_latents is not None:
        teacher_latents = teacher_latents.to(device)
    if teacher_latent_mask is not None:
        teacher_latent_mask = teacher_latent_mask.to(device)
    if contrast_class_id is not None:
        contrast_class_id = contrast_class_id.to(device)
    if relation_features is not None:
        relation_features = relation_features.to(device)
    if judge_features is not None:
        judge_features = judge_features.to(device)
    if judge_mask is not None:
        judge_mask = judge_mask.to(device)

    z_train = z[train_mask]
    base_logit_train = base_logits[train_mask]
    evi_ids_train = evi_ids[train_mask]
    rel_train = relation_features[train_mask] if relation_features is not None else None
    judge_train = judge_features[train_mask] if judge_features is not None else None
    judge_mask_train = judge_mask[train_mask] if judge_mask is not None else None

    outputs = reasoner(
        z_train,
        base_logit_train,
        evi_ids_train,
        relation_features=rel_train,
        judge_features=judge_train,
        judge_mask=judge_mask_train,
        return_debug=True,
    )

    targets_train = {
        "risk_type_id": risk_type_id[train_mask],
        "pos_mask": pos_mask[train_mask],
        "neg_mask": neg_mask[train_mask],
        "accepted_mask": accepted_mask[train_mask],
        "direction_id": direction_id[train_mask],
        "strength_id": strength_id[train_mask],
        "polarity_id": polarity_id[train_mask],
    }
    if teacher_latents is not None and teacher_latent_mask is not None:
        targets_train["teacher_latents"] = teacher_latents[train_mask]
        targets_train["teacher_latent_mask"] = teacher_latent_mask[train_mask]
    if contrast_class_id is not None:
        targets_train["contrast_class_id"] = contrast_class_id[train_mask]
    if judge_mask_train is not None:
        targets_train["judge_mask"] = judge_mask_train

    rc = config.get("reasoner", {})
    risk_type_names = get_reason_types()
    lambda_gate_sparse = float(rc.get("lambda_gate_sparse", 0.0))
    if epoch <= int(rc.get("gate_warmup_epochs", 0)):
        lambda_gate_sparse = 0.0

    train_only_mask = torch.ones(z_train.shape[0], dtype=torch.bool, device=device)
    if rc.get("loss_mode") == "cover_judge":
        loss, loss_dict = compute_cover_judge_loss(
            outputs,
            y=y[train_mask],
            targets=targets_train,
            train_mask=train_only_mask,
            base_logits=base_logit_train,
            pos_weight=rc.get("pos_weight", None),
            lambda_rank=rc.get("lambda_pairrank", rc.get("lambda_rank", 0.2)),
            pairrank_temperature=rc.get("pairrank_temperature", 0.1),
            lambda_judge=rc.get("lambda_judge", 0.1),
            lambda_llm_reg=rc.get("lambda_llm_reg", 0.001),
            llm_alpha_reg=rc.get("llm_alpha_reg", 0.0),
        )
    elif rc.get("loss_mode") == "cover_lift":
        loss, loss_dict = compute_cover_lift_loss(
            outputs,
            y=y[train_mask],
            targets=targets_train,
            train_mask=train_only_mask,
            base_logits=base_logit_train,
            pos_weight=rc.get("pos_weight", None),
            lambda_cve=rc.get("lambda_cve", 0.3),
            lambda_latent=rc.get("lambda_latent", 0.2),
            lambda_intervene=rc.get("lambda_intervene", 0.05),
            lambda_pairrank=rc.get("lambda_pairrank", 0.2),
            pairrank_temperature=rc.get("pairrank_temperature", 0.1),
            delta_scale=rc.get("delta_scale", 2.0),
            max_allowed_shift=rc.get("max_allowed_shift", 0.2),
            latent_contrastive_weight=rc.get("latent_contrastive_weight", 0.0),
            contrast_temperature=rc.get("contrast_temperature", 0.1),
            lambda_gate_sparse=lambda_gate_sparse,
        )
    else:
        loss, loss_dict = compute_cvscd_loss(
            outputs,
            y=y[train_mask],
            targets=targets_train,
            train_mask=train_only_mask,
            base_logits=base_logit_train,
            risk_type_names=risk_type_names,
            pos_weight=rc.get("pos_weight", None),
            lambda_det=rc.get("lambda_det", 1.0),
            lambda_err=rc.get("lambda_err", 0.3),
            lambda_signed=rc.get("lambda_signed", 0.1),
            lambda_corr=rc.get("lambda_corr", 0.1),
            correction_weight=rc.get("correction_weight", 2.0),
            signed_margin=rc.get("signed_margin", 0.2),
            correction_margin=rc.get("correction_margin", 0.05),
            anchor_weight=rc.get("anchor_weight", 0.001),
            max_shift_penalty_weight=rc.get("max_shift_penalty_weight", 0.001),
            max_allowed_shift=rc.get("max_allowed_shift", 0.3),
            direction_ids=direction_id[train_mask],
            strength_ids=strength_id[train_mask],
            polarity_ids=polarity_id[train_mask],
            lambda_direction=rc.get("lambda_direction", 1.0),
            base_correct_anchor_weight=rc.get("base_correct_anchor_weight", None),
            uncertain_anchor_weight=rc.get("uncertain_anchor_weight", None),
            decrease_anchor_weight=rc.get("decrease_anchor_weight", None),
            fp_anchor_weight=rc.get("fp_anchor_weight", None),
            non_fn_positive_anchor_weight=rc.get("non_fn_positive_anchor_weight", None),
            non_accepted_anchor_weight=rc.get("non_accepted_anchor_weight", None),
            enable_fp_negative_correction=rc.get("enable_fp_negative_correction", False),
            residual_cap=rc.get("rho", 0.3) * rc.get("delta_scale", 2.0),
        )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_dict


@torch.no_grad()
def evaluate(reasoner, z, base_logits, targets, mask, y, device):
    reasoner.eval()

    evi_ids = targets["evidence_token_ids"].to(device)

    z_mask = z[mask]
    base_logit_mask = base_logits[mask]
    evi_ids_mask = evi_ids[mask]
    relation_features = targets.get("relation_features")
    rel_mask = relation_features.to(device)[mask] if relation_features is not None else None
    judge_features = targets.get("judge_features")
    judge_mask = targets.get("judge_mask")
    judge_feat_mask = judge_features.to(device)[mask] if judge_features is not None else None
    judge_bool_mask = judge_mask.to(device)[mask] if judge_mask is not None else None

    outputs = reasoner(
        z_mask,
        base_logit_mask,
        evi_ids_mask,
        relation_features=rel_mask,
        judge_features=judge_feat_mask,
        judge_mask=judge_bool_mask,
    )

    y_np = y[mask].cpu().numpy()
    prob = torch.sigmoid(outputs["final_logit"]).cpu().numpy()

    metrics = compute_metrics(y_np, prob)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_name", type=str, default="rule")
    parser.add_argument("--stage3_run_name", type=str, default=None)
    parser.add_argument("--stage2_run_name", type=str, default=None, help="ERR cache run_name to train from; defaults to --run_name")
    parser.add_argument("--stage2_run_names", nargs="+", type=str, default=None, help="Multiple ERR cache run_names to merge for training")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="Override train.device, e.g. cuda:2")
    parser.add_argument("--stratified", action="store_true", help="Use stratified split")
    parser.add_argument("--gate_mode", type=str, default=None, choices=VALID_GATE_MODES)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--delta_scale", type=float, default=None)
    parser.add_argument("--lambda_evi", type=float, default=None)
    parser.add_argument("--residual_l2_weight", type=float, default=None)
    parser.add_argument("--max_shift_penalty_weight", type=float, default=None)
    parser.add_argument("--max_abs_shift", type=float, default=None)
    parser.add_argument("--lambda_det", type=float, default=None)
    parser.add_argument("--lambda_err", type=float, default=None)
    parser.add_argument("--lambda_signed", type=float, default=None)
    parser.add_argument("--lambda_corr", type=float, default=None)
    parser.add_argument("--correction_weight", type=float, default=None)
    parser.add_argument("--signed_margin", type=float, default=None)
    parser.add_argument("--correction_margin", type=float, default=None)
    parser.add_argument("--anchor_weight", type=float, default=None)
    parser.add_argument("--max_allowed_shift", type=float, default=None)
    parser.add_argument("--lambda_direction", type=float, default=None)
    parser.add_argument("--loss_mode", type=str, default=None, choices=["cvscd", "cover_lift", "cover_judge"])
    parser.add_argument("--use_teacher_latents", action="store_true")
    parser.add_argument("--teacher_latents_path", type=str, default=None)
    parser.add_argument("--teacher_latent_meta_path", type=str, default=None)
    parser.add_argument("--use_relation_features", action="store_true")
    parser.add_argument("--relation_features_path", type=str, default=None)
    parser.add_argument("--relation_set", type=str, default="all")
    parser.add_argument("--relation_hidden_dim", type=int, default=None)
    parser.add_argument("--relation_fusion_mode", type=str, default=None, choices=VALID_RELATION_FUSION_MODES)
    parser.add_argument("--anchor_relation", type=str, default=None)
    parser.add_argument("--optional_relations", type=str, default=None)
    parser.add_argument("--lambda_gate_sparse", type=float, default=None)
    parser.add_argument("--relation_dropout", type=float, default=None)
    parser.add_argument("--log_relation_gates", action="store_true")
    parser.add_argument("--gate_hidden_dim", type=int, default=None)
    parser.add_argument("--gate_temperature", type=float, default=None)
    parser.add_argument("--gate_warmup_epochs", type=int, default=None)
    parser.add_argument("--use_llm_judge", action="store_true")
    parser.add_argument("--judge_features_path", type=str, default=None)
    parser.add_argument("--judge_feature_meta_path", type=str, default=None)
    parser.add_argument("--fusion_mode", type=str, default=None, choices=["none", "gated_llm_residual"])
    parser.add_argument("--lambda_judge", type=float, default=None)
    parser.add_argument("--lambda_llm_reg", type=float, default=None)
    parser.add_argument("--llm_delta_scale", type=float, default=None)
    parser.add_argument("--llm_alpha_reg", type=float, default=None)
    parser.add_argument("--lambda_alpha", type=float, default=None, help="Alias for --llm_alpha_reg")
    parser.add_argument("--alpha_max", type=float, default=None)
    parser.add_argument("--strength_aware_alpha", action="store_true")
    parser.add_argument("--allow_missing_judge_features", action="store_true")
    parser.add_argument("--init_reasoner_checkpoint", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--lambda_cve", type=float, default=None)
    parser.add_argument("--lambda_latent", type=float, default=None)
    parser.add_argument("--lambda_intervene", type=float, default=None)
    parser.add_argument("--lambda_pairrank", type=float, default=None)
    parser.add_argument("--pairrank_temperature", type=float, default=None)
    parser.add_argument("--base_correct_anchor_weight", type=float, default=None)
    parser.add_argument("--uncertain_anchor_weight", type=float, default=None)
    parser.add_argument("--decrease_anchor_weight", type=float, default=None)
    parser.add_argument("--fp_anchor_weight", type=float, default=None)
    parser.add_argument("--non_fn_positive_anchor_weight", type=float, default=None)
    parser.add_argument("--non_accepted_anchor_weight", type=float, default=None)
    parser.add_argument("--enable_fp_negative_correction", action="store_true")
    parser.add_argument("--threshold_mode", type=str, default=None, choices=["fixed", "val_f1", "val_macro_f1"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    rc = config.setdefault("reasoner", {})
    if args.gate_mode is not None:
        rc["gate_mode"] = args.gate_mode
    if args.rho is not None:
        rc["rho"] = args.rho
    if args.delta_scale is not None:
        rc["delta_scale"] = args.delta_scale
    if args.lambda_evi is not None:
        rc["lambda_evi"] = args.lambda_evi
    if args.residual_l2_weight is not None:
        rc["residual_l2_weight"] = args.residual_l2_weight
    if args.max_shift_penalty_weight is not None:
        rc["max_shift_penalty_weight"] = args.max_shift_penalty_weight
    if args.max_abs_shift is not None:
        rc["max_abs_shift"] = args.max_abs_shift
    if args.lambda_det is not None:
        rc["lambda_det"] = args.lambda_det
    if args.lambda_err is not None:
        rc["lambda_err"] = args.lambda_err
    if args.lambda_signed is not None:
        rc["lambda_signed"] = args.lambda_signed
    if args.lambda_corr is not None:
        rc["lambda_corr"] = args.lambda_corr
    if args.correction_weight is not None:
        rc["correction_weight"] = args.correction_weight
    if args.signed_margin is not None:
        rc["signed_margin"] = args.signed_margin
    if args.correction_margin is not None:
        rc["correction_margin"] = args.correction_margin
    if args.anchor_weight is not None:
        rc["anchor_weight"] = args.anchor_weight
    if args.max_allowed_shift is not None:
        rc["max_allowed_shift"] = args.max_allowed_shift
    if args.lambda_direction is not None:
        rc["lambda_direction"] = args.lambda_direction
    if args.loss_mode is not None:
        rc["loss_mode"] = args.loss_mode
    if args.use_teacher_latents:
        rc["use_teacher_latents"] = True
    if args.teacher_latents_path is not None:
        rc["teacher_latents_path"] = args.teacher_latents_path
    if args.teacher_latent_meta_path is not None:
        rc["teacher_latent_meta_path"] = args.teacher_latent_meta_path
    if args.use_relation_features:
        rc["use_relation_features"] = True
    if args.relation_features_path is not None:
        rc["relation_features_path"] = args.relation_features_path
    if args.relation_set is not None:
        rc["relation_set"] = args.relation_set
    if args.relation_hidden_dim is not None:
        rc["relation_hidden_dim"] = args.relation_hidden_dim
    if args.relation_fusion_mode is not None:
        rc["relation_fusion_mode"] = args.relation_fusion_mode
    if args.anchor_relation is not None:
        rc["anchor_relation"] = args.anchor_relation.upper()
    if args.optional_relations is not None:
        rc["optional_relations"] = _relation_list(args.optional_relations)
    if args.lambda_gate_sparse is not None:
        rc["lambda_gate_sparse"] = args.lambda_gate_sparse
    if args.relation_dropout is not None:
        rc["relation_dropout"] = args.relation_dropout
    if args.log_relation_gates:
        rc["log_relation_gates"] = True
    if args.gate_hidden_dim is not None:
        rc["gate_hidden_dim"] = args.gate_hidden_dim
    if args.gate_temperature is not None:
        rc["gate_temperature"] = args.gate_temperature
    if args.gate_warmup_epochs is not None:
        rc["gate_warmup_epochs"] = args.gate_warmup_epochs
    if args.use_llm_judge:
        rc["use_llm_judge"] = True
    if args.judge_features_path is not None:
        rc["judge_features_path"] = args.judge_features_path
    if args.judge_feature_meta_path is not None:
        rc["judge_feature_meta_path"] = args.judge_feature_meta_path
    if args.fusion_mode is not None:
        rc["fusion_mode"] = args.fusion_mode
    if args.lambda_judge is not None:
        rc["lambda_judge"] = args.lambda_judge
    if args.lambda_llm_reg is not None:
        rc["lambda_llm_reg"] = args.lambda_llm_reg
    if args.llm_delta_scale is not None:
        rc["llm_delta_scale"] = args.llm_delta_scale
    if args.llm_alpha_reg is not None:
        rc["llm_alpha_reg"] = args.llm_alpha_reg
    if args.lambda_alpha is not None:
        rc["llm_alpha_reg"] = args.lambda_alpha
        rc["lambda_alpha"] = args.lambda_alpha
    if args.alpha_max is not None:
        rc["alpha_max"] = args.alpha_max
    if args.strength_aware_alpha:
        rc["strength_aware_alpha"] = True
    if args.allow_missing_judge_features:
        rc["allow_missing_judge_features"] = True
    if args.init_reasoner_checkpoint is not None:
        rc["init_reasoner_checkpoint"] = args.init_reasoner_checkpoint
    if args.latent_dim is not None:
        rc["latent_dim"] = args.latent_dim
    if args.lambda_cve is not None:
        rc["lambda_cve"] = args.lambda_cve
    if args.lambda_latent is not None:
        rc["lambda_latent"] = args.lambda_latent
    if args.lambda_intervene is not None:
        rc["lambda_intervene"] = args.lambda_intervene
    if args.lambda_pairrank is not None:
        rc["lambda_pairrank"] = args.lambda_pairrank
    if args.pairrank_temperature is not None:
        rc["pairrank_temperature"] = args.pairrank_temperature
    if args.base_correct_anchor_weight is not None:
        rc["base_correct_anchor_weight"] = args.base_correct_anchor_weight
    if args.uncertain_anchor_weight is not None:
        rc["uncertain_anchor_weight"] = args.uncertain_anchor_weight
    if args.decrease_anchor_weight is not None:
        rc["decrease_anchor_weight"] = args.decrease_anchor_weight
    if args.fp_anchor_weight is not None:
        rc["fp_anchor_weight"] = args.fp_anchor_weight
    if args.non_fn_positive_anchor_weight is not None:
        rc["non_fn_positive_anchor_weight"] = args.non_fn_positive_anchor_weight
    if args.non_accepted_anchor_weight is not None:
        rc["non_accepted_anchor_weight"] = args.non_accepted_anchor_weight
    if args.enable_fp_negative_correction:
        rc["enable_fp_negative_correction"] = True
    if args.threshold_mode is not None:
        rc["threshold_mode"] = args.threshold_mode

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    seed = args.seed or config["train"]["seed"]
    model_name = config["model"]["name"]
    run_name = args.stage3_run_name or args.run_name
    stage2_run_names = args.stage2_run_names or [args.stage2_run_name or args.run_name]

    torch.manual_seed(seed)
    requested_device = str(args.device or config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    config.setdefault("train", {})["device"] = requested_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    device_info = runtime_device_info(device)

    if args.debug:
        print("[DEBUG] Using tiny synthetic graph, 3 epochs")
        data = load_fraud_dataset("tiny", seed=seed)
        epochs = 3
    else:
        split_mode = config["dataset"].get("split_mode", "supervised")
        train_ratio = config["dataset"].get("train_ratio", 0.7)
        val_test_ratio = config["dataset"].get("val_test_ratio", [1, 2])
        stratified = config["dataset"].get("stratified", False)

        data = load_fraud_dataset(
            dataset_name, path=dataset_path, seed=seed,
            split_mode=split_mode, train_ratio=train_ratio, val_test_ratio=val_test_ratio,
            stratified=stratified,
        )
        epochs = config["train"].get("epochs", 200)

    extra_kwargs = {}
    if "attention_heads" in config["model"]:
        extra_kwargs["attention_heads"] = config["model"]["attention_heads"]
    base_model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
        **extra_kwargs,
    ).to(device)

    checkpoint_path = get_base_checkpoint_path(dataset_name, model_name, seed)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True)
        base_model.load_state_dict(state)
        print(f"Loaded base checkpoint from {checkpoint_path}")

    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        output = base_model(x, edge_index, return_output=True)
        base_logits = output.logits.detach()
        z = output.embeddings.detach()

    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)

    evidence_cards, accepted_errs, source_stats = load_stage2_sources(
        dataset_name, model_name, stage2_run_names, seed,
    )
    if accepted_errs:
        print(
            f"Loaded {len(evidence_cards)} cards, {len(accepted_errs)} merged accepted ERR "
            f"from {stage2_run_names}"
        )
    else:
        print("Warning: No accepted ERR cache found, using empty targets")

    targets = prepare_targets(
        num_nodes=data.x.shape[0],
        accepted_errs=accepted_errs,
        evidence_cards=evidence_cards,
        train_mask=data.train_mask,
    )

    teacher_latent_meta: dict[str, object] = {}
    loss_mode = rc.get("loss_mode", "cvscd")
    if loss_mode == "cover_lift":
        rc.setdefault("lambda_cve", 0.3)
        rc.setdefault("lambda_latent", 0.2)
        rc.setdefault("lambda_intervene", 0.05)
        rc.setdefault("lambda_pairrank", 0.2)
        rc.setdefault("pairrank_temperature", 0.1)
        rc.setdefault("latent_dim", 256)
        rc.setdefault("rho", 0.1)
        rc.setdefault("delta_scale", 2.0)
        rc.setdefault("max_allowed_shift", 0.2)
        rc.setdefault("pos_weight", compute_train_pos_weight(y, train_mask))
        rc.setdefault("lambda_gate_sparse", 0.0)
        rc.setdefault("gate_warmup_epochs", 0)
    if loss_mode == "cover_judge":
        rc.setdefault("lambda_pairrank", 0.2)
        rc.setdefault("pairrank_temperature", 0.1)
        rc.setdefault("lambda_judge", 0.1)
        rc.setdefault("lambda_llm_reg", 0.001)
        rc.setdefault("llm_alpha_reg", 0.0)
        rc.setdefault("alpha_max", 1.0)
        rc.setdefault("strength_aware_alpha", False)
        rc.setdefault("llm_delta_scale", 1.0)
        rc.setdefault("pos_weight", compute_train_pos_weight(y, train_mask))
        rc.setdefault("fusion_mode", "gated_llm_residual")
        rc.setdefault("use_llm_judge", True)

    teacher_latents = None
    teacher_latent_mask = None
    if rc.get("use_teacher_latents", False):
        teacher_latents_path = rc.get("teacher_latents_path")
        if teacher_latents_path is None:
            teacher_dir = default_teacher_latent_dir(dataset_name, model_name, stage2_run_names[0], seed)
            teacher_latents_path = str(teacher_dir / "teacher_latents.pt")
            rc["teacher_latents_path"] = teacher_latents_path
        teacher_path = Path(str(teacher_latents_path))
        if not teacher_path.exists():
            raise FileNotFoundError(
                f"--use_teacher_latents was set but teacher_latents.pt is missing: {teacher_path}"
            )
        teacher_latents, teacher_latent_mask, teacher_latent_meta = load_teacher_latent_targets(
            teacher_path,
            num_nodes=data.x.shape[0],
        )
        meta_path = rc.get("teacher_latent_meta_path")
        if meta_path is None:
            candidate = teacher_path.parent / "teacher_latent_meta.json"
            if candidate.exists():
                rc["teacher_latent_meta_path"] = str(candidate)
                teacher_latent_meta["teacher_latent_meta_path"] = str(candidate)
                teacher_latent_meta["teacher_latent_meta"] = json.loads(candidate.read_text())
        elif Path(str(meta_path)).exists():
            teacher_latent_meta["teacher_latent_meta_path"] = str(meta_path)
            teacher_latent_meta["teacher_latent_meta"] = json.loads(Path(str(meta_path)).read_text())

    add_teacher_latents_to_targets(targets, teacher_latents, teacher_latent_mask)

    relation_feature_meta: dict[str, object] = {}
    if rc.get("use_relation_features", False):
        relation_features_path = rc.get("relation_features_path")
        if relation_features_path is None:
            relation_features_path = str(default_relation_feature_path(
                dataset_name,
                model_name,
                seed,
                rc.get("relation_set", "all"),
            ))
            rc["relation_features_path"] = relation_features_path
        rel_path = Path(str(relation_features_path))
        if not rel_path.exists():
            raise FileNotFoundError(
                f"--use_relation_features was set but rel_stats.pt is missing: {rel_path}"
            )
        rel_stats, relation_feature_meta = load_relation_stats(rel_path, num_nodes=data.x.shape[0])
        targets["relation_features"] = rel_stats
        rc["relation_dim"] = int(rel_stats.shape[1])
        if relation_feature_meta.get("relations"):
            rc["relation_names"] = [str(name).upper() for name in relation_feature_meta["relations"]]
        if relation_feature_meta.get("stat_names_per_relation"):
            rc["relation_stat_dim"] = len(relation_feature_meta["stat_names_per_relation"])
        relation_feature_meta["relation_features_path"] = str(rel_path)
        print(f"Loaded relation features {tuple(rel_stats.shape)} from {rel_path}")

    judge_feature_meta: dict[str, object] = {}
    if rc.get("use_llm_judge", False):
        judge_features_path = rc.get("judge_features_path")
        if judge_features_path is None:
            if rc.get("allow_missing_judge_features", False):
                targets["judge_features"] = torch.empty(data.x.shape[0], 0)
                targets["judge_mask"] = torch.zeros(data.x.shape[0], dtype=torch.bool)
            else:
                raise FileNotFoundError("--use_llm_judge requires --judge_features_path")
        else:
            judge_path = Path(str(judge_features_path))
            if not judge_path.exists():
                if not rc.get("allow_missing_judge_features", False):
                    raise FileNotFoundError(f"judge_features.pt is missing: {judge_path}")
                targets["judge_features"] = torch.empty(data.x.shape[0], 0)
                targets["judge_mask"] = torch.zeros(data.x.shape[0], dtype=torch.bool)
            else:
                judge_features, loaded_judge_mask = load_judge_features(judge_path, num_nodes=data.x.shape[0])
                targets["judge_features"] = judge_features
                targets["judge_mask"] = loaded_judge_mask
                rc["judge_feature_dim"] = int(judge_features.shape[1])
                meta_path = rc.get("judge_feature_meta_path") or str(judge_path.parent / "judge_feature_meta.json")
                if Path(str(meta_path)).exists():
                    rc["judge_feature_meta_path"] = str(meta_path)
                    judge_feature_meta = json.loads(Path(str(meta_path)).read_text())
                print(
                    f"Loaded judge features {tuple(judge_features.shape)} from {judge_path} "
                    f"accepted={int(loaded_judge_mask.sum().item())}"
                )

    hidden_dim = z.shape[1]
    rho = rc.get("rho", 0.3)
    gate_mode = rc.get("gate_mode", "safe_residual")
    delta_scale = rc.get("delta_scale", 2.0)
    gate_bias_init = rc.get("gate_bias_init", -2.0)
    residual_init_zero = rc.get("residual_init_zero", True)

    reasoner = EvidenceReasoner(
        z_dim=hidden_dim,
        hidden_dim=rc.get("hidden_dim", 128),
        rho=rho,
        gate_mode=gate_mode,
        delta_scale=delta_scale,
        gate_bias_init=gate_bias_init,
        residual_init_zero=residual_init_zero,
        latent_dim=rc.get("latent_dim", 256),
        relation_dim=rc.get("relation_dim", 0),
        relation_hidden_dim=rc.get("relation_hidden_dim", 32),
        relation_fusion_mode=rc.get("relation_fusion_mode", "concat"),
        relation_names=_relation_list(rc.get("relation_names")),
        relation_stat_dim=rc.get("relation_stat_dim"),
        anchor_relation=rc.get("anchor_relation"),
        optional_relations=_relation_list(rc.get("optional_relations")),
        relation_dropout=rc.get("relation_dropout", 0.0),
        gate_hidden_dim=rc.get("gate_hidden_dim", 64),
        gate_temperature=rc.get("gate_temperature", 1.0),
        use_llm_judge=rc.get("use_llm_judge", False),
        judge_feature_dim=rc.get("judge_feature_dim", 0),
        fusion_mode=rc.get("fusion_mode", "none"),
        llm_delta_scale=rc.get("llm_delta_scale", 1.0),
        alpha_max=rc.get("alpha_max", 1.0),
        strength_aware_alpha=rc.get("strength_aware_alpha", False),
    ).to(device)

    if rc.get("init_reasoner_checkpoint"):
        init_path = Path(str(rc["init_reasoner_checkpoint"]))
        if init_path.exists():
            missing, unexpected = reasoner.load_state_dict(
                torch.load(init_path, map_location=device, weights_only=False),
                strict=False,
            )
            print(
                f"Initialized reasoner from {init_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            raise FileNotFoundError(f"init_reasoner_checkpoint not found: {init_path}")

    optimizer = torch.optim.Adam(
        reasoner.parameters(),
        lr=config["train"].get("lr", 0.001),
        weight_decay=config["train"].get("weight_decay", 0.0005),
    )

    patience = config["train"].get("patience", 30)
    select_metric = config["train"].get("select_metric", "roc_auc")
    best_val_score = float("-inf")
    patience_counter = 0
    best_state = None

    start_time = time.time()

    tb_logger = create_logger(dataset_name, model_name, seed, "stage3")
    last_loss_dict: dict[str, float] = {}
    best_loss_dict: dict[str, float] = {}
    epoch_rows: list[dict[str, float]] = []
    best_epoch_idx = 0
    best_val_auprc = float("-inf")

    for epoch in range(1, epochs + 1):
        loss, loss_dict = train_one_epoch(
            reasoner, z, base_logits, targets, y, train_mask, optimizer, config, epoch=epoch
        )
        last_loss_dict = dict(loss_dict)

        val_metrics = evaluate(reasoner, z, base_logits, targets, val_mask, y, device)

        tb_logger.log_scalar("train/loss", loss, epoch)
        tb_logger.log_scalars("train/loss_components", loss_dict, epoch)
        tb_logger.log_metrics(val_metrics, epoch, prefix="val")

        epoch_row: dict[str, float] = {"epoch": float(epoch)}
        for key, value in loss_dict.items():
            epoch_row[key] = float(value)
        epoch_row.update({
            "val_roc_auc": float(val_metrics.get("roc_auc", 0.0)),
            "val_auprc": float(val_metrics.get("auprc", 0.0)),
            "val_f1_at_val": float(val_metrics.get("f1", 0.0)),
            "val_macro_f1_at_val": float(val_metrics.get("macro_f1", 0.0)),
        })
        epoch_rows.append(epoch_row)
        best_val_auprc = max(best_val_auprc, float(val_metrics.get("auprc", 0.0)))

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}")

        val_score = val_metrics.get(select_metric, val_metrics["roc_auc"])
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.cpu().clone() for k, v in reasoner.state_dict().items()}
            best_loss_dict = dict(loss_dict)
            best_epoch_idx = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        reasoner.load_state_dict(best_state)
    else:
        best_state = {k: v.cpu().clone() for k, v in reasoner.state_dict().items()}

    test_mask = data.test_mask.to(device)
    test_metrics = evaluate(reasoner, z, base_logits, targets, test_mask, y, device)

    tb_logger.log_metrics(test_metrics, epoch, prefix="test")
    tb_logger.close()

    elapsed = time.time() - start_time

    print("\n=== Stage 3 Test Results ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    checkpoint_dir = ensure_dir(get_checkpoint_dir(dataset_name, model_name, run_name, seed))
    reasoner_path = checkpoint_dir / "reasoner.pt"
    best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
    torch.save(best_state, reasoner_path)
    torch.save(best_state, best_checkpoint_path)

    log_dir = ensure_dir(get_logs_dir(dataset_name, model_name, run_name, seed))
    write_epoch_logs(log_dir, epoch_rows)

    full_outputs = run_reasoner_full(reasoner, z, base_logits, targets, device)
    full_final = full_outputs["final_logit"].detach().cpu()
    base_cpu = base_logits.detach().cpu()
    y_cpu = y.detach().cpu()
    train_cpu = train_mask.detach().cpu()
    residual_diag = residual_diagnostics(
        full_final,
        base_cpu,
        max_allowed_shift=rc.get("max_allowed_shift", 0.2),
    )
    latent_diag = latent_alignment_diagnostics(full_outputs, targets)
    ranking_diag = ranking_gap_diagnostics(full_final, base_cpu, y_cpu, train_cpu)
    correction_diag = correction_rate_diagnostics(full_final, base_cpu, y_cpu, test_mask.detach().cpu())
    gate_diag: dict[str, object] = {}
    if rc.get("log_relation_gates", False):
        gate_diag = write_relation_gate_diagnostics(
            log_dir=log_dir,
            outputs=full_outputs,
            relation_names=_relation_list(rc.get("relation_names")),
            final_logit=full_final,
            base_logits=base_cpu,
            y=y_cpu,
            seed=seed,
            max_allowed_shift=rc.get("max_allowed_shift", 0.2),
        )
    judge_diag: dict[str, object] = {}
    if rc.get("use_llm_judge", False):
        judge_diag = write_llm_judge_diagnostics(
            log_dir=log_dir,
            outputs=full_outputs,
            judge_mask=targets.get("judge_mask", torch.zeros(data.x.shape[0], dtype=torch.bool)),
            judge_features=targets.get("judge_features", torch.zeros(data.x.shape[0], rc.get("judge_feature_dim", 0))),
            max_allowed_shift=rc.get("max_allowed_shift", 0.2),
        )

    stage3_config = {
        "seed": seed,
        "dataset": dataset_name,
        "split": config["dataset"].get("split_mode", "supervised"),
        "base_artifact_path": str(checkpoint_path),
        "stage2_run_name": stage2_run_names[0],
        "stage2_run_names": stage2_run_names,
        "accepted_err_path": str(get_err_cache_dir(dataset_name, model_name, stage2_run_names[0], seed) / "accepted_err.jsonl"),
        "teacher_latent_path": rc.get("teacher_latents_path"),
        "loss_mode": rc.get("loss_mode", "cvscd"),
        "lambda_cve": rc.get("lambda_cve", 0.3),
        "lambda_latent": rc.get("lambda_latent", 0.2),
        "lambda_intervene": rc.get("lambda_intervene", 0.05),
        "lambda_pairrank": rc.get("lambda_pairrank", 0.2),
        "latent_dim": rc.get("latent_dim", 256),
        "use_relation_features": rc.get("use_relation_features", False),
        "relation_features_path": rc.get("relation_features_path"),
        "relation_dim": rc.get("relation_dim", 0),
        "relation_hidden_dim": rc.get("relation_hidden_dim", 32),
        "relation_fusion_mode": rc.get("relation_fusion_mode", "concat"),
        "relation_names": rc.get("relation_names", []),
        "relation_stat_dim": rc.get("relation_stat_dim", 0),
        "anchor_relation": rc.get("anchor_relation"),
        "optional_relations": rc.get("optional_relations", []),
        "lambda_gate_sparse": rc.get("lambda_gate_sparse", 0.0),
        "relation_dropout": rc.get("relation_dropout", 0.0),
        "log_relation_gates": rc.get("log_relation_gates", False),
        "gate_hidden_dim": rc.get("gate_hidden_dim", 64),
        "gate_temperature": rc.get("gate_temperature", 1.0),
        "gate_warmup_epochs": rc.get("gate_warmup_epochs", 0),
        "use_llm_judge": rc.get("use_llm_judge", False),
        "judge_features_path": rc.get("judge_features_path"),
        "judge_feature_meta_path": rc.get("judge_feature_meta_path"),
        "judge_feature_dim": rc.get("judge_feature_dim", 0),
        "fusion_mode": rc.get("fusion_mode", "none"),
        "lambda_judge": rc.get("lambda_judge", 0.1),
        "lambda_llm_reg": rc.get("lambda_llm_reg", 0.001),
        "llm_delta_scale": rc.get("llm_delta_scale", 1.0),
        "llm_alpha_reg": rc.get("llm_alpha_reg", 0.0),
        "lambda_alpha": rc.get("lambda_alpha", rc.get("llm_alpha_reg", 0.0)),
        "alpha_max": rc.get("alpha_max", 1.0),
        "strength_aware_alpha": rc.get("strength_aware_alpha", False),
        "init_reasoner_checkpoint": rc.get("init_reasoner_checkpoint"),
        "pairrank_temperature": rc.get("pairrank_temperature", 0.1),
        "delta_scale": rc.get("delta_scale", 2.0),
        "max_allowed_shift": rc.get("max_allowed_shift", 0.2),
        "checkpoint_selection_metric": select_metric,
        "git_commit": get_git_hash(),
        "runtime_device": device_info,
        "teacher_projection": {
            "student_latent_dim": rc.get("latent_dim", 256),
            "teacher_latent_dim": teacher_latent_meta.get("teacher_latent_dim", 0),
            "method": "identity_or_truncate_or_zero_pad_in_loss",
        },
    }
    if teacher_latent_meta:
        stage3_config["teacher_latent_meta"] = teacher_latent_meta
    if relation_feature_meta:
        stage3_config["relation_feature_meta"] = relation_feature_meta
    if judge_feature_meta:
        stage3_config["judge_feature_meta"] = judge_feature_meta
    if gate_diag:
        stage3_config["relation_gate_diagnostics"] = gate_diag
    if judge_diag:
        stage3_config["llm_judge_diagnostics"] = judge_diag

    run_info = {
        "config": config,
        "seed": seed,
        "run_name": run_name,
        "source_stage2_run_name": stage2_run_names[0],
        "source_stage2_run_names": stage2_run_names,
        "source_err_cache_stats": source_stats,
        "git_hash": get_git_hash(),
        "runtime_device": device_info,
        "reasoner_checkpoint": str(reasoner_path),
        "best_checkpoint": str(best_checkpoint_path),
        "test_metrics": test_metrics,
        "epochs_trained": epoch,
        "elapsed_seconds": elapsed,
        "select_metric": select_metric,
        "best_val_score": best_val_score,
        "best_val_auprc": best_val_auprc,
        "best_epoch": best_epoch_idx,
        "rho": rho,
        "gate_mode": gate_mode,
        "delta_scale": delta_scale,
        "loss_mode": rc.get("loss_mode", "cvscd"),
        "lambda_cve": rc.get("lambda_cve", 0.3),
        "lambda_latent": rc.get("lambda_latent", 0.2),
        "lambda_intervene": rc.get("lambda_intervene", 0.05),
        "lambda_pairrank": rc.get("lambda_pairrank", 0.2),
        "pairrank_temperature": rc.get("pairrank_temperature", 0.1),
        "latent_dim": rc.get("latent_dim", 256),
        "use_relation_features": rc.get("use_relation_features", False),
        "relation_features_path": rc.get("relation_features_path"),
        "relation_dim": rc.get("relation_dim", 0),
        "relation_hidden_dim": rc.get("relation_hidden_dim", 32),
        "relation_fusion_mode": rc.get("relation_fusion_mode", "concat"),
        "relation_names": rc.get("relation_names", []),
        "relation_stat_dim": rc.get("relation_stat_dim", 0),
        "anchor_relation": rc.get("anchor_relation"),
        "optional_relations": rc.get("optional_relations", []),
        "lambda_gate_sparse": rc.get("lambda_gate_sparse", 0.0),
        "relation_dropout": rc.get("relation_dropout", 0.0),
        "log_relation_gates": rc.get("log_relation_gates", False),
        "gate_hidden_dim": rc.get("gate_hidden_dim", 64),
        "gate_temperature": rc.get("gate_temperature", 1.0),
        "gate_warmup_epochs": rc.get("gate_warmup_epochs", 0),
        "use_llm_judge": rc.get("use_llm_judge", False),
        "judge_features_path": rc.get("judge_features_path"),
        "judge_feature_meta_path": rc.get("judge_feature_meta_path"),
        "judge_feature_dim": rc.get("judge_feature_dim", 0),
        "fusion_mode": rc.get("fusion_mode", "none"),
        "lambda_judge": rc.get("lambda_judge", 0.1),
        "lambda_llm_reg": rc.get("lambda_llm_reg", 0.001),
        "llm_delta_scale": rc.get("llm_delta_scale", 1.0),
        "llm_alpha_reg": rc.get("llm_alpha_reg", 0.0),
        "lambda_alpha": rc.get("lambda_alpha", rc.get("llm_alpha_reg", 0.0)),
        "alpha_max": rc.get("alpha_max", 1.0),
        "strength_aware_alpha": rc.get("strength_aware_alpha", False),
        "init_reasoner_checkpoint": rc.get("init_reasoner_checkpoint"),
        "relation_feature_meta": relation_feature_meta,
        "judge_feature_meta": judge_feature_meta,
        "lambda_evi": rc.get("lambda_evi", 0.5),
        "lambda_det": rc.get("lambda_det", 1.0),
        "lambda_err": rc.get("lambda_err", 0.3),
        "lambda_signed": rc.get("lambda_signed", 0.1),
        "lambda_corr": rc.get("lambda_corr", 0.1),
        "lambda_direction": rc.get("lambda_direction", 1.0),
        "base_correct_anchor_weight": rc.get("base_correct_anchor_weight", None),
        "uncertain_anchor_weight": rc.get("uncertain_anchor_weight", None),
        "decrease_anchor_weight": rc.get("decrease_anchor_weight", None),
        "fp_anchor_weight": rc.get("fp_anchor_weight", None),
        "non_fn_positive_anchor_weight": rc.get("non_fn_positive_anchor_weight", None),
        "non_accepted_anchor_weight": rc.get("non_accepted_anchor_weight", None),
        "enable_fp_negative_correction": rc.get("enable_fp_negative_correction", False),
        "residual_cap": rho * delta_scale,
        "correction_weight": rc.get("correction_weight", 2.0),
        "signed_margin": rc.get("signed_margin", 0.2),
        "correction_margin": rc.get("correction_margin", 0.05),
        "anchor_weight": rc.get("anchor_weight", 0.001),
        "max_shift_penalty_weight": rc.get("max_shift_penalty_weight", 0.0),
        "max_abs_shift": rc.get("max_abs_shift", 2.0),
        "max_allowed_shift": rc.get("max_allowed_shift", 0.3),
        "threshold_mode": rc.get("threshold_mode", "fixed"),
        "num_accepted_err": len(accepted_errs),
        "num_teacher_latents": teacher_latent_meta.get("teacher_latent_count", 0),
        "last_loss_dict": last_loss_dict,
        "best_loss_dict": best_loss_dict,
        "residual_diagnostics": residual_diag,
        "latent_alignment_diagnostics": latent_diag,
        "ranking_gap_diagnostics": ranking_diag,
        "correction_rate_diagnostics": correction_diag,
        "relation_gate_diagnostics": gate_diag,
        "llm_judge_diagnostics": judge_diag,
        "fn_correction_rate": correction_diag["fn_correction_rate"],
        "fp_correction_rate": correction_diag["fp_correction_rate"],
    }

    with open(log_dir / "stage3_config.json", "w") as f:
        json.dump(stage3_config, f, indent=2)
    with open(log_dir / "best_epoch.json", "w") as f:
        json.dump(
            {
                "epoch": best_epoch_idx,
                "select_metric": select_metric,
                "best_val_score": best_val_score,
                "best_val_auprc": best_val_auprc,
            },
            f,
            indent=2,
        )
    with open(log_dir / "final_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(log_dir / "residual_diagnostics.json", "w") as f:
        json.dump(residual_diag, f, indent=2)
    with open(log_dir / "latent_alignment_diagnostics.json", "w") as f:
        json.dump(latent_diag, f, indent=2)
    with open(log_dir / "ranking_gap_diagnostics.json", "w") as f:
        json.dump(ranking_diag, f, indent=2)
    with open(log_dir / "correction_rate_diagnostics.json", "w") as f:
        json.dump(correction_diag, f, indent=2)
    with open(log_dir / "stage3.json", "w") as f:
        json.dump(run_info, f, indent=2)

    results_dir = ensure_dir(get_results_dir(dataset_name, model_name, run_name, seed))

    with open(results_dir / "stage3_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(results_dir / "final_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nReasoner saved to: {reasoner_path}")
    print(f"Metrics saved to: {log_dir / 'stage3.json'}")


if __name__ == "__main__":
    main()
