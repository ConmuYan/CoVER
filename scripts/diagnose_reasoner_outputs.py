"""Comprehensive reasoner diagnosis for CoVER-FD.

Compares base (stage-1) and reasoner (stage-3) outputs, reports gate/residual
statistics, threshold behavior, evidence supervision stats, and loss breakdown.

Usage:
    python scripts/diagnose_reasoner_outputs.py \
        --config configs/yelpchi_bwgnn.yaml \
        --run_name qwen \
        --seed 123
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.vocab import (
    encode_err_targets,
    encode_reasoning,
    get_evidence_slots,
    get_reason_types,
)
from models.gnn import build_detector
from models.reasoner import EvidenceReasoner
from training.losses import compute_reasoner_loss
from utils.paths import (
    ensure_dir,
    get_base_checkpoint_path,
    get_err_cache_dir,
    get_reasoner_checkpoint_path,
    get_split_meta_path,
)
from utils.threshold import evaluate_with_threshold, find_best_threshold


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _stat_dict(arr: np.ndarray) -> dict[str, float]:
    """Return mean/std/min/max for a numpy array."""
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _tensor_stat_dict(t: torch.Tensor) -> dict[str, float]:
    return _stat_dict(t.detach().cpu().float().numpy())


def build_split_info(data, split_meta: dict | None) -> dict:
    """Section A: Split info."""
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    num_train = int(train_mask.sum())
    num_val = int(val_mask.sum())
    num_test = int(test_mask.sum())
    num_total = len(y)

    pos_rate_train = float(y[train_mask].mean()) if num_train else 0.0
    pos_rate_val = float(y[val_mask].mean()) if num_val else 0.0
    pos_rate_test = float(y[test_mask].mean()) if num_test else 0.0
    global_pos_rate = float(y.mean()) if num_total else 0.0

    rates = [pos_rate_train, pos_rate_val, pos_rate_test]
    max_pos_rate_gap = max(rates) - min(rates)
    relative_pos_rate_gap = (
        max_pos_rate_gap / global_pos_rate if global_pos_rate > 0 else 0.0
    )

    stratified = False
    if split_meta is not None:
        stratified = split_meta.get("stratified", False)

    return {
        "stratified": stratified,
        "num_nodes": num_total,
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "pos_rate_train": pos_rate_train,
        "pos_rate_val": pos_rate_val,
        "pos_rate_test": pos_rate_test,
        "global_pos_rate": global_pos_rate,
        "max_pos_rate_gap": max_pos_rate_gap,
        "relative_pos_rate_gap": relative_pos_rate_gap,
    }


def build_logit_shift(
    base_logit: np.ndarray,
    final_logit: np.ndarray,
    mask: np.ndarray,
    split_name: str,
) -> dict:
    """Section B helpers: logit / probability shift for one split."""
    bl = base_logit[mask]
    fl = final_logit[mask]
    delta = fl - bl

    sig_base = 1.0 / (1.0 + np.exp(-bl))
    sig_final = 1.0 / (1.0 + np.exp(-fl))

    return {
        f"{split_name}_base_logit": _stat_dict(bl),
        f"{split_name}_final_logit": _stat_dict(fl),
        f"{split_name}_logit_shift": {
            **_stat_dict(delta),
        },
        f"{split_name}_sigmoid_base": _stat_dict(sig_base),
        f"{split_name}_sigmoid_final": _stat_dict(sig_final),
    }


def build_ranking_correlation(
    base_logit: np.ndarray,
    final_logit: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    split_name: str,
) -> dict:
    """Section C: ranking correlation for one split."""
    bl = base_logit[mask]
    fl = final_logit[mask]
    ym = y[mask]

    from numpy import corrcoef

    pearson = float(corrcoef(bl, fl)[0, 1]) if bl.size > 1 else 0.0

    spearman = None
    try:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(bl, fl).correlation) if bl.size > 1 else 0.0
    except ImportError:
        pass

    from sklearn.metrics import average_precision_score, roc_auc_score

    def _safe_auc(labels, scores):
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return 0.0

    def _safe_auprc(labels, scores):
        try:
            return float(average_precision_score(labels, scores))
        except ValueError:
            return 0.0

    sig_base = 1.0 / (1.0 + np.exp(-bl))
    sig_final = 1.0 / (1.0 + np.exp(-fl))

    return {
        f"{split_name}_pearson": pearson,
        f"{split_name}_spearman": spearman,
        f"{split_name}_roc_auc_base": _safe_auc(ym, sig_base),
        f"{split_name}_roc_auc_final": _safe_auc(ym, sig_final),
        f"{split_name}_auprc_base": _safe_auprc(ym, sig_base),
        f"{split_name}_auprc_final": _safe_auprc(ym, sig_final),
    }


def build_threshold_behavior(
    base_prob_val: np.ndarray,
    final_prob_val: np.ndarray,
    y_val: np.ndarray,
    base_prob_test: np.ndarray,
    final_prob_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Section D: threshold behavior."""
    results: dict = {}

    for label, prob_test, y_t in [
        ("base", base_prob_test, y_test),
        ("final", final_prob_test, y_test),
    ]:
        pred05 = (prob_test >= 0.5).astype(int)
        results[f"{label}_pos_rate_at_0.5"] = (
            float(pred05.mean()) if pred05.size else 0.0
        )

    for metric_name in ("f1", "macro_f1"):
        thr_base, score_base, pr_base = find_best_threshold(
            y_val, base_prob_val, metric=metric_name,
        )
        thr_final, score_final, pr_final = find_best_threshold(
            y_val, final_prob_val, metric=metric_name,
        )

        eval_base = evaluate_with_threshold(y_test, base_prob_test, thr_base)
        eval_final = evaluate_with_threshold(y_test, final_prob_test, thr_final)

        results[f"val_{metric_name}_threshold_base"] = thr_base
        results[f"val_{metric_name}_threshold_final"] = thr_final
        results[f"val_{metric_name}_score_base"] = score_base
        results[f"val_{metric_name}_score_final"] = score_final
        results[f"val_{metric_name}_pos_rate_base"] = pr_base
        results[f"val_{metric_name}_pos_rate_final"] = pr_final

        results[f"test_{metric_name}_base"] = {
            "f1": eval_base["f1"],
            "macro_f1": eval_base["macro_f1"],
            "precision": eval_base["precision"],
            "recall": eval_base["recall"],
            "positive_prediction_rate": eval_base["positive_prediction_rate"],
        }
        results[f"test_{metric_name}_final"] = {
            "f1": eval_final["f1"],
            "macro_f1": eval_final["macro_f1"],
            "precision": eval_final["precision"],
            "recall": eval_final["recall"],
            "positive_prediction_rate": eval_final["positive_prediction_rate"],
        }

    results["threshold_selected_on"] = "validation"

    return results


def build_gate_residual_stats(outputs: dict[str, torch.Tensor]) -> dict:
    """Section E: residual / gate statistics."""
    result: dict = {}

    gate = outputs["gate"]
    residual_raw = outputs["residual_raw"]
    rho = outputs["rho"]

    result["gate"] = _tensor_stat_dict(gate)
    result["residual_raw"] = _tensor_stat_dict(residual_raw)
    result["rho"] = float(rho.item())

    return result


def build_evidence_supervision_stats(
    accepted_errs: list[dict],
    train_mask_np: np.ndarray,
    config: dict,
) -> dict:
    """Section F: evidence supervision stats."""
    num_train = int(train_mask_np.sum())
    num_accepted = len(accepted_errs)

    accepted_node_ids = {err["node_id"] for err in accepted_errs}
    train_indices = set(np.where(train_mask_np)[0].tolist())
    covered = accepted_node_ids & train_indices
    coverage = len(covered) / num_train if num_train > 0 else 0.0

    risk_type_dist = Counter()
    supporting_dist = Counter()
    counter_dist = Counter()

    for err in accepted_errs:
        risk_type_dist[err.get("risk_type", "unknown")] += 1
        for s in err.get("supporting_evidence", []):
            supporting_dist[s] += 1
        for c in err.get("counter_evidence", []):
            counter_dist[c] += 1

    weak_count = sum(
        count
        for rt, count in risk_type_dist.items()
        if "weak" in rt.lower() or "uncertain" in rt.lower()
    )
    total_for_weak = sum(risk_type_dist.values())
    weak_ratio = weak_count / total_for_weak if total_for_weak > 0 else 0.0

    trace_size = config.get("evidence", {}).get("trace_size", 0)

    return {
        "accepted_err_count": num_accepted,
        "accepted_err_over_train_ratio": (
            num_accepted / num_train if num_train > 0 else 0.0
        ),
        "risk_type_distribution": dict(risk_type_dist.most_common()),
        "supporting_evidence_distribution": dict(supporting_dist.most_common(20)),
        "counter_evidence_distribution": dict(counter_dist.most_common(20)),
        "weak_or_uncertain_ratio": weak_ratio,
        "trace_size": trace_size,
        "coverage_of_accepted_err_over_train": coverage,
    }


def build_loss_stats(
    reasoner: EvidenceReasoner,
    z: torch.Tensor,
    base_logits: torch.Tensor,
    targets: dict[str, torch.Tensor],
    train_mask: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    config: dict,
) -> dict:
    """Section G: loss stats computed on the training set."""
    reasoner.eval()

    evi_ids = targets["evidence_token_ids"].to(device)
    risk_type_id = targets["risk_type_id"].to(device)
    pos_mask = targets["pos_mask"].to(device)
    neg_mask = targets["neg_mask"].to(device)
    accepted_mask = targets["accepted_mask"].to(device)

    z_train = z[train_mask]
    bl_train = base_logits[train_mask]
    evi_train = evi_ids[train_mask]
    y_train = y[train_mask].float().to(device)

    with torch.no_grad():
        outputs = reasoner(z_train, bl_train, evi_train)

    accepted_train = accepted_mask[train_mask]
    targets_train = {
        "risk_type_id": risk_type_id[train_mask],
        "pos_mask": pos_mask[train_mask],
        "neg_mask": neg_mask[train_mask],
    }

    lambda_evi = config.get("reasoner", {}).get("lambda_evi", 0.5)
    use_type_loss = config.get("reasoner", {}).get("use_type_loss", True)
    use_evidence_loss = config.get("reasoner", {}).get("use_evidence_loss", True)

    _, loss_dict = compute_reasoner_loss(
        outputs,
        y_train,
        targets_train,
        accepted_train,
        lambda_evi=lambda_evi,
        use_type_loss=use_type_loss,
        use_evidence_loss=use_evidence_loss,
    )

    task_loss = loss_dict["task"]
    type_loss = loss_dict["type"]
    pos_loss = loss_dict["pos"]
    neg_loss = loss_dict["neg"]
    evidence_loss = type_loss + pos_loss + neg_loss

    return {
        "task_loss": task_loss,
        "type_loss": type_loss,
        "pos_mask_loss": pos_loss,
        "neg_mask_loss": neg_loss,
        "evidence_loss": evidence_loss,
        "evidence_over_task_ratio": (
            evidence_loss / task_loss if task_loss > 0 else 0.0
        ),
    }


def render_markdown(report: dict) -> str:
    lines: list[str] = []

    lines.append("# Reasoner Diagnosis Report")
    lines.append("")
    lines.append(
        f"Dataset: {report['dataset']} | Model: {report['model']} "
        f"| Run: {report['run_name']} | Seed: {report['seed']}"
    )
    lines.append(f"Generated: {report.get('git_hash', 'unknown')}")
    lines.append("")

    si = report.get("split_info", {})
    lines.append("## A. Split Info")
    lines.append(f"- Stratified: {si.get('stratified', 'N/A')}")
    lines.append(f"- Train/Val/Test: {si.get('num_train', 0)} / {si.get('num_val', 0)} / {si.get('num_test', 0)}")
    lines.append(f"- Pos rate (train/val/test): {si.get('pos_rate_train', 0):.4f} / {si.get('pos_rate_val', 0):.4f} / {si.get('pos_rate_test', 0):.4f}")
    lines.append(f"- Global pos rate: {si.get('global_pos_rate', 0):.4f}")
    lines.append(f"- Max pos rate gap: {si.get('max_pos_rate_gap', 0):.4f}")
    lines.append(f"- Relative pos rate gap: {si.get('relative_pos_rate_gap', 0):.4f}")
    lines.append("")

    ls = report.get("logit_shift", {})
    lines.append("## B. Logit / Probability Shift")
    for split in ("val", "test"):
        bl = ls.get(f"{split}_base_logit", {})
        fl = ls.get(f"{split}_final_logit", {})
        delta = ls.get(f"{split}_logit_shift", {})
        sb = ls.get(f"{split}_sigmoid_base", {})
        sf = ls.get(f"{split}_sigmoid_final", {})
        lines.append(f"### {split.upper()}")
        lines.append(f"- base_logit mean={bl.get('mean', 0):.4f} std={bl.get('std', 0):.4f}")
        lines.append(f"- final_logit mean={fl.get('mean', 0):.4f} std={fl.get('std', 0):.4f}")
        lines.append(f"- shift mean={delta.get('mean', 0):.4f} std={delta.get('std', 0):.4f} min={delta.get('min', 0):.4f} max={delta.get('max', 0):.4f}")
        lines.append(f"- sigmoid(base) mean={sb.get('mean', 0):.4f} std={sb.get('std', 0):.4f}")
        lines.append(f"- sigmoid(final) mean={sf.get('mean', 0):.4f} std={sf.get('std', 0):.4f}")
        lines.append("")

    rc = report.get("ranking_correlation", {})
    lines.append("## C. Ranking Correlation")
    for split in ("val", "test"):
        lines.append(f"### {split.upper()}")
        lines.append(f"- Pearson: {rc.get(f'{split}_pearson', 'N/A'):.4f}" if rc.get(f"{split}_pearson") is not None else f"- Pearson: N/A")
        sp = rc.get(f"{split}_spearman")
        lines.append(f"- Spearman: {sp:.4f}" if sp is not None else "- Spearman: (scipy not available)")
        lines.append(f"- ROC-AUC base: {rc.get(f'{split}_roc_auc_base', 0):.4f} | final: {rc.get(f'{split}_roc_auc_final', 0):.4f}")
        lines.append(f"- AUPRC base: {rc.get(f'{split}_auprc_base', 0):.4f} | final: {rc.get(f'{split}_auprc_final', 0):.4f}")
        lines.append("")

    tb = report.get("threshold_behavior", {})
    lines.append("## D. Threshold Behavior")
    lines.append(f"- Base pos rate @0.5: {tb.get('base_pos_rate_at_0.5', 0):.4f}")
    lines.append(f"- Final pos rate @0.5: {tb.get('final_pos_rate_at_0.5', 0):.4f}")
    for metric_name in ("f1", "macro_f1"):
        lines.append(f"### Calibrated on val ({metric_name})")
        lines.append(f"- Base threshold: {tb.get(f'val_{metric_name}_threshold_base', 'N/A')}")
        lines.append(f"- Final threshold: {tb.get(f'val_{metric_name}_threshold_final', 'N/A')}")
        base_m = tb.get(f"test_{metric_name}_base", {})
        final_m = tb.get(f"test_{metric_name}_final", {})
        lines.append(f"- Base test F1={base_m.get('f1', 0):.4f} Macro-F1={base_m.get('macro_f1', 0):.4f} PPR={base_m.get('positive_prediction_rate', 0):.4f}")
        lines.append(f"- Final test F1={final_m.get('f1', 0):.4f} Macro-F1={final_m.get('macro_f1', 0):.4f} PPR={final_m.get('positive_prediction_rate', 0):.4f}")
        lines.append("")
    lines.append(f"- Threshold selected on: {tb.get('threshold_selected_on', 'N/A')}")
    lines.append("")

    gr = report.get("gate_residual", {})
    lines.append("## E. Residual / Gate Statistics")
    gate = gr.get("gate", {})
    lines.append(f"- Gate: mean={gate.get('mean', 0):.6f} std={gate.get('std', 0):.6f} min={gate.get('min', 0):.6f} max={gate.get('max', 0):.6f}")
    rr = gr.get("residual_raw", {})
    lines.append(f"- Residual raw: mean={rr.get('mean', 0):.6f} std={rr.get('std', 0):.6f} min={rr.get('min', 0):.6f} max={rr.get('max', 0):.6f}")
    lines.append(f"- rho: {gr.get('rho', 0):.4f}")

    for split in ("val", "test"):
        sg = gr.get(f"{split}_gate", {})
        sr = gr.get(f"{split}_residual_raw", {})
        if sg:
            lines.append(f"  - {split} gate: mean={sg.get('mean', 0):.6f} std={sg.get('std', 0):.6f}")
        if sr:
            lines.append(f"  - {split} residual_raw: mean={sr.get('mean', 0):.6f} std={sr.get('std', 0):.6f}")
    lines.append("")

    ev = report.get("evidence_supervision", {})
    lines.append("## F. Evidence Supervision Stats")
    lines.append(f"- Accepted ERR count: {ev.get('accepted_err_count', 0)}")
    lines.append(f"- Accepted ERR / train ratio: {ev.get('accepted_err_over_train_ratio', 0):.4f}")
    lines.append(f"- Coverage over train nodes: {ev.get('coverage_of_accepted_err_over_train', 0):.4f}")
    lines.append(f"- Trace size: {ev.get('trace_size', 'N/A')}")
    lines.append(f"- Weak/uncertain ratio: {ev.get('weak_or_uncertain_ratio', 0):.4f}")

    rt_dist = ev.get("risk_type_distribution", {})
    if rt_dist:
        lines.append("- Risk type distribution:")
        for rt, count in sorted(rt_dist.items(), key=lambda x: -x[1]):
            lines.append(f"  - {rt}: {count}")

    sup_dist = ev.get("supporting_evidence_distribution", {})
    if sup_dist:
        lines.append("- Top supporting evidence:")
        for s, count in list(sup_dist.items())[:10]:
            lines.append(f"  - {s}: {count}")

    cnt_dist = ev.get("counter_evidence_distribution", {})
    if cnt_dist:
        lines.append("- Top counter evidence:")
        for c, count in list(cnt_dist.items())[:10]:
            lines.append(f"  - {c}: {count}")
    lines.append("")

    lo = report.get("loss_stats", {})
    lines.append("## G. Loss Stats (train set)")
    lines.append(f"- task_loss: {lo.get('task_loss', 0):.6f}")
    lines.append(f"- type_loss: {lo.get('type_loss', 0):.6f}")
    lines.append(f"- pos_mask_loss: {lo.get('pos_mask_loss', 0):.6f}")
    lines.append(f"- neg_mask_loss: {lo.get('neg_mask_loss', 0):.6f}")
    lines.append(f"- evidence_loss: {lo.get('evidence_loss', 0):.6f}")
    lines.append(f"- evidence/task ratio: {lo.get('evidence_over_task_ratio', 0):.4f}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Diagnose reasoner outputs")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    seed = args.seed or config["train"]["seed"]
    model_name = config["model"]["name"]
    run_name = args.run_name

    torch.manual_seed(seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

    split_mode = config["dataset"].get("split_mode", "supervised")
    train_ratio = config["dataset"].get("train_ratio", 0.7)
    val_test_ratio = config["dataset"].get("val_test_ratio", [1, 2])

    data = load_fraud_dataset(
        dataset_name,
        path=dataset_path,
        seed=seed,
        split_mode=split_mode,
        train_ratio=train_ratio,
        val_test_ratio=val_test_ratio,
    )

    y = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    num_nodes = data.x.shape[0]

    y_np = y.numpy()
    train_np = train_mask.numpy()
    val_np = val_mask.numpy()
    test_np = test_mask.numpy()

    base_model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    ).to(device)

    base_path = get_base_checkpoint_path(dataset_name, model_name, seed)
    if base_path.exists():
        state = torch.load(base_path, weights_only=True)
        base_model.load_state_dict(state)
        print(f"Loaded base checkpoint: {base_path}")
    else:
        print(f"WARNING: base checkpoint not found at {base_path}")

    base_model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        output = base_model(x, edge_index, return_output=True)
        base_logits = output.logits
        z = output.embeddings

    err_cache_dir = get_err_cache_dir(dataset_name, model_name, run_name, seed)
    evidence_cards = _load_jsonl(err_cache_dir / "evidence_cards.jsonl")
    accepted_errs = _load_jsonl(err_cache_dir / "accepted_err.jsonl")

    evidence_cards_dict: dict[int, dict] = {
        c["node_id"]: c for c in evidence_cards
    }

    print(f"Evidence cards: {len(evidence_cards)}, Accepted ERRs: {len(accepted_errs)}")

    num_slots = len(get_evidence_slots())
    evidence_token_ids = torch.zeros(num_nodes, num_slots, dtype=torch.long)
    for node_id, card in evidence_cards_dict.items():
        if node_id < num_nodes:
            reasoning = card.get("reasoning", {})
            evidence_token_ids[node_id] = encode_reasoning(reasoning)

    reason_types = get_reason_types()
    accepted_mask = torch.zeros(num_nodes, dtype=torch.float)
    risk_type_ids = torch.zeros(num_nodes, dtype=torch.long)
    pos_masks = torch.zeros(num_nodes, num_slots, dtype=torch.float)
    neg_masks = torch.zeros(num_nodes, num_slots, dtype=torch.float)

    for err_data in accepted_errs:
        nid = err_data["node_id"]
        if nid >= num_nodes:
            continue
        from evidence.schema import ERR

        err = ERR(
            node_id=nid,
            risk_type=err_data["risk_type"],
            supporting_evidence=err_data["supporting_evidence"],
            counter_evidence=err_data["counter_evidence"],
            summary=err_data.get("summary", ""),
        )
        targets_enc = encode_err_targets(err)
        risk_type_ids[nid] = targets_enc["risk_type_id"]
        pos_masks[nid] = targets_enc["pos_mask"]
        neg_masks[nid] = targets_enc["neg_mask"]
        accepted_mask[nid] = 1.0

    targets = {
        "evidence_token_ids": evidence_token_ids,
        "risk_type_id": risk_type_ids,
        "pos_mask": pos_masks,
        "neg_mask": neg_masks,
        "accepted_mask": accepted_mask,
    }

    reasoner_path = get_reasoner_checkpoint_path(
        dataset_name, model_name, run_name, seed
    )
    if not reasoner_path.exists():
        print(f"ERROR: reasoner checkpoint not found at {reasoner_path}")
        sys.exit(1)

    rho = config.get("reasoner", {}).get("rho", 0.3)
    reasoner = EvidenceReasoner(
        z_dim=z.shape[1],
        hidden_dim=config.get("reasoner", {}).get("hidden_dim", 128),
        rho=rho,
    ).to(device)

    r_state = torch.load(reasoner_path, weights_only=True)
    reasoner.load_state_dict(r_state)
    reasoner.eval()
    print(f"Loaded reasoner checkpoint: {reasoner_path}")

    evidence_token_ids_dev = evidence_token_ids.to(device)

    with torch.no_grad():
        outputs_all = reasoner(
            z, base_logits, evidence_token_ids_dev, return_debug=True,
        )

    final_logit_all = outputs_all["final_logit"]
    gate_all = outputs_all["gate"]
    residual_raw_all = outputs_all["residual_raw"]

    base_logit_np = base_logits.cpu().numpy()
    final_logit_np = final_logit_all.cpu().numpy()

    base_prob_np = 1.0 / (1.0 + np.exp(-base_logit_np))
    final_prob_np = 1.0 / (1.0 + np.exp(-final_logit_np))

    split_meta = _load_json(get_split_meta_path(dataset_name, seed))

    split_info = build_split_info(data, split_meta)

    logit_shift = {}
    logit_shift.update(build_logit_shift(base_logit_np, final_logit_np, val_np, "val"))
    logit_shift.update(build_logit_shift(base_logit_np, final_logit_np, test_np, "test"))

    ranking_corr = {}
    ranking_corr.update(
        build_ranking_correlation(base_logit_np, final_logit_np, y_np, val_np, "val")
    )
    ranking_corr.update(
        build_ranking_correlation(base_logit_np, final_logit_np, y_np, test_np, "test")
    )

    threshold_behavior = build_threshold_behavior(
        base_prob_np[val_np],
        final_prob_np[val_np],
        y_np[val_np],
        base_prob_np[test_np],
        final_prob_np[test_np],
        y_np[test_np],
    )

    gate_residual = build_gate_residual_stats(outputs_all)
    for split_name, split_mask_np in [("val", val_np), ("test", test_np)]:
        gate_residual[f"{split_name}_gate"] = _stat_dict(
            gate_all.cpu().numpy()[split_mask_np].flatten()
        )
        gate_residual[f"{split_name}_residual_raw"] = _stat_dict(
            residual_raw_all.cpu().numpy()[split_mask_np].flatten()
        )

    evidence_stats = build_evidence_supervision_stats(
        accepted_errs, train_np, config,
    )

    loss_stats = build_loss_stats(
        reasoner, z, base_logits, targets, train_mask, y, device, config,
    )

    report = {
        "dataset": dataset_name,
        "model": model_name,
        "run_name": run_name,
        "seed": seed,
        "git_hash": _get_git_hash(),
        "split_info": split_info,
        "logit_shift": logit_shift,
        "ranking_correlation": ranking_corr,
        "threshold_behavior": threshold_behavior,
        "gate_residual": gate_residual,
        "evidence_supervision": evidence_stats,
        "loss_stats": loss_stats,
    }

    report_dir = (
        Path("artifacts")
        / "reports"
        / dataset_name
        / model_name
        / run_name
        / f"seed_{seed}"
    )
    ensure_dir(report_dir)

    json_path = report_dir / "reasoner_diagnosis.json"
    md_path = report_dir / "reasoner_diagnosis.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved: {json_path}")

    md_text = render_markdown(report)
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"Markdown report saved: {md_path}")

    print("\n=== Diagnosis Summary ===")
    si = split_info
    print(f"Split: {si['num_train']}/{si['num_val']}/{si['num_test']} (pos rate gap: {si['max_pos_rate_gap']:.4f})")
    print(f"Accepted ERRs: {len(accepted_errs)} | Coverage: {evidence_stats['coverage_of_accepted_err_over_train']:.2%}")
    print(f"Rho: {rho:.4f} | Gate mean: {gate_residual['gate']['mean']:.6f}")
    print(f"Task loss: {loss_stats['task_loss']:.4f} | Evidence/task ratio: {loss_stats['evidence_over_task_ratio']:.4f}")


if __name__ == "__main__":
    main()
