from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.schema import ERR, ReasoningChannel
from evidence.vocab import encode_err_targets, encode_reasoning, get_evidence_slots, get_reason_types
from models.gnn import build_detector
from models.reasoner import EvidenceReasoner
from training.losses import compute_reasoner_loss
from training.metrics import compute_metrics
from utils.tensorboard import create_logger
from utils.paths import get_checkpoint_dir, get_logs_dir, get_results_dir, get_err_cache_dir, get_base_checkpoint_path, ensure_dir


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


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


def prepare_targets(
    num_nodes: int,
    accepted_errs: list[dict],
    evidence_cards: dict[int, dict],
    train_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    num_slots = len(get_evidence_slots())
    num_types = len(get_reason_types())

    evidence_token_ids = torch.zeros(num_nodes, num_slots, dtype=torch.long)
    risk_type_ids = torch.zeros(num_nodes, dtype=torch.long)
    pos_masks = torch.zeros(num_nodes, num_slots, dtype=torch.float)
    neg_masks = torch.zeros(num_nodes, num_slots, dtype=torch.float)
    accepted_mask = torch.zeros(num_nodes, dtype=torch.float)

    for err_data in accepted_errs:
        node_id = err_data["node_id"]
        if node_id >= num_nodes:
            continue

        err = ERR(
            node_id=node_id,
            risk_type=err_data["risk_type"],
            supporting_evidence=err_data["supporting_evidence"],
            counter_evidence=err_data["counter_evidence"],
            summary=err_data.get("summary", ""),
        )

        targets = encode_err_targets(err)
        risk_type_ids[node_id] = targets["risk_type_id"]
        pos_masks[node_id] = targets["pos_mask"]
        neg_masks[node_id] = targets["neg_mask"]
        accepted_mask[node_id] = 1.0

    for node_id, card in evidence_cards.items():
        if node_id >= num_nodes:
            continue
        reasoning = card.get("reasoning", {})
        evidence_token_ids[node_id] = encode_reasoning(reasoning)

    return {
        "evidence_token_ids": evidence_token_ids,
        "risk_type_id": risk_type_ids,
        "pos_mask": pos_masks,
        "neg_mask": neg_masks,
        "accepted_mask": accepted_mask,
    }


def train_one_epoch(reasoner, z, base_logits, targets, train_mask, optimizer, config):
    reasoner.train()
    device = z.device

    evi_ids = targets["evidence_token_ids"].to(device)
    risk_type_id = targets["risk_type_id"].to(device)
    pos_mask = targets["pos_mask"].to(device)
    neg_mask = targets["neg_mask"].to(device)
    accepted_mask = targets["accepted_mask"].to(device)

    z_train = z[train_mask]
    base_logit_train = base_logits[train_mask]
    evi_ids_train = evi_ids[train_mask]
    y_train = torch.zeros(train_mask.sum(), dtype=torch.float, device=device)

    outputs = reasoner(z_train, base_logit_train, evi_ids_train)

    accepted_train = accepted_mask[train_mask]
    targets_train = {
        "risk_type_id": risk_type_id[train_mask],
        "pos_mask": pos_mask[train_mask],
        "neg_mask": neg_mask[train_mask],
    }

    lambda_evi = config.get("reasoner", {}).get("lambda_evi", 0.5)
    use_type_loss = config.get("reasoner", {}).get("use_type_loss", True)
    use_evidence_loss = config.get("reasoner", {}).get("use_evidence_loss", True)

    loss, loss_dict = compute_reasoner_loss(
        outputs, y_train, targets_train, accepted_train,
        lambda_evi=lambda_evi,
        use_type_loss=use_type_loss,
        use_evidence_loss=use_evidence_loss,
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

    outputs = reasoner(z_mask, base_logit_mask, evi_ids_mask)

    y_np = y[mask].cpu().numpy()
    prob = torch.sigmoid(outputs["final_logit"]).cpu().numpy()

    metrics = compute_metrics(y_np, prob)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_name", type=str, default="rule")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    seed = args.seed or config["train"]["seed"]
    model_name = config["model"]["name"]
    run_name = args.run_name

    torch.manual_seed(seed)
    device = torch.device(config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    if args.debug:
        print("[DEBUG] Using tiny synthetic graph, 3 epochs")
        data = load_fraud_dataset("tiny", seed=seed)
        epochs = 3
    else:
        split_mode = config["dataset"].get("split_mode", "supervised")
        train_ratio = config["dataset"].get("train_ratio", 0.7)
        val_test_ratio = config["dataset"].get("val_test_ratio", [1, 2])

        data = load_fraud_dataset(
            dataset_name, path=dataset_path, seed=seed,
            split_mode=split_mode, train_ratio=train_ratio, val_test_ratio=val_test_ratio
        )
        epochs = config["train"].get("epochs", 200)

    base_model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
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
        base_logits = output.logits
        z = output.embeddings

    err_cache_dir = get_err_cache_dir(dataset_name, model_name, run_name, seed)
    cards_path = err_cache_dir / "evidence_cards.jsonl"
    accepted_path = err_cache_dir / "accepted_err.jsonl"

    if cards_path.exists() and accepted_path.exists():
        evidence_cards = load_evidence_cards(cards_path)
        accepted_errs = load_accepted_errs(accepted_path)
        print(f"Loaded {len(evidence_cards)} cards, {len(accepted_errs)} accepted ERR")
    else:
        print(f"Warning: No ERR cache found, using empty targets")
        evidence_cards = {}
        accepted_errs = []

    targets = prepare_targets(
        num_nodes=data.x.shape[0],
        accepted_errs=accepted_errs,
        evidence_cards=evidence_cards,
        train_mask=data.train_mask,
    )

    hidden_dim = z.shape[1]
    rho = config.get("reasoner", {}).get("rho", 0.3)
    reasoner = EvidenceReasoner(
        z_dim=hidden_dim,
        hidden_dim=config.get("reasoner", {}).get("hidden_dim", 128),
        rho=rho,
    ).to(device)

    optimizer = torch.optim.Adam(
        reasoner.parameters(),
        lr=config["train"].get("lr", 0.001),
        weight_decay=config["train"].get("weight_decay", 0.0005),
    )

    patience = config["train"].get("patience", 30)
    best_val_auc = 0.0
    patience_counter = 0
    best_state = None

    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)

    start_time = time.time()

    tb_logger = create_logger(dataset_name, model_name, seed, "stage3")

    for epoch in range(1, epochs + 1):
        loss, loss_dict = train_one_epoch(reasoner, z, base_logits, targets, train_mask, optimizer, config)

        val_metrics = evaluate(reasoner, z, base_logits, targets, val_mask, y, device)

        tb_logger.log_scalar("train/loss", loss, epoch)
        tb_logger.log_scalars("train/loss_components", loss_dict, epoch)
        tb_logger.log_metrics(val_metrics, epoch, prefix="val")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}")

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = {k: v.cpu().clone() for k, v in reasoner.state_dict().items()}
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
    torch.save(best_state, reasoner_path)

    log_dir = ensure_dir(get_logs_dir(dataset_name, model_name, run_name, seed))

    run_info = {
        "config": config,
        "seed": seed,
        "run_name": run_name,
        "git_hash": get_git_hash(),
        "reasoner_checkpoint": str(reasoner_path),
        "test_metrics": test_metrics,
        "epochs_trained": epoch,
        "elapsed_seconds": elapsed,
        "rho": rho,
        "num_accepted_err": len(accepted_errs),
    }

    with open(log_dir / "stage3.json", "w") as f:
        json.dump(run_info, f, indent=2)

    results_dir = ensure_dir(get_results_dir(dataset_name, model_name, run_name, seed))

    with open(results_dir / "stage3_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nReasoner saved to: {reasoner_path}")
    print(f"Metrics saved to: {log_dir / 'stage3.json'}")


if __name__ == "__main__":
    main()
