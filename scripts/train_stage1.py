from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from models.gnn import build_detector
from training.metrics import compute_metrics


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def train_one_epoch(model, data, optimizer, device):
    model.train()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)

    logit_all, _ = model(x, edge_index)
    logit_train = logit_all[train_mask]
    y_train = y[train_mask].float()

    loss = F.binary_cross_entropy_with_logits(logit_train, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask_name, device):
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    mask = getattr(data, f"{mask_name}_mask").to(device)

    logit_all, embedding = model(x, edge_index)
    logit = logit_all[mask]

    y_np = y[mask].cpu().numpy()
    prob = torch.sigmoid(logit).cpu().numpy()
    metrics = compute_metrics(y_np, prob)
    return metrics, embedding[mask].cpu(), logit.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    seed = config["train"]["seed"]
    scarcity_ratio = config["dataset"].get("scarcity_ratio", 1.0)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device(
        config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.debug:
        print("[DEBUG] Using tiny synthetic graph, 3 epochs")
        data = load_fraud_dataset("tiny", seed=seed)
        epochs = config["train"].get("debug_epochs", 3)
    else:
        data = load_fraud_dataset(
            dataset_name, path=dataset_path, seed=seed, scarcity_ratio=scarcity_ratio
        )
        epochs = config["train"]["epochs"]

    model_cfg = config["model"]
    model = build_detector(
        name=model_cfg["name"],
        in_channels=data.x.shape[1],
        hidden_channels=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.5),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    patience = config["train"].get("patience", 50)
    best_val_auc = 0.0
    patience_counter = 0
    best_state = None

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, data, optimizer, device)
        val_metrics, _, _ = evaluate(model, data, "val", device)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                f"Val AUC: {val_metrics['roc_auc']:.4f} | "
                f"Val AUPRC: {val_metrics['auprc']:.4f}"
            )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, _, _ = evaluate(model, data, "test", device)
    train_metrics, _, _ = evaluate(model, data, "train", device)

    elapsed = time.time() - start_time

    print("\n=== Test Results ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    out_dir = Path("artifacts") / "checkpoints" / dataset_name / model_cfg["name"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "base.pt"
    torch.save(best_state, checkpoint_path)

    log_dir = Path("artifacts") / "logs" / dataset_name / model_cfg["name"] / f"seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_info = {
        "config": config,
        "seed": seed,
        "git_hash": get_git_hash(),
        "checkpoint_path": str(checkpoint_path),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "epochs_trained": epoch,
        "elapsed_seconds": elapsed,
    }

    with open(log_dir / "stage1.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\nCheckpoint saved to: {checkpoint_path}")
    print(f"Metrics saved to: {log_dir / 'stage1.json'}")


if __name__ == "__main__":
    main()
