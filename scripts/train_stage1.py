from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Set CUBLAS workspace config BEFORE importing torch to enable deterministic
# cuBLAS reductions. This matches the BWGNN deterministic baseline protocol
# documented in PROGRESS.md "Stage 1 Re-training (Deterministic Baseline)".
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from models.gnn import build_detector
from training.metrics import compute_metrics
from utils.tensorboard import create_logger
from utils.paths import get_checkpoint_dir, get_logs_dir, get_results_dir, ensure_dir


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

    logit_all = model(x, edge_index)
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

    output = model(x, edge_index, return_output=True)
    logit = output.logits[mask]
    embedding = output.embeddings[mask]

    y_np = y[mask].cpu().numpy()
    prob = torch.sigmoid(logit).cpu().numpy()
    metrics = compute_metrics(y_np, prob)
    return metrics, embedding.cpu(), logit.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_name", type=str, default="base")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stratified", action="store_true", help="Use stratified split")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic CUDA ops and save retraining_metrics.json")
    args = parser.parse_args()

    if args.debug and args.run_name == "base":
        args.run_name = "debug"
        print("[DEBUG] Forcing run_name=debug to avoid overwriting production checkpoints")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path")
    seed = args.seed or config["train"]["seed"]
    scarcity_ratio = config["dataset"].get("scarcity_ratio", 1.0)
    run_name = args.run_name

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # GAT's scatter_add_ deterministic implementation requires ~3x more GPU memory,
    # causing OOM on dense graphs (YelpChi 7.7M edges). Skip deterministic_algorithms
    # for GAT; rely on manual seeding + cudnn flags for reproducibility.
    # BWGNN/GCN/SAGE use the full deterministic path.
    model_name = config["model"]["name"]
    if model_name != "gat":
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    requested_device = str(config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    if args.debug:
        print("[DEBUG] Using tiny synthetic graph, 3 epochs")
        data = load_fraud_dataset("tiny", seed=seed)
        epochs = config["train"].get("debug_epochs", 3)
    else:
        split_mode = config["dataset"].get("split_mode", "supervised")
        train_ratio = config["dataset"].get("train_ratio", 0.7)
        val_test_ratio = config["dataset"].get("val_test_ratio", [1, 2])

        data = load_fraud_dataset(
            dataset_name, path=dataset_path, seed=seed, scarcity_ratio=scarcity_ratio,
            split_mode=split_mode, train_ratio=train_ratio, val_test_ratio=val_test_ratio,
            stratified=args.stratified
        )
        epochs = config["train"]["epochs"]

    model_cfg = config["model"]
    extra_kwargs = {}
    if "attention_heads" in model_cfg:
        extra_kwargs["attention_heads"] = model_cfg["attention_heads"]
    model = build_detector(
        name=model_cfg["name"],
        in_channels=data.x.shape[1],
        hidden_channels=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.5),
        **extra_kwargs,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    patience = config["train"].get("patience", 50)
    select_metric = config["train"].get("select_metric", "roc_auc")
    best_val_score = 0.0
    patience_counter = 0
    best_state = None

    start_time = time.time()

    tb_logger = create_logger(dataset_name, model_cfg["name"], seed, "stage1")

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, data, optimizer, device)
        val_metrics, _, _ = evaluate(model, data, "val", device)

        tb_logger.log_scalar("train/loss", loss, epoch)
        tb_logger.log_metrics(val_metrics, epoch, prefix="val")

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                f"Val AUC: {val_metrics['roc_auc']:.4f} | "
                f"Val AUPRC: {val_metrics['auprc']:.4f} | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
            )

        val_score = val_metrics.get(select_metric, val_metrics["roc_auc"])
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    test_metrics, _, _ = evaluate(model, data, "test", device)
    train_metrics, _, _ = evaluate(model, data, "train", device)

    tb_logger.log_metrics(test_metrics, epoch, prefix="test")
    tb_logger.log_metrics(train_metrics, epoch, prefix="train")
    tb_logger.close()

    elapsed = time.time() - start_time

    print("\n=== Test Results ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    model_cfg = config["model"]
    checkpoint_dir = ensure_dir(get_checkpoint_dir(dataset_name, model_cfg["name"], run_name, seed))
    checkpoint_path = checkpoint_dir / "base.pt"
    torch.save(best_state, checkpoint_path)

    log_dir = ensure_dir(get_logs_dir(dataset_name, model_cfg["name"], run_name, seed))

    run_info = {
        "config": config,
        "seed": seed,
        "run_name": run_name,
        "git_hash": get_git_hash(),
        "checkpoint_path": str(checkpoint_path),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "epochs_trained": epoch,
        "elapsed_seconds": elapsed,
    }

    with open(log_dir / "stage1.json", "w") as f:
        json.dump(run_info, f, indent=2)

    results_dir = ensure_dir(get_results_dir(dataset_name, model_cfg["name"], run_name, seed))
    with open(results_dir / "stage1_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nCheckpoint saved to: {checkpoint_path}")
    print(f"Metrics saved to: {log_dir / 'stage1.json'}")

    if args.deterministic:
        retrain_path = checkpoint_dir / "retraining_metrics.json"
        retrain_info = {
            "seed": seed,
            "git_hash": get_git_hash(),
            "deterministic": True,
            "config_path": args.config,
            "run_name": run_name,
            "model_name": model_cfg["name"],
            "dataset": dataset_name,
            "epochs_trained": epoch,
            "elapsed_seconds": elapsed,
            "test_metrics": test_metrics,
        }
        with open(retrain_path, "w") as f:
            json.dump(retrain_info, f, indent=2)
        print(f"Retraining metrics saved to: {retrain_path}")


if __name__ == "__main__":
    main()
