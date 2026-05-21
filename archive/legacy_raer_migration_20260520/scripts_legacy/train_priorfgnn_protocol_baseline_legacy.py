"""Train a single-relation GNN on PriorF-GNN's unified data + 70/10/20 split.

Provides a fair "same-data, same-split" baseline for the PriorF-GNN
comparison. Wraps RAER-FD's existing detectors (BWGNN/GCN/SAGE/GAT)
but loads them with PriorF-GNN's stratified split and HSD-augmented
feature matrix from the unified data file.

Usage:
    python scripts/train_baseline_priorfgnn_protocol.py \\
        --config configs/yelpchi_priorfgnn_h64.yaml \\
        --model bwgnn --hidden_dim 64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gnn import build_detector
from training.metrics import compute_metrics_with_threshold, find_best_macro_f1_threshold
from utils.paths import ensure_dir, get_checkpoint_dir, get_logs_dir, get_results_dir
from utils.tensorboard import create_logger


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_unified(path: Path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    hsd = raw.hsd
    # Same x augmentation as PriorF-GNN pipeline (append HSD)
    x_full = torch.cat([raw.x, hsd.unsqueeze(1)], dim=1)
    # Merge all relations into one homogeneous adjacency
    edge_indexes = [raw.edge_index_dict[k] for k in ("rur", "rtr", "rsr") if k in raw.edge_index_dict]
    edge_index = torch.cat(edge_indexes, dim=1)
    return {
        "x": x_full,
        "y": raw.y,
        "hsd": hsd,
        "edge_index": edge_index,
        "train_mask": raw.train_mask,
        "val_mask": raw.val_mask,
        "test_mask": raw.test_mask,
        "in_dim": x_full.size(1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["bwgnn", "gcn", "sage", "gat"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", default="priorfgnn_protocol")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    dataset_name = cfg["dataset"]["name"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unified_path = Path(cfg["dataset"]["unified_data_path"])
    if "seed_42" in str(unified_path) and seed != 42:
        unified_path = Path(str(unified_path).replace("seed_42", f"seed_{seed}"))
    data = load_unified(unified_path)
    print(f"[data] loaded: {unified_path}")
    print(f"[data] x: {tuple(data['x'].shape)}  edges: {data['edge_index'].size(1)}  "
          f"train: {int(data['train_mask'].sum())}  val: {int(data['val_mask'].sum())}  "
          f"test: {int(data['test_mask'].sum())}")

    extra = {}
    if args.model == "gat":
        extra["attention_heads"] = 4
    model = build_detector(
        name=args.model, in_channels=data["in_dim"],
        hidden_channels=args.hidden_dim, num_layers=args.num_layers,
        dropout=args.dropout, **extra,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model}  params={n_params:,}  device={device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x = data["x"].to(device)
    edge_index = data["edge_index"].to(device)
    y = data["y"].to(device)
    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)

    # pos_weight from training set
    n_pos = int(y[train_mask].sum().item())
    n_neg = int(train_mask.sum().item() - n_pos)
    pos_weight = torch.tensor(max(n_neg / max(n_pos, 1), 1.0), dtype=torch.float32, device=device)
    print(f"[train] pos_weight={pos_weight.item():.3f}")

    tb_logger = create_logger(dataset_name, args.model, seed, args.run_name)

    best_val_score = -float("inf")
    best_state = None
    best_threshold = 0.5
    best_epoch = 0
    patience_counter = 0

    checkpoint_dir = ensure_dir(get_checkpoint_dir(dataset_name, args.model, args.run_name, seed))
    best_ckpt_path = checkpoint_dir / "best.pt"

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], y[train_mask].float(), pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_all = model(x, edge_index)
            val_prob = torch.sigmoid(logits_all[val_mask]).cpu().numpy()
            val_y = y[val_mask].cpu().numpy()
        thr, _ = find_best_macro_f1_threshold(val_y, val_prob)
        val_metrics = compute_metrics_with_threshold(val_y, val_prob, threshold=thr)

        tb_logger.log_scalar("train/loss", float(loss), epoch)
        for k in ("roc_auc", "auprc", "macro_f1", "g_means"):
            tb_logger.log_scalar(f"val_thr/{k}", float(val_metrics[k]), epoch)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss {float(loss):.4f} | "
                  f"val AUC {val_metrics['roc_auc']:.4f} | val AUPRC {val_metrics['auprc']:.4f} | "
                  f"val MaF1@{thr:.2f} {val_metrics['macro_f1']:.4f}")

        cur_score = val_metrics["auprc"]
        if cur_score > best_val_score:
            best_val_score = cur_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = float(thr)
            best_epoch = epoch
            patience_counter = 0
            torch.save(best_state, best_ckpt_path)
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch={best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    model.eval()
    with torch.no_grad():
        logits_all = model(x, edge_index)
        test_prob = torch.sigmoid(logits_all[test_mask]).cpu().numpy()
        test_y = y[test_mask].cpu().numpy()
    test_metrics = compute_metrics_with_threshold(test_y, test_prob, threshold=best_threshold)

    elapsed = time.time() - start
    print(f"\nBest epoch={best_epoch}  val_AUPRC={best_val_score:.4f}  thr={best_threshold:.3f}")
    print("=== Test metrics (best val threshold) ===")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    log_dir = ensure_dir(get_logs_dir(dataset_name, args.model, args.run_name, seed))
    info = {
        "config_path": args.config,
        "model": args.model, "hidden_dim": args.hidden_dim, "dropout": args.dropout,
        "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
        "seed": seed, "run_name": args.run_name, "git_hash": get_git_hash(),
        "n_params": n_params, "elapsed_seconds": elapsed,
        "best_epoch": best_epoch, "best_val_score": float(best_val_score),
        "best_val_threshold": best_threshold,
        "test_metrics": test_metrics,
        "data_protocol": "PriorF-GNN unified (70/10/20 stratified, HSD as feature, multi-rel union)",
    }
    with open(log_dir / "stage1.json", "w") as f:
        json.dump(info, f, indent=2)
    results_dir = ensure_dir(get_results_dir(dataset_name, args.model, args.run_name, seed))
    with open(results_dir / "stage1_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    tb_logger.log_metrics(test_metrics, epoch, prefix="test")
    tb_logger.close()
    print(f"\nMetrics: {log_dir / 'stage1.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
