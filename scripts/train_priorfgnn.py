"""Train PriorF-GNN (LG-HGCL v2) integrated into RAER-FD.

Bridges PriorF-GNN's multi-relation + HSD pipeline with RAER-FD's
checkpoint/metric/results layout. Loads PriorF-GNN's pre-computed
unified data file so the seed-42 result reproduces the reference
(AUPRC=0.8475, AUROC=0.9555).

Usage:
    python scripts/train_priorfgnn.py --config configs/raer_fd/base_detectors/yelpchi_priorfgnn.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import yaml
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gnn import build_detector
from training.metrics import compute_metrics_with_threshold, find_best_macro_f1_threshold
from training.priorfgnn_losses import focal_loss_with_logits, sdcl_loss_v4
from utils.paths import ensure_dir, get_checkpoint_dir, get_logs_dir, get_results_dir
from utils.tensorboard import create_logger


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_unified_data(path: Path, hsd_invert: bool = False):
    """Load PriorF-GNN's pre-computed unified data.

    Returned object has: x (N,D), y (N,), hsd (N,), train/val/test_mask,
    edge_index_dict {rur, rtr, rsr}.

    We collapse multi-relation edges into a single edge_index + edge_type
    tensor pair so RGCNConv can process them in one forward pass.
    Also append HSD as an extra feature column (matches the PriorF-GNN
    pipeline `_stage_unified` step).
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)

    hsd = raw.hsd
    if hsd_invert and hsd.max() > 0:
        hsd = hsd.max() - hsd

    x_full = torch.cat([raw.x, hsd.unsqueeze(1)], dim=1)

    edge_indexes = []
    edge_types = []
    rel2id = {"rur": 0, "rtr": 1, "rsr": 2}
    for rel_name, rel_id in rel2id.items():
        if rel_name in raw.edge_index_dict:
            e = raw.edge_index_dict[rel_name]
            edge_indexes.append(e)
            edge_types.append(torch.full((e.size(1),), rel_id, dtype=torch.long))
    edge_index = torch.cat(edge_indexes, dim=1)
    edge_type = torch.cat(edge_types, dim=0)

    return {
        "x": x_full,
        "y": raw.y,
        "hsd": hsd,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "train_mask": raw.train_mask,
        "val_mask": raw.val_mask,
        "test_mask": raw.test_mask,
        "in_dim": x_full.size(1),
        "num_nodes": x_full.size(0),
        "num_relations": len(rel2id),
    }


def make_sdcl_lambda(epoch: int, total_epochs: int, base: float, schedule: str) -> float:
    if base <= 0:
        return 0.0
    if schedule == "cosine":
        progress = epoch / max(total_epochs - 1, 1)
        return base * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "increasing":
        return base * (epoch / max(total_epochs - 1, 1))
    return base  # constant


def evaluate(model, data, mask_name: str, device, threshold: float = 0.5) -> dict:
    model.eval()
    with torch.no_grad():
        x = data["x"].to(device)
        edge_index = data["edge_index"].to(device)
        edge_type = data["edge_type"].to(device)
        hsd = data["hsd"].to(device)
        mask = data[f"{mask_name}_mask"].to(device)
        y = data["y"].to(device)

        logits = model(x, edge_index, edge_type=edge_type, hsd=hsd)
        prob = torch.sigmoid(logits[mask]).cpu().numpy()
        y_np = y[mask].cpu().numpy()
    return compute_metrics_with_threshold(y_np, prob, threshold=threshold), prob, y_np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default="base")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--select_metric", type=str, default=None)
    parser.add_argument("--no_unified", action="store_true",
                        help="Skip the PriorF-GNN unified data path and rebuild from raw .mat.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    patience = args.patience if args.patience is not None else cfg["train"]["patience"]
    select_metric = args.select_metric or cfg["train"].get("select_metric", "auprc")
    dataset_name = cfg["dataset"]["name"]
    run_name = args.run_name

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # ASDA's softmax uses scatter on CUDA — keep deterministic flag soft.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    requested_device = str(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    # Resolve unified data path for the current seed if it's templated by seed=42
    unified_path = Path(cfg["dataset"]["unified_data_path"])
    if not args.no_unified and "seed_42" in unified_path.name + str(unified_path):
        unified_path = Path(str(unified_path).replace("seed_42", f"seed_{seed}"))

    if not args.no_unified and unified_path.exists():
        data = load_unified_data(unified_path)
        print(f"[data] loaded unified file: {unified_path}")
    else:
        # Rebuild from raw .mat — replicates PriorF-GNN/comp/generate_unified_data.py
        from sklearn.model_selection import train_test_split
        import scipy.io as sio
        import scipy.sparse as sp
        from torch_scatter import scatter_mean

        mat = sio.loadmat(cfg["dataset"]["mat_path"])
        features = mat["features"]
        if sp.issparse(features):
            features = features.toarray()
        x = torch.from_numpy(features.astype(np.float32))
        y = torch.from_numpy(mat["label"].flatten().astype(np.int64))
        rel_keys = ("net_rur", "net_rtr", "net_rsr") if "net_rur" in mat else ("net_upu", "net_usu", "net_uvu")
        rel_names = ("rur", "rtr", "rsr") if "net_rur" in mat else ("upu", "usu", "uvu")
        edge_indexes, edge_types = [], []
        edge_index_dict = {}
        for rid, (mat_key, rel_name) in enumerate(zip(rel_keys, rel_names)):
            adj = mat[mat_key].tocoo()
            row = torch.from_numpy(adj.row.astype(np.int64))
            col = torch.from_numpy(adj.col.astype(np.int64))
            ei = torch.stack([row, col], dim=0)
            mask = ei[0] != ei[1]
            ei = ei[:, mask]
            edge_index_dict[rel_name] = ei
            edge_indexes.append(ei)
            edge_types.append(torch.full((ei.size(1),), rid, dtype=torch.long))
        edge_index = torch.cat(edge_indexes, dim=1)
        edge_type = torch.cat(edge_types, dim=0)

        # HSD: mean L2 distance to neighbours over union edges
        row, col = edge_index
        dist = torch.norm(x[row] - x[col], p=2, dim=1)
        hsd = scatter_mean(dist, row, dim=0, dim_size=x.size(0))
        hsd = torch.nan_to_num(hsd, nan=0.0)

        # stratified 70/10/20
        n = x.size(0)
        idx = np.arange(n)
        train_idx, rem_idx = train_test_split(idx, train_size=0.7, stratify=y.numpy(), random_state=seed)
        val_idx, test_idx = train_test_split(rem_idx, train_size=1.0/3.0, stratify=y.numpy()[rem_idx], random_state=seed)
        train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
        val_mask = torch.zeros(n, dtype=torch.bool); val_mask[val_idx] = True
        test_mask = torch.zeros(n, dtype=torch.bool); test_mask[test_idx] = True

        x_full = torch.cat([x, hsd.unsqueeze(1)], dim=1)
        data = {
            "x": x_full, "y": y, "hsd": hsd,
            "edge_index": edge_index, "edge_type": edge_type,
            "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
            "in_dim": x_full.size(1), "num_nodes": n, "num_relations": len(rel_names),
        }
        print(f"[data] built from raw .mat (seed={seed}): n={n}, in_dim={data['in_dim']}")

    print(f"[data] x: {tuple(data['x'].shape)}  edges: {data['edge_index'].size(1)}  "
          f"train: {int(data['train_mask'].sum())}  val: {int(data['val_mask'].sum())}  "
          f"test: {int(data['test_mask'].sum())}")

    m_cfg = cfg["model"]
    model = build_detector(
        name=m_cfg["name"],
        in_channels=data["in_dim"],
        hidden_channels=m_cfg.get("hidden_dim", 128),
        num_layers=m_cfg.get("num_layers", 2),
        dropout=m_cfg.get("dropout", 0.3),
        num_relations=data["num_relations"],
        use_asda=m_cfg.get("use_asda", True),
        use_mlp_branch=m_cfg.get("use_mlp_branch", True),
        use_gnn_branch=m_cfg.get("use_gnn_branch", True),
        asda_tau=m_cfg.get("asda_tau", 0.1),
        proj_dim=m_cfg.get("proj_dim", 64),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {m_cfg['name']}  params={n_params:,}  device={device}")

    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["train"].get("scheduler_T0", 20),
        T_mult=cfg["train"].get("scheduler_Tmult", 2),
    )

    focal_alpha = m_cfg.get("focal_alpha", 0.25)
    focal_gamma = m_cfg.get("focal_gamma", 2.0)
    lambda_sdcl = m_cfg.get("lambda_sdcl", 0.0)
    sdcl_tau = m_cfg.get("sdcl_tau", 0.1)
    sdcl_anchors = m_cfg.get("sdcl_anchors", 128)
    sdcl_quantile = m_cfg.get("sdcl_quantile", 0.3)
    sdcl_rounds = m_cfg.get("sdcl_rounds", 1)
    sdcl_schedule = m_cfg.get("sdcl_schedule", "cosine")
    grad_clip = cfg["train"].get("grad_clip", 0.0)
    use_amp = bool(cfg["train"].get("use_amp", False)) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    print(f"[train] use_amp={use_amp}  grad_clip={grad_clip}")

    tb_logger = create_logger(dataset_name, m_cfg["name"], seed, "base")

    x_d = data["x"].to(device)
    edge_index_d = data["edge_index"].to(device)
    edge_type_d = data["edge_type"].to(device)
    hsd_d = data["hsd"].to(device)
    y_d = data["y"].to(device)
    train_mask_d = data["train_mask"].to(device)

    best_val_score = -float("inf")
    best_state = None
    best_epoch = 0
    best_threshold = 0.5
    patience_counter = 0

    # Save checkpoint dir up-front so we can persist `best_state` on every
    # improvement — protects against mid-training OOM crashes (we lost a
    # 140-epoch run once because the in-memory `best_state` evaporated).
    checkpoint_dir = ensure_dir(get_checkpoint_dir(dataset_name, m_cfg["name"], run_name, seed))
    best_ckpt_path = checkpoint_dir / "best.pt"
    best_meta_path = checkpoint_dir / "best_meta.json"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        with autocast("cuda", enabled=use_amp):
            logits = model(x_d, edge_index_d, edge_type=edge_type_d, hsd=hsd_d)

            target_logits = logits[train_mask_d]
            target_y = y_d[train_mask_d].float()

            loss_focal = focal_loss_with_logits(target_logits, target_y, alpha=focal_alpha, gamma=focal_gamma)

            loss_sdcl = torch.zeros((), device=device)
            if lambda_sdcl > 0 and model.z_proj is not None:
                z_proj_train = model.z_proj[train_mask_d]
                hsd_train = hsd_d[train_mask_d].float()
                loss_sdcl = sdcl_loss_v4(
                    z_proj_train, target_y, hsd_train,
                    temperature=sdcl_tau, num_anchors=sdcl_anchors,
                    quantile=sdcl_quantile, num_rounds=sdcl_rounds,
                )

            cur_lambda = make_sdcl_lambda(epoch - 1, epochs, lambda_sdcl, sdcl_schedule)
            loss = loss_focal + cur_lambda * loss_sdcl

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Eval: same protocol as RAER-FD base-detector training: threshold search on val.
        val_metrics, val_prob, val_y = evaluate(model, data, "val", device, threshold=0.5)
        thr, _ = find_best_macro_f1_threshold(val_y, val_prob)
        val_metrics_thr = compute_metrics_with_threshold(val_y, val_prob, threshold=thr)

        tb_logger.log_scalar("train/loss_total", float(loss), epoch)
        tb_logger.log_scalar("train/loss_focal", float(loss_focal), epoch)
        tb_logger.log_scalar("train/loss_sdcl", float(loss_sdcl), epoch)
        tb_logger.log_scalar("train/lambda_sdcl", cur_lambda, epoch)
        for k in ("roc_auc", "auprc", "macro_f1", "g_means"):
            tb_logger.log_scalar(f"val_thr/{k}", float(val_metrics_thr[k]), epoch)
        tb_logger.log_scalar("val/best_threshold", thr, epoch)
        tb_logger.log_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss {float(loss):.4f} | "
                  f"val AUC {val_metrics_thr['roc_auc']:.4f} | "
                  f"val AUPRC {val_metrics_thr['auprc']:.4f} | "
                  f"val MaF1@{thr:.2f} {val_metrics_thr['macro_f1']:.4f}")

        cur_score = val_metrics_thr.get(select_metric, val_metrics_thr["auprc"])
        if cur_score > best_val_score:
            best_val_score = cur_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = float(thr)
            best_epoch = epoch
            patience_counter = 0
            # Persist immediately so an OOM later in training does not
            # destroy progress.
            torch.save(best_state, best_ckpt_path)
            with open(best_meta_path, "w") as f:
                json.dump({
                    "best_epoch": best_epoch,
                    "best_val_score": float(best_val_score),
                    "select_metric": select_metric,
                    "best_val_threshold": best_threshold,
                    "val_metrics_thr": {k: float(v) for k, v in val_metrics_thr.items() if isinstance(v, (int, float))},
                }, f, indent=2)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch={best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif best_ckpt_path.exists():
        # Mid-training crash recovery: best_state was lost but checkpoint persisted.
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    test_metrics, _, _ = evaluate(model, data, "test", device, threshold=best_threshold)
    train_metrics, _, _ = evaluate(model, data, "train", device, threshold=best_threshold)
    val_metrics_final, _, _ = evaluate(model, data, "val", device, threshold=best_threshold)

    tb_logger.log_metrics(test_metrics, epoch, prefix="test")
    tb_logger.log_metrics(train_metrics, epoch, prefix="train")
    tb_logger.log_scalar("test/best_val_threshold", best_threshold, epoch)
    tb_logger.log_scalar("test/best_epoch", best_epoch, epoch)
    tb_logger.close()

    elapsed = time.time() - start_time
    print(f"\nBest epoch={best_epoch}  best val {select_metric}={best_val_score:.4f}  thr={best_threshold:.4f}")
    print("\n=== Test (at best val threshold) ===")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    checkpoint_dir = ensure_dir(get_checkpoint_dir(dataset_name, m_cfg["name"], run_name, seed))
    ckpt_path = checkpoint_dir / "base.pt"
    torch.save(best_state if best_state is not None else model.state_dict(), ckpt_path)

    log_dir = ensure_dir(get_logs_dir(dataset_name, m_cfg["name"], run_name, seed))
    run_info = {
        "config": cfg,
        "seed": seed,
        "run_name": run_name,
        "git_hash": get_git_hash(),
        "checkpoint_path": str(ckpt_path),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics_final,
        "test_metrics": test_metrics,
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "best_val_threshold": best_threshold,
        "best_val_score": float(best_val_score),
        "select_metric": select_metric,
        "elapsed_seconds": elapsed,
        "n_params": n_params,
    }
    with open(log_dir / "base_training.json", "w") as f:
        json.dump(run_info, f, indent=2)

    results_dir = ensure_dir(get_results_dir(dataset_name, m_cfg["name"], run_name, seed))
    with open(results_dir / "base_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Metrics:    {log_dir / 'base_training.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
