"""RAER-LREE teacher on top of a frozen PriorF-GNN base.

This slim trainer wires the current project's RAER teacher and LREE module
on top of a *frozen* PriorF-GNN base detector.  It is used for the
strong-base saturation check under PriorF-GNN's 40/20/40 protocol.

Pipeline::

    [frozen PriorF-GNN]                       [LREE]
       (base_logit, base_z=z_fused)              (rel_features per rel)
                 │                                       │
                 └────────► RAER teacher ◄────────────────┘
                                  │
                          L_cls (BCE w/ pos_weight)

Outputs match RAER-FD's existing raer result layout.
"""

from __future__ import annotations

import argparse
import copy
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

from evidence.lree import build_lree_extractor
from models.raer_teacher import RAERTeacher
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


def edge_index_to_sparse_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Build a sparse undirected adjacency tensor in COO layout.

    Mirrors what ``load_relation_adjs_for_extractor`` produces for the
    RAER-FD hand-crafted path: ``A = A + A.T`` then coalesce.
    """
    src, dst = edge_index
    indices = torch.stack([src, dst], dim=0)
    values = torch.ones(src.size(0), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    # Make symmetric (relations in .mat are stored upper-triangular sometimes)
    adj_t = torch.sparse_coo_tensor(
        torch.stack([dst, src], dim=0), values, (num_nodes, num_nodes)
    )
    return (adj + adj_t).coalesce()


def load_unified(path: Path) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    hsd = raw.hsd
    x_full = torch.cat([raw.x, hsd.unsqueeze(1)], dim=1)
    return {
        "x": x_full,
        "x_raw": raw.x,             # 32-dim raw features (no HSD) for extractor
        "y": raw.y,
        "hsd": hsd,
        "edge_index_dict": raw.edge_index_dict,
        "train_mask": raw.train_mask,
        "val_mask": raw.val_mask,
        "test_mask": raw.test_mask,
    }


def build_priorfgnn_base(in_channels: int, m_cfg: dict, ckpt_path: Path, device: torch.device):
    """Build PriorFGNNDetector, load checkpoint, freeze."""
    model = build_detector(
        name="priorfgnn",
        in_channels=in_channels,
        hidden_channels=m_cfg.get("hidden_dim", 64),
        num_layers=m_cfg.get("num_layers", 2),
        dropout=m_cfg.get("dropout", 0.3),
        num_relations=m_cfg.get("num_relations", 3),
        use_asda=m_cfg.get("use_asda", True),
        use_mlp_branch=m_cfg.get("use_mlp_branch", True),
        use_gnn_branch=m_cfg.get("use_gnn_branch", True),
        asda_tau=m_cfg.get("asda_tau", 0.1),
        proj_dim=m_cfg.get("proj_dim", 64),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def precompute_base_outputs(
    model, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
    hsd: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run frozen base once and return (base_logit, base_z)."""
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_type=edge_type, hsd=hsd, return_output=True)
    return out.logits.detach(), out.embeddings.detach()


def evaluate(teacher, base_logit, base_z, rel_features, y, mask, threshold: float):
    teacher.eval()
    with torch.no_grad():
        out = teacher(base_z=base_z, base_logit=base_logit, relation_features=rel_features)
    logits = out["logit"][mask] if "logit" in out else out["z"][mask]
    prob = torch.sigmoid(logits).cpu().numpy()
    y_np = y[mask].cpu().numpy()
    return compute_metrics_with_threshold(y_np, prob, threshold=threshold), prob, y_np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_config", required=True,
                        help="Path to PriorF-GNN base config (defines model architecture).")
    parser.add_argument("--base_ckpt", required=True,
                        help="Path to trained PriorF-GNN base checkpoint (.pt).")
    parser.add_argument("--run_name", default="strong_base_priorfgnn_raer_lree_404020")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--evidence", choices=["learned", "hand_crafted"], default="learned",
                        help="learned = LREE; hand_crafted not yet wired for this entry point.")
    args = parser.parse_args()

    with open(args.config) as f:
        raer_cfg = yaml.safe_load(f)
    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else raer_cfg.get("train", {}).get("seed", 42)
    epochs = args.epochs if args.epochs is not None else raer_cfg["raer_teacher"].get("epochs", 300)
    patience = args.patience if args.patience is not None else raer_cfg["raer_teacher"].get("patience", 50)
    lr = args.lr if args.lr is not None else raer_cfg["raer_teacher"].get("lr", 1e-3)
    weight_decay = raer_cfg["raer_teacher"].get("weight_decay", 1e-4)
    dataset_name = raer_cfg.get("dataset", {}).get("name", "yelpchi")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unified_path = Path(base_cfg["dataset"]["unified_data_path"])
    if "seed_42" in str(unified_path) and seed != 42:
        unified_path = Path(str(unified_path).replace("seed_42", f"seed_{seed}"))
    data = load_unified(unified_path)
    print(f"[data] {unified_path}  x={tuple(data['x'].shape)}  "
          f"train={int(data['train_mask'].sum())} val={int(data['val_mask'].sum())} test={int(data['test_mask'].sum())}")

    relation_names = list(raer_cfg["raer_teacher"].get("relation_names", ["RUR", "RTR", "RSR"]))
    anchor_relation = raer_cfg["raer_teacher"].get("anchor_relation", relation_names[0])
    rel_key_map = {"RUR": "rur", "RSR": "rsr", "RTR": "rtr"}
    num_nodes = int(data["x"].size(0))

    # Build per-relation sparse adj (in extractor's order)
    relation_adjs = []
    for r in relation_names:
        k = rel_key_map[r]
        if k not in data["edge_index_dict"]:
            raise KeyError(f"relation {r} (key {k}) missing in unified data")
        adj = edge_index_to_sparse_adj(data["edge_index_dict"][k], num_nodes).to(device)
        relation_adjs.append(adj)
        print(f"  rel {r}: {adj._nnz()} entries (sym)")

    # Build per-relation merged edge_index+edge_type for the base model forward
    edge_indexes, edge_types = [], []
    rel2id = {"rur": 0, "rtr": 1, "rsr": 2}
    for rel_name, rel_id in rel2id.items():
        if rel_name in data["edge_index_dict"]:
            e = data["edge_index_dict"][rel_name]
            edge_indexes.append(e)
            edge_types.append(torch.full((e.size(1),), rel_id, dtype=torch.long))
    base_edge_index = torch.cat(edge_indexes, dim=1).to(device)
    base_edge_type = torch.cat(edge_types, dim=0).to(device)
    base_x = data["x"].to(device)
    base_hsd = data["hsd"].to(device)
    y = data["y"].to(device)
    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)

    # Load frozen PriorF-GNN base
    base = build_priorfgnn_base(
        in_channels=int(base_x.size(1)),
        m_cfg=base_cfg["model"],
        ckpt_path=Path(args.base_ckpt),
        device=device,
    )
    base_logit, base_z = precompute_base_outputs(
        base, base_x, base_edge_index, base_edge_type, base_hsd, device,
    )
    print(f"[base] PriorF-GNN frozen  base_z={tuple(base_z.shape)}  base_logit={tuple(base_logit.shape)}")

    # Free base_edge memory we no longer need
    del base_edge_index, base_edge_type, base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # LREE on raw 32-dim features (NOT HSD-augmented)
    ext_cfg = raer_cfg["raer_teacher"].get("evidence", {}).get("extractor", {}) or {}
    x_for_ext = data["x_raw"].to(device)
    rel_stat_dim = int(ext_cfg.get("out_dim_per_rel", 9))
    extractor = build_lree_extractor(
        x_dim=int(x_for_ext.size(1)),
        num_relations=len(relation_names),
        cfg={**ext_cfg, "out_dim_per_rel": rel_stat_dim},
    ).to(device)
    extractor.prepare(x_for_ext, relation_adjs)
    print(f"[ext] LearnedRelationEvidenceExtractor  params={sum(p.numel() for p in extractor.parameters()):,}")

    # RAER-LREE teacher
    pr_cfg = raer_cfg["raer_teacher"]
    teacher = RAERTeacher(
        base_z_dim=int(base_z.size(1)),
        relation_names=relation_names,
        anchor_relation=anchor_relation,
        rel_stat_dim=rel_stat_dim,
        rel_hidden_dim=int(pr_cfg.get("rel_hidden_dim", 64)),
        rel_num_layers=int(pr_cfg.get("rel_num_layers", 2)),
        rel_dropout=float(pr_cfg.get("rel_dropout", 0.3)),
        tau_gate=float(pr_cfg.get("tau_gate", 0.7)),
        delta_rel_max=float(pr_cfg.get("delta_rel_max", 2.0)),
        gate_mode=str(pr_cfg.get("gate_mode", "softmax")),
    ).to(device)
    n_params_teacher = sum(p.numel() for p in teacher.parameters())
    n_params_ext = sum(p.numel() for p in extractor.parameters())
    print(f"[raer_teacher] params={n_params_teacher:,}  base_z_dim={base_z.size(1)}")

    # Optimizer (extractor + teacher; base frozen)
    params = list(extractor.parameters()) + list(teacher.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # pos_weight from train labels
    n_pos = int(y[train_mask].sum().item())
    n_neg = int(train_mask.sum().item() - n_pos)
    pos_weight = torch.tensor(max(n_neg / max(n_pos, 1), 1.0), dtype=torch.float32, device=device)

    tb_logger = create_logger(dataset_name, "priorfgnn", seed, args.run_name)

    best_val_score = -float("inf")
    best_threshold = 0.5
    best_epoch = 0
    best_teacher_state = None
    best_ext_state = None
    patience_counter = 0

    checkpoint_dir = ensure_dir(get_checkpoint_dir(dataset_name, "priorfgnn", args.run_name, seed))
    best_ckpt = checkpoint_dir / "raer_lree_teacher_best.pt"

    select_metric = pr_cfg.get("early_stop_metric", "val_auprc").replace("val_", "")
    train_labels_long = y.long()

    start = time.time()
    for epoch in range(1, epochs + 1):
        extractor.train(); teacher.train()
        optimizer.zero_grad()
        rel_feat = extractor(x_for_ext, train_mask, train_labels_long)
        out = teacher(base_z=base_z, base_logit=base_logit, relation_features=rel_feat)
        z = out["final_logit"]
        loss = F.binary_cross_entropy_with_logits(
            z[train_mask], y[train_mask].float(), pos_weight=pos_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # Val eval (use no-grad rel_feat)
        extractor.eval(); teacher.eval()
        with torch.no_grad():
            rel_feat_eval = extractor(x_for_ext, train_mask, train_labels_long)
            out_eval = teacher(base_z=base_z, base_logit=base_logit, relation_features=rel_feat_eval)
            z_eval = out_eval["final_logit"]
            val_prob = torch.sigmoid(z_eval[val_mask]).cpu().numpy()
            val_y = y[val_mask].cpu().numpy()
        thr, _ = find_best_macro_f1_threshold(val_y, val_prob)
        val_metrics = compute_metrics_with_threshold(val_y, val_prob, threshold=thr)

        tb_logger.log_scalar("train/loss", float(loss), epoch)
        for k in ("roc_auc", "auprc", "macro_f1", "g_means"):
            tb_logger.log_scalar(f"val_thr/{k}", float(val_metrics[k]), epoch)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss {float(loss):.4f} | "
                  f"val AUC {val_metrics['roc_auc']:.4f} | val AUPRC {val_metrics['auprc']:.4f} | "
                  f"val MaF1@{thr:.2f} {val_metrics['macro_f1']:.4f}")

        cur_score = val_metrics.get(select_metric, val_metrics["auprc"])
        if cur_score > best_val_score:
            best_val_score = cur_score
            best_threshold = float(thr)
            best_epoch = epoch
            best_teacher_state = copy.deepcopy(teacher.state_dict())
            best_ext_state = copy.deepcopy(extractor.state_dict())
            patience_counter = 0
            torch.save({"raer_teacher": best_teacher_state, "lree": best_ext_state}, best_ckpt)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best={best_epoch})")
            break

    if best_teacher_state is not None:
        teacher.load_state_dict(best_teacher_state)
        extractor.load_state_dict(best_ext_state)

    # Final test eval
    extractor.eval(); teacher.eval()
    with torch.no_grad():
        rel_feat_final = extractor(x_for_ext, train_mask, train_labels_long)
        out_final = teacher(base_z=base_z, base_logit=base_logit, relation_features=rel_feat_final)
        z_final = out_final["final_logit"]
        test_prob = torch.sigmoid(z_final[test_mask]).cpu().numpy()
        test_y = y[test_mask].cpu().numpy()
    test_metrics = compute_metrics_with_threshold(test_y, test_prob, threshold=best_threshold)

    elapsed = time.time() - start
    print(f"\nBest epoch={best_epoch}  val_{select_metric}={best_val_score:.4f}  thr={best_threshold:.3f}")
    print("=== Test metrics ===")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    log_dir = ensure_dir(get_logs_dir(dataset_name, "priorfgnn", args.run_name, seed))
    info = {
        "raer_config": args.config,
        "base_config": args.base_config,
        "base_ckpt": str(args.base_ckpt),
        "seed": seed,
        "git_hash": get_git_hash(),
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_val_score": float(best_val_score),
        "best_val_threshold": best_threshold,
        "test_metrics": test_metrics,
        "n_params": {"raer_teacher": n_params_teacher, "lree": n_params_ext},
        "elapsed_seconds": elapsed,
        "data_protocol": "PriorF-GNN unified 40/20/40 split with multi-relation graph and HSD feature",
        "evidence_source": args.evidence,
        "checkpoint": str(best_ckpt),
    }
    with open(log_dir / "raer_summary.json", "w") as f:
        json.dump(info, f, indent=2)
    results_dir = ensure_dir(get_results_dir(dataset_name, "priorfgnn", args.run_name, seed))
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    tb_logger.log_metrics(test_metrics, epoch, prefix="test")
    tb_logger.close()

    print(f"\nMetrics: {log_dir / 'raer_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
