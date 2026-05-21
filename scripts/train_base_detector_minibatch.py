"""Mini-batch base detector trainer for large RAER-FD datasets.

This entry point is intentionally separate from ``train_base_detector.py`` so
the compact full-graph protocol remains unchanged. It uses a small CPU
neighbor sampler implemented in this file because the current environment may
not have a working PyG ``NeighborLoader`` backend.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from models.gnn import build_detector
from scripts.train_raer_teacher import get_base_output_override_cache_paths
from training.metrics import compute_metrics_with_threshold, find_best_macro_f1_threshold
from utils.paths import ensure_dir, get_checkpoint_dir, get_logs_dir, get_results_dir


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, ValueError):
            pass


class CpuNeighborSampler:
    """CPU fanout sampler that returns layered mini-batch subgraphs.

    The first ``batch_size`` local nodes are always the root nodes whose logits
    are supervised/evaluated. Additional sampled neighbors provide context.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        fanouts: list[int],
        seed: int,
        undirected: bool = False,
        add_reverse_edges: bool = False,
        sampling_direction: str = "in",
    ) -> None:
        if sampling_direction not in {"in", "out"}:
            raise ValueError("sampling_direction must be 'in' or 'out'")
        self.num_nodes = int(num_nodes)
        self.fanouts = [int(v) for v in fanouts]
        self.rng = random.Random(int(seed))
        self.sampling_direction = sampling_direction
        self.add_reverse_edges = bool(add_reverse_edges)

        row = edge_index[0].detach().cpu().long()
        col = edge_index[1].detach().cpu().long()
        if sampling_direction == "in":
            key = col
            neigh = row
        else:
            key = row
            neigh = col

        if undirected:
            key = torch.cat([key, neigh], dim=0)
            neigh = torch.cat([neigh, key[: neigh.numel()]], dim=0)

        valid = (key >= 0) & (key < self.num_nodes) & (neigh >= 0) & (neigh < self.num_nodes)
        if not bool(valid.all()):
            key = key[valid]
            neigh = neigh[valid]
        counts = torch.bincount(key, minlength=self.num_nodes)
        self.indptr = torch.empty(self.num_nodes + 1, dtype=torch.long)
        self.indptr[0] = 0
        self.indptr[1:] = torch.cumsum(counts, dim=0)
        order = torch.argsort(key)
        self.indices = neigh[order].contiguous()

    def sample(self, roots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        roots_list = [int(v) for v in roots.detach().cpu().tolist()]
        nodes: list[int] = list(dict.fromkeys(roots_list))
        node_seen = set(nodes)
        frontier = roots_list
        global_src: list[int] = []
        global_dst: list[int] = []

        for fanout in self.fanouts:
            if fanout == 0 or not frontier:
                break
            next_frontier: list[int] = []
            for node in frontier:
                start = int(self.indptr[node].item())
                end = int(self.indptr[node + 1].item())
                degree = end - start
                if degree <= 0:
                    continue
                if fanout < 0 or degree <= fanout:
                    chosen_tensor = self.indices[start:end]
                else:
                    offsets = self.rng.sample(range(degree), fanout)
                    chosen_tensor = self.indices[torch.tensor(offsets, dtype=torch.long) + start]
                chosen = [int(v) for v in chosen_tensor.tolist()]
                next_frontier.extend(chosen)
                for dst in chosen:
                    if self.sampling_direction == "in":
                        global_src.append(dst)
                        global_dst.append(node)
                        if self.add_reverse_edges and dst != node:
                            global_src.append(node)
                            global_dst.append(dst)
                    else:
                        global_src.append(node)
                        global_dst.append(dst)
                        if self.add_reverse_edges and dst != node:
                            global_src.append(dst)
                            global_dst.append(node)
                    if dst not in node_seen:
                        node_seen.add(dst)
                        nodes.append(dst)
            frontier = next_frontier

        node_ids = torch.tensor(nodes, dtype=torch.long)
        local = {node_id: idx for idx, node_id in enumerate(nodes)}
        if global_src:
            pairs = [
                (local[src], local[dst])
                for src, dst in zip(global_src, global_dst)
                if src in local and dst in local
            ]
            src_local = [src for src, _ in pairs]
            dst_local = [dst for _, dst in pairs]
            sub_edge_index = torch.tensor([src_local, dst_local], dtype=torch.long)
        else:
            sub_edge_index = torch.zeros((2, 0), dtype=torch.long)
        return node_ids, sub_edge_index


def iter_root_batches(
    root_idx: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
):
    if shuffle:
        order = torch.randperm(root_idx.numel(), generator=generator)
        root_idx = root_idx[order]
    for start in range(0, root_idx.numel(), batch_size):
        yield root_idx[start:start + batch_size]


def build_model(config: dict, in_channels: int, device: torch.device):
    model_cfg = config["model"]
    model_name = str(model_cfg["name"]).lower()
    if model_name not in {"sage", "gcn", "gat"}:
        raise ValueError(
            "Mini-batch base training currently supports sage/gcn/gat. "
            f"Got {model_name!r}."
        )
    extra_kwargs = {}
    if model_name == "gat" and "attention_heads" in model_cfg:
        extra_kwargs["attention_heads"] = int(model_cfg["attention_heads"])
    return build_detector(
        name=model_name,
        in_channels=in_channels,
        hidden_channels=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.5)),
        **extra_kwargs,
    ).to(device)


def forward_batch(model, data, sampler: CpuNeighborSampler, roots: torch.Tensor, device: torch.device):
    node_ids, sub_edge_index = sampler.sample(roots)
    x = data.x[node_ids].to(device)
    edge_index = sub_edge_index.to(device)
    out = model(x, edge_index, return_output=True)
    root_count = roots.numel()
    return out.logits[:root_count], out.embeddings[:root_count]


def train_one_epoch(
    model,
    data,
    sampler: CpuNeighborSampler,
    train_idx: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    pos_weight: torch.Tensor,
    generator: torch.Generator,
    grad_clip: float | None,
) -> float:
    model.train()
    losses: list[float] = []
    for roots in iter_root_batches(train_idx, batch_size, shuffle=True, generator=generator):
        logits, _ = forward_batch(model, data, sampler, roots, device)
        y = data.y[roots].to(device=device, dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict_split(
    model,
    data,
    sampler: CpuNeighborSampler,
    root_idx: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    y_all: list[torch.Tensor] = []
    generator = torch.Generator().manual_seed(0)
    for roots in iter_root_batches(root_idx, batch_size, shuffle=False, generator=generator):
        logits, _ = forward_batch(model, data, sampler, roots, device)
        logits_all.append(logits.detach().cpu())
        y_all.append(data.y[roots].detach().cpu())
    return torch.cat(logits_all), torch.cat(y_all)


@torch.no_grad()
def compute_base_output_cache(
    model,
    data,
    sampler: CpuNeighborSampler,
    device: torch.device,
    batch_size: int,
    cache_dtype: str = "float32",
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    num_nodes = int(data.x.shape[0])
    logits = torch.empty(num_nodes, dtype=torch.float32)
    z_dim = int(model.head.in_features)
    embeddings = torch.empty(num_nodes, z_dim, dtype=torch.float32)
    idx = torch.arange(num_nodes, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    for roots in iter_root_batches(idx, batch_size, shuffle=False, generator=generator):
        batch_logits, batch_z = forward_batch(model, data, sampler, roots, device)
        logits[roots] = batch_logits.detach().float().cpu()
        embeddings[roots] = batch_z.detach().float().cpu()
    if cache_dtype == "float16":
        return logits, embeddings.half()
    return logits, embeddings


def load_dataset_from_config(config: dict, seed: int, debug: bool = False):
    if debug:
        return load_fraud_dataset("tiny", seed=seed, stratified=True)
    ds_cfg = config["dataset"]
    return load_fraud_dataset(
        name=ds_cfg["name"],
        path=ds_cfg.get("path"),
        format=ds_cfg.get("format"),
        seed=seed,
        scarcity_ratio=float(ds_cfg.get("scarcity_ratio", 1.0)),
        split_mode=ds_cfg.get("split_mode", "supervised"),
        train_ratio=float(ds_cfg.get("train_ratio", 0.4)),
        val_test_ratio=list(ds_cfg.get("val_test_ratio", [1, 2])),
        stratified=bool(ds_cfg.get("stratified", True)),
        hsd_invert=ds_cfg.get("hsd_invert"),
        hsd_chunk_size=int(ds_cfg.get("hsd_chunk_size", 250_000)),
        append_hsd=bool(ds_cfg.get("append_hsd", True)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Large-graph mini-batch base detector trainer")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--stratified", action="store_true", help="Accepted for CLI parity; config controls the split.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = int(args.seed if args.seed is not None else config["train"].get("seed", 0))
    set_seed(seed)
    device_name = args.device or config["train"].get("device", "cuda:0")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    data = load_dataset_from_config(config, seed=seed, debug=args.debug)
    train_cfg = config.get("minibatch", {}) or {}
    fanouts = [int(v) for v in train_cfg.get("num_neighbors", [15, 10])]
    eval_fanouts = [int(v) for v in train_cfg.get("eval_num_neighbors", fanouts)]
    batch_size = int(train_cfg.get("batch_size", 1024))
    eval_batch_size = int(train_cfg.get("eval_batch_size", batch_size))
    cache_batch_size = int(train_cfg.get("cache_batch_size", eval_batch_size))

    train_idx = torch.nonzero(data.train_mask, as_tuple=False).view(-1)
    val_idx = torch.nonzero(data.val_mask, as_tuple=False).view(-1)
    test_idx = torch.nonzero(data.test_mask, as_tuple=False).view(-1)
    if train_idx.numel() == 0:
        raise ValueError("empty train split")

    model = build_model(config, in_channels=int(data.x.shape[1]), device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"].get("lr", 0.01)),
        weight_decay=float(config["train"].get("weight_decay", 5e-4)),
    )
    y_train = data.y[train_idx]
    n_pos = int((y_train == 1).sum().item())
    n_neg = int((y_train == 0).sum().item())
    pos_weight = torch.tensor(
        max(n_neg / max(n_pos, 1), 1.0),
        dtype=torch.float32,
        device=device,
    )

    epochs = int(args.epochs if args.epochs is not None else config["train"].get("epochs", 100))
    if args.debug:
        epochs = int(config["train"].get("debug_epochs", 3))
    patience = int(args.patience if args.patience is not None else config["train"].get("patience", 20))
    select_metric = str(config["train"].get("select_metric", "macro_f1"))
    run_name = str(args.run_name or train_cfg.get("run_name") or "base_minibatch")
    model_name = str(config["model"]["name"]).lower()
    dataset_name = str(config["dataset"]["name"]).lower() if not args.debug else str(config["dataset"]["name"]).lower()

    train_sampler = CpuNeighborSampler(
        data.edge_index,
        num_nodes=int(data.x.shape[0]),
        fanouts=fanouts,
        seed=seed,
        undirected=bool(train_cfg.get("undirected_sampling", False)),
        add_reverse_edges=bool(train_cfg.get("add_reverse_edges", False)),
        sampling_direction=str(train_cfg.get("sampling_direction", "in")),
    )
    eval_sampler = CpuNeighborSampler(
        data.edge_index,
        num_nodes=int(data.x.shape[0]),
        fanouts=eval_fanouts,
        seed=int(train_cfg.get("eval_seed", seed)),
        undirected=bool(train_cfg.get("undirected_sampling", False)),
        add_reverse_edges=bool(train_cfg.get("add_reverse_edges", False)),
        sampling_direction=str(train_cfg.get("sampling_direction", "in")),
    )

    print(
        f"[Base-MB] dataset={dataset_name} model={model_name} seed={seed} "
        f"device={device} fanouts={fanouts} eval_fanouts={eval_fanouts} "
        f"batch={batch_size} eval_batch={eval_batch_size}"
    )
    print(f"[Base-MB] pos_weight={pos_weight.item():.4f} (n_pos={n_pos}, n_neg={n_neg})")

    best_state = None
    best_val_score = -float("inf")
    best_threshold = 0.5
    best_epoch = 0
    no_improve = 0
    rows: list[dict[str, float]] = []
    start_time = time.time()
    generator = torch.Generator().manual_seed(seed)
    grad_clip = train_cfg.get("grad_clip")

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model=model,
            data=data,
            sampler=train_sampler,
            train_idx=train_idx,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            pos_weight=pos_weight,
            generator=generator,
            grad_clip=float(grad_clip) if grad_clip is not None else None,
        )
        val_logits, val_y = predict_split(model, data, eval_sampler, val_idx, device, eval_batch_size)
        val_prob = torch.sigmoid(val_logits).numpy()
        val_y_np = val_y.numpy()
        threshold, _ = find_best_macro_f1_threshold(val_y_np, val_prob)
        val_metrics = compute_metrics_with_threshold(val_y_np, val_prob, threshold)
        val_score = float(val_metrics.get(select_metric, val_metrics["macro_f1"]))

        row = {
            "epoch": float(epoch),
            "train/loss": float(loss),
            "val/roc_auc": float(val_metrics.get("roc_auc", 0.0)),
            "val/auprc": float(val_metrics.get("auprc", 0.0)),
            "val/macro_f1": float(val_metrics.get("macro_f1", 0.0)),
            "val/threshold": float(threshold),
        }
        rows.append(row)

        if epoch == 1 or epoch % int(train_cfg.get("log_every", 5)) == 0:
            print(
                f"Epoch {epoch:3d} | loss={loss:.4f} | "
                f"val AUPRC={val_metrics['auprc']:.4f} | "
                f"val AUROC={val_metrics['roc_auc']:.4f} | "
                f"val MaF1(thr={threshold:.2f})={val_metrics['macro_f1']:.4f}"
            )

        if val_score > best_val_score:
            best_val_score = val_score
            best_threshold = float(threshold)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Base-MB] Early stopping at epoch {epoch} (best={best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_logits, train_y = predict_split(model, data, eval_sampler, train_idx, device, eval_batch_size)
    val_logits, val_y = predict_split(model, data, eval_sampler, val_idx, device, eval_batch_size)
    test_logits, test_y = predict_split(model, data, eval_sampler, test_idx, device, eval_batch_size)
    train_metrics = compute_metrics_with_threshold(train_y.numpy(), torch.sigmoid(train_logits).numpy(), best_threshold)
    val_metrics_final = compute_metrics_with_threshold(val_y.numpy(), torch.sigmoid(val_logits).numpy(), best_threshold)
    test_metrics = compute_metrics_with_threshold(test_y.numpy(), torch.sigmoid(test_logits).numpy(), best_threshold)

    checkpoint_dir = ensure_dir(get_checkpoint_dir(dataset_name, model_name, run_name, seed))
    checkpoint_path = checkpoint_dir / "base.pt"
    torch.save(best_state, checkpoint_path)

    default_cache = Path("artifacts") / "base_outputs" / dataset_name / model_name / f"seed_{seed}" / "base_outputs.pt"
    cache_path = get_base_output_override_cache_paths(default_cache, checkpoint_path)[0]
    ensure_dir(cache_path.parent)
    cache_dtype = str(train_cfg.get("cache_dtype", "float32")).lower()
    base_logits, base_z = compute_base_output_cache(
        model,
        data,
        eval_sampler,
        device,
        batch_size=cache_batch_size,
        cache_dtype=cache_dtype,
    )
    torch.save(
        {
            "base_logits": base_logits,
            "base_z": base_z,
            "meta": {
                "dataset": dataset_name,
                "model": model_name,
                "seed": int(seed),
                "num_nodes": int(data.x.shape[0]),
                "checkpoint_path": str(checkpoint_path),
                "git_hash": get_git_hash(),
                "training_mode": "cpu_neighbor_minibatch",
                "num_neighbors": fanouts,
                "eval_num_neighbors": eval_fanouts,
                "cache_dtype": cache_dtype,
            },
        },
        cache_path,
    )

    elapsed = time.time() - start_time
    log_dir = ensure_dir(get_logs_dir(dataset_name, model_name, run_name, seed))
    results_dir = ensure_dir(get_results_dir(dataset_name, model_name, run_name, seed))
    run_info = {
        "config": config,
        "seed": seed,
        "run_name": run_name,
        "git_hash": get_git_hash(),
        "checkpoint_path": str(checkpoint_path),
        "base_output_cache_path": str(cache_path),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics_final,
        "test_metrics": test_metrics,
        "epochs_trained": int(rows[-1]["epoch"]) if rows else 0,
        "best_epoch": best_epoch,
        "best_val_threshold": best_threshold,
        "best_val_score": best_val_score,
        "select_metric": select_metric,
        "pos_weight": float(pos_weight.item()),
        "elapsed_seconds": elapsed,
        "training_mode": "cpu_neighbor_minibatch",
        "minibatch": {
            "num_neighbors": fanouts,
            "eval_num_neighbors": eval_fanouts,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "cache_batch_size": cache_batch_size,
            "cache_dtype": cache_dtype,
        },
    }
    (log_dir / "base_training.json").write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n")
    (log_dir / "base_train_log.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (results_dir / "base_metrics.json").write_text(json.dumps(test_metrics, indent=2, sort_keys=True) + "\n")

    print(f"\n[Base-MB] Done in {elapsed:.1f}s | best_epoch={best_epoch} threshold={best_threshold:.3f}")
    print(f"  test AUPRC = {test_metrics.get('auprc', 0.0):.4f}")
    print(f"  test AUROC = {test_metrics.get('roc_auc', 0.0):.4f}")
    print(f"  test macro_f1 = {test_metrics.get('macro_f1', 0.0):.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base cache: {cache_path}")
    print(f"Metrics:    {results_dir / 'base_metrics.json'}")


if __name__ == "__main__":
    main()
