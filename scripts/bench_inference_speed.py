"""Inference efficiency benchmark (Idea 2D).

Measures wall-clock forward pass time (100 iterations, averaged) for:
  (a) frozen base only — sigmoid(base_logit)
  (b) base + full CoVER-REL teacher
  (c) base + distill adapter

CLI::

    python scripts/bench_inference_speed.py \\
        --config configs/phase2_reasoner/ablation/idea1_canonical_clsonly.yaml \\
        --seed 42 \\
        --device cuda:2 \\
        --teacher_ckpt artifacts/checkpoints/yelpchi/bwgnn/idea1_canonical_clsonly/seed_42/reasoner.pt \\
        --adapter_ckpt artifacts/checkpoints/yelpchi/bwgnn/idea2c_distill_adapter/seed_42/adapter.pt \\
        --base_ckpt_path artifacts/checkpoints/yelpchi/bwgnn/fixed_v1_100ep/seed_42/base.pt \\
        --n_iters 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.learned_extractor import build_learned_extractor
from evidence.relation_features import RELATION_SCHEMAS, load_relation_stats
from models.cover_rel_reasoner import CoVERRelReasoner
from models.gnn import build_detector
from models.rel_distill_adapter import RelDistillAdapter


def load_base_outputs(dataset_name, model_name, seed, data, device, ckpt_override=None, config=None):
    """Load cached base outputs (or re-run base model)."""
    from scripts.train_distill_adapter import get_base_output_cache_path
    from utils.paths import get_base_checkpoint_path

    ckpt_path = Path(ckpt_override) if ckpt_override else get_base_checkpoint_path(dataset_name, model_name, seed)
    cache_path = get_base_output_cache_path(dataset_name, model_name, seed)

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        meta = payload.get("meta", {})
        if (
            int(meta.get("num_nodes", -1)) == int(data.x.shape[0])
            and str(meta.get("checkpoint_path", "")) == str(ckpt_path)
        ):
            return payload["base_logits"].to(device).detach(), payload["base_z"].to(device).detach()

    # Fallback: run base model using passed config
    if config is None:
        raise RuntimeError("No cached base outputs and no config provided for fallback")

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
    base_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()
    with torch.no_grad():
        out = base_model(data.x.to(device), data.edge_index.to(device), return_output=True)
    return out.logits.detach(), out.embeddings.detach()


def load_relation_features(dataset_name, model_name, seed, num_nodes):
    rel_path = (
        Path("artifacts") / "relation_features" / dataset_name / model_name
        / f"seed_{seed}" / "all" / "rel_stats.pt"
    )
    rel_stats, rel_meta = load_relation_stats(rel_path, num_nodes=num_nodes)
    return rel_stats, rel_meta


def load_relation_adjs_for_extractor(
    dataset_name: str,
    dataset_path: str | Path,
    relation_names: list[str],
    num_nodes: int,
    device: torch.device,
) -> list[torch.Tensor]:
    from scipy.io import loadmat
    from scipy.sparse import coo_matrix, issparse

    schema = RELATION_SCHEMAS[dataset_name]
    mat = loadmat(str(dataset_path))
    adjs: list[torch.Tensor] = []
    for rel_name in relation_names:
        key = rel_name.upper()
        mat_key = schema[key]["mat_key"]
        m = mat[mat_key]
        sp = m.tocoo() if issparse(m) else coo_matrix(m)
        if sp.shape[0] != num_nodes or sp.shape[1] != num_nodes:
            raise ValueError(f"adj shape {sp.shape} != ({num_nodes}, {num_nodes}) for {key}")
        indices = torch.tensor(
            np.vstack([sp.row, sp.col]), dtype=torch.long, device=device,
        )
        values = torch.tensor(sp.data, dtype=torch.float32, device=device)
        adjs.append(torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce())
    return adjs


@torch.no_grad()
def build_learned_teacher_features(
    config: dict,
    data,
    relation_names: list[str],
    extractor_ckpt_path: Path,
    device: torch.device,
) -> torch.Tensor:
    p2_cfg = config["phase2_reasoner"]
    evidence_cfg = p2_cfg.get("evidence", {}) or {}
    ext_cfg = evidence_cfg.get("extractor", {}) or {}
    extractor = build_learned_extractor(
        x_dim=int(data.x.shape[1]),
        num_relations=len(relation_names),
        cfg={**ext_cfg, "out_dim_per_rel": int(ext_cfg.get("out_dim_per_rel", 9))},
    ).to(device)
    extractor.load_state_dict(torch.load(extractor_ckpt_path, weights_only=True, map_location=device))
    x_dev = data.x.to(device)
    relation_adjs = load_relation_adjs_for_extractor(
        dataset_name=config["dataset"]["name"],
        dataset_path=config["dataset"]["path"],
        relation_names=relation_names,
        num_nodes=int(data.x.shape[0]),
        device=device,
    )
    extractor.prepare(x_dev, relation_adjs)
    extractor.eval()
    return extractor(x_dev, data.train_mask.to(device), data.y.to(device).long()).detach()


@torch.no_grad()
def time_forward(fn, n_iters: int, warmup: int = 10) -> tuple[float, float, float]:
    """Time a forward function n_iters times. Returns (mean_ms, std_ms, total_s)."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    import numpy as np
    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), float(arr.sum() / 1000.0)


def main():
    parser = argparse.ArgumentParser(description="Inference speed benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--adapter_ckpt", required=True)
    parser.add_argument("--teacher_extractor_ckpt", default=None)
    parser.add_argument("--base_ckpt_path", type=str, default=None)
    parser.add_argument("--n_iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    p2_cfg = config["phase2_reasoner"]
    seed = args.seed
    device = torch.device(args.device)

    # Load data
    ds_cfg = config["dataset"]
    data = load_fraud_dataset(
        name=dataset_name,
        path=ds_cfg.get("path"),
        format=ds_cfg.get("format"),
        seed=seed,
        scarcity_ratio=float(ds_cfg.get("scarcity_ratio", 1.0)),
        split_mode=ds_cfg.get("split_mode", "supervised"),
        train_ratio=float(ds_cfg.get("train_ratio", 0.4)),
        val_test_ratio=list(ds_cfg.get("val_test_ratio", [1, 2])),
        stratified=bool(ds_cfg.get("stratified", True)),
    )

    # Load base outputs
    base_logits, base_z = load_base_outputs(
        dataset_name, model_name, seed, data, device,
        ckpt_override=args.base_ckpt_path, config=config,
    )

    # Load relation features
    rel_features, rel_meta = load_relation_features(
        dataset_name, model_name, seed, num_nodes=int(data.x.shape[0]),
    )
    rel_features = rel_features.to(device)

    relation_names = [
        str(name).upper() for name in
        p2_cfg.get("relation_names", rel_meta.get("relations", []))
    ]
    if not relation_names:
        relation_names = list(RELATION_SCHEMAS[dataset_name].keys())

    if args.teacher_extractor_ckpt is not None:
        rel_features = build_learned_teacher_features(
            config=config,
            data=data,
            relation_names=relation_names,
            extractor_ckpt_path=Path(args.teacher_extractor_ckpt),
            device=device,
        )
        print(f"[Bench] Using learned teacher features from {args.teacher_extractor_ckpt}")

    # Use test mask for inference (representative)
    test_mask = data.test_mask.to(device)
    n_test = int(test_mask.sum().item())
    bz_test = base_z[test_mask]
    bl_test = base_logits[test_mask]
    rf_test = rel_features[test_mask]

    print(f"[Bench] N_test={n_test} | n_iters={args.n_iters} | warmup={args.warmup}")
    print(f"[Bench] base_z shape: {bz_test.shape}, rel_features shape: {rf_test.shape}")

    # --- (a) Base only ---
    def fn_base():
        torch.sigmoid(bl_test)

    mean_a, std_a, total_a = time_forward(fn_base, args.n_iters, args.warmup)
    print(f"  (a) Base-only:        {mean_a:.3f} ± {std_a:.3f} ms  (total {total_a:.2f}s)")

    # --- (b) Full CoVER-REL teacher ---
    teacher = CoVERRelReasoner(
        base_z_dim=int(base_z.shape[1]),
        relation_names=relation_names,
        anchor_relation=p2_cfg.get("anchor_relation", relation_names[0]),
        rel_stat_dim=p2_cfg.get("rel_stat_dim", 9),
        rel_hidden_dim=p2_cfg.get("rel_hidden_dim", 64),
        rel_num_layers=p2_cfg.get("rel_num_layers", 2),
        rel_dropout=p2_cfg.get("rel_dropout", 0.3),
        tau_gate=p2_cfg.get("tau_gate", 0.7),
        delta_rel_max=p2_cfg.get("delta_rel_max", 2.0),
        gate_mode=p2_cfg.get("gate_mode", "softmax"),
        evidence_groups=p2_cfg.get("evidence_groups", None),
        expert_shared=p2_cfg.get("expert_shared", False),
        residual_activation=p2_cfg.get("residual_activation", "tanh"),
    ).to(device)
    teacher_raw = torch.load(args.teacher_ckpt, weights_only=False, map_location=device)
    teacher_state = teacher_raw["model_state_dict"] if isinstance(teacher_raw, dict) and "model_state_dict" in teacher_raw else teacher_raw
    teacher.load_state_dict(teacher_state)
    teacher.eval()

    def fn_teacher():
        out = teacher(bz_test, bl_test, rf_test)
        torch.sigmoid(out["final_logit"])

    mean_b, std_b, total_b = time_forward(fn_teacher, args.n_iters, args.warmup)
    print(f"  (b) Full CoVER-REL:   {mean_b:.3f} ± {std_b:.3f} ms  (total {total_b:.2f}s)")

    # --- (c) Distill adapter ---
    adapter_ckpt_path = Path(args.adapter_ckpt)
    if not adapter_ckpt_path.exists():
        print(f"  (c) Adapter ckpt NOT found: {adapter_ckpt_path} — SKIP")
        mean_c, std_c, total_c = float("nan"), float("nan"), float("nan")
    else:
        adapter = RelDistillAdapter(
            base_z_dim=int(base_z.shape[1]),
            num_relations=len(relation_names),
            rel_stat_dim=p2_cfg.get("rel_stat_dim", 9),
            hidden_dim=32,
            num_layers=2,
            dropout=0.3,  # must match training arch; eval() disables dropout
            delta_max=p2_cfg.get("delta_rel_max", 2.0),
        ).to(device)
        adapter.load_state_dict(torch.load(adapter_ckpt_path, weights_only=True, map_location=device))
        adapter.eval()

        def fn_adapter():
            out = adapter(bz_test, bl_test, rf_test)
            torch.sigmoid(out["final_logit"])

        mean_c, std_c, total_c = time_forward(fn_adapter, args.n_iters, args.warmup)
        print(f"  (c) Distill adapter:  {mean_c:.3f} ± {std_c:.3f} ms  (total {total_c:.2f}s)")

    # Speedup summary
    if mean_c == mean_c:  # not NaN
        print(f"\n[Bench] Speedup (b→c): {mean_b / mean_c:.2f}x")
        print(f"[Bench] Overhead (a→b): +{mean_b - mean_a:.3f} ms")
        print(f"[Bench] Overhead (a→c): +{mean_c - mean_a:.3f} ms")

    results = {
        "n_test": n_test,
        "n_iters": args.n_iters,
        "base_only_ms": round(mean_a, 3),
        "base_only_std_ms": round(std_a, 3),
        "teacher_ms": round(mean_b, 3),
        "teacher_std_ms": round(std_b, 3),
        "adapter_ms": round(mean_c, 3),
        "adapter_std_ms": round(std_c, 3),
        "speedup_b_to_c": round(mean_b / mean_c, 2) if mean_c == mean_c else None,
    }
    adapter_run_name = Path(args.adapter_ckpt).parent.parent.name
    out_path = Path("artifacts") / "results" / dataset_name / model_name / adapter_run_name / f"seed_{seed}" / "inference_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\n[Bench] Results saved to {out_path}")


if __name__ == "__main__":
    main()
