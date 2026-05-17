"""Pre-compute and cache GAT base outputs (base_logits, base_z) per seed.

Avoids the trainer's `set_seed → torch.use_deterministic_algorithms(True)` path
which forces PyG scatter into a memory-heavy deterministic algorithm (8-19 GB
peak vs 4 GB normal). Frozen base + eval() + no_grad() is deterministic
by construction, so we can safely skip the deterministic flag here.

Writes the same cache format the trainer reads, so subsequent
`scripts/train_phase2_reasoner.py --base_ckpt_path ...` calls find it and skip
the in-trainer forward entirely.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import yaml

from data.load_fraud import load_fraud_dataset
from models.gnn import build_detector
from utils.paths import ensure_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["yelpchi", "amazon"])
    p.add_argument("--model", required=True, choices=["bwgnn", "sage", "gcn", "gat"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--attention_heads", type=int, default=1, help="GAT only")
    p.add_argument("--ckpt_subdir", default="fixed_v1_100ep")
    args = p.parse_args()

    ds_path = {
        "yelpchi": "datasets/YelpChi.mat",
        "amazon":  "datasets/Amazon.mat",
    }[args.dataset]

    device = torch.device(args.device)
    for seed in args.seeds:
        ckpt_path = Path(f"artifacts/checkpoints/{args.dataset}/{args.model}/{args.ckpt_subdir}/seed_{seed}/base.pt")
        if not ckpt_path.exists():
            print(f"  seed {seed}: SKIP (ckpt missing {ckpt_path})")
            continue
        # Trainer reads cache name from ckpt_path.parent.name + ckpt_path.stem,
        # NOT from the ckpt_subdir, when `--base_ckpt_path` is passed.
        # See load_frozen_base() in scripts/train_phase2_reasoner.py.
        cache_path = Path(
            f"artifacts/base_outputs/{args.dataset}/{args.model}/seed_{seed}/"
            f"_override_{ckpt_path.parent.name}_{ckpt_path.stem}.pt"
        )
        if cache_path.exists():
            print(f"  seed {seed}: SKIP (cache exists {cache_path})")
            continue
        ensure_dir(cache_path.parent)

        data = load_fraud_dataset(
            name=args.dataset, path=ds_path, format="mat",
            seed=seed, scarcity_ratio=1.0, split_mode="supervised",
            train_ratio=0.4, val_test_ratio=[1, 2], stratified=True,
        )
        extra: dict = {}
        if args.model == "gat":
            extra["attention_heads"] = args.attention_heads
        model = build_detector(
            name=args.model,
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            **extra,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
        for p_ in model.parameters():
            p_.requires_grad = False
        model.eval()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device), return_output=True)
            base_logits = out.logits.detach().cpu()
            base_z = out.embeddings.detach().cpu()
        peak = torch.cuda.max_memory_allocated(device) / 1e9

        torch.save({
            "base_logits": base_logits,
            "base_z": base_z,
            "meta": {
                "dataset": args.dataset,
                "model": args.model,
                "seed": int(seed),
                "num_nodes": int(data.x.shape[0]),
                "checkpoint_path": str(ckpt_path),
                "git_hash": "pre_cache_no_deterministic",
            },
        }, cache_path)
        print(f"  seed {seed}: cached {cache_path} (peak={peak:.2f} GB)")
        del model, out, data
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
