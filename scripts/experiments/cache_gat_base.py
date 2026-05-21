#!/usr/bin/env python3
"""Pre-cache GAT base outputs to avoid OOM during teacher training.

For dense graphs (YelpChi ~7.7M edges), GAT's scatter_add in attention
consumes ~19 GB.  Combined with the teacher (~7.5 GB) this exceeds 24 GB.
By pre-caching the base forward pass, the teacher skips it entirely.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml
from data.load_fraud import load_fraud_dataset
from models.gnn import build_detector
from scripts.train_raer_teacher import (
    get_base_output_cache_path,
    get_base_output_override_cache_paths,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base_ckpt_path", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ds_cfg = config["dataset"]
    model_cfg = config["model"]
    ds_name = ds_cfg["name"]
    model_name = model_cfg["name"]
    seed = args.seed

    ckpt_path = Path(args.base_ckpt_path)
    if not ckpt_path.exists():
        print(f"[skip] base ckpt not found: {ckpt_path}")
        return

    default_cache = get_base_output_cache_path(ds_name, model_name, seed)
    override_paths = get_base_output_override_cache_paths(default_cache, ckpt_path)
    # Check if already cached
    for p in override_paths:
        if p.exists():
            payload = torch.load(p, map_location="cpu", weights_only=True)
            meta = payload.get("meta", {})
            if (int(meta.get("num_nodes", -1)) == 0  # will check after load
                    or str(meta.get("checkpoint_path", "")) == str(ckpt_path)):
                print(f"[skip] cached already: {p}")
                return

    data = load_fraud_dataset(
        **{k: v for k, v in ds_cfg.items()
           if k in ("name", "path", "format", "split_mode",
                    "train_ratio", "val_test_ratio", "scarcity_ratio")},
        seed=seed, stratified=True,
    )

    extra_kwargs = {}
    if "attention_heads" in model_cfg:
        extra_kwargs["attention_heads"] = model_cfg["attention_heads"]

    model = build_detector(
        name=model_name,
        in_channels=data.num_features,
        hidden_channels=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.5),
        **extra_kwargs,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    key = "model_state_dict" if "model_state_dict" in state else "model_state_dict"
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index, return_output=True)

    meta = {"num_nodes": data.num_nodes, "checkpoint_path": str(ckpt_path)}
    cache_path = override_paths[0]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "base_z": out.embeddings.cpu(),
        "base_logits": out.logits.cpu(),
        "meta": meta,
    }, cache_path)
    print(f"[cached] {cache_path} (nodes={data.num_nodes})")


if __name__ == "__main__":
    main()
