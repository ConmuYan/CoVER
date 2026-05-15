from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.relation_features import compute_relation_features, get_relation_schema, save_relation_feature_artifacts


def default_output_dir(dataset: str, model: str, seed: int, relation_set: str) -> Path:
    suffix = relation_set.lower()
    return Path("artifacts") / "relation_features" / dataset / model / f"seed_{seed}" / suffix


def seed_root_output_dir(dataset: str, model: str, seed: int) -> Path:
    return Path("artifacts") / "relation_features" / dataset / model / f"seed_{seed}"


def parse_relation_set(value: str, relation_schema: dict[str, dict[str, str]]) -> list[str]:
    value = value.lower()
    if value == "all":
        return list(relation_schema)
    mapping = {name.lower(): name for name in relation_schema}
    rels: list[str] = []
    unknown: list[str] = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key in mapping:
            rels.append(mapping[key])
        else:
            unknown.append(item.strip())
    if unknown:
        raise ValueError(
            f"Unknown relation(s): {', '.join(unknown)}. "
            f"Available relations: {', '.join(name.lower() for name in relation_schema)}"
        )
    if not rels:
        raise ValueError(
            "--relation_set must be all or a comma-separated subset of "
            f"{','.join(name.lower() for name in relation_schema)}"
        )
    return rels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage3_cover_lift.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--relation_set", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--z_threshold", type=float, default=2.0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    dataset_path = Path(args.dataset_path or config["dataset"]["path"])
    seed = int(args.seed if args.seed is not None else config["train"]["seed"])
    relation_schema = get_relation_schema(dataset_name)
    relation_names = parse_relation_set(args.relation_set, relation_schema)
    relation_set_name = (
        "all"
        if set(relation_names) == set(relation_schema)
        else "_".join(name.lower() for name in relation_names)
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
        dataset_name, model_name, seed, relation_set_name
    )

    from scipy.io import loadmat

    mat = loadmat(dataset_path)
    relation_matrices = {
        spec["mat_key"]: mat[spec["mat_key"]]
        for name, spec in relation_schema.items()
        if name in relation_names
    }

    split_mode = config["dataset"].get("split_mode", "supervised")
    train_ratio = config["dataset"].get("train_ratio", 0.7)
    val_test_ratio = config["dataset"].get("val_test_ratio", [1, 2])
    stratified = config["dataset"].get("stratified", False)
    data = load_fraud_dataset(
        dataset_name,
        path=dataset_path,
        seed=seed,
        split_mode=split_mode,
        train_ratio=train_ratio,
        val_test_ratio=val_test_ratio,
        stratified=stratified,
    )

    result = compute_relation_features(
        features=mat["features"],
        relation_matrices=relation_matrices,
        y=data.y,
        train_mask=data.train_mask,
        enabled_relations=relation_names,
        relation_schema=relation_schema,
        z_threshold=args.z_threshold,
    )
    result.meta.update({
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "dataset_path": str(dataset_path),
        "split_mode": split_mode,
        "train_ratio": train_ratio,
        "val_test_ratio": val_test_ratio,
        "stratified": bool(stratified),
        "relation_set": relation_set_name,
        "relation_schema": dataset_name,
    })
    save_relation_feature_artifacts(result, output_dir)
    root_dir = seed_root_output_dir(dataset_name, model_name, seed)
    root_copy_written = False
    if relation_set_name == "all" and root_dir != output_dir:
        save_relation_feature_artifacts(result, root_dir)
        root_copy_written = True
    print(json.dumps({
        "output_dir": str(output_dir),
        "rel_stats_path": str(output_dir / "rel_stats.pt"),
        "rel_tokens_path": str(output_dir / "rel_tokens.jsonl"),
        "rel_feature_meta_path": str(output_dir / "rel_feature_meta.json"),
        "root_copy_written": root_copy_written,
        "root_rel_stats_path": str(root_dir / "rel_stats.pt") if root_copy_written else "",
        "rel_dim": int(result.rel_stats.shape[1]),
        "num_nodes": int(result.rel_stats.shape[0]),
        "relations": relation_names,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
