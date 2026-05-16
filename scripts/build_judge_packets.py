from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.evidence_packet import build_judge_packet, serialize_evidence_packet
from evidence.relation_features import load_relation_stats
from models.gnn import build_detector
from models.reasoner import EvidenceReasoner
from scripts.train_stage3 import load_stage2_sources, prepare_targets
from utils.paths import ensure_dir, get_base_checkpoint_path


DEFAULT_OUTPUT_ROOT = Path("artifacts/judge_packets")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_relation_tokens(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["node_id"]): row for row in load_jsonl(path)}


def select_nodes(
    data,
    num_nodes: int,
    seed: int,
    selection_mode: str = "random",
    base_logits: torch.Tensor | None = None,
) -> list[int]:
    """Pick nodes to send to LLM judge.

    Modes:
        random          — original per-split balanced random (default).
        base_uncertain  — pick nodes with smallest |base_logit| (most
                          uncertain under base classifier). Targets the
                          subset where CoVER stands to gain the most.
                          Restricted to train+val+test (eligible) nodes.
    """
    if num_nodes <= 0:
        return list(range(data.x.shape[0]))

    if selection_mode == "base_uncertain":
        if base_logits is None:
            raise ValueError("base_logits required when selection_mode='base_uncertain'")
        eligible = (data.train_mask | data.val_mask | data.test_mask).cpu()
        elig_idx = torch.where(eligible)[0]
        abs_logit = base_logits.detach().cpu().abs().view(-1)
        # Lowest |logit| first (most uncertain)
        order = abs_logit[elig_idx].argsort()
        selected_idx = elig_idx[order[:num_nodes]]
        selected = [int(x) for x in selected_idx.tolist()]
        return sorted(dict.fromkeys(selected))

    # Original random mode
    gen = torch.Generator().manual_seed(seed)
    per_split = max(1, num_nodes // 3)
    selected: list[int] = []
    for mask in (data.train_mask, data.val_mask, data.test_mask):
        idx = torch.where(mask.cpu())[0]
        perm = idx[torch.randperm(idx.numel(), generator=gen)]
        selected.extend(int(x) for x in perm[:per_split])
    remaining = num_nodes - len(selected)
    if remaining > 0:
        all_idx = torch.tensor([idx for idx in range(data.x.shape[0]) if idx not in set(selected)])
        perm = all_idx[torch.randperm(all_idx.numel(), generator=gen)]
        selected.extend(int(x) for x in perm[:remaining])
    return sorted(dict.fromkeys(selected))


def build_reasoner_from_stage3(stage3_config: dict[str, Any], z_dim: int, checkpoint: Path, device: torch.device):
    reasoner = EvidenceReasoner(
        z_dim=z_dim,
        hidden_dim=stage3_config.get("hidden_dim", 128),
        rho=stage3_config.get("rho", 0.1),
        gate_mode=stage3_config.get("gate_mode", "safe_residual"),
        delta_scale=stage3_config.get("delta_scale", 2.0),
        gate_bias_init=stage3_config.get("gate_bias_init", -2.0),
        residual_init_zero=stage3_config.get("residual_init_zero", True),
        latent_dim=stage3_config.get("latent_dim", 256),
        relation_dim=stage3_config.get("relation_dim", 0),
        relation_hidden_dim=stage3_config.get("relation_hidden_dim", 32),
        relation_fusion_mode=stage3_config.get("relation_fusion_mode", "concat"),
        relation_names=[str(x).upper() for x in stage3_config.get("relation_names", [])],
        relation_stat_dim=stage3_config.get("relation_stat_dim"),
        anchor_relation=stage3_config.get("anchor_relation"),
        optional_relations=[str(x).upper() for x in stage3_config.get("optional_relations", [])],
        relation_dropout=0.0,
        gate_hidden_dim=stage3_config.get("gate_hidden_dim", 64),
        gate_temperature=stage3_config.get("gate_temperature", 1.0),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    reasoner.load_state_dict(state, strict=False)
    reasoner.eval()
    return reasoner


@torch.no_grad()
def compute_gate_values(config: dict[str, Any], stage3_config: dict[str, Any], data, rel_stats: torch.Tensor, device: torch.device, base_run_name: str = "base"):
    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    seed = int(stage3_config["seed"])
    model_cfg = config["model"]
    detector_kwargs = dict(
        name=model_cfg["name"],
        in_channels=data.x.shape[1],
        hidden_channels=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.5),
    )
    if "num_bands" in model_cfg:
        detector_kwargs["num_bands"] = model_cfg["num_bands"]
    if "attention_heads" in model_cfg:
        detector_kwargs["attention_heads"] = model_cfg["attention_heads"]
    model = build_detector(**detector_kwargs).to(device)
    base_path = (
        Path("artifacts/checkpoints") / dataset_name / model_name / base_run_name / f"seed_{seed}" / "base.pt"
    )
    if not base_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_path}")
    model.load_state_dict(torch.load(base_path, map_location=device, weights_only=False))
    model.eval()
    output = model(data.x.to(device), data.edge_index.to(device), return_output=True)
    base_logits = output.logits.detach()
    z = output.embeddings.detach()

    cards, accepted, _ = load_stage2_sources(
        dataset_name,
        model_name,
        stage3_config.get("stage2_run_names", [stage3_config.get("stage2_run_name", "rule")]),
        seed,
    )
    targets = prepare_targets(data.x.shape[0], accepted, cards, data.train_mask)
    reasoner_path = Path("artifacts/checkpoints") / dataset_name / model_name / stage3_config["run_name"] / f"seed_{seed}" / "reasoner.pt"
    reasoner = build_reasoner_from_stage3(stage3_config, z.shape[1], reasoner_path, device)
    outputs = reasoner(
        z,
        base_logits,
        targets["evidence_token_ids"].to(device),
        relation_features=rel_stats.to(device),
        return_debug=True,
    )
    return (
        outputs.get("relation_gate_values", torch.empty(data.x.shape[0], 0, device=device)).cpu(),
        {
            int(node_id): card.get("reasoning", {})
            for node_id, card in cards.items()
            if isinstance(card, dict)
        },
        base_logits.cpu(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cover-rel-gj/stage3_legacy/stage3_cover_rel_gate_nollm.yaml")
    parser.add_argument("--gate_run_name", default="cover_rel_anchor_gate_nollm")
    parser.add_argument("--output_run_name", default="cover_rel_judge")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--base_run_name",
        type=str,
        default="base",
        help="Phase 1 base checkpoint run_name (e.g. 'fixed_v1_100ep').",
    )
    parser.add_argument("--num_nodes", type=int, default=120)
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="random",
        choices=["random", "base_uncertain"],
        help="Node selection strategy. 'random' = per-split balanced random (default); "
             "'base_uncertain' = top-K nodes by smallest |base_logit| (most uncertain under base).",
    )
    parser.add_argument("--primary_relation", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--skip_stage_deps",
        action="store_true",
        help="Skip stage2/stage3 artifacts (cards + stage3 reasoner). "
             "Uses uniform gate_values and empty graph_reasoning. Useful when "
             "rebuilding packets after artifacts cleanup — only base.pt + "
             "relation_features are required.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    seed = int(args.seed if args.seed is not None else config["train"]["seed"])
    config["train"]["seed"] = seed
    device = torch.device(args.device or config["train"].get("device", "cpu"))

    data = load_fraud_dataset(
        dataset_name,
        path=config["dataset"].get("path"),
        seed=seed,
        split_mode=config["dataset"].get("split_mode", "supervised"),
        train_ratio=config["dataset"].get("train_ratio", 0.4),
        val_test_ratio=config["dataset"].get("val_test_ratio", [1, 2]),
        stratified=config["dataset"].get("stratified", False),
    )
    if args.skip_stage_deps:
        # Minimal path: only need base.pt + relation_features. Uses uniform
        # gate_values and empty graph_reasoning. anchor_relation falls back
        # to relation_names[0] if not in config.
        from utils.paths import get_checkpoint_dir
        model_cfg = config["model"]
        detector_kwargs = dict(
            name=model_cfg["name"],
            in_channels=data.x.shape[1],
            hidden_channels=model_cfg.get("hidden_dim", 64),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.5),
        )
        if "num_bands" in model_cfg:
            detector_kwargs["num_bands"] = model_cfg["num_bands"]
        if "attention_heads" in model_cfg:
            detector_kwargs["attention_heads"] = model_cfg["attention_heads"]
        base_model = build_detector(**detector_kwargs).to(device)
        base_path = get_checkpoint_dir(dataset_name, model_name, args.base_run_name, seed) / "base.pt"
        if not base_path.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {base_path}")
        base_model.load_state_dict(torch.load(base_path, map_location=device, weights_only=False))
        base_model.eval()
        with torch.no_grad():
            base_out = base_model(data.x.to(device), data.edge_index.to(device), return_output=True)
            base_logits = base_out.logits.detach().cpu()
        # Use the user-supplied default config for relation schema metadata
        rel_path = Path(config.get("relation_features_path") or
                        f"artifacts/relation_features/{dataset_name}/{model_name}/seed_{seed}/rel_stats.pt")
        rel_stats, rel_meta = load_relation_stats(rel_path, num_nodes=data.x.shape[0])
        token_path = rel_path.parent / "rel_tokens.jsonl"
        rel_tokens = load_relation_tokens(token_path)
        relation_names = [str(x).upper() for x in rel_meta["relations"]]
        stat_names = [str(x) for x in rel_meta["stat_names_per_relation"]]
        primary = (args.primary_relation or relation_names[0]).upper()
        # Uniform gate_values
        gate_values = torch.full((data.x.shape[0], len(relation_names)), 1.0 / len(relation_names))
        card_reasoning: dict[int, dict] = {}
        print(f"[skip_stage_deps] base=ok, rel_features=ok, gate=uniform({1/len(relation_names):.3f}), graph_reasoning=empty")
    else:
        stage3_config_path = Path("artifacts/logs") / dataset_name / model_name / args.gate_run_name / f"seed_{seed}" / "stage3_config.json"
        stage3_config = json.loads(stage3_config_path.read_text())
        stage3_config["run_name"] = args.gate_run_name
        rel_path = Path(stage3_config["relation_features_path"])
        rel_stats, rel_meta = load_relation_stats(rel_path, num_nodes=data.x.shape[0])
        token_path = rel_path.parent / "rel_tokens.jsonl"
        rel_tokens = load_relation_tokens(token_path)
        relation_names = [str(x).upper() for x in rel_meta["relations"]]
        stat_names = [str(x) for x in rel_meta["stat_names_per_relation"]]
        primary = (args.primary_relation or stage3_config.get("anchor_relation") or relation_names[0]).upper()
        gate_values, card_reasoning, base_logits = compute_gate_values(config, stage3_config, data, rel_stats, device, base_run_name=args.base_run_name)

    nodes = select_nodes(
        data,
        args.num_nodes,
        seed,
        selection_mode=args.selection_mode,
        base_logits=base_logits,
    )
    print(f"[selection_mode={args.selection_mode}] picked {len(nodes)} nodes")
    packets: list[dict[str, Any]] = []
    for node_id in nodes:
        gate_by_relation = {
            relation: float(gate_values[node_id, idx].item())
            for idx, relation in enumerate(relation_names)
            if gate_values.numel() > 0
        }
        token_row = rel_tokens.get(node_id, {})
        packets.append(build_judge_packet(
            node_id=node_id,
            dataset=dataset_name,
            relation_schema=relation_names,
            primary_relation=primary,
            relation_fusion_source=args.gate_run_name,
            relation_stats=rel_stats[node_id].numpy(),
            relation_stat_names=stat_names,
            relation_tokens=token_row.get("rel_tokens", []),
            gate_values=gate_by_relation,
            graph_reasoning=card_reasoning.get(node_id, {}),
        ))

    output_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_OUTPUT_ROOT / dataset_name / model_name / args.output_run_name / f"seed_{seed}"
    )
    ensure_dir(output_dir)
    write_jsonl(output_dir / "judge_packets.jsonl", packets)
    (output_dir / "judge_packet_texts.jsonl").write_text(
        "\n".join(serialize_evidence_packet(packet) for packet in packets) + "\n"
    )
    meta = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "num_packets": len(packets),
        "primary_relation": primary,
        "relation_schema": relation_names,
        "relation_fusion_source": args.gate_run_name,
        "score_blind": True,
        "target_label_used": False,
        "test_label_used": False,
        "relation_feature_meta_path": str(rel_path.parent / "rel_feature_meta.json"),
    }
    (output_dir / "judge_packet_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), **meta}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
