#!/usr/bin/env python3
"""E5: Inference efficiency benchmark.

Measures parameter count and forward-pass latency for the three-stage pipeline
(base detector → RAER-LREE teacher → CBR-Flash student) across all model types.

Uses real graph data for base detector timing; dummy tensors for teacher/student
(latency depends on tensor shape, not values).

Usage:
    python scripts/experiments/E5_benchmark_inference.py \
        --dataset yelpchi --split care_712 --seeds 42 123 456 789 2026 \
        --gpu 0 --warmup 5 --repeats 50

Output:
    artifacts/results/E5_efficiency/inference_benchmark.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml

from data.load_fraud import load_fraud_dataset
from models.gnn import build_detector
from models.raer_teacher import RAERTeacher
from models.cbr_flash_adapter import CBRFlashAdapter


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_latency(fn, warmup: int = 5, repeats: int = 50) -> float:
    """Measure median latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def load_base(config, data, ckpt_path, device):
    mc = config["model"]
    model = build_detector(
        name=mc["name"],
        in_channels=data.num_features,
        hidden_channels=mc.get("hidden_dim", 64),
        num_layers=mc.get("num_layers", 2),
        dropout=mc.get("dropout", 0.5),
        **({"attention_heads": mc["attention_heads"]} if "attention_heads" in mc else {}),
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(sd)
    return model.to(device).eval()


def load_teacher(config, ckpt_path, device):
    rc = config["raer_teacher"]
    model = RAERTeacher(
        base_z_dim=config["model"].get("hidden_dim", 64),
        relation_names=rc.get("relation_names", ["RUR", "RSR", "RTR"]),
        anchor_relation=rc.get("anchor_relation", "RUR"),
        rel_stat_dim=rc.get("rel_stat_dim", 9),
        rel_hidden_dim=rc.get("rel_hidden_dim", 64),
        rel_num_layers=rc.get("rel_num_layers", 2),
        rel_dropout=rc.get("rel_dropout", 0.3),
        tau_gate=rc.get("tau_gate", 0.7),
        delta_rel_max=rc.get("delta_rel_max", 2.0),
        gate_mode=rc.get("gate_mode", "softmax"),
        evidence_groups=rc.get("evidence_groups", ["A", "B", "C"]),
    )
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(raw["model_state_dict"] if "model_state_dict" in raw else raw)
    return model.to(device).eval()


def load_student(config, ckpt_path, device):
    rc = config["raer_teacher"]
    sc = config.get("cbr_flash", {})
    num_rel = len(rc.get("relation_names", ["RUR", "RSR", "RTR"]))
    model = CBRFlashAdapter(
        base_z_dim=config["model"].get("hidden_dim", 64),
        num_relations=num_rel,
        rel_stat_dim=rc.get("rel_stat_dim", 9),
        hidden_dim=sc.get("hidden_dim", 32),
        num_layers=sc.get("num_layers", 2),
        dropout=sc.get("dropout", 0.3),
        delta_max=sc.get("delta_max", 2.0),
    )
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(raw["model_state_dict"] if "model_state_dict" in raw else raw)
    return model.to(device).eval()


def load_metrics(ckpt_dir, seed):
    """Load training metrics from results directory."""
    for suffix in ["", f"/seed_{seed}"]:
        path = Path(ckpt_dir) / suffix / "metrics.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yelpchi")
    parser.add_argument("--split", default="care_712")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--config_dir",
                        default="configs/raer_fd/experiments/E1_benchmark/care_712")
    parser.add_argument("--ckpt_root", default="artifacts/checkpoints")
    parser.add_argument("--output",
                        default="artifacts/results/E5_efficiency/inference_benchmark.csv")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # run_name mapping
    if args.split == "care_712":
        base_rn, teacher_rn, student_rn = "base", "raer_lree", "cbr_flash"
    else:
        base_rn = f"base_{args.split}"
        teacher_rn = f"raer_lree_{args.split}"
        student_rn = f"cbr_flash_{args.split}"

    models = ["bwgnn", "sage", "gcn", "gat"]
    ckpt_root = Path(args.ckpt_root)
    config_dir = Path(args.config_dir)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Load dataset once
    base_cfg_path = config_dir / "base_detectors" / f"{args.dataset}_{models[0]}.yaml"
    if not base_cfg_path.exists():
        print(f"[ERROR] no base config: {base_cfg_path}")
        return
    with open(base_cfg_path) as f:
        ref_config = yaml.safe_load(f)
    ds = ref_config["dataset"]
    data = load_fraud_dataset(
        name=ds["name"], path=ds.get("path"),
        format=ds.get("format", "mat"),
        split_mode=ds.get("split_mode", "supervised"),
        train_ratio=ds.get("train_ratio", 0.4),
        val_test_ratio=ds.get("val_test_ratio", [1, 2]),
        scarcity_ratio=ds.get("scarcity_ratio", 1.0),
        seed=args.seeds[0], stratified=True,
    )
    x_dev = data.x.to(device)
    ei_dev = data.edge_index.to(device)
    N = data.num_nodes

    for model_name in models:
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

        bcfg = config_dir / "base_detectors" / f"{args.dataset}_{model_name}.yaml"
        tcfg = config_dir / "teacher" / "raer_lree" / f"{args.dataset}_{model_name}.yaml"
        scfg = config_dir / "student" / f"{args.dataset}_{model_name}.yaml"
        if not bcfg.exists():
            print(f"  [skip] no config"); continue
        with open(bcfg) as f:
            base_config = yaml.safe_load(f)

        hidden_dim = base_config["model"].get("hidden_dim", 64)
        rcfg_names = (yaml.safe_load(open(tcfg))["raer_teacher"].get("relation_names", ["RUR", "RSR", "RTR"])
                      if tcfg.exists() else ["RUR", "RSR", "RTR"])
        num_rel = len(rcfg_names)
        rel_stat_dim = 9
        rel_feat_dim = num_rel * rel_stat_dim

        for seed in args.seeds:
            print(f"  seed={seed}")
            row = {"dataset": args.dataset, "split": args.split, "model": model_name, "seed": seed}

            # Base
            bckpt = ckpt_root / args.dataset / model_name / base_rn / f"seed_{seed}" / "base.pt"
            if not bckpt.exists():
                print(f"    [skip] no base ckpt"); continue

            base_model = load_base(base_config, data, bckpt, device)
            row["base_params"] = count_params(base_model)
            row["base_latency_ms"] = measure_latency(
                lambda: base_model(x_dev, ei_dev),
                warmup=args.warmup, repeats=args.repeats,
            )
            # Get base_z for teacher/student
            with torch.no_grad():
                base_out = base_model(x_dev, ei_dev, return_output=True)
                base_z = base_out.embeddings.clone()
                base_logit = base_out.logits.clone()
            del base_model; torch.cuda.empty_cache()
            print(f"    base: {row['base_params']:,} params, {row['base_latency_ms']:.2f} ms")

            # Teacher
            tckpt = ckpt_root / args.dataset / model_name / teacher_rn / f"seed_{seed}" / "raer_teacher.pt"
            if tckpt.exists() and tcfg.exists():
                with open(tcfg) as f:
                    teacher_config = yaml.safe_load(f)
                teacher_model = load_teacher(teacher_config, tckpt, device)
                row["teacher_params"] = count_params(teacher_model)
                # Dummy relation features of correct shape
                dummy_rel = torch.randn(N, rel_feat_dim, device=device)
                row["teacher_latency_ms"] = measure_latency(
                    lambda: teacher_model(base_z, base_logit, dummy_rel),
                    warmup=args.warmup, repeats=args.repeats,
                )
                del teacher_model; torch.cuda.empty_cache()
                print(f"    teacher: {row['teacher_params']:,} params, {row['teacher_latency_ms']:.2f} ms")
            else:
                print(f"    [skip] no teacher")

            # Student
            sckpt = ckpt_root / args.dataset / model_name / student_rn / f"seed_{seed}" / "cbr_flash_student.pt"
            if sckpt.exists() and scfg.exists():
                with open(scfg) as f:
                    student_config = yaml.safe_load(f)
                student_model = load_student(student_config, sckpt, device)
                row["student_params"] = count_params(student_model)
                dummy_rel = torch.randn(N, rel_feat_dim, device=device)
                row["student_latency_ms"] = measure_latency(
                    lambda: student_model(base_z, base_logit, dummy_rel),
                    warmup=args.warmup, repeats=args.repeats,
                )
                del student_model; torch.cuda.empty_cache()
                print(f"    student: {row['student_params']:,} params, {row['student_latency_ms']:.2f} ms")
            else:
                print(f"    [skip] no student")

            # Ratios
            if "student_params" in row and "teacher_params" in row:
                row["compression_ratio"] = row["teacher_params"] / max(row["student_params"], 1)
            if "student_params" in row and "base_params" in row:
                row["student_vs_base_pct"] = f"{row['student_params'] / max(row['base_params'], 1) * 100:.1f}%"

            # Load metrics from results files
            for stage, rn in [("base", base_rn), ("teacher", teacher_rn), ("student", student_rn)]:
                rdir = Path(f"artifacts/results/{args.dataset}/{model_name}/{rn}/seed_{seed}")
                m = load_metrics(rdir, seed)
                if m:
                    for k in ["test_auprc", "test_auroc", "test_macro_f1"]:
                        if k in m:
                            row[f"{stage}_{k}"] = m[k]

            results.append(row)
            del base_z, base_logit; torch.cuda.empty_cache()

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"\nSaved to {args.output}")

        # Summary
        print(f"\n{'='*95}")
        print(f"{'Model':<8} {'Base#':>10} {'Teach#':>10} {'Stud#':>10} "
              f"{'Base ms':>8} {'Teach ms':>9} {'Stud ms':>8} {'Compress':>9}")
        print("-" * 95)
        for r in results:
            if r.get("seed") != args.seeds[0]:
                continue
            print(f"{r['model']:<8} "
                  f"{r.get('base_params',0):>10,} "
                  f"{r.get('teacher_params',0):>10,} "
                  f"{r.get('student_params',0):>10,} "
                  f"{r.get('base_latency_ms',0):>8.2f} "
                  f"{r.get('teacher_latency_ms',0):>9.2f} "
                  f"{r.get('student_latency_ms',0):>8.2f} "
                  f"{r.get('compression_ratio',0):>8.1f}x")
    else:
        print("No results.")


if __name__ == "__main__":
    main()
