"""Sweep rho and lambda_evi for the evidence-conditioned reasoner.

Reuses existing base checkpoints and ERR cache.  Does NOT re-run Stage 1,
does NOT re-generate Stage 2 ERR, and does NOT call any LLM.

Usage
-----
python scripts/run_reasoner_sweep.py \
    --dataset yelpchi \
    --model bwgnn \
    --run_name qwen \
    --seeds 123 456 789 \
    --rho_values 0.0 0.05 0.1 0.2 0.3 \
    --lambda_values 0.0 0.1 0.3 0.5 \
    --train_gpus 2

Output structure
----------------
artifacts/sweeps/{dataset}/{model}/{base}_rho{r}_lambda{l}/seed_{s}/
  stage3_metrics.json       -- test metrics at fixed 0.5 threshold
  calibrated_metrics.json   -- metrics at fixed / val_f1 / val_macro_f1
  reasoner_diagnosis.json   -- gate/residual stats and rho=0 check
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.vocab import encode_err_targets, encode_reasoning, get_evidence_slots, get_reason_types
from evidence.schema import ERR
from models.gnn import build_detector
from models.reasoner import EvidenceReasoner
from training.metrics import compute_metrics
from utils.paths import (
    ARTIFACTS_ROOT,
    ensure_dir,
    get_base_checkpoint_path,
    get_err_cache_dir,
    get_results_dir,
    get_checkpoint_dir,
)
from utils.threshold import calibrate_and_evaluate


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _sweep_run_name(base: str, rho: float, lambda_evi: float) -> str:
    return f"{base}_rho{rho}_lambda{lambda_evi}"


def _sweep_dir(dataset: str, model: str, sweep_name: str, seed: int) -> Path:
    return ARTIFACTS_ROOT / "sweeps" / dataset / model / sweep_name / f"seed_{seed}"


def _resolve_config(dataset: str, model: str) -> Path:
    p = Path(__file__).parent.parent / "configs" / f"{dataset}_{model}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return p


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# ERR cache mirroring (symlinks to avoid copying large caches)
# ---------------------------------------------------------------------------

def _mirror_err_cache(source: Path, target: Path) -> None:
    """Symlink or copy ERR cache from *source* run into *target* run dir."""
    if not source.exists():
        raise FileNotFoundError(f"Source ERR cache not found: {source}")
    if target.exists() and any(target.iterdir()):
        return  # already mirrored

    for item in source.rglob("*"):
        rel = item.relative_to(source)
        dst = target / rel
        if item.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(item.resolve())
        except OSError:
            shutil.copy2(item, dst)


# ---------------------------------------------------------------------------
# Inline evaluation with calibration
# ---------------------------------------------------------------------------

def _evaluate_with_calibration(
    config: dict,
    dataset_name: str,
    model_name: str,
    run_name: str,
    seed: int,
    device: torch.device,
    debug: bool,
) -> dict:
    """Load trained reasoner, evaluate with multiple threshold modes."""
    if debug:
        data = load_fraud_dataset("tiny", seed=seed)
    else:
        data = load_fraud_dataset(
            dataset_name,
            path=config["dataset"].get("path"),
            seed=seed,
            split_mode=config["dataset"].get("split_mode", "supervised"),
            train_ratio=config["dataset"].get("train_ratio", 0.7),
            val_test_ratio=config["dataset"].get("val_test_ratio", [1, 2]),
        )

    base_model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    ).to(device)

    base_ckpt = get_base_checkpoint_path(dataset_name, model_name, seed)
    if base_ckpt.exists():
        base_model.load_state_dict(torch.load(base_ckpt, weights_only=True))
    base_model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    with torch.no_grad():
        out = base_model(x, edge_index, return_output=True)
        base_logits = out.logits
        z = out.embeddings

    rho = config.get("reasoner", {}).get("rho", 0.3)
    reasoner = EvidenceReasoner(
        z_dim=z.shape[1],
        hidden_dim=config.get("reasoner", {}).get("hidden_dim", 128),
        rho=rho,
    ).to(device)

    ckpt_dir = get_checkpoint_dir(dataset_name, model_name, run_name, seed)
    reasoner_path = ckpt_dir / "reasoner.pt"
    if not reasoner_path.exists():
        raise FileNotFoundError(f"Reasoner checkpoint not found: {reasoner_path}")
    reasoner.load_state_dict(torch.load(reasoner_path, weights_only=True))
    reasoner.eval()

    err_dir = get_err_cache_dir(dataset_name, model_name, run_name, seed)
    cards: dict[int, dict] = {}
    cards_path = err_dir / "evidence_cards.jsonl"
    if cards_path.exists():
        with open(cards_path) as f:
            for line in f:
                c = json.loads(line)
                cards[c["node_id"]] = c

    num_slots = len(get_evidence_slots())
    num_nodes = data.x.shape[0]
    evidence_token_ids = torch.zeros(num_nodes, num_slots, dtype=torch.long)
    for nid, card in cards.items():
        if nid < num_nodes:
            evidence_token_ids[nid] = encode_reasoning(card.get("reasoning", {}))
    evidence_token_ids = evidence_token_ids.to(device)

    y = data.y.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    with torch.no_grad():
        val_out = reasoner(z[val_mask], base_logits[val_mask], evidence_token_ids[val_mask])
        test_out = reasoner(z[test_mask], base_logits[test_mask], evidence_token_ids[test_mask])

    y_val_np = y[val_mask].cpu().numpy()
    prob_val = torch.sigmoid(val_out["final_logit"]).cpu().numpy()
    y_test_np = y[test_mask].cpu().numpy()
    prob_test = torch.sigmoid(test_out["final_logit"]).cpu().numpy()

    fixed_metrics = compute_metrics(y_test_np, prob_test)

    cal_f1 = calibrate_and_evaluate(
        y_val_np, prob_val, y_test_np, prob_test, metric="f1",
    )
    cal_macro_f1 = calibrate_and_evaluate(
        y_val_np, prob_val, y_test_np, prob_test, metric="macro_f1",
    )

    return {
        "rho": rho,
        "fixed": {
            "threshold": 0.5,
            "test": fixed_metrics,
        },
        "val_f1": {
            "best_threshold": cal_f1["best_threshold"],
            "val": cal_f1["val"],
            "test": cal_f1["test"],
        },
        "val_macro_f1": {
            "best_threshold": cal_macro_f1["best_threshold"],
            "val": cal_macro_f1["val"],
            "test": cal_macro_f1["test"],
        },
    }


# ---------------------------------------------------------------------------
# Inline reasoner diagnosis
# ---------------------------------------------------------------------------

def _diagnose_reasoner(
    config: dict,
    dataset_name: str,
    model_name: str,
    run_name: str,
    seed: int,
    base_run_name: str,
    base_metrics: dict,
    stage3_metrics: dict,
    device: torch.device,
    debug: bool,
) -> dict:
    """Load trained reasoner and produce gate/residual diagnostic stats."""
    if debug:
        data = load_fraud_dataset("tiny", seed=seed)
    else:
        data = load_fraud_dataset(
            dataset_name,
            path=config["dataset"].get("path"),
            seed=seed,
            split_mode=config["dataset"].get("split_mode", "supervised"),
            train_ratio=config["dataset"].get("train_ratio", 0.7),
            val_test_ratio=config["dataset"].get("val_test_ratio", [1, 2]),
        )

    base_model = build_detector(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=config["model"].get("hidden_dim", 64),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    ).to(device)

    base_ckpt = get_base_checkpoint_path(dataset_name, model_name, seed)
    if base_ckpt.exists():
        base_model.load_state_dict(torch.load(base_ckpt, weights_only=True))
    base_model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    with torch.no_grad():
        out = base_model(x, edge_index, return_output=True)
        base_logits = out.logits
        z = out.embeddings

    rho = config.get("reasoner", {}).get("rho", 0.3)
    reasoner = EvidenceReasoner(
        z_dim=z.shape[1],
        hidden_dim=config.get("reasoner", {}).get("hidden_dim", 128),
        rho=rho,
    ).to(device)

    ckpt_dir = get_checkpoint_dir(dataset_name, model_name, run_name, seed)
    reasoner_path = ckpt_dir / "reasoner.pt"
    if not reasoner_path.exists():
        raise FileNotFoundError(f"Reasoner checkpoint not found: {reasoner_path}")
    reasoner.load_state_dict(torch.load(reasoner_path, weights_only=True))
    reasoner.eval()

    # Load evidence cards
    err_dir = get_err_cache_dir(dataset_name, model_name, run_name, seed)
    cards: dict[int, dict] = {}
    cards_path = err_dir / "evidence_cards.jsonl"
    if cards_path.exists():
        with open(cards_path) as f:
            for line in f:
                c = json.loads(line)
                cards[c["node_id"]] = c

    num_slots = len(get_evidence_slots())
    num_nodes = data.x.shape[0]
    evidence_token_ids = torch.zeros(num_nodes, num_slots, dtype=torch.long)
    for nid, card in cards.items():
        if nid < num_nodes:
            evidence_token_ids[nid] = encode_reasoning(card.get("reasoning", {}))
    evidence_token_ids = evidence_token_ids.to(device)

    test_mask = data.test_mask.to(device)

    with torch.no_grad():
        outputs = reasoner(
            z[test_mask], base_logits[test_mask],
            evidence_token_ids[test_mask],
            return_debug=True,
        )

    gate = outputs["gate"].cpu().numpy().flatten()
    residual_raw = outputs["residual_raw"].cpu().numpy().flatten()
    residual = outputs["residual"].cpu().numpy().flatten()
    final_logit = outputs["final_logit"].cpu().numpy().flatten()
    base_logit_np = base_logits[test_mask].cpu().numpy().flatten()

    reason_types = get_reason_types()
    type_logits = outputs["type_logits"].cpu().numpy()
    type_probs = np.exp(type_logits) / np.exp(type_logits).sum(axis=-1, keepdims=True)
    pred_types = type_probs.argmax(axis=-1)
    type_dist: dict[str, int] = {}
    for t in pred_types:
        name = reason_types[t] if t < len(reason_types) else f"type_{t}"
        type_dist[name] = type_dist.get(name, 0) + 1

    def _deltas(a: dict, b: dict) -> dict[str, float]:
        keys = ["roc_auc", "auprc", "f1", "macro_f1", "precision", "recall"]
        return {k: round(float(a.get(k, 0.0)) - float(b.get(k, 0.0)), 6) for k in keys}

    rho_check: dict = {"rho": rho, "is_zero": rho == 0.0}
    if rho == 0.0:
        max_diff = float(np.max(np.abs(final_logit - base_logit_np)))
        rho_check["max_diff_from_base"] = max_diff
        rho_check["passes"] = max_diff < 1e-6

    return {
        "dataset": dataset_name,
        "model": model_name,
        "run_name": run_name,
        "source_err_cache": base_run_name,
        "seed": seed,
        "rho": rho,
        "num_test_nodes": int(test_mask.sum().item()),
        "gate": {
            "mean": float(np.mean(gate)),
            "std": float(np.std(gate)),
            "min": float(np.min(gate)),
            "max": float(np.max(gate)),
            "median": float(np.median(gate)),
            "frac_positive": float(np.mean(gate > 0)),
        },
        "residual_raw": {
            "mean": float(np.mean(residual_raw)),
            "std": float(np.std(residual_raw)),
            "min": float(np.min(residual_raw)),
            "max": float(np.max(residual_raw)),
        },
        "residual_scaled": {
            "mean": float(np.mean(residual)),
            "std": float(np.std(residual)),
            "min": float(np.min(residual)),
            "max": float(np.max(residual)),
        },
        "final_logit": {
            "mean": float(np.mean(final_logit)),
            "std": float(np.std(final_logit)),
            "min": float(np.min(final_logit)),
            "max": float(np.max(final_logit)),
        },
        "base_logit": {
            "mean": float(np.mean(base_logit_np)),
            "std": float(np.std(base_logit_np)),
        },
        "predicted_type_distribution": type_dist,
        "rho_zero_check": rho_check,
        "delta_vs_base": _deltas(stage3_metrics, base_metrics),
    }


# ---------------------------------------------------------------------------
# Single sweep-point job
# ---------------------------------------------------------------------------

def run_job(
    *,
    dataset: str,
    model: str,
    base_run_name: str,
    seed: int,
    rho: float,
    lambda_evi: float,
    train_gpus: list[str],
    dry_run: bool,
    skip_existing: bool,
    debug: bool,
) -> tuple[str, bool]:
    """Run one (rho, lambda, seed) sweep point.

    Returns (sweep_run_name, success).
    """
    sweep_name = _sweep_run_name(base_run_name, rho, lambda_evi)
    out_dir = _sweep_dir(dataset, model, sweep_name, seed)

    if skip_existing:
        required = [out_dir / "stage3_metrics.json",
                    out_dir / "calibrated_metrics.json",
                    out_dir / "reasoner_diagnosis.json"]
        if all(p.exists() for p in required):
            print(f"[SKIP] {sweep_name}/seed_{seed} (all outputs exist)")
            return sweep_name, True

    config_path = _resolve_config(dataset, model)
    base_config = yaml.safe_load(open(config_path))
    base_ckpt = get_base_checkpoint_path(dataset, model, seed)
    if not base_ckpt.exists() and not dry_run:
        print(f"[SKIP] Base checkpoint not found: {base_ckpt}")
        return sweep_name, False

    source_cache = get_err_cache_dir(dataset, model, base_run_name, seed)
    if not (source_cache / "evidence_cards.jsonl").exists() and not dry_run:
        print(f"[SKIP] ERR cache not found: {source_cache}")
        return sweep_name, False

    if dry_run:
        target_cache = get_err_cache_dir(dataset, model, sweep_name, seed)
        target_results = get_results_dir(dataset, model, sweep_name, seed)
        print(f"\n[DRY RUN] {sweep_name}/seed_{seed}")
        print(f"  rho={rho}, lambda_evi={lambda_evi}")
        print(f"  source cache: {source_cache}")
        print(f"  target cache: {target_cache}")
        print(f"  sweep output: {out_dir}")
        print(f"  train_stage3.py --config <temp> --run_name {sweep_name} --seed {seed}")
        return sweep_name, True

    target_cache = get_err_cache_dir(dataset, model, sweep_name, seed)
    _mirror_err_cache(source_cache, target_cache)

    patched = json.loads(json.dumps(base_config))
    patched.setdefault("train", {})["seed"] = seed
    patched.setdefault("run", {})["run_name"] = sweep_name
    patched["run"]["seed"] = seed
    patched.setdefault("reasoner", {})["rho"] = rho
    patched["reasoner"]["lambda_evi"] = lambda_evi

    gpu = train_gpus[(seed + int(rho * 1000) + int(lambda_evi * 1000)) % len(train_gpus)]
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    python = sys.executable
    scripts_dir = Path(__file__).parent

    with tempfile.TemporaryDirectory(
        prefix=f"sweep_{dataset}_{model}_{seed}_"
    ) as tmpdir:
        tmp_config = Path(tmpdir) / "config.yaml"
        with open(tmp_config, "w") as f:
            yaml.safe_dump(patched, f, sort_keys=False)

        print(f"\n{'='*60}")
        print(f"Training: {sweep_name}/seed_{seed}  rho={rho} lambda={lambda_evi}")
        print(f"{'='*60}")

        cmd = [
            python, str(scripts_dir / "train_stage3.py"),
            "--config", str(tmp_config),
            "--run_name", sweep_name,
            "--seed", str(seed),
        ]
        if debug:
            cmd.append("--debug")

        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            print(f"[FAILED] train_stage3 for {sweep_name}/seed_{seed}")
            return sweep_name, False

    base_metrics_path = get_results_dir(dataset, model, "base", seed) / "stage1_metrics.json"
    base_metrics = _load_json(base_metrics_path)

    target_results = get_results_dir(dataset, model, sweep_name, seed)
    stage3_metrics = _load_json(target_results / "stage3_metrics.json")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu else "cpu")
    calibrated: dict = {}
    try:
        calibrated = _evaluate_with_calibration(
            patched, dataset, model, sweep_name, seed, device, debug,
        )
    except Exception as e:
        print(f"  [WARN] Calibration eval failed: {e}")
        calibrated = {
            "rho": rho,
            "fixed": {"threshold": 0.5, "test": stage3_metrics},
            "val_f1": {},
            "val_macro_f1": {},
        }

    diagnosis: dict = {}
    try:
        diagnosis = _diagnose_reasoner(
            patched, dataset, model, sweep_name, seed,
            base_run_name, base_metrics,
            stage3_metrics.get("test_metrics", stage3_metrics),
            device, debug,
        )
    except Exception as e:
        print(f"  [WARN] Diagnosis failed: {e}")
        diagnosis = {
            "error": str(e),
            "rho": rho,
            "run_name": sweep_name,
            "seed": seed,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "stage3_metrics.json", {
        "dataset": dataset,
        "model": model,
        "base_run_name": base_run_name,
        "sweep_run_name": sweep_name,
        "seed": seed,
        "rho": rho,
        "lambda_evi": lambda_evi,
        "test_metrics": stage3_metrics.get("test_metrics", stage3_metrics),
        "epochs_trained": stage3_metrics.get("epochs_trained", -1),
        "num_accepted_err": stage3_metrics.get("num_accepted_err", 0),
        "git_hash": _git_hash(),
    })
    _save_json(out_dir / "calibrated_metrics.json", calibrated)
    _save_json(out_dir / "reasoner_diagnosis.json", diagnosis)

    auc = stage3_metrics.get("test_metrics", stage3_metrics).get("roc_auc", 0)
    f1 = stage3_metrics.get("test_metrics", stage3_metrics).get("f1", 0)
    print(f"  Done: AUC={auc:.4f} F1={f1:.4f}")

    return sweep_name, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep rho and lambda_evi for the evidence reasoner.",
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (yelpchi, amazon)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (bwgnn, gcn, gat, sage)")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Base run whose ERR cache to reuse (e.g. qwen)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Random seeds")
    parser.add_argument("--rho_values", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--lambda_values", nargs="+", type=float,
                        default=[0.0, 0.1, 0.3, 0.5])
    parser.add_argument("--train_gpus", type=str, default="0",
                        help="Comma-separated CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--max_jobs", type=int, default=1,
                        help="Max parallel jobs (1 = sequential)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if all sweep outputs already exist")
    parser.add_argument("--debug", action="store_true",
                        help="Use tiny synthetic graph, 3 epochs")
    args = parser.parse_args()

    train_gpus = [g.strip() for g in args.train_gpus.split(",") if g.strip()]
    if not train_gpus:
        train_gpus = [""]

    # Build job list
    jobs = [
        {
            "rho": rho,
            "lambda_evi": lam,
            "seed": seed,
        }
        for rho in args.rho_values
        for lam in args.lambda_values
        for seed in args.seeds
    ]

    print(f"CoVER-FD Reasoner Sweep")
    print(f"{'='*60}")
    print(f"  Dataset:   {args.dataset}")
    print(f"  Model:     {args.model}")
    print(f"  ERR cache: {args.run_name}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Rho:       {args.rho_values}")
    print(f"  Lambda:    {args.lambda_values}")
    print(f"  GPUs:      {train_gpus}")
    print(f"  Max jobs:  {args.max_jobs}")
    print(f"  Total:     {len(jobs)} combinations")
    print(f"{'='*60}")

    if not jobs:
        print("No jobs to run.")
        return

    total_start = time.time()
    results: list[tuple[str, bool]] = []

    if args.dry_run or args.max_jobs <= 1:
        for job in jobs:
            name, ok = run_job(
                dataset=args.dataset,
                model=args.model,
                base_run_name=args.run_name,
                seed=job["seed"],
                rho=job["rho"],
                lambda_evi=job["lambda_evi"],
                train_gpus=train_gpus,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
                debug=args.debug,
            )
            results.append((name, ok))
    else:
        max_workers = max(1, min(args.max_jobs, len(train_gpus)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    run_job,
                    dataset=args.dataset,
                    model=args.model,
                    base_run_name=args.run_name,
                    seed=job["seed"],
                    rho=job["rho"],
                    lambda_evi=job["lambda_evi"],
                    train_gpus=train_gpus,
                    dry_run=False,
                    skip_existing=args.skip_existing,
                    debug=args.debug,
                ): job
                for job in jobs
            }
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    name, ok = fut.result()
                    results.append((name, ok))
                except Exception as e:
                    label = f"rho={job['rho']},lambda={job['lambda_evi']},seed={job['seed']}"
                    print(f"[ERROR] {label}: {e}")
                    results.append((label, False))

    total_elapsed = time.time() - total_start

    succeeded = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    report_dir = ensure_dir(
        ARTIFACTS_ROOT / "sweeps" / args.dataset / args.model / args.run_name,
    )
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "base_run_name": args.run_name,
        "seeds": args.seeds,
        "rho_values": args.rho_values,
        "lambda_values": args.lambda_values,
        "total_combinations": len(jobs),
        "succeeded": succeeded,
        "failed": failed,
        "skipped_by_existing": 0,
        "dry_run": args.dry_run,
        "total_runtime_seconds": round(total_elapsed, 2),
    }
    _save_json(report_dir / "sweep_summary.json", summary)

    print(f"\n{'='*60}")
    print("Sweep Complete")
    print(f"{'='*60}")
    print(f"  Succeeded: {succeeded}/{len(jobs)}")
    print(f"  Failed:    {failed}")
    print(f"  Time:      {total_elapsed:.1f}s")
    print(f"  Report:    {report_dir / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
