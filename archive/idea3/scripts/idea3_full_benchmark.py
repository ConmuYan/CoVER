"""Idea 3 Full Benchmark: LLM-Designed Feature Composites vs AutoFE baselines.

Runs on all 8 cells (YelpChi/Amazon × BWGNN/SAGE/GCN/GAT) × 5 seeds.
Compares:
  1. Base logit only
  2. Base + CoVER-REL (Idea 1)
  3. Base + raw f1..f6 (fraud_nbr_rates + log_degrees)
  4. Base + all pairwise/log/sqrt transforms of f1..f6
  5. Base + LLM-designed composites
  6. Base + random formulas (control)
  7. Base + REL + LLM composites

Usage:
    python scripts/idea3_full_benchmark.py --device cuda:0 --dataset yelpchi --base bwgnn
    python scripts/idea3_full_benchmark.py --device cuda:0 --all  # all 8 cells
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.sparse import issparse, coo_matrix, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_fraud import load_fraud_dataset
from evidence.relation_features import load_relation_stats, RELATION_SCHEMAS
from models.cover_rel_reasoner import CoVERRelReasoner

SEEDS = [42, 123, 456, 789, 2026]
CELLS = [
    ("yelpchi", "bwgnn"), ("yelpchi", "sage"), ("yelpchi", "gcn"), ("yelpchi", "gat"),
    ("amazon", "bwgnn"), ("amazon", "sage"), ("amazon", "gcn"), ("amazon", "gat"),
]

# LLM-designed composites (from Qwen3-4B-Instruct, fixed across seeds)
LLM_EXPRS = [
    ("f1*f3", "RUR×RTR fraud interaction"),
    ("f2/(f5+0.1)", "RSR fraud/degree ratio"),
    ("f4*log(f6+1)", "RUR_deg × log(RTR_deg)"),
    ("sqrt(f1+f2)", "sqrt(RUR+RSR fraud)"),
    ("max(f3,f5)-f4", "max(RTR_fraud,RSR_deg)-RUR_deg"),
]


def load_cell(dataset: str, base: str, seed: int):
    """Load data, base outputs, relation features, and CoVER-REL delta_rel."""
    # Map dataset name to actual filename (YelpChi has capital C, Amazon is title-case)
    mat_name = {"yelpchi": "YelpChi", "amazon": "Amazon"}[dataset]
    data = load_fraud_dataset(
        name=dataset, path=f"datasets/{mat_name}.mat", format="mat",
        seed=seed, train_ratio=0.4, val_test_ratio=[1, 2], stratified=True,
    )

    # Base outputs
    base_path = f"artifacts/base_outputs/{dataset}/{base}/seed_{seed}/_override_seed_{seed}_base.pt"
    base_outputs = torch.load(base_path, weights_only=True, map_location="cpu")
    base_logit = base_outputs["base_logits"].numpy()
    base_z = base_outputs["base_z"].numpy()

    # Relation features + CoVER-REL delta_rel
    canon_run = "idea1_canonical_clsonly" if (dataset == "yelpchi" and base == "bwgnn") else f"idea1_{dataset}_{base}_canonical_clsonly"
    rel_path = f"artifacts/relation_features/{dataset}/{base}/seed_{seed}/all/rel_stats.pt"
    rel_stats, _ = load_relation_stats(rel_path, num_nodes=data.x.shape[0])

    reasoner = CoVERRelReasoner(
        base_z_dim=base_z.shape[1],
        relation_names=list(RELATION_SCHEMAS[dataset].keys()),
        anchor_relation=list(RELATION_SCHEMAS[dataset].keys())[0],
        rel_stat_dim=9, rel_hidden_dim=64, rel_num_layers=2,
        rel_dropout=0.3, tau_gate=0.7, delta_rel_max=2.0,
    )
    ckpt = torch.load(f"artifacts/checkpoints/{dataset}/{base}/{canon_run}/seed_{seed}/reasoner.pt", weights_only=False, map_location="cpu")
    reasoner.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    reasoner.eval()
    device = torch.device("cuda:0")
    reasoner = reasoner.to(device)
    with torch.no_grad():
        out = reasoner(
            torch.tensor(base_z, dtype=torch.float32).to(device),
            torch.tensor(base_logit, dtype=torch.float32).to(device),
            rel_stats.to(device),
        )
    delta_rel = out["delta_rel"].cpu().numpy()

    return data, base_logit, base_z, delta_rel


def compute_graph_stats(dataset: str, data, adjs: dict):
    """Compute fraud_nbr_rates and log_degrees for all relations."""
    N = data.x.shape[0]
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()

    train_fraud = np.zeros(N)
    train_fraud[train_mask & (y == 1)] = 1

    schema = RELATION_SCHEMAS[dataset]
    rel_names = list(schema.keys())

    features = {}
    for name in rel_names:
        A = adjs[name]
        deg = np.array(A.sum(axis=1)).flatten()
        rs = deg.copy()
        rs[rs == 0] = 1
        fraud_nbr = A.dot(train_fraud) / rs
        features[f"fraud_nbr_{name}"] = fraud_nbr
        features[f"log_deg_{name}"] = np.log1p(deg)

    return features, rel_names


def build_composite_features(features: dict, rel_names: list, exprs: list,
                              fail_mode: str = "raise") -> np.ndarray:
    """Evaluate composite feature expressions.

    `exprs` is a list of (expr_str, description) tuples (LLM_EXPRS schema).

    fail_mode controls behaviour when an expression cannot be eval'd:
      - 'raise' (DEFAULT): raise ValueError immediately. Prevents silent
        collapse to all-zero columns (the multi-LLM scaling parser bug
        post-mortem).
      - 'skip'  : drop the failing expression with a warning.
      - 'zero'  : LEGACY zero-substitute. Banned in any new callsite.
    """
    if fail_mode not in {"raise", "skip", "zero"}:
        raise ValueError(f"unknown fail_mode {fail_mode!r}")

    # Map f1..fN to actual features
    f_map = {}
    for i, name in enumerate(rel_names):
        f_map[f"f{i+1}"] = features[f"fraud_nbr_{name}"]
        f_map[f"f{i+1+len(rel_names)}"] = features[f"log_deg_{name}"]

    composites = []
    n_nodes = len(next(iter(features.values())))
    for expr, _ in exprs:
        try:
            feat = eval(expr, {"__builtins__": {}}, {
                **f_map,
                "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
                "max": np.maximum, "min": np.minimum,
            })
            feat = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
            composites.append(feat.reshape(-1, 1))
        except Exception as e:
            if fail_mode == "raise":
                raise ValueError(
                    f"build_composite_features: expression {expr!r} failed: {e}"
                ) from e
            elif fail_mode == "skip":
                print(f"  [WARN] expression {expr!r} skipped: {e}")
            else:  # 'zero' (legacy)
                print(f"  [LEGACY-WARN] expression {expr!r} → zero column: {e}")
                composites.append(np.zeros((n_nodes, 1)))

    return np.hstack(composites) if composites else np.zeros((n_nodes, 0))


def generate_all_pairwise_transforms(features: dict, rel_names: list) -> tuple[np.ndarray, list[str]]:
    """Generate all pairwise/log/sqrt transforms of base features."""
    feat_arrays = [features[f"fraud_nbr_{name}"] for name in rel_names] + \
                  [features[f"log_deg_{name}"] for name in rel_names]
    feat_names = [f"fraud_nbr_{name}" for name in rel_names] + \
                 [f"log_deg_{name}" for name in rel_names]

    transforms = []
    names = []

    # All pairwise products
    for i, j in combinations(range(len(feat_arrays)), 2):
        transforms.append((feat_arrays[i] * feat_arrays[j]).reshape(-1, 1))
        names.append(f"{feat_names[i]}*{feat_names[j]}")

    # All pairwise ratios
    for i, j in combinations(range(len(feat_arrays)), 2):
        transforms.append((feat_arrays[i] / (feat_arrays[j] + 0.01)).reshape(-1, 1))
        names.append(f"{feat_names[i]}/{feat_names[j]}")

    # log and sqrt of each
    for i, arr in enumerate(feat_arrays):
        transforms.append(np.log1p(np.abs(arr)).reshape(-1, 1))
        names.append(f"log({feat_names[i]})")
        transforms.append(np.sqrt(np.abs(arr)).reshape(-1, 1))
        names.append(f"sqrt({feat_names[i]})")

    return np.hstack(transforms), names


def generate_random_formulas(features: dict, rel_names: list, n_formulas: int = 5, seed: int = 42) -> np.ndarray:
    """Generate random composite formulas as control baseline."""
    rng = np.random.RandomState(seed)
    feat_arrays = [features[f"fraud_nbr_{name}"] for name in rel_names] + \
                  [features[f"log_deg_{name}"] for name in rel_names]

    composites = []
    ops = ["add", "mul", "div", "sqrt", "log"]
    for _ in range(n_formulas):
        i, j = rng.randint(0, len(feat_arrays), 2)
        op = rng.choice(ops)
        a, b = feat_arrays[i], feat_arrays[j]
        if op == "add":
            feat = a + b
        elif op == "mul":
            feat = a * b
        elif op == "div":
            feat = a / (b + 0.01)
        elif op == "sqrt":
            feat = np.sqrt(np.abs(a + b))
        else:
            feat = np.log1p(np.abs(a))
        composites.append(np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6).reshape(-1, 1))

    return np.hstack(composites)


def eval_lr(X_train, y_train, X_test, y_test, seed: int) -> dict:
    """Train LR and evaluate."""
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    lr.fit(X_train, y_train)
    prob = lr.predict_proba(X_test)[:, 1]
    return {
        "auprc": float(average_precision_score(y_test, prob)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
    }


def run_cell(dataset: str, base: str, device_str: str = "cuda:0"):
    """Run full benchmark for one cell."""
    print(f"\n{'='*60}")
    print(f"Cell: {dataset}-{base}")
    print(f"{'='*60}")

    # Load .mat once
    mat_name = {"yelpchi": "YelpChi", "amazon": "Amazon"}[dataset]
    mat = loadmat(f"datasets/{mat_name}.mat")
    schema = RELATION_SCHEMAS[dataset]
    rel_names = list(schema.keys())
    N = None

    results_by_seed = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        data, base_logit, base_z, delta_rel = load_cell(dataset, base, seed)
        N = data.x.shape[0]
        y = data.y.numpy()
        train_mask = data.train_mask.numpy()
        test_mask = data.test_mask.numpy()

        # Build adj CSR
        adjs = {}
        for name in rel_names:
            key = schema[name]["mat_key"]
            sp = mat[key].tocoo() if issparse(mat[key]) else coo_matrix(mat[key])
            adjs[name] = csr_matrix((np.ones(len(sp.row)), (sp.row, sp.col)), shape=(N, N))

        # Compute graph stats
        features, _ = compute_graph_stats(dataset, data, adjs)

        # Build feature sets
        raw_feats = np.column_stack([features[f"fraud_nbr_{name}"] for name in rel_names] +
                                     [features[f"log_deg_{name}"] for name in rel_names])
        llm_feats = build_composite_features(features, rel_names, LLM_EXPRS)
        all_transforms, _ = generate_all_pairwise_transforms(features, rel_names)
        random_feats = generate_random_formulas(features, rel_names, n_formulas=5, seed=seed)

        # Evaluate all combinations
        r = {}
        X_base = base_logit.reshape(-1, 1)

        r["base_only"] = eval_lr(X_base[train_mask], y[train_mask], X_base[test_mask], y[test_mask], seed)
        r["base_rel"] = eval_lr(np.column_stack([X_base, delta_rel])[train_mask], y[train_mask],
                                np.column_stack([X_base, delta_rel])[test_mask], y[test_mask], seed)
        r["base_raw"] = eval_lr(np.column_stack([X_base, raw_feats])[train_mask], y[train_mask],
                                np.column_stack([X_base, raw_feats])[test_mask], y[test_mask], seed)
        r["base_pairwise"] = eval_lr(np.column_stack([X_base, all_transforms])[train_mask], y[train_mask],
                                     np.column_stack([X_base, all_transforms])[test_mask], y[test_mask], seed)
        r["base_llm"] = eval_lr(np.column_stack([X_base, llm_feats])[train_mask], y[train_mask],
                                np.column_stack([X_base, llm_feats])[test_mask], y[test_mask], seed)
        r["base_random"] = eval_lr(np.column_stack([X_base, random_feats])[train_mask], y[train_mask],
                                   np.column_stack([X_base, random_feats])[test_mask], y[test_mask], seed)
        r["base_rel_llm"] = eval_lr(np.column_stack([X_base, delta_rel, llm_feats])[train_mask], y[train_mask],
                                    np.column_stack([X_base, delta_rel, llm_feats])[test_mask], y[test_mask], seed)

        results_by_seed[seed] = r
        print(f"  base={r['base_only']['auprc']:.4f}  REL={r['base_rel']['auprc']:.4f}  "
              f"raw={r['base_raw']['auprc']:.4f}  pairwise={r['base_pairwise']['auprc']:.4f}  "
              f"LLM={r['base_llm']['auprc']:.4f}  random={r['base_random']['auprc']:.4f}  "
              f"REL+LLM={r['base_rel_llm']['auprc']:.4f}")

    return results_by_seed


def aggregate_results(all_results: dict):
    """Aggregate across cells and seeds, compute paired-t."""
    from scipy import stats as sp_stats

    lines = []
    a = lines.append

    a("# Idea 3 Full Benchmark — LLM Feature Composites vs AutoFE Baselines\n")
    a("**Method**: Qwen3-4B-Instruct designs 5 composite features from graph statistics.\n")
    a("**Baselines**: raw features, all pairwise/log/sqrt transforms, random formulas.\n")
    a("**Significance (df=4)**: |t|>2.78 → ★; |t|>4.60 → ★★; |t|>8.61 → ★★★\n")

    # Per-cell summary table
    a("## AUPRC Summary (mean ± sd across 5 seeds)\n")
    methods = ["base_only", "base_rel", "base_raw", "base_pairwise", "base_random", "base_llm", "base_rel_llm"]
    method_labels = {
        "base_only": "Base only", "base_rel": "Base+REL", "base_raw": "Base+raw",
        "base_pairwise": "Base+pairwise", "base_random": "Base+random",
        "base_llm": "Base+LLM", "base_rel_llm": "Base+REL+LLM",
    }

    header = "| Cell | " + " | ".join(method_labels[m] for m in methods) + " |"
    sep = "|---|" + "|".join(["---:" for _ in methods]) + "|"
    a(header)
    a(sep)

    # Collect per-method arrays for paired-t
    method_arrays = {m: [] for m in methods}

    for (dataset, base), seed_results in all_results.items():
        row = [f"{dataset}-{base}"]
        for m in methods:
            vals = [seed_results[s][m]["auprc"] for s in SEEDS]
            method_arrays[m].extend(vals)
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            row.append(f"{mean:.4f} ± {std:.4f}")
        a("| " + " | ".join(row) + " |")

    # Paired-t: LLM vs each baseline
    a("\n## Paired-t: Base+LLM vs baselines (across all 8 cells × 5 seeds = 40 pairs)\n")
    a("| Comparison | Δ | t | p | sig |")
    a("|---|---:|---:|---:|:---:|")

    llm_vals = np.array(method_arrays["base_llm"])
    for m in ["base_only", "base_rel", "base_raw", "base_pairwise", "base_random"]:
        other_vals = np.array(method_arrays[m])
        diff = llm_vals - other_vals
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
        p = 2 * (1 - sp_stats.t.cdf(abs(t), df=len(diff) - 1))
        sig = "★★★" if abs(t) > 8.61 else ("★★" if abs(t) > 4.60 else ("★" if abs(t) > 2.78 else "ns"))
        a(f"| Base+LLM vs {method_labels[m]} | {diff.mean():+.4f} | {t:+.3f} | {p:.4f} | {sig} |")

    # REL+LLM vs REL alone
    rel_vals = np.array(method_arrays["base_rel"])
    rel_llm_vals = np.array(method_arrays["base_rel_llm"])
    diff = rel_llm_vals - rel_vals
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    p = 2 * (1 - sp_stats.t.cdf(abs(t), df=len(diff) - 1))
    sig = "★★★" if abs(t) > 8.61 else ("★★" if abs(t) > 4.60 else ("★" if abs(t) > 2.78 else "ns"))
    a(f"| Base+REL+LLM vs Base+REL | {diff.mean():+.4f} | {t:+.3f} | {p:.4f} | {sig} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--base", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        cells = CELLS
    elif args.dataset and args.base:
        cells = [(args.dataset, args.base)]
    else:
        cells = CELLS

    all_results = {}
    for dataset, base in cells:
        try:
            seed_results = run_cell(dataset, base, args.device)
            all_results[(dataset, base)] = seed_results
        except Exception as e:
            print(f"\n[ERROR] {dataset}-{base}: {e}")
            continue

    # Aggregate
    report = aggregate_results(all_results)
    print("\n" + report)

    # Save
    out_dir = Path("artifacts/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "idea3_full_benchmark.md").write_text(report)
    print(f"\nSaved to {out_dir / 'idea3_full_benchmark.md'}")


if __name__ == "__main__":
    main()
