"""Idea 3 AutoFE Baselines: XGBoost, LightGBM, Random Formula Search.

Compares LLM-designed composites against:
1. XGBoost on raw features (f1..f6)
2. LightGBM on raw features
3. Best of 100 random composite formulas (val-selected)
4. Best of all pairwise/log/sqrt transforms (val-selected)

Usage:
    python scripts/idea3_autofe_baselines.py --device cuda:0
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
from evidence.relation_features import RELATION_SCHEMAS

SEEDS = [42, 123, 456, 789, 2026]
CELLS = [
    ("yelpchi", "bwgnn"), ("yelpchi", "sage"), ("yelpchi", "gcn"), ("yelpchi", "gat"),
    ("amazon", "bwgnn"), ("amazon", "sage"), ("amazon", "gcn"), ("amazon", "gat"),
]

# LLM-designed composites (from Qwen3-4B-Instruct)
LLM_EXPRS = [
    ("f1*f3", "RUR×RTR fraud interaction"),
    ("f2/(f5+0.1)", "RSR fraud/degree ratio"),
    ("f4*log(f6+1)", "RUR_deg × log(RTR_deg)"),
    ("sqrt(f1+f2)", "sqrt(RUR+RSR fraud)"),
    ("max(f3,f5)-f4", "max(RTR_fraud,RSR_deg)-RUR_deg"),
]


def load_cell(dataset: str, base: str, seed: int):
    mat_name = {"yelpchi": "YelpChi", "amazon": "Amazon"}[dataset]
    data = load_fraud_dataset(
        name=dataset, path=f"datasets/{mat_name}.mat", format="mat",
        seed=seed, train_ratio=0.4, val_test_ratio=[1, 2], stratified=True,
    )
    base_path = f"artifacts/base_outputs/{dataset}/{base}/seed_{seed}/_override_seed_{seed}_base.pt"
    base_outputs = torch.load(base_path, weights_only=True, map_location="cpu")
    base_logit = base_outputs["base_logits"].numpy()
    return data, base_logit


def compute_graph_stats(dataset: str, data, mat):
    N = data.x.shape[0]
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    train_fraud = np.zeros(N)
    train_fraud[train_mask & (y == 1)] = 1

    schema = RELATION_SCHEMAS[dataset]
    rel_names = list(schema.keys())
    features = {}
    adjs = {}

    for name in rel_names:
        key = schema[name]["mat_key"]
        sp = mat[key].tocoo() if issparse(mat[key]) else coo_matrix(mat[key])
        A = csr_matrix((np.ones(len(sp.row)), (sp.row, sp.col)), shape=(N, N))
        adjs[name] = A
        deg = np.array(A.sum(axis=1)).flatten()
        rs = deg.copy(); rs[rs == 0] = 1
        features[f"fraud_nbr_{name}"] = A.dot(train_fraud) / rs
        features[f"log_deg_{name}"] = np.log1p(deg)

    return features, rel_names


def eval_metric(y_true, prob, mask):
    auprc = average_precision_score(y_true[mask], prob[mask])
    return auprc


def generate_random_formulas(feat_arrays, n_formulas=100, seed=42):
    """Generate random composite formulas and evaluate on val, return best on test."""
    rng = np.random.RandomState(seed)
    ops = ["add", "mul", "div", "sqrt_sum", "log_diff", "max_diff"]

    formulas = []
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
        elif op == "sqrt_sum":
            feat = np.sqrt(np.abs(a + b))
        elif op == "log_diff":
            feat = np.log1p(np.abs(a - b))
        else:
            feat = np.maximum(a, b) - np.minimum(a, b)
        formulas.append(np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6))

    return formulas


def generate_systematic_transforms(feat_arrays, feat_names):
    """Generate all pairwise products, ratios, log, sqrt transforms."""
    transforms = []
    names = []

    # Pairwise products
    for i, j in combinations(range(len(feat_arrays)), 2):
        transforms.append(feat_arrays[i] * feat_arrays[j])
        names.append(f"{feat_names[i]}*{feat_names[j]}")

    # Pairwise ratios
    for i, j in combinations(range(len(feat_arrays)), 2):
        transforms.append(feat_arrays[i] / (feat_arrays[j] + 0.01))
        names.append(f"{feat_names[i]}/{feat_names[j]}")

    # Individual transforms
    for i, arr in enumerate(feat_arrays):
        transforms.append(np.log1p(np.abs(arr)))
        names.append(f"log({feat_names[i]})")
        transforms.append(np.sqrt(np.abs(arr)))
        names.append(f"sqrt({feat_names[i]})")

    return transforms, names


def run_cell(dataset: str, base: str):
    print(f"\n{'='*60}")
    print(f"Cell: {dataset}-{base}")
    print(f"{'='*60}")

    mat_name = {"yelpchi": "YelpChi", "amazon": "Amazon"}[dataset]
    mat = loadmat(f"datasets/{mat_name}.mat")

    results_by_seed = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        data, base_logit = load_cell(dataset, base, seed)
        y = data.y.numpy()
        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()

        features, rel_names = compute_graph_stats(dataset, data, mat)
        feat_arrays = [features[f"fraud_nbr_{name}"] for name in rel_names] + \
                       [features[f"log_deg_{name}"] for name in rel_names]
        feat_names = [f"fraud_nbr_{name}" for name in rel_names] + \
                      [f"log_deg_{name}" for name in rel_names]

        X_base = base_logit.reshape(-1, 1)

        r = {}

        # 1. Base only
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        lr.fit(X_base[train_mask], y[train_mask])
        r["base_only"] = eval_metric(y, lr.predict_proba(X_base)[:, 1], test_mask)

        # 2. Base + raw features
        X_raw = np.column_stack([X_base] + [f.reshape(-1, 1) for f in feat_arrays])
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        lr.fit(X_raw[train_mask], y[train_mask])
        r["base_raw"] = eval_metric(y, lr.predict_proba(X_raw)[:, 1], test_mask)

        # 3. XGBoost on raw features
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                scale_pos_weight=(y[train_mask] == 0).sum() / max((y[train_mask] == 1).sum(), 1),
                random_state=seed, eval_metric="aucpr", verbosity=0,
            )
            xgb_model.fit(X_raw[train_mask], y[train_mask])
            r["xgboost_raw"] = eval_metric(y, xgb_model.predict_proba(X_raw)[:, 1], test_mask)
        except ImportError:
            r["xgboost_raw"] = float("nan")
            print("  XGBoost not available")

        # 4. LightGBM on raw features
        try:
            import lightgbm as lgb
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                scale_pos_weight=(y[train_mask] == 0).sum() / max((y[train_mask] == 1).sum(), 1),
                random_state=seed, verbose=-1,
            )
            lgb_model.fit(X_raw[train_mask], y[train_mask])
            r["lightgbm_raw"] = eval_metric(y, lgb_model.predict_proba(X_raw)[:, 1], test_mask)
        except ImportError:
            r["lightgbm_raw"] = float("nan")
            print("  LightGBM not available")

        # 5. LLM-designed composites
        llm_feats = []
        for expr, _ in LLM_EXPRS:
            f_map = {f"f{i+1}": feat_arrays[i] for i in range(len(feat_arrays))}
            f_map.update({"log": np.log, "sqrt": np.sqrt, "abs": np.abs,
                          "max": np.maximum, "min": np.minimum})
            feat = eval(expr, f_map)
            llm_feats.append(np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6))
        X_llm = np.column_stack([X_base] + [f.reshape(-1, 1) for f in llm_feats])
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        lr.fit(X_llm[train_mask], y[train_mask])
        r["base_llm"] = eval_metric(y, lr.predict_proba(X_llm)[:, 1], test_mask)

        # 6. Random formula search (100 random, val-select best, test-eval)
        random_formulas = generate_random_formulas(feat_arrays, n_formulas=100, seed=seed)
        best_val_score = -1
        best_test_score = -1
        for formula in random_formulas:
            X_rand = np.column_stack([X_base, formula.reshape(-1, 1)])
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            lr.fit(X_rand[train_mask], y[train_mask])
            val_score = eval_metric(y, lr.predict_proba(X_rand)[:, 1], val_mask)
            if val_score > best_val_score:
                best_val_score = val_score
                best_test_score = eval_metric(y, lr.predict_proba(X_rand)[:, 1], test_mask)
        r["random_formula_best"] = best_test_score

        # 7. Systematic transforms (val-select best)
        sys_transforms, sys_names = generate_systematic_transforms(feat_arrays, feat_names)
        best_val_score = -1
        best_test_score = -1
        for transform in sys_transforms:
            X_sys = np.column_stack([X_base, transform.reshape(-1, 1)])
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            lr.fit(X_sys[train_mask], y[train_mask])
            val_score = eval_metric(y, lr.predict_proba(X_sys)[:, 1], val_mask)
            if val_score > best_val_score:
                best_val_score = val_score
                best_test_score = eval_metric(y, lr.predict_proba(X_sys)[:, 1], test_mask)
        r["systematic_best"] = best_test_score

        results_by_seed[seed] = r
        print(f"  base={r['base_only']:.4f}  raw={r['base_raw']:.4f}  "
              f"XGB={r['xgboost_raw']:.4f}  LGB={r['lightgbm_raw']:.4f}  "
              f"LLM={r['base_llm']:.4f}  random100={r['random_formula_best']:.4f}  "
              f"systematic={r['systematic_best']:.4f}")

    return results_by_seed


def aggregate(all_results: dict):
    from scipy import stats as sp_stats

    lines = []
    a = lines.append

    a("# Idea 3 AutoFE Baselines — LLM vs XGBoost/LightGBM/Random/Systematic\n")
    a("**Comparison**: LLM-designed composites vs automated feature engineering baselines.\n")
    a("**All methods**: LR(base_logit + features) on test set.\n")

    methods = ["base_only", "base_raw", "xgboost_raw", "lightgbm_raw",
               "random_formula_best", "systematic_best", "base_llm"]
    labels = {
        "base_only": "Base only", "base_raw": "Base+raw (LR)",
        "xgboost_raw": "XGBoost (raw)", "lightgbm_raw": "LightGBM (raw)",
        "random_formula_best": "Random formula (best of 100)",
        "systematic_best": "Systematic transforms (best)",
        "base_llm": "Base+LLM composites",
    }

    a("## AUPRC Summary (mean ± sd across 5 seeds)\n")
    header = "| Cell | " + " | ".join(labels[m] for m in methods) + " |"
    sep = "|---|" + "|".join(["---:" for _ in methods]) + "|"
    a(header)
    a(sep)

    method_arrays = {m: [] for m in methods}
    for (dataset, base), seed_results in all_results.items():
        row = [f"{dataset}-{base}"]
        for m in methods:
            vals = [seed_results[s][m] for s in SEEDS]
            method_arrays[m].extend(vals)
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            row.append(f"{mean:.4f} ± {std:.4f}")
        a("| " + " | ".join(row) + " |")

    # Paired-t: LLM vs each baseline
    a("\n## Paired-t: Base+LLM vs baselines (8 cells × 5 seeds = 40 pairs)\n")
    a("| Comparison | Δ | t | p | sig |")
    a("|---|---:|---:|---:|:---:|")

    llm_vals = np.array(method_arrays["base_llm"])
    for m in ["base_only", "base_raw", "xgboost_raw", "lightgbm_raw", "random_formula_best", "systematic_best"]:
        other_vals = np.array(method_arrays[m])
        diff = llm_vals - other_vals
        if np.isnan(diff).any():
            a(f"| Base+LLM vs {labels[m]} | — | — | — | NaN in data |")
            continue
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
        p = 2 * (1 - sp_stats.t.cdf(abs(t), df=len(diff) - 1))
        sig = "★★★" if abs(t) > 8.61 else ("★★" if abs(t) > 4.60 else ("★" if abs(t) > 2.78 else "ns"))
        a(f"| Base+LLM vs {labels[m]} | {diff.mean():+.4f} | {t:+.3f} | {p:.4f} | {sig} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--base", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cells = CELLS if args.all else [(args.dataset, args.base)] if args.dataset and args.base else CELLS

    all_results = {}
    for dataset, base in cells:
        try:
            seed_results = run_cell(dataset, base)
            all_results[(dataset, base)] = seed_results
        except Exception as e:
            print(f"\n[ERROR] {dataset}-{base}: {e}")
            import traceback; traceback.print_exc()
            continue

    report = aggregate(all_results)
    print("\n" + report)

    out_dir = Path("artifacts/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "idea3_autofe_baselines.md").write_text(report)
    print(f"\nSaved to {out_dir / 'idea3_autofe_baselines.md'}")


if __name__ == "__main__":
    main()
