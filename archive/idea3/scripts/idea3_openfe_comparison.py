"""Idea 3 OpenFE + Symbolic Regression Comparison.

Compares LLM-designed composites vs SOTA AutoFE methods:
1. OpenFE (NeurIPS 2023) — automatic feature engineering via mutual information
2. Symbolic Regression (gplearn) — genetic programming for formula discovery
3. LLM-designed composites (Qwen3-4B-Instruct)
4. Random formula (best of 100)
5. Systematic pairwise/log/sqrt transforms (best)

All evaluated as: LR(base_logit + features) on test set.
8 cells × 5 seeds.

Usage:
    python scripts/idea3_openfe_comparison.py --device cuda:0 --all
    python scripts/idea3_openfe_comparison.py --device cuda:1 --dataset yelpchi --base bwgnn
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.sparse import issparse, coo_matrix, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

    for name in rel_names:
        key = schema[name]["mat_key"]
        sp = mat[key].tocoo() if issparse(mat[key]) else coo_matrix(mat[key])
        A = csr_matrix((np.ones(len(sp.row)), (sp.row, sp.col)), shape=(N, N))
        deg = np.array(A.sum(axis=1)).flatten()
        rs = deg.copy(); rs[rs == 0] = 1
        features[f"fraud_nbr_{name}"] = A.dot(train_fraud) / rs
        features[f"log_deg_{name}"] = np.log1p(deg)

    return features, rel_names


def eval_metric(y_true, prob, mask):
    return average_precision_score(y_true[mask], prob[mask])


def run_openfe(X_full_df, y_full, train_mask, test_mask, seed, top_k=5):
    """Run OpenFE on full data with proper train/val indices. Returns new features for all N nodes."""
    try:
        from openfe import OpenFE, tree_to_formula

        openfe_model = OpenFE()

        train_idx = np.where(train_mask)[0]
        # Use a subset of train as val (last 30% of train indices)
        n_val = max(int(len(train_idx) * 0.3), 1)
        val_idx = train_idx[-n_val:]
        train_idx_fit = train_idx[:-n_val] if n_val > 1 else train_idx

        openfe_model.fit(
            data=X_full_df,
            label=pd.DataFrame(y_full, columns=["label"]),
            task="classification",
            train_index=pd.Index(train_idx_fit),
            val_index=pd.Index(val_idx),
            n_jobs=1,
            seed=seed,
            verbose=False,
        )

        new_features = openfe_model.new_features_list[:top_k]
        feature_names = []
        for i, feat in enumerate(new_features):
            try:
                name = tree_to_formula(feat)
                feature_names.append(f"openfe_{name[:30]}")
            except Exception:
                feature_names.append(f"openfe_feat_{i}")

        if len(new_features) > 0:
            # Apply features directly via calculate() — avoids transform() API issues
            feat_cols = []
            for feat in new_features:
                feat.calculate(X_full_df, is_root=True)
                feat_cols.append(np.nan_to_num(feat.data.values.ravel(), nan=0.0, posinf=1e6, neginf=-1e6))
            return np.column_stack(feat_cols), feature_names
        else:
            return None, []

    except Exception as e:
        print(f"  OpenFE failed: {e}")
        return None, []


def run_symbolic_regression(X_train, y_train, X_test, n_components=5, seed=42):
    """Run gplearn SymbolicRegression to discover composite features."""
    try:
        from gplearn.genetic import SymbolicTransformer

        gp = SymbolicTransformer(
            population_size=500,
            hall_of_fame=50,
            n_components=n_components,
            generations=15,
            tournament_size=20,
            stopping_criteria=1.0,
            const_range=(-1.0, 1.0),
            init_depth=(2, 4),
            function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log'),
            metric='spearman',
            parsimony_coefficient=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            p_point_replace=0.05,
            max_samples=0.9,
            n_jobs=1,
            verbose=0,
            random_state=seed,
        )

        gp.fit(X_train, y_train)

        # Transform both train and test
        train_feats = gp.transform(X_train)
        test_feats = gp.transform(X_test)

        # Get program names
        feature_names = []
        for i, prog in enumerate(gp._best_programs):
            # gplearn 0.4.x: _best_programs[i] is a single _Program (not a list)
            if hasattr(prog, 'fitness_'):
                feature_names.append(f"gp_{str(prog)[:30]}")
            else:
                feature_names.append(f"gp_feat_{i}")

        return train_feats, test_feats, feature_names

    except Exception as e:
        print(f"  SymbolicRegression failed: {e}")
        return None, None, []


def generate_random_formulas(feat_arrays, n_formulas=100, seed=42):
    """Generate random composite formulas."""
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
    for i, j in combinations(range(len(feat_arrays)), 2):
        transforms.append(feat_arrays[i] * feat_arrays[j])
        names.append(f"{feat_names[i]}*{feat_names[j]}")
    for i, j in combinations(range(len(feat_arrays)), 2):
        transforms.append(feat_arrays[i] / (feat_arrays[j] + 0.01))
        names.append(f"{feat_names[i]}/{feat_names[j]}")
    for i, arr in enumerate(feat_arrays):
        transforms.append(np.log1p(np.abs(arr)))
        names.append(f"log({feat_names[i]})")
        transforms.append(np.sqrt(np.abs(arr)))
        names.append(f"sqrt({feat_names[i]})")
    return transforms, names


def run_cell(dataset: str, base: str, device_str: str = "cuda:0"):
    print(f"\n{'='*60}")
    print(f"Cell: {dataset}-{base}")
    print(f"{'='*60}")

    mat_name = {"yelpchi": "YelpChi", "amazon": "Amazon"}[dataset]
    mat = loadmat(f"datasets/{mat_name}.mat")

    results_by_seed = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()
        data, base_logit = load_cell(dataset, base, seed)
        base_logit = np.nan_to_num(base_logit, nan=0.0, posinf=1e6, neginf=-1e6)
        y = data.y.numpy()
        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()

        features, rel_names = compute_graph_stats(dataset, data, mat)
        feat_arrays = [np.nan_to_num(features[f"fraud_nbr_{name}"], nan=0.0) for name in rel_names] + \
                       [np.nan_to_num(features[f"log_deg_{name}"], nan=0.0) for name in rel_names]
        feat_names = [f"fraud_nbr_{name}" for name in rel_names] + \
                      [f"log_deg_{name}" for name in rel_names]

        X_base = base_logit.reshape(-1, 1)

        r = {}

        # --- 1. Base only ---
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        lr.fit(X_base[train_mask], y[train_mask])
        r["base_only"] = eval_metric(y, lr.predict_proba(X_base)[:, 1], test_mask)

        # --- 2. Base + raw features ---
        X_raw = np.column_stack([X_base] + [f.reshape(-1, 1) for f in feat_arrays])
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        lr.fit(X_raw[train_mask], y[train_mask])
        r["base_raw"] = eval_metric(y, lr.predict_proba(X_raw)[:, 1], test_mask)

        # --- 3. LLM-designed composites ---
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

        # --- 4. Random formula search (100 random, val-select best, test-eval) ---
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

        # --- 5. Systematic transforms (val-select best) ---
        sys_transforms, sys_names_list = generate_systematic_transforms(feat_arrays, feat_names)
        best_val_score = -1
        best_test_score = -1
        for transform_feat in sys_transforms:
            X_sys = np.column_stack([X_base, transform_feat.reshape(-1, 1)])
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            lr.fit(X_sys[train_mask], y[train_mask])
            val_score = eval_metric(y, lr.predict_proba(X_sys)[:, 1], val_mask)
            if val_score > best_val_score:
                best_val_score = val_score
                best_test_score = eval_metric(y, lr.predict_proba(X_sys)[:, 1], test_mask)
        r["systematic_best"] = best_test_score

        # --- 6. OpenFE (top-5 new features) ---
        print("  Running OpenFE...")
        base_feat_names = [f"f{i+1}" for i in range(len(feat_arrays))]
        X_df = pd.DataFrame(
            np.column_stack([base_logit.reshape(-1, 1)] + [f.reshape(-1, 1) for f in feat_arrays]),
            columns=["base_logit"] + base_feat_names,
        )
        openfe_feats, openfe_names = run_openfe(X_df, y, train_mask, test_mask, seed=seed, top_k=5)
        if openfe_feats is not None and openfe_feats.shape[1] > 0:
            X_openfe = np.column_stack([X_base, openfe_feats])
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            lr.fit(X_openfe[train_mask], y[train_mask])
            r["base_openfe"] = eval_metric(y, lr.predict_proba(X_openfe)[:, 1], test_mask)
        else:
            r["base_openfe"] = float("nan")

        # --- 7. Symbolic Regression (GP) ---
        print("  Running Symbolic Regression...")
        X_train_sr = np.column_stack([X_base[train_mask]] + [f[train_mask].reshape(-1, 1) for f in feat_arrays])
        X_test_sr = np.column_stack([X_base[test_mask]] + [f[test_mask].reshape(-1, 1) for f in feat_arrays])
        gp_train, gp_test, gp_names = run_symbolic_regression(
            X_train_sr, y[train_mask], X_test_sr, n_components=5, seed=seed,
        )
        if gp_train is not None and gp_train.shape[1] > 0:
            # Build full-feature arrays: GP features for train and test
            N = len(y)
            n_gp = gp_train.shape[1]
            gp_feats_full = np.zeros((N, n_gp))
            gp_feats_full[train_mask] = np.nan_to_num(gp_train, nan=0.0, posinf=1e6, neginf=-1e6)
            gp_feats_full[test_mask] = np.nan_to_num(gp_test, nan=0.0, posinf=1e6, neginf=-1e6)
            X_gp = np.column_stack([X_base, gp_feats_full])
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            lr.fit(X_gp[train_mask], y[train_mask])
            r["base_gp"] = eval_metric(y, lr.predict_proba(X_gp)[:, 1], test_mask)
        else:
            r["base_gp"] = float("nan")

        elapsed = time.time() - t0
        results_by_seed[seed] = r
        print(f"  [{elapsed:.0f}s] base={r['base_only']:.4f}  raw={r['base_raw']:.4f}  "
              f"LLM={r['base_llm']:.4f}  random100={r['random_formula_best']:.4f}  "
              f"systematic={r['systematic_best']:.4f}  OpenFE={r.get('base_openfe', float('nan')):.4f}  "
              f"GP={r.get('base_gp', float('nan')):.4f}")

    return results_by_seed


def aggregate(all_results: dict):
    from scipy import stats as sp_stats

    lines = []
    a = lines.append

    a("# Idea 3 OpenFE + Symbolic Regression Comparison\n")
    a("**Goal**: Prove LLM-designed composites > SOTA AutoFE methods (OpenFE, Symbolic Regression).\n")
    a("**Methods compared**:\n")
    a("- **OpenFE** (NeurIPS 2023): Automatic feature engineering via mutual information + gradient boosting.\n")
    a("- **Symbolic Regression** (gplearn): Genetic programming to discover mathematical formulas.\n")
    a("- **LLM composites**: Qwen3-4B-Instruct designs 5 fraud-aware composite features.\n")
    a("- **Random formula**: Best of 100 random binary operations (val-selected).\n")
    a("- **Systematic**: Best of all pairwise products/ratios + log/sqrt transforms (val-selected).\n")
    a("- All evaluated: LR(base_logit + features) on test set.\n")
    a("- **Significance (df=4)**: |t|>2.78 → ★; |t|>4.60 → ★★; |t|>8.61 → ★★★\n")

    methods = ["base_only", "base_raw", "random_formula_best", "systematic_best",
               "base_openfe", "base_gp", "base_llm"]
    labels = {
        "base_only": "Base only",
        "base_raw": "Base+raw (LR)",
        "random_formula_best": "Random formula (best of 100)",
        "systematic_best": "Systematic transforms (best)",
        "base_openfe": "Base+OpenFE",
        "base_gp": "Base+GP (gplearn)",
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
            vals = [seed_results[s].get(m, float("nan")) for s in SEEDS]
            valid_vals = [v for v in vals if not np.isnan(v)]
            if valid_vals:
                method_arrays[m].extend(valid_vals)
                mean = np.mean(valid_vals)
                std = np.std(valid_vals, ddof=1) if len(valid_vals) > 1 else 0.0
                row.append(f"{mean:.4f} ± {std:.4f}")
            else:
                method_arrays[m].extend([float("nan")] * len(vals))
                row.append("—")
        a("| " + " | ".join(row) + " |")

    # Overall means
    a("\n## Overall Mean AUPRC (across all cells)\n")
    a("| Method | Mean ± SD | N |")
    a("|---|---:|---:|")
    for m in methods:
        vals = [v for v in method_arrays[m] if not np.isnan(v)]
        if vals:
            a(f"| {labels[m]} | {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f} | {len(vals)} |")
        else:
            a(f"| {labels[m]} | — | 0 |")

    # Paired-t: LLM vs each baseline
    a("\n## Paired-t: Base+LLM vs baselines\n")
    a("| Comparison | Δ | t | p | sig |")
    a("|---|---:|---:|---:|:---:|")

    llm_all = []
    for (dataset, base), seed_results in all_results.items():
        for s in SEEDS:
            llm_all.append(seed_results[s].get("base_llm", float("nan")))
    llm_arr = np.array(llm_all)

    for m in ["base_only", "base_raw", "random_formula_best", "systematic_best", "base_openfe", "base_gp"]:
        other_all = []
        for (dataset, base), seed_results in all_results.items():
            for s in SEEDS:
                other_all.append(seed_results[s].get(m, float("nan")))
        other_arr = np.array(other_all)

        # Filter paired non-NaN
        mask = ~np.isnan(llm_arr) & ~np.isnan(other_arr)
        if mask.sum() < 3:
            a(f"| Base+LLM vs {labels[m]} | — | — | — | insufficient data |")
            continue

        diff = llm_arr[mask] - other_arr[mask]
        n = len(diff)
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
        p = 2 * (1 - sp_stats.t.cdf(abs(t), df=n - 1))
        sig = "★★★" if abs(t) > 8.61 else ("★★" if abs(t) > 4.60 else ("★" if abs(t) > 2.78 else "ns"))
        a(f"| Base+LLM vs {labels[m]} | {diff.mean():+.4f} | {t:+.3f} | {p:.4f} ({n} pairs) | {sig} |")

    # Win-rate table
    a("\n## LLM Win Rate vs Baselines\n")
    a("| vs Method | Win | Tie | Loss | Win% |")
    a("|---|---:|---:|---:|---:|")

    for m in ["base_only", "base_raw", "random_formula_best", "systematic_best", "base_openfe", "base_gp"]:
        wins = ties = losses = 0
        for (dataset, base), seed_results in all_results.items():
            for s in SEEDS:
                llm_val = seed_results[s].get("base_llm", float("nan"))
                other_val = seed_results[s].get(m, float("nan"))
                if np.isnan(llm_val) or np.isnan(other_val):
                    continue
                if llm_val > other_val + 1e-6:
                    wins += 1
                elif other_val > llm_val + 1e-6:
                    losses += 1
                else:
                    ties += 1
        total = wins + ties + losses
        if total > 0:
            a(f"| vs {labels[m]} | {wins} | {ties} | {losses} | {100*wins/total:.0f}% |")
        else:
            a(f"| vs {labels[m]} | — | — | — | — |")

    # Per-cell detailed table
    a("\n## Per-Cell Detailed Results\n")
    a("| Cell | Seed | Base | Raw | Random | Systematic | OpenFE | GP | LLM |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (dataset, base), seed_results in all_results.items():
        for s in SEEDS:
            r = seed_results[s]
            row = [f"{dataset}-{base}", str(s)]
            for m in ["base_only", "base_raw", "random_formula_best", "systematic_best",
                       "base_openfe", "base_gp", "base_llm"]:
                v = r.get(m, float("nan"))
                row.append(f"{v:.4f}" if not np.isnan(v) else "—")
            a("| " + " | ".join(row) + " |")

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
            seed_results = run_cell(dataset, base, args.device)
            all_results[(dataset, base)] = seed_results
        except Exception as e:
            print(f"\n[ERROR] {dataset}-{base}: {e}")
            import traceback; traceback.print_exc()
            continue

    report = aggregate(all_results)
    print("\n" + report)

    out_dir = Path("artifacts/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "idea3_openfe_comparison.md").write_text(report)
    print(f"\nSaved to {out_dir / 'idea3_openfe_comparison.md'}")

    # Also save raw JSON
    json_results = {}
    for (dataset, base), seed_results in all_results.items():
        key = f"{dataset}_{base}"
        json_results[key] = {}
        for s, r in seed_results.items():
            json_results[key][str(s)] = {k: float(v) if not np.isnan(v) else None for k, v in r.items()}
    (out_dir / "idea3_openfe_comparison.json").write_text(json.dumps(json_results, indent=2))
    print(f"JSON saved to {out_dir / 'idea3_openfe_comparison.json'}")


if __name__ == "__main__":
    main()
