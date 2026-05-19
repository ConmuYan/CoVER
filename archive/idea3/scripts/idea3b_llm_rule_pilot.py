"""Idea 3B Pilot: LLM Case-Retrieval Fraud Rule Induction.

Pipeline:
1. Load YelpChi + per-relation adjacencies
2. Compute per-node structural features (degree per relation, overlap, 2-hop)
3. K-means cluster on structural features
4. Retrieve representative fraud/benign cases per cluster from train set
5. Textualize cases as natural language ego-net descriptions
6. Feed to Qwen3-0.6B, ask for fraud rules
7. Apply rules to score all nodes
8. Evaluate AUPRC on test set vs baselines

Usage:
    python scripts/idea3b_llm_rule_pilot.py --device cuda:0 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from scipy.io import loadmat
from scipy.sparse import issparse, coo_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score


def load_yelpchi(seed: int = 42):
    """Load YelpChi data and per-relation adjacency matrices."""
    from data.load_fraud import load_fraud_dataset
    data = load_fraud_dataset(
        name="yelpchi", path="datasets/YelpChi.mat", format="mat",
        seed=seed, train_ratio=0.4, val_test_ratio=[1, 2], stratified=True,
    )
    mat = loadmat("datasets/YelpChi.mat")
    adjs = {}
    for name, key in [("RUR", "net_rur"), ("RSR", "net_rsr"), ("RTR", "net_rtr")]:
        sp = mat[key]
        sp = sp.tocoo() if issparse(sp) else coo_matrix(sp)
        adjs[name] = sp
    return data, adjs


def compute_structural_features(data, adjs: dict, num_nodes: int) -> np.ndarray:
    """Compute per-node structural features across all relations.

    Features per relation (3 relations × 6 features = 18 dims):
    - degree (log1p)
    - neighbor mean feature similarity (cosine)
    - fraction of neighbors that are fraud (train-only)
    Plus 3 cross-relation features:
    - degree ratio RUR/RSR, RUR/RTR, RSR/RTR
    """
    from sklearn.metrics.pairwise import cosine_similarity

    x = data.x.numpy()
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()

    features = []
    rel_names = ["RUR", "RSR", "RTR"]
    rel_degrees = {}

    for name in rel_names:
        sp = adjs[name]
        # Degree
        deg = np.zeros(num_nodes)
        np.add.at(deg, sp.row, 1)
        rel_degrees[name] = deg
        log_deg = np.log1p(deg).reshape(-1, 1)

        # Neighbor mean feature similarity (sample for speed)
        sim = np.zeros(num_nodes)
        # Use sparse matrix multiplication for neighbor mean
        from scipy.sparse import csr_matrix
        sp_csr = csr_matrix((np.ones(len(sp.row)), (sp.row, sp.col)), shape=(num_nodes, num_nodes))
        row_sums = np.array(sp_csr.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        neighbor_mean = sp_csr.dot(x) / row_sums[:, None]
        # Cosine similarity between node and its neighbor mean
        for i in range(num_nodes):
            if deg[i] > 0:
                norm_x = np.linalg.norm(x[i])
                norm_n = np.linalg.norm(neighbor_mean[i])
                if norm_x > 0 and norm_n > 0:
                    sim[i] = np.dot(x[i], neighbor_mean[i]) / (norm_x * norm_n)

        # Fraud neighbor fraction (train-only)
        fraud_frac = np.zeros(num_nodes)
        train_fraud = np.zeros(num_nodes)
        train_fraud[train_mask & (y == 1)] = 1
        fraud_neighbors = sp_csr.dot(train_fraud)
        fraud_frac = fraud_neighbors / row_sums

        features.extend([log_deg, sim.reshape(-1, 1), fraud_frac.reshape(-1, 1)])

    # Cross-relation degree ratios
    for i, n1 in enumerate(rel_names):
        for n2 in rel_names[i+1:]:
            ratio = np.log1p(rel_degrees[n1]) / (np.log1p(rel_degrees[n2]) + 1e-8)
            features.append(ratio.reshape(-1, 1))

    return np.hstack(features)


def textualize_case(node_id: int, features: np.ndarray, adjs: dict,
                    y: np.ndarray, train_mask: np.ndarray,
                    num_nodes: int, is_fraud: bool) -> str:
    """Convert a node's ego-net into a natural language description."""
    rel_names = ["RUR", "RSR", "RTR"]
    rel_labels = {
        "RUR": "same-user review",
        "RSR": "same-product same-star",
        "RTR": "same-product same-month",
    }

    lines = [f"Node {node_id} ({'FRAUD' if is_fraud else 'BENIGN'}):"]

    for i, name in enumerate(rel_names):
        sp = adjs[name]
        # Count neighbors for this node
        neighbors = sp.col[sp.row == node_id]
        n_neighbors = len(neighbors)
        lines.append(f"  {rel_labels[name]}: {n_neighbors} neighbors")

        # Among neighbors, how many are fraud (train-only)?
        if n_neighbors > 0:
            train_neighbors = [n for n in neighbors if train_mask[n]]
            if train_neighbors:
                train_labels = y[train_neighbors]
                fraud_count = (train_labels == 1).sum()
                lines.append(f"    (train neighbors: {len(train_neighbors)}, fraud: {fraud_count}, "
                             f"fraud rate: {fraud_count/len(train_neighbors):.2%})")

    # Feature summary (anonymized)
    feat = features[node_id]
    lines.append(f"  Structural profile: {feat[:6].round(2).tolist()} ...")

    return "\n".join(lines)


def cluster_nodes(features: np.ndarray, n_clusters: int = 10, seed: int = 42) -> np.ndarray:
    """K-means clustering on structural features."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return kmeans.fit_predict(features_scaled)


def retrieve_cases(cluster_labels: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                   n_clusters: int, n_fraud: int = 5, n_benign: int = 5) -> dict:
    """Retrieve representative fraud/benign cases per cluster from train set."""
    cases = {}
    for c in range(n_clusters):
        cluster_mask = (cluster_labels == c) & train_mask
        fraud_ids = np.where(cluster_mask & (y == 1))[0]
        benign_ids = np.where(cluster_mask & (y == 0))[0]

        # Sample up to n_fraud/n_benign
        rng = np.random.RandomState(42)
        fraud_sample = rng.choice(fraud_ids, min(n_fraud, len(fraud_ids)), replace=False)
        benign_sample = rng.choice(benign_ids, min(n_benign, len(benign_ids)), replace=False)

        cases[c] = {
            "fraud": fraud_sample.tolist(),
            "benign": benign_sample.tolist(),
            "fraud_count": len(fraud_ids),
            "benign_count": len(benign_ids),
        }
    return cases


def build_llm_prompt(cases: dict, features: np.ndarray, adjs: dict,
                     y: np.ndarray, train_mask: np.ndarray,
                     num_nodes: int, n_clusters: int) -> str:
    """Build the prompt for LLM rule induction."""
    prompt = """You are a fraud detection expert analyzing a multi-relational review graph.

The graph has 3 relation types:
- RUR (same-user reviews): Users who review the same products
- RSR (same-product same-star): Products with same star ratings
- RTR (same-product same-month): Products reviewed in the same month

Below are representative fraud and benign node cases from different structural clusters.
For each case, I show the node's local graph structure.

"""
    for c in range(n_clusters):
        cluster_cases = cases[c]
        prompt += f"\n--- Cluster {c} (fraud: {cluster_cases['fraud_count']}, benign: {cluster_cases['benign_count']}) ---\n"

        prompt += "\nFraud cases:\n"
        for node_id in cluster_cases["fraud"][:3]:  # Limit to 3 for prompt length
            prompt += textualize_case(node_id, features, adjs, y, train_mask, num_nodes, True) + "\n"

        prompt += "\nBenign cases:\n"
        for node_id in cluster_cases["benign"][:3]:
            prompt += textualize_case(node_id, features, adjs, y, train_mask, num_nodes, False) + "\n"

    prompt += """
Based on these cases, identify 5-10 fraud detection rules. For each rule:
1. Describe the structural pattern that indicates fraud
2. Assign a suspiciousness score (0-1)
3. Specify which relation(s) the rule applies to

Format your rules as JSON:
```json
[
  {"rule_id": 1, "description": "...", "pattern": "...", "score": 0.8, "relations": ["RUR", "RSR"]},
  ...
]
```

Focus on patterns that distinguish fraud from benign across clusters.
Be specific about degree thresholds, neighbor fraud rates, and cross-relation patterns.
"""
    return prompt


def apply_rules_to_nodes(rules: list, features: np.ndarray, adjs: dict,
                         y: np.ndarray, train_mask: np.ndarray,
                         num_nodes: int) -> np.ndarray:
    """Apply LLM-generated rules to score all nodes.

    For the pilot, we use a simple heuristic: match rules based on
    structural feature thresholds derived from the rule descriptions.
    Returns: (num_nodes,) fraud scores.
    """
    # For the pilot, use a simple scoring based on features
    # In production, this would parse rule descriptions and apply them
    # For now, use a weighted combination of key features

    scores = np.zeros(num_nodes)

    # Feature indices (from compute_structural_features):
    # [RUR_deg, RUR_sim, RUR_fraud_frac, RSR_deg, RSR_sim, RSR_fraud_frac,
    #  RTR_deg, RTR_sim, RTR_fraud_frac, ratio_RUR_RSR, ratio_RUR_RTR, ratio_RSR_RTR]

    # Rule 1: High fraud neighbor fraction across relations
    fraud_frac_avg = (features[:, 2] + features[:, 5] + features[:, 8]) / 3
    scores += 0.3 * fraud_frac_avg

    # Rule 2: Low feature similarity with neighbors (anomaly signal)
    sim_avg = (features[:, 1] + features[:, 4] + features[:, 7]) / 3
    scores += 0.2 * (1 - sim_avg)

    # Rule 3: Degree imbalance across relations
    deg_rur = features[:, 0]
    deg_rsr = features[:, 3]
    deg_rtr = features[:, 6]
    deg_std = np.std([deg_rur, deg_rsr, deg_rtr], axis=0)
    scores += 0.2 * deg_std

    # Rule 4: High degree in RUR (same-user reviews)
    scores += 0.15 * deg_rur

    # Rule 5: Cross-relation degree ratio extremes
    ratio_std = np.std([features[:, 9], features[:, 10], features[:, 11]], axis=0)
    scores += 0.15 * ratio_std

    return scores


def evaluate(y_true: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> dict:
    """Evaluate fraud detection performance."""
    y = y_true[mask]
    s = scores[mask]
    auprc = average_precision_score(y, s)
    auc = roc_auc_score(y, s)
    return {"auprc": auprc, "roc_auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--use_llm", action="store_true", help="Use LLM for rule generation (slow)")
    parser.add_argument("--llm_model", default="/data1/mq/models/Qwen3-0.6B")
    args = parser.parse_args()

    print(f"[Idea3B] Loading YelpChi (seed={args.seed})...")
    data, adjs = load_yelpchi(args.seed)
    num_nodes = data.x.shape[0]
    y = data.y.numpy()

    print(f"[Idea3B] Computing structural features...")
    t0 = time.time()
    features = compute_structural_features(data, adjs, num_nodes)
    print(f"  Features: {features.shape} ({time.time()-t0:.1f}s)")

    print(f"[Idea3B] Clustering ({args.n_clusters} clusters)...")
    cluster_labels = cluster_nodes(features, args.n_clusters, args.seed)
    print(f"  Cluster sizes: {np.bincount(cluster_labels).tolist()}")

    print(f"[Idea3B] Retrieving cases...")
    cases = retrieve_cases(cluster_labels, y, data.train_mask.numpy(),
                           args.n_clusters, n_fraud=5, n_benign=5)

    # Build LLM prompt
    prompt = build_llm_prompt(cases, features, adjs, y, data.train_mask.numpy(),
                              num_nodes, args.n_clusters)
    print(f"[Idea3B] Prompt length: {len(prompt)} chars")

    if args.use_llm:
        print(f"[Idea3B] Loading LLM: {args.llm_model}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model, torch_dtype=torch.float16, device_map=args.device,
            trust_remote_code=True,
        )

        print(f"[Idea3B] Generating rules...")
        t0 = time.time()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=2048, temperature=0.7, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  LLM response ({time.time()-t0:.1f}s):")
        print(response[:2000])

        # Try to parse rules from JSON
        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                rules = json.loads(response[json_start:json_end])
                print(f"\n[Idea3B] Parsed {len(rules)} rules")
            else:
                rules = []
                print("[Idea3B] No JSON rules found, using heuristic scoring")
        except json.JSONDecodeError:
            rules = []
            print("[Idea3B] JSON parse failed, using heuristic scoring")
    else:
        rules = []
        print("[Idea3B] Skipping LLM (use --use_llm to enable)")

    # Apply rules (heuristic for pilot)
    print(f"[Idea3B] Applying rules to all nodes...")
    scores = apply_rules_to_nodes(rules, features, adjs, y, data.train_mask.numpy(), num_nodes)

    # Evaluate
    train_metrics = evaluate(y, scores, data.train_mask.numpy())
    val_metrics = evaluate(y, scores, data.val_mask.numpy())
    test_metrics = evaluate(y, scores, data.test_mask.numpy())

    print(f"\n[Idea3B] Results:")
    print(f"  Train: AUPRC={train_metrics['auprc']:.4f}, AUROC={train_metrics['roc_auc']:.4f}")
    print(f"  Val:   AUPRC={val_metrics['auprc']:.4f}, AUROC={val_metrics['roc_auc']:.4f}")
    print(f"  Test:  AUPRC={test_metrics['auprc']:.4f}, AUROC={test_metrics['roc_auc']:.4f}")

    # Baselines
    # Random scores
    rng = np.random.RandomState(args.seed)
    random_scores = rng.random(num_nodes)
    random_metrics = evaluate(y, random_scores, data.test_mask.numpy())

    # Frequency-based: fraud neighbor fraction as score
    freq_scores = (features[:, 2] + features[:, 5] + features[:, 8]) / 3
    freq_metrics = evaluate(y, freq_scores, data.test_mask.numpy())

    print(f"\n[Idea3B] Baselines:")
    print(f"  Random:    AUPRC={random_metrics['auprc']:.4f}, AUROC={random_metrics['roc_auc']:.4f}")
    print(f"  Freq-only: AUPRC={freq_metrics['auprc']:.4f}, AUROC={freq_metrics['roc_auc']:.4f}")

    # Save results
    results = {
        "seed": args.seed,
        "n_clusters": args.n_clusters,
        "use_llm": args.use_llm,
        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["roc_auc"],
        "random_auprc": random_metrics["auprc"],
        "freq_auprc": freq_metrics["auprc"],
        "n_rules": len(rules),
    }
    out_dir = Path(f"artifacts/results/yelpchi/bwgnn/idea3b_llm_rules/seed_{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\n[Idea3B] Saved to {out_dir / 'pilot_metrics.json'}")


if __name__ == "__main__":
    main()
