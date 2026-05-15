from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import ensure_dir


TEXT_KEY_HINTS = ("text", "review_content", "content", "comment", "body", "doc")
USER_KEY_HINTS = ("user", "userid", "user_id", "reviewer")
PRODUCT_KEY_HINTS = ("product", "prod", "item", "business", "shop", "seller")
RATING_KEY_HINTS = ("rating", "star", "stars", "score_rating")
TIME_KEY_HINTS = ("time", "timestamp", "date", "month", "year")
FEATURE_NAME_HINTS = ("feature_name", "feature_names", "feat_name", "attr_name", "attribute")


@dataclass(frozen=True)
class MatKeyInfo:
    key: str
    shape: tuple[int, ...]
    dtype: str
    kind: str
    nnz: int | None = None

    @property
    def lower_key(self) -> str:
        return self.key.lower()

    @property
    def shape_text(self) -> str:
        return "x".join(str(x) for x in self.shape) if self.shape else "scalar"


def _has_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def _is_text_like(info: MatKeyInfo) -> bool:
    dtype = info.dtype.lower()
    return info.kind in {"U", "S", "O"} or "str" in dtype or "object" in dtype


def _source_row(
    source: str,
    available: bool,
    evidence: str,
    keys: list[str],
    decision_note: str,
) -> dict[str, str]:
    return {
        "source": source,
        "available": str(bool(available)).lower(),
        "evidence": evidence,
        "keys": ";".join(keys),
        "decision_note": decision_note,
    }


def summarize_mat_keys(mat: dict[str, Any]) -> list[MatKeyInfo]:
    from scipy import sparse

    infos: list[MatKeyInfo] = []
    for key, value in sorted(mat.items()):
        if key.startswith("__"):
            continue
        shape = tuple(int(x) for x in getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", type(value).__name__))
        kind = str(getattr(getattr(value, "dtype", None), "kind", ""))
        nnz = int(value.nnz) if sparse.issparse(value) else None
        infos.append(MatKeyInfo(key=key, shape=shape, dtype=dtype, kind=kind, nnz=nnz))
    return infos


def audit_sources_from_key_info(infos: list[MatKeyInfo]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    by_key = {info.key: info for info in infos}

    raw_text = [i for i in infos if _has_any(i.lower_key, TEXT_KEY_HINTS) and _is_text_like(i)]
    user_ids = [i for i in infos if _has_any(i.lower_key, USER_KEY_HINTS)]
    product_ids = [i for i in infos if _has_any(i.lower_key, PRODUCT_KEY_HINTS)]
    ratings = [i for i in infos if _has_any(i.lower_key, RATING_KEY_HINTS)]
    times = [i for i in infos if _has_any(i.lower_key, TIME_KEY_HINTS)]
    feature_names = [i for i in infos if _has_any(i.lower_key, FEATURE_NAME_HINTS)]

    features = by_key.get("features")
    labels = by_key.get("label")
    homo = by_key.get("homo")
    net_rur = by_key.get("net_rur")
    net_rsr = by_key.get("net_rsr")
    net_rtr = by_key.get("net_rtr")

    feature_count = features.shape[1] if features is not None and len(features.shape) == 2 else None
    feature_note = (
        f"{feature_count} anonymous numeric features"
        if feature_count is not None and not feature_names
        else "named feature metadata present"
        if feature_names
        else "features matrix not found"
    )

    rows = [
        _source_row(
            "raw_review_text",
            bool(raw_text),
            "text-like key with review/content naming" if raw_text else "no text-like review/content key in .mat",
            [i.key for i in raw_text],
            "use raw text" if raw_text else "fall back to feature buckets",
        ),
        _source_row(
            "user_id",
            bool(user_ids),
            "key name contains user/reviewer" if user_ids else "only R-U-R relation is available",
            [i.key for i in user_ids],
            "direct user metadata available" if user_ids else "summarize same-user relation graph",
        ),
        _source_row(
            "product_id",
            bool(product_ids),
            "key name contains product/item/business" if product_ids else "only product-derived relations are available",
            [i.key for i in product_ids],
            "direct product metadata available" if product_ids else "summarize R-S-R/R-T-R relation graphs",
        ),
        _source_row(
            "rating_or_star",
            bool(ratings),
            "key name contains rating/star" if ratings else "rating appears only through R-S-R relation semantics",
            [i.key for i in ratings],
            "direct rating metadata available" if ratings else "use R-S-R relation summary",
        ),
        _source_row(
            "timestamp_or_month",
            bool(times),
            "key name contains time/date/month" if times else "time appears only through R-T-R relation semantics",
            [i.key for i in times],
            "direct temporal metadata available" if times else "use R-T-R relation summary",
        ),
        _source_row(
            "relation_R_U_R",
            net_rur is not None,
            _matrix_evidence(net_rur) if net_rur is not None else "net_rur key missing",
            ["net_rur"] if net_rur is not None else [],
            "same-user neighbor summary",
        ),
        _source_row(
            "relation_R_S_R",
            net_rsr is not None,
            _matrix_evidence(net_rsr) if net_rsr is not None else "net_rsr key missing",
            ["net_rsr"] if net_rsr is not None else [],
            "same-product-same-rating neighbor summary",
        ),
        _source_row(
            "relation_R_T_R",
            net_rtr is not None,
            _matrix_evidence(net_rtr) if net_rtr is not None else "net_rtr key missing",
            ["net_rtr"] if net_rtr is not None else [],
            "same-product-same-month neighbor summary",
        ),
        _source_row(
            "homogeneous_graph",
            homo is not None,
            _matrix_evidence(homo) if homo is not None else "homo key missing",
            ["homo"] if homo is not None else [],
            "existing structural graph prior",
        ),
        _source_row(
            "handcrafted_features",
            features is not None,
            _matrix_evidence(features) if features is not None else "features key missing",
            ["features"] if features is not None else [],
            feature_note,
        ),
        _source_row(
            "feature_names",
            bool(feature_names),
            "feature-name metadata key present" if feature_names else feature_note,
            [i.key for i in feature_names],
            "semanticize by name" if feature_names else "semanticize as feature_00..feature_31 buckets",
        ),
        _source_row(
            "labels",
            labels is not None,
            _matrix_evidence(labels) if labels is not None else "label key missing",
            ["label"] if labels is not None else [],
            "labels are for train-only retrieval/prototypes, never target packet fields",
        ),
    ]

    decision = "use_raw_text" if raw_text else "use_feature_buckets_relation_summaries"
    summary = {
        "decision": decision,
        "raw_text_available": bool(raw_text),
        "feature_count": feature_count,
        "relation_keys_available": {
            "R-U-R": net_rur is not None,
            "R-S-R": net_rsr is not None,
            "R-T-R": net_rtr is not None,
        },
        "feature_names_available": bool(feature_names),
        "num_nodes": features.shape[0] if features is not None and features.shape else None,
    }
    return rows, summary


def _matrix_evidence(info: MatKeyInfo) -> str:
    if info.nnz is None:
        return f"shape={info.shape_text}, dtype={info.dtype}"
    density = 0.0
    if len(info.shape) == 2 and info.shape[0] * info.shape[1] > 0:
        density = info.nnz / float(info.shape[0] * info.shape[1])
    return f"shape={info.shape_text}, dtype={info.dtype}, nnz={info.nnz}, density={density:.6g}"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    fields = ["source", "available", "evidence", "keys", "decision_note"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    dataset_path: Path,
    infos: list[MatKeyInfo],
    rows: list[dict[str, str]],
    summary: dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    lines = [
        "# YelpChi Semantic Source Audit",
        "",
        f"- Dataset path: `{dataset_path}`",
        f"- Decision: `{summary['decision']}`",
        f"- Raw review text available: `{str(summary['raw_text_available']).lower()}`",
        f"- Feature count: `{summary['feature_count']}`",
        f"- Feature names available: `{str(summary['feature_names_available']).lower()}`",
        f"- Relation keys available: `{json.dumps(summary['relation_keys_available'], sort_keys=True)}`",
        "",
        "## Source Availability",
        "",
        "| Source | Available | Evidence | Keys | Decision note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {source} | {available} | {evidence} | {keys} | {decision_note} |".format(
                source=row["source"],
                available=row["available"],
                evidence=row["evidence"].replace("|", "\\|"),
                keys=row["keys"] or "-",
                decision_note=row["decision_note"].replace("|", "\\|"),
            )
        )

    lines.extend([
        "",
        "## MAT Key Inventory",
        "",
        "| Key | Shape | Dtype | NNZ |",
        "| --- | --- | --- | --- |",
    ])
    for info in infos:
        lines.append(
            f"| {info.key} | {info.shape_text} | {info.dtype} | {info.nnz if info.nnz is not None else '-'} |"
        )

    lines.extend([
        "",
        "## CoVER-META-Lite Implication",
        "",
        "The local YelpChi artifact does not expose raw text or direct user/product/rating/time columns.",
        "The first evidence-packet version should therefore use anonymous 32-feature buckets plus relation-specific summaries from `net_rur`, `net_rsr`, and `net_rtr`.",
        "Train labels may only be used to build train-only prototype or retrieval summaries; target labels and split identity must not appear in packet fields.",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/YelpChi.mat")
    parser.add_argument(
        "--output_report",
        type=str,
        default="artifacts/reports/yelpchi_semantic_source_audit.md",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="artifacts/tables/yelpchi_semantic_source_audit.csv",
    )
    args = parser.parse_args()

    from scipy.io import loadmat

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path not found: {dataset_path}")
    mat = loadmat(dataset_path)
    infos = summarize_mat_keys(mat)
    rows, summary = audit_sources_from_key_info(infos)

    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_report), dataset_path, infos, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {args.output_report}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
