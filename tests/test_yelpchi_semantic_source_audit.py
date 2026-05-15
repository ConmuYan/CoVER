from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.audit_yelpchi_semantic_sources import MatKeyInfo, audit_sources_from_key_info


def test_audit_falls_back_to_feature_buckets_for_local_yelpchi_shape():
    infos = [
        MatKeyInfo("features", (45954, 32), "float64", "f", nnz=1469088),
        MatKeyInfo("label", (1, 45954), "int64", "i"),
        MatKeyInfo("homo", (45954, 45954), "float64", "f", nnz=7693958),
        MatKeyInfo("net_rur", (45954, 45954), "float64", "f", nnz=98630),
        MatKeyInfo("net_rtr", (45954, 45954), "float64", "f", nnz=1147232),
        MatKeyInfo("net_rsr", (45954, 45954), "float64", "f", nnz=6805486),
    ]

    rows, summary = audit_sources_from_key_info(infos)
    by_source = {row["source"]: row for row in rows}

    assert summary["decision"] == "use_feature_buckets_relation_summaries"
    assert summary["raw_text_available"] is False
    assert summary["feature_count"] == 32
    assert summary["relation_keys_available"] == {
        "R-U-R": True,
        "R-S-R": True,
        "R-T-R": True,
    }
    assert by_source["raw_review_text"]["available"] == "false"
    assert by_source["feature_names"]["decision_note"] == "semanticize as feature_00..feature_31 buckets"
    assert by_source["relation_R_U_R"]["available"] == "true"


def test_audit_prefers_raw_text_when_review_text_key_exists():
    infos = [
        MatKeyInfo("review_text", (10, 1), "object", "O"),
        MatKeyInfo("features", (10, 32), "float64", "f", nnz=320),
        MatKeyInfo("feature_names", (1, 32), "object", "O"),
        MatKeyInfo("net_rur", (10, 10), "float64", "f", nnz=12),
    ]

    rows, summary = audit_sources_from_key_info(infos)
    by_source = {row["source"]: row for row in rows}

    assert summary["decision"] == "use_raw_text"
    assert summary["raw_text_available"] is True
    assert summary["feature_names_available"] is True
    assert by_source["raw_review_text"]["available"] == "true"
    assert by_source["raw_review_text"]["decision_note"] == "use raw text"
