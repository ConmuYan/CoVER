"""Compatibility shim for the former learned evidence extractor module.

Canonical implementation now lives in :mod:`evidence.lree`.
"""

from __future__ import annotations

from evidence.lree import (
    LREE,
    LearnedRelationEvidenceExtractor,
    RelationGCNEncoder,
    build_learned_extractor,
    row_normalize_sparse_adj,
    symmetric_normalize_sparse_adj,
)

__all__ = [
    "LREE",
    "LearnedRelationEvidenceExtractor",
    "RelationGCNEncoder",
    "build_learned_extractor",
    "symmetric_normalize_sparse_adj",
    "row_normalize_sparse_adj",
]
