"""Compatibility shim for the former CoVER-REL teacher module.

Canonical implementation now lives in :mod:`models.raer_teacher`.
"""

from __future__ import annotations

from models.raer_teacher import CoVERRelReasoner, RAERTeacher

__all__ = ["CoVERRelReasoner", "RAERTeacher"]
