"""Compatibility shim for the former FlashAdapter module.

Canonical CBR-Flash implementation now lives in :mod:`models.cbr_flash_adapter`.
"""

from __future__ import annotations

from models.cbr_flash_adapter import CBRFlashAdapter, FlashAdapter, RelDistillAdapter

__all__ = ["CBRFlashAdapter", "FlashAdapter", "RelDistillAdapter"]
