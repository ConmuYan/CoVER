"""Back-compat shim — RelDistillAdapter now lives in ``models.flash_adapter``.

This module re-exports ``FlashAdapter`` (aliased as ``RelDistillAdapter``)
so all existing import sites in ``scripts/train_distill_adapter.py`` and
``scripts/aggregate_idea2c_2b_distill.py`` continue to work unchanged.

The v1 ``RelDistillAdapter`` semantics are preserved when ``forward`` is
called without ``return_heads=True``: the returned dict matches the v1
signature (``delta_phi, gate_logits, gate_probs, final_logit``) key-for-key.

The G-OPD-Flash three-head matching interface (``return_heads=True``) is
opt-in and only consumed by ``scripts/train_g_opd_flash.py``.

See ``models/flash_adapter.py`` and ``docs/OPD_FLASH_DESIGN_v3.md`` §3.3.
"""

from __future__ import annotations

from models.flash_adapter import FlashAdapter as _FlashAdapter

#: Back-compat alias.  v1 RelDistillAdapter is now a FlashAdapter instance.
RelDistillAdapter = _FlashAdapter


__all__ = ["RelDistillAdapter"]
