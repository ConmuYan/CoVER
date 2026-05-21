"""Back-compat shim — RelDistillAdapter now lives in ``models.cbr_flash_adapter``.

This module re-exports ``FlashAdapter`` (aliased as ``RelDistillAdapter``)
so all existing import sites in ``scripts/train_distill_adapter.py`` and
``scripts/aggregate_idea2c_2b_distill.py`` continue to work unchanged.

The v1 ``RelDistillAdapter`` semantics are preserved when ``forward`` is
called without ``return_heads=True``: the returned dict matches the v1
signature (``delta_phi, gate_logits, gate_probs, final_logit``) key-for-key.

The CBR-Flash three-head matching interface (``return_heads=True``) is
opt-in and consumed by ``scripts/train_cbr_flash.py``.

See ``models/cbr_flash_adapter.py`` and the CBR-Flash distillation narrative.
"""

from __future__ import annotations

from models.cbr_flash_adapter import FlashAdapter as _FlashAdapter

#: Back-compat alias.  v1 RelDistillAdapter is now a FlashAdapter instance.
RelDistillAdapter = _FlashAdapter


__all__ = ["RelDistillAdapter"]
