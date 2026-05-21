"""Compatibility entry point for the old CBR-Flash aggregator name.

Canonical script: :mod:`scripts.aggregate_cbr_flash`.
"""

from __future__ import annotations

from scripts.aggregate_cbr_flash import *  # noqa: F401,F403
from scripts.aggregate_cbr_flash import main


if __name__ == "__main__":
    main()
