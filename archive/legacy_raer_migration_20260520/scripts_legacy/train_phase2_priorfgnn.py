"""Compatibility entry point for the old PriorF-adapted phase-2 script.

Canonical script: :mod:`scripts.train_raer_priorfgnn`.
"""

from __future__ import annotations

from scripts.train_raer_priorfgnn import *  # noqa: F401,F403
from scripts.train_raer_priorfgnn import main


if __name__ == "__main__":
    raise SystemExit(main())
