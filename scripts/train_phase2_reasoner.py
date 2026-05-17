#!/usr/bin/env python
"""DEPRECATED: use scripts/train_phase3_reasoner.py for Phase 3 LEQA workflow.

The Phase 2 trainer is preserved at this path for git-blame continuity but
raises immediately. See PHASE2_DEPRECATION_PLAN_CORRECTION.md.
"""
import sys
print("ERROR: scripts/train_phase2_reasoner.py is deprecated. "
      "Use scripts/train_phase3_reasoner.py.", file=sys.stderr)
sys.exit(2)
