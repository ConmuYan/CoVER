#!/usr/bin/env bash
# Compatibility wrapper. Canonical runner: scripts/run_cbr_flash_deployshift.sh
exec "$(dirname "$0")/run_cbr_flash_deployshift.sh" "$@"
