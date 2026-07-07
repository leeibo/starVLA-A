#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANNER_OFT_STAGE=planner exec "${SCRIPT_DIR}/run_train.sh" "$@"
