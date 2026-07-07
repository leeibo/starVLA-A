#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="$(basename "${SCRIPT_DIR}")"
export PLANNER_OFT_STAGE="${PLANNER_OFT_STAGE:-vla}"
exec bash "${SCRIPT_DIR}/../start_managed_train.sh" "${RUN_NAME}" "$@"
