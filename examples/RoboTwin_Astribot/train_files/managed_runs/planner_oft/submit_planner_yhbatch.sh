#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/XYAIFS00/HDD_POOL/hlkj_zql/hlkj_zql_8/code/starVLA/examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft"
if [[ -n "${STARGVLA_REPO_ROOT:-}" ]]; then
  SCRIPT_DIR="${STARGVLA_REPO_ROOT}/examples/RoboTwin_Astribot/train_files/managed_runs/planner_oft"
fi

PLANNER_OFT_STAGE=planner exec "${SCRIPT_DIR}/submit_yhbatch.sh" "$@"
