#!/usr/bin/env bash
set -euo pipefail

# ======================
# Config
# ======================
REPO_ID="conroy1201/robotwin-ckpt"
REPO_TYPE="model"

SRC_DIR="/HOME/hlkj_zql/hlkj_zql_8/HDD_POOL/data/RoboTwin_Astribot/lerobot_data/local/astribot/beat_block_hammer_rotate_view"

# Upload files under SRC_DIR to the repository root.
REMOTE_BASE=""

UPLOAD_JOBS="${UPLOAD_JOBS:-4}"
MAX_RETRY="${MAX_RETRY:-10}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-20}"

# Keep this consistent with upload_robotwin_ckpt.sh.
export PATH="/HOME/hlkj_zql/hlkj_zql_8/.local/bin:$PATH"

# ======================
# Helpers
# ======================
validate_positive_int() {
  local name="$1"
  local value="$2"

  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${name} must be a positive integer, got: ${value}"
    exit 1
  fi
}

check_requirements() {
  if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: source directory not found:"
    echo "  $SRC_DIR"
    exit 1
  fi

  if ! command -v modelscope >/dev/null 2>&1; then
    echo "ERROR: modelscope command not found."
    echo "Install ModelScope CLI or add it to PATH, then rerun this script."
    exit 1
  fi
}

remote_path_for_rel_path() {
  local rel_path="$1"

  if [ -n "$REMOTE_BASE" ]; then
    printf '%s/%s\n' "$REMOTE_BASE" "$rel_path"
  else
    printf '%s\n' "$rel_path"
  fi
}

ms_upload_retry() {
  local local_path="$1"
  local remote_path="$2"

  local retry=1

  while [ "$retry" -le "$MAX_RETRY" ]; do
    echo
    echo "Upload attempt ${retry}/${MAX_RETRY}"
    echo "  local : ${local_path}"
    echo "  remote: ${remote_path}"

    if modelscope upload \
      "${REPO_ID}" \
      "${local_path}" \
      "${remote_path}" \
      --repo-type "${REPO_TYPE}"; then
      echo "Upload success: ${remote_path}"
      return 0
    fi

    echo "Upload failed, retrying in ${RETRY_SLEEP_SECONDS} seconds..."
    retry=$((retry + 1))
    sleep "$RETRY_SLEEP_SECONDS"
  done

  echo "ERROR: upload failed after ${MAX_RETRY} attempts:"
  echo "  ${local_path}"
  return 1
}

wait_for_oldest_job() {
  local -n pids_ref="$1"
  local -n failed_ref="$2"
  local pid="${pids_ref[0]}"
  local remaining=("${pids_ref[@]:1}")

  if ! wait "$pid"; then
    failed_ref=1
  fi

  pids_ref=("${remaining[@]}")
}

wait_for_all_jobs() {
  local -n pids_ref="$1"
  local -n failed_ref="$2"

  while [ "${#pids_ref[@]}" -gt 0 ]; do
    wait_for_oldest_job "$1" "$2"
  done
}

upload_one_file() {
  local local_file="$1"
  local rel_path="${local_file#${SRC_DIR}/}"
  local remote_path
  remote_path="$(remote_path_for_rel_path "$rel_path")"

  ms_upload_retry "$local_file" "$remote_path"
}

upload_all_files() {
  local upload_pids=()
  local upload_failed=0
  local file_count=0
  local file

  while IFS= read -r -d '' file; do
    (
      upload_one_file "$file"
    ) &

    upload_pids+=("$!")
    file_count=$((file_count + 1))

    if [ "${#upload_pids[@]}" -ge "$UPLOAD_JOBS" ]; then
      wait_for_oldest_job upload_pids upload_failed
    fi
  done < <(find "$SRC_DIR" -type f -print0 | sort -z)

  wait_for_all_jobs upload_pids upload_failed

  if [ "$upload_failed" -ne 0 ]; then
    echo
    echo "ERROR: one or more file uploads failed."
    return 1
  fi

  echo
  echo "Uploaded files: ${file_count}"
}

validate_positive_int "UPLOAD_JOBS" "$UPLOAD_JOBS"
validate_positive_int "MAX_RETRY" "$MAX_RETRY"
validate_positive_int "RETRY_SLEEP_SECONDS" "$RETRY_SLEEP_SECONDS"
check_requirements

echo "========================================"
echo "Repo:        ${REPO_ID}"
echo "Repo type:   ${REPO_TYPE}"
echo "Source dir:  ${SRC_DIR}"
echo "Remote base: <repo root>"
echo "Upload jobs: ${UPLOAD_JOBS}"
echo "Max retry:   ${MAX_RETRY}"
echo "========================================"

upload_all_files

echo
echo "Upload finished."
