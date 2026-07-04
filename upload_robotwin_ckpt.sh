#!/usr/bin/env bash
set -euo pipefail

# ======================
# Config
# ======================
REPO_ID="conroy1201/robotwin-ckpt"
REPO_TYPE="model"

BASE_DIR="/HOME/hlkj_zql/hlkj_zql_8/HDD_POOL/code/starVLA/results/Checkpoints"

# .pt 文件切片大小
PART_SIZE="500M"

# 临时切片目录
TMP_BASE="/tmp/modelscope_pt_upload_parts"

# 同时上传多少个 checkpoint .pt
CHECKPOINT_UPLOAD_JOBS="${CHECKPOINT_UPLOAD_JOBS:-4}"

# 每个 .pt 的切片同时上传多少个 part
PART_UPLOAD_JOBS="${PART_UPLOAD_JOBS:-4}"

# 如果 modelscope 不在 PATH，自动补一下
export PATH="/HOME/hlkj_zql/hlkj_zql_8/.local/bin:$PATH"

# ======================
# Args
# ======================
if [ "$#" -lt 1 ]; then
  echo "Usage:"
  echo "  bash $0 <folder_name> [step_number|final|all ...]"
  echo
  echo "Examples:"
  echo "  bash $0 fast_subtask_action_12_wos"
  echo "  bash $0 fast_subtask_action_12_wos 5000"
  echo "  bash $0 fast_subtask_action_12_wos 5000 10000 15000"
  echo "  bash $0 fast_subtask_action_12_wos all"
  echo "  bash $0 fast_subtask_action_12_wos final"
  echo
  echo "Optional concurrency envs:"
  echo "  CHECKPOINT_UPLOAD_JOBS=4 PART_UPLOAD_JOBS=4 bash $0 fast_subtask_action_12_wos all"
  exit 1
fi

EXP_NAME="$1"
TARGETS=("${@:2}")

SRC_DIR="${BASE_DIR}/${EXP_NAME}"
REMOTE_BASE="ckpts/${EXP_NAME}"

if [ ! -d "$SRC_DIR" ]; then
  echo "ERROR: source directory not found:"
  echo "  $SRC_DIR"
  exit 1
fi

# ======================
# Upload with retry
# ======================
ms_upload_retry() {
  local local_path="$1"
  local remote_path="$2"

  local max_retry=10
  local retry=1

  while [ "$retry" -le "$max_retry" ]; do
    echo
    echo "Upload attempt ${retry}/${max_retry}"
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

    echo "Upload failed, retrying in 20 seconds..."
    retry=$((retry + 1))
    sleep 20
  done

  echo "ERROR: upload failed after ${max_retry} attempts:"
  echo "  ${local_path}"
  return 1
}

validate_positive_int() {
  local name="$1"
  local value="$2"

  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${name} must be a positive integer, got: ${value}"
    exit 1
  fi
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

# ======================
# Upload one non-pt file directly
# ======================
upload_normal_file() {
  local local_file="$1"
  local rel_path="$2"
  local remote_path="${REMOTE_BASE}/${rel_path}"

  echo
  echo "Uploading normal file:"
  echo "  local : ${local_file}"
  echo "  remote: ${remote_path}"

  ms_upload_retry "${local_file}" "${remote_path}"
}

# ======================
# Upload one .pt file by chunks
# ======================
upload_pt_by_parts() {
  local local_file="$1"
  local rel_path="$2"

  local filename
  filename="$(basename "$local_file")"

  local remote_file_path="${REMOTE_BASE}/${rel_path}"
  local remote_parts_dir="${remote_file_path}.parts"

  local safe_name
  safe_name="$(echo "${EXP_NAME}_${rel_path}" | sed 's#[/ ]#_#g')"

  local tmp_dir="${TMP_BASE}/${safe_name}"

  echo
  echo "Uploading .pt file by parts:"
  echo "  local file      : ${local_file}"
  echo "  remote parts dir: ${remote_parts_dir}"
  echo "  tmp dir         : ${tmp_dir}"

  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"

  echo
  echo "Computing sha256..."
  sha256sum "${local_file}" > "${tmp_dir}/${filename}.sha256"

  echo
  echo "Splitting ${filename} into ${PART_SIZE} parts..."
  split \
    -b "${PART_SIZE}" \
    -d \
    -a 5 \
    "${local_file}" \
    "${tmp_dir}/part_"

  echo
  echo "Generated parts:"
  ls -lh "${tmp_dir}"/part_*

  echo
  echo "Uploading sha256 file..."
  ms_upload_retry \
    "${tmp_dir}/${filename}.sha256" \
    "${remote_file_path}.sha256"

  echo
  echo "Uploading parts in parallel..."
  echo "  part upload jobs: ${PART_UPLOAD_JOBS}"
  local part_count=0
  local part_failed=0
  local part_pids=()

  while IFS= read -r -d '' part_file; do
    local part_name
    part_name="$(basename "$part_file")"

    (
      ms_upload_retry \
        "${part_file}" \
        "${remote_parts_dir}/${part_name}"
    ) &

    part_pids+=("$!")
    part_count=$((part_count + 1))

    if [ "${#part_pids[@]}" -ge "$PART_UPLOAD_JOBS" ]; then
      wait_for_oldest_job part_pids part_failed
    fi
  done < <(find "${tmp_dir}" -maxdepth 1 -type f -name "part_*" -print0 | sort -z)

  wait_for_all_jobs part_pids part_failed

  if [ "$part_failed" -ne 0 ]; then
    echo
    echo "ERROR: one or more parts failed to upload:"
    echo "  ${local_file}"
    rm -rf "${tmp_dir}"
    return 1
  fi

  echo
  echo "Uploaded ${part_count} parts for:"
  echo "  ${local_file}"

  echo
  echo "To restore after download:"
  echo "  cat ${filename}.parts/part_* > ${filename}"
  echo "  sha256sum -c ${filename}.sha256"

  rm -rf "${tmp_dir}"
}

# ======================
# Upload file auto
# .pt -> split upload
# others -> normal upload
# ======================
upload_one_file() {
  local local_file="$1"
  local rel_path="$2"

  if [[ "${local_file}" == *.pt ]]; then
    upload_pt_by_parts "${local_file}" "${rel_path}"
  else
    upload_normal_file "${local_file}" "${rel_path}"
  fi
}

# ======================
# Target parsing
# ======================
FINAL_REQUESTED=0
CHECKPOINT_STEPS=()

add_checkpoint_step() {
  local step="$1"
  local existing

  for existing in "${CHECKPOINT_STEPS[@]}"; do
    if [ "$existing" = "$step" ]; then
      return 0
    fi
  done

  CHECKPOINT_STEPS+=("$step")
}

collect_all_checkpoint_steps() {
  local checkpoints_dir="${SRC_DIR}/checkpoints"
  local found=0
  local file
  local filename

  if [ ! -d "$checkpoints_dir" ]; then
    echo "ERROR: checkpoints directory not found:"
    echo "  $checkpoints_dir"
    exit 1
  fi

  while IFS= read -r -d '' file; do
    filename="$(basename "$file")"
    if [[ "$filename" =~ ^steps_([0-9]+)_pytorch_model\.pt$ ]]; then
      add_checkpoint_step "${BASH_REMATCH[1]}"
      found=1
    fi
  done < <(find "$checkpoints_dir" -maxdepth 1 -type f -name "steps_*_pytorch_model.pt" -print0 | sort -z -V)

  if [ "$found" -eq 0 ]; then
    echo "ERROR: no checkpoint files found:"
    echo "  ${checkpoints_dir}/steps_*_pytorch_model.pt"
    exit 1
  fi
}

parse_one_target() {
  local target="$1"

  if [[ "$target" == *final* ]]; then
    FINAL_REQUESTED=1
    return 0
  fi

  if [ "$target" = "all" ]; then
    collect_all_checkpoint_steps
    return 0
  fi

  if [[ "$target" =~ ^[0-9]+$ ]]; then
    add_checkpoint_step "$target"
    return 0
  fi

  echo "ERROR: target must be a number, 'all', or contain 'final'."
  echo "Got: $target"
  exit 1
}

parse_targets() {
  local raw_target
  local target
  local step
  local ckpt_file
  local final_dir="${SRC_DIR}/final_model"

  for raw_target in "${TARGETS[@]}"; do
    local split_targets=()
    IFS=',' read -r -a split_targets <<< "$raw_target"

    for target in "${split_targets[@]}"; do
      if [ -z "$target" ]; then
        echo "ERROR: empty target in: $raw_target"
        exit 1
      fi

      parse_one_target "$target"
    done
  done

  if [ "$FINAL_REQUESTED" -eq 1 ] && [ ! -d "$final_dir" ]; then
    echo "ERROR: final_model directory not found:"
    echo "  $final_dir"
    exit 1
  fi

  for step in "${CHECKPOINT_STEPS[@]}"; do
    ckpt_file="${SRC_DIR}/checkpoints/steps_${step}_pytorch_model.pt"
    if [ ! -f "$ckpt_file" ]; then
      echo "ERROR: checkpoint file not found:"
      echo "  $ckpt_file"
      exit 1
    fi
  done
}

# ======================
# Upload root-level files only
# 不上传任何子目录
# 如果根目录有 .pt，也会切片上传
# ======================
upload_root_files() {
  echo
  echo "[1] Uploading root-level single files..."

  local root_file_count=0
  local file
  local filename

  while IFS= read -r -d '' file; do
    filename="$(basename "$file")"
    upload_one_file "$file" "$filename"
    root_file_count=$((root_file_count + 1))
  done < <(find "$SRC_DIR" -maxdepth 1 -type f -print0 | sort -z)

  echo
  echo "Uploaded root-level files: ${root_file_count}"
}

# ======================
# Upload final_model
# 保持 final_model 内部相对路径
# 其中 .pt 自动切片
# ======================
upload_final_model() {
  local final_dir="${SRC_DIR}/final_model"
  local final_file_count=0
  local file
  local rel_path

  echo
  echo "[2] Uploading final_model files..."
  echo "This will NOT upload wandb."
  echo ".pt files inside final_model will be split into ${PART_SIZE} parts."

  while IFS= read -r -d '' file; do
    rel_path="${file#${SRC_DIR}/}"
    upload_one_file "$file" "$rel_path"
    final_file_count=$((final_file_count + 1))
  done < <(find "$final_dir" -type f -print0 | sort -z)

  echo
  echo "Uploaded final_model files: ${final_file_count}"
}

# ======================
# Upload checkpoints
# 不能上传整个 checkpoints 目录
# .pt 自动切片
# ======================
upload_checkpoint_step() {
  local step="$1"
  local ckpt_file="${SRC_DIR}/checkpoints/steps_${step}_pytorch_model.pt"
  local rel_path="checkpoints/steps_${step}_pytorch_model.pt"

  echo
  echo "Uploading checkpoint step ${step}..."

  upload_one_file "$ckpt_file" "$rel_path"
}

upload_checkpoint_steps_parallel() {
  if [ "${#CHECKPOINT_STEPS[@]}" -eq 0 ]; then
    return 0
  fi

  echo
  echo "[3] Uploading checkpoint files in parallel..."
  echo "This will NOT upload the whole checkpoints directory."
  echo "  checkpoint upload jobs: ${CHECKPOINT_UPLOAD_JOBS}"
  echo "  checkpoint steps      : ${CHECKPOINT_STEPS[*]}"

  local ckpt_pids=()
  local ckpt_failed=0
  local step

  for step in "${CHECKPOINT_STEPS[@]}"; do
    (
      upload_checkpoint_step "$step"
    ) &

    ckpt_pids+=("$!")

    if [ "${#ckpt_pids[@]}" -ge "$CHECKPOINT_UPLOAD_JOBS" ]; then
      wait_for_oldest_job ckpt_pids ckpt_failed
    fi
  done

  wait_for_all_jobs ckpt_pids ckpt_failed

  if [ "$ckpt_failed" -ne 0 ]; then
    echo
    echo "ERROR: one or more checkpoint uploads failed."
    return 1
  fi

  echo
  echo "Uploaded checkpoint files: ${#CHECKPOINT_STEPS[@]}"
}

validate_positive_int "CHECKPOINT_UPLOAD_JOBS" "$CHECKPOINT_UPLOAD_JOBS"
validate_positive_int "PART_UPLOAD_JOBS" "$PART_UPLOAD_JOBS"
parse_targets

TARGET_DESC="<root files only>"
if [ "$FINAL_REQUESTED" -eq 1 ] || [ "${#CHECKPOINT_STEPS[@]}" -gt 0 ]; then
  TARGET_DESC=""
  if [ "$FINAL_REQUESTED" -eq 1 ]; then
    TARGET_DESC="final"
  fi
  if [ "${#CHECKPOINT_STEPS[@]}" -gt 0 ]; then
    if [ -n "$TARGET_DESC" ]; then
      TARGET_DESC="${TARGET_DESC} "
    fi
    TARGET_DESC="${TARGET_DESC}steps:${CHECKPOINT_STEPS[*]}"
  fi
fi

echo "========================================"
echo "Repo:                  ${REPO_ID}"
echo "Source dir:            ${SRC_DIR}"
echo "Remote base:           ${REMOTE_BASE}"
echo "Target:                ${TARGET_DESC}"
echo "Part size:             ${PART_SIZE}"
echo "Checkpoint jobs:       ${CHECKPOINT_UPLOAD_JOBS}"
echo "Part jobs per .pt:     ${PART_UPLOAD_JOBS}"
echo "========================================"

upload_root_files

if [ "$FINAL_REQUESTED" -eq 0 ] && [ "${#CHECKPOINT_STEPS[@]}" -eq 0 ]; then
  echo
  echo "[2] No target specified. Will NOT upload checkpoints, final_model, or wandb."
  echo
  echo "Upload finished."
  exit 0
fi

if [ "$FINAL_REQUESTED" -eq 1 ]; then
  upload_final_model
fi

upload_checkpoint_steps_parallel

echo
echo "Upload finished."
