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

# 如果 modelscope 不在 PATH，自动补一下
export PATH="/HOME/hlkj_zql/hlkj_zql_8/.local/bin:$PATH"

# ======================
# Args
# ======================
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage:"
  echo "  bash $0 <folder_name> [step_number|final]"
  echo
  echo "Examples:"
  echo "  bash $0 fast_subtask_action_12_wos"
  echo "  bash $0 fast_subtask_action_12_wos 5000"
  echo "  bash $0 fast_subtask_action_12_wos final"
  exit 1
fi

EXP_NAME="$1"
TARGET="${2:-}"

SRC_DIR="${BASE_DIR}/${EXP_NAME}"
REMOTE_BASE="ckpts/${EXP_NAME}"

if [ ! -d "$SRC_DIR" ]; then
  echo "ERROR: source directory not found:"
  echo "  $SRC_DIR"
  exit 1
fi

echo "========================================"
echo "Repo:        ${REPO_ID}"
echo "Source dir:  ${SRC_DIR}"
echo "Remote base: ${REMOTE_BASE}"
echo "Target:      ${TARGET:-<root files only>}"
echo "Part size:   ${PART_SIZE}"
echo "========================================"

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
  echo "Uploading parts..."
  local part_count=0

  while IFS= read -r -d '' part_file; do
    local part_name
    part_name="$(basename "$part_file")"

    ms_upload_retry \
      "${part_file}" \
      "${remote_parts_dir}/${part_name}"

    part_count=$((part_count + 1))
  done < <(find "${tmp_dir}" -maxdepth 1 -type f -name "part_*" -print0 | sort -z)

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
# 1. Always upload root-level files only
#    不上传任何子目录
#    如果根目录有 .pt，也会切片上传
# ======================
echo
echo "[1/3] Uploading root-level single files..."

ROOT_FILE_COUNT=0

while IFS= read -r -d '' file; do
  filename="$(basename "$file")"
  upload_one_file "$file" "$filename"
  ROOT_FILE_COUNT=$((ROOT_FILE_COUNT + 1))
done < <(find "$SRC_DIR" -maxdepth 1 -type f -print0)

echo
echo "Uploaded root-level files: ${ROOT_FILE_COUNT}"

# ======================
# 2. Optional target
# ======================
if [ -z "$TARGET" ]; then
  echo
  echo "[2/3] No target specified. Will NOT upload checkpoints, final_model, or wandb."
  echo
  echo "Upload finished."
  exit 0
fi

# ======================
# 3A. If target is final, upload final_model only
#     保持 final_model 内部相对路径
#     其中 .pt 自动切片
# ======================
if [[ "$TARGET" == *final* ]]; then
  FINAL_DIR="${SRC_DIR}/final_model"

  if [ ! -d "$FINAL_DIR" ]; then
    echo "ERROR: final_model directory not found:"
    echo "  $FINAL_DIR"
    exit 1
  fi

  echo
  echo "[2/3] Uploading final_model files..."
  echo "This will NOT upload checkpoints or wandb."
  echo ".pt files inside final_model will be split into ${PART_SIZE} parts."

  FINAL_FILE_COUNT=0

  while IFS= read -r -d '' file; do
    rel_path="${file#${SRC_DIR}/}"
    upload_one_file "$file" "$rel_path"
    FINAL_FILE_COUNT=$((FINAL_FILE_COUNT + 1))
  done < <(find "$FINAL_DIR" -type f -print0)

  echo
  echo "Uploaded final_model files: ${FINAL_FILE_COUNT}"
  echo
  echo "Upload finished."
  exit 0
fi

# ======================
# 3B. If target is number, upload exactly one checkpoint .pt
#     不能上传整个 checkpoints 目录
#     该 .pt 自动切片
# ======================
if [[ "$TARGET" =~ ^[0-9]+$ ]]; then
  CKPT_FILE="${SRC_DIR}/checkpoints/steps_${TARGET}_pytorch_model.pt"

  if [ ! -f "$CKPT_FILE" ]; then
    echo "ERROR: checkpoint file not found:"
    echo "  $CKPT_FILE"
    exit 1
  fi

  echo
  echo "[2/3] Uploading exactly one checkpoint file by parts..."
  echo "This will NOT upload the whole checkpoints directory."

  upload_one_file \
    "$CKPT_FILE" \
    "checkpoints/steps_${TARGET}_pytorch_model.pt"

  echo
  echo "Upload finished."
  exit 0
fi

echo "ERROR: second argument must be empty, a number, or contain 'final'."
echo "Got: $TARGET"
exit 1