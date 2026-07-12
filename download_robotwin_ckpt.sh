#!/usr/bin/env bash
set -euo pipefail

# ======================
# Config
# ======================
REPO_ID="conroy1201/robotwin-ckpt"
REPO_TYPE="model"

# 新机器本地保存目录
LOCAL_BASE="/private/zjb/workspace/starVLA-A/results"

# 临时下载目录，避免 ModelScope 把 ckpts/ 前缀直接混进目标目录
TMP_DOWNLOAD_DIR="/tmp/modelscope_robotwin_download"

# 如果 modelscope 不在 PATH，可以取消下面这行注释
# export PATH="$HOME/.local/bin:$PATH"

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

REMOTE_BASE="ckpts/${EXP_NAME}"
FINAL_LOCAL_DIR="${LOCAL_BASE}/${EXP_NAME}"

echo "========================================"
echo "Repo:             ${REPO_ID}"
echo "Remote base:      ${REMOTE_BASE}"
echo "Final local dir:  ${FINAL_LOCAL_DIR}"
echo "Target:           ${TARGET:-<all available under root>}"
echo "========================================"

mkdir -p "${LOCAL_BASE}"
rm -rf "${TMP_DOWNLOAD_DIR}"
mkdir -p "${TMP_DOWNLOAD_DIR}"

# ======================
# Download with retry
# ======================
ms_download_retry() {
  local include_pattern="$1"

  local max_retry=10
  local retry=1

  while [ "$retry" -le "$max_retry" ]; do
    echo
    echo "Download attempt ${retry}/${max_retry}"
    echo "  include: ${include_pattern}"
    echo "  tmp dir: ${TMP_DOWNLOAD_DIR}"

    if modelscope download \
      "${REPO_ID}" \
      --local_dir "${TMP_DOWNLOAD_DIR}" \
      --include "${include_pattern}"; then
      echo "Download success: ${include_pattern}"
      return 0
    fi

    echo "Download failed, retrying in 20 seconds..."
    retry=$((retry + 1))
    sleep 20
  done

  echo "ERROR: download failed after ${max_retry} attempts:"
  echo "  ${include_pattern}"
  return 1
}

# ======================
# Choose download target
# ======================
if [ -z "${TARGET}" ]; then
  echo
  echo "[1/3] Downloading whole experiment directory..."
  ms_download_retry "${REMOTE_BASE}/**"

elif [[ "${TARGET}" == *final* ]]; then
  echo
  echo "[1/3] Downloading root files + final_model..."

  # 根目录单文件
  ms_download_retry "${REMOTE_BASE}/*"

  # final_model
  ms_download_retry "${REMOTE_BASE}/final_model/**"

elif [[ "${TARGET}" =~ ^[0-9]+$ ]]; then
  echo
  echo "[1/3] Downloading root files + checkpoint step ${TARGET}..."

  # 根目录单文件
  ms_download_retry "${REMOTE_BASE}/*"

  # 指定 step 的切片和 sha256
  ms_download_retry "${REMOTE_BASE}/checkpoints/steps_${TARGET}_pytorch_model.pt.parts/**"
  ms_download_retry "${REMOTE_BASE}/checkpoints/steps_${TARGET}_pytorch_model.pt.sha256"

else
  echo "ERROR: second argument must be empty, a number, or contain 'final'."
  echo "Got: ${TARGET}"
  exit 1
fi

# ======================
# Move downloaded directory to final location
# ======================
echo
echo "[2/3] Moving files to final local dir..."

DOWNLOADED_EXP_DIR="${TMP_DOWNLOAD_DIR}/${REMOTE_BASE}"

if [ ! -d "${DOWNLOADED_EXP_DIR}" ]; then
  echo "ERROR: downloaded experiment directory not found:"
  echo "  ${DOWNLOADED_EXP_DIR}"
  echo
  echo "Debug: files under tmp dir:"
  find "${TMP_DOWNLOAD_DIR}" -maxdepth 5 -print
  exit 1
fi

mkdir -p "${FINAL_LOCAL_DIR}"

# 拷贝内容，保持实验目录内部相对路径
rsync -av "${DOWNLOADED_EXP_DIR}/" "${FINAL_LOCAL_DIR}/"

echo
echo "Files are now under:"
echo "  ${FINAL_LOCAL_DIR}"

# ======================
# Restore .pt from .parts and verify sha256
# ======================
echo
echo "[3/3] Restoring .pt files from .parts directories..."

PART_DIRS="$(find "${FINAL_LOCAL_DIR}" -type d -name "*.pt.parts" || true)"

if [ -z "${PART_DIRS}" ]; then
  echo "No .pt.parts directory found. Nothing to restore."
  echo
  echo "Download finished."
  exit 0
fi

while IFS= read -r PART_DIR; do
  [ -z "${PART_DIR}" ] && continue

  PT_FILE="${PART_DIR%.parts}"
  SHA_FILE="${PT_FILE}.sha256"

  echo
  echo "Restoring:"
  echo "  parts: ${PART_DIR}"
  echo "  pt   : ${PT_FILE}"

  if [ -f "${PT_FILE}" ]; then
    echo "Existing .pt found, removing:"
    echo "  ${PT_FILE}"
    rm -f "${PT_FILE}"
  fi

  # 按 part_00000, part_00001 ... 顺序合并
  find "${PART_DIR}" -maxdepth 1 -type f -name "part_*" | sort | xargs cat > "${PT_FILE}"

  if [ -f "${SHA_FILE}" ]; then
    echo "Checking sha256..."

    expected_hash="$(awk '{print $1}' "${SHA_FILE}")"
    actual_hash="$(sha256sum "${PT_FILE}" | awk '{print $1}')"

    echo "  expected: ${expected_hash}"
    echo "  actual  : ${actual_hash}"

    if [ "${expected_hash}" = "${actual_hash}" ]; then
      echo "  sha256 OK"
    else
      echo "  ERROR: sha256 mismatch"
      exit 1
    fi
  else
    echo "WARNING: sha256 file not found:"
    echo "  ${SHA_FILE}"
  fi

  echo "Restored:"
  echo "  ${PT_FILE}"

done <<< "${PART_DIRS}"

echo
echo "Download, merge, and verification finished."
echo "Final directory:"
echo "  ${FINAL_LOCAL_DIR}"