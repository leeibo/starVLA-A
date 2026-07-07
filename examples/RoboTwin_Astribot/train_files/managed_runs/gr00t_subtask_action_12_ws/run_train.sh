#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_YAML="${SCRIPT_DIR}/config.yaml"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-starVLA}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-28501}}"

if [[ -z "${CUDA_HOME:-}" && -d /APP/u22/ai_x86/CUDA/12.4 ]]; then
  export CUDA_HOME=/APP/u22/ai_x86/CUDA/12.4
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  if [[ -d /sys/class/net/bond0 ]]; then
    NCCL_SOCKET_IFNAME="bond0"
  elif [[ -d /sys/class/net/ib0 ]]; then
    NCCL_SOCKET_IFNAME="ib0"
  else
    NCCL_SOCKET_IFNAME="eth0"
  fi
fi

export NCCL_SOCKET_IFNAME
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"


EXPLICIT_WANDB_MODE="${WANDB_MODE:-}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ -n "${EXPLICIT_WANDB_MODE}" ]]; then
  export WANDB_MODE="${EXPLICIT_WANDB_MODE}"
else
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

if [[ -n "${ACCELERATE_BIN:-}" ]]; then
  ACCELERATE_CMD=("${ACCELERATE_BIN}")
elif command -v accelerate >/dev/null 2>&1; then
  ACCELERATE_CMD=("accelerate")
else
  CONDA_BIN="${CONDA_EXE:-${HOME}/HDD_POOL/miniconda3/bin/conda}"
  ACCELERATE_CMD=("${CONDA_BIN}" "run" "--no-capture-output" "-n" "${CONDA_ENV_NAME}" "accelerate")
fi

"${ACCELERATE_CMD[@]}" launch \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla_cotrain.py \
  --config_yaml "${CONFIG_YAML}"
