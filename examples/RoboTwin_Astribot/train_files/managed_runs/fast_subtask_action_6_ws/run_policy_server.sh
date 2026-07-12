#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="fast_subtask_action_6_ws"
MANAGED_RUNS_ROOT="${MANAGED_RUNS_ROOT:-/private/zjb/workspace/starVLA-A/examples/RoboTwin_Astribot/train_files/managed_runs}"
if [[ -n "${STARGVLA_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${STARGVLA_REPO_ROOT}"
else
  REPO_ROOT="$(cd "${MANAGED_RUNS_ROOT}/../../../.." && pwd -P)"
fi
RUN_OUTPUT="${RUN_OUTPUT:-${REPO_ROOT}/results/Checkpoints/${RUN_NAME}}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export STARGVLA_BASE_VLM="${STARGVLA_BASE_VLM:-${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action}"
if [[ ! -d "${STARGVLA_BASE_VLM}" ]]; then
  cat >&2 <<EOF
Missing base VLM directory:
  ${STARGVLA_BASE_VLM}

This checkpoint was trained with Qwen3-VL-2B-Instruct-Action. Prepare it at the
path above, or set STARGVLA_BASE_VLM to an existing local action-token model.
To generate it from the plain Qwen checkpoint:

  python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py \\
    --model-id Qwen/Qwen3-VL-2B-Instruct \\
    --save-dir "${STARGVLA_BASE_VLM}" \\
    --tokens-file starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt \\
    --init-strategy normal \\
    --device auto
EOF
  exit 1
fi

STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
if [[ ! -x "${STARVLA_PYTHON}" ]]; then
  STARVLA_PYTHON="python"
fi

resolve_checkpoint() {
  if [[ -n "${POLICY_CKPT_PATH:-}" ]]; then
    printf '%s\n' "${POLICY_CKPT_PATH}"
    return
  fi

  local latest_checkpoint=""
  if [[ -d "${RUN_OUTPUT}/checkpoints" ]]; then
    latest_checkpoint="$(
      find "${RUN_OUTPUT}/checkpoints" -maxdepth 1 -type f \( -name 'steps_*_pytorch_model.pt' -o -name 'steps_*.safetensors' \) \
        | sort -V \
        | tail -n 1
    )"
  fi
  if [[ -n "${latest_checkpoint}" ]]; then
    printf '%s\n' "${latest_checkpoint}"
    return
  fi

  if [[ -f "${RUN_OUTPUT}/final_model/pytorch_model.pt" ]]; then
    printf '%s\n' "${RUN_OUTPUT}/final_model/pytorch_model.pt"
    return
  fi
  if [[ -f "${RUN_OUTPUT}/final_model/model.safetensors" ]]; then
    printf '%s\n' "${RUN_OUTPUT}/final_model/model.safetensors"
    return
  fi

  echo "No checkpoint found under ${RUN_OUTPUT}. Set POLICY_CKPT_PATH explicitly." >&2
  exit 1
}

CKPT_PATH="$(resolve_checkpoint)"
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint does not exist: ${CKPT_PATH}" >&2
  exit 1
fi

PORT="${POLICY_PORT:-7980}"
GPU_ID="${POLICY_GPU_ID:-0}"
USE_BF16="${USE_BF16:-1}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-1800}"

CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT_PATH}"
  --port "${PORT}"
  --idle_timeout "${IDLE_TIMEOUT}"
)

if [[ "${USE_BF16}" != "0" ]]; then
  CMD+=(--use_bf16)
fi

echo "Run name: ${RUN_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Port: ${PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
echo "Python: ${STARVLA_PYTHON}"
echo "USE_BF16: ${USE_BF16}"
echo "IDLE_TIMEOUT: ${IDLE_TIMEOUT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}" "${CMD[@]}"
