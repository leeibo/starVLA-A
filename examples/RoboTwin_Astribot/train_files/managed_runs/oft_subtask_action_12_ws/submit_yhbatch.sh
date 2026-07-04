#!/usr/bin/env bash
#SBATCH --job-name=oft_subtask_action_12_ws
#SBATCH -p a800x
#SBATCH -c 8
#SBATCH --gres=gpu:1
#
# Submit:
#   yhbatch examples/RoboTwin_Astribot/train_files/managed_runs/oft_subtask_action_12_ws/submit_yhbatch.sh

set -euo pipefail

REPO_ROOT="${STARGVLA_REPO_ROOT:-/XYAIFS00/HDD_POOL/hlkj_zql/hlkj_zql_8/code/starVLA}"
RUN_NAME="oft_subtask_action_12_ws"
SCRIPT_DIR="${REPO_ROOT}/examples/RoboTwin_Astribot/train_files/managed_runs/${RUN_NAME}"
RUN_TRAIN="${SCRIPT_DIR}/run_train.sh"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
JOB_ID="${YH_JOB_ID:-${SLURM_JOB_ID:-${PBS_JOBID:-manual}}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="${LOG_DIR}/${RUN_NAME}-${JOB_ID}-${TIMESTAMP}.out"
STDERR_LOG="${LOG_DIR}/${RUN_NAME}-${JOB_ID}-${TIMESTAMP}.err"

exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)

cd "${REPO_ROOT}"

DEFAULT_ACCELERATE_BIN="/HOME/hlkj_zql/hlkj_zql_8/HDD_POOL/conda_envs/starVLA/bin/accelerate"
if [[ -z "${ACCELERATE_BIN:-}" && -x "${DEFAULT_ACCELERATE_BIN}" ]]; then
  export ACCELERATE_BIN="${DEFAULT_ACCELERATE_BIN}"
fi

if [[ -z "${NUM_PROCESSES:-}" ]]; then
  if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ && "${SLURM_GPUS_ON_NODE}" -gt 0 ]]; then
    NUM_PROCESSES="${SLURM_GPUS_ON_NODE}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    IFS=',' read -r -a CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_PROCESSES="${#CUDA_DEVICE_LIST[@]}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t NVIDIA_SMI_GPUS < <(nvidia-smi -L 2>/dev/null || true)
    if [[ "${#NVIDIA_SMI_GPUS[@]}" -gt 0 ]]; then
      NUM_PROCESSES="${#NVIDIA_SMI_GPUS[@]}"
    else
      NUM_PROCESSES=1
    fi
  else
    NUM_PROCESSES=1
  fi
fi

export NUM_PROCESSES
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
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

echo "Run name: ${RUN_NAME}"
echo "Job id: ${JOB_ID}"
echo "Repo root: ${REPO_ROOT}"
echo "Run script: ${RUN_TRAIN}"
echo "Stdout log: ${STDOUT_LOG}"
echo "Stderr log: ${STDERR_LOG}"
echo "ACCELERATE_BIN: ${ACCELERATE_BIN:-<run_train fallback>}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "WANDB_MODE: ${WANDB_MODE}"

bash "${RUN_TRAIN}"
