#!/usr/bin/env bash
# Multi-node launcher for gr00tdual_subtask_action_12_ws.
# Designed for 百舸 PyTorch 训练任务：平台自动注入以下环境变量，
# 在每个节点上各执行一次本脚本：
#   WORLD_SIZE       节点总数（含 Master），对应 accelerate --num_machines
#   RANK             节点编号，Master=0, Worker-0=1, ...，对应 accelerate --machine_rank
#   NPROC_PER_NODE   每节点 GPU 数
#   MASTER_ADDR      Master 节点地址
#   MASTER_PORT      通信端口（默认 23456）
#
# 本脚本把这些变量翻译成 accelerate 多机参数，然后委托给 run_train.sh。
# 训练语义全部来自 config.yaml，本脚本不传任何 --framework/--datasets/--trainer 覆盖。
#
# 手动测试时可显式覆盖 NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT/NPROC_PER_NODE。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${STARGVLA_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
RUN_NAME="$(basename "${SCRIPT_DIR}")"
RUN_TRAIN="${SCRIPT_DIR}/run_train.sh"

cd "${REPO_ROOT}"

# -------------------- conda --------------------
source /data/lmz/miniconda3/etc/profile.d/conda.sh
conda activate /data/lmz/miniconda3/envs/starVLA
export ACCELERATE_BIN="/data/lmz/miniconda3/envs/starVLA/bin/accelerate"

# -------------------- CUDA / HF offline --------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/data/lmz/hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

# -------------------- distributed env (platform-injected) --------------------
# 优先用平台注入的 WORLD_SIZE / RANK；允许 NNODES / NODE_RANK 覆盖以便手动测试。
NUM_MACHINES="${WORLD_SIZE:-${NNODES:-2}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MACHINE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is not set by platform; set MASTER_ADDR to master node ip}"
MASTER_PORT="${MASTER_PORT:-23456}"

TOTAL_GPUS=$((NUM_MACHINES * NPROC_PER_NODE))

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"

# -------------------- W&B (respect explicit WANDB_MODE + repo .env) --------------------
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

# -------------------- translate to accelerate multi-node args --------------------
export NUM_MACHINES
export MACHINE_RANK
export MAIN_PROCESS_IP="${MASTER_ADDR}"
export MAIN_PROCESS_PORT="${MASTER_PORT}"
export NUM_PROCESSES="${TOTAL_GPUS}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# -------------------- logs --------------------
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="${LOG_DIR}/${RUN_NAME}-multinode-rank${MACHINE_RANK}-${TIMESTAMP}.out"
STDERR_LOG="${LOG_DIR}/${RUN_NAME}-multinode-rank${MACHINE_RANK}-${TIMESTAMP}.err"

exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)

echo "Run name: ${RUN_NAME}"
echo "Repo root: ${REPO_ROOT}"
echo "Run script: ${RUN_TRAIN}"
echo "WORLD_SIZE/NNODES=${NUM_MACHINES}  RANK/NODE_RANK=${MACHINE_RANK}  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "TOTAL_GPUS=${TOTAL_GPUS}  MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "Stdout log: ${STDOUT_LOG}"
echo "Stderr log: ${STDERR_LOG}"

bash "${RUN_TRAIN}"
