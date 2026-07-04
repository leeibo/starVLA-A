#!/usr/bin/env bash
set -u -o pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-starVLA}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/lmz/miniconda3/envs/starVLA/bin/accelerate}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/memory_tests/oft_cotrain_2gpu}"
DATA_MIX="${DATA_MIX:-robotwin_astribot_long}"
REQUIRE_FULL_HISTORY="${REQUIRE_FULL_HISTORY:-false}"
MAX_CAP="${MAX_CAP:-16}"
TRIAL_TIMEOUT="${TRIAL_TIMEOUT:-1200}"
GPU_IDS=(${GPU_IDS:-0 1})
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-doc/memory_tests/logs/oft_cotrain_2gpu_${STAMP}}"
REPORT_PATH="${REPORT_PATH:-doc/memory_tests/oft_cotrain_2gpu_batch_size_${STAMP}.md}"

if [[ -n "${CASES_OVERRIDE:-}" ]]; then
  read -r -a CASES <<< "${CASES_OVERRIDE}"
else
  CASES=(
    "oft_subtask_no_0_ws"
    "oft_subtask_action_6_ws"
    "oft_subtask_action_12_ws"
    "oft_instruction_action_12_ws"
    "oft_subtask_action_12_wos"
    "oft_subtask_motion_6_ws"
    "oft_subtask_motion_12_ws"
    "oft_subtask_subtask_6_ws"
    "oft_subtask_subtask_12_ws"
  )
fi

mkdir -p "${LOG_DIR}" "${RUN_ROOT_DIR}"

cat > "${REPORT_PATH}" <<EOF
# OFT Cotrain 2-GPU Batch Size Probe

- Time UTC: ${STAMP}
- GPUs: ${GPU_IDS[*]}
- Max cap: ${MAX_CAP}
- Trial steps: 5
- Eval: enabled at step 5
- Data mix override: \`${DATA_MIX}\`
- Require full history: \`${REQUIRE_FULL_HISTORY}\`
- Run root: \`${RUN_ROOT_DIR}\`

## Trial Log

| Time UTC | GPU | Run | Batch Size | Status | Exit | Peak MiB | Log |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- |
EOF

status_from_log() {
  local exit_code="$1"
  local log_file="$2"
  if grep -Eiq "out of memory|cuda.*oom|CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "${log_file}"; then
    printf 'OOM'
  elif grep -Fq "Unable to sample a valid item after 200 attempts" "${log_file}"; then
    printf 'DATA_FAIL'
  elif [[ "${exit_code}" -eq 124 ]]; then
    printf 'TIMEOUT'
  elif [[ "${exit_code}" -eq 0 ]]; then
    printf 'OK'
  else
    printf 'FAIL'
  fi
}

monitor_gpu() {
  local gpu_id="$1"
  local stop_file="$2"
  local mem_log="$3"
  while [[ ! -f "${stop_file}" ]]; do
    nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits >> "${mem_log}" 2>/dev/null || true
    sleep 1
  done
}

run_trial() {
  local gpu_id="$1"
  local run_name="$2"
  local batch_size="$3"
  local trial_id="${run_name}_gpu${gpu_id}_bs${batch_size}_${STAMP}"
  local run_dir="examples/RoboTwin_Astribot/train_files/managed_runs/${run_name}"
  local log_file="${LOG_DIR}/${trial_id}.log"
  local mem_log="${LOG_DIR}/${trial_id}.mem"
  local stop_file="${LOG_DIR}/${trial_id}.stop"
  local time_utc
  local exit_code
  local status
  local peak_mib

  rm -f "${stop_file}"
  : > "${mem_log}"
  time_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  monitor_gpu "${gpu_id}" "${stop_file}" "${mem_log}" &
  local monitor_pid=$!

  (
    cd "${REPO_ROOT}" || exit 1
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    NUM_PROCESSES=1 \
    MAIN_PROCESS_PORT="$((29600 + gpu_id))" \
    ACCELERATE_BIN="${ACCELERATE_BIN}" \
    WANDB_MODE=disabled \
    WANDB_DISABLED=true \
    TOKENIZERS_PARALLELISM=false \
    NO_ALBUMENTATIONS_UPDATE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    timeout "${TRIAL_TIMEOUT}" \
    bash "${run_dir}/run_train.sh" \
      --run_id "${trial_id}" \
      --run_root_dir "${RUN_ROOT_DIR}" \
      --datasets.vla_data.data_mix "${DATA_MIX}" \
      --datasets.vla_data.history.require_full_frames "${REQUIRE_FULL_HISTORY}" \
      --datasets.vla_data.per_device_batch_size "${batch_size}" \
      --datasets.vla_data.num_workers 0 \
      --datasets.vla_data.pin_memory false \
      --datasets.vlm_data.per_device_batch_size "${batch_size}" \
      --datasets.vlm_data.num_workers 0 \
      --datasets.vlm_data.pin_memory false \
      --trainer.max_train_steps 5 \
      --trainer.eval_interval 5 \
      --trainer.save_interval 999999 \
      --trainer.skip_final_save true \
      --trainer.logging_frequency 1 \
      --trainer.num_warmup_steps 0
  ) > "${log_file}" 2>&1
  exit_code=$?

  touch "${stop_file}"
  wait "${monitor_pid}" 2>/dev/null || true

  peak_mib="$(sort -nr "${mem_log}" 2>/dev/null | head -n 1)"
  if [[ -z "${peak_mib}" ]]; then
    peak_mib="0"
  fi
  status="$(status_from_log "${exit_code}" "${log_file}")"

  printf '| %s | %s | `%s` | %s | %s | %s | %s | `%s` |\n' \
    "${time_utc}" "${gpu_id}" "${run_name}" "${batch_size}" "${status}" "${exit_code}" "${peak_mib}" "${log_file}" \
    >> "${REPORT_PATH}"
  printf '%s gpu=%s run=%s bs=%s status=%s exit=%s peak_mib=%s log=%s\n' \
    "${time_utc}" "${gpu_id}" "${run_name}" "${batch_size}" "${status}" "${exit_code}" "${peak_mib}" "${log_file}"

  [[ "${status}" == "OK" ]]
}

find_max_for_run() {
  local gpu_id="$1"
  local run_name="$2"
  local low=0
  local high="${MAX_CAP}"
  local first_fail=""

  while [[ "${low}" -lt "${high}" ]]; do
    local mid=$(((low + high + 1) / 2))
    if run_trial "${gpu_id}" "${run_name}" "${mid}"; then
      low="${mid}"
    else
      first_fail="${mid}"
      high=$((mid - 1))
    fi
  done

  printf '%s,%s,%s,%s\n' "${run_name}" "${low}" "${first_fail:-none}" "${gpu_id}" >> "${LOG_DIR}/summary.csv"
}

run_worker() {
  local gpu_id="$1"
  shift
  local case
  for case in "$@"; do
    find_max_for_run "${gpu_id}" "${case}"
  done
}

: > "${LOG_DIR}/summary.csv"

GPU0_CASES=()
GPU1_CASES=()
for i in "${!CASES[@]}"; do
  if (( i % 2 == 0 )); then
    GPU0_CASES+=("${CASES[$i]}")
  else
    GPU1_CASES+=("${CASES[$i]}")
  fi
done

run_worker "${GPU_IDS[0]}" "${GPU0_CASES[@]}" &
PID0=$!
run_worker "${GPU_IDS[1]}" "${GPU1_CASES[@]}" &
PID1=$!

wait "${PID0}"
STATUS0=$?
wait "${PID1}"
STATUS1=$?

{
  printf '\n## Summary\n\n'
  printf '| Run | Max OK Batch Size | First Failing Batch Size | GPU Worker |\n'
  printf '| --- | ---: | --- | ---: |\n'
  sort "${LOG_DIR}/summary.csv" | awk -F, '{printf "| `%s` | %s | %s | %s |\n", $1, $2, $3, $4}'
} >> "${REPORT_PATH}"

printf 'Report: %s\n' "${REPORT_PATH}"
exit $((STATUS0 || STATUS1))
