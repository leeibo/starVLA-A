#!/usr/bin/env bash
set -u -o pipefail

GPU_ID="${GPU_ID:-1}"
MAX_CAP="${MAX_CAP:-64}"
TRIAL_TIMEOUT="${TRIAL_TIMEOUT:-900}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-starVLA}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/memory_tests}"
DATA_MIX="${DATA_MIX:-robotwin_astribot_long}"
REQUIRE_FULL_HISTORY="${REQUIRE_FULL_HISTORY:-true}"
CASES="${CASES:-oft_subtask_action_6_ws:6 oft_subtask_action_12_ws:12}"
DOC_PATH="${DOC_PATH:-doc/memory_tests/oft_subtask_ws_gpu1_batch_size.md}"
LOG_DIR="${LOG_DIR:-doc/memory_tests/logs}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "${LOG_DIR}"

if ! grep -q "^## Test Nodes" "${DOC_PATH}"; then
  {
    printf '\n## Test Nodes\n\n'
    printf '| Time UTC | Run | History | Batch Size | Status | Exit | Peak GPU1 MiB | Log |\n'
    printf '| --- | --- | ---: | ---: | --- | ---: | ---: | --- |\n'
  } >> "${DOC_PATH}"
fi

append_node() {
  local time_utc="$1"
  local run_name="$2"
  local history="$3"
  local batch_size="$4"
  local status="$5"
  local exit_code="$6"
  local peak_mib="$7"
  local log_file="$8"
  printf '| %s | `%s` | %s | %s | %s | %s | %s | `%s` |\n' \
    "${time_utc}" "${run_name}" "${history}" "${batch_size}" "${status}" "${exit_code}" "${peak_mib}" "${log_file}" \
    >> "${DOC_PATH}"
}

monitor_gpu() {
  local stop_file="$1"
  local mem_log="$2"
  while [[ ! -f "${stop_file}" ]]; do
    nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits >> "${mem_log}" 2>/dev/null || true
    sleep 1
  done
}

run_trial() {
  local run_name="$1"
  local history="$2"
  local batch_size="$3"
  local config_path="examples/RoboTwin_Astribot/train_files/managed_runs/${run_name}/config.yaml"
  local trial_id="${run_name}_fullhist_gpu${GPU_ID}_bs${batch_size}_${STAMP}"
  local log_file="${LOG_DIR}/${trial_id}.log"
  local mem_log="${LOG_DIR}/${trial_id}.mem"
  local stop_file="${LOG_DIR}/${trial_id}.stop"
  local time_utc
  local exit_code
  local status
  local peak_mib

  time_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  : > "${mem_log}"
  monitor_gpu "${stop_file}" "${mem_log}" &
  local monitor_pid=$!

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  WANDB_MODE=disabled \
  WANDB_DISABLED=true \
  TOKENIZERS_PARALLELISM=false \
  NO_ALBUMENTATIONS_UPDATE=1 \
  NCCL_DEBUG=WARN \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout "${TRIAL_TIMEOUT}" \
  conda run --no-capture-output -n "${CONDA_ENV_NAME}" accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes 1 \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_path}" \
    --run_id "${trial_id}" \
    --run_root_dir "${RUN_ROOT_DIR}" \
    --datasets.vla_data.data_mix "${DATA_MIX}" \
    --datasets.vla_data.history.require_full_frames "${REQUIRE_FULL_HISTORY}" \
    --datasets.vla_data.per_device_batch_size "${batch_size}" \
    --datasets.vla_data.num_workers 0 \
    --datasets.vla_data.pin_memory false \
    --trainer.max_train_steps 1 \
    --trainer.eval_interval 1 \
    --trainer.save_interval 999999 \
    --trainer.skip_final_save true \
    --trainer.logging_frequency 1 \
    --trainer.num_warmup_steps 0 \
    > "${log_file}" 2>&1
  exit_code=$?

  touch "${stop_file}"
  wait "${monitor_pid}" 2>/dev/null || true

  peak_mib="$(sort -nr "${mem_log}" 2>/dev/null | head -n 1)"
  if [[ -z "${peak_mib}" ]]; then
    peak_mib="0"
  fi

  if grep -Eiq "out of memory|cuda.*oom|CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "${log_file}"; then
    status="OOM"
  elif [[ "${exit_code}" -eq 124 ]]; then
    status="TIMEOUT"
  elif [[ "${exit_code}" -eq 0 ]]; then
    status="OK"
  else
    status="FAIL"
  fi

  append_node "${time_utc}" "${run_name}" "${history}" "${batch_size}" "${status}" "${exit_code}" "${peak_mib}" "${log_file}"
  printf '%s %s bs=%s status=%s exit=%s peak_mib=%s log=%s\n' \
    "${time_utc}" "${run_name}" "${batch_size}" "${status}" "${exit_code}" "${peak_mib}" "${log_file}"

  [[ "${status}" == "OK" ]]
}

find_max_for_case() {
  local run_name="$1"
  local history="$2"
  local low_ok=0
  local high_fail=0
  local bs=1

  while [[ "${bs}" -le "${MAX_CAP}" ]]; do
    if run_trial "${run_name}" "${history}" "${bs}"; then
      low_ok="${bs}"
      bs=$((bs * 2))
    else
      high_fail="${bs}"
      break
    fi
  done

  if [[ "${high_fail}" -eq 0 ]]; then
    printf '| %s | `%s` | %s | >=%s | reached cap without OOM |\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_name}" "${history}" "${low_ok}" >> "${DOC_PATH}"
    return 0
  fi

  local left=$((low_ok + 1))
  local right=$((high_fail - 1))
  while [[ "${left}" -le "${right}" ]]; do
    local mid=$(((left + right) / 2))
    if run_trial "${run_name}" "${history}" "${mid}"; then
      low_ok="${mid}"
      left=$((mid + 1))
    else
      high_fail="${mid}"
      right=$((mid - 1))
    fi
  done

  printf '| %s | `%s` | %s | %s | first failing batch size: %s |\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_name}" "${history}" "${low_ok}" "${high_fail}" >> "${DOC_PATH}"
}

{
  printf '\n## Summary Updates\n\n'
  printf '| Time UTC | Run | History | Max OK Batch Size | Notes |\n'
  printf '| --- | --- | ---: | ---: | --- |\n'
} >> "${DOC_PATH}"

for case_entry in ${CASES}; do
  run_name="${case_entry%%:*}"
  history="${case_entry##*:}"
  find_max_for_case "${run_name}" "${history}"
done
