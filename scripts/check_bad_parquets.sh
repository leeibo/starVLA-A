#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-playground/dataset/RoboTwin_Astribot_lerobot}"
JOBS="${JOBS:-16}"
PYTHON_BIN="${PYTHON_BIN:-/data/lmz/miniconda3/envs/starVLA/bin/python}"
RETRY_ON_FAIL="${RETRY_ON_FAIL:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="${REPORT:-bad_parquets_${STAMP}.tsv}"
PROGRESS="${PROGRESS:-1}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/check_bad_parquets.XXXXXX")"
FILE_LIST="${TMP_DIR}/parquets.txt"
RESULT_DIR="${TMP_DIR}/results"
DONE_DIR="${TMP_DIR}/done"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT not found: ${DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/path/to/starVLA/bin/python if needed." >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}" "${DONE_DIR}"
find -L "${DATA_ROOT}" -name '*.parquet' -type f | sort > "${FILE_LIST}"
TOTAL="$(wc -l < "${FILE_LIST}" | tr -d ' ')"

echo "Data root: ${DATA_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Jobs: ${JOBS}"
echo "Retry on fail: ${RETRY_ON_FAIL}"
echo "Total parquet files: ${TOTAL}"
echo "Report: ${REPORT}"

if [[ "${TOTAL}" -eq 0 ]]; then
  : > "${REPORT}"
  echo "No parquet files found."
  exit 0
fi

export PYTHON_BIN RESULT_DIR DONE_DIR RETRY_ON_FAIL
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

run_probe() {
  local path="$1"
  local err_file="$2"
  local shell_err_file="$3"

  set +e
  (
    "${PYTHON_BIN}" -c 'import pyarrow.parquet as pq, sys; pq.read_table(sys.argv[1], use_threads=False)' "${path}" \
      >/dev/null 2>"${err_file}"
  ) 2>"${shell_err_file}"
  local rc="$?"
  set -e
  return "${rc}"
}
export -f run_probe

check_one() {
  local idx="$1"
  local path="$2"
  local err_file
  local shell_err_file
  local result_file="${RESULT_DIR}/${idx}.tsv"
  local done_file="${DONE_DIR}/${idx}.done"
  local rc
  err_file="$(mktemp "${TMPDIR:-/tmp}/parquet_check.XXXXXX.err")"
  shell_err_file="$(mktemp "${TMPDIR:-/tmp}/parquet_check_shell.XXXXXX.err")"

  if run_probe "${path}" "${err_file}" "${shell_err_file}"; then
    rc=0
  else
    rc="$?"
  fi

  if [[ "${rc}" -ne 0 && "${RETRY_ON_FAIL}" == "1" ]]; then
    : > "${err_file}"
    : > "${shell_err_file}"
    if run_probe "${path}" "${err_file}" "${shell_err_file}"; then
      rc=0
    else
      rc="$?"
    fi
  fi

  if [[ "${rc}" -ne 0 ]]; then
    local msg
    msg="$(cat "${err_file}" "${shell_err_file}" | tail -n 1 | tr '\t' ' ')"
    if [[ -z "${msg}" ]]; then
      msg="read_parquet failed"
    fi
    printf '%s\t%s\t%s\n' "${path}" "${rc}" "${msg}" > "${result_file}"
  fi
  : > "${done_file}"
  rm -f "${err_file}" "${shell_err_file}"
}
export -f check_one

show_progress() {
  local done_count=0
  local bad_count=0
  while [[ "${done_count}" -lt "${TOTAL}" ]]; do
    done_count="$(find "${DONE_DIR}" -type f 2>/dev/null | wc -l | tr -d ' ')"
    bad_count="$(find "${RESULT_DIR}" -name '*.tsv' -type f 2>/dev/null | wc -l | tr -d ' ')"
    printf '\rChecked %s/%s parquet files | bad=%s' "${done_count}" "${TOTAL}" "${bad_count}" >&2
    sleep 1
  done
  printf '\n' >&2
}

if [[ "${PROGRESS}" != "0" ]]; then
  show_progress &
  PROGRESS_PID="$!"
else
  PROGRESS_PID=""
fi

nl -ba "${FILE_LIST}" | xargs -r -n 2 -P "${JOBS}" bash -c 'check_one "$1" "$2"' _

if [[ -n "${PROGRESS_PID}" ]]; then
  wait "${PROGRESS_PID}" 2>/dev/null || true
fi

find "${RESULT_DIR}" -name '*.tsv' -type f -print0 \
  | xargs -0 -r cat \
  | sort > "${REPORT}"

BAD_COUNT="$(wc -l < "${REPORT}" | tr -d ' ')"
echo "Bad parquet files: ${BAD_COUNT}"

if [[ "${BAD_COUNT}" -gt 0 ]]; then
  echo
  cat "${REPORT}"
fi
