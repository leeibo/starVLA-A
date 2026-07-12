#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ID="conroy1201/robotwin-ckpt"
REMOTE_ROOT="starvla_ckpt"
RESULTS_DIR="/private/zjb/workspace/starVLA-A/results"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-modelscope}"
MAX_RETRY="${MAX_RETRY:-3}"
RETRY_SLEEP="${RETRY_SLEEP:-20}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: bash download_starvla_ckpt.sh

Downloads every missing direct child of the ModelScope directory
conroy1201/robotwin-ckpt/starvla_ckpt into starVLA-A/results.

Environment variables:
  MODELSCOPE_PYTHON  Python that can import modelscope_hub
  MODELSCOPE_BIN     ModelScope CLI command (default: modelscope)
  MAX_RETRY          Retries per remote child directory (default: 3)
  RETRY_SLEEP        Seconds between retries (default: 20)
  DRY_RUN=1          List the planned downloads without downloading
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    usage >&2
    exit 2
fi
if ! [[ "${MAX_RETRY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_RETRY must be a positive integer: ${MAX_RETRY}" >&2
    exit 2
fi
if ! [[ "${RETRY_SLEEP}" =~ ^[0-9]+$ ]]; then
    echo "RETRY_SLEEP must be a non-negative integer: ${RETRY_SLEEP}" >&2
    exit 2
fi
if ! command -v "${MODELSCOPE_BIN}" >/dev/null 2>&1; then
    echo "ModelScope CLI was not found: ${MODELSCOPE_BIN}" >&2
    exit 1
fi

can_import_modelscope_hub() {
    local python_bin="$1"
    [[ -n "${python_bin}" && -x "${python_bin}" ]] || return 1
    "${python_bin}" -c 'import modelscope_hub' >/dev/null 2>&1
}

resolve_modelscope_python() {
    local python_bin
    if [[ -n "${MODELSCOPE_PYTHON:-}" ]]; then
        if can_import_modelscope_hub "${MODELSCOPE_PYTHON}"; then
            printf '%s\n' "${MODELSCOPE_PYTHON}"
            return 0
        fi
        echo "MODELSCOPE_PYTHON cannot import modelscope_hub: ${MODELSCOPE_PYTHON}" >&2
        return 1
    fi

    for python_bin in "$(command -v python 2>/dev/null || true)" "$(command -v python3 2>/dev/null || true)"; do
        if can_import_modelscope_hub "${python_bin}"; then
            printf '%s\n' "${python_bin}"
            return 0
        fi
    done

    local modelscope_path shebang_python
    modelscope_path="$(command -v "${MODELSCOPE_BIN}")"
    shebang_python="$(sed -n '1s/^#!//p' "${modelscope_path}")"
    if can_import_modelscope_hub "${shebang_python}"; then
        printf '%s\n' "${shebang_python}"
        return 0
    fi

    echo "Cannot find a Python interpreter that can import modelscope_hub." >&2
    echo "Activate the ai environment or set MODELSCOPE_PYTHON explicitly." >&2
    return 1
}

MODELSCOPE_PYTHON="$(resolve_modelscope_python)"

list_remote_children() {
    "${MODELSCOPE_PYTHON}" - "${REPO_ID}" "${REMOTE_ROOT}" <<'PY'
import sys

from modelscope_hub import HubApi

repo_id, remote_root = sys.argv[1:]
prefix = remote_root.rstrip("/") + "/"
children = set()

for item in HubApi().list_repo_files(repo_id, "model", recursive=True):
    path = getattr(item, "path", "")
    item_type = getattr(item, "type", "")
    if item_type != "tree" or not path.startswith(prefix):
        continue
    relative = path[len(prefix):]
    if relative and "/" not in relative:
        children.add(relative)

for child in sorted(children):
    print(child)
PY
}

REMOTE_CHILDREN_TEXT="$(list_remote_children)"
if [[ -z "${REMOTE_CHILDREN_TEXT}" ]]; then
    echo "No child directories found under ${REPO_ID}/${REMOTE_ROOT}." >&2
    exit 1
fi
mapfile -t REMOTE_CHILDREN <<< "${REMOTE_CHILDREN_TEXT}"

mkdir -p "${RESULTS_DIR}"
PENDING_CHILDREN=()
for child in "${REMOTE_CHILDREN[@]}"; do
    if [[ ! "${child}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Unsafe remote child name: ${child}" >&2
        exit 1
    fi
    if [[ -e "${RESULTS_DIR}/${child}" || -L "${RESULTS_DIR}/${child}" ]]; then
        echo "[skip] ${child}: ${RESULTS_DIR}/${child} already exists"
    else
        PENDING_CHILDREN+=("${child}")
    fi
done

echo "========================================"
echo "Repo:              ${REPO_ID}"
echo "Remote directory:  ${REMOTE_ROOT}"
echo "Results directory: ${RESULTS_DIR}"
echo "ModelScope Python: ${MODELSCOPE_PYTHON}"
echo "Download workers:  1"
echo "Existing skipped:  $((${#REMOTE_CHILDREN[@]} - ${#PENDING_CHILDREN[@]}))"
echo "To download:       ${#PENDING_CHILDREN[@]}"
echo "========================================"

if (( ${#PENDING_CHILDREN[@]} == 0 )); then
    echo "All remote child directories already exist under results. Nothing to download."
    exit 0
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] would download: %s\n' "${PENDING_CHILDREN[@]}"
    exit 0
fi

# ModelScope preserves remote paths below --local-dir. This link strips the
# remote starvla_ckpt/ prefix while files are written directly into results/.
PREFIX_LINK="${RESULTS_DIR}/${REMOTE_ROOT}"
if [[ -e "${PREFIX_LINK}" || -L "${PREFIX_LINK}" ]]; then
    echo "Refusing to replace existing path: ${PREFIX_LINK}" >&2
    exit 1
fi
PREFIX_LINK_CREATED=0
cleanup() {
    if [[ "${PREFIX_LINK_CREATED}" == "1" ]]; then
        rm -f -- "${PREFIX_LINK}"
    fi
}
trap cleanup EXIT INT TERM
ln -s . "${PREFIX_LINK}"
PREFIX_LINK_CREATED=1

# A single process owns this destination, so no downloader file locks or
# parallel workers are needed.
export MODELSCOPE_DOWNLOAD_FILE_LOCK=false

download_child() {
    local child="$1"
    local include_pattern="${REMOTE_ROOT}/${child}/**"
    local attempt

    for ((attempt = 1; attempt <= MAX_RETRY; attempt++)); do
        echo
        echo "[download] ${child} (attempt ${attempt}/${MAX_RETRY})"
        if "${MODELSCOPE_BIN}" download \
            "${REPO_ID}" \
            --local-dir "${RESULTS_DIR}" \
            --max-workers 1 \
            --include "${include_pattern}"; then
            if [[ -d "${RESULTS_DIR}/${child}" ]]; then
                echo "[done] ${RESULTS_DIR}/${child}"
                return 0
            fi
            echo "Download reported success but destination is missing: ${RESULTS_DIR}/${child}" >&2
        fi

        if (( attempt < MAX_RETRY )); then
            echo "Retrying ${child} in ${RETRY_SLEEP}s..." >&2
            sleep "${RETRY_SLEEP}"
        fi
    done

    echo "Failed to download ${child} after ${MAX_RETRY} attempt(s)." >&2
    return 1
}

for child in "${PENDING_CHILDREN[@]}"; do
    download_child "${child}"
done

echo
echo "Download finished. Child directories are directly under: ${RESULTS_DIR}"
