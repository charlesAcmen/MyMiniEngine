#!/usr/bin/env bash
# Quiet Part-B verification gate: unit tests plus two local benchmark smokes.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
model_dir="${MODEL_DIR:-${repo_root}/models/Qwen3-0.6B}"
log_dir="$(mktemp -d)"
server_pid=""

cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -INT "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    # log_dir is created by mktemp and must remain an exact child of /tmp.
    if [[ "${log_dir}" == /tmp/tmp.* && -d "${log_dir}" ]]; then
        rm -rf -- "${log_dir}"
    fi
}
trap cleanup EXIT INT TERM

fail_with_log() {
    local stage="$1"
    local log_file="$2"
    printf 'Part-B check failed during %s.\n' "${stage}" >&2
    if [[ -f "${log_file}" ]]; then
        printf '%s\n' '--- captured output ---' >&2
        cat "${log_file}" >&2
    fi
    exit 1
}

if [[ ! -x "${python_bin}" ]]; then
    printf 'Part-B check requires the project interpreter at %s.\n' "${python_bin}" >&2
    exit 1
fi
if [[ ! -d "${model_dir}" ]]; then
    printf 'Part-B check requires the local model at %s.\n' "${model_dir}" >&2
    exit 1
fi
if [[ ! -d "${repo_root}/data/WildChat-1M/data" ]]; then
    printf 'Part-B check requires local WildChat data.\n' >&2
    exit 1
fi
if [[ ! -d "${repo_root}/data/mmlu" ]]; then
    printf 'Part-B check requires local MMLU data.\n' >&2
    exit 1
fi

cd "${repo_root}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

unit_log="${log_dir}/unit.log"
if ! "${python_bin}" -m unittest discover -s tests >"${unit_log}" 2>&1; then
    fail_with_log "unit tests" "${unit_log}"
fi
if ! "${python_bin}" -m py_compile miniengine/*.py benchmark/*.py >>"${unit_log}" 2>&1; then
    fail_with_log "Python compilation" "${unit_log}"
fi

# Ask the OS for an unused loopback port so concurrent developer tools do not
# collide with a fixed benchmark port.
port="$("${python_bin}" - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
base_url="http://127.0.0.1:${port}"
server_log="${log_dir}/server.log"

"${python_bin}" -m miniengine \
    --model "${model_dir}" \
    --mode paged \
    --dtype bfloat16 \
    --page-size 256 \
    --mem-fraction-static 0.50 \
    --max-running 2 \
    --host 127.0.0.1 \
    --port "${port}" >"${server_log}" 2>&1 &
server_pid=$!

server_ready=false
for _ in $(seq 1 120); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        fail_with_log "paged server startup" "${server_log}"
    fi
    if curl --silent --fail "${base_url}/health" >/dev/null 2>&1; then
        server_ready=true
        break
    fi
    sleep 0.25
done
if [[ "${server_ready}" != true ]]; then
    fail_with_log "paged server readiness" "${server_log}"
fi

serving_log="${log_dir}/serving.log"
if ! "${python_bin}" -m benchmark.bench_serving \
    --base-url "${base_url}" \
    --model "${model_dir}" \
    --input-len 32 \
    --output-len 2 \
    --num-requests 2 \
    --input-len-min-ratio 0.5 \
    --concurrencies 1,2 >"${serving_log}" 2>&1; then
    fail_with_log "serving benchmark smoke" "${serving_log}"
fi

accuracy_log="${log_dir}/accuracy.log"
if ! "${python_bin}" -m benchmark.bench_accuracy \
    --dataset mmlu \
    --num-samples 2 \
    --max-tokens 4 \
    --concurrency 2 \
    --base-url "${base_url}" >"${accuracy_log}" 2>&1; then
    fail_with_log "accuracy benchmark smoke" "${accuracy_log}"
fi

printf '%s\n' 'Part-B checks passed (unit, serving smoke, accuracy smoke).'
