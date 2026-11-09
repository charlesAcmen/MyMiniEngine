#!/usr/bin/env bash
# Part-C verification gate: preserve Part B, then exercise compile + CUDA graph.

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
    if [[ "${log_dir}" == /tmp/tmp.* && -d "${log_dir}" ]]; then
        rm -rf -- "${log_dir}"
    fi
}
trap cleanup EXIT INT TERM

fail_with_log() {
    local stage="$1"
    local log_file="$2"
    printf 'Part-C check failed during %s.\n' "${stage}" >&2
    if [[ -f "${log_file}" ]]; then
        printf '%s\n' '--- captured output ---' >&2
        cat "${log_file}" >&2
    fi
    exit 1
}

cd "${repo_root}"
HF_HUB_OFFLINE=1 scripts/check_part_b.sh

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
    --mem-fraction-static 0.60 \
    --max-running 4 \
    --sampling-config "${repo_root}/configs/sampling-greedy.json" \
    --torch-compile \
    --cuda-graph \
    --cuda-graph-batch-sizes 1,2,4 \
    --host 127.0.0.1 \
    --port "${port}" >"${server_log}" 2>&1 &
server_pid=$!

server_ready=false
for _ in $(seq 1 240); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        fail_with_log "compiled graph server startup" "${server_log}"
    fi
    if curl --silent --fail "${base_url}/health" >/dev/null 2>&1; then
        server_ready=true
        break
    fi
    sleep 0.25
done
if [[ "${server_ready}" != true ]]; then
    fail_with_log "compiled graph server readiness" "${server_log}"
fi

serving_log="${log_dir}/serving.log"
if ! "${python_bin}" -m benchmark.bench_serving \
    --base-url "${base_url}" \
    --model "${model_dir}" \
    --input-len 32 \
    --output-len 4 \
    --num-requests 4 \
    --concurrencies 3 >"${serving_log}" 2>&1; then
    fail_with_log "compiled graph serving smoke" "${serving_log}"
fi

if ! grep -q '"cuda_graph": {"batch_sizes": \[1, 2, 4\], "enabled": true}' "${serving_log}"; then
    fail_with_log "execution metadata reporting" "${serving_log}"
fi

printf '%s\n' 'Part-C checks passed (Part B, compile, CUDA graph, shrinking batch).'
