# Milestone 3 Part A Experiment Plan

Date: 2026-11-17

## Experiment scope

All experiments in this plan use the local RTX 3060 and Qwen3-0.6B. Keep the
following evidence separate:

- functional: chunking fires and requests complete;
- OOM avoidance: the same workload fails unchunked and succeeds chunked;
- performance: repeated successful runs where both variants fit;
- accuracy: greedy MMLU on an identical fixed sample set.

## Fixed controls

- mode: `paged`
- dtype: `bfloat16`
- page size: `256` locally (required by the installed paged FlashAttention path)
- sampling: `configs/sampling-greedy.json`
- offline mode: `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`
- compare `--prefill-chunk-size 0` against `512`
- restart the server between variants so allocator state and startup policy match
- use the same benchmark seed and request set for both variants

## RTX 3060 / Qwen3-0.6B matrix

### P0 — Chunk-triggering fit check

Use Qwen3-0.6B, input length 1024, output length 16, and concurrency 1. Both
variants must complete; chunk size 512 must execute two prompt chunks.

### P1 — OOM boundary search

Use `--mem-fraction-static 0.75`, output length 2, and one request per worker.
Test the unchunked path progressively and stop at the first reproducible CUDA
OOM:

| Input length | Concurrency | Total offered prompt tokens |
|---:|---:|---:|
| 4096 | 4 | 16384 |
| 4096 | 8 | 32768 |
| 8192 | 4 | 32768 |
| 8192 | 8 | 65536 |

For the first failing workload, restart the same server configuration with
chunk size 512 and repeat the identical client command. Success requires all
requests to complete with no scheduler/server error.

Record both sides:

- server log containing the CUDA OOM for chunk size 0;
- client HTTP/stream failure summary;
- chunked server log with no OOM;
- chunked benchmark table with every request successful.

If the server cannot start because unrelated processes consume GPU memory,
lower `mem-fraction-static` equally for both variants instead of changing only
one side.

### P2 — Successful performance comparison

Use input length 1024 and output length 32 so chunking fires twice while the
unchunked baseline still fits at concurrency 16. Sweep concurrency 1, 4, and
16. Run each variant three times after one warm-up run.

Report median across the three measured runs for:

- TTFT p50 / p99;
- TPOT p50 / p99;
- generation throughput;
- successful request count.

Use run-to-run spread to define “within noise”; do not claim no regression
from a single run.

### P3 — Accuracy comparison

Run the same 200 shuffled MMLU samples with concurrency 8 and greedy sampling
for chunk sizes 0 and 512. Compare parsed answers per sample as well as aggregate
accuracy. Any changed answers should be listed separately from HTTP failures.

## Interpretation

Chunk size 512 is the initial compromise: it is small enough to cap per-step
query activations substantially, but large enough to avoid extremely small
transformer launches. Input length 1024 makes chunking fire twice while retaining
enough headroom for the unchunked concurrency-16 comparison on the RTX 3060.
The OOM workload is discovered empirically because activation headroom depends
on the static KV budget, concurrency, allocator state, and other GPU users.
