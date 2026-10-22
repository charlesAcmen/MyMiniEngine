# M1 Batching Benchmark Record — 2026-10-21

## Scope and environment

This record covers the local M1 implementation only. It is not an 8B, L4, or
A6000 performance claim.

- GPU: NVIDIA GeForce RTX 3060 12GB
- Model: repository-local `models/Qwen3-0.6B`
- Dtype: `bfloat16`
- Server capacity: `--max-running 16`
- Client: `python -m benchmark.bench_serving`
- Scheduler: M1 iteration-level continuous batching

The serving benchmark reports end-to-end client-observed metrics. Completion
latency includes queueing, prefill, decode, and SSE delivery. Generation
throughput is the aggregate completed output-token rate for one concurrency
level.

## Fixed-length scaling run

Each concurrency level received 50 requests with exact 512-token inputs and
256-token output caps (the former `--randomness 1.0` flag). This is the appropriate run for
comparing throughput across batch sizes. The 50 requests are reissued for each
concurrency level, so the full sweep sends 200 requests.

| Concurrency | TTFT p50 | TTFT p99 | Completion p50 | Completion p99 | TPOT p50 | TPOT p99 | Generation throughput | Successes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59 ms | 87 ms | 8246 ms | 9961 ms | 32.9 ms | 41.3 ms | 31 tok/s | 50/50 |
| 4 | 86 ms | 222 ms | 9856 ms | 11451 ms | 38.4 ms | 46.8 ms | 99 tok/s | 50/50 |
| 8 | 181 ms | 414 ms | 14891 ms | 16266 ms | 57.9 ms | 100.5 ms | 131 tok/s | 50/50 |
| 16 | 377 ms | 850 ms | 25513 ms | 25785 ms | 98.0 ms | 154.2 ms | 158 tok/s | 50/50 |

`conc=16` reached `158 / 31 = 5.10x` the generation throughput of `conc=1`.
Throughput increased monotonically with concurrency, while TTFT, TPOT, and
per-request completion latency increased. This is the expected throughput versus
per-request latency trade-off for batching. The 5.10x result is within the M1
working expectation of roughly 5–7x at concurrency 16 on this local setup.

## Variable-length continuous-batching run

This separate run used 50 requests at `--concurrencies 16` with
the former `--randomness 0.5` flag. Its measured request lengths were intentionally variable:

- Input: min 259, max 511, mean 383 tokens
- Output cap: min 134, max 254, mean 184 tokens

| Concurrency | TTFT p50 | TTFT p99 | Completion p50 | Completion p99 | TPOT p50 | TPOT p99 | Generation throughput | Successes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 149 ms | 1964 ms | 15566 ms | 22307 ms | 95.8 ms | 106.3 ms | 150 tok/s | 50/50 |

This run confirms that the continuous scheduler completed a mixed-length,
over-capacity workload without request failures. Its throughput must **not** be
directly compared with the fixed-length `conc=16` result: it intentionally
contains fewer input and output tokens per request. Its useful signals are the
successful 50/50 completion and its TTFT/completion-latency distribution under
mixed request lengths.

## Correctness and evidence boundary

The CPU M1 suite passed 9/9 tests before recording this summary. It covers
heterogeneous KV-cache parity against independent decode, compact KV storage,
same-iteration admission into batched decode, next-iteration slot refill, and
prefill-stop handling.

These measurements are one local run per condition, not repeated-trial
statistics. Preserve future raw terminal summaries with `tee` and rerun the
same fixed-length parameters when comparing later scheduler revisions.

```bash
mkdir -p benchmark/results
run_id=$(date +%Y%m%d_%H%M%S)
python -m benchmark.bench_serving --model models/Qwen3-0.6B \
  --input-len 512 --output-len 256 --num-requests 50 \
  --input-len-min-ratio 1.0 --concurrencies 1,4,8,16 \
  | tee "benchmark/results/m1_fixed_${run_id}.txt"
```
