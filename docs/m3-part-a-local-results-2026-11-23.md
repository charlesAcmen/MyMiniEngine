# Milestone 3 Part A Local Experiment Results

Date: 2026-11-23

## Scope and environment

This is local development evidence using Qwen3-0.6B on an RTX 3060. Future
RTX 4060 results must be reported separately because they come from a different
GPU configuration.

- GPU: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 616.92
- PyTorch: 2.15.0+cu132; CUDA runtime reported by PyTorch: 13.2
- model: `models/Qwen3-0.6B`
- mode/dtype/page size: paged / bfloat16 / 256
- static memory fraction: 0.75 except for explicitly labelled OOM probes
- sampling: `configs/sampling-greedy.json` (`temperature=0`)
- variants: `prefill_chunk_size=0` and `512`
- offline mode enabled for Hugging Face, datasets, and transformers
- raw artifacts: `outputs/m3-parta-2026-11-23/`

Chunk size 512 was chosen as a local compromise. It bounds each prefill step
well below the tested 1,024-token prompts while avoiding very small transformer
launches. Input length 1,024 makes chunking execute twice per request and still
allows the unchunked concurrency-16 baseline to fit on the RTX 3060.

## P0: chunk-triggering functional check

Both variants used one 1,024-token prompt, 16 fixed output tokens, concurrency
1, and seed 42. Both completed successfully.

| Variant | Success | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | Gen tok/s |
|---|---:|---:|---:|---:|
| chunk 0 | 1/1 | 1366 / 1366 | 27.4 / 27.4 | 9 |
| chunk 512 | 1/1 | 1253 / 1253 | 29.1 / 29.1 | 9 |

This establishes functional execution only. A single request is not used for a
performance conclusion.

## P1: local OOM boundary search

A qualifying activation OOM was not reproduced locally. The strongest clean
baseline point from the original 0.75 matrix, 8 requests of 8,192 input tokens,
completed 8/8 without a server error.

| Input x concurrency | Static fraction | Client outcome | Server CUDA OOM |
|---|---:|---:|---:|
| 8192 x 8 | 0.75 | 8/8 success | no |
| 8192 x 8 | 0.92 | 1/8 success, 7 socket timeouts | no |
| 16384 x 5 | 0.90 | 1/5 success, 4 socket timeouts | no |
| 40960 x 2 | 0.90 | 0/2, socket timeouts | no |

The high-fraction runs reached approximately 12.0 GiB device usage and became
very slow, but their server logs contained neither `OutOfMemoryError` nor
`CUDA out of memory`. They therefore do not satisfy the milestone's OOM
avoidance evidence requirement. A 0.94 probe was manually terminated after it
continued computing without an explicit CUDA OOM and is excluded from results.

The local limit is constrained by the approximately 90K-token KV capacity and
Qwen3's 40,960-token per-request context limit. With the small 0.6B model and
FlashAttention, the tested activation path remained executable until client
timeouts became the practical limit. The required baseline-OOM/chunk-success
pair should be probed again on the RTX 4060 to determine whether its different
memory limit exposes the intended failure/success boundary.

## P2: repeated serving comparison

Configuration: input length 1,024, fixed output length 32, 16 identical seeded
requests per concurrency, concurrency 1/4/16. Each server received one excluded
warm-up, followed by three measured runs. The table reports the median of the
three run-level p50/p99 statistics. Every cell completed 16/16 requests.

| Conc | Variant | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Gen tok/s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | chunk 0 | 75.5 ms | 209.6 ms | 19.9 ms | 22.6 ms | 45.0 |
| 1 | chunk 512 | 81.5 ms | 240.6 ms | 19.7 ms | 22.1 ms | 45.0 |
| 4 | chunk 0 | 289.4 ms | 294.4 ms | 21.0 ms | 29.7 ms | 135.5 |
| 4 | chunk 512 | 160.7 ms | 438.0 ms | 26.6 ms | 28.4 ms | 126.7 |
| 16 | chunk 0 | 1014.7 ms | 1016.2 ms | 24.1 ms | 49.0 ms | 291.6 |
| 16 | chunk 512 | 984.4 ms | 1867.9 ms | 40.3 ms | 58.4 ms | 201.8 |

Relative chunked change versus baseline:

| Conc | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Gen tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | +7.9% | +14.8% | -0.9% | -2.6% | +0.1% |
| 4 | -44.5% | +48.8% | +26.5% | -4.3% | -6.5% |
| 16 | -3.0% | +83.8% | +67.6% | +19.2% | -30.8% |

The local no-regression target is not met at concurrency 16. The result is
repeatable: chunked throughput was 201.4-202.9 tok/s versus baseline
275.8-297.3 tok/s, so the gap is much larger than run-to-run spread. The
completion-first FCFS implementation gives earlier requests prompt progress
while later requests wait behind additional prefill iterations. This explains
the improved or similar TTFT p50 together with substantially worse TTFT p99.
Extra prefill iterations also interleave with decode, increasing TPOT and
kernel-launch overhead under high concurrency.

## P3: MMLU accuracy comparison

Both variants evaluated the same 200 local MMLU samples at concurrency 8 with
greedy sampling and a 32-token output cap.

| Variant | HTTP success | Correct | Accuracy | Mean latency |
|---|---:|---:|---:|---:|
| chunk 0 | 200/200 | 75 | 37.5% | 0.665 s |
| chunk 512 | 200/200 | 77 | 38.5% | 0.668 s |

The +1 percentage-point aggregate difference is within sampling noise for only
200 questions and is not evidence that chunking improves accuracy. The saved
artifact truncates prompts to 80 characters, leaving 189 unique comparison
keys. Among those keys, seven parsed predictions differed; two changed from
incorrect to correct and none changed from correct to incorrect. Forty-five
keys had some output-text difference, often only wording or formatting. This
is consistent with different packed batch/chunk shapes changing floating-point
rounding near token decision boundaries even with greedy sampling.

## Conclusion

The local experiments verify that chunked prefill fires, completes requests,
and preserves aggregate MMLU accuracy within noise. They do not yet satisfy all
Milestone 3 Part A targets:

1. No qualifying local CUDA activation OOM was found, so the required
   unchunked-OOM/chunked-success evidence remains open for the local GPUs.
2. The current completion-first FCFS chunk scheduler has a clear high-
   concurrency performance regression, especially TTFT p99, TPOT, and
   throughput at concurrency 16. This should be treated as an implementation
   result to optimize, not reported as "within noise."
3. Accuracy is acceptable at the aggregate level, but the result format should
   retain a stable sample identifier or full prompt hash for exact per-question
   parity analysis in the final local-GPU experiment.
