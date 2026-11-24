# M3 Part A Healthy-Baseline Chunk-Size Sweep

Date: 2026-11-23

## Question

The earlier 4,096-token sweep allowed the unchunked concurrency-16 path to
pack 65,536 query tokens into one forward. That baseline was already in a
severe pressure region on the RTX 3060, so it was not suitable for deciding
whether chunking has no regression. This sweep uses 1,024-token prompts so
the unchunked baseline remains healthy, then asks whether the original global
budget of 512 tokens was simply too small to keep the GPU well utilized.

## Controls

- RTX 3060 12 GiB, local Qwen3-0.6B, paged bfloat16
- page size 256, static memory fraction 0.75, max running 16
- fixed input length 1,024 and fixed output length 32
- 16 requests, seed 42, concurrency 1/4/16
- greedy sampling
- 20 ms fixed arrival-coalescing window
- completion-first prefill planner
- one excluded warm-up and three measured runs per server configuration
- values below are medians of the three run-level statistics
- raw artifacts: `outputs/m3-parta-healthy-sweep-2026-11-23/`

The total offered prefill at concurrency 16 is 16,384 tokens. Budgets below
16,384 therefore execute the chunked path over multiple scheduler iterations;
16,384 uses the chunked data path but can finish the whole offered prefill in
one iteration.

## Results

| Chunk | Conc | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Gen tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 98.5 | 259.8 | 21.6 | 25.4 | 40.2 |
| 512 | 1 | 105.5 | 269.8 | 22.2 | 31.4 | 39.2 |
| 1024 | 1 | 99.2 | 257.3 | 21.5 | 25.4 | 40.2 |
| 2048 | 1 | 100.6 | 258.7 | 22.1 | 26.2 | 39.6 |
| 4096 | 1 | 101.3 | 231.8 | 23.7 | 30.2 | 36.8 |
| 8192 | 1 | 101.7 | 237.4 | 23.5 | 25.8 | 38.1 |
| 16384 | 1 | 101.9 | 266.6 | 23.0 | 25.1 | 38.1 |
| 0 | 4 | 277.9 | 280.4 | 21.6 | 22.9 | 134.8 |
| 512 | 4 | 167.7 | 472.8 | 29.4 | 33.4 | 115.1 |
| 1024 | 4 | 176.0 | 375.3 | 28.7 | 31.7 | 118.5 |
| 2048 | 4 | 231.3 | 320.7 | 25.3 | 28.1 | 125.1 |
| 4096 | 4 | 284.6 | 291.9 | 22.8 | 23.9 | 128.6 |
| 8192 | 4 | 291.0 | 307.8 | 23.6 | 24.7 | 125.4 |
| 16384 | 4 | 292.6 | 296.5 | 23.0 | 25.8 | 125.1 |
| 0 | 16 | 1041.3 | 1042.6 | 24.7 | 24.7 | 284.6 |
| 512 | 16 | 1095.9 | 2097.1 | 44.5 | 64.2 | 179.2 |
| 1024 | 16 | 869.4 | 1622.0 | 43.0 | 61.1 | 215.2 |
| 2048 | 16 | 731.0 | 1295.9 | 40.7 | 56.0 | 246.8 |
| 4096 | 16 | 724.8 | 1189.0 | 40.1 | 52.6 | 255.1 |
| 8192 | 16 | 879.5 | 1172.0 | 38.8 | 47.8 | 243.7 |
| 16384 | 16 | 1088.1 | 1089.4 | 27.5 | 27.5 | 263.4 |

## Concurrency-16 change versus unchunked

| Chunk | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Gen tok/s |
|---:|---:|---:|---:|---:|---:|
| 512 | +5.2% | +101.1% | +80.2% | +159.5% | -37.0% |
| 1024 | -16.5% | +55.6% | +74.1% | +147.2% | -24.4% |
| 2048 | -29.8% | +24.3% | +64.6% | +126.4% | -13.3% |
| 4096 | -30.4% | +14.0% | +62.3% | +112.6% | -10.3% |
| 8192 | -15.5% | +12.4% | +57.1% | +93.3% | -14.3% |
| 16384 | +4.5% | +4.5% | +11.3% | +11.4% | -7.4% |

The three concurrency-16 throughput runs were 276.8-287.2 tok/s for the
unchunked baseline and 255.1-276.0 tok/s for chunk 16,384. Their ranges only
touch at the boundary, so the median 7.4% loss should not be called within
noise from these data.

## Interpretation

The 512-token global budget was too small for this GPU/model pair. Raising it
to 2,048-4,096 recovered most of the lost throughput while still splitting
the 16,384-token offered prefill into multiple forwards. This supports using
the chunk limit as a large activation-memory safety bound rather than as a
tiny microbatch size.

A larger budget alone does not make every metric match. With the
completion-first planner, a budget below 16,384 completes only an FCFS prefix
of requests. Those requests begin decoding while later prompts still consume
prefill iterations. Their decode gaps inflate TPOT, while early completion
improves TTFT p50 and later completion worsens TTFT p99. This explains why
throughput approaches the baseline faster than the latency distribution.

Chunk 16,384 finishes all prompts together and consequently comes closest on
TTFT and TPOT, but it does not actually bound this workload below its complete
offered prefill. Its remaining 7.4% median throughput loss estimates the
combination of run variance and fixed overhead in the chunked metadata/model
path; it is not evidence that a useful multi-iteration setting is within
noise.

## Conclusion

The hypothesis is partly confirmed: 512 is an unnecessarily restrictive
local setting, and 2,048-4,096 is a substantially better throughput region on
the RTX 3060 with Qwen3-0.6B. However, the current completion-first schedule
still cannot meet the milestone's all-metrics-within-noise target when
chunking actually splits the concurrency-16 prefill. The next discriminating
experiment is an all-active-request fair-share planner with a large budget:
it should keep prompt completion synchronized and reveal whether the remaining
TPOT gap is scheduling policy rather than the chunked forward itself.
