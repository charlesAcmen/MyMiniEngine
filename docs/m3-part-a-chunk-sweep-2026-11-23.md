# M3 Part A Chunk-Size Sweep

Date: 2026-11-23

## Controls

- RTX 3060 12 GiB, local Qwen3-0.6B, paged bfloat16
- page size 256, static memory fraction 0.75, max running 16
- fixed input length 4096 and fixed output length 32
- 16 requests, seed 42, concurrency 1/4/16
- greedy sampling
- one excluded warm-up and three measured runs per chunk size
- reported values are medians of the three run-level statistics
- raw artifacts: `outputs/m3-parta-chunk-sweep-2026-11-23/`

Chunk sizes 512, 1024, and 2048 split every 4096-token prompt. Size 4096 is
the per-request boundary. Sizes 8192 and 16384 do not split an individual
prompt, but still cap the concurrent 65,536-token offered prefill into 8 and 4
scheduler iterations respectively.

## Results

| Chunk | Conc | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Gen tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 432.9 | 648.0 | 22.9 | 27.0 | 27.3 |
| 512 | 1 | 400.8 | 574.7 | 21.2 | 22.9 | 29.7 |
| 1024 | 1 | 374.3 | 564.7 | 20.9 | 23.1 | 30.7 |
| 2048 | 1 | 345.1 | 563.1 | 20.6 | 21.7 | 32.0 |
| 4096 | 1 | 335.9 | 529.5 | 21.8 | 32.9 | 30.1 |
| 8192 | 1 | 336.3 | 542.6 | 20.9 | 22.4 | 31.9 |
| 16384 | 1 | 335.9 | 509.1 | 21.7 | 23.0 | 31.3 |
| 0 | 4 | 1584.6 | 1650.3 | 25.1 | 63.7 | 53.9 |
| 512 | 4 | 636.9 | 2034.5 | 58.7 | 61.7 | 50.3 |
| 1024 | 4 | 564.9 | 1659.5 | 54.9 | 58.6 | 55.6 |
| 2048 | 4 | 574.7 | 1442.7 | 48.7 | 56.4 | 60.1 |
| 4096 | 4 | 680.5 | 1332.4 | 43.7 | 54.5 | 62.5 |
| 8192 | 4 | 1005.6 | 1314.6 | 33.9 | 55.3 | 62.2 |
| 16384 | 4 | 1312.2 | 1317.4 | 23.2 | 55.3 | 62.9 |
| 0 | 16 | 36537.7 | 36539.2 | 52.8 | 1043.8 | 13.4 |
| 512 | 16 | 4827.0 | 9219.7 | 71.5 | 73.4 | 51.3 |
| 1024 | 16 | 3832.4 | 7427.7 | 112.6 | 118.7 | 62.1 |
| 2048 | 16 | 3251.0 | 6282.6 | 118.0 | 193.2 | 70.2 |
| 4096 | 16 | 2994.9 | 5690.2 | 118.9 | 194.7 | 74.6 |
| 8192 | 16 | 3064.6 | 5441.4 | 114.7 | 194.6 | 76.3 |
| 16384 | 16 | 2961.6 | 5292.7 | 114.7 | 190.5 | 78.0 |

All cells completed 16/16 requests.

## Interpretation

No tested chunk size is within noise of the unchunked path on every metric.
For chunk sizes that actually split each prompt, 2048 is the best local
throughput point at concurrency 16, but its TPOT p50 is still 118.0 ms versus
52.8 ms for the baseline. Chunk 512 protects TPOT better at concurrency 16,
but pays more launch and scheduling overhead and has lower throughput.

The unchunked concurrency-16 result is not a healthy high-throughput baseline.
It processes one early HTTP arrival first, then performs a very large prefill
for the remaining 15 requests. In a representative run, request 0 had TTFT
456 ms and TPOT 1094 ms, while the other requests had TTFT about 32.8 seconds
and TPOT about 50.8 ms. The first request cannot decode while the later packed
prefill occupies the scheduler thread.

This exposes two independent scheduler effects:

1. Immediate Condition wakeup provides no short arrival-coalescing window, so
   the first HTTP request can be peeled away from an otherwise concurrent
   batch. The old polling delay accidentally supplied such a window.
2. Completion-first FCFS chunking staggers prompt completion. Early requests
   decode while later prompts continue prefill, producing small and changing
   decode batches. This improves memory pressure and TTFT for early requests
   but can increase TPOT and tail TTFT.

The sweep does demonstrate the intended memory/performance trade-off. At
concurrency 16, the 65,536-token unchunked prefill enters severe pressure and
delivers only 13.4 tok/s. Capping packed q-tokens raises throughput
monotonically over the tested chunked range from 51.3 tok/s at 512 to 78.0
tok/s at 16384, while smaller chunks provide a tighter activation bound.

## Next experiments

1. Add iteration telemetry for prefill token count, number of chunks, decode
   batch size, and prefill/decode duration.
2. Add a small explicit request-coalescing delay implemented with the
   Condition variable, not polling, and repeat the unchunked/chunked sweep.
3. Compare completion-first with a fair-share planner that packs multiple
   requests into the same prefill budget so prompts finish closer together.
4. Define a total iteration token budget where decode-ready rows consume one
   token each and remaining budget is assigned to prefill.
5. Re-select the chunk size independently on the RTX 4060. The RTX 3060
   optimum is not automatically transferable because launch overhead and
   activation headroom differ between GPUs.
