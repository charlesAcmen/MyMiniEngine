# M3 Part A Prefill-Policy Sweep

Date: 2026-11-23

## Question

The healthy-baseline chunk sweep showed that a 512-token global budget was
too small, but completion-first still inflated TPOT when chunking actually
split the concurrency-16 prefill. This experiment compares completion-first
with full fair-share at larger budgets. Full fair-share divides each iteration
across every active prefilling request so their prompts complete together.

## Controls

- RTX 3060 12 GiB, local Qwen3-0.6B, paged bfloat16
- page size 256, static memory fraction 0.75, max running 16
- fixed input length 1,024 and fixed output length 32
- 16 requests, seed 42, concurrency 1/4/16
- greedy sampling and a 20 ms fixed arrival-coalescing window
- budgets 4,096 and 8,192, versus a same-run unchunked baseline
- one excluded warm-up and three measured runs per server configuration
- values are medians of the three run-level statistics
- raw artifacts: `outputs/m3-parta-policy-sweep-2026-11-23/`

At concurrency 16 the total offered prefill is 16,384 tokens. Budget 4,096
therefore requires four prefill iterations and budget 8,192 requires two.

## Results

| Policy | Chunk | Conc | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Gen tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| unchunked | 0 | 1 | 98.9 | 243.1 | 21.6 | 22.8 | 41.1 |
| unchunked | 0 | 4 | 281.7 | 283.9 | 22.7 | 23.7 | 130.6 |
| unchunked | 0 | 16 | 1041.2 | 1041.9 | 24.6 | 24.6 | 285.2 |
| completion-first | 4096 | 1 | 100.9 | 256.8 | 23.1 | 29.0 | 38.0 |
| completion-first | 4096 | 4 | 294.1 | 306.4 | 24.0 | 27.2 | 123.4 |
| completion-first | 4096 | 16 | 742.6 | 1172.3 | 39.6 | 52.6 | 253.6 |
| fair-share | 4096 | 1 | 101.8 | 223.4 | 22.3 | 26.7 | 39.6 |
| fair-share | 4096 | 4 | 292.7 | 295.2 | 22.8 | 25.4 | 126.4 |
| fair-share | 4096 | 16 | 1129.1 | 1130.6 | 26.2 | 26.2 | 264.3 |
| completion-first | 8192 | 1 | 101.9 | 248.3 | 23.5 | 26.7 | 37.8 |
| completion-first | 8192 | 4 | 290.5 | 298.6 | 22.2 | 22.4 | 131.1 |
| completion-first | 8192 | 16 | 858.9 | 1140.8 | 34.7 | 43.1 | 264.4 |
| fair-share | 8192 | 1 | 104.0 | 245.9 | 23.8 | 28.5 | 37.1 |
| fair-share | 8192 | 4 | 297.2 | 307.7 | 23.9 | 26.0 | 120.8 |
| fair-share | 8192 | 16 | 1142.8 | 1144.1 | 26.5 | 26.6 | 260.3 |

## Concurrency-16 change versus same-run unchunked baseline

| Policy | Chunk | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Gen tok/s |
|---|---:|---:|---:|---:|---:|---:|
| completion-first | 4096 | -28.7% | +12.5% | +60.8% | +113.5% | -11.1% |
| fair-share | 4096 | +8.4% | +8.5% | +6.2% | +6.2% | -7.3% |
| completion-first | 8192 | -17.5% | +9.5% | +41.0% | +75.0% | -7.3% |
| fair-share | 8192 | +9.8% | +9.8% | +7.8% | +7.8% | -8.7% |

## Fair-share change versus completion-first

At concurrency 16:

| Chunk | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Gen tok/s |
|---:|---:|---:|---:|---:|---:|
| 4096 | +52.1% | -3.6% | -33.9% | -50.2% | +4.2% |
| 8192 | +33.1% | +0.3% | -23.6% | -38.4% | -1.6% |

## Interpretation

Fair-share confirms that completion timing, not only chunk-forward overhead,
caused the earlier TPOT regression. At budget 4,096, completion-first finishes
an FCFS prefix early and lets it decode while the remaining prompts prefill.
This improves TTFT p50 but produces long gaps between output tokens. Fair-share
advances all 16 prompts by 256 tokens per iteration; they finish together, so
TTFT p50 and p99 reconverge and TPOT returns to within about 6% of unchunked.

Fair-share does not eliminate all overhead. Its 4,096-token result remains
7.3% below the unchunked throughput median and about 6-8.5% above the baseline
latencies. These differences should not be claimed as within noise from this
experiment. The unchunked concurrency-16 throughput runs were 285.2, 285.9,
and 250.4 tok/s; fair-share 4,096 produced the much tighter 264.3, 268.2, and
262.1 tok/s. The baseline has one low outlier, but its first two runs are still
clearly faster.

Budget 8,192 is not automatically better. Completion-first only needs two
prefill iterations at this size, so its interleaving penalty is already
smaller. Fair-share makes each packed forward contain more, shorter query
sequences; the extra ragged-sequence metadata and less favorable kernel shape
can outweigh the scheduling benefit. It restores TPOT but loses 1.6% throughput
against completion-first and is worse than fair-share 4,096 overall.

## Conclusion

For the local RTX 3060/Qwen3-0.6B workload, full fair-share with a 4,096-token
budget is the best tested compromise. It truly caps the 16,384-token offered
prefill, recovers 4.2% throughput over completion-first, and removes most of
the TPOT and tail-latency distortion. It is much closer to the milestone's
no-regression target, but the remaining 6-8.5% gaps are still measurable and
must be reported rather than labeled noise. The final policy and chunk size
must be re-selected on the required L4/Qwen3-8B environment.
