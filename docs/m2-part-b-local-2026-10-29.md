# M2 Part B Local Acceptance — 2026-10-29

## Evidence boundary

This is a local checkpoint for the paged-KV and packed-prefill implementation,
using Qwen3-0.6B on an RTX 3060 12GB. It is not an L4/Qwen3-8B, Part C, or
final course-performance claim. Historical M1 one-shot numbers are not reused:
all speedups below pair the current `batched` and `paged` paths on the same
request set and configuration.

The initially planned full matrix was deliberately compressed after c32 and
long-prefill pilots reached the memory safety threshold. No CUDA OOM occurred,
and failed conditions were not retried. The accepted raw evidence is
`outputs/2026-10-29-partb-compact-v3/`; earlier interrupted and safety-pilot
directories must not be mixed into its aggregate.

## Controlled environment

| Item | Local value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12GB |
| Model | repository-local `models/Qwen3-0.6B` |
| Dtype/backend | BF16 / FlashAttention 2.8.4 |
| Page size | 256 tokens |
| Static fraction | 0.68 |
| Paged capacity | 256 usable pages / 65,536 token slots |
| Client | seed 42, temperature 0, exact fixed output length |
| Safety limits | 11,520 MiB FB, 83 C, trial timeout guard |
| Telemetry | `nvidia-smi dmon`, one-second samples |

Every trial JSON records the Git state, configuration hash, commands,
request-level timings, errors, and telemetry. The runner starts a fresh server
per mode/trial, performs warmup, writes results atomically, and cleans up the
process group. Fatal failures are excluded from numeric aggregates.

## Validation and correctness

| Check | Result |
|---|---|
| Unit tests | 36/36 passed |
| Python compilation and `git diff --check` | Passed |
| `scripts/check_part_b.sh` | Passed before collection |
| Formal trials | 21 attempted; 16 successful; 5 safety-limit failures |
| CUDA OOM / pool exhaustion / timeout | 0 / 0 / 0 |
| MMLU-50 batched | 18/50 = 36% |
| MMLU-50 paged | 17/50 = 34% |

The MMLU runs used the same first 50 samples and deterministic client settings,
but differ by one answer. This small smoke confirms both paths complete; it is
not strong enough to claim exact output parity or a stable accuracy delta. The
original 200-sample acceptance was intentionally not run in this compressed
session.

## Successful A/B results

`balanced` is the only repeated condition. Its throughput is reported as
mean ± sample standard deviation with min-max; other rows are one-shot results.

| Workload | Batched Gen tok/s | Paged Gen tok/s | Paired speedup | Batched / paged peak FB MiB | Status |
|---|---:|---:|---:|---:|---|
| 512/128, c8 | 164.2 | 242.6 | 1.48x | 4,728 / 10,614 | 1 pair |
| 512/128, c16 | 175.9 | 507.9 | 2.89x | 7,802 / 11,510 | 1 pair |
| 512/256, c16 | 168.2 ± 19.7 (154.3-182.2) | 586.1 ± 11.9 (577.6-594.5) | 3.51x mean (3.17-3.85) | 10,184 / 11,510 max | 2 pairs |
| ragged 268-472/256, c16 | 181.2 | 537.8 | 2.97x | 8,844 / 10,566 | 1 pair |
| 256/512, c16 | 176.0 | 613.6 | 3.49x | 10,016 / 10,128 | 1 pair |

For the repeated balanced condition, paged reduced the mean of trial TTFT p50
values from 905 ms to 771 ms, and the mean of trial TPOT p50 values from
92.6 ms to 24.4 ms. The two paged runs completed in 6.89-7.09 s p50 versus
22.48-26.54 s for batched.

## Telemetry interpretation

| Workload/mode | Avg SM | Avg power | Max temperature | Interpretation |
|---|---:|---:|---:|---|
| balanced batched | 72.6% | 104.9 W | 71 C | Higher SM/power, but much slower decode path |
| balanced paged | 37.3% | 69.3 W | 57 C | 3.51x throughput despite lower sampled utilization |
| decode-heavy batched | 70.4% | 101.0 W | 76 C | Repeated KV concatenation remains expensive |
| decode-heavy paged | 40.6% | 71.6 W | 60 C | 3.49x throughput and similar peak FB |

The lower paged SM/power samples do not mean the GPU is saturated. They show
that this 0.6B workload is still launch/CPU/short-kernel limited while the
batched baseline spends more GPU work on its less efficient cache path. Part C
or CUDA Graph/compile work is still needed before claiming full GPU utilization.

## Capacity, prefill, and overload

| Condition | Result |
|---|---|
| c32, 512/128 pilot | Both modes crossed 11,520 MiB and were stopped before OOM; c32 was not retried in the formal v3 matrix |
| 2048/128, c12 | Both modes stopped at the safety line (11,938 / 11,794 MiB) |
| 4096/64, c8 | Both modes stopped at the safety line (12,030 / 11,920 MiB) |
| 2048/32, c12 prefill-heavy | Batched passed at 51.7 Gen tok/s and 10,888 MiB; paged crossed 11,914 MiB, so no fair speedup is reported |
| client c32, max-running 16 | Paged passed: 572.4 Gen tok/s, TTFT p50/p99 2.20/4.07 s, peak 11,390 MiB |

The long-context failures are conservative capacity evidence, not OOMs and not
performance samples. They also show that page-count preflight alone is
insufficient: packed-prefill activations and FA workspace can exceed the FB
safety budget even when KV pages fit (for example, 4K c8 needs only 136/256
usable pages).

## Stage conclusion

Part B demonstrates a clear local benefit on successful c16 decode-bearing
workloads: 2.89-3.51x paired generation throughput, lower TTFT, and sharply
lower TPOT. The 2x target is therefore met for balanced, ragged, and
decode-heavy c16 on this machine, but not at c8.

The main unresolved issue is high-concurrency packed-prefill memory headroom.
The 0.68 static pool leaves enough room for 512-token c16 and client-overload
queueing, but not the chosen 2K/4K concurrent prefills under the 768 MiB reserve.
Future work should measure chunked prefill or a workload-aware pool budget;
raising the memory fraction or retrying OOM-adjacent conditions would weaken,
not improve, this acceptance result.

## README-standard follow-up — 2026-10-30

The follow-up uses the README workload exactly: 512 input tokens, 256 fixed
output tokens, 50 requests, and concurrency 1/4/8/16. Each serving condition
was scheduled for three paired trials. The paged static fraction was 0.67,
providing 252 usable pages; the c16 workload needs only 48 pages.

| Concurrency | Batched Gen tok/s | Paged Gen tok/s | Paired speedup | Batched / paged TTFT p50 | Batched / paged TPOT p50 | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 38.7 ± 3.9 | 44.4 ± 3.7 | 1.15x mean | 47.9 / 47.4 ms | 24.2 / 21.5 ms | 3 pairs |
| 4 | 118.1 ± 9.0 | 158.8 ± 15.1 | 1.34x mean | 169.5 / 153.4 ms | 32.3 / 22.7 ms | 3 pairs |
| 8 | 146.5 ± 2.8 | 272.7 ± 34.3 | 1.86x mean | 371.0 / 317.4 ms | 50.7 / 24.7 ms | 3 pairs |
| 16 | 165.8 ± 3.7 | 413.3 ± 70.8 | 2.49x mean | 744.0 / 611.4 ms | 88.4 / 29.0 ms | 3 corrective pairs |

Paged throughput improved monotonically with concurrency. The c8 paired ratios
ranged from 1.61x to 2.10x. The original c16 batched attempt was invalidated
because the first runner policy incorrectly treated NVIDIA's 83 C target
temperature as a hard ceiling. It is preserved as a failed trial and was not
included in statistics.

Three fresh batched c16 trials were collected after changing the unattended
policy to: start at or below 55 C, warn above 83 C, stop after two consecutive
samples at or above 88 C, and stop immediately at 90 C. No serving or model
code changed. The corrective trials reached only 75/77/80 C and completed all
50 requests. Their paired paged speedups were 2.58x, 2.05x, and 2.85x (2.49x
mean). Paged reduced TTFT p50 by 17.8%, TPOT p50 by 67.2%, and completion p50
by 65.4%. Batched used 10,149 MiB mean peak FB and reached 77.3 C mean maximum;
paged used 11,297 MiB and reached 65.7 C. The README-standard c16 result now
meets the milestone's 2x throughput target.

The earlier compact c16 result remains separate evidence and is not mixed with
this 50-request aggregate.

### README accuracy datasets

| Dataset | Samples | Batched | Paged | Difference |
|---|---:|---:|---:|---:|
| MMLU | 200 | 76/200 = 38.0% | 75/200 = 37.5% | -0.5 percentage point |
| GSM8K | 100 | 15/100 = 15.0% | 12/100 = 12.0% | -3.0 percentage points |

Both modes completed both README datasets without request failures. The MMLU
difference is one question. The three-question GSM8K difference is within the
large sampling uncertainty of this 100-question subset, but the paths are not
token-for-token identical; future compile/graph comparisons must use the same
fixed samples and retain per-question outputs rather than claiming exact
parity.
