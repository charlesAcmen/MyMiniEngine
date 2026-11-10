# M2 Part C Local Acceptance — 2026-11-09

## Evidence boundary

This report validates the Part C paged-decode implementation locally with
Qwen3-0.6B on an RTX 3060 12GB. It compares the current eager, TorchInductor,
and manual CUDA Graph paths. It is not an L4/Qwen3-8B course-performance
claim, and it does not attribute the results to FlashAttention itself: all
three paths preserve the same native FlashAttention KV-cache call.

The final collection uses `mem_fraction_static=0.60`. Earlier 0.67 runs are
retained as historical evidence but are not mixed into the aggregate below,
because they did not record GPU-memory telemetry and approached the physical
memory limit.

## Controlled environment

| Item | Local value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12GB |
| Model | repository-local `models/Qwen3-0.6B` |
| Dtype | BF16 |
| Serving path | paged KV, page size 256 |
| Static fraction | 0.60 |
| KV capacity | 221 usable pages / 56,576 token slots |
| Sampling | server-wide greedy configuration, seed 42 |
| Workload | 512 input / 256 fixed output tokens, 50 requests |
| Concurrency sweep | 4, 8, 16, 32 |
| Repetitions | three fresh server processes per execution mode |
| Telemetry | `nvidia-smi` every 0.2 seconds from before server startup |

Each server first ran a 32-input/2-output warmup across the same concurrency
sweep. The formal aggregate contains nine independent server runs and 1,800
successful requests: three eager, three compiled, and three CUDA Graph runs.

## Validation and correctness

| Check | Result |
|---|---|
| Formal requests | 1,800/1,800 successful |
| CUDA OOM / server crash | 0 / 0 |
| Dynamo recompile-limit failure | 0 |
| Server-log error scan | Clean |
| `git -c core.whitespace=cr-at-eol diff --check` | Passed; respects the existing CRLF `model.py` baseline |
| `scripts/check_part_c.sh` | Passed on the final working tree |
| Part B regression gate | Unit, paged serving, and accuracy smoke passed |
| Part C smoke | Compile, CUDA Graph, and shrinking batch passed |

The CUDA Graph smoke uses buckets 1/2/4 and a client concurrency of three, so
it exercises padding into the next graph bucket rather than only exact bucket
sizes. Unit coverage separately checks dynamic batch compilation and the graph
bucket-selection policy.

## Formal performance results

Values are the mean of three runs plus or minus sample standard deviation.
Generation throughput is output tokens per second; TPOT and TTFT are p50.

| Concurrency | Eager Gen tok/s | Compile Gen tok/s | Graph Gen tok/s |
|---:|---:|---:|---:|
| 4 | 161.5 ± 5.8 | 211.1 ± 4.7 | 408.0 ± 9.4 |
| 8 | 287.9 ± 9.2 | 384.1 ± 27.4 | 686.7 ± 7.3 |
| 16 | 490.7 ± 18.0 | 574.9 ± 54.6 | 953.1 ± 26.1 |
| 32 | 890.2 ± 14.7 | 1,065.0 ± 56.2 | 1,260.8 ± 23.2 |

| Concurrency | Eager TPOT | Compile TPOT | Graph TPOT |
|---:|---:|---:|---:|
| 4 | 23.12 ms | 17.77 ms | 9.05 ms |
| 8 | 23.74 ms | 18.02 ms | 9.86 ms |
| 16 | 24.73 ms | 20.73 ms | 12.91 ms |
| 32 | 27.12 ms | 22.20 ms | 19.17 ms |

| Concurrency | Compile vs eager throughput | Graph vs compile throughput | Graph vs eager throughput |
|---:|---:|---:|---:|
| 4 | +30.7% | +93.3% | +152.7% |
| 8 | +33.4% | +78.8% | +138.5% |
| 16 | +17.2% | +65.8% | +94.2% |
| 32 | +19.6% | +18.4% | +41.6% |

Compile reduces TPOT by 16.2%-24.1% relative to eager. CUDA Graph reduces
TPOT by a further 37.7%-49.1% at concurrency 4-16. At concurrency 32, its
incremental reduction is 13.7%, so the graph path does not meet a 20% target
for every metric at every concurrency. This is consistent with launch overhead
becoming a smaller fraction of each larger batched decode step.

TTFT remains broadly unchanged across execution modes. This is expected: Part
C compiles and captures paged decode, while packed prefill remains outside the
optimized regions.

## GPU-memory results

| Mode | Three absolute peaks | Maximum peak |
|---|---:|---:|
| Eager | 9,806 / 9,816 / 9,758 MiB | 9,816 MiB |
| Compile | 9,949 / 9,878 / 9,951 MiB | 9,951 MiB |
| CUDA Graph | 9,977 / 10,138 / 10,301 MiB | 10,301 MiB |

The highest sampled peak leaves 1,987 MiB below the GPU's 12,288 MiB physical
capacity. The graph path has the highest peak, as expected from its static
inputs and private capture pools. The 0.2-second external sampler can miss a
shorter transient, but it covers loading, warmup, and all formal workloads;
all runs also completed without OOM.

Changing the static fraction from 0.67 to 0.60 changed graph throughput by
-0.4% to -2.1% across the four concurrency levels. This is within ordinary
run-to-run variation and shows that the safer KV-pool budget does not materially
reduce performance for this workload.

## Stage conclusion

Part C meets its local goals. Compiling the stable tensor regions around the
unchanged FlashAttention call improves generation throughput by 17.2%-33.4%
over eager. Manual CUDA Graph replay then removes enough per-step launch and
dispatch overhead to improve throughput by 65.8%-93.3% over compile at
concurrency 4-16, and by 18.4% at concurrency 32.

The implementation preserves eager fallback behavior, supports dynamic batch
sizes without unbounded Dynamo recompilation, and handles non-exact graph
batches through fixed buckets. The final 0.60 configuration also restores
reasonable local memory headroom. The remaining work is code review and any
review-driven cleanup; additional performance collection is not required for
this local Part C acceptance.

Raw formal JSON is under `outputs/2026-11-09/`. Server logs, warmup logs,
formal console logs, and per-run GPU CSV telemetry are under
`outputs/2026-11-09-partc-mem060/`.
