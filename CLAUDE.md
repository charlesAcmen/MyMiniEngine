# MiniEngine Project Notes

Use the rules in AGENTS.md.

## Current M2 State

- The paged path uses iteration-level continuous batching. The scheduler owns
  request-state transitions; the paged KV pool owns GPU KV-page allocation.
- Preserve scheduler/request alignment: a token returned at batch index `i`
  belongs to the request at batch index `i`.
- Preserve GPU-memory ownership: weights and the static paged KV pool stay on
  CUDA, while the scheduler retains only request metadata and generated token
  IDs on the host.

## Environment

- Work in ubuntu26, never the ubuntu22 research environment.
- The editable project `.venv` is the local CUDA environment. The verified
  laptop stack is CPython 3.12, PyTorch 2.8.0+cu128, and FlashAttention
  2.8.3.post2 from its matching prebuilt wheel.
- Use `models/Qwen3-0.6B`, not a Hub model ID, for local serving. Set
  `HF_HUB_OFFLINE=1` to prevent tokenizer/model metadata checks from contacting
  the Hub.
- Treat package bounds in `pyproject.toml` as the supported compatibility
  range. A machine-specific lock or install record must keep the exact
  PyTorch/CUDA/FlashAttention wheel tuple used to run a benchmark.
- The CUDA device-properties memory query is intended to work across RTX
  30/40/50 GPUs. Only the RTX 4060 Laptop has been smoke-tested here; run the
  paged-serving smoke again after changing GPU, driver, CUDA wheel, or Python.

## Review Rules

- Do not commit automatically.
- Do not delete existing code or comments without explicit approval.
- Prefer additive interfaces and English comments about ownership, shapes, and
  invariants over comments that merely repeat Python syntax.
- For Python serving, scheduling, model, or benchmark edits, give Charles the
  `scripts/check_part_b.sh` command and its expected result; do not claim it
  passed unless Charles supplies the result or an explicitly authorized run
  produced it.
