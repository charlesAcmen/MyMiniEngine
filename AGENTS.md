# MiniEngine Learning Contract

- Work only in the ubuntu26 ext4 checkout at
  /home/charlesAcmen/dev/repos/CS349D-miniengine.
- origin is CharlesAcmen/MyMiniEngine. Do not rewrite history, reset, push, or
  commit unless Charles explicitly asks.
- Stay on the independent M1 learning path. Never inspect, copy, checkout, or
  compare upstream M2-M4 reference implementations.
- Current baseline is m1-foundation at 6062777: one request runs to completion
  before the next request is scheduled.
- Use Qwen/Qwen3-0.6B for local smoke/debug. It is in the default Hugging Face
  cache. Use HF_HUB_OFFLINE=1 for local serving to avoid online tokenizer checks.
- Make one bounded, reviewable step at a time. First explain state transitions,
  tensor shapes, GPU-memory ownership, and invariants in GPU-system terms.
- Preserve old code and learning comments. Add concise English comments for
  non-obvious interfaces. Ask before deleting or materially replacing code.
- Keep smoke behavior, accuracy, and performance evidence separate. RTX 3060
  12GB is for Qwen3-0.6B debug; final 8B performance belongs on L4 or A6000.
- After changing Python serving, scheduling, model, or benchmark code, run
  `scripts/check_part_b.sh` before reporting completion. It is the automatic
  local gate for unit tests, paged serving smoke, and MMLU accuracy smoke.
