# MiniEngine Learning Contract

- origin is CharlesAcmen/MyMiniEngine. Do not rewrite history, reset, push, or
  commit unless Charles explicitly asks.
- Stay on the independent M2 learning path. Never inspect, copy, checkout, or
  compare upstream M2-M4 reference implementations.
- The current code includes paged continuous batching. Preserve its scheduler,
  request-state, and paged-KV ownership invariants unless the task explicitly
  changes them.
- Use the local `models/Qwen3-0.6B` directory for smoke/debug. Set
  `HF_HUB_OFFLINE=1` for local serving; do not make an online Hub lookup part
  of the smoke path.
- Make one bounded, reviewable step at a time. First explain state transitions,
  tensor shapes, GPU-memory ownership, and invariants in GPU-system terms.
- Preserve old code and learning comments. Add concise English comments for
  non-obvious interfaces. Ask before deleting or materially replacing code.
- Keep smoke behavior, accuracy, and performance evidence separate. Qwen3-0.6B
  is the local debug model; final 8B performance belongs on L4 or A6000.
- CUDA portability: obtain static VRAM capacity through CUDA device properties,
  not `torch.accelerator`. This keeps the code path compatible with NVIDIA
  RTX 30/40/50 families. A binary FlashAttention wheel must still match the
  installed Python, PyTorch, CUDA, and C++ ABI, so each GPU family needs its
  own paged-serving smoke test before being called verified.
- After changing Python serving, scheduling, model, or benchmark code, ask the
  user to run `scripts/check_part_b.sh` before reporting completion. It is the
  intended local gate for unit tests, paged serving smoke, and MMLU accuracy
  smoke.
