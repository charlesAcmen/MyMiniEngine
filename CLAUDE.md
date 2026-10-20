# MiniEngine Project Notes

Use the rules in AGENTS.md.

## Current M1 State

- Baseline: single-request FCFS, then per-request KV-cache release.
- Target: batched decode followed by iteration-level continuous batching.
- First reviewable change: optional attention_mask from CausalLM to Attention.
  None preserves the single-request path. Later batched decode builds a
  boolean mask shaped (batch, 1, 1, kv_len); True means visible KV data.

## Environment

- Work in ubuntu26, never the ubuntu22 research environment.
- The editable project .venv has CUDA PyTorch.
- WSL proxy dynamically uses the Windows gateway on port 7892.
- Qwen3-0.6B is in the default cache. Use HF_HUB_OFFLINE=1 for serving.

## Review Rules

- Do not commit automatically.
- Do not delete existing code or comments without explicit approval.
- Prefer additive interfaces and English comments about ownership, shapes, and
  invariants over comments that merely repeat Python syntax.
