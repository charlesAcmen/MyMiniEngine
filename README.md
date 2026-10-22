# MiniEngine

A minimal LLM serving engine for educational purposes, inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang).

The engine serves an OpenAI-compatible API and is structured so that the **scheduler** is the primary optimization target across a series of assignments.

## Architecture

```
┌──────────┐     ┌─────────────┐     ┌───────────┐
│  Client   │────▶│  HTTP Server │────▶│ Scheduler  │
│ (curl/py) │◀────│  (FastAPI)   │◀────│  (Thread)  │
└──────────┘     └─────────────┘     └─────┬─────┘
                                           │
                                     ┌─────▼─────┐
                                     │   Engine   │
                                     │ (HF Model) │
                                     └───────────┘
```

| Module | File | Role |
|--------|------|------|
| **core** | `miniengine/core.py` | Data structures: `Request`, `SamplingParams`, `TokenOutput` |
| **engine** | `miniengine/engine.py` | HuggingFace model wrapper — prefill, decode, tokenize |
| **sampler** | `miniengine/sampler.py` | Top-k / top-p / temperature / repetition-penalty sampling |
| **scheduler** | `miniengine/scheduler.py` | Request scheduling loop (**optimization target**) |
| **server** | `miniengine/server.py` | OpenAI-compatible FastAPI server with SSE streaming |

## Quick Start

```bash
# Install
pip install -e .

# Launch the local model (no Hugging Face cache or network access).
python -m miniengine --model models/Qwen3-0.6B --dtype bfloat16 --max-running 16

# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
    "stream": true
  }'
```

## Benchmarks

Two benchmarks are provided. Both send requests to a **running** MiniEngine server.
```bash
python -m miniengine --model models/Qwen3-0.6B --dtype bfloat16 --page-size 32
```

`--page-size` is the number of **tokens** stored by one physical KV-cache
page (default: 32).  It is a server-startup setting: the benchmark programs
only send HTTP requests to an already-running server, so changing page size
means restarting that server with the desired flag.  `--mem-fraction-static`
(default: `0.80`) bounds model weights plus the fixed pre-allocated KV pool as
a fraction of total GPU memory.

### Serving Benchmark (`bench_serving.py`)

Measures throughput and latency under varying concurrency. Prompts are read
only from the local WildChat Parquet files in data/WildChat-1M/data/ and are
truncated/padded to the target input length. The serving benchmark has no
online dataset fallback.

```bash
python -m benchmark.bench_serving --model models/Qwen3-0.6B \
  --input-len 512 --output-len 256 --num-requests 50 \
  --input-len-min-ratio 1.0 --concurrencies 1,4,8,16
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input-len` | 512 | Target input length (tokens);|
| `--output-len` | unset | Fixed sampled output length. Enables benchmark-only `ignore_eos`, so every request performs exactly this many decode steps. Mutually exclusive with `--max-tokens`. |
| `--max-tokens` | 256 | Natural-mode output cap. The server stops earlier at EOS; use this to observe a production-like output-length distribution. |
| `--input-len-min-ratio` | 1.0 | Shortest input as a fraction of `--input-len`: `1.0` = fixed input length; `0.5` = uniform from `[50%, 100%]`; `0.0` = uniform from chat-template overhead to 100%. It never changes an explicit `--output-len`. |
| `--concurrencies` | 1,2,4,8,16,32 | Concurrency levels to sweep |
| `--num-requests` | 100 | Total requests per concurrency level |

**Reports per concurrency level:** TTFT p50/p99, completion latency p50/p99, TPOT p50/p99, generation throughput (tok/s). Completion-token metrics come from the server's final `usage.completion_tokens`, not decoded text chunks.

### Accuracy Benchmark (`bench_accuracy.py`)

Evaluates model correctness on MMLU (multiple choice) or GSM8K (math).

```bash
# MMLU — 200 questions, fast (short outputs)
python -m benchmark.bench_accuracy --dataset mmlu --num-samples 200

# GSM8K — 100 math word problems
python -m benchmark.bench_accuracy --dataset gsm8k --num-samples 100

# Filter MMLU by subject
python -m benchmark.bench_accuracy --dataset mmlu --subjects abstract_algebra,anatomy
```
