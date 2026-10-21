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
python -m miniengine --model models/Qwen3-0.6B --dtype bfloat16

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
python -m miniengine --model models/Qwen3-0.6B --dtype bfloat16
```

### Serving Benchmark (`bench_serving.py`)

Measures throughput and latency under varying concurrency. Prompts are read
only from the local WildChat Parquet files in data/WildChat-1M/data/ and are
truncated/padded to the target input length. The serving benchmark has no
online dataset fallback.

```bash
python -m benchmark.bench_serving --model models/Qwen3-0.6B \ --num-requests 50 \
  --randomness 0.5 --concurrencies 16
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input-len` | 512 | Target input length (tokens);|
| `--output-len` | 256 | Target output length (tokens); |
| `--randomness` | 1.0 | Length randomness: `1.0` = all requests use exact target lengths; `0.0` = uniform random from `[1, target]`; `0.5` = uniform from `[target/2, target]` |
| `--concurrencies` | 1,2,4,8,16,32 | Concurrency levels to sweep |
| `--num-requests` | 100 | Total requests per concurrency level |

**Reports per concurrency level:** TTFT p50/p99, completion latency p50/p99, TPOT p50/p99, generation throughput (tok/s).

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
