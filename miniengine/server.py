"""
OpenAI-compatible HTTP server.

Endpoints:
  POST /v1/chat/completions   — chat completion (streaming & non-streaming)
  GET  /v1/models             — list available models
  GET  /health                — liveness check

The server is intentionally thin — it only translates HTTP ↔ internal
Request objects.  All scheduling and model logic lives elsewhere.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from miniengine.core import Request, SamplingParams, TokenOutput
from miniengine.engine import Engine
from miniengine.scheduler import Scheduler

logger = logging.getLogger(__name__)

# ── FastAPI app ─────────────────────────────────────────────────────────

app = FastAPI(title="MiniEngine", version="0.1.0")

# These are set by __main__.py before the server starts.
engine: Engine | None = None
scheduler: Scheduler | None = None
model_id: str = "unknown"

# ── Request / Response schemas (OpenAI-compatible subset) ───────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=256)
    temperature: float | None = Field(default=0.6)
    top_p: float | None = Field(default=0.95)
    top_k: int | None = Field(default=20)
    repetition_penalty: float | None = Field(default=1.0)
    # Kept false for normal chat.  The serving benchmark enables it to make
    # every request execute an exactly controlled number of decode steps.
    ignore_eos: bool = False
    stream: bool = False


# ── Endpoints ───────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "miniengine",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(raw: ChatCompletionRequest):
    assert engine is not None and scheduler is not None

    # Tokenize the conversation using the model's chat template
    messages = [{"role": m.role, "content": m.content} for m in raw.messages]
    input_ids = engine.tokenize_messages(messages)

    sampling_params = SamplingParams(
        max_new_tokens=raw.max_tokens if raw.max_tokens is not None else 256,
        temperature=raw.temperature if raw.temperature is not None else 0.6,
        top_p=raw.top_p if raw.top_p is not None else 0.95,
        top_k=raw.top_k if raw.top_k is not None else 20,
        repetition_penalty=(
            raw.repetition_penalty if raw.repetition_penalty is not None else 1.0
        ),
        ignore_eos=raw.ignore_eos,
    )

    req = Request(
        request_id=str(uuid.uuid4()),
        input_ids=input_ids,
        sampling_params=sampling_params,
    )
    scheduler.add_request(req)

    if raw.stream:
        return StreamingResponse(
            _stream_response(req, raw.model or model_id),
            media_type="text/event-stream",
        )

    # Non-streaming: collect full response
    full_text = await _collect_full_response(req)
    return _make_completion_response(req, raw.model or model_id, full_text)


# ── Streaming helpers ───────────────────────────────────────────────────


async def _stream_response(
    req: Request, model: str
) -> AsyncGenerator[str, None]:
    """Yield SSE chunks as tokens arrive from the scheduler."""
    loop = asyncio.get_event_loop()
    while True:
        output: TokenOutput = await loop.run_in_executor(
            None, req.token_queue.get
        )
        if output.finished:
            chunk = _make_stream_chunk(
                req.request_id,
                model,
                content="",
                finish_reason="length" if req.is_finished else "stop",
                usage=_usage_for(req),
            )
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return
        chunk = _make_stream_chunk(req.request_id, model, content=output.token_text)
        yield f"data: {json.dumps(chunk)}\n\n"


async def _collect_full_response(req: Request) -> str:
    """Block until the request finishes and return the full text."""
    loop = asyncio.get_event_loop()
    parts: list[str] = []
    while True:
        output: TokenOutput = await loop.run_in_executor(
            None, req.token_queue.get
        )
        if output.finished:
            break
        parts.append(output.token_text)
    return "".join(parts)


# ── Response builders ───────────────────────────────────────────────────


def _make_stream_chunk(
    request_id: str,
    model: str,
    content: str,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict:
    chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _usage_for(req: Request) -> dict[str, int]:
    return {
        "prompt_tokens": req.num_input_tokens,
        "completion_tokens": req.num_output_tokens,
        "total_tokens": req.num_input_tokens + req.num_output_tokens,
    }


def _make_completion_response(req: Request, model: str, text: str) -> dict:
    return {
        "id": f"chatcmpl-{req.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length" if req.is_finished else "stop",
            }
        ],
        "usage": _usage_for(req),
    }
