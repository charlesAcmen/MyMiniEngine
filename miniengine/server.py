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
from dataclasses import asdict
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from tokenizers.decoders import DecodeStream

from miniengine.config import SamplingConfig
from miniengine.core import Request, TokenOutput
from miniengine.engine import Engine
from miniengine.scheduler import Scheduler

logger = logging.getLogger(__name__)

# ── FastAPI app ─────────────────────────────────────────────────────────

app = FastAPI(title="MiniEngine", version="0.1.0")

# These are set by __main__.py before the server starts.
engine: Engine | None = None
scheduler: Scheduler | None = None
model_id: str = "unknown"


def _require_runtime() -> tuple[Engine, Scheduler]:
    """Return the initialized serving runtime or reject premature traffic."""
    if engine is None or scheduler is None:
        raise HTTPException(status_code=503, detail="service is not initialized")
    return engine, scheduler

# ── Request / Response schemas (OpenAI-compatible subset) ───────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    #ge: greater than or equal to, gt: greater than, le: less than or equal to
    max_tokens: int | None = Field(default=256, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    repetition_penalty: float | None = Field(default=None, gt=0)
    # Kept false for normal chat.  The serving benchmark enables it to make
    # every request execute an exactly controlled number of decode steps.
    ignore_eos: bool = False
    stream: bool = False


# ── Endpoints ───────────────────────────────────────────────────────────


def validate_sampling_overrides(
    raw: ChatCompletionRequest, config: SamplingConfig
) -> None:
    """Reject request-level sampling changes without silently ignoring them."""
    for field in ("temperature", "top_p", "top_k", "repetition_penalty"):
        requested = getattr(raw, field)
        configured = getattr(config, field)
        if requested is not None and requested != configured:
            raise HTTPException(
                status_code=422,
                detail=f"{field} is fixed by this service at {configured}",
            )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    current_engine, current_scheduler = _require_runtime()
    engine_config = asdict(current_engine.config)
    engine_config["max_running"] = current_scheduler.max_running
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "miniengine",
                "sampling_config": asdict(current_engine.sampling_config),
                "engine_config": engine_config,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(raw: ChatCompletionRequest):
    current_engine, current_scheduler = _require_runtime()
    validate_sampling_overrides(raw, current_engine.sampling_config)

    # Tokenize the conversation using the model's chat template
    messages = [{"role": m.role, "content": m.content} for m in raw.messages]
    input_ids = current_engine.tokenize_messages(messages)

    req = Request(
        request_id=str(uuid.uuid4()),
        input_ids=input_ids,
        max_new_tokens=raw.max_tokens if raw.max_tokens is not None else 256,
        ignore_eos=raw.ignore_eos,
    )
    current_scheduler.add_request(req)

    if raw.stream:
        return StreamingResponse(
            _stream_response(req, raw.model or model_id, current_engine),
            media_type="text/event-stream",
        )

    # Non-streaming: collect full response
    full_text = await _collect_full_response(req, current_engine)
    return _make_completion_response(req, raw.model or model_id, full_text)


# ── Streaming helpers ───────────────────────────────────────────────────


async def _stream_response(
    req: Request, model: str, response_engine: Engine
) -> AsyncGenerator[str, None]:
    """Yield SSE chunks as tokens arrive from the scheduler."""
    loop = asyncio.get_event_loop()
    decode_stream: DecodeStream | None = None
    while True:
        output: TokenOutput = await loop.run_in_executor(
            None, req.token_queue.get
        )
        if output.error is not None:
            error = {
                "error": {
                    "message": output.error,
                    "type": "server_error",
                    "request_id": req.request_id,
                }
            }
            yield f"data: {json.dumps(error)}\n\n"
            yield "data: [DONE]\n\n"
            return
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
        # Byte-fallback tokenizers may need several token ids before a complete
        # Unicode fragment is available for the HTTP stream.
        if decode_stream is None:
            decode_stream = DecodeStream([], skip_special_tokens=True)
        text = decode_stream.step(
            response_engine.tokenizer._tokenizer, output.token_id
        )
        if text is None:
            continue
        chunk = _make_stream_chunk(req.request_id, model, content=text)
        yield f"data: {json.dumps(chunk)}\n\n"


async def _collect_full_response(req: Request, response_engine: Engine) -> str:
    """Block until the request finishes and return the full text."""
    loop = asyncio.get_event_loop()
    while True:
        output: TokenOutput = await loop.run_in_executor(
            None, req.token_queue.get
        )
        if output.error is not None:
            raise HTTPException(status_code=500, detail=output.error)
        if output.finished:
            break
    return response_engine.decode_tokens(req.output_ids)


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
