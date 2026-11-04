"""
Core data structures for the serving engine.

These structures flow through the entire system:
  Client request → Server → Scheduler → Engine → Scheduler → Server → Client response

Key types:
  - SamplingConfig: server-wide token generation policy
  - Request: tracks the full lifecycle of a single generation request
  - TokenOutput: a single streamed token delivered to the client
"""

import json
import math
import queue
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


class RequestStatus(Enum):
    """Lifecycle states for a generation request."""

    WAITING = auto()  # Queued, not yet scheduled
    RUNNING = auto()  # Actively being prefilled or decoded
    FINISHED = auto()  # Generation complete


@dataclass(frozen=True)
class SamplingConfig:
    """One immutable token-sampling policy shared by a server instance."""

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0

    #hook:validate for every source of sampling config,including test,server config and request config
    def __post_init__(self) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a finite number >= 0")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be a finite number in (0, 1]")
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 0
        ):
            raise ValueError("top_k must be a non-negative integer")
        if (
            isinstance(self.repetition_penalty, bool)
            or not isinstance(self.repetition_penalty, (int, float))
            or not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0
        ):
            raise ValueError("repetition_penalty must be a finite number > 0")

    @classmethod
    def from_json(cls, path: str | Path) -> "SamplingConfig":
        """Load a service sampling policy from a JSON object."""
        config_path = Path(path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot load sampling config {config_path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError("sampling config must be a JSON object")
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ValueError(f"invalid sampling config fields: {exc}") from exc

    def validate_vocab_size(self, vocab_size: int) -> None:
        """Reject a finite top-k that is equivalent to the full vocabulary."""
        if self.top_k >= vocab_size and self.top_k != 0:
            raise ValueError(
                f"top_k must be 0 or smaller than vocab_size ({vocab_size})"
            )


@dataclass
class Request:
    """
    A single generation request flowing through the engine.

    The scheduler moves requests through WAITING → RUNNING → FINISHED.
    During RUNNING, the engine fills output_ids one token at a time and
    pushes TokenOutput objects into token_queue for streaming.
    """

    request_id: str
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    max_new_tokens: int = 256
    # Benchmark-only: continue to the length limit even after an EOS token.
    ignore_eos: bool = False
    status: RequestStatus = RequestStatus.WAITING
    arrival_time: float = field(default_factory=time.time)

    # Per-request KV cache (set by Engine during prefill, updated on each decode)
    kv_cache: Any = None

    # M2 paged-KV ownership. ``page_table`` maps logical pages for this
    # request to physical KVMemoryPool page indices; ``kv_seq_len`` counts
    # committed token slots in those pages.
    page_table: list[int] = field(default_factory=list)
    kv_seq_len: int = 0

    # Streaming output channel — scheduler pushes, server consumes
    token_queue: queue.Queue = field(default_factory=queue.Queue)

    # ── Derived properties ─────────────────────────────────────────────

    @property
    def num_input_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_ids)

    @property
    def is_finished(self) -> bool:
        return self.num_output_tokens >= self.max_new_tokens


@dataclass
class TokenOutput:
    """One streamed token pushed from the scheduler to the server."""

    token_id: int
    token_text: str
    finished: bool
