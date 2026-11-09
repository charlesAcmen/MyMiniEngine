"""Immutable service configuration owned by a MiniEngine instance."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SamplingConfig:
    """One immutable token-sampling policy shared by a server instance."""

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0

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
    def from_json(cls, path: str | Path) -> SamplingConfig:
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


@dataclass(frozen=True)
class CompileConfig:
    """Stable torch.compile policy for paged decode tensor regions."""

    enabled: bool = False
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = True
    dynamic_batch: bool = True
    target: str = "paged_decode_tensor_regions"


@dataclass(frozen=True)
class CudaGraphConfig:
    """Manual CUDA graph policy for fixed-size paged decode buckets."""

    enabled: bool = False
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    def __post_init__(self) -> None:
        if not self.batch_sizes:
            raise ValueError("CUDA graph batch sizes must not be empty")
        if any(size < 1 for size in self.batch_sizes):
            raise ValueError("CUDA graph batch sizes must be positive")
        if tuple(sorted(set(self.batch_sizes))) != self.batch_sizes:
            raise ValueError(
                "CUDA graph batch sizes must be unique and strictly increasing"
            )


@dataclass(frozen=True)
class EngineConfig:
    """Serializable model-execution and accelerator-resource policy."""

    mode: str = "batched"
    dtype: str = "bfloat16"
    device: str = "cuda"
    page_size: int = 256
    mem_fraction_static: float = 0.80
    compile: CompileConfig = field(default_factory=CompileConfig)
    cuda_graph: CudaGraphConfig = field(default_factory=CudaGraphConfig)

    def __post_init__(self) -> None:
        if self.mode not in {"batched", "paged"}:
            raise ValueError(f"unsupported engine mode: {self.mode}")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.page_size < 1:
            raise ValueError("page_size must be positive")
        if not 0 < self.mem_fraction_static < 1:
            raise ValueError("mem_fraction_static must be in (0, 1)")
        if self.compile.enabled and self.mode != "paged":
            raise ValueError("torch.compile is supported only in paged mode")
        if self.compile.enabled and not self.device.startswith("cuda"):
            raise ValueError("torch.compile paged decode requires a CUDA device")
        if self.cuda_graph.enabled and not self.compile.enabled:
            raise ValueError("CUDA graph mode requires torch.compile")
