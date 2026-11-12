"""Stable pure-tensor regions used by paged decode and torch.compile."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from miniengine.config import CompileConfig
from miniengine.functional import apply_rotary_emb, rms_norm, swiglu_mlp


def paged_decode_pre_attention(
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    input_norm_weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize, project, and rotate one paged-decode token per row."""
    batch_size, query_len, _ = hidden.shape
    normalized = rms_norm(hidden, input_norm_weight, eps)
    q = F.linear(normalized, q_weight).view(
        batch_size, query_len, num_heads, head_dim
    )
    k = F.linear(normalized, k_weight).view(
        batch_size, query_len, num_kv_heads, head_dim
    )
    v = F.linear(normalized, v_weight).view(
        batch_size, query_len, num_kv_heads, head_dim
    )

    q = apply_rotary_emb(rms_norm(q, q_norm_weight, eps), cos, sin)
    k = apply_rotary_emb(rms_norm(k, k_norm_weight, eps), cos, sin)
    return q.contiguous(), k.contiguous(), v.contiguous()


def paged_decode_post_attention(
    attention_output: torch.Tensor,
    residual: torch.Tensor,
    output_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply the attention projection, residual, norm, and SwiGLU MLP."""
    batch_size, query_len, _, _ = attention_output.shape
    projected = F.linear(
        attention_output.reshape(batch_size, query_len, -1), output_weight
    )
    hidden = residual + projected
    normalized = rms_norm(hidden, post_norm_weight, eps)
    return hidden + swiglu_mlp(
        normalized, gate_weight, up_weight, down_weight
    )


@dataclass(frozen=True)
class PagedDecodeOps:
    """One pair of eager or compiled pure-tensor decode operations."""

    pre_attention: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    post_attention: Callable[..., torch.Tensor]


EAGER_PAGED_DECODE_OPS = PagedDecodeOps(
    pre_attention=paged_decode_pre_attention,
    post_attention=paged_decode_post_attention,
)


def build_paged_decode_ops(config: CompileConfig) -> PagedDecodeOps:
    """Build the fixed paged-decode execution policy once at startup."""
    if not config.enabled:
        return EAGER_PAGED_DECODE_OPS

    compile_options = {
        "backend": config.backend,
        "fullgraph": config.fullgraph,
        "dynamic": config.dynamic_batch,
    }
    if config.backend == "inductor":
        # Manual CudaGraphRunner owns graph capture so compile-only and graph
        # modes remain separately measurable.
        compile_options["options"] = {"triton.cudagraphs": False}
    return PagedDecodeOps(
        pre_attention=torch.compile(paged_decode_pre_attention, **compile_options),
        post_attention=torch.compile(paged_decode_post_attention, **compile_options),
    )
