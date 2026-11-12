"""Shared stateless tensor math for eager and compiled model paths."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm(
    value: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Apply RMS normalization over the last dimension."""
    return value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps) * weight


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    """Rotate split halves of the last dimension for RoPE."""
    first, second = value.chunk(2, dim=-1)
    # x'=xcos(theta)-ysin(theta)
    # y'=xsin(theta)+ycos(theta)
    #[x y]T cos(theta)+[-y x]T sin(theta)
    return torch.cat((-second, first), dim=-1)

def apply_rotary_emb(
    value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE using cos and sin already aligned for broadcasting."""
    return value * cos + rotate_half(value) * sin


def swiglu_mlp(
    value: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply a bias-free SwiGLU MLP from explicit projection weights."""
    gated = F.silu(F.linear(value, gate_weight)) * F.linear(value, up_weight)
    return F.linear(gated, down_weight)
