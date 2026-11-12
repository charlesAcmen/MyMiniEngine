"""Tests for tensor math shared by eager and compiled model paths."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from miniengine.functional import apply_rotary_emb, rms_norm, swiglu_mlp


class FunctionalCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_rms_norm_matches_reference_math(self) -> None:
        value = torch.randn(2, 3, 8)
        weight = torch.randn(8)
        eps = 1e-6

        expected = (
            value
            * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
            * weight
        )

        torch.testing.assert_close(rms_norm(value, weight, eps), expected)

    def test_rotary_embedding_matches_reference_math(self) -> None:
        value = torch.randn(2, 3, 4, 8)
        cos = torch.randn(2, 3, 1, 8)
        sin = torch.randn_like(cos)
        first, second = value.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)

        expected = value * cos + rotated * sin

        torch.testing.assert_close(apply_rotary_emb(value, cos, sin), expected)

    def test_swiglu_mlp_matches_reference_math(self) -> None:
        value = torch.randn(2, 3, 8)
        gate_weight = torch.randn(16, 8)
        up_weight = torch.randn(16, 8)
        down_weight = torch.randn(8, 16)

        expected = F.linear(
            F.silu(F.linear(value, gate_weight))
            * F.linear(value, up_weight),
            down_weight,
        )

        torch.testing.assert_close(
            swiglu_mlp(value, gate_weight, up_weight, down_weight), expected
        )


if __name__ == "__main__":
    unittest.main()
