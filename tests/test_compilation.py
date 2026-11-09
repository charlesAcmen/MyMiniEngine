"""Correctness tests for the pure-tensor paged decode compile boundary."""

from __future__ import annotations

import unittest

import torch

from miniengine.compilation import build_paged_decode_ops
from miniengine.config import CompileConfig
from miniengine.cuda_graph import select_graph_bucket
from miniengine.model import (
    ModelConfig,
    PagedAttentionOps,
    TransformerBlock,
    apply_rotary_emb,
)


class PagedDecodeCompilationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.config = ModelConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            tie_word_embeddings=False,
        )

    @staticmethod
    def _identity_attention(q, _k_cache, _v_cache, **_kwargs):
        return q

    def test_refactored_decode_block_matches_original_math(self) -> None:
        block = TransformerBlock(self.config).eval()
        block.self_attn.set_paged_ops(
            PagedAttentionOps(
                prefill=lambda *args, **kwargs: None,
                decode=self._identity_attention,
            )
        )
        hidden = torch.randn(3, 1, self.config.hidden_size)
        position_ids = torch.tensor([[0], [1], [2]])
        cos, sin = torch.cos(position_ids.float()).view(3, 1, 1, 1), torch.sin(
            position_ids.float()
        ).view(3, 1, 1, 1)
        cos = cos.expand(-1, -1, -1, self.config.head_dim)
        sin = sin.expand_as(cos)
        cache = torch.zeros(1)
        cache_seqlens = torch.zeros(3, dtype=torch.int32)
        block_table = torch.zeros((3, 1), dtype=torch.int32)

        actual = block.forward_paged_decode(
            hidden, cos, sin, cache, cache, cache_seqlens, block_table
        )

        normalized = block.input_layernorm(hidden)
        q = block.self_attn.q_proj(normalized).view(3, 1, 2, 4).transpose(1, 2)
        q = apply_rotary_emb(block.self_attn.q_norm(q), cos, sin)
        attention_output = q.transpose(1, 2).contiguous().reshape(3, 1, 8)
        residual = hidden + block.self_attn.o_proj(attention_output)
        expected = residual + block.mlp(block.post_attention_layernorm(residual))

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_compiled_callables_preserve_eager_results(self) -> None:
        eager = build_paged_decode_ops(CompileConfig())
        compiled = build_paged_decode_ops(
            CompileConfig(enabled=True, backend="eager", mode="default")
        )
        block = TransformerBlock(self.config).eval()
        hidden = torch.randn(2, 1, self.config.hidden_size)
        cos = torch.randn(2, 1, 1, self.config.head_dim)
        sin = torch.randn_like(cos)
        pre_args = (
            hidden,
            cos,
            sin,
            block.input_layernorm.weight,
            block.self_attn.q_proj.weight,
            block.self_attn.k_proj.weight,
            block.self_attn.v_proj.weight,
            block.self_attn.q_norm.weight,
            block.self_attn.k_norm.weight,
            self.config.rms_norm_eps,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        expected_pre = eager.pre_attention(*pre_args)
        actual_pre = compiled.pre_attention(*pre_args)
        for actual, expected in zip(actual_pre, expected_pre):
            torch.testing.assert_close(actual, expected)

        attention_output = torch.randn(2, 1, 2, 4)
        post_args = (
            attention_output,
            hidden,
            block.self_attn.o_proj.weight,
            block.post_attention_layernorm.weight,
            block.mlp.gate_proj.weight,
            block.mlp.up_proj.weight,
            block.mlp.down_proj.weight,
            self.config.rms_norm_eps,
        )
        torch.testing.assert_close(
            compiled.post_attention(*post_args), eager.post_attention(*post_args)
        )

    def test_graph_bucket_selects_smallest_fitting_shape(self) -> None:
        buckets = (1, 3, 7)
        self.assertEqual(select_graph_bucket(1, buckets), 1)
        self.assertEqual(select_graph_bucket(2, buckets), 3)
        self.assertEqual(select_graph_bucket(7, buckets), 7)
        with self.assertRaises(ValueError):
            select_graph_bucket(8, buckets)

    def test_compiled_region_bounds_graphs_across_batch_sizes(self) -> None:
        compile_count = 0

        def counting_backend(graph, _example_inputs):
            nonlocal compile_count
            compile_count += 1
            return graph.forward

        torch._dynamo.reset()
        compiled = build_paged_decode_ops(
            CompileConfig(enabled=True, backend=counting_backend)
        )
        block = TransformerBlock(self.config).eval()
        for batch_size in range(1, 13):
            hidden = torch.randn(batch_size, 1, self.config.hidden_size)
            cos = torch.randn(batch_size, 1, 1, self.config.head_dim)
            sin = torch.randn_like(cos)
            compiled.pre_attention(
                hidden,
                cos,
                sin,
                block.input_layernorm.weight,
                block.self_attn.q_proj.weight,
                block.self_attn.k_proj.weight,
                block.self_attn.v_proj.weight,
                block.self_attn.q_norm.weight,
                block.self_attn.k_norm.weight,
                self.config.rms_norm_eps,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )

        # Dynamo specializes batch size one, then reuses one symbolic graph for
        # every larger batch instead of compiling each continuous-batch shape.
        self.assertEqual(compile_count, 2)


if __name__ == "__main__":
    unittest.main()
