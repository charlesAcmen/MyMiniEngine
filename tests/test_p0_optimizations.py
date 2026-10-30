"""CPU tests for the Part-B hot-path optimizations."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from miniengine.core import SamplingParams
from miniengine.model import CausalLM, ModelConfig, RotaryEmbedding
from miniengine.sampler import sample_batch, sample_token


class BatchSamplingTests(unittest.TestCase):
    def test_pure_greedy_batch_uses_one_tensor_to_list(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 2.0], [8.0, 3.0, 5.0]])
        params = [SamplingParams(temperature=0), SamplingParams(temperature=0)]

        self.assertEqual(sample_batch(logits, params, [[], [1]]), [1, 0])

    def test_repetition_penalty_keeps_per_request_fallback(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 2.0], [8.0, 3.0, 5.0]])
        params = [
            SamplingParams(temperature=0),
            SamplingParams(temperature=0, repetition_penalty=2.0),
        ]

        with patch(
            "miniengine.sampler.sample_token", wraps=sample_token
        ) as single_sampler:
            tokens = sample_batch(logits, params, [[], [0]])

        self.assertEqual(tokens, [1, 2])
        self.assertEqual(single_sampler.call_count, 2)


class PrefillProjectionTests(unittest.TestCase):
    def test_paged_prefill_projects_only_segment_ends(self) -> None:
        config = ModelConfig(
            vocab_size=7,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            tie_word_embeddings=False,
        )
        model = CausalLM(config).eval()
        packed_hidden = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

        with patch.object(
            model.model,
            "forward_paged_prefill",
            return_value=packed_hidden,
        ):
            logits = model.forward_paged_prefill(
                input_ids=torch.zeros(8, dtype=torch.long),
                position_ids=torch.zeros(8, dtype=torch.long),
                kv_caches=[],
                slot_mapping=torch.zeros(8, dtype=torch.long),
                cu_seqlens=cu_seqlens,
                max_seqlen=5,
                flash_attn_varlen_func=lambda *args, **kwargs: None,
            )

        expected = model._project_logits(packed_hidden[[2, 7]])
        self.assertEqual(logits.shape, (2, config.vocab_size))
        torch.testing.assert_close(logits, expected)


class RotaryCacheTests(unittest.TestCase):
    def test_cpu_cache_bound_avoids_tensor_item(self) -> None:
        rotary = RotaryEmbedding(head_dim=4)
        position_ids = torch.tensor([[2], [5]])

        with patch.object(torch.Tensor, "item", side_effect=AssertionError):
            cos, sin = rotary(position_ids, required_cache_len=6)

        self.assertEqual(cos.shape, (2, 1, 1, 4))
        self.assertEqual(sin.shape, (2, 1, 1, 4))
        self.assertGreaterEqual(rotary._cached_len, 6)


if __name__ == "__main__":
    unittest.main()
