"""CPU tests for the Part-B hot-path optimizations."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from miniengine.core import SamplingConfig
from miniengine.model import CausalLM, ModelConfig, RotaryEmbedding
from miniengine.sampler import (
    apply_repetition_penalty_batch,
    sample_batch,
    sample_token,
)


class BatchSamplingTests(unittest.TestCase):
    def test_pure_greedy_batch_uses_one_tensor_to_list(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 2.0], [8.0, 3.0, 5.0]])
        config = SamplingConfig(temperature=0)

        self.assertEqual(sample_batch(logits, config, [[], [1]]), [1, 0])

    def test_ragged_repetition_penalty_keeps_rows_independent(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 2.0], [8.0, 3.0, 5.0]])
        original = logits.clone()

        adjusted = apply_repetition_penalty_batch(
            logits, [[1, 1], [0, 2]], penalty=2.0
        )

        torch.testing.assert_close(logits, original)
        torch.testing.assert_close(
            adjusted, torch.tensor([[1.0, 2.0, 2.0], [4.0, 3.0, 2.5]])
        )

    def test_single_sampler_does_not_clone_without_penalty(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 2.0]])
        with patch.object(torch.Tensor, "clone", side_effect=AssertionError):
            token = sample_token(logits, SamplingConfig(temperature=0), [1])
        self.assertEqual(token, 1)

    def test_single_sampler_clones_when_penalty_is_active(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 2.0]])
        original = logits.clone()

        token = sample_token(
            logits,
            SamplingConfig(temperature=0, repetition_penalty=10.0),
            [1],
        )

        self.assertEqual(token, 2)
        torch.testing.assert_close(logits, original)

    def test_top_p_paths_launch_softmax_once(self) -> None:
        logits = torch.tensor([[10.0, 9.0, 1.0], [8.0, 7.0, 2.0]])
        config = SamplingConfig(temperature=1.0, top_k=3, top_p=0.8)
        real_softmax = torch.softmax

        with patch("torch.softmax", wraps=real_softmax) as single_softmax:
            sample_token(logits[:1], config)
        self.assertEqual(single_softmax.call_count, 1)

        with patch("torch.softmax", wraps=real_softmax) as batch_softmax:
            sample_batch(logits, config, [[], []])
        self.assertEqual(batch_softmax.call_count, 1)

    def test_batch_sampler_performs_one_host_transfer(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 2.0], [8.0, 3.0, 5.0]])
        with patch.object(
            torch.Tensor,
            "tolist",
            return_value=[1, 0],
        ) as tolist:
            tokens = sample_batch(logits, SamplingConfig(temperature=0), [[], []])

        self.assertEqual(tokens, [1, 0])
        self.assertEqual(tolist.call_count, 1)

    def test_batch_penalty_with_all_empty_histories_does_not_clone(self) -> None:
        logits = torch.tensor([[1.0, 4.0], [8.0, 3.0]])
        with patch.object(torch.Tensor, "clone", side_effect=AssertionError):
            adjusted = apply_repetition_penalty_batch(logits, [[], []], 2.0)
        self.assertIs(adjusted, logits)

    def test_batch_sampling_maps_candidates_back_to_vocab_ids(self) -> None:
        logits = torch.tensor([[1.0, 8.0, 3.0, 7.0], [9.0, 2.0, 8.0, 1.0]])
        config = SamplingConfig(temperature=1.0, top_k=2, top_p=1.0)
        chosen_candidates = torch.tensor([[1], [0]])

        with patch("torch.multinomial", return_value=chosen_candidates):
            tokens = sample_batch(logits, config, [[], []])

        self.assertEqual(tokens, [3, 0])

    def test_top_k_runs_before_temperature_scaling(self) -> None:
        logits = torch.tensor([[1.0, 8.0, 3.0, 7.0]])
        config = SamplingConfig(temperature=0.5, top_k=2, top_p=1.0)

        with patch("torch.topk", wraps=torch.topk) as topk:
            sample_batch(logits, config, [[]])

        self.assertIs(topk.call_args.args[0], logits)

    def test_top_k_zero_maps_sorted_candidate_back_to_vocab_id(self) -> None:
        logits = torch.tensor([[1.0, 8.0, 3.0, 7.0]])
        config = SamplingConfig(temperature=1.0, top_k=0, top_p=0.95)

        with patch("torch.multinomial", return_value=torch.tensor([[2]])):
            tokens = sample_batch(logits, config, [[]])

        self.assertEqual(tokens, [2])

    def test_penalty_changes_the_top_k_candidate(self) -> None:
        logits = torch.tensor([[10.0, 9.0, 1.0]])
        config = SamplingConfig(
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=2.0,
        )

        self.assertEqual(sample_batch(logits, config, [[0]]), [1])

    def test_single_and_batch_top_k_keep_exactly_k_tied_candidates(self) -> None:
        logits = torch.tensor([[10.0, 9.0, 9.0, 9.0]])
        config = SamplingConfig(temperature=1.0, top_k=2, top_p=1.0)
        captured_shapes = []

        def choose_first(probs: torch.Tensor, num_samples: int) -> torch.Tensor:
            captured_shapes.append(tuple(probs.shape))
            return torch.zeros((probs.shape[0], num_samples), dtype=torch.long)

        with patch("torch.multinomial", side_effect=choose_first):
            sample_token(logits, config)
            sample_batch(logits, config, [[]])

        self.assertEqual(captured_shapes, [(1, 2), (1, 2)])

    def test_top_p_never_removes_the_highest_candidate(self) -> None:
        logits = torch.tensor([[10.0, 9.0, 1.0]])
        config = SamplingConfig(temperature=1.0, top_k=3, top_p=0.01)

        for _ in range(10):
            self.assertEqual(sample_batch(logits, config, [[]]), [0])


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
