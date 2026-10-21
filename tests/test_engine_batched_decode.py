"""CPU correctness tests for M1's Engine.batched_decode()."""

import unittest

import torch

from miniengine.core import Request, SamplingParams
from miniengine.engine import Engine
from miniengine.model import CausalLM, ModelConfig


class BatchedDecodeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        config = ModelConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            tie_word_embeddings=False,
        )
        self.engine = Engine.__new__(Engine)
        self.engine.device = "cpu"
        self.engine.dtype = torch.float32
        self.engine.model = CausalLM(config).eval()

    @staticmethod
    def _clone_request(source: Request) -> Request:
        clone = Request(
            request_id=f"clone-{source.request_id}",
            input_ids=list(source.input_ids),
            output_ids=list(source.output_ids),
            sampling_params=SamplingParams(temperature=0),
        )
        clone.kv_cache = [
            (key.clone(), value.clone()) for key, value in source.kv_cache
        ]
        return clone

    def _prefilled_request(self, request_id: str, input_ids: list[int]) -> Request:
        request = Request(
            request_id=request_id,
            input_ids=input_ids,
            sampling_params=SamplingParams(temperature=0),
        )
        request.output_ids.append(self.engine.prefill(request))
        return request

    def test_empty_batch(self) -> None:
        self.assertEqual(self.engine.batched_decode([]), [])

    def test_heterogeneous_cache_parity_and_compact_storage(self) -> None:
        batched_requests = [
            self._prefilled_request("short", [2, 3]),
            self._prefilled_request("long", [4, 5, 6, 7]),
        ]
        cache_lengths = [
            request.kv_cache[0][0].shape[2] for request in batched_requests
        ]
        self.assertLess(cache_lengths[0], cache_lengths[1])

        independent_requests = [
            self._clone_request(request) for request in batched_requests
        ]
        expected_tokens = [
            self.engine.decode_step(request) for request in independent_requests
        ]
        actual_tokens = self.engine.batched_decode(batched_requests)
        self.assertEqual(actual_tokens, expected_tokens)

        for batched, independent, old_length in zip(
            batched_requests, independent_requests, cache_lengths
        ):
            for (batched_key, batched_value), (
                independent_key,
                independent_value,
            ) in zip(batched.kv_cache, independent.kv_cache):
                self.assertEqual(
                    batched_key.shape, (1, 2, old_length + 1, 4)
                )
                self.assertEqual(
                    batched_value.shape, (1, 2, old_length + 1, 4)
                )
                self.assertEqual(
                    batched_key.untyped_storage().nbytes(),
                    batched_key.numel() * batched_key.element_size(),
                )
                self.assertEqual(
                    batched_value.untyped_storage().nbytes(),
                    batched_value.numel() * batched_value.element_size(),
                )
                torch.testing.assert_close(
                    batched_key, independent_key, rtol=1e-5, atol=1e-6
                )
                torch.testing.assert_close(
                    batched_value, independent_value, rtol=1e-5, atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
