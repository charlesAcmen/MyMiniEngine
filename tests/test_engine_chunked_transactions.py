"""Transaction tests for publishing incremental chunked-prefill KV state."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from miniengine.config import SamplingConfig
from miniengine.core import Request
from miniengine.engine import Engine
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.paged_attention import PrefillChunk


class _FakeChunkedModel:
    fail_forward = False

    @classmethod
    def forward_paged_chunked_prefill(
        cls,
        _input_ids,
        _position_ids,
        _kv_caches,
        _slot_mapping,
        _cu_seqlens_q,
        _cu_seqlens_k,
        _block_table,
        sample_indices,
        _max_seqlen_q,
        _max_seqlen_k,
    ) -> torch.Tensor:
        if cls.fail_forward:
            raise RuntimeError("forward failed")
        return torch.zeros((sample_indices.numel(), 8))


class ChunkedPrefillTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = KVMemoryPool(
            num_pages=5,
            page_size=2,
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        self.engine = Engine.__new__(Engine)
        self.engine.kv_pool = self.pool
        self.engine.paged_attention_ops = object()
        self.engine.model = _FakeChunkedModel()
        self.engine.sampling_config = SamplingConfig(temperature=0)
        _FakeChunkedModel.fail_forward = False

    @staticmethod
    def _request() -> Request:
        return Request(request_id="request", input_ids=[1, 2, 3])

    def test_partial_chunk_commits_without_sampling_then_final_chunk_samples(self) -> None:
        request = self._request()

        with patch("miniengine.engine.sample_batch", return_value=[7]) as sample:
            partial_result = self.engine.paged_prefill_chunks(
                [PrefillChunk(request, 0, 2)]
            )
            self.assertEqual(partial_result, [None])
            self.assertEqual(request.kv_seq_len, 2)
            self.assertEqual(request.page_table, [1])
            sample.assert_not_called()

            final_result = self.engine.paged_prefill_chunks(
                [PrefillChunk(request, 2, 3)]
            )

        self.assertEqual(final_result, [7])
        self.assertEqual(request.kv_seq_len, 3)
        self.assertEqual(request.page_table, [1, 2])

    def test_sampling_failure_keeps_prior_chunk_and_returns_new_page(self) -> None:
        request = self._request()
        self.engine.paged_prefill_chunks([PrefillChunk(request, 0, 2)])
        original_table = request.page_table.copy()
        original_free = self.pool.num_free

        with (
            patch(
                "miniengine.engine.sample_batch",
                side_effect=RuntimeError("sampling failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "sampling failed"),
        ):
            self.engine.paged_prefill_chunks([PrefillChunk(request, 2, 3)])

        self.assertEqual(request.kv_seq_len, 2)
        self.assertEqual(request.page_table, original_table)
        self.assertEqual(self.pool.num_free, original_free)

    def test_forward_failure_keeps_prior_chunk_and_returns_new_page(self) -> None:
        request = self._request()
        self.engine.paged_prefill_chunks([PrefillChunk(request, 0, 2)])
        original_table = request.page_table.copy()
        original_free = self.pool.num_free
        _FakeChunkedModel.fail_forward = True

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            self.engine.paged_prefill_chunks([PrefillChunk(request, 2, 3)])

        self.assertEqual(request.kv_seq_len, 2)
        self.assertEqual(request.page_table, original_table)
        self.assertEqual(self.pool.num_free, original_free)


if __name__ == "__main__":
    unittest.main()
