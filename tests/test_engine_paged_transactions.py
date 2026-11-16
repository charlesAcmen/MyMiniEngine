"""Transaction tests for publishing paged KV state after sampling."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from miniengine.config import SamplingConfig
from miniengine.core import Request
from miniengine.engine import Engine
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.paged_attention import allocate_prefill_pages, commit_prefill


class _FakePagedModel:
    @staticmethod
    def forward_paged_prefill(
        _input_ids,
        _position_ids,
        _kv_caches,
        _slot_mapping,
        cu_seqlens,
        _max_seqlen,
    ) -> torch.Tensor:
        batch_size = cu_seqlens.numel() - 1
        return torch.zeros((batch_size, 8))

    @staticmethod
    def forward_paged_decode(
        input_ids,
        _position_ids,
        _kv_caches,
        _cache_seqlens,
        _block_table,
        _rope_cache_len,
    ) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], 1, 8))


class PagedEngineTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = KVMemoryPool(
            num_pages=5,
            page_size=1,
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        self.engine = Engine.__new__(Engine)
        self.engine.kv_pool = self.pool
        self.engine.paged_attention_ops = object()
        self.engine.cuda_graph_runner = None
        self.engine.model = _FakePagedModel()
        self.engine.sampling_config = SamplingConfig(temperature=0)

    @staticmethod
    def _request() -> Request:
        return Request(
            request_id="request",
            input_ids=[1],
            max_new_tokens=4,
        )

    def test_prefill_sampling_failure_does_not_publish_or_retain_pages(self) -> None:
        request = self._request()
        initial_free = self.pool.num_free

        def fail_sampling(*_args, **_kwargs):
            self.assertEqual(request.kv_seq_len, 0)
            raise RuntimeError("sampling failed")

        with (
            patch("miniengine.engine.sample_batch", side_effect=fail_sampling),
            self.assertRaisesRegex(RuntimeError, "sampling failed"),
        ):
            self.engine.paged_prefill([request])

        self.assertEqual(request.kv_seq_len, 0)
        self.assertEqual(request.page_table, [])
        self.assertEqual(self.pool.num_free, initial_free)

    def test_decode_sampling_failure_keeps_old_commit_and_pages(self) -> None:
        request = self._request()
        allocate_prefill_pages([request], self.pool)
        commit_prefill([request])
        request.output_ids.append(7)
        original_table = request.page_table.copy()
        original_free = self.pool.num_free

        def fail_sampling(*_args, **_kwargs):
            self.assertEqual(request.kv_seq_len, 1)
            raise RuntimeError("sampling failed")

        with (
            patch("miniengine.engine.sample_batch", side_effect=fail_sampling),
            self.assertRaisesRegex(RuntimeError, "sampling failed"),
        ):
            self.engine.paged_decode([request])

        self.assertEqual(request.kv_seq_len, 1)
        self.assertEqual(request.page_table, original_table)
        self.assertEqual(self.pool.num_free, original_free)


if __name__ == "__main__":
    unittest.main()
