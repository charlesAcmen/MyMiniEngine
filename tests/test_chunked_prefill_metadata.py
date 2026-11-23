"""CPU-only metadata and page-transaction tests for chunked prefill."""

from __future__ import annotations

import unittest

import torch

from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.paged_attention import (
    PrefillChunk,
    allocate_prefill_chunk_pages,
    build_packed_chunked_prefill,
    commit_prefill_chunks,
    rollback_prefill_chunk_pages,
)


class ChunkedPrefillMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = KVMemoryPool(
            num_pages=8,
            page_size=4,
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )

    def test_chunks_use_absolute_positions_and_full_kv_lengths(self) -> None:
        a = Request(request_id="a", input_ids=list(range(7)))
        b = Request(request_id="b", input_ids=list(range(10, 15)))

        first = PrefillChunk(a, 0, 4)
        allocate_prefill_chunk_pages([first], self.pool)
        commit_prefill_chunks([first])

        chunks = [PrefillChunk(a, 4, 7), PrefillChunk(b, 0, 2)]
        allocate_prefill_chunk_pages(chunks, self.pool)
        packed = build_packed_chunked_prefill(chunks, self.pool)

        self.assertEqual(packed.input_ids.tolist(), [4, 5, 6, 10, 11])
        self.assertEqual(packed.position_ids.tolist(), [4, 5, 6, 0, 1])
        self.assertEqual(packed.cu_seqlens_q.tolist(), [0, 3, 5])
        self.assertEqual(packed.cu_seqlens_k.tolist(), [0, 7, 9])
        self.assertEqual(packed.slot_mapping.tolist(), [8, 9, 10, 12, 13])
        self.assertEqual(packed.block_table.tolist(), [[1, 2], [3, 0]])
        self.assertEqual(packed.sample_indices.tolist(), [2])
        self.assertEqual((packed.max_seqlen_q, packed.max_seqlen_k), (3, 7))

        commit_prefill_chunks(chunks)
        self.assertEqual((a.kv_seq_len, b.kv_seq_len), (7, 2))

    def test_failed_or_rolled_back_chunk_preserves_committed_prefix(self) -> None:
        request = Request(request_id="request", input_ids=list(range(9)))
        first = PrefillChunk(request, 0, 4)
        allocate_prefill_chunk_pages([first], self.pool)
        commit_prefill_chunks([first])
        original_table = request.page_table.copy()
        original_free = self.pool.num_free

        second = PrefillChunk(request, 4, 9)
        original_counts = allocate_prefill_chunk_pages([second], self.pool)
        self.assertGreater(len(request.page_table), len(original_table))

        rollback_prefill_chunk_pages([second], original_counts, self.pool)
        self.assertEqual(request.page_table, original_table)
        self.assertEqual(request.kv_seq_len, 4)
        self.assertEqual(self.pool.num_free, original_free)

    def test_invalid_chunk_start_does_not_leak_pages(self) -> None:
        request = Request(request_id="request", input_ids=list(range(8)))
        initial_free = self.pool.num_free

        with self.assertRaisesRegex(ValueError, "committed KV length"):
            allocate_prefill_chunk_pages(
                [PrefillChunk(request, 1, 4)], self.pool
            )

        self.assertEqual(request.page_table, [])
        self.assertEqual(self.pool.num_free, initial_free)


if __name__ == "__main__":
    unittest.main()
