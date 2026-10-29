"""Part-B paged request ownership and packed-prefill metadata tests."""

from __future__ import annotations

import unittest

import torch

from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.paged_attention import (
    allocate_prefill_pages,
    build_packed_prefill,
    build_paged_decode,
    commit_decode_batch,
    commit_decode_token,
    commit_prefill,
    release_request_pages,
    reserve_decode_slot,
)


class PackedPrefillTests(unittest.TestCase):
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
        self.a = Request(request_id="a", input_ids=[10, 11, 12])
        self.b = Request(request_id="b", input_ids=[20, 21, 22, 23, 24])

    def test_packed_prefill_uses_page_tables_without_padding(self) -> None:
        allocate_prefill_pages([self.a, self.b], self.pool)
        self.assertEqual(self.a.page_table, [1])
        self.assertEqual(self.b.page_table, [2, 3])

        packed = build_packed_prefill([self.a, self.b], self.pool)
        self.assertEqual(packed.input_ids.tolist(), [10, 11, 12, 20, 21, 22, 23, 24])
        self.assertEqual(packed.position_ids.tolist(), [0, 1, 2, 0, 1, 2, 3, 4])
        self.assertEqual(packed.cu_seqlens.tolist(), [0, 3, 8])
        self.assertEqual(packed.slot_mapping.tolist(), [4, 5, 6, 8, 9, 10, 11, 12])
        self.assertEqual(packed.max_seqlen, 5)
        self.assertEqual(self.a.kv_seq_len, 0)

        commit_prefill([self.a, self.b])
        self.assertEqual((self.a.kv_seq_len, self.b.kv_seq_len), (3, 5))

    def test_decode_grows_only_at_a_page_boundary_and_release_returns_pages(self) -> None:
        allocate_prefill_pages([self.b], self.pool)
        commit_prefill([self.b])

        self.assertEqual(reserve_decode_slot(self.b, self.pool), 9)
        self.assertEqual(self.b.page_table, [1, 2])
        commit_decode_token(self.b)
        self.assertEqual(self.b.kv_seq_len, 6)

        release_request_pages(self.b, self.pool)
        self.assertEqual(self.b.page_table, [])
        self.assertEqual(self.b.kv_seq_len, 0)
        self.assertEqual(self.pool.num_free, 7)

    def test_failed_multi_request_allocation_rolls_back_prior_page_ownership(self) -> None:
        small_pool = KVMemoryPool(
            num_pages=3,  # scratch + two usable pages
            page_size=4,
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        a = Request(request_id="a", input_ids=[1, 2, 3, 4, 5])  # two pages
        b = Request(request_id="b", input_ids=[6])  # one unavailable page

        with self.assertRaises(RuntimeError):
            allocate_prefill_pages([a, b], small_pool)
        self.assertEqual(a.page_table, [])
        self.assertEqual(b.page_table, [])
        self.assertEqual(small_pool.num_free, 2)

    def test_decode_metadata_uses_pre_append_lengths_and_scratch_padding(self) -> None:
        allocate_prefill_pages([self.a, self.b], self.pool)
        commit_prefill([self.a, self.b])
        self.a.output_ids.append(30)
        self.b.output_ids.append(40)

        decode = build_paged_decode([self.a, self.b], self.pool)

        self.assertEqual(decode.input_ids.tolist(), [[30], [40]])
        self.assertEqual(decode.position_ids.tolist(), [[3], [5]])
        self.assertEqual(decode.cache_seqlens.tolist(), [3, 5])
        self.assertEqual(decode.block_table.tolist(), [[1, 0], [2, 3]])
        commit_decode_batch([self.a, self.b])
        self.assertEqual((self.a.kv_seq_len, self.b.kv_seq_len), (4, 6))

    def test_decode_page_growth_rolls_back_for_the_whole_batch(self) -> None:
        pool = KVMemoryPool(
            num_pages=4,  # scratch + two prompt pages + one growth page
            page_size=4,
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        a = Request(request_id="a", input_ids=[1, 2, 3, 4], output_ids=[5])
        b = Request(request_id="b", input_ids=[6, 7, 8, 9], output_ids=[10])
        allocate_prefill_pages([a, b], pool)
        commit_prefill([a, b])
        original_tables = [list(a.page_table), list(b.page_table)]

        with self.assertRaises(RuntimeError):
            build_paged_decode([a, b], pool)

        self.assertEqual(a.page_table, original_tables[0])
        self.assertEqual(b.page_table, original_tables[1])
        self.assertEqual(pool.num_free, 1)
