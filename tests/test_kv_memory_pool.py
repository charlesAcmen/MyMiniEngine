"""CPU and CUDA checks for M2 Part A pool capacity accounting."""

from __future__ import annotations

import unittest

import torch

from miniengine.kv_memory_pool import KVMemoryPool


class KVMemoryPoolBudgetTests(unittest.TestCase):
    def test_from_budget_uses_only_whole_pages_above_minimum(self) -> None:
        # K + V, 2 layers, 2 KV heads, 4 values/head, float32.
        bytes_per_page = 2 * 2 * 2 * 32 * 4 * 4
        pool = KVMemoryPool.from_budget(
            num_layers=2,
            num_kv_heads=2,
            head_dim=4,
            page_size=32,
            dtype=torch.float32,
            device="cpu",
            bytes_budget=bytes_per_page * 20 + 17,
        )

        self.assertEqual(pool.num_pages, 20)
        self.assertEqual(pool.num_free, 19)  # page 0 is the scratch page.
        self.assertEqual(pool.k_caches[0].shape, (20, 32, 2, 4))

    def test_from_budget_preserves_minimum_sixteen_page_policy(self) -> None:
        pool = KVMemoryPool.from_budget(
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            page_size=32,
            dtype=torch.float32,
            device="cpu",
            bytes_budget=2 * 1 * 1 * 32 * 1 * 4 + 1,
        )

        self.assertEqual(pool.num_pages, 16)
        self.assertEqual(pool.num_free, 15)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class KVMemoryPoolCudaTests(unittest.TestCase):
    def test_cuda_pool_has_fixed_layout_and_recycles_page_indices(self) -> None:
        # Deliberately tiny: validates device allocation without loading a model
        # or consuming the production Engine pool budget.
        pool = KVMemoryPool(
            num_pages=3,
            page_size=2,
            num_layers=2,
            num_kv_heads=1,
            head_dim=4,
            dtype=torch.float16,
            device="cuda",
        )

        self.assertTrue(pool.k_caches[0].is_cuda)
        self.assertEqual(pool.k_caches[0].dtype, torch.float16)
        self.assertEqual(pool.k_caches[0].shape, (3, 2, 1, 4))

        pages = pool.allocate(2)
        self.assertEqual(pages, [1, 2])
        self.assertEqual(pool.num_free, 0)
        pool.free(pages)
        self.assertEqual(pool.num_free, 2)
