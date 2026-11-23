"""CPU-only FCFS scheduler tests for the M3 chunked-prefill path."""

from __future__ import annotations

from collections import deque
import unittest

import torch

from miniengine.config import SchedulerConfig
from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.paged_attention import (
    PrefillChunk,
    allocate_prefill_chunk_pages,
    build_paged_decode,
    commit_decode_batch,
    commit_prefill_chunks,
    release_request_pages,
)
from miniengine.scheduler import Scheduler


class FakeChunkedEngine:
    def __init__(
        self,
        *,
        prefill_tokens: dict[str, int],
        decode_tokens: dict[str, list[int]],
        num_pages: int = 8,
    ) -> None:
        self.mode = "paged"
        self.kv_pool = KVMemoryPool(
            num_pages=num_pages,
            page_size=4,
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
        )
        self.prefill_tokens = prefill_tokens
        self.decode_tokens = {
            request_id: deque(tokens)
            for request_id, tokens in decode_tokens.items()
        }
        self.chunk_batches: list[list[tuple[str, int, int]]] = []
        self.decode_batches: list[list[str]] = []
        self.decode_batch = self.paged_decode

    def paged_prefill_chunks(
        self, chunks: list[PrefillChunk]
    ) -> list[int | None]:
        self.chunk_batches.append(
            [
                (chunk.request.request_id, chunk.start, chunk.end)
                for chunk in chunks
            ]
        )
        allocate_prefill_chunk_pages(chunks, self.kv_pool)
        commit_prefill_chunks(chunks)
        return [
            self.prefill_tokens[chunk.request.request_id]
            if chunk.completes_prompt
            else None
            for chunk in chunks
        ]

    def paged_decode(self, requests: list[Request]) -> list[int]:
        self.decode_batches.append(
            [request.request_id for request in requests]
        )
        build_paged_decode(requests, self.kv_pool)
        commit_decode_batch(requests)
        return [
            self.decode_tokens[request.request_id].popleft()
            for request in requests
        ]

    def release_request(self, request: Request) -> None:
        release_request_pages(request, self.kv_pool)

    @staticmethod
    def is_stop_token(_token_id: int) -> bool:
        return False


class ChunkedPrefillSchedulerTests(unittest.TestCase):
    @staticmethod
    def _request(
        request_id: str, prompt_len: int, max_new_tokens: int = 2
    ) -> Request:
        return Request(
            request_id=request_id,
            input_ids=list(range(prompt_len)),
            max_new_tokens=max_new_tokens,
        )

    def test_long_prompt_advances_in_budgeted_chunks_before_sampling(self) -> None:
        engine = FakeChunkedEngine(
            prefill_tokens={"a": 10},
            decode_tokens={"a": [11]},
        )
        scheduler = Scheduler(
            engine, SchedulerConfig(max_running=1, prefill_chunk_size=4)
        )
        request = self._request("a", 9)
        scheduler.add_request(request)

        self.assertEqual(scheduler._step_impl(), [])
        self.assertEqual((request.kv_seq_len, request.output_ids), (4, []))
        self.assertEqual(scheduler._step_impl(), [])
        self.assertEqual((request.kv_seq_len, request.output_ids), (8, []))
        self.assertEqual(scheduler._step_impl(), [request])

        self.assertEqual(
            engine.chunk_batches,
            [[("a", 0, 4)], [("a", 4, 8)], [("a", 8, 9)]],
        )
        self.assertEqual(engine.decode_batches, [["a"]])
        self.assertEqual(request.output_ids, [10, 11])
        self.assertEqual(engine.kv_pool.num_free, 7)

    def test_fcfs_completion_passes_remaining_budget_to_next_request(self) -> None:
        engine = FakeChunkedEngine(
            prefill_tokens={"a": 10, "b": 20},
            decode_tokens={},
        )
        scheduler = Scheduler(
            engine, SchedulerConfig(max_running=2, prefill_chunk_size=4)
        )
        a = self._request("a", 6, max_new_tokens=1)
        b = self._request("b", 5, max_new_tokens=1)
        scheduler.add_request(a)
        scheduler.add_request(b)

        self.assertEqual(scheduler._step_impl(), [])
        self.assertEqual(scheduler._step_impl(), [a])
        self.assertEqual((a.output_ids, b.kv_seq_len), ([10], 2))
        self.assertEqual(scheduler._step_impl(), [b])

        self.assertEqual(
            engine.chunk_batches,
            [
                [("a", 0, 4)],
                [("a", 4, 6), ("b", 0, 2)],
                [("b", 2, 5)],
            ],
        )
        self.assertTrue(
            all(
                sum(end - start for _, start, end in batch) <= 4
                for batch in engine.chunk_batches
            )
        )
        self.assertEqual(engine.kv_pool.num_free, 7)

    def test_prompt_larger_than_total_pool_capacity_fails_early(self) -> None:
        engine = FakeChunkedEngine(
            prefill_tokens={"large": 10},
            decode_tokens={},
            num_pages=3,
        )
        scheduler = Scheduler(
            engine, SchedulerConfig(max_running=1, prefill_chunk_size=4)
        )
        scheduler.add_request(self._request("large", 9))

        with self.assertRaisesRegex(RuntimeError, "only 2 usable pages"):
            scheduler._step_impl()

        self.assertEqual(engine.chunk_batches, [])
        self.assertEqual(engine.kv_pool.num_free, 2)


if __name__ == "__main__":
    unittest.main()
