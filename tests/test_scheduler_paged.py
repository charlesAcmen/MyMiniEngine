"""CPU-only scheduler tests for the M2 paged execution path."""

from __future__ import annotations

from collections import deque
import unittest

import torch

from miniengine.core import Request, SamplingParams
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.paged_attention import (
    allocate_prefill_pages,
    build_paged_decode,
    commit_decode_batch,
    commit_prefill,
    release_request_pages,
)
from miniengine.scheduler import Scheduler


class FakePagedEngine:
    """Exercise real page metadata while replacing model computation."""

    def __init__(
        self,
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
        self.prefill_batches: list[list[str]] = []
        self.decode_batches: list[list[str]] = []
        self.prefill_batch = self.paged_prefill
        self.decode_batch = self.paged_decode

    def paged_prefill(self, requests: list[Request]) -> list[int]:
        self.prefill_batches.append([request.request_id for request in requests])
        allocate_prefill_pages(requests, self.kv_pool)
        commit_prefill(requests)
        return [self.prefill_tokens[request.request_id] for request in requests]

    def paged_decode(self, requests: list[Request]) -> list[int]:
        self.decode_batches.append([request.request_id for request in requests])
        build_paged_decode(requests, self.kv_pool)
        commit_decode_batch(requests)
        return [
            self.decode_tokens[request.request_id].popleft()
            for request in requests
        ]

    def release_request(self, request: Request) -> None:
        release_request_pages(request, self.kv_pool)

    @staticmethod
    def is_stop_token(token_id: int) -> bool:
        return False

    @staticmethod
    def decode_token(token_id: int) -> str:
        return f"<{token_id}>"


class PagedSchedulerTest(unittest.TestCase):
    @staticmethod
    def _request(request_id: str, prompt_len: int) -> Request:
        return Request(
            request_id=request_id,
            input_ids=list(range(prompt_len)),
            sampling_params=SamplingParams(max_new_tokens=2),
        )

    def test_packed_prefill_decode_and_finish_release_all_pages(self) -> None:
        engine = FakePagedEngine(
            prefill_tokens={"a": 10, "b": 20},
            decode_tokens={"a": [11], "b": [21]},
        )
        scheduler = Scheduler(engine, max_running=2)
        a = self._request("a", 3)
        b = self._request("b", 5)
        scheduler.add_request(a)
        scheduler.add_request(b)

        self.assertEqual(scheduler.step_paged(), [a, b])
        self.assertEqual(engine.prefill_batches, [["a", "b"]])
        self.assertEqual(engine.decode_batches, [["a", "b"]])
        self.assertEqual(a.output_ids, [10, 11])
        self.assertEqual(b.output_ids, [20, 21])
        self.assertEqual(a.page_table, [])
        self.assertEqual(b.page_table, [])
        self.assertEqual(engine.kv_pool.num_free, 7)

    def test_fcfs_head_request_is_not_bypassed_when_pool_is_too_small(self) -> None:
        engine = FakePagedEngine(
            prefill_tokens={"large": 10, "small": 20},
            decode_tokens={"large": [11], "small": [21]},
            num_pages=3,  # scratch + two usable pages
        )
        scheduler = Scheduler(engine, max_running=2)
        large = self._request("large", 9)  # three pages: cannot fit
        small = self._request("small", 1)
        scheduler.add_request(large)
        scheduler.add_request(small)

        self.assertEqual(scheduler.step_paged(), [])
        self.assertEqual(engine.prefill_batches, [])
        self.assertEqual(
            [request.request_id for request in scheduler.waiting],
            ["large", "small"],
        )


if __name__ == "__main__":
    unittest.main()
