"""CPU-only scheduler tests for the M2 paged execution path."""

from __future__ import annotations

from collections import deque
import unittest
from unittest.mock import patch

import torch

from miniengine.config import SchedulerConfig
from miniengine.core import (
    KV_CAPACITY_EXHAUSTED,
    Request,
    RequestStatus,
    RequestTooLargeError,
)
from miniengine.engine import Engine
from miniengine.kv_memory_pool import KVMemoryPool, KVPoolExhaustedError
from miniengine.paged_attention import (
    PrefillSuffix,
    allocate_prefill_suffix_pages,
    allocate_prefill_pages,
    build_paged_decode,
    commit_decode_batch,
    commit_prefill,
    commit_prefill_suffixes,
    release_request_pages,
)
from miniengine.radix_cache import MatchResult, RadixCache
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
        self.radix_cache = None
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

    def validate_request_capacity(self, request: Request) -> None:
        Engine.validate_request_capacity(self, request)

    def try_prepare_paged_prefill(
        self,
        request: Request,
        prepared,
    ):
        return Engine.try_prepare_paged_prefill(
            self,
            request,
            prepared,
        )

    def execute_paged_prefill(self, admissions):
        return Engine.execute_paged_prefill(self, admissions)

    def prepare_retracted_request(self, request: Request) -> None:
        Engine.prepare_retracted_request(self, request)

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

    def finish_request(self, request: Request) -> None:
        self.release_request(request)

    @staticmethod
    def is_stop_token(token_id: int) -> bool:
        return False


class PrefixCachingFakePagedEngine(FakePagedEngine):
    """Use the real cache ownership transitions around fake model work."""

    def __init__(
        self,
        prefill_tokens: dict[str, int],
        decode_tokens: dict[str, list[int]],
    ) -> None:
        super().__init__(prefill_tokens, decode_tokens, num_pages=12)
        self.radix_cache = RadixCache(self.kv_pool)
        self.kv_pool.set_page_reclaimer(self.radix_cache.evict)
        self.suffix_batches: list[list[tuple[str, int, int]]] = []
        self.finish_request = self._finish_paged_request

    def prepare_prefix_cache_match(
        self, request: Request
    ) -> tuple[PrefillSuffix, MatchResult]:
        return Engine.prepare_prefix_cache_match(self, request)

    def record_prefix_cache_match(
        self, request: Request, match: MatchResult
    ) -> None:
        Engine.record_prefix_cache_match(self, request, match)

    def rollback_prefix_cache_match(self, request: Request) -> None:
        Engine.rollback_prefix_cache_match(self, request)

    def publish_prompt_prefix(self, request: Request) -> None:
        Engine.publish_prompt_prefix(self, request)

    def paged_prefill_suffixes(
        self, suffixes: list[PrefillSuffix]
    ) -> list[int | None]:
        self.suffix_batches.append(
            [
                (suffix.request.request_id, suffix.start, suffix.end)
                for suffix in suffixes
            ]
        )
        allocate_prefill_suffix_pages(suffixes, self.kv_pool)
        commit_prefill_suffixes(suffixes)
        return [
            self.prefill_tokens[suffix.request.request_id]
            if suffix.completes_prompt
            else None
            for suffix in suffixes
        ]

    def _release_paged_request(self, request: Request) -> None:
        Engine._release_paged_request(self, request)

    def _finish_paged_request(self, request: Request) -> None:
        Engine._finish_paged_request(self, request)

    def release_request(self, request: Request) -> None:
        self._release_paged_request(request)


class ReplayFakePagedEngine(FakePagedEngine):
    """Return a distinct next token when a retracted context is replayed."""

    def __init__(
        self,
        prefill_tokens: dict[str, list[int]],
        decode_tokens: dict[str, list[int]],
        num_pages: int,
    ) -> None:
        super().__init__({}, decode_tokens, num_pages=num_pages)
        self.replay_prefill_tokens = {
            request_id: deque(tokens)
            for request_id, tokens in prefill_tokens.items()
        }

    def paged_prefill(self, requests: list[Request]) -> list[int]:
        self.prefill_batches.append([request.request_id for request in requests])
        allocate_prefill_pages(requests, self.kv_pool)
        commit_prefill(requests)
        return [
            self.replay_prefill_tokens[request.request_id].popleft()
            for request in requests
        ]

class PagedSchedulerTest(unittest.TestCase):
    @staticmethod
    def _request(request_id: str, prompt_len: int) -> Request:
        return Request(
            request_id=request_id,
            input_ids=list(range(prompt_len)),
            max_new_tokens=2,
        )

    def test_packed_prefill_decode_and_finish_release_all_pages(self) -> None:
        engine = FakePagedEngine(
            prefill_tokens={"a": 10, "b": 20},
            decode_tokens={"a": [11], "b": [21]},
        )
        scheduler = Scheduler(engine, SchedulerConfig(max_running=2))
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

    def test_impossible_prompt_is_rejected_before_it_can_block_fcfs(self) -> None:
        engine = FakePagedEngine(
            prefill_tokens={"large": 10, "small": 20},
            decode_tokens={"large": [11], "small": [21]},
            num_pages=3,  # scratch + two usable pages
        )
        scheduler = Scheduler(engine, SchedulerConfig(max_running=2))
        large = self._request("large", 9)  # three pages: cannot fit
        small = self._request("small", 1)

        with self.assertRaisesRegex(RequestTooLargeError, "requires 3 KV pages"):
            scheduler.add_request(large)

        self.assertEqual(list(scheduler.waiting), [])
        self.assertNotIn(large.request_id, scheduler._requests)

        scheduler.add_request(small)
        self.assertEqual(scheduler.step_paged(), [small])

    def test_later_request_hits_prompt_published_before_first_request_finishes(self) -> None:
        engine = PrefixCachingFakePagedEngine(
            prefill_tokens={"a": 10, "b": 20},
            decode_tokens={"a": [11, 12], "b": [21, 22]},
        )
        scheduler = Scheduler(engine, SchedulerConfig(max_running=2))
        a = Request(
            request_id="a",
            input_ids=[0, 1, 2, 3, 4, 5],
            max_new_tokens=3,
        )
        scheduler.add_request(a)

        # A completes prompt prefill, publishes page [0, 1, 2, 3], and then
        # remains decode-ready with two generated tokens.
        self.assertEqual(scheduler.step_paged(), [])
        self.assertIn(a, scheduler.decode_ready)
        self.assertEqual(engine.radix_cache.num_cached_pages, 1)

        b = Request(
            request_id="b",
            input_ids=[0, 1, 2, 3, 4, 99],
            max_new_tokens=3,
        )
        scheduler.add_request(b)

        # Admission happens before this step decodes and finishes A. Therefore
        # B's hit can only come from A's earlier post-prefill publication.
        self.assertEqual(scheduler.step_paged(), [a])
        self.assertEqual(b.cache_hit_tokens, 4)
        self.assertEqual(engine.suffix_batches, [[("a", 0, 6)], [("b", 4, 6)]])
        self.assertIn(b, scheduler.decode_ready)

        self.assertEqual(scheduler.step_paged(), [b])
        self.assertEqual(engine.radix_cache.root.ref_count, 0)

    def test_publication_failure_releases_committed_prefill_before_requeue(self) -> None:
        engine = PrefixCachingFakePagedEngine(
            prefill_tokens={"a": 10},
            decode_tokens={"a": [11]},
        )
        scheduler = Scheduler(engine, SchedulerConfig(max_running=1))
        request = Request(
            request_id="a",
            input_ids=[0, 1, 2, 3, 4, 5],
            max_new_tokens=2,
        )
        scheduler.add_request(request)

        with (
            patch.object(
                engine,
                "publish_prompt_prefix",
                side_effect=RuntimeError("publication failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "publication failed"),
        ):
            scheduler.step_paged()

        self.assertEqual(list(scheduler.waiting), [request])
        self.assertEqual(request.page_table, [])
        self.assertEqual(request.kv_seq_len, 0)
        self.assertEqual(request.cache_hit_tokens, 0)
        self.assertEqual(engine.radix_cache.root.ref_count, 0)
        self.assertEqual(engine.kv_pool.num_free, 11)

    def test_decode_exhaustion_fails_only_one_request_without_retraction(self) -> None:
        engine = ReplayFakePagedEngine(
            prefill_tokens={"a": [10], "b": [20]},
            decode_tokens={"a": [11], "b": [21]},
            num_pages=12,  # scratch + eleven usable four-token pages
        )
        scheduler = Scheduler(
            engine,
            SchedulerConfig(max_running=2, retraction_enabled=False),
        )
        a = self._request("a", 20)
        b = self._request("b", 20)
        scheduler.add_request(a)
        scheduler.add_request(b)

        self.assertEqual(scheduler.step_paged(), [])

        # Decode allocation is transactional, then one request is released.
        self.assertEqual(a.status, RequestStatus.RUNNING)
        self.assertEqual(b.status, RequestStatus.FAILED)
        self.assertEqual(len(a.page_table), 5)
        self.assertEqual(b.page_table, [])
        self.assertEqual(
            [request.request_id for request in scheduler.decode_ready], ["a"]
        )
        self.assertEqual(scheduler.total_failed, 1)
        self.assertFalse(b.token_queue.get_nowait().finished)
        failure = b.token_queue.get_nowait()
        self.assertTrue(failure.finished)
        self.assertEqual(failure.error_code, KV_CAPACITY_EXHAUSTED)

    def test_retraction_replays_context_without_duplicate_stream_tokens(self) -> None:
        engine = ReplayFakePagedEngine(
            prefill_tokens={"a": [10], "b": [20, 21]},
            decode_tokens={"a": [11], "b": [22]},
            num_pages=12,
        )
        scheduler = Scheduler(
            engine,
            SchedulerConfig(max_running=2, retraction_enabled=True),
        )
        a = self._request("a", 20)
        b = Request(
            request_id="b",
            input_ids=list(range(20)),
            max_new_tokens=3,
            arrival_time=a.arrival_time + 1.0,
        )
        scheduler.add_request(a)
        scheduler.add_request(b)

        # The first two-request decode needs two growth pages but only one is
        # free. B is younger and has more remaining work, so it is retracted.
        self.assertEqual(scheduler.step_paged(), [])
        self.assertEqual(b.retraction_count, 1)
        self.assertEqual(b.input_ids[-1], 20)
        self.assertEqual(b.recomputed_output_tokens, 1)
        self.assertEqual([request.request_id for request in scheduler.waiting], ["b"])
        self.assertEqual([request.request_id for request in scheduler.decode_ready], ["a"])

        # Cooldown leaves B waiting until A completes and frees its pages.
        self.assertEqual(scheduler.step_paged(), [a])
        self.assertEqual(engine.prefill_batches, [["a", "b"]])
        self.assertEqual(scheduler.step_paged(), [b])

        self.assertEqual(a.output_ids, [10, 11])
        self.assertEqual(b.output_ids, [20, 21, 22])
        self.assertEqual(b.num_prompt_tokens, 20)
        self.assertEqual(b.num_input_tokens, 21)
        self.assertEqual(scheduler.total_retracted, 1)
        self.assertEqual(engine.kv_pool.num_free, 11)

    def test_lone_request_exhaustion_fails_and_waiting_request_continues(self) -> None:
        engine = ReplayFakePagedEngine(
            prefill_tokens={"a": [10], "b": [20]},
            decode_tokens={"a": [11], "b": [21]},
            num_pages=6,  # prompt owns every usable page
        )
        scheduler = Scheduler(
            engine,
            SchedulerConfig(max_running=1, retraction_enabled=True),
        )
        a = self._request("a", 20)
        b = self._request("b", 4)
        scheduler.add_request(a)
        scheduler.add_request(b)

        self.assertEqual(scheduler.step_paged(), [])
        self.assertEqual(scheduler.total_retracted, 0)
        self.assertEqual(scheduler.total_failed, 1)
        self.assertEqual(a.status, RequestStatus.FAILED)
        self.assertEqual(a.page_table, [])
        self.assertEqual(engine.kv_pool.num_free, 5)
        self.assertEqual([request.request_id for request in scheduler.waiting], ["b"])
        self.assertNotIn("a", scheduler._requests)

        first = a.token_queue.get_nowait()
        failure = a.token_queue.get_nowait()
        self.assertEqual(first.token_id, 10)
        self.assertEqual(failure.error_code, KV_CAPACITY_EXHAUSTED)
        self.assertIn("KVPoolExhaustedError", failure.error)

        self.assertEqual(scheduler.step_paged(), [b])
        self.assertEqual(b.status, RequestStatus.FINISHED)
        self.assertEqual(engine.kv_pool.num_free, 5)

    def test_capacity_failure_releases_prefix_cache_pin(self) -> None:
        engine = PrefixCachingFakePagedEngine(
            prefill_tokens={"a": 10},
            decode_tokens={"a": [11]},
        )
        scheduler = Scheduler(
            engine,
            SchedulerConfig(max_running=1, retraction_enabled=True),
        )
        request = Request(
            request_id="a",
            input_ids=list(range(44)),  # owns every usable four-token page
            max_new_tokens=2,
        )
        scheduler.add_request(request)

        self.assertEqual(scheduler.step_paged(), [])

        self.assertEqual(request.status, RequestStatus.FAILED)
        self.assertEqual(request.page_table, [])
        self.assertIsNone(request.cache_node)
        self.assertEqual(engine.radix_cache.root.ref_count, 0)
        self.assertEqual(engine.radix_cache.num_evictable_pages(), 11)
        self.assertEqual(scheduler.decode_ready, [])
        self.assertNotIn(request.request_id, scheduler._requests)

    def test_retraction_publishes_committed_cache_pages_and_releases_pin(self) -> None:
        engine = PrefixCachingFakePagedEngine(
            prefill_tokens={"victim": 10},
            decode_tokens={"victim": [11]},
        )
        scheduler = Scheduler(
            engine,
            SchedulerConfig(max_running=2, retraction_enabled=True),
        )
        victim = Request(
            request_id="victim",
            input_ids=[0, 1, 2, 3, 4, 5],
            max_new_tokens=10,
        )
        scheduler.add_request(victim)
        self.assertEqual(scheduler.step_paged(), [])
        self.assertIsNotNone(victim.cache_node)
        self.assertGreater(engine.radix_cache.root.ref_count, 0)

        survivor = Request(
            request_id="survivor",
            input_ids=[99],
            output_ids=[98],
            max_new_tokens=2,
            arrival_time=victim.arrival_time - 1.0,
        )
        scheduler.decode_ready.append(survivor)
        self.assertTrue(
            scheduler._retract_decode_victim(
                [victim, survivor],
                KVPoolExhaustedError("KV pool exhausted: test"),
            )
        )

        self.assertEqual(victim.page_table, [])
        self.assertIsNone(victim.cache_node)
        self.assertEqual(engine.radix_cache.root.ref_count, 0)
        self.assertGreaterEqual(engine.radix_cache.num_cached_pages, 1)
        self.assertEqual(victim.input_ids[-2:], [10, 11])


if __name__ == "__main__":
    unittest.main()
