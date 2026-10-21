"""Unit tests for M1 Part A's static-batch Scheduler behavior."""

from collections import deque
import unittest

from miniengine.core import Request, RequestStatus, SamplingParams
from miniengine.scheduler import Scheduler


class FakeEngine:
    """Deterministic CPU-only engine that records scheduler interactions."""

    def __init__(
        self,
        prefill_tokens: dict[str, int],
        decode_tokens: dict[str, list[int]],
        stop_tokens: set[int] | None = None,
    ) -> None:
        self.prefill_tokens = prefill_tokens
        self.decode_tokens = {
            request_id: deque(tokens)
            for request_id, tokens in decode_tokens.items()
        }
        self.stop_tokens = stop_tokens or set()
        self.prefill_calls: list[str] = []
        self.batched_decode_calls: list[list[str]] = []

    def prefill(self, request: Request) -> int:
        self.prefill_calls.append(request.request_id)
        request.kv_cache = object()
        return self.prefill_tokens[request.request_id]

    def batched_decode(self, requests: list[Request]) -> list[int]:
        self.batched_decode_calls.append([request.request_id for request in requests])
        return [
            self.decode_tokens[request.request_id].popleft()
            for request in requests
        ]

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_tokens

    @staticmethod
    def decode_token(token_id: int) -> str:
        return f"<{token_id}>"


class StaticBatchSchedulerTest(unittest.TestCase):
    @staticmethod
    def _request(request_id: str, max_new_tokens: int = 3) -> Request:
        return Request(
            request_id=request_id,
            input_ids=[1, 2],
            sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        )

    @staticmethod
    def _drain_stream(request: Request) -> list[tuple[int, str, bool]]:
        items: list[tuple[int, str, bool]] = []
        while not request.token_queue.empty():
            token = request.token_queue.get_nowait()
            items.append((token.token_id, token.token_text, token.finished))
        return items

    def test_admission_is_capped_and_each_active_step_has_one_batch_decode(self) -> None:
        engine = FakeEngine(
            prefill_tokens={"one": 11, "two": 21, "three": 31},
            decode_tokens={
                "one": [12, 13],
                "two": [22, 23],
                "three": [32, 33],
            },
        )
        scheduler = Scheduler(engine, max_running=2)
        requests = [self._request(name) for name in ("one", "two", "three")]
        for request in requests:
            scheduler.add_request(request)

        self.assertEqual(scheduler.step_static_batch(), [])
        self.assertEqual(engine.prefill_calls, ["one", "two"])
        self.assertEqual(
            [request.request_id for request in scheduler.running], ["one", "two"]
        )
        self.assertEqual(
            [request.request_id for request in scheduler.waiting], ["three"]
        )
        self.assertEqual(engine.batched_decode_calls, [])

        self.assertEqual(scheduler.step_static_batch(), [])
        self.assertEqual(engine.batched_decode_calls, [["one", "two"]])
        self.assertEqual(scheduler.step_static_batch(), requests[:2])
        self.assertEqual(
            engine.batched_decode_calls, [["one", "two"], ["one", "two"]]
        )
        self.assertEqual(scheduler.running, [])
        self.assertEqual(
            [request.request_id for request in scheduler.waiting], ["three"]
        )

        self.assertEqual(scheduler.step_static_batch(), [])
        self.assertEqual(engine.prefill_calls, ["one", "two", "three"])

    def test_finished_slot_waits_for_the_static_batch_to_drain(self) -> None:
        engine = FakeEngine(
            prefill_tokens={"short": 10, "long": 20, "next": 30},
            decode_tokens={"short": [11], "long": [21, 22], "next": [31]},
        )
        scheduler = Scheduler(engine, max_running=2)
        short = self._request("short", max_new_tokens=2)
        long = self._request("long", max_new_tokens=3)
        next_request = self._request("next", max_new_tokens=2)
        for request in (short, long, next_request):
            scheduler.add_request(request)

        scheduler.step_static_batch()
        self.assertEqual(scheduler.step_static_batch(), [short])
        self.assertEqual(
            [request.request_id for request in scheduler.running], ["long"]
        )
        self.assertEqual(
            [request.request_id for request in scheduler.waiting], ["next"]
        )
        self.assertEqual(engine.prefill_calls, ["short", "long"])

        self.assertEqual(scheduler.step_static_batch(), [long])
        self.assertEqual(scheduler.running, [])
        self.assertEqual(engine.prefill_calls, ["short", "long"])

        scheduler.step_static_batch()
        self.assertEqual(engine.prefill_calls, ["short", "long", "next"])

    def test_completion_releases_kv_emits_sentinel_and_updates_stats(self) -> None:
        engine = FakeEngine(prefill_tokens={"done": 40}, decode_tokens={"done": [41]})
        scheduler = Scheduler(engine, max_running=2)
        request = self._request("done", max_new_tokens=2)
        scheduler.add_request(request)

        scheduler.step_static_batch()
        self.assertIsNotNone(request.kv_cache)
        self.assertEqual(scheduler.step_static_batch(), [request])

        self.assertEqual(request.status, RequestStatus.FINISHED)
        self.assertIsNone(request.kv_cache)
        self.assertEqual(
            self._drain_stream(request),
            [(40, "<40>", False), (41, "<41>", False), (-1, "", True)],
        )
        self.assertEqual(scheduler.total_finished, 1)
        self.assertEqual(scheduler.total_generated_tokens, 2)

    def test_prefill_stop_finishes_without_entering_batched_decode(self) -> None:
        engine = FakeEngine(
            prefill_tokens={"stop": 99},
            decode_tokens={"stop": []},
            stop_tokens={99},
        )
        scheduler = Scheduler(engine, max_running=2)
        request = self._request("stop", max_new_tokens=8)
        scheduler.add_request(request)

        self.assertEqual(scheduler.step_static_batch(), [request])
        self.assertEqual(scheduler.running, [])
        self.assertEqual(engine.batched_decode_calls, [])
        self.assertEqual(request.status, RequestStatus.FINISHED)
        self.assertIsNone(request.kv_cache)
        self.assertEqual(self._drain_stream(request)[-1], (-1, "", True))
        self.assertEqual(scheduler.total_finished, 1)
        self.assertEqual(scheduler.total_generated_tokens, 1)


class ContinuousBatchSchedulerTest(unittest.TestCase):
    """Deterministic tests for M1 Part B iteration-level batch admission."""

    @staticmethod
    def _request(request_id: str, max_new_tokens: int = 3) -> Request:
        return Request(
            request_id=request_id,
            input_ids=[1, 2],
            sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        )

    def test_newly_prefilled_requests_join_decode_in_the_same_iteration(self) -> None:
        engine = FakeEngine(
            prefill_tokens={"one": 11, "two": 21},
            decode_tokens={"one": [12], "two": [22]},
        )
        scheduler = Scheduler(engine, max_running=2)
        requests = [self._request(name) for name in ("one", "two")]
        for request in requests:
            scheduler.add_request(request)

        self.assertEqual(scheduler.step_continuous_batch(), [])
        self.assertEqual(engine.prefill_calls, ["one", "two"])
        self.assertEqual(engine.batched_decode_calls, [["one", "two"]])
        self.assertEqual(
            [request.request_id for request in scheduler.running], ["one", "two"]
        )

    def test_finished_slot_is_refilled_on_the_next_iteration(self) -> None:
        engine = FakeEngine(
            prefill_tokens={"short": 10, "long": 20, "next": 30},
            decode_tokens={
                "short": [11],
                "long": [21, 22],
                "next": [31],
            },
        )
        scheduler = Scheduler(engine, max_running=2)
        short = self._request("short", max_new_tokens=2)
        long = self._request("long", max_new_tokens=3)
        next_request = self._request("next", max_new_tokens=2)
        for request in (short, long, next_request):
            scheduler.add_request(request)

        self.assertEqual(scheduler.step_continuous_batch(), [short])
        self.assertEqual(
            [request.request_id for request in scheduler.running], ["long"]
        )
        self.assertEqual(engine.prefill_calls, ["short", "long"])

        self.assertEqual(
            scheduler.step_continuous_batch(), [long, next_request]
        )
        self.assertEqual(engine.prefill_calls, ["short", "long", "next"])
        self.assertEqual(engine.batched_decode_calls, [["short", "long"], ["long", "next"]])
        self.assertEqual(scheduler.running, [])
        self.assertEqual(list(scheduler.waiting), [])

    def test_prefill_stop_does_not_enter_the_decode_batch(self) -> None:
        engine = FakeEngine(
            prefill_tokens={"stop": 99},
            decode_tokens={"stop": []},
            stop_tokens={99},
        )
        scheduler = Scheduler(engine, max_running=2)
        request = self._request("stop", max_new_tokens=8)
        scheduler.add_request(request)

        self.assertEqual(scheduler.step_continuous_batch(), [request])
        self.assertEqual(engine.batched_decode_calls, [])
        self.assertEqual(request.status, RequestStatus.FINISHED)
        self.assertIsNone(request.kv_cache)


if __name__ == "__main__":
    unittest.main()
