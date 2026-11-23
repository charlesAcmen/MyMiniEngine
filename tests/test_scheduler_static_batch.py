"""Unit tests for M1 Part A's static-batch Scheduler behavior."""

from collections import deque
import threading
import unittest

from miniengine.config import SchedulerConfig
from miniengine.core import Request, RequestStatus
from miniengine.scheduler import Scheduler


class FakeEngine:
    """Deterministic CPU-only engine that records scheduler interactions."""

    def __init__(
        self,
        prefill_tokens: dict[str, int],
        decode_tokens: dict[str, list[int]],
        stop_tokens: set[int] | None = None,
    ) -> None:
        self.mode = "batched"
        self.prefill_tokens = prefill_tokens
        self.decode_tokens = {
            request_id: deque(tokens)
            for request_id, tokens in decode_tokens.items()
        }
        self.stop_tokens = stop_tokens or set()
        self.prefill_calls: list[str] = []
        self.batched_decode_calls: list[list[str]] = []
        self.decode_admission_validations: list[str] = []
        self.prefill_batch = self._prefill_batch
        self.decode_batch = self.batched_decode

    def _prefill_batch(self, requests: list[Request]) -> list[int]:
        token_ids = []
        for request in requests:
            token_ids.append(self.prefill(request))
            self.validate_decode_admission(request)
        return token_ids

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

    def validate_decode_admission(self, request: Request) -> None:
        self.decode_admission_validations.append(request.request_id)

    @staticmethod
    def release_request(request: Request) -> None:
        request.kv_cache = None

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_tokens


class StaticBatchSchedulerTest(unittest.TestCase):
    @staticmethod
    def _request(request_id: str, max_new_tokens: int = 3) -> Request:
        return Request(
            request_id=request_id,
            input_ids=[1, 2],
            max_new_tokens=max_new_tokens,
        )

    @staticmethod
    def _drain_stream(request: Request) -> list[tuple[int, bool]]:
        items: list[tuple[int, bool]] = []
        while not request.token_queue.empty():
            token = request.token_queue.get_nowait()
            items.append((token.token_id, token.finished))
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
        scheduler = Scheduler(engine, SchedulerConfig(max_running=2))
        requests = [self._request(name) for name in ("one", "two", "three")]
        for request in requests:
            scheduler.add_request(request)

        self.assertEqual(scheduler.step_static_batch(), [])
        self.assertEqual(engine.prefill_calls, ["one", "two"])
        self.assertEqual(engine.decode_admission_validations, ["one", "two"])
        self.assertEqual(
            [request.request_id for request in scheduler.decode_ready], ["one", "two"]
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
        self.assertEqual(scheduler.decode_ready, [])
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
        scheduler = Scheduler(engine, SchedulerConfig(max_running=2))
        short = self._request("short", max_new_tokens=2)
        long = self._request("long", max_new_tokens=3)
        next_request = self._request("next", max_new_tokens=2)
        for request in (short, long, next_request):
            scheduler.add_request(request)

        scheduler.step_static_batch()
        self.assertEqual(scheduler.step_static_batch(), [short])
        self.assertEqual(
            [request.request_id for request in scheduler.decode_ready], ["long"]
        )
        self.assertEqual(
            [request.request_id for request in scheduler.waiting], ["next"]
        )
        self.assertEqual(engine.prefill_calls, ["short", "long"])

        self.assertEqual(scheduler.step_static_batch(), [long])
        self.assertEqual(scheduler.decode_ready, [])
        self.assertEqual(engine.prefill_calls, ["short", "long"])

        scheduler.step_static_batch()
        self.assertEqual(engine.prefill_calls, ["short", "long", "next"])

    def test_completion_releases_kv_emits_sentinel_and_updates_stats(self) -> None:
        engine = FakeEngine(prefill_tokens={"done": 40}, decode_tokens={"done": [41]})
        scheduler = Scheduler(engine, SchedulerConfig(max_running=2))
        request = self._request("done", max_new_tokens=2)
        scheduler.add_request(request)

        scheduler.step_static_batch()
        self.assertIsNotNone(request.kv_cache)
        self.assertEqual(scheduler.step_static_batch(), [request])

        self.assertEqual(request.status, RequestStatus.FINISHED)
        self.assertIsNone(request.kv_cache)
        self.assertEqual(
            self._drain_stream(request),
            [(40, False), (41, False), (-1, True)],
        )
        self.assertEqual(scheduler.total_finished, 1)
        self.assertEqual(scheduler.total_generated_tokens, 2)

    def test_step_failure_fails_all_outstanding_requests_once(self) -> None:
        engine = FakeEngine(
            prefill_tokens={"running": 10, "waiting": 20},
            decode_tokens={"running": [], "waiting": []},
        )
        scheduler = Scheduler(engine, SchedulerConfig(max_running=2))
        decode_ready_request = self._request("running")
        waiting = self._request("waiting")
        scheduler.add_request(decode_ready_request)
        scheduler.add_request(waiting)

        scheduler.waiting.popleft()
        decode_ready_request.status = RequestStatus.RUNNING
        decode_ready_request.kv_cache = object()
        scheduler.decode_ready.append(decode_ready_request)
        scheduler._fail_outstanding_requests(RuntimeError("kernel failed"))

        self.assertEqual(decode_ready_request.status, RequestStatus.FAILED)
        self.assertEqual(waiting.status, RequestStatus.FAILED)
        self.assertIsNone(decode_ready_request.kv_cache)
        self.assertEqual(list(scheduler.waiting), [])
        self.assertEqual(scheduler.decode_ready, [])
        self.assertEqual(scheduler.total_failed, 2)
        for request in (decode_ready_request, waiting):
            terminal = request.token_queue.get_nowait()
            self.assertTrue(terminal.finished)
            self.assertEqual(terminal.error, "RuntimeError: kernel failed")

    def test_prefill_stop_finishes_without_entering_batched_decode(self) -> None:
        engine = FakeEngine(
            prefill_tokens={"stop": 99},
            decode_tokens={"stop": []},
            stop_tokens={99},
        )
        scheduler = Scheduler(engine, SchedulerConfig(max_running=2))
        request = self._request("stop", max_new_tokens=8)
        scheduler.add_request(request)

        self.assertEqual(scheduler.step_static_batch(), [request])
        self.assertEqual(scheduler.decode_ready, [])
        self.assertEqual(engine.batched_decode_calls, [])
        self.assertEqual(request.status, RequestStatus.FINISHED)
        self.assertIsNone(request.kv_cache)
        self.assertEqual(self._drain_stream(request)[-1], (-1, True))
        self.assertEqual(scheduler.total_finished, 1)
        self.assertEqual(scheduler.total_generated_tokens, 1)

    def test_ignore_eos_runs_to_the_configured_length(self) -> None:
        engine = FakeEngine(
            prefill_tokens={"fixed": 99},
            decode_tokens={"fixed": [10, 11]},
            stop_tokens={99},
        )
        scheduler = Scheduler(engine, SchedulerConfig(max_running=1))
        request = self._request("fixed", max_new_tokens=3)
        request.ignore_eos = True
        scheduler.add_request(request)

        self.assertEqual(scheduler.step_static_batch(), [])
        self.assertEqual(scheduler.step_static_batch(), [])
        self.assertEqual(scheduler.step_static_batch(), [request])
        self.assertEqual(request.output_ids, [99, 10, 11])


class ContinuousBatchSchedulerTest(unittest.TestCase):
    """Deterministic tests for M1 Part B iteration-level batch admission."""

    @staticmethod
    def _request(request_id: str, max_new_tokens: int = 3) -> Request:
        return Request(
            request_id=request_id,
            input_ids=[1, 2],
            max_new_tokens=max_new_tokens,
        )

    def test_newly_prefilled_requests_join_decode_in_the_same_iteration(self) -> None:
        engine = FakeEngine(
            prefill_tokens={"one": 11, "two": 21},
            decode_tokens={"one": [12], "two": [22]},
        )
        scheduler = Scheduler(engine, SchedulerConfig(max_running=2))
        requests = [self._request(name) for name in ("one", "two")]
        for request in requests:
            scheduler.add_request(request)

        self.assertEqual(scheduler.step_continuous_batch(), [])
        self.assertEqual(engine.prefill_calls, ["one", "two"])
        self.assertEqual(engine.batched_decode_calls, [["one", "two"]])
        self.assertEqual(
            [request.request_id for request in scheduler.decode_ready], ["one", "two"]
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
        scheduler = Scheduler(engine, SchedulerConfig(max_running=2))
        short = self._request("short", max_new_tokens=2)
        long = self._request("long", max_new_tokens=3)
        next_request = self._request("next", max_new_tokens=2)
        for request in (short, long, next_request):
            scheduler.add_request(request)

        self.assertEqual(scheduler.step_continuous_batch(), [short])
        self.assertEqual(
            [request.request_id for request in scheduler.decode_ready], ["long"]
        )
        self.assertEqual(engine.prefill_calls, ["short", "long"])

        self.assertEqual(
            scheduler.step_continuous_batch(), [long, next_request]
        )
        self.assertEqual(engine.prefill_calls, ["short", "long", "next"])
        self.assertEqual(engine.batched_decode_calls, [["short", "long"], ["long", "next"]])
        self.assertEqual(scheduler.decode_ready, [])
        self.assertEqual(list(scheduler.waiting), [])

    def test_prefill_stop_does_not_enter_the_decode_batch(self) -> None:
        engine = FakeEngine(
            prefill_tokens={"stop": 99},
            decode_tokens={"stop": []},
            stop_tokens={99},
        )
        scheduler = Scheduler(engine, SchedulerConfig(max_running=2))
        request = self._request("stop", max_new_tokens=8)
        scheduler.add_request(request)

        self.assertEqual(scheduler.step_continuous_batch(), [request])
        self.assertEqual(engine.batched_decode_calls, [])
        self.assertEqual(request.status, RequestStatus.FINISHED)
        self.assertIsNone(request.kv_cache)


class SchedulerLoopWakeupTests(unittest.TestCase):
    def test_idle_loop_wakes_for_new_work_and_stop_wakes_idle_loop(self) -> None:
        engine = FakeEngine(prefill_tokens={}, decode_tokens={})
        scheduler = Scheduler(engine, SchedulerConfig(max_running=1))
        step_called = threading.Event()

        def record_one_step() -> list[Request]:
            # Consume the synthetic work so the loop blocks again after this step.
            with scheduler._lock:
                scheduler.waiting.clear()
                scheduler._requests.clear()
            step_called.set()
            return []

        scheduler._step_impl = record_one_step
        scheduler.start()
        try:
            scheduler.add_request(
                Request(
                    request_id="wake",
                    input_ids=[1],
                    max_new_tokens=1,
                )
            )
            self.assertTrue(
                step_called.wait(timeout=1.0),
                "new work did not wake the idle scheduler",
            )
        finally:
            scheduler.stop()

        self.assertIsNotNone(scheduler._thread)
        self.assertFalse(scheduler._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
