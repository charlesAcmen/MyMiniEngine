"""
Request scheduler — the core orchestrator of the serving engine.

The scheduler sits between the HTTP server and the model engine:

    Server  ──add_request()──▶  Scheduler  ──prefill/decode──▶  Engine
      ▲                            │
      └─── token_queue (stream) ◄──┘

It runs in a background thread, repeatedly calling step() which:
  1. Admits waiting requests and prefills them  (WAITING → RUNNING)
  2. Runs one decode step on every decode-ready request
  3. Retires finished requests                    (RUNNING → FINISHED)

An unrecoverable step exception fails all outstanding requests once and
notifies their HTTP waiters instead of retrying the same broken step forever.

"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Generic, Protocol, TypeVar

from miniengine.config import SchedulerConfig
from miniengine.core import (
    KV_CAPACITY_EXHAUSTED,
    KVPoolExhaustedError,
    Request,
    RequestStatus,
    TokenOutput,
)

logger = logging.getLogger(__name__)

#Protocol in Python
#type checker
#duck typing
AdmissionT = TypeVar("AdmissionT")


class SchedulerEngine(Protocol[AdmissionT]):
    """The orchestration capabilities Scheduler requires from an engine."""

    mode: str

    def validate_request_capacity(self, request: Request) -> None: ...
    def prefill(self, request: Request) -> int: ...
    def validate_decode_admission(self, request: Request) -> None: ...
    def decode_step(self, request: Request) -> int: ...
    def batched_decode(self, requests: list[Request]) -> list[int]: ...
    def prefill_batch(self, requests: list[Request]) -> list[int]: ...
    def decode_batch(self, requests: list[Request]) -> list[int]: ...

    def try_prepare_paged_prefill(
        self,
        request: Request,
        prepared: list[AdmissionT],
    ) -> AdmissionT | None: ...

    def execute_paged_prefill(
        self,
        admissions: list[AdmissionT],
    ) -> list[int]: ...

    def prepare_retracted_request(self, request: Request) -> None: ...
    def release_request(self, request: Request) -> None: ...
    def finish_request(self, request: Request) -> None: ...
    def is_stop_token(self, token_id: int) -> bool: ...

#accepts AdmissionT as a type parameter
class Scheduler(Generic[AdmissionT]):
    """
    FCFS scheduler with naive and static-batch implementations.

    Public API (thread-safe):
        add_request(req)   — enqueue a new request
        start()            — launch the background scheduling loop
        stop()             — gracefully shut down

    Internal (called from the scheduler thread):
        step()                    — preserved naive once-at-a-time baseline
        step_static_batch()       — M1 static-batch iteration
        step_continuous_batch()   — M1 Part B iteration-level batching
    """

    def __init__(
        self, engine: SchedulerEngine, config: SchedulerConfig = SchedulerConfig()
    ) -> None:
        self.engine = engine
        self.config = config

        # Engine.mode is the single source of truth. Store the selected bound
        # method once so every scheduler iteration performs a direct callable
        # invocation instead of repeating mode dispatch in the hot loop.
        step_strategies = {
            "batched": self.step_continuous_batch,
            "paged": self.step_paged,
        }
        try:
            self._step_impl = step_strategies[engine.mode]
        except KeyError as exc:
            raise ValueError(f"unsupported engine mode: {engine.mode}") from exc

        # Queues
        self.waiting: deque[Request] = deque()
        self.decode_ready: list[Request] = []
        # Registry includes waiting, decode-ready, and requests held in local step
        # variables, so a step failure can terminate every waiter.
        self._requests: dict[str, Request] = {}
        self._pending_cancellations: set[str] = set()

        # Thread control
        self._lock = threading.Lock()
        self._work_available = threading.Condition(self._lock)
        self._running_flag = False
        self._thread: threading.Thread | None = None

        # Stats
        self.total_finished: int = 0
        self.total_failed: int = 0
        self.total_generated_tokens: int = 0
        self.total_retracted: int = 0
        self.total_cancelled: int = 0

    # ── Public API (thread-safe) ────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """Enqueue a request for scheduling."""
        with self._work_available:
            if request.request_id in self._requests:
                raise ValueError(f"duplicate request id: {request.request_id}")
            self._requests[request.request_id] = request
            self.waiting.append(request)
            logger.debug(
                "Enqueued request %s  (prompt_len=%d, waiting=%d)",
                request.request_id,
                request.num_input_tokens,
                len(self.waiting),
            )
            self._work_available.notify()
    #HTTP thread send cancel request,rather clearup directly
    def cancel_request(self, request_id: str) -> bool:
        """Queue cancellation without releasing engine resources off-thread."""
        #scheduler thread might step() remove _req, read _pending
        with self._work_available:
            if request_id not in self._requests:
                return False
            self._pending_cancellations.add(request_id)
            # scheduler thread:
            # self._work_available.wait_for(
            #     lambda:
            #         not self._running_flag
            #         or bool(self.waiting)
            #         or bool(self.decode_ready)
            #         or bool(self._pending_cancellations)
            # )
            self._work_available.notify()
            return True

    def start(self) -> None:
        """Start the scheduler loop in a background daemon thread."""
        with self._work_available:
            self._running_flag = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread to join."""
        with self._work_available:
            self._running_flag = False
            # An idle scheduler may be blocked in wait_for indefinitely.
            self._work_available.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped")

    # ── Main loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            with self._work_available:
                self._work_available.wait_for(
                    lambda: not self._running_flag
                    or bool(self.waiting)
                    or bool(self.decode_ready)
                    or bool(self._pending_cancellations)
                )
                if not self._running_flag:
                    return
            try:
                self._step_impl()
            except Exception as exc:
                logger.exception(
                    "Scheduler step failed; failing %d outstanding request(s)",
                    len(self._requests),
                )
                self._fail_outstanding_requests(exc)

    # ── Scheduling step ─────────────────────────────────────────────────

    def step(self) -> list[Request]:
        """
        One scheduling iteration — maximally naive.

        Takes one request from the waiting queue, prefills it, and decodes
        it to completion before moving on to the next request.  No batching,
        no interleaving.

        Returns list of requests that finished in this step.
        """
        self._drain_cancelled_requests()
        finished: list[Request] = []

        # ── Pick one request ────────────────────────────────────────────
        with self._lock:
            if not self.waiting:
                return finished
            req = self.waiting.popleft()

        # ── Prefill ─────────────────────────────────────────────────────
        req.status = RequestStatus.RUNNING
        token_id = self.engine.prefill(req)
        req.output_ids.append(token_id)
        self.engine.validate_decode_admission(req)
        self._stream_token(req, token_id)

        # ── Decode until finished ───────────────────────────────────────
        while not self._check_finished(req, token_id):
            token_id = self.engine.decode_step(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

        self._finish_request(req, finished)
        return finished

    def step_static_batch(self) -> list[Request]:
        """
        Run one static-batch scheduling iteration.

        A batch is admitted only after the previous batch fully drains. Its
        prompt prefill is per request; later iterations decode all survivors
        together once.

        Returns list of requests that finished in this step.
        """
        self._drain_cancelled_requests()
        finished: list[Request] = []

        if not self.decode_ready:
            # Static-batch admission: finished slots are intentionally not refilled.
            with self._lock:
                batch = [
                    self.waiting.popleft()
                    for _ in range(
                        min(self.config.max_running, len(self.waiting))
                    )
                ]

            # Prompt lengths are ragged, so prefill keeps GPU KV ownership per request.
            for req in batch:
                req.status = RequestStatus.RUNNING
                token_id = self.engine.prefill(req)
                #new token id appended
                req.output_ids.append(token_id)
                self.engine.validate_decode_admission(req)
                #Push a generated token into the request's streaming queue.
                self._stream_token(req, token_id)

                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    self.decode_ready.append(req)

            return finished

        active_requests = self.decode_ready
        # batched_decode uses a temporary padded KV workspace, then restores compact KV.
        token_ids = self.engine.batched_decode(active_requests)
        if len(token_ids) != len(active_requests):
            raise RuntimeError(
                "batched_decode must return one token for every decode-ready request"
            )

        decode_survivors: list[Request] = []
        #active_requests[i] ↔ token_ids[i]
        for req, token_id in zip(active_requests, token_ids):
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                decode_survivors.append(req)

        self.decode_ready = decode_survivors
        return finished

    def step_continuous_batch(self) -> list[Request]:
        """Run one M1 Part B iteration-level continuous-batching step.

        Phase 1 fills every currently free scheduler slot in FCFS order. Each
        prompt still needs its own prefill forward pass because prompt lengths
        are ragged, and therefore each request retains ownership of a compact
        batch-size-one KV cache. Phase 2 immediately decodes every surviving
        request, including requests admitted during this same iteration.

        A completed request releases its KV cache before this method returns.
        Its slot can consequently be admitted on the next scheduler iteration,
        without waiting for unrelated requests in the batch to finish.
        """
        self._drain_cancelled_requests()
        finished: list[Request] = []

        # A slot is free whenever no decode-ready request owns its KV cache.
        with self._lock:
            prefill_batch = [
                self.waiting.popleft()
                for _ in range(
                    min(
                        self.config.max_running - len(self.decode_ready),
                        len(self.waiting),
                    )
                )
            ]

        for req in prefill_batch:
            req.status = RequestStatus.RUNNING

        # Engine bound this call to M1's preserved per-request prefill strategy.
        token_ids = self.engine.prefill_batch(prefill_batch)
        if len(token_ids) != len(prefill_batch):
            raise RuntimeError(
                "prefill_batch must return one token for every admitted request"
            )
        for req, token_id in zip(prefill_batch, token_ids):
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                self.decode_ready.append(req)

        if not self.decode_ready:
            return finished

        # The engine builds only a temporary padded GPU KV workspace for this call.
        # It restores one compact KV cache per request before returning token_ids.
        active_requests = self.decode_ready
        token_ids = self.engine.decode_batch(active_requests)
        if len(token_ids) != len(active_requests):
            raise RuntimeError(
                "batched_decode must return one token for every decode-ready request"
            )

        decode_survivors: list[Request] = []
        # Request/token index alignment is the scheduler-to-engine batch invariant.
        for req, token_id in zip(active_requests, token_ids):
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                decode_survivors.append(req)

        self.decode_ready = decode_survivors
        return finished

    def step_paged(self) -> list[Request]:
        """Run one continuous-batching iteration over the paged KV pool.

        Admission is FCFS and capacity-aware. Newly admitted prompts are
        flattened into one packed prefill batch, then all surviving requests
        participate in one paged decode call. Engine owns capacity,
        allocation, cache borrowing, and transactional resource rollback.
        """
        self._drain_cancelled_requests()
        # M2 baseline: the original single-shot packed-prefill path.
        finished: list[Request] = []
        prefill_batch: list[Request] = []
        pool = self.engine.kv_pool
        if pool is None:
            raise RuntimeError("paged scheduler requires Engine(mode='paged')")

        available_pages = pool.num_free
        available_slots = self.config.max_running - len(self.decode_ready)

        # Stop at the first request that does not fit to preserve strict FCFS.
        # Later requests never bypass a large prompt at the queue head.
        with self._lock:
            for _ in range(min(available_slots, len(self.waiting))):
                request = self.waiting[0]
                pages_needed = pool.pages_needed(request.num_input_tokens)
                if pages_needed > available_pages:
                    break
                prefill_batch.append(self.waiting.popleft())
                available_pages -= pages_needed

        if prefill_batch:
            for request in prefill_batch:
                request.status = RequestStatus.RUNNING
            try:
                #packed prefill
                token_ids = self.engine.prefill_batch(prefill_batch)
            except Exception:
                # Engine rolls back page ownership. Restore FCFS queue order so
                # a transient kernel failure does not silently lose requests.
                with self._lock:
                    for request in reversed(prefill_batch):
                        request.status = RequestStatus.WAITING
                        self.waiting.appendleft(request)
                raise

            if len(token_ids) != len(prefill_batch):
                raise RuntimeError(
                    "paged_prefill must return one token for every admitted request"
                )
            for request, token_id in zip(prefill_batch, token_ids):
                request.output_ids.append(token_id)
                self._stream_token(request, token_id)
                if self._check_finished(request, token_id):
                    self._finish_request(request, finished)
                else:
                    self.decode_ready.append(request)

        if not self.decode_ready:
            return finished

        active_requests = self.decode_ready
        try:
            token_ids = self.engine.decode_batch(active_requests)
        except KVPoolExhaustedError as exc:
            if self._retract_decode_victim(active_requests, exc):
                #RUNNING -> WAITING,Release KV cache, and requeue the victim for later replay.
                return finished
            #retraction is not helping
            self._fail_decode_capacity_victim(active_requests, exc)
            #RUNNING -> FAILED,fail one request and continue with the rest of the active set.
            return finished
        if len(token_ids) != len(active_requests):
            raise RuntimeError(
                "paged_decode must return one token for every decode-ready request"
            )

        decode_survivors: list[Request] = []
        for request, token_id in zip(active_requests, token_ids):
            request.output_ids.append(token_id)
            self._stream_token(request, token_id)
            if self._check_finished(request, token_id):
                self._finish_request(request, finished)
            else:
                decode_survivors.append(request)

        self.decode_ready = decode_survivors
        return finished

    # ── Helpers ─────────────────────────────────────────────────────────

    def _drain_cancelled_requests(self) -> None:
        """Release cancelled requests on the scheduler's owning thread."""
        #HTTP thread might have called cancel_request() to enqueue a cancellation
        #new req might be added to waiting
        #scheduler itself might change registry and queue
        with self._lock:
        #goal:remove cancelled requests from registry and queues, then release their resources off-thread
            #cancel_request() put id in _pending_cancellations
            request_ids = self._pending_cancellations
            #set helps to remove duplicates
            self._pending_cancellations = set()
            requests = [
                request
                for request_id in request_ids
                #None:return default value if key not found,avoid KeyError
                #this is quite important:req just finished and added to _pending queue
                #should not be popped here,so None is required here
                if (request := self._requests.pop(request_id, None)) is not None
            ]
            #all cancelled requests are finished
            if not requests:
                return
            cancelled_ids = {request.request_id for request in requests}
            #remove those requests to be cancelled from waiting and decode_ready
            self.waiting = deque(
                request
                for request in self.waiting
                if request.request_id not in cancelled_ids
            )
            self.decode_ready = [
                request
                for request in self.decode_ready
                if request.request_id not in cancelled_ids
            ]
        #exit the lock(resource release may be slow)
        for request in requests:
            #return private kv pages,do not free prefix-cache pages directly
            #cancel cache_node pin,clear up page_table and kv_seq_len
            self.engine.release_request(request)
            #WAITING/RUNNING → CANCELLED
            request.status = RequestStatus.CANCELLED
            request.token_queue.put(TokenOutput(token_id=-1, finished=True))
        #exclude finished reqs
        self.total_cancelled += len(requests)
        logger.info("Cancelled %d disconnected request(s)", len(requests))

    def _retract_decode_victim(
        self,
        active_requests: list[Request],
        exc: KVPoolExhaustedError,
    ) -> bool:
        """Reclaim one active request and replay its emitted context later.

        A single active request cannot benefit: releasing and immediately
        rebuilding the same sequence cannot create capacity for its next page.
        """
        if not self.config.retraction_enabled or len(active_requests) <= 1:
            #not enabled or single req retract wont work
            return False

        victim = max(
            active_requests,
            key=lambda request: (
                #select req with most remaining tokens to generate first 
                request.max_new_tokens - request.num_output_tokens,
                #select latest arrived request to retract firsts to secure FCFS order
                request.arrival_time,
            ),
        )
        self.engine.prepare_retracted_request(victim)
        #RUNNING -> WAITING
        victim.status = RequestStatus.WAITING

        #sync with queue statusd
        self.decode_ready = [
            request for request in self.decode_ready if request is not victim
        ]
        with self._lock:
            self.waiting.appendleft(victim)
        self.total_retracted += 1
        # Example message body:
        # Retracted request req-b after KV pool exhausted: requested 1, have 0
        # free pages (remaining_tokens=80, replay_tokens=769, active_survivors=1)
        logger.warning(
            "Retracted request %s after %s (remaining_tokens=%d, "
            "replay_tokens=%d, active_survivors=%d)",
            victim.request_id,
            exc,
            victim.max_new_tokens - victim.num_output_tokens,
            victim.num_input_tokens,
            len(self.decode_ready),
        )
        return True

    def _fail_decode_capacity_victim(
        self,
        active_requests: list[Request],
        exc: KVPoolExhaustedError,
    ) -> None:
        """Fail one request when decode cannot reclaim enough KV capacity."""
        if not active_requests:
            raise RuntimeError("KV exhaustion reported without an active request")

        victim = max(
            active_requests,
            key=lambda request: (
                #same as _retract_decode_victim
                request.max_new_tokens - request.num_output_tokens,
                request.arrival_time,
            ),
        )
        #finish_request() is used for successfully finished requests
        self.engine.release_request(victim)
        victim.status = RequestStatus.FAILED
        self.decode_ready = [
            request for request in self.decode_ready if request is not victim
        ]
        with self._lock:
            #global registry
            self._requests.pop(victim.request_id, None)

        message = f"{type(exc).__name__}: {exc}"[:2000]
        victim.token_queue.put(
            TokenOutput(
                token_id=-1,
                finished=True,
                error=message,
                error_code=KV_CAPACITY_EXHAUSTED,
            )
        )
        self.total_failed += 1
        logger.error(
            "Failed request %s with %s; scheduler continues with %d active and "
            "%d waiting request(s)",
            victim.request_id,
            KV_CAPACITY_EXHAUSTED,
            len(self.decode_ready),
            len(self.waiting),
        )

    def _check_finished(self, req: Request, token_id: int) -> bool:
        """Decide whether a request should stop generating."""
        if req.is_finished:
            return True
        if not req.ignore_eos and self.engine.is_stop_token(token_id):
            return True
        return False

    def _stream_token(self, req: Request, token_id: int) -> None:
        """Push a generated token id into the request's response queue."""
        req.token_queue.put(TokenOutput(token_id=token_id, finished=False))

    def _finish_request(self, req: Request, finished_list: list[Request]) -> None:
        """Mark a request as finished and free its resources."""
        req.status = RequestStatus.FINISHED
        self.engine.finish_request(req)
        req.token_queue.put(TokenOutput(token_id=-1, finished=True))
        finished_list.append(req)

        with self._lock:
            self._requests.pop(req.request_id, None)

        self.total_finished += 1
        self.total_generated_tokens += req.num_output_tokens
        logger.debug(
            "Finished request %s  (output_len=%d, decode_ready=%d, waiting=%d)",
            req.request_id,
            req.num_output_tokens,
            len(self.decode_ready),
            len(self.waiting),
        )

    def _fail_outstanding_requests(self, exc: Exception) -> None:
        """Fail all requests owned by the step that raised, exactly once.

        The registry also covers requests temporarily removed from ``waiting``
        during admission. Clearing all outstanding work is deliberately
        fail-stop: retrying an unchanged CUDA or model failure would otherwise
        spin, flood logs, and leave HTTP clients blocked forever.
        """
        message = f"{type(exc).__name__}: {exc}"[:2000]
        with self._lock:
            requests = list(self._requests.values())
            self._requests.clear()
            self.waiting.clear()
            self.decode_ready = []
            self._pending_cancellations.clear()

        for request in requests:
            try:
                self.engine.release_request(request)
            except Exception:
                logger.exception(
                    "Failed to release resources for request %s",
                    request.request_id,
                )
            request.status = RequestStatus.FAILED
            request.token_queue.put(
                TokenOutput(
                    token_id=-1,
                    finished=True,
                    error=message,
                )
            )
        self.total_failed += len(requests)
