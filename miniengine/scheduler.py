"""
Request scheduler — the core orchestrator of the serving engine.

The scheduler sits between the HTTP server and the model engine:

    Server  ──add_request()──▶  Scheduler  ──prefill/decode──▶  Engine
      ▲                            │
      └─── token_queue (stream) ◄──┘

It runs in a background thread, repeatedly calling step() which:
  1. Admits waiting requests and prefills them  (WAITING → RUNNING)
  2. Runs one decode step on every running request
  3. Retires finished requests                    (RUNNING → FINISHED)

"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from miniengine.core import Request, RequestStatus, TokenOutput
from miniengine.engine import Engine

logger = logging.getLogger(__name__)


class Scheduler:
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

    def __init__(self, engine: Engine, max_running: int = 16):
        self.engine = engine
        self.max_running = max_running

        # Queues
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        # Thread control
        self._lock = threading.Lock()
        self._running_flag = False
        self._thread: threading.Thread | None = None

        # Stats
        self.total_finished: int = 0
        self.total_generated_tokens: int = 0

    # ── Public API (thread-safe) ────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """Enqueue a request for scheduling."""
        with self._lock:
            self.waiting.append(request)
            logger.info(
                "Enqueued request %s  (prompt_len=%d, waiting=%d)",
                request.request_id,
                request.num_input_tokens,
                len(self.waiting),
            )

    def start(self) -> None:
        """Start the scheduler loop in a background daemon thread."""
        self._running_flag = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread to join."""
        self._running_flag = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped")

    # ── Main loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running_flag:
            has_work = bool(self.waiting) or bool(self.running)
            if not has_work:
                time.sleep(0.005)  # idle sleep to avoid busy-waiting
                continue
            try:
                self.step_continuous_batch()
            except Exception:
                logger.exception("Scheduler step failed")

    # ── Scheduling step ─────────────────────────────────────────────────

    def step(self) -> list[Request]:
        """
        One scheduling iteration — maximally naive.

        Takes one request from the waiting queue, prefills it, and decodes
        it to completion before moving on to the next request.  No batching,
        no interleaving.

        Returns list of requests that finished in this step.
        """
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
        finished: list[Request] = []

        if not self.running:
            # Static-batch admission: finished slots are intentionally not refilled.
            with self._lock:
                batch = [
                    self.waiting.popleft()
                    for _ in range(min(self.max_running, len(self.waiting)))
                ]

            # Prompt lengths are ragged, so prefill keeps GPU KV ownership per request.
            for req in batch:
                req.status = RequestStatus.RUNNING
                token_id = self.engine.prefill(req)
                #new token id appended
                req.output_ids.append(token_id)
                #Push a generated token into the request's streaming queue.
                self._stream_token(req, token_id)

                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    self.running.append(req)

            return finished

        active_requests = self.running
        # batched_decode uses a temporary padded KV workspace, then restores compact KV.
        token_ids = self.engine.batched_decode(active_requests)
        if len(token_ids) != len(active_requests):
            raise RuntimeError(
                "batched_decode must return one token for every running request"
            )

        surviving_requests: list[Request] = []
        #active_requests[i] ↔ token_ids[i]
        for req, token_id in zip(active_requests, token_ids):
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                surviving_requests.append(req)

        self.running = surviving_requests
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
        finished: list[Request] = []

        # A slot is free whenever no RUNNING request currently owns its KV cache.
        with self._lock:
            admitted_requests = [
                self.waiting.popleft()
                for _ in range(
                    min(self.max_running - len(self.running), len(self.waiting))
                )
            ]

        # Prefill produces each request's first sampled token and compact KV cache.
        for req in admitted_requests:
            req.status = RequestStatus.RUNNING
            token_id = self.engine.prefill(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                self.running.append(req)

        if not self.running:
            return finished

        # The engine builds only a temporary padded GPU KV workspace for this call.
        # It restores one compact KV cache per request before returning token_ids.
        active_requests = self.running
        token_ids = self.engine.batched_decode(active_requests)
        if len(token_ids) != len(active_requests):
            raise RuntimeError(
                "batched_decode must return one token for every running request"
            )

        surviving_requests: list[Request] = []
        # Request/token index alignment is the scheduler-to-engine batch invariant.
        for req, token_id in zip(active_requests, token_ids):
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                surviving_requests.append(req)

        self.running = surviving_requests
        return finished

    # ── Helpers ─────────────────────────────────────────────────────────

    def _check_finished(self, req: Request, token_id: int) -> bool:
        """Decide whether a request should stop generating."""
        if req.is_finished:
            return True
        if self.engine.is_stop_token(token_id):
            return True
        return False

    def _stream_token(self, req: Request, token_id: int) -> None:
        """Push a generated token into the request's streaming queue."""
        text = self.engine.decode_token(token_id)
        req.token_queue.put(TokenOutput(token_id=token_id, token_text=text, finished=False))

    def _finish_request(self, req: Request, finished_list: list[Request]) -> None:
        """Mark a request as finished and free its resources."""
        req.status = RequestStatus.FINISHED
        req.kv_cache = None  # release GPU memory
        req.token_queue.put(TokenOutput(token_id=-1, token_text="", finished=True))
        finished_list.append(req)

        self.total_finished += 1
        self.total_generated_tokens += req.num_output_tokens
        logger.info(
            "Finished request %s  (output_len=%d, running=%d, waiting=%d)",
            req.request_id,
            req.num_output_tokens,
            len(self.running),
            len(self.waiting),
        )
