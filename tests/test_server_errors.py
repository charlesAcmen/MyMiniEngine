"""HTTP response helpers surface terminal scheduler failures."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from miniengine.config import SamplingConfig
from miniengine.core import (
    KV_CAPACITY_EXHAUSTED,
    REQUEST_TOO_LARGE,
    Request,
    RequestTooLargeError,
    TokenOutput,
)
from miniengine.server import (
    ChatCompletionRequest,
    Runtime,
    _collect_full_response,
    _require_runtime,
    _stream_response,
    app,
    chat_completions,
    lifespan,
)


class ServerFailureResponseTest(unittest.TestCase):
    _unused_engine = object()

    @staticmethod
    def _failed_request() -> Request:
        request = Request(request_id="failed", input_ids=[1])
        request.token_queue.put(
            TokenOutput(
                token_id=-1,
                finished=True,
                error="RuntimeError: kernel failed",
            )
        )
        return request

    @staticmethod
    def _capacity_failed_request() -> Request:
        request = Request(request_id="capacity-failed", input_ids=[1])
        request.token_queue.put(
            TokenOutput(
                token_id=-1,
                finished=True,
                error="KVPoolExhaustedError: KV pool exhausted",
                error_code=KV_CAPACITY_EXHAUSTED,
            )
        )
        return request

    def test_uninitialized_runtime_becomes_http_503(self) -> None:
        with patch.object(app.state, "runtime", None, create=True):
            with self.assertRaises(HTTPException) as raised:
                _require_runtime()
        self.assertEqual(raised.exception.status_code, 503)

    def test_lifespan_starts_and_stops_installed_scheduler(self) -> None:
        class FakeScheduler:
            def __init__(self) -> None:
                self.started = 0
                self.stopped = 0

            def start(self) -> None:
                self.started += 1

            def stop(self) -> None:
                self.stopped += 1

        scheduler = FakeScheduler()
        runtime = Runtime(
            engine=object(),
            scheduler=scheduler,
            model_id="test-model",
        )

        async def exercise_lifespan() -> None:
            async with lifespan(app):
                self.assertEqual(scheduler.started, 1)
                self.assertEqual(scheduler.stopped, 0)
                self.assertIs(_require_runtime(), runtime)

        with patch.object(app.state, "runtime", runtime, create=True):
            asyncio.run(exercise_lifespan())

        self.assertEqual(scheduler.started, 1)
        self.assertEqual(scheduler.stopped, 1)

    def test_non_streaming_failure_becomes_http_500(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                _collect_full_response(self._failed_request(), self._unused_engine)
            )
        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(raised.exception.detail, "RuntimeError: kernel failed")

    def test_non_streaming_capacity_failure_becomes_http_503(self) -> None:
        request = self._capacity_failed_request()
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(_collect_full_response(request, self._unused_engine))
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            raised.exception.detail,
            {
                "code": KV_CAPACITY_EXHAUSTED,
                "message": "KVPoolExhaustedError: KV pool exhausted",
                "request_id": request.request_id,
            },
        )

    def test_impossible_prompt_becomes_http_422_before_streaming(self) -> None:
        class RejectingScheduler:
            @staticmethod
            def add_request(_request: Request) -> None:
                raise RequestTooLargeError(required_pages=6, usable_pages=5)

        runtime = Runtime(
            engine=SimpleNamespace(
                sampling_config=SamplingConfig(),
                tokenize_messages=lambda _messages: list(range(21)),
            ),
            scheduler=RejectingScheduler(),
            model_id="test-model",
        )
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "too large"}],
            stream=True,
        )

        with patch.object(app.state, "runtime", runtime, create=True):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(chat_completions(request))

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail["code"], REQUEST_TOO_LARGE)
        self.assertEqual(raised.exception.detail["required_pages"], 6)
        self.assertEqual(raised.exception.detail["usable_pages"], 5)

    def test_streaming_failure_emits_error_event_and_done(self) -> None:
        async def collect() -> list[str]:
            return [item async for item in _stream_response(
                self._failed_request(), "model", self._unused_engine
            )]

        events = asyncio.run(collect())
        payload = json.loads(events[0].removeprefix("data: "))
        self.assertEqual(payload["error"]["type"], "server_error")
        self.assertEqual(payload["error"]["message"], "RuntimeError: kernel failed")
        self.assertEqual(events[1], "data: [DONE]\n\n")

    def test_streaming_capacity_failure_emits_stable_code_and_done(self) -> None:
        async def collect() -> list[str]:
            return [item async for item in _stream_response(
                self._capacity_failed_request(), "model", self._unused_engine
            )]

        events = asyncio.run(collect())
        payload = json.loads(events[0].removeprefix("data: "))
        self.assertEqual(payload["error"]["code"], KV_CAPACITY_EXHAUSTED)
        self.assertEqual(payload["error"]["request_id"], "capacity-failed")
        self.assertEqual(events[1], "data: [DONE]\n\n")

    def test_closing_stream_queues_request_cancellation(self) -> None:
        request = Request(request_id="disconnected", input_ids=[1])
        request.token_queue.put(TokenOutput(token_id=7, finished=False))
        cancelled: list[str] = []
        decoder = SimpleNamespace(step=lambda _token_id: "token")
        response_engine = SimpleNamespace(
            create_incremental_decoder=lambda: decoder
        )

        async def consume_one_then_disconnect() -> None:
            stream = _stream_response(
                request,
                "model",
                response_engine,
                lambda request_id: not cancelled.append(request_id),
            )
            self.assertIn("token", await anext(stream))
            await stream.aclose()

        asyncio.run(consume_one_then_disconnect())
        self.assertEqual(cancelled, ["disconnected"])

    def test_terminal_stream_does_not_cancel_completed_request(self) -> None:
        request = Request(request_id="complete", input_ids=[1])
        request.token_queue.put(TokenOutput(token_id=-1, finished=True))
        cancelled: list[str] = []

        async def collect() -> list[str]:
            return [item async for item in _stream_response(
                request,
                "model",
                self._unused_engine,
                lambda request_id: not cancelled.append(request_id),
            )]

        events = asyncio.run(collect())
        self.assertEqual(events[-1], "data: [DONE]\n\n")
        self.assertEqual(cancelled, [])
