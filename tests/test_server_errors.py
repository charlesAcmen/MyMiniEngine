"""HTTP response helpers surface terminal scheduler failures."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from miniengine.core import Request, TokenOutput
from miniengine.server import (
    _collect_full_response,
    _require_runtime,
    _stream_response,
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

    def test_uninitialized_runtime_becomes_http_503(self) -> None:
        with (
            patch("miniengine.server.engine", None),
            patch("miniengine.server.scheduler", None),
            self.assertRaises(HTTPException) as raised,
        ):
            _require_runtime()
        self.assertEqual(raised.exception.status_code, 503)

    def test_non_streaming_failure_becomes_http_500(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                _collect_full_response(self._failed_request(), self._unused_engine)
            )
        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(raised.exception.detail, "RuntimeError: kernel failed")

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
