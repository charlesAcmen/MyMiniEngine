"""Unicode-correct HTTP decoding with the real Qwen tokenizer."""

from __future__ import annotations

import asyncio
import json
import unittest

from transformers import AutoTokenizer

from miniengine.core import Request, TokenOutput
from miniengine.server import _collect_full_response, _stream_response


class _TokenizerOnlyEngine:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            trust_remote_code=True,
            local_files_only=True,
        )

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


class ServerUnicodeDecodingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = _TokenizerOnlyEngine()

    @staticmethod
    def _completed_request(token_ids: list[int]) -> Request:
        request = Request(request_id="unicode", input_ids=[1])
        request.output_ids.extend(token_ids)
        for token_id in token_ids:
            request.token_queue.put(TokenOutput(token_id=token_id, finished=False))
        request.token_queue.put(TokenOutput(token_id=-1, finished=True))
        return request

    def test_non_streaming_decodes_complete_unicode_sequence(self) -> None:
        request = self._completed_request([82912, 246])
        text = asyncio.run(_collect_full_response(request, self.engine))
        self.assertEqual(text, "龘")

    def test_streaming_chunks_join_to_complete_unicode_sequence(self) -> None:
        request = self._completed_request([82912, 246])

        async def collect() -> list[str]:
            return [
                item
                async for item in _stream_response(request, "qwen", self.engine)
            ]

        events = asyncio.run(collect())
        chunks = [
            json.loads(event.removeprefix("data: "))
            for event in events
            if event != "data: [DONE]\n\n"
        ]
        text = "".join(
            chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
        )
        self.assertEqual(text, "龘")

    def test_non_streaming_decodes_split_emoji(self) -> None:
        request = self._completed_request([9284, 104, 254])
        text = asyncio.run(_collect_full_response(request, self.engine))
        self.assertEqual(text, "🫠")


if __name__ == "__main__":
    unittest.main()
