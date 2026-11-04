"""CLI boundary tests: invalid numeric flags never construct an Engine."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
from pydantic import ValidationError

from miniengine.__main__ import parse_args
from miniengine.core import SamplingConfig
from miniengine.server import (
    ChatCompletionRequest,
    ChatMessage,
    validate_sampling_overrides,
)


class ServerCliTests(unittest.TestCase):
    def test_default_page_size_matches_flash_attention_paged_kv(self) -> None:
        args = parse_args(["--model", "local-model"])
        self.assertEqual(args.page_size, 256)
        self.assertEqual(args.mode, "batched")
        self.assertEqual(args.sampling_config, SamplingConfig())

    def test_paged_mode_is_selectable(self) -> None:
        args = parse_args(["--model", "local-model", "--mode", "paged"])
        self.assertEqual(args.mode, "paged")

    def test_valid_pool_flags_are_parsed_before_engine_construction(self) -> None:
        args = parse_args(
            [
                "--model",
                "local-model",
                "--page-size",
                "16",
                "--mem-fraction-static",
                "0.75",
            ]
        )
        self.assertEqual(args.page_size, 16)
        self.assertEqual(args.mem_fraction_static, 0.75)

    def test_invalid_numeric_flags_exit_from_argparse(self) -> None:
        invalid_flags = [
            ["--page-size", "0"],
            ["--mem-fraction-static", "1.0"],
            ["--max-running", "0"],
            ["--port", "65536"],
        ]
        for flags in invalid_flags:
            with (
                self.subTest(flags=flags),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(["--model", "local-model", *flags])

    def test_sampling_config_is_loaded_from_json(self) -> None:
        config_path = Path(__file__).parents[1] / "configs" / "sampling-greedy.json"
        args = parse_args(
            ["--model", "local-model", "--sampling-config", str(config_path)]
        )
        self.assertEqual(args.sampling_config.temperature, 0.0)

    def test_invalid_sampling_config_fails_during_argument_parsing(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            handle.write('{"top_p": 0}')
            handle.flush()
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(
                    ["--model", "local-model", "--sampling-config", handle.name]
                )


class SamplingOverrideTests(unittest.TestCase):
    @staticmethod
    def _request(**overrides: float | int) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")], **overrides
        )

    def test_omitted_or_matching_overrides_are_accepted(self) -> None:
        config = SamplingConfig()
        validate_sampling_overrides(self._request(), config)
        validate_sampling_overrides(
            self._request(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.0,
            ),
            config,
        )

    def test_conflicting_override_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            validate_sampling_overrides(
                self._request(temperature=1.0), SamplingConfig()
            )
        self.assertEqual(raised.exception.status_code, 422)

    def test_non_positive_max_tokens_is_rejected_at_http_boundary(self) -> None:
        with self.assertRaises(ValidationError):
            self._request(max_tokens=0)
