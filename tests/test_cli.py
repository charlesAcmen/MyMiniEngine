"""CLI boundary tests: invalid numeric flags never construct an Engine."""

from __future__ import annotations

import contextlib
import io
import unittest

from miniengine.__main__ import parse_args


class ServerCliTests(unittest.TestCase):
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
