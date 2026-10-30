"""Unit tests for machine-readable benchmark output and statistics."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from benchmark.bench_accuracy import (
    EvalResult,
    build_json_report as accuracy_report,
    send_eval_request,
)
from benchmark.bench_serving import (
    RequestMetrics,
    build_json_report as serving_report,
    summarize_concurrency,
)
from benchmark.result_artifacts import (
    MAX_RESULT_FILENAME_BYTES,
    atomic_write_json,
    validate_result_filename,
    write_benchmark_result,
)
from benchmark.trial_statistics import describe
from miniengine.cli import positive_int_csv


class ResultArtifactTests(unittest.TestCase):
    def test_atomic_json_and_describe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "nested" / "result.json"
            atomic_write_json(path, {"value": 3})
            self.assertEqual(json.loads(path.read_text()), {"value": 3})
            self.assertFalse(list(path.parent.glob("*.tmp")))

        stats = describe([1.0, 2.0, 3.0])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["mean"], 2.0)
        self.assertEqual(stats["median"], 2.0)
        self.assertEqual(describe([])["mean"], None)

    def test_benchmark_result_uses_date_directory_and_parameter_briefing(self) -> None:
        payload = {
            "benchmark": "serving",
            "parameters": {
                "input_len": 512,
                "output_len": 256,
                "max_tokens": None,
                "concurrencies": [1, 8],
                "num_requests": 50,
            },
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = write_benchmark_result(payload, root)
            second = write_benchmark_result(payload, root)
            self.assertRegex(first.parent.name, r"^\d{4}-\d{2}-\d{2}$")
            self.assertEqual(first.name, "serving-in512-out256-c1-8-n50.json")
            self.assertEqual(second.name, "serving-in512-out256-c1-8-n50-r02.json")

    def test_accuracy_subject_order_is_preserved_in_filename(self) -> None:
        payload = {
            "benchmark": "accuracy",
            "parameters": {
                "dataset": "mmlu",
                "subjects": "anatomy,abstract_algebra",
                "num_samples": 200,
                "concurrency": 8,
            },
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = write_benchmark_result(payload, Path(temporary_dir))
        self.assertEqual(
            path.name,
            "accuracy-mmlu-anatomy-abstract_algebra-n200-c8.json",
        )

    def test_result_filename_enforces_255_utf8_bytes(self) -> None:
        parameters = {
            "dataset": "mmlu",
            "subjects": "x",
            "num_samples": 1,
            "concurrency": 1,
        }
        one_char_stem = validate_result_filename("accuracy", parameters)
        fixed_bytes = len(f"{one_char_stem}.json".encode("utf-8")) - 1

        parameters["subjects"] = "x" * (
            MAX_RESULT_FILENAME_BYTES - fixed_bytes
        )
        stem = validate_result_filename("accuracy", parameters)
        self.assertEqual(
            len(f"{stem}.json".encode("utf-8")),
            MAX_RESULT_FILENAME_BYTES,
        )

        parameters["subjects"] += "x"
        with self.assertRaisesRegex(ValueError, "256 bytes"):
            validate_result_filename("accuracy", parameters)

    def test_concurrency_csv_preserves_order_and_rejects_duplicates(self) -> None:
        self.assertEqual(positive_int_csv("1,2,4,8,16,32"), [1, 2, 4, 8, 16, 32])
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "duplicate"):
            positive_int_csv("1,2,2,4")


class ServingJsonTests(unittest.TestCase):
    @staticmethod
    def _result(start: float, first: float, end: float, tokens: int) -> RequestMetrics:
        return RequestMetrics(
            input_len=32,
            target_output_len=4,
            start_time=start,
            first_token_time=first,
            end_time=end,
            num_output_tokens=tokens,
        )

    def test_summary_matches_request_durations(self) -> None:
        results = [self._result(0.0, 0.1, 1.0, 4), self._result(0.0, 0.2, 2.0, 4)]
        summary = summarize_concurrency(results)
        self.assertEqual(summary["successes"], 2)
        self.assertEqual(summary["total_output_tokens"], 8)
        self.assertEqual(summary["generation_tokens_per_second"], 4.0)
        self.assertAlmostEqual(summary["ttft_ms"]["p50"], 150.0)

    def test_serving_report_contains_raw_and_aggregate_results(self) -> None:
        args = Namespace(
            base_url="http://127.0.0.1:8000",
            model="local",
            num_requests=1,
            input_len=32,
            output_len=4,
            max_tokens=256,
            input_len_min_ratio=1.0,
            concurrencies=[1],
            prompt_data_dir=Path("data"),
            seed=42,
        )
        report = serving_report(args, {1: [self._result(0.0, 0.1, 1.0, 4)]})
        self.assertEqual(report["benchmark"], "serving")
        self.assertEqual(report["results"]["1"]["summary"]["successes"], 1)
        self.assertEqual(report["results"]["1"]["requests"][0]["tpot_seconds"], 0.3)
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "serving.json"
            atomic_write_json(path, report)
            self.assertEqual(json.loads(path.read_text())["benchmark"], "serving")

    def test_accuracy_report_counts_every_completed_result(self) -> None:
        args = Namespace(
            dataset="mmlu",
            num_samples=2,
            max_tokens=32,
            concurrency=2,
            base_url="http://127.0.0.1:8000",
            model="local-model",
            subjects=None,
        )
        results = [
            EvalResult("q", 0, 0, True, "A", 1.0),
            EvalResult("q2", 1, None, False, "B", 2.0),
        ]
        report = accuracy_report(args, results)
        self.assertEqual(report["summary"]["successes"], 2)
        self.assertEqual(report["summary"]["accuracy"], 0.5)
        self.assertEqual(report["summary"]["average_latency_seconds"], 1.5)
        self.assertEqual(report["parameters"]["model"], "local-model")


class AccuracyTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_failure_is_not_converted_into_a_wrong_answer(self) -> None:
        class FailingResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            def raise_for_status(self) -> None:
                raise RuntimeError("HTTP 503")

        class FakeSession:
            def post(self, *_args, **_kwargs):
                return FailingResponse()

        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            await send_eval_request(
                FakeSession(), "http://127.0.0.1:8000", "question", 4
            )


if __name__ == "__main__":
    unittest.main()
