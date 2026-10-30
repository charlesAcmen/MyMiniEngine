"""CPU-only validation for the Part-B experiment runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.result_artifacts import atomic_write_json
from benchmark.run_experiments import (
    ConfigError,
    TelemetryMonitor,
    build_summary,
    load_config,
    required_pages,
    result_path_from_log,
    validate_capacity,
    validate_resume_environment,
)


def minimal_config() -> dict:
    return {
        "schema_version": 1,
        "name": "test-run",
        "model": "models/Qwen3-0.6B",
        "common": {
            "dtype": "bfloat16",
            "page_size": 256,
            "mem_fraction_static": 0.72,
            "seed": 42,
            "warmup": {
                "input_len": 32,
                "output_len": 2,
                "num_requests": 2,
                "concurrency": 1,
            },
            "safety": {
                "max_fb_mib": 11520,
                "max_temperature_c": 83,
                "max_idle_fb_mib": 2048,
                "idle_wait_seconds": 10,
                "trial_timeout_seconds": 30,
            },
        },
        "conditions": [
            {
                "id": "balanced",
                "group": "acceptance",
                "modes": ["batched", "paged"],
                "input_len": 512,
                "output_len": 256,
                "input_len_min_ratio": 1.0,
                "concurrency": 32,
                "max_running": 32,
                "num_requests": 64,
                "repeats": 2,
            }
        ],
    }


class ConfigValidationTests(unittest.TestCase):
    def _load(self, payload: dict) -> dict:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_config(path)

    def test_valid_config_and_page_preflight(self) -> None:
        config = self._load(minimal_config())
        condition = config["conditions"][0]
        self.assertEqual(required_pages(condition, 256), 96)
        validate_capacity(config, {"usable_pages": 96})
        with self.assertRaises(ConfigError):
            validate_capacity(config, {"usable_pages": 95})

    def test_output_modes_are_mutually_exclusive(self) -> None:
        config = minimal_config()
        config["conditions"][0]["max_tokens"] = 32
        with self.assertRaises(ConfigError):
            self._load(config)

    def test_client_overload_requires_explicit_group(self) -> None:
        config = minimal_config()
        config["conditions"][0]["concurrency"] = 64
        with self.assertRaises(ConfigError):
            self._load(config)
        config["conditions"][0]["group"] = "overload"
        self._load(config)

    def test_duplicate_condition_ids_are_rejected(self) -> None:
        config = minimal_config()
        config["conditions"].append(dict(config["conditions"][0]))
        with self.assertRaises(ConfigError):
            self._load(config)

    def test_resume_rejects_source_drift(self) -> None:
        previous = {
            "git_sha": "a",
            "git_diff_sha256": "one",
            "config_sha256": "config",
            "python": "3.14",
            "torch": "2.14",
            "torch_cuda": "13.0",
            "flash_attn": "2.8.4",
        }
        validate_resume_environment(previous, dict(previous))
        current = dict(previous)
        current["git_diff_sha256"] = "two"
        with self.assertRaises(RuntimeError):
            validate_resume_environment(previous, current)


class CliResultPathTests(unittest.TestCase):
    def test_result_path_parser_handles_spaces_and_missing_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            log_path = Path(temporary_dir) / "client.log"
            log_path.write_text(
                "benchmark output\n"
                "  JSON result: /tmp/result directory/serving.json\n",
                encoding="utf-8",
            )
            self.assertEqual(
                result_path_from_log(log_path),
                Path("/tmp/result directory/serving.json"),
            )
            log_path.write_text("benchmark output only\n", encoding="utf-8")
            self.assertIsNone(result_path_from_log(log_path))


class TelemetryTests(unittest.TestCase):
    def test_dmon_line_parser_handles_missing_fields(self) -> None:
        sample = TelemetryMonitor.parse_line(
            "0 150 72 - 96 40 0 0 0 0 7000 1800 11200 4 -"
        )
        assert sample is not None
        self.assertEqual(sample["power_w"], 150.0)
        self.assertEqual(sample["sm_util_pct"], 96.0)
        self.assertEqual(sample["fb_memory_mib"], 11200.0)
        self.assertIsNone(sample["memory_temp_c"])
        self.assertIsNone(TelemetryMonitor.parse_line("# header"))

    @staticmethod
    def _sample(temperature: float) -> dict[str, float | None]:
        sample = {key: 0.0 for key in TelemetryMonitor.FIELDS}
        sample["gpu_temp_c"] = temperature
        sample["fb_memory_mib"] = 1000.0
        return sample

    def test_temperature_policy_warns_then_requires_two_hot_samples(self) -> None:
        monitor = TelemetryMonitor(
            11520,
            88,
            warning_temperature_c=83,
            emergency_temperature_c=90,
            temperature_consecutive_samples=2,
        )
        monitor.record_sample(self._sample(86))
        self.assertIsNone(monitor.violation)
        self.assertEqual(monitor.temperature_warning_count, 1)
        monitor.record_sample(self._sample(88))
        self.assertIsNone(monitor.violation)
        monitor.record_sample(self._sample(87))
        monitor.record_sample(self._sample(88))
        monitor.record_sample(self._sample(89))
        self.assertIn("for 2 samples", monitor.violation or "")

    def test_temperature_policy_stops_immediately_at_emergency_limit(self) -> None:
        monitor = TelemetryMonitor(
            11520,
            88,
            warning_temperature_c=83,
            emergency_temperature_c=90,
            temperature_consecutive_samples=2,
        )
        monitor.record_sample(self._sample(90))
        self.assertIn("emergency", monitor.violation or "")


class SummaryTests(unittest.TestCase):
    @staticmethod
    def _trial(mode: str, repeat: int, throughput: float) -> dict:
        summary = {
            "generation_tokens_per_second": throughput,
            "ttft_ms": {"p50": 10.0, "p99": 20.0},
            "completion_ms": {"p50": 100.0, "p99": 120.0},
            "tpot_ms": {"p50": 2.0, "p99": 3.0},
        }
        return {
            "status": "success",
            "failure_class": None,
            "condition": {"id": "balanced", "concurrency": 32},
            "mode": mode,
            "repeat": repeat,
            "benchmark": {"benchmark": "serving", "results": {"32": {"summary": summary}}},
            "telemetry": {
                "peak_fb_memory_mib": 11000.0,
                "average_sm_util_pct": 90.0,
                "average_power_w": 150.0,
                "max_temperature_c": 75.0,
            },
        }

    def test_summary_uses_trial_statistics_and_paired_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for mode, throughput in (("batched", 100.0), ("paged", 200.0)):
                path = root / mode / "balanced" / "r01.json"
                atomic_write_json(path, self._trial(mode, 1, throughput))
            summary = build_summary(root, {"git_sha": "abc"}, {"usable_pages": 1})
        condition = summary["conditions"]["balanced"]
        self.assertEqual(condition["batched"]["successful"], 1)
        self.assertEqual(condition["paired_generation_speedup"]["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
