"""Naming and durable storage for benchmark result artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


RESULT_PATH_PREFIX = "JSON result: "
MAX_RESULT_FILENAME_BYTES = 255


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON beside its final destination, then atomically replace it."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _result_stem(benchmark: str, parameters: dict[str, Any]) -> str:
    """Build a compact filename from the parameters that define one run."""
    if benchmark == "serving":
        output = parameters.get("output_len")
        output_brief = (
            f"out{output}" if output is not None else f"max{parameters['max_tokens']}"
        )
        concurrencies = "-".join(
            str(value) for value in parameters["concurrencies"]
        )
        return (
            f"serving-in{parameters['input_len']}-{output_brief}"
            f"-c{concurrencies}-n{parameters['num_requests']}"
        )
    if benchmark == "accuracy":
        subjects = parameters.get("subjects")
        subject_brief = ""
        if subjects:
            ordered_subjects = [item.strip() for item in subjects.split(",")]
            subject_brief = "-" + "-".join(ordered_subjects)
        return (
            f"accuracy-{parameters['dataset']}{subject_brief}"
            f"-n{parameters['num_samples']}"
            f"-c{parameters['concurrency']}"
        )
    raise ValueError(f"unsupported benchmark result type: {benchmark}")


def validate_result_filename(benchmark: str, parameters: dict[str, Any]) -> str:
    """Return the result stem after enforcing the filesystem byte limit."""
    stem = _result_stem(benchmark, parameters)
    filename = f"{stem}.json"
    size = len(filename.encode("utf-8"))
    if size > MAX_RESULT_FILENAME_BYTES:
        raise ValueError(
            f"benchmark result filename is {size} bytes; "
            f"the limit is {MAX_RESULT_FILENAME_BYTES}. Shorten the parameter list."
        )
    return stem


def _validate_candidate(path: Path) -> None:
    size = len(path.name.encode("utf-8"))
    if size > MAX_RESULT_FILENAME_BYTES:
        raise ValueError(
            f"benchmark result filename is {size} bytes after adding the repeat suffix; "
            f"the limit is {MAX_RESULT_FILENAME_BYTES}"
        )


def write_benchmark_result(
    payload: dict[str, Any], root: Path = Path("outputs")
) -> Path:
    """Name and atomically store one benchmark report under its local date."""
    benchmark = payload["benchmark"]
    date_dir = root / datetime.now().astimezone().date().isoformat()
    stem = validate_result_filename(benchmark, payload["parameters"])
    path = date_dir / f"{stem}.json"
    repeat = 2
    while path.exists():
        path = date_dir / f"{stem}-r{repeat:02d}.json"
        repeat += 1
    _validate_candidate(path)
    atomic_write_json(path, payload)
    return path


def announce_result(path: Path) -> None:
    """Expose the saved artifact path to people and batch callers."""
    print(f"  {RESULT_PATH_PREFIX}{path.resolve()}")
