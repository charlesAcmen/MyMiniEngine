"""Descriptive statistics for repeated benchmark trials."""

from __future__ import annotations

import statistics
from collections.abc import Iterable


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    """Return trial-level descriptive statistics without hiding sample count."""
    samples = [float(value) for value in values]
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stddev": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "stddev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }
