"""Small, deterministic response previews for benchmark diagnostics.

These helpers never derive performance counters from decoded text. They only
print a bounded number of post-run examples so malformed, empty, or unrelated
model outputs are visible beside the aggregate metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticSample:
    """One bounded prompt/response preview printed after benchmark timing."""

    label: str
    prompt: str
    output: str
    # gold= parsed=,
    # latency=,
    details: tuple[str, ...] = ()


def clip_text(text: str, limit: int) -> str:
    """Normalize whitespace and bound terminal output without hiding emptiness."""
    #continual space,enter,tab, etc. -> single space
    normalized = " ".join(text.split())
    if not normalized:
        return "<empty>"
    if len(normalized) <= limit:
        return normalized
    #actual clip
    return normalized[: max(0, limit - 3)] + "..."


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    """Choose up to ``count`` stable positions, including both ends."""
    if length <= 0 or count <= 0:
        return []
    count = min(length, count)
    if count == 1:
        return [0]
    return sorted({round(i * (length - 1) / (count - 1)) for i in range(count)})


def print_diagnostic_samples(
    samples: Sequence[DiagnosticSample],
    *,
    #pass by keyward to avoid accidental positional use
    title: str = "Diagnostic output samples",
) -> None:
    """Print already-selected samples outside the measured request interval."""
    if not samples:
        return
    print(f"\n  {title} ({len(samples)}):")
    for index, sample in enumerate(samples, start=1):
        print(f"\n  [{index}] {sample.label}")
        for detail in sample.details:
            print(f"      {detail}")
        print(f"      Prompt: {clip_text(sample.prompt, 240)}")
        print(f"      Output: {clip_text(sample.output, 800)}")
