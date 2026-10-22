"""Argparse validators shared by MiniEngine command-line entry points."""

from __future__ import annotations

import argparse


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be an integer of at least 1")
    return value


def non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def port_number(raw: str) -> int:
    value = int(raw)
    if not 1 <= value <= 65_535:
        raise argparse.ArgumentTypeError("must be an integer in [1, 65535]")
    return value


def unit_interval(raw: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be a number in [0.0, 1.0]")
    return value


def open_unit_interval(raw: str) -> float:
    value = float(raw)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("must be a number strictly between 0.0 and 1.0")
    return value


def positive_int_csv(raw: str) -> list[int]:
    try:
        values = [positive_int(item.strip()) for item in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be comma-separated positive integers"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one integer")
    return values
