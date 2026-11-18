"""Command-line parsing for the MiniEngine server and benchmark tools."""

from __future__ import annotations

import argparse

from miniengine.config import (
    CompileConfig,
    CudaGraphConfig,
    EngineConfig,
    SamplingConfig,
)


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
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("must not contain duplicate integers")
    return values


def increasing_positive_int_csv(raw: str) -> tuple[int, ...]:
    values = positive_int_csv(raw)
    if values != sorted(values):
        raise argparse.ArgumentTypeError("must be strictly increasing")
    return tuple(values)


def build_server_parser() -> argparse.ArgumentParser:
    """Build the server CLI without starting or importing the model engine."""
    parser = argparse.ArgumentParser(
        prog="miniengine",
        description="Minimal LLM serving engine",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model id or local path"
    )
    parser.add_argument(
        "--mode",
        choices=["batched", "paged"],
        default="batched",
        help="Serving engine path: preserved M1 baseline or M2 paged KV",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=port_number, default=8000)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to load model on"
    )
    parser.add_argument(
        "--max-running",
        type=positive_int,
        default=16,
        help="Max concurrent requests in the scheduler",
    )
    parser.add_argument(
        "--page-size",
        type=positive_int,
        default=256,
        help="KV-cache tokens per physical page (default: 256)",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=open_unit_interval,
        default=0.80,
        help="Fraction of GPU memory reserved for model weights plus the fixed KV pool",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=non_negative_int,
        default=0,
        metavar="N",
        help=(
            "Maximum prefill query tokens scheduled per step; "
            "0 preserves the M2 single-shot prefill path"
        ),
    )
    parser.add_argument(
        "--sampling-config",
        type=SamplingConfig.from_json,
        default=SamplingConfig(),
        metavar="PATH",
        help="JSON file containing the server-wide token sampling policy",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile stable paged-decode tensor regions with TorchInductor",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Replay paged decode through fixed-size manual CUDA graphs",
    )
    parser.add_argument(
        "--cuda-graph-batch-sizes",
        type=increasing_positive_int_csv,
        default=(1, 2, 4, 8, 16, 32),
        metavar="SIZES",
        help="Strictly increasing CUDA graph batch buckets",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse server arguments and construct the immutable engine config."""
    parser = build_server_parser()
    args = parser.parse_args(argv)
    try:
        args.engine_config = EngineConfig(
            mode=args.mode,
            dtype=args.dtype,
            device=args.device,
            page_size=args.page_size,
            mem_fraction_static=args.mem_fraction_static,
            prefill_chunk_size=args.prefill_chunk_size,
            compile=CompileConfig(enabled=args.torch_compile),
            cuda_graph=CudaGraphConfig(
                enabled=args.cuda_graph,
                batch_sizes=args.cuda_graph_batch_sizes,
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.cuda_graph and max(args.cuda_graph_batch_sizes) < args.max_running:
        parser.error("largest CUDA graph batch size must cover --max-running")
    return args
