"""
CLI entry point — launch the MiniEngine server.

Usage:
    python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507
    python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 --port 8080 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from miniengine.cli import (
    increasing_positive_int_csv,
    open_unit_interval,
    port_number,
    positive_int,
)
from miniengine.config import (
    CompileConfig,
    CudaGraphConfig,
    EngineConfig,
    SamplingConfig,
)
from miniengine.engine import Engine
from miniengine.scheduler import Scheduler
from miniengine import server as srv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="miniengine",
        description="Minimal LLM serving engine",
    )
    p.add_argument(
        "--model", type=str, required=True, help="HuggingFace model id or local path"
    )
    p.add_argument(
        "--mode",
        choices=["batched", "paged"],
        default="batched",
        help="Serving engine path: preserved M1 baseline or M2 paged KV",
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=port_number, default=8000)
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument(
        "--device", type=str, default="cuda", help="Device to load model on"
    )
    p.add_argument(
        "--max-running",
        type=positive_int,
        default=16,
        help="Max concurrent requests in the scheduler",
    )
    p.add_argument(
        "--page-size",
        type=positive_int,
        default=256,
        help="KV-cache tokens per physical page (default: 256)",
    )
    p.add_argument(
        "--mem-fraction-static",
        type=open_unit_interval,
        default=0.80,
        help="Fraction of GPU memory reserved for model weights plus the fixed KV pool",
    )
    p.add_argument(
        "--sampling-config",
        type=SamplingConfig.from_json,
        default=SamplingConfig(),
        metavar="PATH",
        help="JSON file containing the server-wide token sampling policy",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile stable paged-decode tensor regions with TorchInductor",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Replay paged decode through fixed-size manual CUDA graphs",
    )
    p.add_argument(
        "--cuda-graph-batch-sizes",
        type=increasing_positive_int_csv,
        default=(1, 2, 4, 8, 16, 32),
        metavar="SIZES",
        help="Strictly increasing CUDA graph batch buckets",
    )
    args = p.parse_args(argv)
    try:
        args.engine_config = EngineConfig(
            mode=args.mode,
            dtype=args.dtype,
            device=args.device,
            page_size=args.page_size,
            mem_fraction_static=args.mem_fraction_static,
            compile=CompileConfig(enabled=args.torch_compile),
            cuda_graph=CudaGraphConfig(
                enabled=args.cuda_graph,
                batch_sizes=args.cuda_graph_batch_sizes,
            ),
        )
    except ValueError as exc:
        p.error(str(exc))
    if args.cuda_graph and max(args.cuda_graph_batch_sizes) < args.max_running:
        p.error("largest CUDA graph batch size must cover --max-running")
    return args


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("miniengine")

    logger.info("Initializing engine  model=%s  dtype=%s", args.model, args.dtype)

    engine = Engine(
        model_path=args.model,
        engine_config=args.engine_config,
        sampling_config=args.sampling_config,
    )
    sched = Scheduler(engine=engine, max_running=args.max_running)

    # Wire up the server module globals
    srv.engine = engine
    srv.scheduler = sched
    srv.model_id = args.model

    # Start scheduler background thread
    sched.start()

    logger.info("Starting server on %s:%d", args.host, args.port)
    try:
        uvicorn.run(srv.app, host=args.host, port=args.port, log_level="info")
    finally:
        sched.stop()


if __name__ == "__main__":
    main()
