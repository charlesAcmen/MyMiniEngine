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
