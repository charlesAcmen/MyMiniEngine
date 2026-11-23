"""
CLI entry point — launch the MiniEngine server.

Usage:
    python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507
    python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 --port 8080 --dtype bfloat16
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

import uvicorn

from miniengine import server as srv
from miniengine.cli import parse_args
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

    server_config = {
        "model": args.model,
        "host": args.host,
        "port": args.port,
    }
    logger.info("Server config: %s", json.dumps(server_config, sort_keys=True))
    logger.info(
        "Engine config: %s",
        json.dumps(asdict(args.engine_config), sort_keys=True),
    )
    logger.info(
        "Sampling config: %s",
        json.dumps(asdict(args.sampling_config), sort_keys=True),
    )

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
        uvicorn.run(
            srv.app,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=False,
        )
    finally:
        sched.stop()


if __name__ == "__main__":
    main()
