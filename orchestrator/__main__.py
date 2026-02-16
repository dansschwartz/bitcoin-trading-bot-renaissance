"""
Standalone entry point for the BotOrchestrator.

Usage::
    python -m orchestrator
    python -m orchestrator --config config/config.json
"""

import argparse
import asyncio
import json
import logging
import sys

from orchestrator.bot_manager import BotOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Renaissance Bot Orchestrator")
    parser.add_argument(
        "--config", default="config/config.json",
        help="Path to config file (reads orchestrator section)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = {}
    try:
        with open(args.config) as f:
            full_config = json.load(f)
        config = full_config.get("orchestrator", {})
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logging.warning("Could not load config %s: %s — using defaults", args.config, exc)

    orchestrator = BotOrchestrator(config)

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        orchestrator.stop()
        logging.info("Orchestrator shut down by user")


if __name__ == "__main__":
    main()
