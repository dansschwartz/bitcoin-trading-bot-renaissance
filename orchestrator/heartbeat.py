"""
Heartbeat Writer
================
Each bot instance writes a JSON heartbeat file every N seconds so the
BotOrchestrator (running as a separate process) can monitor health
and compute aggregate risk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Any

logger = logging.getLogger(__name__)


class HeartbeatWriter:
    """Writes periodic JSON heartbeat files for multi-bot coordination.

    Parameters
    ----------
    bot_id : str
        Unique identifier for this bot instance (e.g. "bot-01").
    heartbeat_dir : str
        Directory where heartbeat files are written.
    """

    def __init__(self, bot_id: str, heartbeat_dir: str = "data/heartbeats") -> None:
        self._bot_id = bot_id
        self._heartbeat_dir = heartbeat_dir
        self._running = False

        os.makedirs(self._heartbeat_dir, exist_ok=True)
        logger.info("HeartbeatWriter init  bot_id=%s  dir=%s", bot_id, heartbeat_dir)

    async def start(self, bot_ref: Any, interval: float = 5.0) -> None:
        """Run heartbeat loop: extract state from bot_ref and write every *interval* seconds."""
        self._running = True
        logger.info(
            "HeartbeatWriter started  bot_id=%s  interval=%.1fs",
            self._bot_id, interval,
        )

        while self._running:
            try:
                positions = getattr(bot_ref, "positions", {}) or {}
                daily_pnl = getattr(bot_ref, "daily_pnl", 0.0) or 0.0
                equity = getattr(bot_ref, "portfolio_value", 0.0) or 0.0

                net_exposure = sum(
                    abs(float(p.get("net_exposure_usd", 0)))
                    if isinstance(p, dict) else 0
                    for p in positions.values()
                ) if isinstance(positions, dict) else 0.0

                heartbeat = {
                    "bot_id": self._bot_id,
                    "timestamp": time.time(),
                    "status": "running",
                    "equity_usd": round(float(equity), 2),
                    "daily_pnl_usd": round(float(daily_pnl), 2),
                    "positions": positions if isinstance(positions, dict) else {},
                    "net_exposure_usd": round(net_exposure, 2),
                    "open_orders": 0,
                    "regime": "unknown",
                }

                # Atomic write: tmp file then os.replace
                target = os.path.join(self._heartbeat_dir, f"{self._bot_id}.json")
                fd, tmp_path = tempfile.mkstemp(dir=self._heartbeat_dir, suffix=".tmp")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(heartbeat, f, indent=2)
                    os.replace(tmp_path, target)
                except Exception:
                    logger.exception("Failed to write heartbeat for %s", self._bot_id)
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("HeartbeatWriter cycle failed")

            await asyncio.sleep(interval)

        self._running = False
        logger.info("HeartbeatWriter stopped  bot_id=%s", self._bot_id)

    def stop(self) -> None:
        """Signal the heartbeat loop to stop."""
        self._running = False
