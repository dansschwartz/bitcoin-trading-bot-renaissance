"""
Bot Orchestrator
================
Multi-bot coordination layer.  Runs as a standalone process that monitors
heartbeat files from individual bot instances, computes aggregate risk,
and generates combined reports.

Usage
-----
::
    python -m orchestrator
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BotInstance:
    """State snapshot of a single bot instance, populated from heartbeat."""

    bot_id: str
    status: str = "unknown"
    exchange_account: str = "primary"
    assigned_pairs: list = field(default_factory=list)
    capital_usd: float = 0.0
    current_equity_usd: float = 0.0
    net_exposure: float = 0.0
    daily_pnl_usd: float = 0.0
    sharpe_7d: float = 0.0
    last_heartbeat: float = 0.0
    positions: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class BotOrchestrator:
    """Monitors bot heartbeats and computes aggregate risk across instances.

    Parameters
    ----------
    config : dict
        Configuration dict (typically ``config["orchestrator"]``).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._heartbeat_dir = cfg.get("heartbeat_dir", "data/heartbeats")
        self._poll_interval = float(cfg.get("poll_interval_seconds", 5))

        limits = cfg.get("aggregate_limits", {})
        self._max_single_asset_pct = float(limits.get("max_single_asset_pct", 30))
        self._max_total_exposure_pct = float(limits.get("max_total_exposure_pct", 80))
        self._max_drawdown_pct = float(limits.get("max_drawdown_pct", 10))

        self.bots: Dict[str, BotInstance] = {}
        self._running = False

        logger.info(
            "BotOrchestrator init  heartbeat_dir=%s  max_single_asset=%.0f%%  "
            "max_total_exposure=%.0f%%",
            self._heartbeat_dir,
            self._max_single_asset_pct,
            self._max_total_exposure_pct,
        )

    # ----- heartbeat reading ------------------------------------------------

    def _read_heartbeats(self) -> None:
        """Read all JSON heartbeat files from the heartbeat directory."""
        hb_dir = Path(self._heartbeat_dir)
        if not hb_dir.is_dir():
            return

        for fpath in hb_dir.glob("*.json"):
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)

                bot_id = data.get("bot_id", fpath.stem)
                bot = self.bots.get(bot_id)
                if bot is None:
                    bot = BotInstance(bot_id=bot_id)
                    self.bots[bot_id] = bot

                bot.last_heartbeat = float(data.get("timestamp", 0))
                bot.current_equity_usd = float(data.get("equity_usd", 0))
                bot.daily_pnl_usd = float(data.get("daily_pnl_usd", 0))
                bot.net_exposure = float(data.get("net_exposure_usd", 0))
                bot.positions = data.get("positions", {})
                bot.status = data.get("status", "running")

            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read heartbeat %s: %s", fpath, exc)

    # ----- health monitoring ------------------------------------------------

    def monitor_health(self) -> None:
        """Check heartbeat freshness and update bot status.

        - >30s without heartbeat -> unresponsive
        - >120s without heartbeat -> dead
        """
        now = time.time()
        for bot in self.bots.values():
            age = now - bot.last_heartbeat
            if age > 120:
                bot.status = "dead"
            elif age > 30:
                bot.status = "unresponsive"
            elif bot.status in ("unresponsive", "dead", "unknown"):
                bot.status = "running"

        dead = [b.bot_id for b in self.bots.values() if b.status == "dead"]
        unresponsive = [b.bot_id for b in self.bots.values() if b.status == "unresponsive"]
        if dead:
            logger.error("DEAD bots (>120s no heartbeat): %s", dead)
        if unresponsive:
            logger.warning("Unresponsive bots (>30s no heartbeat): %s", unresponsive)

    # ----- aggregate risk ---------------------------------------------------

    def check_aggregate_exposure(self) -> Dict[str, Any]:
        """Sum exposure per asset across all bots and check limits.

        Returns
        -------
        dict
            ``breaches``: list of breach descriptions,
            ``per_asset``: dict of asset -> total exposure USD,
            ``total_exposure_usd``, ``total_equity_usd``.
        """
        per_asset: Dict[str, float] = {}
        total_exposure = 0.0
        total_equity = 0.0

        for bot in self.bots.values():
            if bot.status == "dead":
                continue
            total_equity += bot.current_equity_usd
            total_exposure += abs(bot.net_exposure)

            for pair, pos in bot.positions.items():
                exposure = abs(
                    float(pos.get("net_exposure_usd", 0))
                    if isinstance(pos, dict) else 0
                )
                per_asset[pair] = per_asset.get(pair, 0.0) + exposure

        breaches: List[str] = []

        if total_equity > 0:
            for asset, exposure in per_asset.items():
                pct = (exposure / total_equity) * 100
                if pct > self._max_single_asset_pct:
                    breaches.append(
                        f"{asset}: {pct:.1f}% (limit {self._max_single_asset_pct:.0f}%)"
                    )

            total_pct = (total_exposure / total_equity) * 100
            if total_pct > self._max_total_exposure_pct:
                breaches.append(
                    f"Total exposure: {total_pct:.1f}% "
                    f"(limit {self._max_total_exposure_pct:.0f}%)"
                )

        if breaches:
            logger.warning("Aggregate exposure breaches: %s", breaches)

        return {
            "breaches": breaches,
            "per_asset": per_asset,
            "total_exposure_usd": round(total_exposure, 2),
            "total_equity_usd": round(total_equity, 2),
        }

    # ----- pair assignment --------------------------------------------------

    def assign_pairs(self, available_pairs: List[str]) -> Dict[str, List[str]]:
        """Distribute pairs across bots by splitting evenly (round-robin)."""
        active_bots = [b for b in self.bots.values() if b.status != "dead"]
        if not active_bots:
            return {}

        assignment: Dict[str, List[str]] = {b.bot_id: [] for b in active_bots}
        bot_ids = list(assignment.keys())

        for i, pair in enumerate(available_pairs):
            target = bot_ids[i % len(bot_ids)]
            assignment[target].append(pair)

        return assignment

    # ----- aggregate report -------------------------------------------------

    def get_aggregate_report(self) -> Dict[str, Any]:
        """Combined P&L, exposure, and bot status summary."""
        total_equity = sum(
            b.current_equity_usd for b in self.bots.values() if b.status != "dead"
        )
        total_pnl = sum(
            b.daily_pnl_usd for b in self.bots.values() if b.status != "dead"
        )
        exposure_info = self.check_aggregate_exposure()

        bot_summaries = []
        for bot in self.bots.values():
            bot_summaries.append({
                "bot_id": bot.bot_id,
                "status": bot.status,
                "equity_usd": bot.current_equity_usd,
                "daily_pnl_usd": bot.daily_pnl_usd,
                "net_exposure": bot.net_exposure,
                "positions": len(bot.positions),
            })

        return {
            "timestamp": time.time(),
            "total_equity_usd": round(total_equity, 2),
            "total_daily_pnl_usd": round(total_pnl, 2),
            "total_exposure_usd": exposure_info["total_exposure_usd"],
            "breaches": exposure_info["breaches"],
            "bots": bot_summaries,
            "active_bots": sum(1 for b in self.bots.values() if b.status == "running"),
            "total_bots": len(self.bots),
        }

    # ----- main loop --------------------------------------------------------

    async def run(self) -> None:
        """Main async loop: read heartbeats, check exposure, monitor health."""
        self._running = True
        logger.info("BotOrchestrator run loop started")

        while self._running:
            try:
                self._read_heartbeats()
                self.check_aggregate_exposure()
                self.monitor_health()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Orchestrator cycle failed")

            await asyncio.sleep(self._poll_interval)

        self._running = False
        logger.info("BotOrchestrator run loop stopped")

    def stop(self) -> None:
        """Signal the run loop to stop."""
        self._running = False


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    )
    orch = BotOrchestrator()
    try:
        asyncio.run(orch.run())
    except KeyboardInterrupt:
        orch.stop()
        logger.info("Orchestrator stopped by user")
