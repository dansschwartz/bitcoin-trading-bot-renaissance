"""
Signal Auto-Throttle -- Daily P&L-Based Signal Health Monitor (E1)
===================================================================
Monitors rolling daily P&L per signal type and automatically throttles
or disables signals that have been consistently losing money.

Unlike the root-level ``signal_auto_throttle.py`` (which uses cycle-based
directional accuracy), this module tracks *realised daily P&L* aggregated
from the ``trades`` table and maintains its own ``signal_daily_pnl``
ledger.  Decisions are made on calendar-day boundaries.

Design principle (inherited from DevilTracker):
  "if unexpected, do nothing" -- every public method catches exceptions
  internally and logs rather than raising, so a tracker failure never
  kills a live trading loop.

Throttle rules (configurable):
  - 3+ consecutive losing days  -> throttled (allocation * 0.5)
  - 5+ consecutive losing days  -> disabled  (allocation * 0.0)
  - 3  consecutive profitable days after disable -> re-enabled

This module is self-contained: it owns its own SQLite table and does NOT
depend on the broader DatabaseManager.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SignalThrottle:
    """
    Tracks daily P&L per signal type and throttles/disables losing signals.

    Usage::

        throttle = SignalThrottle(config, "data/renaissance_bot.db")
        # At end of each trading day:
        throttle.update_daily()
        # Before placing a trade:
        if throttle.is_signal_allowed("stat_arb"):
            size *= throttle.get_allocation_multiplier("stat_arb")
    """

    TABLE = "signal_daily_pnl"

    _CREATE_TABLE = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_type TEXT NOT NULL,
        date TEXT NOT NULL,
        pnl REAL NOT NULL,
        num_trades INTEGER NOT NULL,
        win_rate REAL,
        UNIQUE(signal_type, date)
    )
    """

    _CREATE_IDX_SIGNAL_DATE = (
        f"CREATE INDEX IF NOT EXISTS idx_signal_daily_pnl_type_date "
        f"ON {TABLE} (signal_type, date)"
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, config: Dict[str, Any], db_path: str):
        self.db_path = db_path
        cfg = config.get("medallion_signal_throttle", {})

        self._throttle_after = int(cfg.get("throttle_after_consecutive_losing_days", 3))
        self._throttle_reduction = float(cfg.get("throttle_reduction", 0.5))
        self._disable_after = int(cfg.get("disable_after_consecutive_losing_days", 5))
        self._re_enable_after = int(cfg.get("re_enable_after_profitable_days", 3))
        self._evaluation_time_utc = cfg.get("evaluation_time_utc", "00:00")

        self._ensure_table()
        logger.info(
            "SignalThrottle initialised (db=%s) throttle_after=%d days, "
            "disable_after=%d days, re_enable_after=%d profitable days",
            db_path, self._throttle_after, self._disable_after, self._re_enable_after,
        )

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        """Yield a short-lived connection with WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_table(self):
        """Create the signal_daily_pnl table and indexes if they do not exist."""
        try:
            with self._conn() as conn:
                conn.execute(self._CREATE_TABLE)
                conn.execute(self._CREATE_IDX_SIGNAL_DATE)
                conn.commit()
        except Exception as exc:
            logger.error("SignalThrottle: failed to create table: %s", exc)

    # ------------------------------------------------------------------
    # Core: daily P&L retrieval
    # ------------------------------------------------------------------

    def _get_daily_pnl_by_signal(
        self, signal_type: str, lookback_days: int = 7
    ) -> List[Tuple[str, float]]:
        """
        Return a list of ``(date_str, pnl)`` tuples for the given signal type
        over the most recent *lookback_days*, ordered oldest-first.
        """
        try:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=lookback_days)
            ).strftime("%Y-%m-%d")

            with self._conn() as conn:
                rows = conn.execute(
                    f"""
                    SELECT date, pnl
                    FROM {self.TABLE}
                    WHERE signal_type = ? AND date >= ?
                    ORDER BY date ASC
                    """,
                    (signal_type, cutoff),
                ).fetchall()
            return [(row[0], row[1]) for row in rows]
        except Exception as exc:
            logger.error(
                "SignalThrottle._get_daily_pnl_by_signal failed for %s: %s",
                signal_type, exc,
            )
            return []

    # ------------------------------------------------------------------
    # Core: consecutive losing day counter
    # ------------------------------------------------------------------

    @staticmethod
    def _count_consecutive_losing_days(daily_pnls: List[Tuple[str, float]]) -> int:
        """
        Count the number of consecutive losing days (pnl < 0) at the END
        of the provided daily P&L list.

        ``daily_pnls`` is expected to be sorted oldest-first: [(date, pnl), ...].
        """
        if not daily_pnls:
            return 0

        count = 0
        for _date, pnl in reversed(daily_pnls):
            if pnl < 0:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _count_consecutive_profitable_days(daily_pnls: List[Tuple[str, float]]) -> int:
        """
        Count the number of consecutive profitable days (pnl >= 0) at the END
        of the provided daily P&L list.
        """
        if not daily_pnls:
            return 0

        count = 0
        for _date, pnl in reversed(daily_pnls):
            if pnl >= 0:
                count += 1
            else:
                break
        return count

    # ------------------------------------------------------------------
    # Health check per signal
    # ------------------------------------------------------------------

    def check_signal_health(self, signal_type: str) -> Dict[str, Any]:
        """
        Assess the health of a single signal type based on its daily P&L
        history.

        Returns a dict with keys:
            signal_type, consecutive_losing_days, status,
            allocation_multiplier, action_taken, daily_pnl_history
        """
        try:
            daily_pnls = self._get_daily_pnl_by_signal(signal_type, lookback_days=7)
            consecutive_losing = self._count_consecutive_losing_days(daily_pnls)
            consecutive_profitable = self._count_consecutive_profitable_days(daily_pnls)

            # Determine status
            if consecutive_losing >= self._disable_after:
                status = "disabled"
                allocation_multiplier = 0.0
                action_taken = (
                    f"Signal disabled: {consecutive_losing} consecutive losing days "
                    f"(threshold: {self._disable_after})"
                )
            elif consecutive_losing >= self._throttle_after:
                status = "throttled"
                allocation_multiplier = self._throttle_reduction
                action_taken = (
                    f"Signal throttled to {self._throttle_reduction:.0%}: "
                    f"{consecutive_losing} consecutive losing days "
                    f"(threshold: {self._throttle_after})"
                )
            else:
                status = "active"
                allocation_multiplier = 1.0
                action_taken = None

            # Check for re-enable: if we were previously disabled but have
            # enough consecutive profitable days, upgrade to active
            if status == "disabled" and consecutive_profitable >= self._re_enable_after:
                status = "active"
                allocation_multiplier = 1.0
                action_taken = (
                    f"Signal re-enabled: {consecutive_profitable} consecutive "
                    f"profitable days (threshold: {self._re_enable_after})"
                )

            pnl_history = [pnl for _date, pnl in daily_pnls]

            return {
                "signal_type": signal_type,
                "consecutive_losing_days": consecutive_losing,
                "status": status,
                "allocation_multiplier": allocation_multiplier,
                "action_taken": action_taken,
                "daily_pnl_history": pnl_history,
            }
        except Exception as exc:
            logger.error(
                "SignalThrottle.check_signal_health failed for %s: %s",
                signal_type, exc,
            )
            # Safe fallback: allow signal at full allocation
            return {
                "signal_type": signal_type,
                "consecutive_losing_days": 0,
                "status": "active",
                "allocation_multiplier": 1.0,
                "action_taken": None,
                "daily_pnl_history": [],
            }

    # ------------------------------------------------------------------
    # Batch health check
    # ------------------------------------------------------------------

    def get_all_signal_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Return health dicts for every signal type that has daily P&L records.
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    f"SELECT DISTINCT signal_type FROM {self.TABLE}"
                ).fetchall()

            signal_types = [row[0] for row in rows]
            result = {}
            for sig in signal_types:
                result[sig] = self.check_signal_health(sig)
            return result
        except Exception as exc:
            logger.error("SignalThrottle.get_all_signal_health failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # End-of-day aggregation
    # ------------------------------------------------------------------

    def update_daily(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggregate trades from the ``trades`` table for a given date (default:
        yesterday UTC), group by signal type (``algo_used`` column), compute
        daily P&L and win rate, and insert/update the ``signal_daily_pnl``
        ledger.

        The ``trades`` table schema is::

            trades(id, timestamp, product_id, side, size, price, status,
                   algo_used, slippage, execution_time)

        ``algo_used`` maps to signal_type.  Trades with NULL algo_used are
        skipped.

        Returns a summary dict: {signal_type: {pnl, num_trades, win_rate, action}}.
        """
        if target_date is None:
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            target_date = yesterday.strftime("%Y-%m-%d")

        logger.info("SignalThrottle.update_daily: aggregating trades for %s", target_date)
        summary: Dict[str, Any] = {}

        try:
            with self._conn() as conn:
                # Check if the trades table exists
                table_check = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
                ).fetchone()
                if table_check is None:
                    logger.warning(
                        "SignalThrottle.update_daily: 'trades' table does not exist"
                    )
                    return summary

                # Fetch all trades for the target date grouped by algo_used
                rows = conn.execute(
                    """
                    SELECT
                        algo_used,
                        side,
                        size,
                        price,
                        slippage
                    FROM trades
                    WHERE date(timestamp) = ?
                      AND algo_used IS NOT NULL
                      AND algo_used != ''
                    ORDER BY algo_used, timestamp ASC
                    """,
                    (target_date,),
                ).fetchall()

            if not rows:
                logger.info(
                    "SignalThrottle.update_daily: no trades found for %s", target_date
                )
                return summary

            # Group trades by signal type and compute P&L
            # P&L approximation: SELL trades contribute +size*price, BUY trades
            # contribute -size*price.  Slippage is subtracted.
            from collections import defaultdict

            signal_trades: Dict[str, List[Dict]] = defaultdict(list)
            for algo_used, side, size, price, slippage in rows:
                signal_trades[algo_used].append({
                    "side": side,
                    "size": size or 0.0,
                    "price": price or 0.0,
                    "slippage": slippage or 0.0,
                })

            with self._conn() as conn:
                for signal_type, trades in signal_trades.items():
                    pnl = 0.0
                    winning_trades = 0
                    num_trades = len(trades)

                    for t in trades:
                        trade_value = t["size"] * t["price"]
                        slip_cost = abs(t["slippage"]) if t["slippage"] else 0.0

                        if t["side"] and t["side"].upper() == "SELL":
                            trade_pnl = trade_value - slip_cost
                            if trade_pnl > 0:
                                winning_trades += 1
                        elif t["side"] and t["side"].upper() == "BUY":
                            trade_pnl = -trade_value - slip_cost
                        else:
                            trade_pnl = 0.0

                        pnl += trade_pnl

                    win_rate = (
                        winning_trades / num_trades if num_trades > 0 else 0.0
                    )

                    # Upsert into signal_daily_pnl
                    conn.execute(
                        f"""
                        INSERT INTO {self.TABLE}
                            (signal_type, date, pnl, num_trades, win_rate)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(signal_type, date) DO UPDATE SET
                            pnl = excluded.pnl,
                            num_trades = excluded.num_trades,
                            win_rate = excluded.win_rate
                        """,
                        (signal_type, target_date, pnl, num_trades, win_rate),
                    )

                    # Check health after update
                    health = self.check_signal_health(signal_type)

                    summary[signal_type] = {
                        "pnl": round(pnl, 6),
                        "num_trades": num_trades,
                        "win_rate": round(win_rate, 4),
                        "action": health.get("action_taken"),
                        "status": health.get("status"),
                    }

                    if health.get("action_taken"):
                        logger.warning(
                            "SignalThrottle [%s]: %s",
                            signal_type, health["action_taken"],
                        )
                    else:
                        logger.info(
                            "SignalThrottle [%s]: pnl=%.4f trades=%d status=%s",
                            signal_type, pnl, num_trades, health.get("status"),
                        )

                conn.commit()

            logger.info(
                "SignalThrottle.update_daily complete for %s: %d signal types processed",
                target_date, len(summary),
            )
            return summary

        except Exception as exc:
            logger.error("SignalThrottle.update_daily failed: %s", exc)
            return summary

    # ------------------------------------------------------------------
    # Query helpers for the trading loop
    # ------------------------------------------------------------------

    def is_signal_allowed(self, signal_type: str) -> bool:
        """
        Return ``False`` if the signal is disabled (5+ consecutive losing days),
        ``True`` otherwise (including when no data is available).
        """
        try:
            health = self.check_signal_health(signal_type)
            return health["status"] != "disabled"
        except Exception as exc:
            logger.error(
                "SignalThrottle.is_signal_allowed failed for %s: %s",
                signal_type, exc,
            )
            return True  # fail-open: allow the signal

    def get_allocation_multiplier(self, signal_type: str) -> float:
        """
        Return the allocation multiplier for a signal type:
          - 1.0 if active
          - 0.5 if throttled (configurable)
          - 0.0 if disabled
          - 1.0 if no data (fail-open)
        """
        try:
            health = self.check_signal_health(signal_type)
            return health["allocation_multiplier"]
        except Exception as exc:
            logger.error(
                "SignalThrottle.get_allocation_multiplier failed for %s: %s",
                signal_type, exc,
            )
            return 1.0  # fail-open
