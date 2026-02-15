"""
Leverage Manager -- Consistency-Based Leverage Control (E2)
============================================================
"Leverage is a function of CONSISTENCY, not edge size.  A tiny edge with
low volatility can support huge leverage."

This module reads recent trade returns from the SQLite database, computes
a consistency score (fraction of winning days), rolling Sharpe ratio, and
a Kelly-fraction-based maximum safe leverage.  It then compares the
current portfolio leverage against that ceiling to produce actionable
guidance (ok / warning / overleveraged).

Self-contained: owns no tables but reads from the ``trades`` table
maintained by the broader trading system.  All DB access uses context
managers and WAL mode.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LeverageManager:
    """
    Computes safe leverage limits from trade-history consistency.

    Usage::

        lm = LeverageManager(config, "data/renaissance_bot.db")
        report = lm.get_leverage_report()
        if lm.should_reduce_leverage():
            # de-lever
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, config: Dict[str, Any], db_path: str):
        self.db_path = db_path
        cfg = config.get("leverage_manager", {})

        self._enabled = bool(cfg.get("enabled", True))
        self._max_leverage = float(cfg.get("max_leverage", 3.0))
        self._target_sharpe = float(cfg.get("target_sharpe", 1.5))
        self._kelly_fraction = float(cfg.get("kelly_fraction", 0.25))
        self._consistency_window_days = int(cfg.get("consistency_window_days", 30))
        self._check_interval_minutes = int(cfg.get("check_interval_minutes", 60))
        self._auto_deleverage = bool(cfg.get("auto_deleverage", False))

        # Cached state (updated on each computation)
        self._current_leverage: float = 0.0
        self._positions: List[Dict[str, Any]] = []
        self._equity: float = 0.0

        logger.info(
            "LeverageManager initialised (db=%s) max_leverage=%.1f "
            "target_sharpe=%.2f kelly_fraction=%.2f window=%d days",
            db_path, self._max_leverage, self._target_sharpe,
            self._kelly_fraction, self._consistency_window_days,
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

    # ------------------------------------------------------------------
    # Daily return extraction
    # ------------------------------------------------------------------

    def _get_daily_returns(
        self, window_days: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Compute net daily P&L from the ``trades`` table for the rolling
        window, returning ``[(date_str, daily_pnl), ...]`` sorted oldest-first.

        Daily P&L is approximated as:
          SUM(CASE side='SELL' THEN size*price ELSE -size*price END)
        per calendar date.
        """
        if window_days is None:
            window_days = self._consistency_window_days

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=window_days)
        ).strftime("%Y-%m-%d")

        try:
            with self._conn() as conn:
                # Check if the trades table exists
                table_check = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
                ).fetchone()
                if table_check is None:
                    logger.warning(
                        "LeverageManager._get_daily_returns: 'trades' table does not exist"
                    )
                    return []

                rows = conn.execute(
                    """
                    SELECT
                        date(timestamp) AS trade_date,
                        SUM(
                            CASE
                                WHEN upper(side) = 'SELL' THEN size * price
                                WHEN upper(side) = 'BUY'  THEN -size * price
                                ELSE 0
                            END
                        ) AS daily_pnl
                    FROM trades
                    WHERE date(timestamp) >= ?
                      AND status IS NOT NULL
                    GROUP BY trade_date
                    ORDER BY trade_date ASC
                    """,
                    (cutoff,),
                ).fetchall()
            return [(row[0], row[1]) for row in rows if row[0] is not None]
        except Exception as exc:
            logger.error("LeverageManager._get_daily_returns failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Consistency score
    # ------------------------------------------------------------------

    def compute_consistency_score(self, window_days: int = 30) -> float:
        """
        Consistency score = 1 - (num_losing_days / total_days).

        Returns a value in [0.0, 1.0].  Returns 0.0 if no data is available.
        """
        try:
            daily_returns = self._get_daily_returns(window_days)
            if not daily_returns:
                logger.info(
                    "LeverageManager.compute_consistency_score: no daily returns "
                    "available (window=%d days)", window_days,
                )
                return 0.0

            total_days = len(daily_returns)
            losing_days = sum(1 for _date, pnl in daily_returns if pnl < 0)
            score = 1.0 - (losing_days / total_days)
            return max(0.0, min(1.0, score))
        except Exception as exc:
            logger.error(
                "LeverageManager.compute_consistency_score failed: %s", exc
            )
            return 0.0

    # ------------------------------------------------------------------
    # Rolling Sharpe
    # ------------------------------------------------------------------

    def _compute_rolling_sharpe(
        self, window_days: Optional[int] = None
    ) -> float:
        """
        Annualised Sharpe ratio from daily P&L.

        Sharpe = (mean_daily_pnl / std_daily_pnl) * sqrt(252)

        Returns 0.0 if there is insufficient data (< 5 days) or zero
        standard deviation.
        """
        if window_days is None:
            window_days = self._consistency_window_days

        try:
            daily_returns = self._get_daily_returns(window_days)
            if len(daily_returns) < 5:
                logger.info(
                    "LeverageManager._compute_rolling_sharpe: insufficient data "
                    "(%d days, need >= 5)", len(daily_returns),
                )
                return 0.0

            pnls = [pnl for _date, pnl in daily_returns]
            mean_pnl = sum(pnls) / len(pnls)

            variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
            std_pnl = math.sqrt(variance)

            if std_pnl < 1e-12:
                # Zero volatility -- either all flat or all identical returns.
                # If mean is positive, treat as infinite Sharpe capped to
                # target; if zero or negative, return 0.
                if mean_pnl > 0:
                    return self._target_sharpe * 2.0  # very consistent
                return 0.0

            sharpe = (mean_pnl / std_pnl) * math.sqrt(252)
            return sharpe
        except Exception as exc:
            logger.error("LeverageManager._compute_rolling_sharpe failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Max safe leverage
    # ------------------------------------------------------------------

    def compute_max_safe_leverage(self) -> float:
        """
        Compute the maximum safe leverage multiplier.

        Formula:
            max_leverage = min(
                config.max_leverage,
                target_sharpe * kelly_fraction * consistency_score
            )

        The result is clamped to [0.0, config.max_leverage].  If no trade
        history is available, returns 0.0 (conservatively disallow leverage).
        """
        try:
            consistency = self.compute_consistency_score(self._consistency_window_days)
            sharpe = self._compute_rolling_sharpe()

            if consistency <= 0.0:
                logger.info(
                    "LeverageManager.compute_max_safe_leverage: consistency=0 "
                    "-> max_safe_leverage=0.0"
                )
                return 0.0

            # Core formula: leverage scales with both Sharpe quality and
            # consistency.  The kelly_fraction acts as a conservative
            # fractional Kelly damper.
            raw = self._target_sharpe * self._kelly_fraction * consistency

            # If the rolling Sharpe is negative, further penalise
            if sharpe < 0:
                raw *= 0.0
            elif sharpe < self._target_sharpe:
                # Scale linearly between 0 and full
                raw *= sharpe / self._target_sharpe

            max_safe = max(0.0, min(self._max_leverage, raw))

            logger.debug(
                "LeverageManager: consistency=%.3f sharpe=%.3f raw=%.3f "
                "max_safe=%.3f (cap=%.1f)",
                consistency, sharpe, raw, max_safe, self._max_leverage,
            )
            return round(max_safe, 4)
        except Exception as exc:
            logger.error(
                "LeverageManager.compute_max_safe_leverage failed: %s", exc
            )
            return 0.0

    # ------------------------------------------------------------------
    # Current leverage
    # ------------------------------------------------------------------

    def get_current_leverage(
        self,
        positions: List[Dict[str, Any]],
        equity: float,
    ) -> float:
        """
        Compute current portfolio leverage:
            leverage = total_position_value / equity

        ``positions`` is a list of dicts with at least ``size`` and
        ``price`` (or ``entry_price``) keys.

        Returns 0.0 if equity <= 0 or no positions.
        """
        try:
            self._positions = positions
            self._equity = equity

            if equity <= 0:
                logger.warning(
                    "LeverageManager.get_current_leverage: equity=%.2f <= 0",
                    equity,
                )
                self._current_leverage = 0.0
                return 0.0

            total_value = 0.0
            for pos in positions:
                size = abs(float(pos.get("size", 0.0)))
                price = float(
                    pos.get("price", pos.get("entry_price", 0.0))
                )
                total_value += size * price

            leverage = total_value / equity
            self._current_leverage = leverage
            return round(leverage, 4)
        except Exception as exc:
            logger.error(
                "LeverageManager.get_current_leverage failed: %s", exc
            )
            self._current_leverage = 0.0
            return 0.0

    # ------------------------------------------------------------------
    # Leverage check
    # ------------------------------------------------------------------

    def should_reduce_leverage(self) -> bool:
        """
        Return ``True`` if current leverage exceeds the max safe leverage.

        Note: ``get_current_leverage()`` must have been called first
        (or ``get_leverage_report()``) to populate ``_current_leverage``.
        If never called, this returns ``False`` conservatively.

        Edge case: if there is no trade history (max_safe computes to 0
        because consistency is 0), we return ``False`` rather than
        signalling a spurious deleverage.
        """
        try:
            if not self._enabled:
                return False

            # If no current leverage is set, nothing to reduce
            if self._current_leverage <= 0:
                return False

            # If there is no trade history, we cannot compute a meaningful
            # max safe leverage.  Return False to avoid false alarms.
            daily_returns = self._get_daily_returns()
            if not daily_returns:
                return False

            max_safe = self.compute_max_safe_leverage()
            return self._current_leverage > max_safe
        except Exception as exc:
            logger.error(
                "LeverageManager.should_reduce_leverage failed: %s", exc
            )
            return False

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def get_leverage_report(
        self,
        positions: Optional[List[Dict[str, Any]]] = None,
        equity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Produce a comprehensive leverage report.

        If ``positions`` and ``equity`` are provided, current leverage is
        recomputed; otherwise the last cached values are used.

        Returns a dict with:
            current_leverage, max_safe_leverage, consistency_score,
            rolling_sharpe, num_losing_days, total_days,
            status ("ok" / "warning" / "overleveraged"),
            recommended_action
        """
        try:
            # Optionally refresh current leverage
            if positions is not None and equity is not None:
                self.get_current_leverage(positions, equity)

            consistency = self.compute_consistency_score(self._consistency_window_days)
            sharpe = self._compute_rolling_sharpe()
            max_safe = self.compute_max_safe_leverage()
            current = self._current_leverage

            # Count losing / total days for the report
            daily_returns = self._get_daily_returns(self._consistency_window_days)
            total_days = len(daily_returns)
            num_losing_days = sum(1 for _d, p in daily_returns if p < 0)

            # Determine status
            if current <= 0 or max_safe <= 0:
                # No positions or no data -- just report
                status = "ok"
                recommended_action = None
            elif current > max_safe:
                status = "overleveraged"
                if max_safe > 0:
                    reduction_pct = ((current - max_safe) / current) * 100.0
                else:
                    reduction_pct = 100.0
                recommended_action = (
                    f"reduce by {reduction_pct:.1f}% "
                    f"(current {current:.2f}x -> target {max_safe:.2f}x)"
                )
            elif current > max_safe * 0.85:
                status = "warning"
                headroom = ((max_safe - current) / max_safe) * 100.0
                recommended_action = (
                    f"approaching limit -- {headroom:.1f}% headroom remaining "
                    f"(current {current:.2f}x, max {max_safe:.2f}x)"
                )
            else:
                status = "ok"
                recommended_action = None

            report = {
                "current_leverage": round(current, 4),
                "max_safe_leverage": round(max_safe, 4),
                "consistency_score": round(consistency, 4),
                "rolling_sharpe": round(sharpe, 4),
                "num_losing_days": num_losing_days,
                "total_days": total_days,
                "status": status,
                "recommended_action": recommended_action,
            }

            logger.info(
                "LeverageManager report: current=%.2fx max_safe=%.2fx "
                "consistency=%.3f sharpe=%.2f status=%s",
                current, max_safe, consistency, sharpe, status,
            )
            return report

        except Exception as exc:
            logger.error("LeverageManager.get_leverage_report failed: %s", exc)
            return {
                "current_leverage": 0.0,
                "max_safe_leverage": 0.0,
                "consistency_score": 0.0,
                "rolling_sharpe": 0.0,
                "num_losing_days": 0,
                "total_days": 0,
                "status": "ok",
                "recommended_action": None,
            }
