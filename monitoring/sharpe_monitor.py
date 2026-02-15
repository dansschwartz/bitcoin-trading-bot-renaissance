"""
Rolling Sharpe Ratio Health Monitor (E3)
=========================================
The primary system health metric.  Computes rolling Sharpe ratios over
multiple time windows (7d, 30d, 90d), the Sortino ratio, maximum drawdown,
and Calmar ratio.  When the Sharpe ratio deteriorates below configurable
thresholds, the monitor recommends reducing position sizes or halting
trading entirely.

Daily P&L is derived from the ``trades`` table by summing the net cash flow
(SELL proceeds minus BUY costs) for each calendar day.

All public methods catch exceptions internally and log rather than raise,
following the same resilience pattern used throughout the Renaissance bot.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "healthy_sharpe": 1.0,
    "warning_sharpe": 0.5,
    "critical_sharpe": 0.0,
    "risk_free_rate": 0.05,
    "check_interval_minutes": 60,
    "auto_reduce_exposure": True,
}


class SharpeMonitor:
    """Monitors rolling Sharpe ratios and derived risk metrics to assess
    overall system health.

    Args:
        config: Full bot configuration dict.  Sharpe-specific settings are
                read from ``config["sharpe_monitor"]``.
        db_path: Path to the SQLite database (``data/renaissance_bot.db``).
    """

    def __init__(self, config: Dict[str, Any], db_path: str) -> None:
        self._cfg: Dict[str, Any] = {
            **_DEFAULTS,
            **config.get("sharpe_monitor", {}),
        }
        self._db_path: str = db_path

        self._healthy_sharpe: float = float(self._cfg["healthy_sharpe"])
        self._warning_sharpe: float = float(self._cfg["warning_sharpe"])
        self._critical_sharpe: float = float(self._cfg["critical_sharpe"])
        self._risk_free_rate: float = float(self._cfg["risk_free_rate"])

        # Cache for the most recent report
        self._last_report: Optional[Dict[str, Any]] = None

        logger.info(
            "SharpeMonitor initialised (db=%s, healthy=%.2f, warning=%.2f, "
            "critical=%.2f, rf=%.4f)",
            db_path,
            self._healthy_sharpe,
            self._warning_sharpe,
            self._critical_sharpe,
            self._risk_free_rate,
        )

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a short-lived WAL-mode connection."""
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def _fetch_daily_pnl(self, days: int) -> List[Tuple[str, float]]:
        """Return (date_str, daily_net_pnl) pairs for the last N days.

        Daily P&L is approximated as:
            SUM(SELL trades: size*price) - SUM(BUY trades: size*price)
        for each calendar day.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )
        query = """
            SELECT date(timestamp) AS trade_date,
                   SUM(CASE WHEN UPPER(side) = 'SELL' THEN size * price
                            WHEN UPPER(side) = 'BUY'  THEN -size * price
                            ELSE 0 END) AS daily_pnl
            FROM trades
            WHERE date(timestamp) >= ?
              AND status != 'FAILED'
            GROUP BY trade_date
            ORDER BY trade_date ASC
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(query, (cutoff,)).fetchall()
            return [
                (str(r[0]), float(r[1])) for r in rows
                if r[0] is not None and r[1] is not None
            ]
        except Exception:
            logger.exception("SharpeMonitor: failed to fetch daily P&L")
            return []

    def _fetch_portfolio_equity_series(self, days: int) -> np.ndarray:
        """Build a cumulative equity curve from daily P&L.

        Returns an array of daily equity values starting from a base of
        1000.0 (arbitrary normalisation for drawdown computation).
        """
        pnl_series = self._fetch_daily_pnl(days)
        if not pnl_series:
            return np.array([], dtype=np.float64)

        daily_pnl = np.array([p for _, p in pnl_series], dtype=np.float64)
        equity = np.cumsum(daily_pnl) + 1000.0  # start from 1000
        return equity

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def compute_rolling_sharpe(self, window_days: int = 30) -> float:
        """Compute the annualised Sharpe ratio over the last *window_days*.

        Formula:
            Sharpe = (mean_daily_return - daily_risk_free) / std_daily_return
                     * sqrt(365)

        Returns 0.0 when there is insufficient data or zero volatility.
        """
        try:
            pnl_series = self._fetch_daily_pnl(window_days)

            if len(pnl_series) < 2:
                logger.info(
                    "SharpeMonitor: only %d day(s) of data for %d-day Sharpe",
                    len(pnl_series),
                    window_days,
                )
                return 0.0

            daily_returns = np.array(
                [p for _, p in pnl_series], dtype=np.float64
            )

            # Convert annualised risk-free rate to daily
            daily_rf = self._risk_free_rate / 365.0

            mean_return = float(np.mean(daily_returns))
            std_return = float(np.std(daily_returns, ddof=1))

            if std_return < 1e-12:
                logger.info(
                    "SharpeMonitor: zero volatility over %d days; "
                    "Sharpe set to 0.0",
                    window_days,
                )
                return 0.0

            sharpe = (mean_return - daily_rf) / std_return * math.sqrt(365)

            logger.debug(
                "SharpeMonitor: %d-day Sharpe = %.4f "
                "(mean=%.6f, std=%.6f, rf_daily=%.8f, n=%d)",
                window_days,
                sharpe,
                mean_return,
                std_return,
                daily_rf,
                len(daily_returns),
            )
            return round(sharpe, 4)

        except Exception:
            logger.exception("SharpeMonitor.compute_rolling_sharpe failed")
            return 0.0

    def _compute_sortino(self, window_days: int = 30) -> float:
        """Compute the annualised Sortino ratio over the last *window_days*.

        Like Sharpe, but uses downside deviation (std of negative returns
        only) instead of total standard deviation.
        """
        try:
            pnl_series = self._fetch_daily_pnl(window_days)

            if len(pnl_series) < 2:
                return 0.0

            daily_returns = np.array(
                [p for _, p in pnl_series], dtype=np.float64
            )

            daily_rf = self._risk_free_rate / 365.0
            excess_returns = daily_returns - daily_rf

            # Downside deviation: std of returns below the target (0)
            downside = excess_returns[excess_returns < 0]

            if len(downside) < 1:
                # No negative returns -- effectively infinite Sortino
                # Cap at a high but finite value
                return 10.0

            downside_std = float(np.std(downside, ddof=1))

            if downside_std < 1e-12:
                return 10.0

            mean_excess = float(np.mean(excess_returns))
            sortino = mean_excess / downside_std * math.sqrt(365)

            return round(sortino, 4)

        except Exception:
            logger.exception("SharpeMonitor._compute_sortino failed")
            return 0.0

    def _compute_max_drawdown(self, days: int = 90) -> float:
        """Compute the maximum drawdown percentage over the last N days.

        Returns a non-negative float representing the drawdown as a
        fraction (e.g. 0.15 for 15%).
        """
        try:
            equity = self._fetch_portfolio_equity_series(days)

            if len(equity) < 2:
                return 0.0

            # Running maximum
            running_max = np.maximum.accumulate(equity)

            # Drawdown at each point
            drawdowns = (running_max - equity) / running_max

            # Replace NaN/inf (from division by zero when equity starts at 0)
            drawdowns = np.where(np.isfinite(drawdowns), drawdowns, 0.0)

            max_dd = float(np.max(drawdowns))
            return round(max(0.0, max_dd), 6)

        except Exception:
            logger.exception("SharpeMonitor._compute_max_drawdown failed")
            return 0.0

    def _compute_calmar_ratio(self, days: int = 90) -> float:
        """Compute the Calmar ratio: annualised return / max drawdown.

        Returns 0.0 if max drawdown is zero or data is insufficient.
        """
        try:
            pnl_series = self._fetch_daily_pnl(days)

            if len(pnl_series) < 2:
                return 0.0

            daily_returns = np.array(
                [p for _, p in pnl_series], dtype=np.float64
            )
            total_return = float(np.sum(daily_returns))

            # Annualise: scale to 365 days
            n_days = len(daily_returns)
            annualised_return = total_return * (365.0 / n_days) if n_days > 0 else 0.0

            max_dd = self._compute_max_drawdown(days)

            if max_dd < 1e-12:
                # No drawdown; cap Calmar at a high value if returns are positive
                return 10.0 if annualised_return > 0 else 0.0

            # For Calmar, we use the absolute $ drawdown relative to equity
            # But since we already have max_dd as a fraction, normalise
            equity = self._fetch_portfolio_equity_series(days)
            if len(equity) == 0:
                return 0.0

            peak_equity = float(np.max(equity))
            absolute_dd = max_dd * peak_equity

            if absolute_dd < 1e-12:
                return 10.0 if annualised_return > 0 else 0.0

            calmar = annualised_return / absolute_dd

            return round(calmar, 4)

        except Exception:
            logger.exception("SharpeMonitor._compute_calmar_ratio failed")
            return 0.0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        """Generate a comprehensive risk/health report.

        Returns a dict with:
            sharpe_7d, sharpe_30d, sharpe_90d,
            sortino_30d, max_drawdown_pct, calmar_ratio,
            status ("healthy" | "warning" | "critical"),
            trend ("improving" | "stable" | "deteriorating"),
            exposure_multiplier
        """
        try:
            sharpe_7d = self.compute_rolling_sharpe(window_days=7)
            sharpe_30d = self.compute_rolling_sharpe(window_days=30)
            sharpe_90d = self.compute_rolling_sharpe(window_days=90)
            sortino_30d = self._compute_sortino(window_days=30)
            max_dd = self._compute_max_drawdown(days=90)
            calmar = self._compute_calmar_ratio(days=90)

            # Status based on 30-day Sharpe
            if sharpe_30d >= self._healthy_sharpe:
                status = "healthy"
            elif sharpe_30d >= self._warning_sharpe:
                status = "warning"
            else:
                status = "critical"

            # Trend: compare 7d vs 30d Sharpe
            sharpe_diff = sharpe_7d - sharpe_30d
            if sharpe_diff > 0.2:
                trend = "improving"
            elif sharpe_diff < -0.2:
                trend = "deteriorating"
            else:
                trend = "stable"

            # Exposure multiplier
            exposure_mult = self._compute_exposure_multiplier(sharpe_30d)

            report: Dict[str, Any] = {
                "sharpe_7d": sharpe_7d,
                "sharpe_30d": sharpe_30d,
                "sharpe_90d": sharpe_90d,
                "sortino_30d": sortino_30d,
                "max_drawdown_pct": round(max_dd * 100.0, 4),
                "calmar_ratio": calmar,
                "status": status,
                "trend": trend,
                "exposure_multiplier": exposure_mult,
                "healthy_threshold": self._healthy_sharpe,
                "warning_threshold": self._warning_sharpe,
                "critical_threshold": self._critical_sharpe,
                "risk_free_rate": self._risk_free_rate,
                "auto_reduce_exposure": self._cfg.get("auto_reduce_exposure", True),
            }

            self._last_report = report
            logger.info(
                "SharpeMonitor report: 7d=%.2f 30d=%.2f 90d=%.2f "
                "sortino=%.2f maxDD=%.2f%% calmar=%.2f "
                "status=%s trend=%s mult=%.2f",
                sharpe_7d,
                sharpe_30d,
                sharpe_90d,
                sortino_30d,
                max_dd * 100.0,
                calmar,
                status,
                trend,
                exposure_mult,
            )
            return report

        except Exception:
            logger.exception("SharpeMonitor.get_report failed")
            fallback: Dict[str, Any] = {
                "sharpe_7d": 0.0,
                "sharpe_30d": 0.0,
                "sharpe_90d": 0.0,
                "sortino_30d": 0.0,
                "max_drawdown_pct": 0.0,
                "calmar_ratio": 0.0,
                "status": "critical",
                "trend": "unknown",
                "exposure_multiplier": 0.0,
                "healthy_threshold": self._healthy_sharpe,
                "warning_threshold": self._warning_sharpe,
                "critical_threshold": self._critical_sharpe,
                "risk_free_rate": self._risk_free_rate,
                "auto_reduce_exposure": self._cfg.get("auto_reduce_exposure", True),
            }
            self._last_report = fallback
            return fallback

    # ------------------------------------------------------------------
    # Exposure decisions
    # ------------------------------------------------------------------

    def should_reduce_exposure(self) -> bool:
        """Return True if the 30-day Sharpe is below the warning threshold.

        When True, the bot should reduce position sizes or stop opening
        new positions until performance recovers.
        """
        try:
            if self._last_report is None:
                self.get_report()
            report = self._last_report or {}
            sharpe_30d = report.get("sharpe_30d", 0.0)
            should_reduce = sharpe_30d < self._warning_sharpe
            if should_reduce:
                logger.warning(
                    "SharpeMonitor: REDUCE EXPOSURE recommended "
                    "(sharpe_30d=%.2f < warning=%.2f)",
                    sharpe_30d,
                    self._warning_sharpe,
                )
            return should_reduce
        except Exception:
            logger.exception("SharpeMonitor.should_reduce_exposure failed")
            # Fail-safe: recommend reducing if we cannot determine health
            return True

    def get_exposure_multiplier(self) -> float:
        """Return a position-size multiplier based on current system health.

        - 1.0 if healthy (Sharpe >= healthy_sharpe)
        - 0.5 if warning (warning_sharpe <= Sharpe < healthy_sharpe)
        - 0.0 if critical (Sharpe < warning_sharpe)

        Callers should multiply their target position size by this value.
        """
        try:
            if self._last_report is None:
                self.get_report()
            report = self._last_report or {}
            sharpe_30d = report.get("sharpe_30d", 0.0)
            return self._compute_exposure_multiplier(sharpe_30d)
        except Exception:
            logger.exception("SharpeMonitor.get_exposure_multiplier failed")
            # Fail-safe: halt trading if we cannot determine health
            return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_exposure_multiplier(self, sharpe: float) -> float:
        """Map a Sharpe ratio value to an exposure multiplier."""
        if sharpe >= self._healthy_sharpe:
            return 1.0
        elif sharpe >= self._warning_sharpe:
            # Linear interpolation between 0.5 and 1.0
            range_width = self._healthy_sharpe - self._warning_sharpe
            if range_width > 1e-12:
                frac = (sharpe - self._warning_sharpe) / range_width
                return round(0.5 + 0.5 * frac, 4)
            return 0.5
        else:
            # Below warning threshold
            # Linear interpolation between 0.0 and 0.5
            range_width = self._warning_sharpe - self._critical_sharpe
            if range_width > 1e-12 and sharpe > self._critical_sharpe:
                frac = (sharpe - self._critical_sharpe) / range_width
                return round(0.5 * frac, 4)
            return 0.0
