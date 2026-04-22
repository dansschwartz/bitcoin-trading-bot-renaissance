"""
Portfolio Beta Monitor (D1)
===========================
Tracks the portfolio's correlation to BTC price movements.  A delta-neutral
portfolio should have beta close to 0.0.  When |beta| exceeds the configured
threshold an alert is triggered so that operators can hedge or reduce
directional exposure.

The module reads hourly portfolio returns (derived from the ``trades`` and
``decisions`` tables) and hourly BTC returns (from the ``market_data`` table)
to compute a rolling OLS regression:

    portfolio_returns = alpha + beta * btc_returns

All public methods catch exceptions internally and log rather than raise,
following the same resilience pattern used throughout the Renaissance bot.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "target_beta": 0.0,
    "alert_threshold": 0.3,
    "window_hours": 24,
    "check_interval_minutes": 60,
    "auto_hedge": False,
}


class BetaMonitor:
    """Monitors the portfolio's beta to BTC and recommends hedging actions.

    Args:
        config: Full bot configuration dict.  Beta-specific settings are
                read from ``config["beta_monitor"]``.
        db_path: Path to the SQLite database (``data/renaissance_bot.db``).
    """

    def __init__(self, config: Dict[str, Any], db_path: str) -> None:
        self._cfg: Dict[str, Any] = {**_DEFAULTS, **config.get("beta_monitor", {})}
        self._db_path: str = db_path

        self._target_beta: float = float(self._cfg["target_beta"])
        self._alert_threshold: float = float(self._cfg["alert_threshold"])
        self._window_hours: int = int(self._cfg["window_hours"])

        # Cache latest computation for get_report / should_alert
        self._last_result: Optional[Dict[str, Any]] = None

        logger.info(
            "BetaMonitor initialised (db=%s, threshold=%.2f, window=%dh)",
            db_path,
            self._alert_threshold,
            self._window_hours,
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

    def _fetch_hourly_btc_returns(
        self, hours: int
    ) -> List[tuple]:
        """Return (hour_bucket, hourly_return) pairs for BTC-USD.

        Each row is the percentage return from the first to the last price
        observation within a calendar-hour bucket.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        query = """
            SELECT strftime('%Y-%m-%d %H', timestamp) AS hour_bucket,
                   MIN(CASE WHEN sub.rn = 1 THEN price END)  AS open_price,
                   MAX(CASE WHEN sub.rn = sub.cnt THEN price END) AS close_price
            FROM (
                SELECT price, timestamp,
                       ROW_NUMBER() OVER (
                           PARTITION BY strftime('%Y-%m-%d %H', timestamp)
                           ORDER BY timestamp ASC
                       ) AS rn,
                       COUNT(*) OVER (
                           PARTITION BY strftime('%Y-%m-%d %H', timestamp)
                       ) AS cnt
                FROM market_data
                WHERE product_id = 'BTC-USD'
                  AND timestamp >= ?
            ) sub
            GROUP BY hour_bucket
            ORDER BY hour_bucket ASC
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(query, (cutoff,)).fetchall()
        except Exception:
            logger.exception("BetaMonitor: failed to fetch BTC market data")
            return []

        results: List[tuple] = []
        for bucket, open_price, close_price in rows:
            if open_price and close_price and open_price > 0:
                ret = (close_price - open_price) / open_price
                results.append((bucket, ret))
        return results

    def _fetch_hourly_portfolio_returns(
        self, hours: int
    ) -> List[tuple]:
        """Derive hourly portfolio returns from the trades table.

        For each hour bucket we compute the net cash-flow from trades:
            net = SUM(SELL size*price) - SUM(BUY size*price)
        Then we normalise by the total traded notional in that hour to get
        a return-like metric.  Where no trades occurred the hour is skipped.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        query = """
            SELECT strftime('%Y-%m-%d %H', timestamp) AS hour_bucket,
                   SUM(CASE WHEN UPPER(side) = 'SELL' THEN size * price
                            WHEN UPPER(side) = 'BUY'  THEN -size * price
                            ELSE 0 END) AS net_pnl,
                   SUM(size * price) AS total_notional
            FROM trades
            WHERE timestamp >= ?
              AND status != 'FAILED'
            GROUP BY hour_bucket
            ORDER BY hour_bucket ASC
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(query, (cutoff,)).fetchall()
        except Exception:
            logger.exception("BetaMonitor: failed to fetch portfolio returns")
            return []

        results: List[tuple] = []
        for bucket, net_pnl, total_notional in rows:
            if total_notional and total_notional > 0:
                ret = net_pnl / total_notional
                results.append((bucket, ret))
        return results

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_beta(self, window_hours: Optional[int] = None) -> Dict[str, Any]:
        """Compute portfolio beta to BTC over a rolling window.

        Returns a dict with keys:
            beta, alpha, r_squared, p_value, status, message

        Status is one of ``"ok"``, ``"warning"``, ``"alert"``.
        """
        hours = window_hours if window_hours is not None else self._window_hours

        empty_result: Dict[str, Any] = {
            "beta": 0.0,
            "alpha": 0.0,
            "r_squared": 0.0,
            "p_value": 1.0,
            "status": "ok",
            "message": "Insufficient data to compute beta.",
            "window_hours": hours,
            "n_observations": 0,
        }

        try:
            btc_data = self._fetch_hourly_btc_returns(hours)
            port_data = self._fetch_hourly_portfolio_returns(hours)

            if not btc_data or not port_data:
                logger.info("BetaMonitor: no data available for beta computation")
                self._last_result = empty_result
                return empty_result

            # Align on common hour buckets
            btc_map = dict(btc_data)
            port_map = dict(port_data)
            common_buckets = sorted(set(btc_map.keys()) & set(port_map.keys()))

            if len(common_buckets) < 3:
                empty_result["message"] = (
                    f"Only {len(common_buckets)} common hour(s) found; "
                    "need at least 3 for regression."
                )
                logger.info("BetaMonitor: %s", empty_result["message"])
                self._last_result = empty_result
                return empty_result

            btc_returns = np.array([btc_map[b] for b in common_buckets], dtype=np.float64)
            port_returns = np.array([port_map[b] for b in common_buckets], dtype=np.float64)

            # Sanitise: drop any NaN / inf
            valid_mask = np.isfinite(btc_returns) & np.isfinite(port_returns)
            btc_returns = btc_returns[valid_mask]
            port_returns = port_returns[valid_mask]

            if len(btc_returns) < 3:
                empty_result["message"] = "Fewer than 3 valid observations after filtering."
                self._last_result = empty_result
                return empty_result

            # OLS via scipy
            from scipy import stats as sp_stats

            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
                btc_returns, port_returns
            )

            beta = float(slope)
            alpha = float(intercept)
            r_squared = float(r_value ** 2)
            p_val = float(p_value)

            # Determine status
            abs_beta = abs(beta - self._target_beta)
            warning_zone = self._alert_threshold * 0.7  # 70% of threshold

            if abs_beta >= self._alert_threshold:
                status = "alert"
                message = (
                    f"Portfolio beta is {beta:+.4f} (target {self._target_beta:.2f}). "
                    f"|beta - target| = {abs_beta:.4f} EXCEEDS threshold "
                    f"{self._alert_threshold:.2f}. Consider hedging."
                )
            elif abs_beta >= warning_zone:
                status = "warning"
                message = (
                    f"Portfolio beta is {beta:+.4f} (target {self._target_beta:.2f}). "
                    f"|beta - target| = {abs_beta:.4f} is approaching threshold "
                    f"{self._alert_threshold:.2f}."
                )
            else:
                status = "ok"
                message = (
                    f"Portfolio beta is {beta:+.4f} (target {self._target_beta:.2f}). "
                    f"Within acceptable range."
                )

            result: Dict[str, Any] = {
                "beta": round(beta, 6),
                "alpha": round(alpha, 6),
                "r_squared": round(r_squared, 6),
                "p_value": round(p_val, 6),
                "std_err": round(float(std_err), 6),
                "status": status,
                "message": message,
                "window_hours": hours,
                "n_observations": int(len(btc_returns)),
            }

            self._last_result = result
            logger.info(
                "BetaMonitor: beta=%.4f alpha=%.6f R2=%.4f p=%.4f status=%s (%d obs)",
                beta, alpha, r_squared, p_val, status, len(btc_returns),
            )
            return result

        except ImportError:
            logger.error("BetaMonitor: scipy is not installed; cannot compute beta")
            empty_result["message"] = "scipy is not installed."
            self._last_result = empty_result
            return empty_result
        except Exception:
            logger.exception("BetaMonitor.compute_beta failed")
            self._last_result = empty_result
            return empty_result

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    def should_alert(self) -> bool:
        """Return True if the latest beta exceeds the alert threshold."""
        try:
            if self._last_result is None:
                self.compute_beta()
            if self._last_result is None:
                return False
            return self._last_result.get("status") == "alert"
        except Exception:
            logger.exception("BetaMonitor.should_alert failed")
            return False

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        """Return a comprehensive beta report including rolling 7-day beta
        and trend analysis.

        Keys: current_beta, beta_7d_rolling, trend, current_status, message.
        """
        try:
            current = self.compute_beta(window_hours=self._window_hours)
            rolling_7d = self.compute_beta(window_hours=168)  # 7 days

            current_beta = current.get("beta", 0.0)
            beta_7d = rolling_7d.get("beta", 0.0)

            # Trend: compare short-term to long-term beta
            beta_diff = abs(current_beta) - abs(beta_7d)
            if beta_diff > 0.05:
                trend = "increasing_exposure"
            elif beta_diff < -0.05:
                trend = "decreasing_exposure"
            else:
                trend = "stable"

            report: Dict[str, Any] = {
                "current_beta": current_beta,
                "current_alpha": current.get("alpha", 0.0),
                "current_r_squared": current.get("r_squared", 0.0),
                "current_p_value": current.get("p_value", 1.0),
                "current_status": current.get("status", "ok"),
                "current_message": current.get("message", ""),
                "current_window_hours": current.get("window_hours", self._window_hours),
                "current_n_observations": current.get("n_observations", 0),
                "beta_7d_rolling": beta_7d,
                "beta_7d_r_squared": rolling_7d.get("r_squared", 0.0),
                "beta_7d_n_observations": rolling_7d.get("n_observations", 0),
                "trend": trend,
                "target_beta": self._target_beta,
                "alert_threshold": self._alert_threshold,
            }

            logger.info(
                "BetaMonitor report: current=%.4f 7d=%.4f trend=%s",
                current_beta, beta_7d, trend,
            )
            return report

        except Exception:
            logger.exception("BetaMonitor.get_report failed")
            return {
                "current_beta": 0.0,
                "current_alpha": 0.0,
                "current_r_squared": 0.0,
                "current_p_value": 1.0,
                "current_status": "ok",
                "current_message": "Report generation failed.",
                "current_window_hours": self._window_hours,
                "current_n_observations": 0,
                "beta_7d_rolling": 0.0,
                "beta_7d_r_squared": 0.0,
                "beta_7d_n_observations": 0,
                "trend": "unknown",
                "target_beta": self._target_beta,
                "alert_threshold": self._alert_threshold,
            }

    # ------------------------------------------------------------------
    # Hedge recommendation
    # ------------------------------------------------------------------

    def get_hedge_recommendation(self) -> Dict[str, Any]:
        """Suggest a position adjustment to move the portfolio beta closer
        to the target.

        Returns a dict with:
            needs_hedge (bool), direction, recommended_size_usd,
            current_beta, target_beta, rationale
        """
        try:
            if self._last_result is None:
                self.compute_beta()

            result = self._last_result or {}
            current_beta = result.get("beta", 0.0)
            beta_deviation = current_beta - self._target_beta

            # No hedge needed if within acceptable range
            if abs(beta_deviation) < self._alert_threshold * 0.5:
                return {
                    "needs_hedge": False,
                    "direction": "none",
                    "recommended_size_usd": 0.0,
                    "current_beta": current_beta,
                    "target_beta": self._target_beta,
                    "beta_deviation": round(beta_deviation, 6),
                    "rationale": (
                        f"Beta deviation ({beta_deviation:+.4f}) is within "
                        f"acceptable range. No hedge required."
                    ),
                }

            # Estimate portfolio notional from recent trades
            portfolio_notional = self._estimate_portfolio_notional()

            # Hedge size = |beta_deviation| * portfolio_notional
            # If beta is positive (long BTC exposure), we need to SELL BTC
            # If beta is negative (short BTC exposure), we need to BUY BTC
            if beta_deviation > 0:
                direction = "SELL"
            else:
                direction = "BUY"

            hedge_size_usd = abs(beta_deviation) * portfolio_notional
            # Cap at portfolio notional to avoid over-hedging
            hedge_size_usd = min(hedge_size_usd, portfolio_notional)

            recommendation: Dict[str, Any] = {
                "needs_hedge": True,
                "direction": direction,
                "recommended_size_usd": round(hedge_size_usd, 2),
                "current_beta": round(current_beta, 6),
                "target_beta": self._target_beta,
                "beta_deviation": round(beta_deviation, 6),
                "estimated_portfolio_notional": round(portfolio_notional, 2),
                "auto_hedge_enabled": self._cfg.get("auto_hedge", False),
                "rationale": (
                    f"Beta is {current_beta:+.4f} vs target {self._target_beta:.2f}. "
                    f"{direction} ~${hedge_size_usd:,.0f} of BTC-USD to reduce "
                    f"directional exposure."
                ),
            }

            logger.info(
                "BetaMonitor hedge recommendation: %s $%.0f (beta deviation %.4f)",
                direction, hedge_size_usd, beta_deviation,
            )
            return recommendation

        except Exception:
            logger.exception("BetaMonitor.get_hedge_recommendation failed")
            return {
                "needs_hedge": False,
                "direction": "none",
                "recommended_size_usd": 0.0,
                "current_beta": 0.0,
                "target_beta": self._target_beta,
                "beta_deviation": 0.0,
                "rationale": "Hedge recommendation computation failed.",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_portfolio_notional(self) -> float:
        """Estimate total portfolio notional from recent open positions
        and trades.

        Falls back to the sum of recent trade notional over the past 24h
        if open_positions data is unavailable.
        """
        try:
            with self._conn() as conn:
                # Try open_positions first
                row = conn.execute(
                    """
                    SELECT COALESCE(SUM(size * entry_price), 0.0)
                    FROM open_positions
                    WHERE status = 'OPEN'
                    """
                ).fetchone()
                if row and row[0] and row[0] > 0:
                    return float(row[0])

                # Fallback: average hourly notional * hours
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(hours=24)
                ).isoformat()
                row = conn.execute(
                    """
                    SELECT COALESCE(SUM(size * price), 0.0)
                    FROM trades
                    WHERE timestamp >= ?
                      AND status != 'FAILED'
                    """,
                    (cutoff,),
                ).fetchone()
                if row and row[0]:
                    return float(row[0])
        except Exception:
            logger.debug("BetaMonitor: failed to estimate portfolio notional")

        # Absolute fallback
        return 1000.0
