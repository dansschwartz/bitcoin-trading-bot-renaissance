"""
Capacity Wall Detector (D2)
============================
Tracks the relationship between trade size and realised slippage for each
trading pair.  As trade sizes grow, slippage eventually rises
disproportionately -- the "capacity wall".  This module fits a linear
regression of slippage_bps vs trade_size_usd and flags pairs as
capacity-constrained when the estimated slippage at current average size
approaches or exceeds the configured wall (default 5 bps).

Data is read from the ``devil_tracker`` table which records per-trade
slippage and fill details.

All public methods catch exceptions internally and log rather than raise,
following the same resilience pattern used throughout the Renaissance bot.
"""

from __future__ import annotations

import logging
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
    "slippage_wall_bps": 5.0,
    "check_interval_minutes": 60,
    "min_trades_for_analysis": 20,
    "alert_on_constrained": True,
}


class CapacityMonitor:
    """Detects capacity constraints per trading pair by analysing the
    relationship between trade size and realised slippage.

    Args:
        config: Full bot configuration dict.  Capacity-specific settings are
                read from ``config["capacity_monitor"]``.
        db_path: Path to the SQLite database (``data/renaissance_bot.db``).
    """

    def __init__(self, config: Dict[str, Any], db_path: str) -> None:
        self._cfg: Dict[str, Any] = {
            **_DEFAULTS,
            **config.get("capacity_monitor", {}),
        }
        self._db_path: str = db_path

        self._slippage_wall_bps: float = float(self._cfg["slippage_wall_bps"])
        self._min_trades: int = int(self._cfg["min_trades_for_analysis"])

        # Cache per-pair analysis results
        self._cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "CapacityMonitor initialised (db=%s, wall=%.1f bps, min_trades=%d)",
            db_path,
            self._slippage_wall_bps,
            self._min_trades,
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

    def _fetch_pair_trades(self, pair: str) -> List[Tuple[float, float]]:
        """Return (trade_size_usd, slippage_bps) tuples for a given pair.

        Only includes filled trades with valid slippage and price/quantity.
        """
        query = """
            SELECT fill_price * fill_quantity AS trade_size_usd,
                   slippage_bps
            FROM devil_tracker
            WHERE pair = ?
              AND fill_price IS NOT NULL
              AND fill_quantity IS NOT NULL
              AND slippage_bps IS NOT NULL
              AND fill_price > 0
              AND fill_quantity > 0
            ORDER BY signal_timestamp ASC
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(query, (pair,)).fetchall()
            return [(float(r[0]), float(r[1])) for r in rows if r[0] > 0]
        except Exception:
            logger.exception(
                "CapacityMonitor: failed to fetch trades for pair=%s", pair
            )
            return []

    def _fetch_all_pairs(self) -> List[str]:
        """Return a list of all distinct pairs in the devil_tracker table."""
        query = """
            SELECT DISTINCT pair
            FROM devil_tracker
            WHERE fill_price IS NOT NULL
              AND fill_quantity IS NOT NULL
              AND slippage_bps IS NOT NULL
            ORDER BY pair
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(query).fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception:
            logger.exception("CapacityMonitor: failed to fetch pairs list")
            return []

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze_capacity(self, pair: str) -> Dict[str, Any]:
        """Analyse the capacity characteristics of a single trading pair.

        Returns a dict with:
            pair, slope_bps_per_1000usd, intercept_bps,
            capacity_status ("ok" | "warning" | "constrained"),
            estimated_max_size_usd, current_avg_size_usd,
            headroom_pct, n_trades, r_squared
        """
        insufficient: Dict[str, Any] = {
            "pair": pair,
            "slope_bps_per_1000usd": 0.0,
            "intercept_bps": 0.0,
            "capacity_status": "ok",
            "estimated_max_size_usd": float("inf"),
            "current_avg_size_usd": 0.0,
            "headroom_pct": 100.0,
            "n_trades": 0,
            "r_squared": 0.0,
            "message": "Insufficient data for capacity analysis.",
        }

        try:
            data = self._fetch_pair_trades(pair)

            if len(data) < self._min_trades:
                insufficient["n_trades"] = len(data)
                insufficient["message"] = (
                    f"Only {len(data)} trades for {pair}; "
                    f"need at least {self._min_trades}."
                )
                logger.info("CapacityMonitor: %s", insufficient["message"])
                self._cache[pair] = insufficient
                return insufficient

            sizes = np.array([d[0] for d in data], dtype=np.float64)
            slippages = np.array([d[1] for d in data], dtype=np.float64)

            # Filter out extreme outliers (beyond 3 std devs)
            if len(sizes) > 10:
                size_mean, size_std = np.mean(sizes), np.std(sizes)
                slip_mean, slip_std = np.mean(slippages), np.std(slippages)

                if size_std > 0 and slip_std > 0:
                    size_mask = np.abs(sizes - size_mean) < 3 * size_std
                    slip_mask = np.abs(slippages - slip_mean) < 3 * slip_std
                    combined_mask = size_mask & slip_mask

                    # Only apply if we keep enough data points
                    if np.sum(combined_mask) >= self._min_trades:
                        sizes = sizes[combined_mask]
                        slippages = slippages[combined_mask]

            # Sanitise
            valid = np.isfinite(sizes) & np.isfinite(slippages) & (sizes > 0)
            sizes = sizes[valid]
            slippages = slippages[valid]

            if len(sizes) < 3:
                insufficient["message"] = "Fewer than 3 valid data points after filtering."
                self._cache[pair] = insufficient
                return insufficient

            # Linear regression: slippage_bps = intercept + slope * trade_size_usd
            coeffs = np.polyfit(sizes, slippages, 1)
            slope = float(coeffs[0])      # bps per 1 USD
            intercept = float(coeffs[1])   # bps at zero size

            # R-squared
            predicted = np.polyval(coeffs, sizes)
            ss_res = np.sum((slippages - predicted) ** 2)
            ss_tot = np.sum((slippages - np.mean(slippages)) ** 2)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            r_squared = max(0.0, min(1.0, r_squared))

            # Slope per $1000
            slope_per_1000 = slope * 1000.0

            # Current average trade size
            current_avg_size = float(np.mean(sizes))

            # Estimated max size: size where slippage = wall
            # slippage_wall = intercept + slope * max_size
            # max_size = (slippage_wall - intercept) / slope
            if slope > 1e-12:
                estimated_max_size = (self._slippage_wall_bps - intercept) / slope
                # If intercept already exceeds wall, max_size will be negative
                estimated_max_size = max(0.0, estimated_max_size)
            elif intercept >= self._slippage_wall_bps:
                # Flat or negative slope but already above wall
                estimated_max_size = 0.0
            else:
                # Flat or negative slope and below wall -- effectively unlimited
                estimated_max_size = float("inf")

            # Headroom: how much room before hitting the wall
            if estimated_max_size == float("inf"):
                headroom_pct = 100.0
            elif estimated_max_size <= 0:
                headroom_pct = 0.0
            elif current_avg_size > 0:
                headroom_pct = max(
                    0.0,
                    ((estimated_max_size - current_avg_size) / estimated_max_size) * 100.0,
                )
            else:
                headroom_pct = 100.0

            # Predicted slippage at current average size
            current_predicted_slip = intercept + slope * current_avg_size

            # Determine status
            if current_predicted_slip >= self._slippage_wall_bps:
                status = "constrained"
            elif headroom_pct < 30.0 or current_predicted_slip >= self._slippage_wall_bps * 0.7:
                status = "warning"
            else:
                status = "ok"

            # Build message
            if status == "constrained":
                message = (
                    f"{pair} is CAPACITY CONSTRAINED. Predicted slippage at avg size "
                    f"${current_avg_size:,.0f} is {current_predicted_slip:.2f} bps "
                    f"(wall={self._slippage_wall_bps:.1f} bps). "
                    f"Reduce trade sizes or split orders."
                )
            elif status == "warning":
                message = (
                    f"{pair} is approaching capacity wall. "
                    f"Headroom: {headroom_pct:.1f}%. "
                    f"Estimated max size: ${estimated_max_size:,.0f}."
                )
            else:
                message = (
                    f"{pair} has adequate capacity. "
                    f"Headroom: {headroom_pct:.1f}%. "
                    f"Slope: {slope_per_1000:.3f} bps/$1000."
                )

            result: Dict[str, Any] = {
                "pair": pair,
                "slope_bps_per_1000usd": round(slope_per_1000, 6),
                "intercept_bps": round(intercept, 4),
                "capacity_status": status,
                "estimated_max_size_usd": (
                    round(estimated_max_size, 2)
                    if estimated_max_size != float("inf")
                    else float("inf")
                ),
                "current_avg_size_usd": round(current_avg_size, 2),
                "current_predicted_slippage_bps": round(current_predicted_slip, 4),
                "headroom_pct": round(headroom_pct, 2),
                "n_trades": int(len(sizes)),
                "r_squared": round(r_squared, 6),
                "message": message,
            }

            self._cache[pair] = result
            logger.info(
                "CapacityMonitor [%s]: status=%s slope=%.4f bps/$1k "
                "max_size=$%.0f headroom=%.1f%% (%d trades, R2=%.3f)",
                pair,
                status,
                slope_per_1000,
                estimated_max_size if estimated_max_size != float("inf") else -1,
                headroom_pct,
                len(sizes),
                r_squared,
            )
            return result

        except Exception:
            logger.exception(
                "CapacityMonitor.analyze_capacity failed for pair=%s", pair
            )
            self._cache[pair] = insufficient
            return insufficient

    # ------------------------------------------------------------------
    # Bulk analysis
    # ------------------------------------------------------------------

    def get_all_capacities(self) -> Dict[str, Dict[str, Any]]:
        """Analyse capacity for every pair found in the devil_tracker table.

        Returns a dict mapping pair -> capacity analysis result.
        """
        try:
            pairs = self._fetch_all_pairs()
            if not pairs:
                logger.info("CapacityMonitor: no pairs found in devil_tracker")
                return {}

            results: Dict[str, Dict[str, Any]] = {}
            for pair in pairs:
                results[pair] = self.analyze_capacity(pair)

            constrained = [
                p for p, r in results.items()
                if r.get("capacity_status") == "constrained"
            ]
            warning = [
                p for p, r in results.items()
                if r.get("capacity_status") == "warning"
            ]

            logger.info(
                "CapacityMonitor: analysed %d pairs — %d constrained, %d warning, %d ok",
                len(results),
                len(constrained),
                len(warning),
                len(results) - len(constrained) - len(warning),
            )
            return results

        except Exception:
            logger.exception("CapacityMonitor.get_all_capacities failed")
            return {}

    # ------------------------------------------------------------------
    # Recommended max size
    # ------------------------------------------------------------------

    def get_recommended_max_size(self, pair: str) -> float:
        """Return the maximum recommended trade size (in USD) for a pair
        before slippage exceeds the configured wall.

        Returns ``float('inf')`` if the pair is unconstrained or if there
        is insufficient data.  Returns ``0.0`` if already constrained at
        any size.
        """
        try:
            if pair not in self._cache:
                self.analyze_capacity(pair)

            result = self._cache.get(pair, {})
            max_size = result.get("estimated_max_size_usd", float("inf"))

            # Apply a safety margin of 80% of the theoretical max
            if max_size != float("inf") and max_size > 0:
                return round(max_size * 0.8, 2)

            return max_size

        except Exception:
            logger.exception(
                "CapacityMonitor.get_recommended_max_size failed for pair=%s", pair
            )
            return float("inf")
