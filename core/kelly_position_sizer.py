"""
Fractional Kelly Position Sizer (A3)
=====================================
Uses the Kelly Criterion to determine optimal position sizes based on
historical win rate and payoff distribution.

Two modes:
  1. **Binary Kelly** -- for arb trades with a clear "win / lose" outcome.
     f* = p - q/b  where  p = win_rate, q = 1-p, b = avg_win/avg_loss.

  2. **Continuous Kelly** -- for variable-return signals.
     Uses ``scipy.optimize.minimize_scalar`` to maximise the expected
     log-growth of the bankroll.

Both modes operate on per-signal-type, per-pair statistics pulled from the
``trades`` table in SQLite.

Falls back to a fixed default (1 % of equity) when fewer than
``min_trades`` samples are available.

Caps at ``max_position_pct`` and applies a fractional Kelly multiplier
(default 25 % of full Kelly) to control tail risk.

Config lives under ``kelly_sizer`` in config.json.

NOTE: The root-level ``position_sizer.py`` (``RenaissancePositionSizer``)
handles *capacity-aware* sizing with market-impact modelling.  This module
is complementary: it sizes purely from the statistical edge the signal has
demonstrated historically.

Design principle: "if unexpected, do nothing" -- every public method
catches exceptions internally and returns a conservative fallback.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Attempt scipy import -- graceful fallback if absent.
try:
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning(
        "scipy not installed; continuous Kelly will fall back to binary Kelly"
    )


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "kelly_fraction": 0.25,
    "min_trades": 30,
    "lookback_trades": 200,
    "max_position_pct": 10.0,
    "default_position_pct": 1.0,
    "recalculate_interval_minutes": 15,
    "negative_kelly_action": "halt",        # "halt" | "reduce" | "default"
}


# ---------------------------------------------------------------------------
# Data class for trade statistics
# ---------------------------------------------------------------------------

@dataclass
class TradeStats:
    """Aggregate statistics for a (signal_type, pair) bucket."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win_bps: float = 0.0
    avg_loss_bps: float = 0.0
    returns_bps: Optional[List[float]] = None  # individual trade returns
    full_kelly_fraction: float = 0.0
    fractional_kelly: float = 0.0
    recommended_position_pct: float = 0.0
    expectancy_per_trade_bps: float = 0.0


# ---------------------------------------------------------------------------
# KellyPositionSizer
# ---------------------------------------------------------------------------

class KellyPositionSizer:
    """
    Fractional Kelly position sizer backed by SQLite trade history.

    Usage::

        sizer = KellyPositionSizer(config, "data/renaissance_bot.db")
        size_usd = sizer.get_position_size(
            signal_dict={"signal_type": "stat_arb", "pair": "BTC-USD",
                         "side": "BUY", "confidence": 0.72},
            equity=25_000.0,
        )
    """

    TRADES_TABLE = "trades"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db_path: str = "data/renaissance_bot.db",
    ):
        raw = (config or {}).get("kelly_sizer", {})
        self.cfg: Dict[str, Any] = {**_DEFAULT_CONFIG, **raw}
        self.db_path = db_path

        # In-memory cache: (signal_type, pair) -> (TradeStats, epoch_calculated)
        self._cache: Dict[Tuple[str, str], Tuple[TradeStats, float]] = {}
        self._recalc_interval = self.cfg["recalculate_interval_minutes"] * 60.0

        logger.info(
            "KellyPositionSizer initialised: fraction=%.0f%%, min_trades=%d, "
            "lookback=%d, max_pos=%.1f%%, db=%s",
            self.cfg["kelly_fraction"] * 100,
            self.cfg["min_trades"],
            self.cfg["lookback_trades"],
            self.cfg["max_position_pct"],
            db_path,
        )

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
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
    # Fetch trade statistics from the DB
    # ------------------------------------------------------------------

    def _fetch_returns(
        self, signal_type: str, pair: str
    ) -> List[float]:
        """
        Fetch per-trade return in basis points from the ``trades`` table.

        We look for rows whose ``product_id`` matches *pair* and whose
        ``algo_used`` contains *signal_type* (the ``algo_used`` column is
        the closest the existing schema has to a signal-type tag).

        If no matching rows exist, returns an empty list.
        """
        try:
            lookback = self.cfg["lookback_trades"]
            with self._conn() as conn:
                rows = conn.execute(
                    f"""
                    SELECT side, price, slippage
                    FROM {self.TRADES_TABLE}
                    WHERE product_id = ?
                      AND (algo_used LIKE ? OR algo_used LIKE ?)
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (pair, f"%{signal_type}%", f"%{signal_type.lower()}%", lookback),
                ).fetchall()

            if not rows:
                return []

            # Build pairwise return: consecutive (BUY, SELL) as a round trip.
            # Each row is (side, price, slippage).
            returns_bps: List[float] = []
            buy_price: Optional[float] = None

            # Process oldest-first for pairing
            for side, price, slippage in reversed(rows):
                if price is None or price <= 0:
                    continue
                slippage = slippage or 0.0

                if side and side.upper() == "BUY":
                    buy_price = price
                elif side and side.upper() == "SELL" and buy_price is not None:
                    ret = ((price - buy_price) / buy_price - slippage) * 10_000.0
                    returns_bps.append(ret)
                    buy_price = None

            return returns_bps

        except Exception as exc:
            logger.error("KellyPositionSizer._fetch_returns failed: %s", exc)
            return []

    def _build_stats(self, signal_type: str, pair: str) -> TradeStats:
        """Build a TradeStats object from historical returns."""
        returns = self._fetch_returns(signal_type, pair)
        stats = TradeStats(returns_bps=returns)
        stats.total_trades = len(returns)

        if not returns:
            return stats

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        stats.wins = len(wins)
        stats.losses = len(losses)
        stats.win_rate = len(wins) / len(returns) if returns else 0.0
        stats.avg_win_bps = sum(wins) / len(wins) if wins else 0.0
        stats.avg_loss_bps = abs(sum(losses) / len(losses)) if losses else 0.0

        # Expectancy
        stats.expectancy_per_trade_bps = (
            stats.win_rate * stats.avg_win_bps
            - (1 - stats.win_rate) * stats.avg_loss_bps
        )

        # Full Kelly (binary)
        fk = self._binary_kelly(stats.win_rate, stats.avg_win_bps, stats.avg_loss_bps)
        stats.full_kelly_fraction = fk
        stats.fractional_kelly = fk * self.cfg["kelly_fraction"]
        stats.recommended_position_pct = min(
            stats.fractional_kelly * 100.0,
            self.cfg["max_position_pct"],
        )

        return stats

    def _get_cached_stats(self, signal_type: str, pair: str) -> TradeStats:
        """Return cached stats or rebuild if stale."""
        key = (signal_type, pair)
        now = time.time()
        cached = self._cache.get(key)
        if cached and (now - cached[1]) < self._recalc_interval:
            return cached[0]

        stats = self._build_stats(signal_type, pair)
        self._cache[key] = (stats, now)
        return stats

    # ------------------------------------------------------------------
    # Binary Kelly
    # ------------------------------------------------------------------

    @staticmethod
    def _binary_kelly(
        win_rate: float, avg_win_bps: float, avg_loss_bps: float
    ) -> float:
        """
        Classic Kelly fraction for binary outcomes.

        f* = p - q / b
        where p = win_rate, q = 1 - p, b = avg_win / avg_loss.
        """
        if win_rate <= 0 or avg_loss_bps <= 0:
            return 0.0
        b = avg_win_bps / avg_loss_bps
        q = 1.0 - win_rate
        f = win_rate - q / b
        return max(f, 0.0)

    def compute_kelly_binary(self, signal_type: str, pair: str) -> float:
        """
        Public API: binary Kelly fraction for a given signal_type and pair.

        Returns a fraction in [0, 1].  Falls back to 0 if insufficient data.
        """
        try:
            stats = self._get_cached_stats(signal_type, pair)
            if stats.total_trades < self.cfg["min_trades"]:
                return 0.0
            return max(stats.full_kelly_fraction, 0.0)
        except Exception as exc:
            logger.error("compute_kelly_binary failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Continuous Kelly
    # ------------------------------------------------------------------

    def compute_kelly_continuous(self, signal_type: str, pair: str) -> float:
        """
        Continuous Kelly via optimisation of expected log-growth.

        Maximises  E[log(1 + f * r_i)]  over the observed return distribution.

        Falls back to binary Kelly if scipy is unavailable or if the
        optimisation fails.
        """
        try:
            stats = self._get_cached_stats(signal_type, pair)
            if stats.total_trades < self.cfg["min_trades"]:
                return 0.0

            returns = stats.returns_bps
            if not returns:
                return 0.0

            # Convert bps to decimal fraction for the optimiser
            rets = [r / 10_000.0 for r in returns]

            if not SCIPY_AVAILABLE:
                return self.compute_kelly_binary(signal_type, pair)

            def neg_log_growth(f: float) -> float:
                """Negative expected log-growth (we minimise)."""
                total = 0.0
                for r in rets:
                    val = 1.0 + f * r
                    if val <= 0:
                        return 1e10  # penalty for ruin
                    total += math.log(val)
                return -total / len(rets)

            result = minimize_scalar(
                neg_log_growth,
                bounds=(0.0, 2.0),
                method="bounded",
            )

            if result.success:
                f_star = max(result.x, 0.0)
                return f_star
            else:
                logger.debug(
                    "Continuous Kelly optimisation did not converge for %s/%s; "
                    "falling back to binary Kelly.",
                    signal_type, pair,
                )
                return max(stats.full_kelly_fraction, 0.0)

        except Exception as exc:
            logger.error("compute_kelly_continuous failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Main sizing API
    # ------------------------------------------------------------------

    def get_position_size(
        self,
        signal_dict: Dict[str, Any],
        equity: float,
    ) -> float:
        """
        Determine the dollar amount to allocate to this signal.

        Parameters
        ----------
        signal_dict : dict
            Must contain ``signal_type`` and ``pair``.
            May contain ``confidence`` (0-1) for scaling.
        equity : float
            Current account equity in USD.

        Returns
        -------
        float
            Dollar amount (>= 0) to allocate.  Zero means "do not trade".
        """
        try:
            if equity <= 0:
                return 0.0

            signal_type = signal_dict.get("signal_type", "unknown")
            pair = signal_dict.get("pair", "BTC-USD")
            confidence = signal_dict.get("confidence", 0.5)

            stats = self._get_cached_stats(signal_type, pair)

            # Not enough data: use the conservative default
            if stats.total_trades < self.cfg["min_trades"]:
                default_pct = self.cfg["default_position_pct"]
                size = equity * (default_pct / 100.0) * confidence
                logger.debug(
                    "KellySizer: insufficient data (%d < %d trades) for %s/%s, "
                    "using default %.1f%% -> $%.2f",
                    stats.total_trades, self.cfg["min_trades"],
                    signal_type, pair, default_pct, size,
                )
                return max(size, 0.0)

            # Negative expectancy handling
            if stats.expectancy_per_trade_bps <= 0:
                action = self.cfg["negative_kelly_action"]
                if action == "halt":
                    logger.info(
                        "KellySizer: negative expectancy for %s/%s (%.1f bps), "
                        "halting.",
                        signal_type, pair, stats.expectancy_per_trade_bps,
                    )
                    return 0.0
                elif action == "reduce":
                    size = equity * (self.cfg["default_position_pct"] / 100.0) * 0.25
                    return max(size, 0.0)
                else:
                    # "default"
                    size = equity * (self.cfg["default_position_pct"] / 100.0)
                    return max(size, 0.0)

            # Try continuous Kelly first, fall back to binary.
            kelly_f = self.compute_kelly_continuous(signal_type, pair)
            if kelly_f <= 0:
                kelly_f = self.compute_kelly_binary(signal_type, pair)
            if kelly_f <= 0:
                return 0.0

            # Apply fractional Kelly
            frac = kelly_f * self.cfg["kelly_fraction"]

            # Scale by confidence
            frac *= confidence

            # Cap at max position pct
            max_frac = self.cfg["max_position_pct"] / 100.0
            frac = min(frac, max_frac)

            size = equity * frac
            logger.debug(
                "KellySizer: %s/%s kelly_f=%.4f frac=%.4f conf=%.2f -> $%.2f "
                "(%.2f%% of $%.0f equity)",
                signal_type, pair, kelly_f, frac, confidence,
                size, frac * 100, equity,
            )
            return max(size, 0.0)

        except Exception as exc:
            logger.error("KellyPositionSizer.get_position_size failed: %s", exc)
            # Conservative fallback
            return max(equity * (self.cfg["default_position_pct"] / 100.0) * 0.5, 0.0)

    # ------------------------------------------------------------------
    # Statistics API
    # ------------------------------------------------------------------

    def get_statistics(self, signal_type: str, pair: str) -> Dict[str, Any]:
        """
        Return a JSON-safe dictionary of Kelly sizing statistics for
        a given signal_type and pair.
        """
        try:
            stats = self._get_cached_stats(signal_type, pair)

            # Also compute continuous Kelly for comparison
            cont_kelly = 0.0
            if stats.total_trades >= self.cfg["min_trades"]:
                cont_kelly = self.compute_kelly_continuous(signal_type, pair)

            return {
                "total_trades": stats.total_trades,
                "win_rate": round(stats.win_rate, 4),
                "avg_win_bps": round(stats.avg_win_bps, 2),
                "avg_loss_bps": round(stats.avg_loss_bps, 2),
                "full_kelly_fraction": round(stats.full_kelly_fraction, 4),
                "continuous_kelly_fraction": round(cont_kelly, 4),
                "fractional_kelly": round(
                    stats.full_kelly_fraction * self.cfg["kelly_fraction"], 4,
                ),
                "recommended_position_pct": round(stats.recommended_position_pct, 2),
                "expectancy_per_trade_bps": round(stats.expectancy_per_trade_bps, 2),
                "sufficient_data": stats.total_trades >= self.cfg["min_trades"],
                "config": {
                    "kelly_fraction": self.cfg["kelly_fraction"],
                    "min_trades": self.cfg["min_trades"],
                    "lookback_trades": self.cfg["lookback_trades"],
                    "max_position_pct": self.cfg["max_position_pct"],
                    "default_position_pct": self.cfg["default_position_pct"],
                    "negative_kelly_action": self.cfg["negative_kelly_action"],
                },
            }
        except Exception as exc:
            logger.error("KellyPositionSizer.get_statistics failed: %s", exc)
            return {"error": str(exc)}
