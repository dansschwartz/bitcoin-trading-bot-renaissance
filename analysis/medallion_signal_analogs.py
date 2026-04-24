"""
Medallion Signal Analogs — Crypto-Adapted Versions of Classic Renaissance Strategies
=====================================================================================
Three micro-strategies inspired by patterns described in "The Man Who Solved the Market":

1. Sharp Move Reversion — Mean reversion after large moves (>3%)
2. Hour-of-Day Seasonality — Time-based return patterns
3. Funding Settlement Timing — Pre-settlement positioning trades

"The signal doesn't have to be big. It has to be real."
"""

from __future__ import annotations

import logging
import math
import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional


class MedallionSignalAnalogs:
    """
    Three Medallion-inspired micro-strategies adapted for crypto markets.

    Usage:
        analogs = MedallionSignalAnalogs(config, logger)
        signals = analogs.get_signals(
            product_id="BTC-USD",
            current_price=97000.0,
            price_history=[96000, 96500, 97000, ...],
            funding_rate=0.0003,
        )
        # Returns: {"sharp_move_reversion": -0.4, "hourly_seasonality": 0.02, "funding_timing": 0.3}
    """

    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Sharp Move Reversion parameters
        self._reversion_threshold = float(config.get("reversion_threshold", 0.03))  # 3% move
        self._reversion_lookback = int(config.get("reversion_lookback", 48))  # 48 cycles = 4 hours at 5min
        self._reversion_decay_cycles = int(config.get("reversion_decay_cycles", 12))

        # Track active reversion signals per product
        # {product_id: {"direction": float, "cycles_remaining": int, "magnitude": float}}
        self._active_reversions: Dict[str, Dict] = {}

        # Hourly seasonality — rolling return averages by hour
        # {product_id: {hour: deque of returns}}
        self._hourly_returns: Dict[str, Dict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=500))
        )
        self._last_prices: Dict[str, float] = {}

        # Funding settlement times (UTC hours)
        self._settlement_hours = [0, 8, 16]  # Standard crypto funding settlement
        self._settlement_window_minutes = 30  # Signal active 30 min before settlement

        self._cycle_count = 0
        self.logger.info(
            f"MedallionSignalAnalogs initialized: "
            f"reversion>{self._reversion_threshold:.0%}, "
            f"decay={self._reversion_decay_cycles} cycles"
        )

    def get_signals(
        self,
        product_id: str,
        current_price: float,
        price_history: Optional[list] = None,
        funding_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute all three Medallion analog signals.

        Returns dict with keys: sharp_move_reversion, hourly_seasonality, funding_timing
        All values in [-1, 1].
        """
        self._cycle_count += 1
        result = {
            "sharp_move_reversion": 0.0,
            "hourly_seasonality": 0.0,
            "funding_timing": 0.0,
        }

        try:
            # 1. Sharp Move Reversion
            result["sharp_move_reversion"] = self._sharp_move_reversion(
                product_id, current_price, price_history
            )

            # 2. Hourly Seasonality
            result["hourly_seasonality"] = self._hourly_seasonality(
                product_id, current_price
            )

            # 3. Funding Settlement Timing
            result["funding_timing"] = self._funding_settlement_timing(
                funding_rate
            )

        except Exception as e:
            self.logger.debug(f"MedallionSignalAnalogs error: {e}")

        return result

    def _sharp_move_reversion(
        self, product_id: str, current_price: float, price_history: Optional[list]
    ) -> float:
        """
        Signal 1: Mean reversion after sharp moves.
        If price moved >3% in the lookback window, expect reversion.
        Signal decays linearly over _reversion_decay_cycles.
        """
        # Check for new sharp moves
        if price_history and len(price_history) >= self._reversion_lookback:
            lookback_price = price_history[-self._reversion_lookback]
            if lookback_price > 0:
                move_pct = (current_price - lookback_price) / lookback_price
                if abs(move_pct) >= self._reversion_threshold:
                    # New sharp move detected — create or update reversion signal
                    direction = -np.sign(move_pct)  # Revert in opposite direction
                    magnitude = min(abs(move_pct) / self._reversion_threshold, 2.0)  # Scale by severity

                    self._active_reversions[product_id] = {
                        "direction": float(direction),
                        "cycles_remaining": self._reversion_decay_cycles,
                        "magnitude": float(magnitude),
                    }

                    if self._cycle_count <= 10 or self._cycle_count % 100 == 0:
                        self.logger.info(
                            f"SHARP MOVE: {product_id} moved {move_pct:+.2%} in "
                            f"{self._reversion_lookback} cycles — reversion signal active"
                        )

        # Decay existing reversion signals
        active = self._active_reversions.get(product_id)
        if active and active["cycles_remaining"] > 0:
            decay = active["cycles_remaining"] / self._reversion_decay_cycles
            signal = active["direction"] * active["magnitude"] * decay
            active["cycles_remaining"] -= 1
            return float(np.clip(signal, -1.0, 1.0))
        elif active and active["cycles_remaining"] <= 0:
            del self._active_reversions[product_id]

        return 0.0

    def _hourly_seasonality(self, product_id: str, current_price: float) -> float:
        """
        Signal 2: Hour-of-day return patterns.
        Maintains rolling average returns per hour, uses the bias as a weak signal.
        """
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        # Record return from last observation
        last_price = self._last_prices.get(product_id, 0.0)
        if last_price > 0 and current_price > 0:
            ret = (current_price - last_price) / last_price
            self._hourly_returns[product_id][current_hour].append(ret)
        self._last_prices[product_id] = current_price

        # Compute seasonal bias for current hour
        hourly_data = self._hourly_returns[product_id].get(current_hour)
        if not hourly_data or len(hourly_data) < 20:
            return 0.0

        avg_return = float(np.mean(list(hourly_data)))
        # Scale to [-1, 1] — a 0.1% average hourly bias maps to full signal
        signal = float(np.clip(avg_return * 1000, -1.0, 1.0))

        return signal

    def _funding_settlement_timing(self, funding_rate: float) -> float:
        """
        Signal 3: Pre-funding-settlement positioning.
        High positive funding = longs pay shorts = likely long unwinding before settlement.
        Signal is active only in the 30-minute window before settlement.
        """
        now = datetime.now(timezone.utc)
        minutes_to_next_settlement = float('inf')

        for settlement_hour in self._settlement_hours:
            settlement_time = now.replace(hour=settlement_hour, minute=0, second=0, microsecond=0)
            if settlement_time <= now:
                settlement_time += timedelta(days=1)
            delta = (settlement_time - now).total_seconds() / 60.0
            minutes_to_next_settlement = min(minutes_to_next_settlement, delta)

        # Only active in the window before settlement
        if minutes_to_next_settlement > self._settlement_window_minutes:
            return 0.0

        if abs(funding_rate) < 0.0001:
            return 0.0  # Funding rate too small to be meaningful

        # High positive funding = crowded longs = expect sell pressure before settlement
        # High negative funding = crowded shorts = expect buy pressure
        # Proximity amplifier: closer to settlement = stronger signal
        proximity = 1.0 - (minutes_to_next_settlement / self._settlement_window_minutes)
        signal = -np.sign(funding_rate) * min(abs(funding_rate) * 500, 1.0) * proximity

        return float(np.clip(signal, -1.0, 1.0))

    def get_stats(self) -> Dict[str, Any]:
        """Return current signal statistics."""
        return {
            "active_reversions": {k: v for k, v in self._active_reversions.items() if v.get("cycles_remaining", 0) > 0},
            "hourly_data_points": {
                pid: sum(len(d) for d in hours.values())
                for pid, hours in self._hourly_returns.items()
            },
            "cycle_count": self._cycle_count,
        }
