"""
Fast Mean Reversion Scanner
============================
Detects sub-bar price dislocations (0.05%-0.15%) that revert in seconds.

Runs as a background task evaluating all tracked pairs every 1 second.
The main bot feeds price updates via ``on_price_update()``; the scanner
maintains per-pair rolling statistics (VWAP, stddev) and caches the latest
signal for injection into ``generate_signals()``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _PairStats:
    """Rolling per-pair state for VWAP / stddev calculation."""

    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    volumes: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

    @property
    def sample_count(self) -> int:
        return len(self.prices)

    @property
    def rolling_vwap(self) -> float:
        if not self.prices or not self.volumes:
            return 0.0
        total_pv = sum(p * v for p, v in zip(self.prices, self.volumes))
        total_v = sum(self.volumes)
        if total_v <= 0:
            return sum(self.prices) / len(self.prices)
        return total_pv / total_v

    @property
    def rolling_stdev(self) -> float:
        if len(self.prices) < 2:
            return 0.0
        mean = sum(self.prices) / len(self.prices)
        variance = sum((p - mean) ** 2 for p in self.prices) / (len(self.prices) - 1)
        return math.sqrt(variance) if variance > 0 else 0.0


@dataclass
class FastReversionSignal:
    """Output signal from the fast mean reversion scanner."""

    pair: str
    direction: str  # "long" or "short"
    dislocation_pct: float
    mean_price: float
    current_price: float
    expected_reversion_bps: float
    confidence: float
    ttl_seconds: float
    timestamp: float


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class FastMeanReversionScanner:
    """Detects fast mean-reversion opportunities from real-time price feeds.

    Parameters
    ----------
    config : dict
        Configuration from ``config.fast_mean_reversion``.
    bar_aggregator : optional
        Reference to BarAggregator (unused currently, reserved for future).
    """

    def __init__(self, config: Dict[str, Any], bar_aggregator: Any = None) -> None:
        self._enabled = config.get("enabled", False)
        self._eval_interval = float(config.get("evaluation_interval_seconds", 1.0))
        self._window_seconds = int(config.get("rolling_window_seconds", 60))
        self._sigma_threshold = float(config.get("deviation_threshold_sigma", 2.0))
        self._min_bps = float(config.get("min_deviation_bps", 5.0))
        self._max_ttl = float(config.get("max_signal_ttl_seconds", 120))
        self._min_confidence = float(config.get("min_confidence", 0.52))
        self._min_samples = int(config.get("min_samples", 30))
        self._pairs = list(config.get("pairs", ["BTC-USD", "ETH-USD"]))

        self._bar_aggregator = bar_aggregator

        # Per-pair rolling stats
        self._stats: Dict[str, _PairStats] = {}

        # Cached latest signal per pair
        self._latest_signals: Dict[str, FastReversionSignal] = {}

        self._running = False

        logger.info(
            "FastMeanReversionScanner init  pairs=%s  window=%ds  "
            "sigma=%.1f  min_bps=%.1f  enabled=%s",
            self._pairs,
            self._window_seconds,
            self._sigma_threshold,
            self._min_bps,
            self._enabled,
        )

    # ----- price feed ---------------------------------------------------------

    def on_price_update(
        self, pair: str, price: float, volume: float, timestamp: float
    ) -> None:
        """Called by the main bot on every ticker update."""
        if not self._enabled or price <= 0:
            return

        stats = self._stats.get(pair)
        if stats is None:
            maxlen = self._window_seconds * 2
            stats = _PairStats(
                prices=deque(maxlen=maxlen),
                volumes=deque(maxlen=maxlen),
                timestamps=deque(maxlen=maxlen),
            )
            self._stats[pair] = stats

        # Trim entries older than the rolling window
        cutoff = timestamp - self._window_seconds
        while stats.timestamps and stats.timestamps[0] < cutoff:
            stats.timestamps.popleft()
            stats.prices.popleft()
            stats.volumes.popleft()

        stats.prices.append(price)
        stats.volumes.append(max(volume, 0.0))
        stats.timestamps.append(timestamp)

    # ----- evaluation ---------------------------------------------------------

    def evaluate(self, pair: str) -> Optional[FastReversionSignal]:
        """Pure function: check if *pair* has a mean-reversion dislocation."""
        stats = self._stats.get(pair)
        if stats is None or stats.sample_count < self._min_samples:
            return None

        vwap = stats.rolling_vwap
        stdev = stats.rolling_stdev
        if vwap <= 0 or stdev <= 0:
            return None

        current_price = stats.prices[-1]
        deviation_sigma = (current_price - vwap) / stdev
        deviation_bps = abs(current_price - vwap) / vwap * 10_000.0

        # Must exceed both sigma and bps thresholds
        if abs(deviation_sigma) < self._sigma_threshold:
            return None
        if deviation_bps < self._min_bps:
            return None

        # Direction: price below VWAP -> long (buy the dip), above -> short
        direction = "short" if deviation_sigma > 0 else "long"

        # Confidence scales with sigma, capped at 0.85
        confidence = min(0.50 + abs(deviation_sigma) * 0.07, 0.85)
        if confidence < self._min_confidence:
            return None

        expected_reversion_bps = deviation_bps * confidence

        return FastReversionSignal(
            pair=pair,
            direction=direction,
            dislocation_pct=round(deviation_bps / 100.0, 4),
            mean_price=round(vwap, 2),
            current_price=round(current_price, 2),
            expected_reversion_bps=round(expected_reversion_bps, 2),
            confidence=round(confidence, 4),
            ttl_seconds=self._max_ttl,
            timestamp=time.time(),
        )

    def get_latest_signal(self, pair: str) -> Optional[FastReversionSignal]:
        """Return cached latest signal for signal injection, or None if expired."""
        sig = self._latest_signals.get(pair)
        if sig is None:
            return None
        if time.time() - sig.timestamp > sig.ttl_seconds:
            self._latest_signals.pop(pair, None)
            return None
        return sig

    # ----- background loop ----------------------------------------------------

    async def run_loop(self) -> None:
        """Async background loop: evaluate all tracked pairs every interval."""
        self._running = True
        logger.info(
            "FastMeanReversionScanner run_loop started (%.1fs interval)",
            self._eval_interval,
        )

        while self._running:
            try:
                for pair in list(self._stats.keys()):
                    signal = self.evaluate(pair)
                    if signal is not None:
                        self._latest_signals[pair] = signal
                    # Don't clear old signal -- let TTL handle expiry

                await asyncio.sleep(self._eval_interval)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("FastMeanReversionScanner eval cycle failed")
                await asyncio.sleep(self._eval_interval)

        self._running = False
        logger.info("FastMeanReversionScanner run_loop stopped")

    def stop(self) -> None:
        """Signal the run loop to stop."""
        self._running = False
