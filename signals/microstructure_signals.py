"""
Advanced Microstructure Signals
===============================
Specialized signal classes for order book microstructure analysis.

These signals complement the existing MicrostructureEngine by providing
granular, independently configurable detectors that produce standardized
MicrostructureSignal output suitable for aggregation and fusion.

Dependencies: stdlib + numpy only.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standardized signal container
# ---------------------------------------------------------------------------

@dataclass
class MicrostructureSignal:
    """Standardized output produced by every microstructure signal class.

    Attributes:
        signal_name: Identifier for the originating signal class.
        direction: One of ``"long"``, ``"short"``, or ``"neutral"``.
        confidence: Confidence level in ``[0, 1]``.
        time_horizon_minutes: Expected validity window in minutes.
        metadata: Arbitrary extra data specific to the signal.
        timestamp: When the signal was generated.
    """
    signal_name: str
    direction: str  # "long", "short", or "neutral"
    confidence: float  # 0-1
    time_horizon_minutes: int
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_numeric(self) -> float:
        """Convert to a signed numeric value in ``[-1, 1]``.

        ``+confidence`` for long, ``-confidence`` for short, ``0`` for neutral.
        """
        if self.direction == "long":
            return self.confidence
        elif self.direction == "short":
            return -self.confidence
        return 0.0


# ---------------------------------------------------------------------------
# 1. Order Book Imbalance Signal
# ---------------------------------------------------------------------------

class OrderBookImbalanceSignal:
    """Measures bid/ask volume ratio at the top *N* price levels.

    When the bid side dominates (bids >> asks), the order book is
    interpreted as bullish.  When the ask side dominates, bearish.

    The signal fires only when one side controls more than
    ``IMBALANCE_THRESHOLD`` (default 65 %) of the total volume across
    the observed levels.
    """

    LOOKBACK_LEVELS: int = 10
    IMBALANCE_THRESHOLD: float = 0.65  # 65 % on one side triggers signal
    SIGNAL_HORIZON_MINUTES: int = 5

    def calculate(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> Optional[MicrostructureSignal]:
        """Evaluate order book imbalance.

        Args:
            bids: Descending list of ``(price, size)`` tuples (best bid first).
            asks: Ascending list of ``(price, size)`` tuples (best ask first).

        Returns:
            A :class:`MicrostructureSignal` if the imbalance exceeds the
            threshold, or ``None`` otherwise.
        """
        try:
            if not bids or not asks:
                return None

            levels = min(self.LOOKBACK_LEVELS, len(bids), len(asks))
            if levels == 0:
                return None

            bid_volume = sum(size for _, size in bids[:levels])
            ask_volume = sum(size for _, size in asks[:levels])
            total_volume = bid_volume + ask_volume

            if total_volume <= 0.0:
                return None

            bid_ratio = bid_volume / total_volume
            ask_ratio = ask_volume / total_volume

            # Determine whether the imbalance is strong enough
            if bid_ratio >= self.IMBALANCE_THRESHOLD:
                direction = "long"
                # Scale confidence: at threshold it's low; at 100 % it's 1.0
                raw_confidence = (bid_ratio - self.IMBALANCE_THRESHOLD) / (
                    1.0 - self.IMBALANCE_THRESHOLD
                )
                confidence = float(np.clip(raw_confidence, 0.0, 1.0))
            elif ask_ratio >= self.IMBALANCE_THRESHOLD:
                direction = "short"
                raw_confidence = (ask_ratio - self.IMBALANCE_THRESHOLD) / (
                    1.0 - self.IMBALANCE_THRESHOLD
                )
                confidence = float(np.clip(raw_confidence, 0.0, 1.0))
            else:
                return None  # Below threshold -- no signal

            return MicrostructureSignal(
                signal_name="order_book_imbalance",
                direction=direction,
                confidence=confidence,
                time_horizon_minutes=self.SIGNAL_HORIZON_MINUTES,
                metadata={
                    "bid_volume": bid_volume,
                    "ask_volume": ask_volume,
                    "bid_ratio": round(bid_ratio, 4),
                    "ask_ratio": round(ask_ratio, 4),
                    "levels_analysed": levels,
                },
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error("OrderBookImbalanceSignal.calculate failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# 2. Trade Flow Imbalance Signal
# ---------------------------------------------------------------------------

@dataclass
class _TradeRecord:
    """Internal record kept in the trade buffer."""
    price: float
    size: float
    side: str  # "buy" or "sell"
    timestamp: datetime


class TradeFlowImbalanceSignal:
    """Net buying vs. selling pressure over a rolling window.

    Tracks aggressive buyers (taker buys) vs. aggressive sellers
    (taker sells) and fires when the ratio exceeds ``THRESHOLD_RATIO``.
    """

    WINDOW_SECONDS: int = 60
    THRESHOLD_RATIO: float = 0.60
    SIGNAL_HORIZON_MINUTES: int = 3

    def __init__(self) -> None:
        self.trade_buffer: deque[_TradeRecord] = deque(maxlen=10_000)

    # -- data ingestion -----------------------------------------------------

    def add_trade(
        self,
        price: float,
        size: float,
        side: str,
        timestamp: datetime,
    ) -> None:
        """Register a trade in the internal buffer.

        Args:
            price: Execution price.
            size: Trade size.
            side: ``"buy"`` or ``"sell"``.
            timestamp: Time of the trade.
        """
        if side not in ("buy", "sell"):
            logger.warning(
                "TradeFlowImbalanceSignal.add_trade: ignoring unknown side %r",
                side,
            )
            return
        if size <= 0.0 or price <= 0.0:
            return
        self.trade_buffer.append(
            _TradeRecord(price=price, size=size, side=side, timestamp=timestamp)
        )

    # -- signal calculation -------------------------------------------------

    def calculate(self) -> Optional[MicrostructureSignal]:
        """Evaluate trade flow imbalance over the rolling window.

        Returns:
            A :class:`MicrostructureSignal` when the buy or sell ratio
            exceeds the threshold, or ``None`` otherwise.
        """
        try:
            if len(self.trade_buffer) == 0:
                return None

            cutoff = datetime.now() - timedelta(seconds=self.WINDOW_SECONDS)
            buy_volume = 0.0
            sell_volume = 0.0

            for trade in self.trade_buffer:
                if trade.timestamp < cutoff:
                    continue
                if trade.side == "buy":
                    buy_volume += trade.size
                else:
                    sell_volume += trade.size

            total_volume = buy_volume + sell_volume
            if total_volume <= 0.0:
                return None

            buy_ratio = buy_volume / total_volume
            sell_ratio = sell_volume / total_volume

            if buy_ratio >= self.THRESHOLD_RATIO:
                direction = "long"
                raw_confidence = (buy_ratio - self.THRESHOLD_RATIO) / (
                    1.0 - self.THRESHOLD_RATIO
                )
                confidence = float(np.clip(raw_confidence, 0.0, 1.0))
            elif sell_ratio >= self.THRESHOLD_RATIO:
                direction = "short"
                raw_confidence = (sell_ratio - self.THRESHOLD_RATIO) / (
                    1.0 - self.THRESHOLD_RATIO
                )
                confidence = float(np.clip(raw_confidence, 0.0, 1.0))
            else:
                return None

            return MicrostructureSignal(
                signal_name="trade_flow_imbalance",
                direction=direction,
                confidence=confidence,
                time_horizon_minutes=self.SIGNAL_HORIZON_MINUTES,
                metadata={
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "buy_ratio": round(buy_ratio, 4),
                    "sell_ratio": round(sell_ratio, 4),
                    "window_seconds": self.WINDOW_SECONDS,
                    "trade_count_in_window": sum(
                        1 for t in self.trade_buffer if t.timestamp >= cutoff
                    ),
                },
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error("TradeFlowImbalanceSignal.calculate failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# 3. Large Order Detector
# ---------------------------------------------------------------------------

class LargeOrderDetector:
    """Detects unusually large orders in the book ("whale spotting").

    A large bid appearing is treated as an accumulation signal.
    A large ask appearing is treated as a distribution signal.

    "Large" is defined as any order whose size exceeds the
    ``SIZE_PERCENTILE_THRESHOLD`` of the running history of order sizes.
    """

    SIZE_PERCENTILE_THRESHOLD: int = 95  # top 5 % by size
    SIGNAL_HORIZON_MINUTES: int = 10

    def __init__(self) -> None:
        self.historical_sizes: deque[float] = deque(maxlen=5_000)

    def calculate(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> Optional[MicrostructureSignal]:
        """Scan for whale-size orders on either side.

        Args:
            bids: ``(price, size)`` descending from best bid.
            asks: ``(price, size)`` ascending from best ask.

        Returns:
            A :class:`MicrostructureSignal` if whale orders are detected
            on one side only (or disproportionately), otherwise ``None``.
        """
        try:
            if not bids and not asks:
                return None

            # Feed current order sizes into the historical buffer
            all_sizes: List[float] = []
            for _, size in bids:
                if size > 0.0:
                    all_sizes.append(size)
            for _, size in asks:
                if size > 0.0:
                    all_sizes.append(size)

            if not all_sizes:
                return None

            self.historical_sizes.extend(all_sizes)

            # Need enough history to compute a meaningful percentile
            if len(self.historical_sizes) < 50:
                return None

            size_threshold = float(
                np.percentile(
                    list(self.historical_sizes), self.SIZE_PERCENTILE_THRESHOLD
                )
            )
            if size_threshold <= 0.0:
                return None

            # Count large-order volume on each side
            large_bid_volume = 0.0
            large_bid_count = 0
            for _, size in bids:
                if size >= size_threshold:
                    large_bid_volume += size
                    large_bid_count += 1

            large_ask_volume = 0.0
            large_ask_count = 0
            for _, size in asks:
                if size >= size_threshold:
                    large_ask_volume += size
                    large_ask_count += 1

            total_large = large_bid_volume + large_ask_volume
            if total_large <= 0.0:
                return None

            bid_dominance = large_bid_volume / total_large
            ask_dominance = large_ask_volume / total_large

            # Only fire when large orders lean meaningfully to one side
            DOMINANCE_THRESHOLD = 0.65

            if bid_dominance >= DOMINANCE_THRESHOLD:
                direction = "long"
                raw_confidence = (bid_dominance - DOMINANCE_THRESHOLD) / (
                    1.0 - DOMINANCE_THRESHOLD
                )
                confidence = float(np.clip(raw_confidence, 0.0, 1.0))
            elif ask_dominance >= DOMINANCE_THRESHOLD:
                direction = "short"
                raw_confidence = (ask_dominance - DOMINANCE_THRESHOLD) / (
                    1.0 - DOMINANCE_THRESHOLD
                )
                confidence = float(np.clip(raw_confidence, 0.0, 1.0))
            else:
                return None

            return MicrostructureSignal(
                signal_name="large_order_detector",
                direction=direction,
                confidence=confidence,
                time_horizon_minutes=self.SIGNAL_HORIZON_MINUTES,
                metadata={
                    "size_threshold": round(size_threshold, 6),
                    "large_bid_volume": large_bid_volume,
                    "large_ask_volume": large_ask_volume,
                    "large_bid_count": large_bid_count,
                    "large_ask_count": large_ask_count,
                    "bid_dominance": round(bid_dominance, 4),
                    "ask_dominance": round(ask_dominance, 4),
                    "historical_sample_size": len(self.historical_sizes),
                },
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error("LargeOrderDetector.calculate failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# 4. Spread Dynamics Signal
# ---------------------------------------------------------------------------

@dataclass
class _SpreadObservation:
    """Internal record for the spread history ring buffer."""
    spread: float
    mid_price: float
    timestamp: datetime


class SpreadDynamicsSignal:
    """Analyses spread behaviour for predictive signals.

    * **Widening spread** -- uncertainty, potential for large move.
    * **Narrowing spread** -- conviction building, directional move likely.
    * **Sudden spike after calm** -- volatility incoming.

    The signal uses a z-score of the current spread relative to recent
    history and detects compression/expansion regimes.
    """

    HISTORY_SIZE: int = 500
    SIGNAL_HORIZON_MINUTES: int = 5
    # Minimum observations before producing a signal
    MIN_OBSERVATIONS: int = 30

    def __init__(self) -> None:
        self.spread_history: deque[_SpreadObservation] = deque(
            maxlen=self.HISTORY_SIZE
        )

    # -- data ingestion -----------------------------------------------------

    def update(self, bid: float, ask: float, timestamp: datetime) -> None:
        """Record a spread observation.

        Args:
            bid: Best bid price.
            ask: Best ask price.
            timestamp: Observation time.
        """
        if ask <= bid or bid <= 0.0:
            return
        spread = ask - bid
        mid_price = (bid + ask) / 2.0
        self.spread_history.append(
            _SpreadObservation(spread=spread, mid_price=mid_price, timestamp=timestamp)
        )

    # -- signal calculation -------------------------------------------------

    def calculate(self) -> Optional[MicrostructureSignal]:
        """Evaluate spread dynamics.

        Returns:
            A :class:`MicrostructureSignal` characterising the spread
            regime (compression = bullish, expansion = uncertain),
            or ``None`` if not enough data.
        """
        try:
            if len(self.spread_history) < self.MIN_OBSERVATIONS:
                return None

            # Build numpy arrays for analysis
            spreads = np.array(
                [obs.spread for obs in self.spread_history], dtype=np.float64
            )
            mid_prices = np.array(
                [obs.mid_price for obs in self.spread_history], dtype=np.float64
            )

            # Normalise spread to basis points (spread / mid) for comparability
            with np.errstate(divide="ignore", invalid="ignore"):
                spread_bps = np.where(
                    mid_prices > 0.0, spreads / mid_prices, 0.0
                )

            if len(spread_bps) == 0 or np.all(spread_bps == 0.0):
                return None

            # --- z-score of current spread relative to recent window -------
            current_spread_bps = float(spread_bps[-1])
            mean_spread = float(np.mean(spread_bps))
            std_spread = float(np.std(spread_bps))

            if std_spread <= 0.0:
                return None

            z_score = (current_spread_bps - mean_spread) / std_spread

            # --- spread trend (linear regression slope over last N obs) ----
            window = min(50, len(spread_bps))
            recent = spread_bps[-window:]
            x = np.arange(window, dtype=np.float64)
            if window >= 2:
                # Simple least-squares slope
                x_mean = x.mean()
                y_mean = recent.mean()
                slope = float(
                    np.sum((x - x_mean) * (recent - y_mean))
                    / (np.sum((x - x_mean) ** 2) + 1e-12)
                )
            else:
                slope = 0.0

            # --- volatility of the spread itself ---------------------------
            spread_volatility = float(np.std(np.diff(spread_bps[-window:]))) if window >= 3 else 0.0

            # --- decision logic -------------------------------------------
            # Compression regime: spread is narrowing (negative slope)
            #   AND current spread is below average (negative z-score).
            #   Interpretation: conviction building, directional move ahead.
            # Expansion regime: spread is widening AND z-score is high.
            #   Interpretation: uncertainty / volatility incoming.
            # Spike detection: z-score > 2 after a calm period (low spread_volatility).

            direction: str
            confidence: float

            if z_score < -1.0 and slope < 0.0:
                # Spread compression -- typically precedes directional move (bullish lean)
                direction = "long"
                raw_confidence = min(abs(z_score) / 3.0, 1.0) * min(abs(slope) * 1e4, 1.0)
                confidence = float(np.clip(raw_confidence, 0.0, 1.0))
            elif z_score > 2.0 and spread_volatility < std_spread * 0.5:
                # Sudden spike after calm -- volatility burst
                direction = "neutral"
                confidence = float(np.clip(z_score / 4.0, 0.0, 1.0))
            elif z_score > 1.0 and slope > 0.0:
                # Spread expansion -- uncertainty, slight bearish lean
                direction = "short"
                raw_confidence = min(z_score / 3.0, 1.0) * min(slope * 1e4, 1.0)
                confidence = float(np.clip(raw_confidence, 0.0, 1.0))
            else:
                return None

            return MicrostructureSignal(
                signal_name="spread_dynamics",
                direction=direction,
                confidence=confidence,
                time_horizon_minutes=self.SIGNAL_HORIZON_MINUTES,
                metadata={
                    "z_score": round(z_score, 4),
                    "slope": round(slope, 8),
                    "spread_volatility": round(spread_volatility, 8),
                    "current_spread_bps": round(current_spread_bps, 8),
                    "mean_spread_bps": round(mean_spread, 8),
                    "observations": len(self.spread_history),
                },
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error("SpreadDynamicsSignal.calculate failed: %s", e)
            return None
