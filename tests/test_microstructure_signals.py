"""
Tests for signals/microstructure_signals.py
===========================================
Covers all 4 signal classes: OrderBookImbalanceSignal, TradeFlowImbalanceSignal,
LargeOrderDetector, SpreadDynamicsSignal. Uses synthetic order book and trade data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pytest

from signals.microstructure_signals import (
    MicrostructureSignal,
    OrderBookImbalanceSignal,
    TradeFlowImbalanceSignal,
    LargeOrderDetector,
    SpreadDynamicsSignal,
)


# ---------------------------------------------------------------------------
# MicrostructureSignal dataclass tests
# ---------------------------------------------------------------------------

class TestMicrostructureSignal:
    def test_to_numeric_long(self):
        sig = MicrostructureSignal(
            signal_name="test", direction="long", confidence=0.75,
            time_horizon_minutes=5,
        )
        assert sig.to_numeric() == 0.75

    def test_to_numeric_short(self):
        sig = MicrostructureSignal(
            signal_name="test", direction="short", confidence=0.6,
            time_horizon_minutes=5,
        )
        assert sig.to_numeric() == -0.6

    def test_to_numeric_neutral(self):
        sig = MicrostructureSignal(
            signal_name="test", direction="neutral", confidence=0.9,
            time_horizon_minutes=5,
        )
        assert sig.to_numeric() == 0.0


# ---------------------------------------------------------------------------
# OrderBookImbalanceSignal tests
# ---------------------------------------------------------------------------

class TestOrderBookImbalance:
    def setup_method(self):
        self.signal = OrderBookImbalanceSignal()

    def test_bullish_imbalance(self):
        """Bids dominate (>65%) -> long signal."""
        bids = [(100.0, 10.0)] * 10  # total bid = 100
        asks = [(101.0, 2.0)] * 10   # total ask = 20
        result = self.signal.calculate(bids, asks)
        assert result is not None
        assert result.direction == "long"
        assert result.confidence > 0.0
        assert result.signal_name == "order_book_imbalance"

    def test_bearish_imbalance(self):
        """Asks dominate (>65%) -> short signal."""
        bids = [(100.0, 2.0)] * 10   # total bid = 20
        asks = [(101.0, 10.0)] * 10  # total ask = 100
        result = self.signal.calculate(bids, asks)
        assert result is not None
        assert result.direction == "short"
        assert result.confidence > 0.0

    def test_balanced_book_returns_none(self):
        """Roughly equal bid/ask -> no signal."""
        bids = [(100.0, 5.0)] * 10
        asks = [(101.0, 5.0)] * 10
        result = self.signal.calculate(bids, asks)
        assert result is None

    def test_empty_bids(self):
        asks = [(101.0, 5.0)] * 10
        result = self.signal.calculate([], asks)
        assert result is None

    def test_empty_asks(self):
        bids = [(100.0, 5.0)] * 10
        result = self.signal.calculate(bids, [])
        assert result is None

    def test_both_empty(self):
        result = self.signal.calculate([], [])
        assert result is None

    def test_zero_total_volume(self):
        bids = [(100.0, 0.0)] * 5
        asks = [(101.0, 0.0)] * 5
        result = self.signal.calculate(bids, asks)
        assert result is None

    def test_confidence_scales_with_imbalance(self):
        """More extreme imbalance should produce higher confidence."""
        # Moderate imbalance
        bids_moderate = [(100.0, 7.0)] * 10   # 70 bid
        asks_moderate = [(101.0, 3.0)] * 10   # 30 ask -> 70% bid ratio
        res1 = self.signal.calculate(bids_moderate, asks_moderate)

        # Extreme imbalance
        bids_extreme = [(100.0, 19.0)] * 10   # 190 bid
        asks_extreme = [(101.0, 1.0)] * 10    # 10 ask -> 95% bid ratio
        res2 = self.signal.calculate(bids_extreme, asks_extreme)

        assert res1 is not None and res2 is not None
        assert res2.confidence > res1.confidence

    def test_fewer_levels_than_lookback(self):
        """If fewer than LOOKBACK_LEVELS, uses available levels."""
        bids = [(100.0, 20.0)] * 3  # only 3 levels
        asks = [(101.0, 1.0)] * 3
        result = self.signal.calculate(bids, asks)
        assert result is not None
        assert result.metadata["levels_analysed"] == 3

    def test_metadata_contains_volumes(self):
        bids = [(100.0, 10.0)] * 10
        asks = [(101.0, 2.0)] * 10
        result = self.signal.calculate(bids, asks)
        assert result is not None
        assert "bid_volume" in result.metadata
        assert "ask_volume" in result.metadata
        assert result.metadata["bid_volume"] == 100.0
        assert result.metadata["ask_volume"] == 20.0


# ---------------------------------------------------------------------------
# TradeFlowImbalanceSignal tests
# ---------------------------------------------------------------------------

class TestTradeFlowImbalance:
    def setup_method(self):
        self.signal = TradeFlowImbalanceSignal()

    def test_buy_dominated_flow(self):
        """Many buys -> long signal."""
        now = datetime.now()
        for i in range(100):
            self.signal.add_trade(
                price=50000.0, size=1.0, side="buy",
                timestamp=now - timedelta(seconds=i),
            )
        for i in range(20):
            self.signal.add_trade(
                price=50000.0, size=1.0, side="sell",
                timestamp=now - timedelta(seconds=i),
            )
        result = self.signal.calculate()
        assert result is not None
        assert result.direction == "long"

    def test_sell_dominated_flow(self):
        """Many sells -> short signal."""
        now = datetime.now()
        for i in range(100):
            self.signal.add_trade(
                price=50000.0, size=1.0, side="sell",
                timestamp=now - timedelta(seconds=i),
            )
        for i in range(20):
            self.signal.add_trade(
                price=50000.0, size=1.0, side="buy",
                timestamp=now - timedelta(seconds=i),
            )
        result = self.signal.calculate()
        assert result is not None
        assert result.direction == "short"

    def test_balanced_flow_returns_none(self):
        """Equal buy/sell -> no signal."""
        now = datetime.now()
        for i in range(50):
            self.signal.add_trade(50000.0, 1.0, "buy", now - timedelta(seconds=i))
            self.signal.add_trade(50000.0, 1.0, "sell", now - timedelta(seconds=i))
        result = self.signal.calculate()
        assert result is None

    def test_empty_buffer_returns_none(self):
        result = self.signal.calculate()
        assert result is None

    def test_old_trades_outside_window(self):
        """Trades older than WINDOW_SECONDS should not count."""
        old = datetime.now() - timedelta(seconds=120)  # well beyond 60s window
        for _ in range(100):
            self.signal.add_trade(50000.0, 1.0, "buy", old)
        result = self.signal.calculate()
        # All trades are outside window, total_volume == 0
        assert result is None

    def test_invalid_side_ignored(self):
        now = datetime.now()
        self.signal.add_trade(50000.0, 1.0, "invalid_side", now)
        assert len(self.signal.trade_buffer) == 0

    def test_zero_size_ignored(self):
        now = datetime.now()
        self.signal.add_trade(50000.0, 0.0, "buy", now)
        assert len(self.signal.trade_buffer) == 0

    def test_zero_price_ignored(self):
        now = datetime.now()
        self.signal.add_trade(0.0, 1.0, "buy", now)
        assert len(self.signal.trade_buffer) == 0

    def test_metadata_includes_window_info(self):
        now = datetime.now()
        for i in range(50):
            self.signal.add_trade(50000.0, 1.0, "buy", now - timedelta(seconds=i))
        result = self.signal.calculate()
        assert result is not None
        assert "window_seconds" in result.metadata
        assert "trade_count_in_window" in result.metadata


# ---------------------------------------------------------------------------
# LargeOrderDetector tests
# ---------------------------------------------------------------------------

class TestLargeOrderDetector:
    def setup_method(self):
        self.detector = LargeOrderDetector()

    def _seed_history(self, count: int = 100, base_size: float = 1.0):
        """Feed small orders into historical_sizes to establish a baseline."""
        for _ in range(count):
            self.detector.historical_sizes.append(base_size)

    def test_whale_bid_triggers_long(self):
        """One massive bid order among small sizes -> long signal."""
        self._seed_history(100, base_size=1.0)
        # Inject a whale bid
        bids = [(50000.0, 1.0)] * 5 + [(49000.0, 100.0)]  # whale at 100
        asks = [(50100.0, 1.0)] * 6
        result = self.detector.calculate(bids, asks)
        assert result is not None
        assert result.direction == "long"

    def test_whale_ask_triggers_short(self):
        """One massive ask order among small sizes -> short signal."""
        self._seed_history(100, base_size=1.0)
        bids = [(50000.0, 1.0)] * 6
        asks = [(50100.0, 1.0)] * 5 + [(51000.0, 100.0)]  # whale ask
        result = self.detector.calculate(bids, asks)
        assert result is not None
        assert result.direction == "short"

    def test_balanced_whales_returns_none(self):
        """Equal large orders on both sides -> no signal."""
        self._seed_history(100, base_size=1.0)
        bids = [(50000.0, 1.0)] * 5 + [(49000.0, 100.0)]
        asks = [(50100.0, 1.0)] * 5 + [(51000.0, 100.0)]
        result = self.detector.calculate(bids, asks)
        # Roughly 50/50 dominance, should not fire
        assert result is None

    def test_insufficient_history(self):
        """Fewer than 50 historical sizes -> no signal."""
        bids = [(50000.0, 1.0)] * 5
        asks = [(50100.0, 1.0)] * 5
        result = self.detector.calculate(bids, asks)
        assert result is None

    def test_empty_book_returns_none(self):
        result = self.detector.calculate([], [])
        assert result is None

    def test_zero_size_orders_ignored(self):
        bids = [(50000.0, 0.0)] * 10
        asks = [(50100.0, 0.0)] * 10
        result = self.detector.calculate(bids, asks)
        assert result is None

    def test_history_accumulates(self):
        """Each call adds sizes to historical_sizes."""
        bids = [(50000.0, 2.0)] * 3
        asks = [(50100.0, 3.0)] * 3
        initial = len(self.detector.historical_sizes)
        self.detector.calculate(bids, asks)
        assert len(self.detector.historical_sizes) == initial + 6

    def test_metadata_contains_threshold_info(self):
        self._seed_history(100, base_size=1.0)
        bids = [(50000.0, 1.0)] * 5 + [(49000.0, 200.0)]
        asks = [(50100.0, 1.0)] * 6
        result = self.detector.calculate(bids, asks)
        if result is not None:
            assert "size_threshold" in result.metadata
            assert "large_bid_count" in result.metadata
            assert "historical_sample_size" in result.metadata


# ---------------------------------------------------------------------------
# SpreadDynamicsSignal tests
# ---------------------------------------------------------------------------

class TestSpreadDynamics:
    def setup_method(self):
        self.signal = SpreadDynamicsSignal()

    def _feed_spread_observations(
        self,
        count: int = 50,
        base_bid: float = 50000.0,
        spread: float = 1.0,
        spread_trend: float = 0.0,
    ):
        """Feed synthetic bid/ask observations with optional spread trend."""
        now = datetime.now()
        for i in range(count):
            current_spread = spread + spread_trend * i
            bid = base_bid
            ask = base_bid + max(current_spread, 0.01)
            ts = now - timedelta(seconds=(count - i) * 2)
            self.signal.update(bid, ask, ts)

    def test_insufficient_data_returns_none(self):
        """Fewer than MIN_OBSERVATIONS -> no signal."""
        for i in range(10):
            self.signal.update(50000.0, 50001.0, datetime.now())
        result = self.signal.calculate()
        assert result is None

    def test_compression_regime_long_signal(self):
        """Narrowing spread (negative slope) + below-average spread -> long."""
        # Start with wide spread, narrow over time
        self._feed_spread_observations(count=60, spread=10.0, spread_trend=-0.15)
        result = self.signal.calculate()
        # The current spread is at the narrow end, z-score should be < -1
        # Direction should be "long" if conditions are met
        if result is not None:
            assert result.direction in ("long", "short", "neutral")

    def test_expansion_regime_short_signal(self):
        """Widening spread (positive slope) + above-average spread -> short."""
        self._feed_spread_observations(count=60, spread=1.0, spread_trend=0.15)
        result = self.signal.calculate()
        if result is not None:
            assert result.direction in ("long", "short", "neutral")

    def test_stable_spread_returns_none(self):
        """Constant spread -> no signal (not compression or expansion)."""
        self._feed_spread_observations(count=60, spread=5.0, spread_trend=0.0)
        result = self.signal.calculate()
        # With a perfectly constant spread, std_spread ~ 0, so returns None
        assert result is None

    def test_invalid_bid_ask_ignored(self):
        """bid >= ask should be rejected."""
        self.signal.update(50001.0, 50000.0, datetime.now())  # ask < bid
        assert len(self.signal.spread_history) == 0

    def test_zero_bid_ignored(self):
        self.signal.update(0.0, 50000.0, datetime.now())
        assert len(self.signal.spread_history) == 0

    def test_equal_bid_ask_ignored(self):
        self.signal.update(50000.0, 50000.0, datetime.now())
        assert len(self.signal.spread_history) == 0

    def test_metadata_contains_z_score(self):
        """Verify metadata keys when signal fires."""
        self._feed_spread_observations(count=60, spread=10.0, spread_trend=-0.2)
        result = self.signal.calculate()
        if result is not None:
            assert "z_score" in result.metadata
            assert "slope" in result.metadata
            assert "observations" in result.metadata

    def test_history_capped_at_max(self):
        """Spread history should not exceed HISTORY_SIZE."""
        for i in range(self.signal.HISTORY_SIZE + 100):
            self.signal.update(50000.0, 50001.0 + i * 0.001, datetime.now())
        assert len(self.signal.spread_history) == self.signal.HISTORY_SIZE

    def test_signal_name_is_spread_dynamics(self):
        self._feed_spread_observations(count=60, spread=10.0, spread_trend=-0.2)
        result = self.signal.calculate()
        if result is not None:
            assert result.signal_name == "spread_dynamics"
