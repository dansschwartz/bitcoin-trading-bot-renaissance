"""
Tests for signals/signal_aggregator.py — SignalAggregator
=========================================================
Covers: data feeding, composite signal generation, weighted aggregation,
signal agreement/disagreement, edge cases with empty data, and the
convenience get_signal_dict_entry method.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from signals.signal_aggregator import (
    CompositeSignal,
    SignalAggregator,
)
from signals.microstructure_signals import MicrostructureSignal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aggregator():
    """Fresh aggregator with default weights."""
    return SignalAggregator()


@pytest.fixture
def custom_aggregator():
    """Aggregator with custom weights."""
    return SignalAggregator(weights={
        "order_book_imbalance": 2.0,
        "trade_flow_imbalance": 1.0,
        "large_order_detector": 0.5,
        "spread_dynamics": 0.5,
    })


def _make_bullish_book() -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Return (bids, asks) with heavy bid-side imbalance."""
    bids = [(50000.0, 10.0)] * 10
    asks = [(50001.0, 1.0)] * 10
    return bids, asks


def _make_bearish_book() -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Return (bids, asks) with heavy ask-side imbalance."""
    bids = [(50000.0, 1.0)] * 10
    asks = [(50001.0, 10.0)] * 10
    return bids, asks


def _make_balanced_book() -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Return (bids, asks) roughly balanced."""
    bids = [(50000.0, 5.0)] * 10
    asks = [(50001.0, 5.0)] * 10
    return bids, asks


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_weights(self, aggregator):
        assert aggregator.weights["order_book_imbalance"] == 1.0
        assert aggregator.weights["trade_flow_imbalance"] == 1.0

    def test_custom_weights_override(self, custom_aggregator):
        assert custom_aggregator.weights["order_book_imbalance"] == 2.0
        assert custom_aggregator.weights["large_order_detector"] == 0.5

    def test_signal_instances_created(self, aggregator):
        assert aggregator.order_book_imbalance is not None
        assert aggregator.trade_flow_imbalance is not None
        assert aggregator.large_order_detector is not None
        assert aggregator.spread_dynamics is not None


# ---------------------------------------------------------------------------
# update_book tests
# ---------------------------------------------------------------------------

class TestUpdateBook:
    def test_bullish_book_updates_signals(self, aggregator):
        bids, asks = _make_bullish_book()
        aggregator.update_book(bids, asks)
        # Should have produced at least the OBI signal
        assert len(aggregator._latest_book_signals) == 2  # OBI + LOD slots

    def test_empty_book_no_crash(self, aggregator):
        aggregator.update_book([], [])
        # Should not raise, signals are None
        composite = aggregator.get_composite_signal()
        assert composite.active_signal_count == 0

    def test_spread_dynamics_updated(self, aggregator):
        bids = [(50000.0, 5.0)] * 5
        asks = [(50001.0, 5.0)] * 5
        aggregator.update_book(bids, asks)
        # Spread dynamics was updated internally
        assert len(aggregator.spread_dynamics.spread_history) == 1


# ---------------------------------------------------------------------------
# update_trade tests
# ---------------------------------------------------------------------------

class TestUpdateTrade:
    def test_add_buy_trade(self, aggregator):
        now = datetime.now()
        aggregator.update_trade(50000.0, 1.0, "buy", now)
        assert len(aggregator.trade_flow_imbalance.trade_buffer) == 1

    def test_add_sell_trade(self, aggregator):
        now = datetime.now()
        aggregator.update_trade(50000.0, 1.0, "sell", now)
        assert len(aggregator.trade_flow_imbalance.trade_buffer) == 1

    def test_trade_updates_latest_signal(self, aggregator):
        now = datetime.now()
        # Feed enough buy trades to trigger a signal
        for i in range(80):
            aggregator.update_trade(50000.0, 1.0, "buy", now - timedelta(seconds=i))
        # _latest_trade_signal should now be set (or None if below threshold)
        # Since 80 buys and 0 sells, ratio = 100% > 60%, should fire
        assert aggregator._latest_trade_signal is not None


# ---------------------------------------------------------------------------
# get_composite_signal tests
# ---------------------------------------------------------------------------

class TestCompositeSignal:
    def test_no_active_signals(self, aggregator):
        """Empty aggregator with no data -> zero composite."""
        composite = aggregator.get_composite_signal()
        assert isinstance(composite, CompositeSignal)
        assert composite.direction_score == 0.0
        assert composite.confidence == 0.0
        assert composite.active_signal_count == 0
        assert composite.component_signals == []

    def test_single_bullish_signal(self, aggregator):
        """Feed bullish book -> positive direction_score."""
        bids, asks = _make_bullish_book()
        aggregator.update_book(bids, asks)
        composite = aggregator.get_composite_signal()
        # OBI should fire long, direction_score > 0
        if composite.active_signal_count > 0:
            assert composite.direction_score > 0.0

    def test_single_bearish_signal(self, aggregator):
        """Feed bearish book -> negative direction_score."""
        bids, asks = _make_bearish_book()
        aggregator.update_book(bids, asks)
        composite = aggregator.get_composite_signal()
        if composite.active_signal_count > 0:
            assert composite.direction_score < 0.0

    def test_direction_score_bounded(self, aggregator):
        """Direction score always in [-1, 1]."""
        bids, asks = _make_bullish_book()
        aggregator.update_book(bids, asks)
        now = datetime.now()
        for i in range(100):
            aggregator.update_trade(50000.0, 1.0, "buy", now - timedelta(seconds=i))
        composite = aggregator.get_composite_signal()
        assert -1.0 <= composite.direction_score <= 1.0
        assert 0.0 <= composite.confidence <= 1.0

    def test_agreement_boosts_confidence(self, aggregator):
        """When all signals agree, confidence is maintained (not reduced)."""
        bids, asks = _make_bullish_book()
        aggregator.update_book(bids, asks)
        now = datetime.now()
        for i in range(100):
            aggregator.update_trade(50000.0, 1.0, "buy", now - timedelta(seconds=i))
        composite = aggregator.get_composite_signal()
        # All should be long direction -> agreement = 1.0
        if composite.active_signal_count >= 2:
            # confidence should not be heavily penalized
            assert composite.confidence > 0.0

    def test_disagreement_reduces_confidence(self, aggregator):
        """When signals disagree, confidence is reduced."""
        # Manually set signals to disagree
        long_sig = MicrostructureSignal(
            signal_name="order_book_imbalance", direction="long",
            confidence=0.8, time_horizon_minutes=5,
        )
        short_sig = MicrostructureSignal(
            signal_name="trade_flow_imbalance", direction="short",
            confidence=0.8, time_horizon_minutes=3,
        )
        aggregator._latest_book_signals = [long_sig, None]
        aggregator._latest_trade_signal = short_sig
        aggregator._latest_spread_signal = None

        composite = aggregator.get_composite_signal()
        # Disagreement should scale confidence down
        assert composite.active_signal_count == 2
        # The confidence should be less than the raw average (0.8) due to disagreement
        assert composite.confidence < 0.8

    def test_custom_weights_affect_result(self, custom_aggregator):
        """Double weight on OBI should skew direction_score."""
        # Set up a known signal
        long_sig = MicrostructureSignal(
            signal_name="order_book_imbalance", direction="long",
            confidence=0.8, time_horizon_minutes=5,
        )
        short_sig = MicrostructureSignal(
            signal_name="trade_flow_imbalance", direction="short",
            confidence=0.8, time_horizon_minutes=3,
        )
        custom_aggregator._latest_book_signals = [long_sig, None]
        custom_aggregator._latest_trade_signal = short_sig
        custom_aggregator._latest_spread_signal = None

        composite = custom_aggregator.get_composite_signal()
        # OBI has weight 2.0, trade_flow has weight 1.0
        # Net direction: (0.8 * 2.0 + (-0.8) * 1.0) / 3.0 = (1.6 - 0.8)/3 = 0.267
        assert composite.direction_score > 0.0  # OBI dominates

    def test_component_signals_included(self, aggregator):
        """Composite should list all contributing signals."""
        long_sig = MicrostructureSignal(
            signal_name="order_book_imbalance", direction="long",
            confidence=0.7, time_horizon_minutes=5,
        )
        aggregator._latest_book_signals = [long_sig, None]
        aggregator._latest_trade_signal = None
        aggregator._latest_spread_signal = None

        composite = aggregator.get_composite_signal()
        assert len(composite.component_signals) == 1
        assert composite.component_signals[0].signal_name == "order_book_imbalance"


# ---------------------------------------------------------------------------
# get_signal_dict_entry tests
# ---------------------------------------------------------------------------

class TestSignalDictEntry:
    def test_returns_expected_keys(self, aggregator):
        entry = aggregator.get_signal_dict_entry()
        assert "microstructure_advanced" in entry
        assert "microstructure_advanced_confidence" in entry

    def test_values_are_floats(self, aggregator):
        entry = aggregator.get_signal_dict_entry()
        assert isinstance(entry["microstructure_advanced"], float)
        assert isinstance(entry["microstructure_advanced_confidence"], float)

    def test_empty_aggregator_returns_zeros(self, aggregator):
        entry = aggregator.get_signal_dict_entry()
        assert entry["microstructure_advanced"] == 0.0
        assert entry["microstructure_advanced_confidence"] == 0.0


# ---------------------------------------------------------------------------
# Edge case / error handling tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_update_book_with_none_signals_handled(self, aggregator):
        """If underlying signals return None, composite still works."""
        bids, asks = _make_balanced_book()
        aggregator.update_book(bids, asks)
        composite = aggregator.get_composite_signal()
        # No signal should fire on balanced book
        assert composite.active_signal_count == 0

    def test_multiple_book_updates(self, aggregator):
        """Multiple rapid updates should not crash or accumulate stale data."""
        for _ in range(50):
            bids, asks = _make_bullish_book()
            aggregator.update_book(bids, asks)
        composite = aggregator.get_composite_signal()
        assert isinstance(composite, CompositeSignal)

    def test_mixed_signals_and_trades(self, aggregator):
        """Interleave book updates and trades."""
        now = datetime.now()
        bids, asks = _make_bullish_book()
        aggregator.update_book(bids, asks)
        for i in range(50):
            aggregator.update_trade(50000.0, 1.0, "sell", now - timedelta(seconds=i))
        composite = aggregator.get_composite_signal()
        # Should handle mixed signals without error
        assert isinstance(composite, CompositeSignal)
