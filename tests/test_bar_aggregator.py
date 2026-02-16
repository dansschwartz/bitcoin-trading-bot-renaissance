"""
Tests for data_module/bar_aggregator.py — BarAggregator.

Covers tick-to-bar aggregation, 5-minute boundary detection, OHLCV calculation,
derived features (VWAP, log-return, spread, buy/sell ratio), and DB persistence.
"""

import math
import os
import sqlite3
import tempfile
import time

import pytest

from data_module.bar_aggregator import (
    BarAggregator,
    FiveMinuteBar,
    BAR_DURATION_S,
    _BarAccumulator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_aggregator(tmp_path) -> BarAggregator:
    """Create a BarAggregator with a temp SQLite database."""
    db_path = str(tmp_path / "test_bars.db")
    return BarAggregator(config={}, db_path=db_path)


def _bar_start(ts: float) -> float:
    return float(int(ts) // BAR_DURATION_S * BAR_DURATION_S)


# ---------------------------------------------------------------------------
# Tests — FiveMinuteBar dataclass
# ---------------------------------------------------------------------------

class TestFiveMinuteBar:
    def test_fields(self):
        bar = FiveMinuteBar(
            pair="BTC-USD", exchange="mexc",
            bar_start=1000.0, bar_end=1300.0,
            open=100.0, high=110.0, low=95.0, close=105.0,
            volume=50.0, num_trades=10,
            vwap=103.0, log_return=0.01,
            avg_spread_bps=5.0, buy_sell_ratio=0.6,
            funding_rate=0.0001,
        )
        assert bar.pair == "BTC-USD"
        assert bar.close == 105.0
        assert bar.num_trades == 10


# ---------------------------------------------------------------------------
# Tests — _BarAccumulator
# ---------------------------------------------------------------------------

class TestBarAccumulator:
    def test_initial_state(self):
        acc = _BarAccumulator(bar_start=0.0)
        assert acc.bar_start == 0.0
        assert acc.bar_end == BAR_DURATION_S
        assert acc.open is None
        assert acc.high == -math.inf
        assert acc.low == math.inf
        assert acc.volume == 0.0
        assert acc.num_trades == 0


# ---------------------------------------------------------------------------
# Tests — _bar_start_for
# ---------------------------------------------------------------------------

class TestBarStartFor:
    def test_aligned_timestamp(self):
        # Timestamp exactly at a 5-minute boundary
        ts = 1800.0  # = 6 * 300
        assert BarAggregator._bar_start_for(ts) == 1800.0

    def test_mid_bar_timestamp(self):
        ts = 1900.0  # 1800 + 100
        assert BarAggregator._bar_start_for(ts) == 1800.0

    def test_just_before_boundary(self):
        ts = 2099.0  # Just before 2100 boundary
        assert BarAggregator._bar_start_for(ts) == 1800.0


# ---------------------------------------------------------------------------
# Tests — on_trade (OHLCV accumulation)
# ---------------------------------------------------------------------------

class TestOnTrade:
    def test_single_trade_sets_ohlc(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        agg.on_trade("BTC-USD", "mexc", 50000.0, 1.0, "buy", ts)

        key = ("BTC-USD", "mexc")
        acc = agg._current_bars[key]
        assert acc.open == 50000.0
        assert acc.high == 50000.0
        assert acc.low == 50000.0
        assert acc.close == 50000.0
        assert acc.volume == 1.0
        assert acc.num_trades == 1

    def test_multiple_trades_update_ohlc(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        agg.on_trade("BTC-USD", "mexc", 100.0, 1.0, "buy", ts)
        agg.on_trade("BTC-USD", "mexc", 110.0, 2.0, "sell", ts + 10)
        agg.on_trade("BTC-USD", "mexc", 95.0, 0.5, "buy", ts + 20)
        agg.on_trade("BTC-USD", "mexc", 105.0, 1.5, "sell", ts + 30)

        key = ("BTC-USD", "mexc")
        acc = agg._current_bars[key]
        assert acc.open == 100.0
        assert acc.high == 110.0
        assert acc.low == 95.0
        assert acc.close == 105.0
        assert acc.volume == pytest.approx(5.0)
        assert acc.num_trades == 4

    def test_buy_sell_volume_tracking(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        agg.on_trade("ETH-USD", "binance", 3000.0, 2.0, "buy", ts)
        agg.on_trade("ETH-USD", "binance", 3010.0, 1.0, "sell", ts + 5)

        key = ("ETH-USD", "binance")
        acc = agg._current_bars[key]
        assert acc.buy_volume == 2.0
        assert acc.sell_volume == 1.0

    def test_bar_boundary_triggers_flush(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        bar1_start = 1800.0
        bar2_start = bar1_start + BAR_DURATION_S

        # Trade in first bar
        agg.on_trade("BTC-USD", "mexc", 50000.0, 1.0, "buy", bar1_start + 10)
        # Trade in second bar triggers flush of first
        agg.on_trade("BTC-USD", "mexc", 51000.0, 0.5, "buy", bar2_start + 10)

        # First bar should have been flushed to DB
        bars = agg.get_bars("BTC-USD", "mexc", n_bars=10)
        assert len(bars) == 1
        assert bars[0].open == 50000.0
        assert bars[0].close == 50000.0


# ---------------------------------------------------------------------------
# Tests — on_orderbook_snapshot (spread tracking)
# ---------------------------------------------------------------------------

class TestOnOrderbookSnapshot:
    def test_spread_calculation(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        # First need a trade to create the bar
        agg.on_trade("BTC-USD", "mexc", 50000.0, 1.0, "buy", ts)
        # Then add spread observation
        agg.on_orderbook_snapshot("BTC-USD", "mexc", 49990.0, 50010.0, ts + 1)

        key = ("BTC-USD", "mexc")
        acc = agg._current_bars[key]
        assert acc.spread_samples == 1
        # mid = 50000, spread = 20, bps = 20/50000 * 10000 = 4 bps
        assert acc.spread_sum == pytest.approx(4.0, abs=0.01)

    def test_multiple_spread_observations(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        agg.on_trade("BTC-USD", "mexc", 100.0, 1.0, "buy", ts)
        agg.on_orderbook_snapshot("BTC-USD", "mexc", 99.0, 101.0, ts + 1)
        agg.on_orderbook_snapshot("BTC-USD", "mexc", 98.0, 102.0, ts + 2)

        key = ("BTC-USD", "mexc")
        acc = agg._current_bars[key]
        assert acc.spread_samples == 2


# ---------------------------------------------------------------------------
# Tests — _flush_bar (derived features)
# ---------------------------------------------------------------------------

class TestFlushBar:
    def test_vwap_computation(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        # Two trades: 100@1.0 and 200@2.0 => VWAP = (100*1+200*2)/(1+2) = 500/3
        agg.on_trade("BTC-USD", "mexc", 100.0, 1.0, "buy", ts)
        agg.on_trade("BTC-USD", "mexc", 200.0, 2.0, "buy", ts + 10)

        # Trigger flush by sending trade in next bar
        agg.on_trade("BTC-USD", "mexc", 150.0, 0.1, "buy", ts + BAR_DURATION_S + 1)

        bars = agg.get_bars("BTC-USD", "mexc", n_bars=10)
        assert len(bars) == 1
        expected_vwap = (100.0 * 1.0 + 200.0 * 2.0) / 3.0
        assert bars[0].vwap == pytest.approx(expected_vwap, abs=0.01)

    def test_log_return_first_bar_is_zero(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        agg.on_trade("BTC-USD", "mexc", 100.0, 1.0, "buy", ts)
        agg.on_trade("BTC-USD", "mexc", 100.0, 0.1, "buy", ts + BAR_DURATION_S + 1)

        bars = agg.get_bars("BTC-USD", "mexc", n_bars=10)
        assert bars[0].log_return == 0.0  # No previous close

    def test_log_return_second_bar_computed(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        bar1_start = 1800.0
        bar2_start = bar1_start + BAR_DURATION_S
        bar3_start = bar2_start + BAR_DURATION_S

        agg.on_trade("BTC-USD", "mexc", 100.0, 1.0, "buy", bar1_start + 1)
        agg.on_trade("BTC-USD", "mexc", 110.0, 1.0, "buy", bar2_start + 1)
        agg.on_trade("BTC-USD", "mexc", 120.0, 1.0, "buy", bar3_start + 1)

        bars = agg.get_bars("BTC-USD", "mexc", n_bars=10)
        assert len(bars) == 2  # Two completed bars
        # Second bar's log_return = ln(110/100)
        expected_lr = math.log(110.0 / 100.0)
        assert bars[1].log_return == pytest.approx(expected_lr, abs=0.001)

    def test_buy_sell_ratio(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        agg.on_trade("BTC-USD", "mexc", 100.0, 3.0, "buy", ts)
        agg.on_trade("BTC-USD", "mexc", 101.0, 1.0, "sell", ts + 10)
        agg.on_trade("BTC-USD", "mexc", 102.0, 0.1, "buy", ts + BAR_DURATION_S + 1)

        bars = agg.get_bars("BTC-USD", "mexc", n_bars=10)
        # buy_volume=3, sell_volume=1, total=4, ratio = 3/4 = 0.75
        assert bars[0].buy_sell_ratio == pytest.approx(0.75, abs=0.01)

    def test_avg_spread_bps(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        agg.on_trade("BTC-USD", "mexc", 100.0, 1.0, "buy", ts)
        agg.on_orderbook_snapshot("BTC-USD", "mexc", 99.0, 101.0, ts + 1)
        agg.on_orderbook_snapshot("BTC-USD", "mexc", 99.5, 100.5, ts + 2)
        agg.on_trade("BTC-USD", "mexc", 100.0, 0.1, "buy", ts + BAR_DURATION_S + 1)

        bars = agg.get_bars("BTC-USD", "mexc", n_bars=10)
        # First snapshot: mid=100, spread=2, bps=200
        # Second snapshot: mid=100, spread=1, bps=100
        # Average = 150 bps
        assert bars[0].avg_spread_bps == pytest.approx(150.0, abs=1.0)

    def test_empty_bar_skipped(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        # Create bar with no trades, just orderbook
        ts = 1800.0
        agg.on_orderbook_snapshot("BTC-USD", "mexc", 99.0, 101.0, ts)
        # Flush by going to next bar
        agg.on_orderbook_snapshot("BTC-USD", "mexc", 99.0, 101.0, ts + BAR_DURATION_S + 1)

        bars = agg.get_bars("BTC-USD", "mexc", n_bars=10)
        assert len(bars) == 0  # Empty bar should be skipped


# ---------------------------------------------------------------------------
# Tests — get_bars / get_latest_bar
# ---------------------------------------------------------------------------

class TestGetBars:
    def test_returns_chronological_order(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        for i in range(4):
            bar_start = ts + i * BAR_DURATION_S
            agg.on_trade("BTC-USD", "mexc", 100.0 + i, 1.0, "buy", bar_start + 1)

        # Flush all by going to 5th bar
        final_ts = ts + 4 * BAR_DURATION_S + 1
        agg.on_trade("BTC-USD", "mexc", 104.0, 0.1, "buy", final_ts)

        bars = agg.get_bars("BTC-USD", "mexc", n_bars=10)
        assert len(bars) == 4
        # Should be sorted by bar_start ascending
        for i in range(len(bars) - 1):
            assert bars[i].bar_start < bars[i + 1].bar_start

    def test_n_bars_limit(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        for i in range(5):
            bar_start = ts + i * BAR_DURATION_S
            agg.on_trade("BTC-USD", "mexc", 100.0, 1.0, "buy", bar_start + 1)
        agg.on_trade("BTC-USD", "mexc", 100.0, 0.1, "buy", ts + 5 * BAR_DURATION_S + 1)

        bars = agg.get_bars("BTC-USD", "mexc", n_bars=3)
        assert len(bars) == 3

    def test_get_latest_bar(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        agg.on_trade("BTC-USD", "mexc", 100.0, 1.0, "buy", ts + 1)
        agg.on_trade("BTC-USD", "mexc", 200.0, 1.0, "buy", ts + BAR_DURATION_S + 1)
        agg.on_trade("BTC-USD", "mexc", 300.0, 0.1, "buy", ts + 2 * BAR_DURATION_S + 1)

        latest = agg.get_latest_bar("BTC-USD", "mexc")
        assert latest is not None
        assert latest.open == 200.0

    def test_get_latest_bar_empty(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        assert agg.get_latest_bar("BTC-USD", "mexc") is None


# ---------------------------------------------------------------------------
# Tests — Multiple pairs
# ---------------------------------------------------------------------------

class TestMultiplePairs:
    def test_separate_bars_per_pair(self, tmp_path):
        agg = _make_aggregator(tmp_path)
        ts = 1800.0
        agg.on_trade("BTC-USD", "mexc", 50000.0, 1.0, "buy", ts + 1)
        agg.on_trade("ETH-USD", "mexc", 3000.0, 10.0, "sell", ts + 2)

        # Flush both
        next_ts = ts + BAR_DURATION_S + 1
        agg.on_trade("BTC-USD", "mexc", 51000.0, 0.1, "buy", next_ts)
        agg.on_trade("ETH-USD", "mexc", 3100.0, 0.1, "buy", next_ts)

        btc_bars = agg.get_bars("BTC-USD", "mexc", n_bars=10)
        eth_bars = agg.get_bars("ETH-USD", "mexc", n_bars=10)

        assert len(btc_bars) == 1
        assert len(eth_bars) == 1
        assert btc_bars[0].open == 50000.0
        assert eth_bars[0].open == 3000.0
