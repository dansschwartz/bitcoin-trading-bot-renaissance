"""
Tests for intelligence/microstructure_predictor.py — MicrostructurePredictor
=============================================================================
Covers orderbook imbalance, trade flow, spread dynamics, tick momentum
predictions, and HorizonEstimate output validation.
"""

import math
import time

import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.microstructure_predictor import MicrostructurePredictor
from core.data_structures import HorizonEstimate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "max_history_seconds": 120,
        "max_ticks": 2000,
        "orderbook_levels": 5,
        "probability_clamp": [0.40, 0.70],
    }


@pytest.fixture
def predictor(config):
    return MicrostructurePredictor(config)


def _seed_orderbook(pred, pair, bid_depth=10.0, ask_depth=10.0, ts=None):
    ts = ts or time.time()
    bids = [[50000 - i * 10, bid_depth / 5] for i in range(5)]
    asks = [[50000 + i * 10, ask_depth / 5] for i in range(5)]
    pred.on_orderbook_update(pair, bids, asks, ts)


def _seed_trades(pred, pair, n=100, base_price=50000.0, ts_start=None):
    ts = ts_start or time.time() - n
    for i in range(n):
        side = "buy" if i % 2 == 0 else "sell"
        price = base_price + (i * 0.1 if side == "buy" else -i * 0.1)
        pred.on_trade(pair, price, 0.01, side, ts + i)


def _seed_spreads(pred, pair, n=20, spread_bps=5.0, ts_start=None):
    ts = ts_start or time.time() - n
    mid = 50000.0
    half_spread = mid * spread_bps / 20000.0
    for i in range(n):
        pred.on_spread_update(pair, mid - half_spread, mid + half_spread, ts + i)


# ---------------------------------------------------------------------------
# Orderbook Tests
# ---------------------------------------------------------------------------

class TestOrderbookUpdate:

    def test_creates_snapshot_for_new_pair(self, predictor):
        _seed_orderbook(predictor, "BTC-USD")
        assert "BTC-USD" in predictor.orderbook_snapshots
        assert len(predictor.orderbook_snapshots["BTC-USD"]) == 1

    def test_imbalance_balanced_book(self, predictor):
        _seed_orderbook(predictor, "BTC-USD", bid_depth=10.0, ask_depth=10.0)
        snapshot = predictor.orderbook_snapshots["BTC-USD"][-1]
        assert snapshot["imbalance"] == pytest.approx(0.5, abs=0.01)

    def test_imbalance_bid_heavy(self, predictor):
        _seed_orderbook(predictor, "BTC-USD", bid_depth=30.0, ask_depth=10.0)
        snapshot = predictor.orderbook_snapshots["BTC-USD"][-1]
        assert snapshot["imbalance"] > 0.5  # More bids -> > 0.5

    def test_imbalance_ask_heavy(self, predictor):
        _seed_orderbook(predictor, "BTC-USD", bid_depth=10.0, ask_depth=30.0)
        snapshot = predictor.orderbook_snapshots["BTC-USD"][-1]
        assert snapshot["imbalance"] < 0.5

    def test_imbalance_zero_depth_returns_half(self, predictor):
        # Both sides empty
        predictor.on_orderbook_update("BTC-USD", [], [], time.time())
        snapshot = predictor.orderbook_snapshots["BTC-USD"][-1]
        assert snapshot["imbalance"] == 0.5


# ---------------------------------------------------------------------------
# Trade Flow Tests
# ---------------------------------------------------------------------------

class TestTradeFlow:

    def test_on_trade_creates_entries(self, predictor):
        predictor.on_trade("BTC-USD", 50000.0, 0.1, "buy", time.time())
        assert "BTC-USD" in predictor.trade_flow
        assert "BTC-USD" in predictor.tick_history

    def test_trade_flow_all_buys(self, predictor):
        now = time.time()
        for i in range(10):
            predictor.on_trade("BTC-USD", 50000.0, 1.0, "buy", now - 5 + i)
        flow = predictor._compute_trade_flow("BTC-USD", now, 10)
        assert flow == pytest.approx(1.0, abs=0.01)

    def test_trade_flow_all_sells(self, predictor):
        now = time.time()
        for i in range(10):
            predictor.on_trade("BTC-USD", 50000.0, 1.0, "sell", now - 5 + i)
        flow = predictor._compute_trade_flow("BTC-USD", now, 10)
        assert flow == pytest.approx(0.0, abs=0.01)

    def test_trade_flow_no_data_returns_half(self, predictor):
        flow = predictor._compute_trade_flow("UNKNOWN", time.time(), 10)
        assert flow == 0.5


# ---------------------------------------------------------------------------
# Spread Tests
# ---------------------------------------------------------------------------

class TestSpreadDynamics:

    def test_spread_update_creates_entries(self, predictor):
        predictor.on_spread_update("BTC-USD", 49999.0, 50001.0, time.time())
        assert "BTC-USD" in predictor.spread_history
        snap = predictor.spread_history["BTC-USD"][-1]
        assert snap["mid"] == pytest.approx(50000.0, rel=1e-6)
        assert snap["spread_bps"] > 0

    def test_spread_trend_no_data_returns_zero(self, predictor):
        assert predictor._compute_spread_trend("BTC-USD", time.time()) == 0.0

    def test_spread_trend_insufficient_data(self, predictor):
        for i in range(5):
            predictor.on_spread_update("BTC-USD", 49999.0, 50001.0, time.time() + i)
        # Less than 10 entries -> return 0.0
        assert predictor._compute_spread_trend("BTC-USD", time.time()) == 0.0


# ---------------------------------------------------------------------------
# Tick Momentum Tests
# ---------------------------------------------------------------------------

class TestTickMomentum:

    def test_no_data_returns_zero(self, predictor):
        assert predictor._compute_tick_momentum("BTC-USD", time.time(), 10) == 0.0

    def test_all_buy_ticks_positive_momentum(self, predictor):
        now = time.time()
        for i in range(20):
            predictor.on_trade("BTC-USD", 50000.0 + i, 1.0, "buy", now - 10 + i)
        mom = predictor._compute_tick_momentum("BTC-USD", now, 15)
        assert mom > 0

    def test_momentum_clamped(self, predictor):
        now = time.time()
        for i in range(50):
            predictor.on_trade("BTC-USD", 50000.0, 100.0, "buy", now - 25 + i)
        mom = predictor._compute_tick_momentum("BTC-USD", now, 30)
        assert -1.0 <= mom <= 1.0


# ---------------------------------------------------------------------------
# Prediction Tests
# ---------------------------------------------------------------------------

class TestPredict:

    def test_predict_returns_horizon_estimate(self, predictor):
        _seed_orderbook(predictor, "BTC-USD")
        _seed_trades(predictor, "BTC-USD", n=60)
        _seed_spreads(predictor, "BTC-USD", n=20)

        result = predictor.predict("BTC-USD", "long", 1)
        assert isinstance(result, HorizonEstimate)
        assert result.horizon_seconds == 1
        assert result.horizon_label == "1s"

    def test_predict_probability_clamped(self, predictor):
        _seed_orderbook(predictor, "BTC-USD")
        _seed_trades(predictor, "BTC-USD", n=60)
        _seed_spreads(predictor, "BTC-USD", n=20)

        result = predictor.predict("BTC-USD", "long", 5)
        assert 0.40 <= result.p_profit <= 0.70
        assert 0.30 <= result.p_loss <= 0.60

    def test_predict_short_side_inverted(self, predictor):
        _seed_orderbook(predictor, "BTC-USD", bid_depth=30.0, ask_depth=10.0)
        _seed_trades(predictor, "BTC-USD", n=60)
        _seed_spreads(predictor, "BTC-USD", n=20)

        long_est = predictor.predict("BTC-USD", "long", 1)
        short_est = predictor.predict("BTC-USD", "short", 1)
        # p_profit for long + p_profit for short should conceptually be complementary
        # within the clamp range
        assert long_est.p_profit + short_est.p_profit == pytest.approx(1.0, abs=0.01)

    def test_predict_different_horizons(self, predictor):
        _seed_orderbook(predictor, "BTC-USD")
        _seed_trades(predictor, "BTC-USD", n=60)

        est_1s = predictor.predict("BTC-USD", "long", 1)
        est_30s = predictor.predict("BTC-USD", "long", 30)
        # Longer horizons should have different weights
        assert est_1s.dominant_signal in ("imbalance", "flow", "spread", "momentum")
        assert est_30s.dominant_signal in ("imbalance", "flow", "spread", "momentum")

    def test_data_quality_improves_with_more_data(self, predictor):
        # Few ticks
        predictor.on_trade("BTC-USD", 50000.0, 1.0, "buy", time.time())
        _seed_orderbook(predictor, "BTC-USD")
        est_few = predictor.predict("BTC-USD", "long", 1)

        # Many ticks
        _seed_trades(predictor, "ETH-USD", n=100)
        _seed_orderbook(predictor, "ETH-USD")
        est_many = predictor.predict("ETH-USD", "long", 1)

        assert est_many.estimate_confidence > est_few.estimate_confidence


# ---------------------------------------------------------------------------
# Signal Conversion Tests
# ---------------------------------------------------------------------------

class TestSignalConversion:

    def test_imbalance_to_signal(self):
        # 0.5 -> 0.0 (neutral), 0.75 -> 0.5, 1.0 -> 1.0
        assert MicrostructurePredictor._imbalance_to_signal(0.5) == pytest.approx(0.0)
        assert MicrostructurePredictor._imbalance_to_signal(0.75) == pytest.approx(0.5)
        assert MicrostructurePredictor._imbalance_to_signal(1.0) == pytest.approx(1.0)

    def test_flow_to_signal(self):
        assert MicrostructurePredictor._flow_to_signal(0.5) == pytest.approx(0.0)
        assert MicrostructurePredictor._flow_to_signal(1.0) == pytest.approx(1.0)
        assert MicrostructurePredictor._flow_to_signal(0.0) == pytest.approx(-1.0)

    def test_format_horizon(self):
        assert MicrostructurePredictor._format_horizon(1) == "1s"
        assert MicrostructurePredictor._format_horizon(30) == "30s"
        assert MicrostructurePredictor._format_horizon(60) == "1m"
        assert MicrostructurePredictor._format_horizon(300) == "5m"

    def test_tail_probability_zero_sigma(self):
        result = MicrostructurePredictor._tail_probability(0.0, 10, 0.5)
        assert result == 0.0

    def test_dominant_signal_name(self):
        weights = {"imbalance": 0.10, "flow": 0.25, "spread": 0.15, "momentum": 0.50}
        assert MicrostructurePredictor._dominant_signal_name(weights) == "momentum"
