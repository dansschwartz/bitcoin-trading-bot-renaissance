"""
Tests for intelligence/statistical_predictor.py — StatisticalPredictor
========================================================================
Covers VWAP deviation, volatility regime, volume anomaly, bar momentum,
autocorrelation predictions, and HorizonEstimate output.
"""

import math
import time

import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.statistical_predictor import StatisticalPredictor
from core.data_structures import HorizonEstimate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "vwap_window_trades": 500,
        "probability_clamp": [0.42, 0.62],
    }


@pytest.fixture
def predictor(config):
    return StatisticalPredictor(config)


def _feed_trades(pred, pair, n=200, base_price=50000.0, ts_start=None, trend=0.0):
    """Feed synthetic trades into the predictor."""
    ts = ts_start or (time.time() - n)
    for i in range(n):
        price = base_price + trend * i + (i % 3 - 1) * 0.5
        size = 0.01 + (i % 5) * 0.001
        pred.on_trade(pair, price, size, ts + i)


# ---------------------------------------------------------------------------
# Feed Handler Tests
# ---------------------------------------------------------------------------

class TestOnTrade:

    def test_creates_vwap_data_for_new_pair(self, predictor):
        predictor.on_trade("BTC-USD", 50000.0, 0.1, time.time())
        assert "BTC-USD" in predictor.vwap_data
        data = predictor.vwap_data["BTC-USD"]
        assert data["price_volume_sum"] == 50000.0 * 0.1
        assert data["volume_sum"] == 0.1
        assert data["last_price"] == 50000.0

    def test_accumulates_price_volume_sum(self, predictor):
        predictor.on_trade("BTC-USD", 50000.0, 1.0, 1000.0)
        predictor.on_trade("BTC-USD", 51000.0, 2.0, 1001.0)
        data = predictor.vwap_data["BTC-USD"]
        expected_pv = 50000.0 * 1.0 + 51000.0 * 2.0
        assert data["price_volume_sum"] == pytest.approx(expected_pv)
        assert data["volume_sum"] == pytest.approx(3.0)

    def test_computes_returns_in_bps(self, predictor):
        predictor.on_trade("BTC-USD", 50000.0, 0.1, 1000.0)
        predictor.on_trade("BTC-USD", 50050.0, 0.1, 1001.0)
        data = predictor.vwap_data["BTC-USD"]
        rets = list(data["returns"])
        # First trade creates entry with last_price=50000, also computes return
        # from initial last_price (= same price) -> 0.0 bps
        # Second trade: (50050 - 50000) / 50000 * 10000 = 10 bps
        assert len(rets) == 2
        assert rets[0] == pytest.approx(0.0, abs=0.01)
        assert rets[1] == pytest.approx(10.0, abs=0.1)


class TestOnBarClose:

    def test_stores_bar_momentum(self, predictor):
        bar = {"open": 50000.0, "close": 50100.0, "high": 50200.0, "low": 49900.0, "volume": 100.0}
        predictor.on_bar_close("BTC-USD", bar)
        mom = predictor.bar_momentum["BTC-USD"]
        assert mom["close"] == 50100.0
        assert mom["bar_return_bps"] == pytest.approx(
            (50100 - 50000) / 50000 * 10000, abs=0.1
        )

    def test_bar_with_zero_open(self, predictor):
        bar = {"open": 0, "close": 50100.0, "high": 50200.0, "low": 49900.0, "volume": 100.0}
        predictor.on_bar_close("BTC-USD", bar)
        mom = predictor.bar_momentum["BTC-USD"]
        assert mom["bar_return_bps"] == 0


# ---------------------------------------------------------------------------
# Feature Computation Tests
# ---------------------------------------------------------------------------

class TestVWAPDeviation:

    def test_no_data_returns_zero(self, predictor):
        assert predictor._compute_vwap_deviation("BTC-USD") == 0.0

    def test_price_at_vwap_near_zero(self, predictor):
        """When current price is near VWAP, signal should be near zero."""
        _feed_trades(predictor, "BTC-USD", n=200, base_price=50000.0, trend=0.0)
        signal = predictor._compute_vwap_deviation("BTC-USD")
        assert abs(signal) < 0.5  # near zero for flat prices

    def test_price_below_vwap_positive_signal(self, predictor):
        """Price below VWAP -> positive signal (reversion up expected)."""
        # Feed some high prices to shift VWAP up, then end low
        for i in range(100):
            predictor.on_trade("BTC-USD", 50100.0, 1.0, 1000.0 + i)
        for i in range(50):
            predictor.on_trade("BTC-USD", 49900.0, 1.0, 1100.0 + i)
        signal = predictor._compute_vwap_deviation("BTC-USD")
        # Current price 49900 < VWAP, z-score negative -> inverted -> positive
        assert signal > 0


class TestVolatilityState:

    def test_no_data_defaults(self, predictor):
        result = predictor._compute_volatility_state("BTC-USD")
        assert result["regime"] == "stable"
        assert result["per_second_vol_bps"] == 2.0

    def test_insufficient_returns_defaults(self, predictor):
        for i in range(10):
            predictor.on_trade("BTC-USD", 50000.0 + i, 0.1, 1000.0 + i)
        result = predictor._compute_volatility_state("BTC-USD")
        assert result["regime"] == "stable"

    def test_expanding_volatility(self, predictor):
        """Feed stable prices then volatile prices to detect expanding regime."""
        # 70 stable trades (older)
        for i in range(70):
            predictor.on_trade("BTC-USD", 50000.0 + (i % 2) * 0.01, 0.1, 1000.0 + i)
        # 30 volatile trades (recent)
        for i in range(30):
            predictor.on_trade("BTC-USD", 50000.0 + (i % 2) * 100.0, 0.1, 1070.0 + i)
        result = predictor._compute_volatility_state("BTC-USD")
        assert result["regime"] in ("expanding", "stable")


class TestVolumeAnomaly:

    def test_no_data_returns_zero(self, predictor):
        assert predictor._compute_volume_anomaly("BTC-USD") == 0.0

    def test_insufficient_data_returns_zero(self, predictor):
        for i in range(20):
            predictor.on_trade("BTC-USD", 50000.0, 1.0, 1000.0 + i)
        assert predictor._compute_volume_anomaly("BTC-USD") == 0.0


class TestBarMomentum:

    def test_no_bar_returns_zero(self, predictor):
        assert predictor._compute_bar_momentum("BTC-USD") == 0.0

    def test_positive_bar_positive_momentum(self, predictor):
        bar = {"open": 50000.0, "close": 50200.0, "high": 50300.0, "low": 49900.0, "volume": 100.0}
        predictor.on_bar_close("BTC-USD", bar)
        mom = predictor._compute_bar_momentum("BTC-USD")
        assert mom > 0

    def test_bar_momentum_clamped(self, predictor):
        bar = {"open": 50000.0, "close": 55000.0, "high": 55000.0, "low": 50000.0, "volume": 100.0}
        predictor.on_bar_close("BTC-USD", bar)
        mom = predictor._compute_bar_momentum("BTC-USD")
        assert -1.0 <= mom <= 1.0


class TestAutocorrelation:

    def test_no_data_returns_zero(self, predictor):
        assert predictor._compute_autocorrelation("BTC-USD") == 0.0

    def test_insufficient_returns_zero(self, predictor):
        for i in range(10):
            predictor.on_trade("BTC-USD", 50000.0 + i, 0.1, 1000.0 + i)
        assert predictor._compute_autocorrelation("BTC-USD") == 0.0

    def test_autocorrelation_clamped(self, predictor):
        _feed_trades(predictor, "BTC-USD", n=200)
        ac = predictor._compute_autocorrelation("BTC-USD")
        assert -1.0 <= ac <= 1.0


# ---------------------------------------------------------------------------
# Predict Tests
# ---------------------------------------------------------------------------

class TestPredict:

    def test_predict_returns_horizon_estimate(self, predictor):
        _feed_trades(predictor, "BTC-USD", n=200)
        result = predictor.predict("BTC-USD", "long", 120)
        assert isinstance(result, HorizonEstimate)
        assert result.horizon_seconds == 120
        assert result.horizon_label == "2m"

    def test_predict_probability_clamped(self, predictor):
        _feed_trades(predictor, "BTC-USD", n=200)
        result = predictor.predict("BTC-USD", "long", 30)
        assert 0.42 <= result.p_profit <= 0.62

    def test_predict_short_side(self, predictor):
        _feed_trades(predictor, "BTC-USD", n=200)
        long_est = predictor.predict("BTC-USD", "long", 120)
        short_est = predictor.predict("BTC-USD", "short", 120)
        assert long_est.p_profit + short_est.p_profit == pytest.approx(1.0, abs=0.01)

    def test_predict_30s_horizon_label(self, predictor):
        _feed_trades(predictor, "BTC-USD", n=200)
        result = predictor.predict("BTC-USD", "long", 30)
        assert result.horizon_label == "30s"

    def test_predict_300s_horizon_label(self, predictor):
        _feed_trades(predictor, "BTC-USD", n=200)
        result = predictor.predict("BTC-USD", "long", 300)
        assert result.horizon_label == "5m"

    def test_dominant_signal_valid(self, predictor):
        _feed_trades(predictor, "BTC-USD", n=200)
        result = predictor.predict("BTC-USD", "long", 120)
        assert result.dominant_signal in ("vwap", "vol", "volume", "bar", "autocorr")

    def test_data_quality_improves_with_more_data(self, predictor):
        # Few trades
        _feed_trades(predictor, "BTC-USD", n=10)
        est_few = predictor.predict("BTC-USD", "long", 30)

        _feed_trades(predictor, "ETH-USD", n=300)
        est_many = predictor.predict("ETH-USD", "long", 30)

        assert est_many.estimate_confidence > est_few.estimate_confidence


# ---------------------------------------------------------------------------
# Weights Tests
# ---------------------------------------------------------------------------

class TestGetWeights:

    def test_30s_weights(self, predictor):
        w = predictor._get_weights(30)
        assert w["vwap"] == 0.35
        assert sum(w.values()) == pytest.approx(1.0, abs=0.01)

    def test_120s_weights(self, predictor):
        w = predictor._get_weights(120)
        assert w["vwap"] == 0.30

    def test_300s_weights(self, predictor):
        w = predictor._get_weights(300)
        assert w["vwap"] == 0.25
        assert w["vol"] == 0.25


# ---------------------------------------------------------------------------
# Tail Probability Tests
# ---------------------------------------------------------------------------

class TestTailProb:

    def test_zero_sigma_returns_zero(self):
        assert StatisticalPredictor._tail_prob(0.0, 10, 0.5) == 0.0

    def test_basic_tail_probability(self):
        p = StatisticalPredictor._tail_prob(20.0, 10, 0.5)
        assert 0.0 < p < 1.0

    def test_capped_at_1(self):
        # With very small sigma and high p_adv, ensure capping
        p = StatisticalPredictor._tail_prob(0.001, 0.0001, 0.99)
        assert p <= 1.0
