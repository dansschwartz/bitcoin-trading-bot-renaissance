"""
Tests for intelligence/multi_horizon_estimator.py — MultiHorizonEstimator
==========================================================================
Covers horizon routing, blending, cone analysis, action determination,
and the reevaluator integration interface.
"""

import time
from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock, patch
from dataclasses import dataclass

import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.multi_horizon_estimator import (
    MultiHorizonEstimator,
    DEFAULT_HORIZONS,
)
from core.data_structures import HorizonEstimate, ProbabilityCone, ConeAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_estimate(horizon_seconds, e_net_bps=1.0, p_profit=0.55, sigma_bps=10.0,
                   p_adverse_25bps=0.05, estimate_confidence=0.5):
    label = f"{horizon_seconds}s" if horizon_seconds < 60 else f"{horizon_seconds // 60}m"
    return HorizonEstimate(
        horizon_seconds=horizon_seconds,
        horizon_label=label,
        p_profit=p_profit,
        p_loss=1.0 - p_profit,
        e_favorable_bps=e_net_bps * 2,
        e_adverse_bps=abs(e_net_bps),
        e_net_bps=e_net_bps,
        sigma_bps=sigma_bps,
        p_adverse_10bps=0.02,
        p_adverse_25bps=p_adverse_25bps,
        p_adverse_50bps=0.001,
        estimate_confidence=estimate_confidence,
        dominant_signal="test",
    )


class FakePosition:
    """Minimal position-like object for testing."""
    def __init__(self, pair="BTC-USD", side="long", position_id="test-1",
                 age_seconds=10, signal_ttl_seconds=300):
        self.pair = pair
        self.side = side
        self.position_id = position_id
        self.age_seconds = age_seconds
        self.signal_ttl_seconds = signal_ttl_seconds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {"horizons": DEFAULT_HORIZONS}


@pytest.fixture
def micro():
    m = MagicMock()
    m.predict.return_value = _make_estimate(1, e_net_bps=2.0)
    return m


@pytest.fixture
def stat():
    s = MagicMock()
    s.predict.return_value = _make_estimate(120, e_net_bps=1.5)
    return s


@pytest.fixture
def regime():
    r = MagicMock()
    r.predict.return_value = _make_estimate(900, e_net_bps=0.5)
    return r


@pytest.fixture
def mhpe(config, micro, stat, regime):
    return MultiHorizonEstimator(config, micro, stat, regime)


# ---------------------------------------------------------------------------
# Horizon Routing Tests
# ---------------------------------------------------------------------------

class TestHorizonRouting:

    def test_1s_uses_micro_only(self, mhpe, micro):
        est = mhpe._estimate_horizon("BTC-USD", "long", 1)
        micro.predict.assert_called_with("BTC-USD", "long", 1)

    def test_5s_uses_micro_only(self, mhpe, micro):
        est = mhpe._estimate_horizon("BTC-USD", "long", 5)
        micro.predict.assert_called_with("BTC-USD", "long", 5)

    def test_30s_blends_micro_and_stat(self, config, micro, stat, regime):
        micro.predict.return_value = _make_estimate(30, e_net_bps=2.0, p_profit=0.60)
        stat.predict.return_value = _make_estimate(30, e_net_bps=1.0, p_profit=0.55)
        mhpe = MultiHorizonEstimator(config, micro, stat, regime)
        est = mhpe._estimate_horizon("BTC-USD", "long", 30)
        # Blended: 0.3 * micro + 0.7 * stat
        expected_p = round(0.60 * 0.3 + 0.55 * 0.7, 4)
        assert est.p_profit == pytest.approx(expected_p, abs=0.001)

    def test_120s_uses_stat_only(self, mhpe, stat):
        est = mhpe._estimate_horizon("BTC-USD", "long", 120)
        stat.predict.assert_called_with("BTC-USD", "long", 120)

    def test_300s_blends_stat_and_regime(self, config, micro, stat, regime):
        stat.predict.return_value = _make_estimate(300, e_net_bps=1.0, p_profit=0.55)
        regime.predict.return_value = _make_estimate(300, e_net_bps=0.5, p_profit=0.52)
        mhpe = MultiHorizonEstimator(config, micro, stat, regime)
        est = mhpe._estimate_horizon("BTC-USD", "long", 300)
        # Blended: 0.4 * stat + 0.6 * regime
        expected_p = round(0.55 * 0.4 + 0.52 * 0.6, 4)
        assert est.p_profit == pytest.approx(expected_p, abs=0.001)

    def test_900s_uses_regime_only(self, mhpe, regime):
        est = mhpe._estimate_horizon("BTC-USD", "long", 900)
        regime.predict.assert_called_with("BTC-USD", "long", 900)

    def test_3600s_uses_regime_only(self, mhpe, regime):
        est = mhpe._estimate_horizon("BTC-USD", "long", 3600)
        regime.predict.assert_called_with("BTC-USD", "long", 3600)

    def test_fallback_to_neutral_when_no_predictors(self, config):
        mhpe = MultiHorizonEstimator(config)  # no predictors
        est = mhpe._estimate_horizon("BTC-USD", "long", 1)
        assert est.p_profit == 0.50
        assert est.e_net_bps == 0.0
        assert est.dominant_signal == "none"


# ---------------------------------------------------------------------------
# Neutral Estimate Tests
# ---------------------------------------------------------------------------

class TestNeutralEstimate:

    def test_neutral_estimate_seconds(self):
        est = MultiHorizonEstimator._neutral_estimate(30)
        assert est.horizon_label == "30s"
        assert est.p_profit == 0.5

    def test_neutral_estimate_minutes(self):
        est = MultiHorizonEstimator._neutral_estimate(300)
        assert est.horizon_label == "5m"


# ---------------------------------------------------------------------------
# Blending Tests
# ---------------------------------------------------------------------------

class TestBlendEstimates:

    def test_blend_50_50(self):
        a = _make_estimate(30, e_net_bps=2.0, p_profit=0.60)
        b = _make_estimate(30, e_net_bps=0.0, p_profit=0.50)
        result = MultiHorizonEstimator._blend_estimates(a, b, 0.5, 0.5)
        assert result.p_profit == pytest.approx(0.55, abs=0.001)
        assert result.e_net_bps == pytest.approx(1.0, abs=0.01)

    def test_blend_dominant_signal_follows_larger_weight(self):
        a = _make_estimate(30, e_net_bps=2.0)
        b = _make_estimate(30, e_net_bps=1.0)
        a_dom = a.dominant_signal
        b_dom = b.dominant_signal

        result_a_heavy = MultiHorizonEstimator._blend_estimates(a, b, 0.8, 0.2)
        assert result_a_heavy.dominant_signal == a_dom

        result_b_heavy = MultiHorizonEstimator._blend_estimates(a, b, 0.2, 0.8)
        assert result_b_heavy.dominant_signal == b_dom


# ---------------------------------------------------------------------------
# Full Estimate Tests
# ---------------------------------------------------------------------------

class TestEstimate:

    def test_estimate_returns_probability_cone(self, mhpe):
        pos = FakePosition()
        cone = mhpe.estimate(pos)
        assert isinstance(cone, ProbabilityCone)
        assert len(cone.horizons) == len(DEFAULT_HORIZONS)
        assert cone.pair == "BTC-USD"
        assert cone.side == "long"
        assert cone.computation_time_ms >= 0

    def test_estimate_increments_metrics(self, mhpe):
        pos = FakePosition()
        assert mhpe._total_estimates == 0
        mhpe.estimate(pos)
        assert mhpe._total_estimates == 1
        mhpe.estimate(pos)
        assert mhpe._total_estimates == 2

    def test_estimate_short_position(self, mhpe):
        pos = FakePosition(side="short")
        cone = mhpe.estimate(pos)
        assert cone.side == "short"


# ---------------------------------------------------------------------------
# Cone Analysis Tests
# ---------------------------------------------------------------------------

class TestConeAnalysis:

    def test_all_negative_ev_close_now(self, config):
        """When EV negative at all horizons, action should be close_now."""
        micro = MagicMock()
        micro.predict.return_value = _make_estimate(1, e_net_bps=-1.0)
        stat = MagicMock()
        stat.predict.return_value = _make_estimate(120, e_net_bps=-2.0)
        regime = MagicMock()
        regime.predict.return_value = _make_estimate(900, e_net_bps=-3.0)

        mhpe = MultiHorizonEstimator(config, micro, stat, regime)
        pos = FakePosition()
        cone = mhpe.estimate(pos)
        assert cone.recommended_action == "close_now"
        assert cone.action_urgency == "high"


# ---------------------------------------------------------------------------
# Reevaluator Interface Tests
# ---------------------------------------------------------------------------

class TestAnalyzeForReevaluator:

    def test_returns_cone_analysis(self, mhpe):
        pos = FakePosition()
        analysis = mhpe.analyze_for_reevaluator(pos)
        assert isinstance(analysis, ConeAnalysis)
        assert analysis.position_id == "test-1"

    def test_close_now_gives_zero_multiplier(self, config):
        micro = MagicMock()
        micro.predict.return_value = _make_estimate(1, e_net_bps=-5.0)
        stat = MagicMock()
        stat.predict.return_value = _make_estimate(120, e_net_bps=-5.0)
        regime = MagicMock()
        regime.predict.return_value = _make_estimate(900, e_net_bps=-5.0)

        mhpe = MultiHorizonEstimator(config, micro, stat, regime)
        pos = FakePosition()
        analysis = mhpe.analyze_for_reevaluator(pos)
        assert analysis.size_multiplier == 0.0


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_get_metrics_initial(self, mhpe):
        metrics = mhpe.get_metrics()
        assert metrics["total_estimates"] == 0
        assert metrics["avg_computation_ms"] == 0
        assert metrics["predictors"]["micro"] is True
        assert metrics["predictors"]["stat"] is True
        assert metrics["predictors"]["regime"] is True

    def test_get_metrics_no_predictors(self, config):
        mhpe = MultiHorizonEstimator(config)
        metrics = mhpe.get_metrics()
        assert metrics["predictors"]["micro"] is False
        assert metrics["predictors"]["stat"] is False
        assert metrics["predictors"]["regime"] is False
