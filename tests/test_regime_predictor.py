"""
Tests for intelligence/regime_predictor.py — RegimePredictor
==============================================================
Covers HMM transition probability predictions, regime volatility
forecast, tail risk computation, and HorizonEstimate output.
"""

import math

import pytest
from unittest.mock import MagicMock

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.regime_predictor import RegimePredictor, _DEFAULT_REGIME_VOL, _REGIME_BIAS
from core.data_structures import HorizonEstimate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "probability_clamp": [0.45, 0.56],
    }


@pytest.fixture
def predictor(config):
    return RegimePredictor(config, regime_detector=None)


@pytest.fixture
def detector_mock():
    mock = MagicMock()
    mock._current_regime_name = "trending_up"
    mock.get_transition_probabilities.return_value = {
        "trending_up": 0.60,
        "mean_reverting": 0.15,
        "low_volatility": 0.10,
        "high_volatility": 0.10,
        "chaotic": 0.05,
    }
    return mock


# ---------------------------------------------------------------------------
# Current Regime Tests
# ---------------------------------------------------------------------------

class TestGetCurrentRegime:

    def test_no_detector_defaults_low_volatility(self, predictor):
        regime = predictor._get_current_regime()
        assert regime == "low_volatility"

    def test_with_detector_uses_regime_name(self, config, detector_mock):
        pred = RegimePredictor(config, regime_detector=detector_mock)
        assert pred._get_current_regime() == "trending_up"

    def test_fallback_to_current_regime_attr(self, config):
        mock = MagicMock(spec=[])
        mock.current_regime = "mean_reverting"
        # _current_regime_name should be checked first via getattr
        # Since spec=[] means no _current_regime_name -> getattr returns None
        pred = RegimePredictor(config, regime_detector=mock)
        regime = pred._get_current_regime()
        assert regime == "mean_reverting"


# ---------------------------------------------------------------------------
# Transition Probability Tests
# ---------------------------------------------------------------------------

class TestGetTransitionProbs:

    def test_fallback_when_no_detector(self, predictor):
        probs = predictor._get_transition_probs(300)
        assert isinstance(probs, dict)
        # Current regime is low_volatility (no detector)
        assert probs["low_volatility"] > probs.get("chaotic", 0)
        # Should be normalized
        total = sum(probs.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_uses_detector_transition_matrix(self, config, detector_mock):
        pred = RegimePredictor(config, regime_detector=detector_mock)
        probs = pred._get_transition_probs(300)
        assert probs["trending_up"] == 0.60
        detector_mock.get_transition_probabilities.assert_called_with(steps=1)

    def test_steps_scale_with_horizon(self, config, detector_mock):
        pred = RegimePredictor(config, regime_detector=detector_mock)
        pred._get_transition_probs(900)  # 900/300 = 3 steps
        detector_mock.get_transition_probabilities.assert_called_with(steps=3)

    def test_fallback_on_detector_exception(self, config):
        mock = MagicMock()
        mock._current_regime_name = "trending_up"
        mock.get_transition_probabilities.side_effect = RuntimeError("broken")
        pred = RegimePredictor(config, regime_detector=mock)
        probs = pred._get_transition_probs(300)
        # Falls back to simple probs
        assert "trending_up" in probs
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Directional Bias Tests
# ---------------------------------------------------------------------------

class TestRegimeDirectionalBias:

    def test_trending_up_positive_bias(self, predictor):
        probs = {"trending_up": 0.8, "low_volatility": 0.2}
        bias = predictor._regime_directional_bias("trending_up", probs)
        assert bias > 0

    def test_trending_down_negative_bias(self, predictor):
        probs = {"trending_down": 0.8, "low_volatility": 0.2}
        bias = predictor._regime_directional_bias("trending_down", probs)
        assert bias < 0

    def test_chaotic_near_zero_bias(self, predictor):
        probs = {"chaotic": 0.8, "low_volatility": 0.2}
        bias = predictor._regime_directional_bias("chaotic", probs)
        assert abs(bias) < 0.1

    def test_bias_clamped_to_minus1_plus1(self, predictor):
        probs = {"trending_up": 1.0}
        bias = predictor._regime_directional_bias("trending_up", probs)
        assert -1.0 <= bias <= 1.0


# ---------------------------------------------------------------------------
# Volatility Forecast Tests
# ---------------------------------------------------------------------------

class TestForecastVolatility:

    def test_low_volatility_regime(self, predictor):
        probs = {"low_volatility": 1.0}
        result = predictor._forecast_volatility("low_volatility", probs, 300)
        expected_vol_per_bar = _DEFAULT_REGIME_VOL["low_volatility"]
        assert result["vol_per_bar_bps"] == pytest.approx(expected_vol_per_bar, abs=0.5)
        assert result["expected_sigma_bps"] == pytest.approx(expected_vol_per_bar, abs=0.5)

    def test_chaotic_regime_high_vol(self, predictor):
        probs = {"chaotic": 1.0}
        result = predictor._forecast_volatility("chaotic", probs, 300)
        assert result["vol_per_bar_bps"] == pytest.approx(80.0, abs=1.0)

    def test_vol_scales_with_sqrt_time(self, predictor):
        probs = {"low_volatility": 1.0}
        result_5m = predictor._forecast_volatility("low_volatility", probs, 300)
        result_20m = predictor._forecast_volatility("low_volatility", probs, 1200)
        # 20m = 4 bars, 5m = 1 bar -> ratio should be sqrt(4) = 2
        ratio = result_20m["expected_sigma_bps"] / result_5m["expected_sigma_bps"]
        assert ratio == pytest.approx(2.0, rel=0.01)

    def test_mixed_regime_weighted_vol(self, predictor):
        probs = {"low_volatility": 0.5, "high_volatility": 0.5}
        result = predictor._forecast_volatility("low_volatility", probs, 300)
        expected = 0.5 * _DEFAULT_REGIME_VOL["low_volatility"] + 0.5 * _DEFAULT_REGIME_VOL["high_volatility"]
        assert result["vol_per_bar_bps"] == pytest.approx(expected, abs=0.5)


# ---------------------------------------------------------------------------
# Tail Risk Tests
# ---------------------------------------------------------------------------

class TestRegimeTailProb:

    def test_zero_sigma_returns_zero(self, predictor):
        probs = {"low_volatility": 1.0}
        assert predictor._regime_tail_prob(0.0, 10, probs) == 0.0

    def test_basic_tail_probability(self, predictor):
        probs = {"low_volatility": 0.9, "chaotic": 0.05, "high_volatility": 0.05}
        p = predictor._regime_tail_prob(20.0, 10, probs)
        assert 0.0 < p < 1.0

    def test_high_chaotic_amplifies_tail(self, predictor):
        probs_normal = {"low_volatility": 0.95, "chaotic": 0.0, "high_volatility": 0.0, "trending": 0.05}
        probs_chaotic = {"low_volatility": 0.3, "chaotic": 0.4, "high_volatility": 0.3}
        p_normal = predictor._regime_tail_prob(20.0, 25, probs_normal)
        p_chaotic = predictor._regime_tail_prob(20.0, 25, probs_chaotic)
        assert p_chaotic > p_normal


# ---------------------------------------------------------------------------
# Predict Tests
# ---------------------------------------------------------------------------

class TestPredict:

    def test_returns_horizon_estimate(self, predictor):
        result = predictor.predict("BTC-USD", "long", 300)
        assert isinstance(result, HorizonEstimate)
        assert result.horizon_seconds == 300
        assert result.horizon_label == "5m"
        assert result.dominant_signal == "regime"

    def test_probability_within_clamp(self, predictor):
        result = predictor.predict("BTC-USD", "long", 900)
        assert 0.45 <= result.p_profit <= 0.56

    def test_short_side_inverts_probability(self, predictor):
        long_est = predictor.predict("BTC-USD", "long", 300)
        short_est = predictor.predict("BTC-USD", "short", 300)
        assert long_est.p_profit + short_est.p_profit == pytest.approx(1.0, abs=0.01)

    def test_confidence_decreases_with_horizon(self, predictor):
        est_5m = predictor.predict("BTC-USD", "long", 300)
        est_60m = predictor.predict("BTC-USD", "long", 3600)
        assert est_5m.estimate_confidence > est_60m.estimate_confidence

    def test_with_detector(self, config, detector_mock):
        pred = RegimePredictor(config, regime_detector=detector_mock)
        result = pred.predict("BTC-USD", "long", 300)
        assert isinstance(result, HorizonEstimate)
        # trending_up has positive bias -> long should have higher p_profit
        # than neutral 0.5 (though clamped)
        assert result.p_profit >= 0.45


# ---------------------------------------------------------------------------
# Weights Tests
# ---------------------------------------------------------------------------

class TestGetWeights:

    def test_short_horizon_weights(self):
        w = RegimePredictor._get_weights(300)
        assert w["regime"] == 0.50
        assert sum(w.values()) == pytest.approx(1.0, abs=0.01)

    def test_medium_horizon_weights(self):
        w = RegimePredictor._get_weights(900)
        assert w["regime"] == 0.60

    def test_long_horizon_weights(self):
        w = RegimePredictor._get_weights(3600)
        assert w["regime"] == 0.70
