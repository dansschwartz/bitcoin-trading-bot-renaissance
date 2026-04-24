"""
Tests for bot/signals.py — signal generation and continuous signal conversion functions.

Tests the continuous RSI, MACD, Bollinger, and OBV signal conversions as well
as the weighted signal fusion and edge cases.
"""

import logging
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import the continuous signal functions directly
from renaissance_trading_bot import (
    _continuous_rsi_signal,
    _continuous_macd_signal,
    _continuous_bollinger_signal,
    _continuous_obv_signal,
    _signed_strength,
)
from analysis.enhanced_technical_indicators import IndicatorOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_indicator(value=50.0, signal="HOLD", strength=0.5, confidence=0.5,
                    metadata=None) -> IndicatorOutput:
    """Create a test IndicatorOutput."""
    return IndicatorOutput(
        value=value,
        signal=signal,
        strength=strength,
        confidence=confidence,
        metadata=metadata or {},
    )


def _make_bot() -> MagicMock:
    """Create a minimal mock bot for signal tests."""
    bot = MagicMock()
    bot.logger = logging.getLogger("test_bot_signals")
    bot.signal_weights = {
        "order_flow": 0.14,
        "volume": 0.08,
        "macd": 0.05,
        "rsi": 0.05,
        "bollinger": 0.05,
        "alternative": 0.01,
        "ml_ensemble": 0.20,
        "ml_cnn": 0.0,
        "crash_regime": 0.15,
    }
    bot._force_float = lambda v: float(v) if v is not None else 0.0
    # signal_fusion mock
    bot.signal_fusion = MagicMock()
    bot.signal_fusion.fuse_signals_with_ml.return_value = (0.05, 0.6, {"contributions": {}})
    return bot


# ---------------------------------------------------------------------------
# Tests: _continuous_rsi_signal
# ---------------------------------------------------------------------------


class TestContinuousRSISignal:
    """Tests for _continuous_rsi_signal conversion."""

    def test_rsi_0_returns_plus_one(self):
        """RSI 0 (max oversold) → +1.0 (strong buy)."""
        signal = _make_indicator(value=0.0)
        result = _continuous_rsi_signal(signal)
        assert result == pytest.approx(1.0)

    def test_rsi_100_returns_minus_one(self):
        """RSI 100 (max overbought) → -1.0 (strong sell)."""
        signal = _make_indicator(value=100.0)
        result = _continuous_rsi_signal(signal)
        assert result == pytest.approx(-1.0)

    def test_rsi_50_returns_zero(self):
        """RSI 50 (neutral) → 0.0."""
        signal = _make_indicator(value=50.0)
        result = _continuous_rsi_signal(signal)
        assert result == pytest.approx(0.0)

    def test_rsi_30_returns_positive(self):
        """RSI 30 (oversold zone) → +0.4."""
        signal = _make_indicator(value=30.0)
        result = _continuous_rsi_signal(signal)
        assert result == pytest.approx(0.4, abs=0.01)

    def test_rsi_70_returns_negative(self):
        """RSI 70 (overbought zone) → -0.4."""
        signal = _make_indicator(value=70.0)
        result = _continuous_rsi_signal(signal)
        assert result == pytest.approx(-0.4, abs=0.01)

    def test_rsi_none_returns_zero(self):
        """None input → 0.0."""
        result = _continuous_rsi_signal(None)
        assert result == 0.0

    def test_rsi_nan_returns_zero(self):
        """NaN RSI value → 0.0."""
        signal = _make_indicator(value=float("nan"))
        result = _continuous_rsi_signal(signal)
        assert result == 0.0

    def test_rsi_inf_returns_zero(self):
        """Inf RSI value → 0.0."""
        signal = _make_indicator(value=float("inf"))
        result = _continuous_rsi_signal(signal)
        assert result == 0.0

    def test_rsi_none_value_treated_as_50(self):
        """IndicatorOutput with value=None → treated as RSI 50 → 0.0."""
        signal = _make_indicator(value=None)
        result = _continuous_rsi_signal(signal)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: _continuous_macd_signal
# ---------------------------------------------------------------------------


class TestContinuousMACDSignal:
    """Tests for _continuous_macd_signal conversion."""

    def test_positive_histogram_returns_positive(self):
        """Positive MACD histogram → positive signal."""
        signal = _make_indicator(
            metadata={"histogram": 2.0, "signal_line": 1.0}
        )
        result = _continuous_macd_signal(signal)
        assert result > 0
        assert result <= 1.0

    def test_negative_histogram_returns_negative(self):
        """Negative MACD histogram → negative signal."""
        signal = _make_indicator(
            metadata={"histogram": -3.0, "signal_line": 1.0}
        )
        result = _continuous_macd_signal(signal)
        assert result < 0
        assert result >= -1.0

    def test_zero_histogram_returns_zero(self):
        """Zero histogram → 0.0."""
        signal = _make_indicator(metadata={"histogram": 0.0, "signal_line": 1.0})
        result = _continuous_macd_signal(signal)
        assert result == pytest.approx(0.0)

    def test_none_input_returns_zero(self):
        """None input → 0.0."""
        result = _continuous_macd_signal(None)
        assert result == 0.0

    def test_missing_metadata_returns_zero(self):
        """Signal with no metadata → 0.0."""
        signal = _make_indicator(metadata=None)
        result = _continuous_macd_signal(signal)
        assert result == 0.0

    def test_clipped_to_range(self):
        """Large histogram value should be clipped to [-1, 1]."""
        signal = _make_indicator(
            metadata={"histogram": 100.0, "signal_line": 1.0}
        )
        result = _continuous_macd_signal(signal)
        assert result == pytest.approx(1.0)

    def test_nan_histogram_returns_zero(self):
        """NaN histogram → 0.0."""
        signal = _make_indicator(
            metadata={"histogram": float("nan"), "signal_line": 1.0}
        )
        result = _continuous_macd_signal(signal)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Tests: _continuous_bollinger_signal
# ---------------------------------------------------------------------------


class TestContinuousBollingerSignal:
    """Tests for _continuous_bollinger_signal conversion."""

    def test_lower_band_returns_plus_one(self):
        """Position 0 (at lower band) → +1.0 (buy)."""
        signal = _make_indicator(value=0.0)
        result = _continuous_bollinger_signal(signal)
        assert result == pytest.approx(1.0)

    def test_upper_band_returns_minus_one(self):
        """Position 1 (at upper band) → -1.0 (sell)."""
        signal = _make_indicator(value=1.0)
        result = _continuous_bollinger_signal(signal)
        assert result == pytest.approx(-1.0)

    def test_middle_returns_zero(self):
        """Position 0.5 (middle) → 0.0."""
        signal = _make_indicator(value=0.5)
        result = _continuous_bollinger_signal(signal)
        assert result == pytest.approx(0.0)

    def test_none_returns_zero(self):
        """None input → 0.0."""
        result = _continuous_bollinger_signal(None)
        assert result == 0.0

    def test_nan_returns_zero(self):
        """NaN value → 0.0."""
        signal = _make_indicator(value=float("nan"))
        result = _continuous_bollinger_signal(signal)
        assert result == 0.0

    def test_below_lower_clipped(self):
        """Value below 0 should be clipped to +1.0."""
        signal = _make_indicator(value=-0.5)
        result = _continuous_bollinger_signal(signal)
        assert result == pytest.approx(1.0)

    def test_above_upper_clipped(self):
        """Value above 1 should be clipped to -1.0."""
        signal = _make_indicator(value=1.5)
        result = _continuous_bollinger_signal(signal)
        assert result == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Tests: _continuous_obv_signal
# ---------------------------------------------------------------------------


class TestContinuousOBVSignal:
    """Tests for _continuous_obv_signal conversion."""

    def test_positive_momentum_returns_positive(self):
        """Positive OBV momentum → positive signal."""
        signal = _make_indicator(
            metadata={
                "obv_momentum": 0.2,
                "obv_change": 0.1,
                "divergence": 0,
            }
        )
        result = _continuous_obv_signal(signal)
        assert result > 0

    def test_negative_momentum_returns_negative(self):
        """Negative OBV momentum → negative signal."""
        signal = _make_indicator(
            metadata={
                "obv_momentum": -0.3,
                "obv_change": -0.1,
                "divergence": 0,
            }
        )
        result = _continuous_obv_signal(signal)
        assert result < 0

    def test_clipped_to_range(self):
        """Large values should be clipped to [-1, 1]."""
        signal = _make_indicator(
            metadata={
                "obv_momentum": 10.0,
                "obv_change": 5.0,
                "divergence": 3.0,
            }
        )
        result = _continuous_obv_signal(signal)
        assert -1.0 <= result <= 1.0

    def test_none_returns_zero(self):
        """None input → 0.0."""
        result = _continuous_obv_signal(None)
        assert result == 0.0

    def test_no_metadata_uses_fallback(self):
        """Signal with no metadata → falls back to _signed_strength."""
        signal = _make_indicator(
            value=0.5, signal="BUY", strength=0.3,
            metadata=None,
        )
        result = _continuous_obv_signal(signal)
        # Should use _signed_strength fallback
        assert isinstance(result, float)

    def test_nan_momentum_returns_zero(self):
        """NaN momentum → 0.0."""
        signal = _make_indicator(
            metadata={
                "obv_momentum": float("nan"),
                "obv_change": 0.0,
                "divergence": 0,
            }
        )
        result = _continuous_obv_signal(signal)
        assert result == 0.0

    def test_zero_values_returns_zero(self):
        """All zero metadata → 0.0."""
        signal = _make_indicator(
            metadata={
                "obv_momentum": 0.0,
                "obv_change": 0.0,
                "divergence": 0,
            }
        )
        result = _continuous_obv_signal(signal)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: _signed_strength (internal helper)
# ---------------------------------------------------------------------------


class TestSignedStrength:
    """Tests for _signed_strength helper."""

    def test_buy_positive(self):
        """BUY signal → positive strength."""
        signal = _make_indicator(signal="BUY", strength=0.8)
        result = _signed_strength(signal)
        assert result == pytest.approx(0.8)

    def test_sell_negative(self):
        """SELL signal → negative strength."""
        signal = _make_indicator(signal="SELL", strength=0.6)
        result = _signed_strength(signal)
        assert result == pytest.approx(-0.6)

    def test_hold_zero(self):
        """HOLD signal → 0.0."""
        signal = _make_indicator(signal="HOLD", strength=0.5)
        result = _signed_strength(signal)
        assert result == pytest.approx(0.0)

    def test_none_returns_zero(self):
        """None → 0.0."""
        result = _signed_strength(None)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Tests: calculate_weighted_signal
# ---------------------------------------------------------------------------


class TestCalculateWeightedSignal:
    """Tests for calculate_weighted_signal function."""

    def test_returns_float_tuple(self):
        """calculate_weighted_signal should return (float, dict)."""
        from bot.signals import calculate_weighted_signal

        bot = _make_bot()
        signals = {"order_flow": 0.5, "volume": -0.3, "rsi": 0.1}

        result = calculate_weighted_signal(bot, signals)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], dict)

    def test_delegates_to_signal_fusion(self):
        """calculate_weighted_signal should call signal_fusion.fuse_signals_with_ml."""
        from bot.signals import calculate_weighted_signal

        bot = _make_bot()
        signals = {"order_flow": 0.5, "volume": -0.3}
        calculate_weighted_signal(bot, signals)

        bot.signal_fusion.fuse_signals_with_ml.assert_called_once()

    def test_handles_none_signal_values(self):
        """None values in signals dict should be converted to 0.0."""
        from bot.signals import calculate_weighted_signal

        bot = _make_bot()
        signals = {"order_flow": None, "volume": 0.3}

        # Should not raise
        result = calculate_weighted_signal(bot, signals)
        assert isinstance(result[0], float)

    def test_handles_non_numeric_values(self):
        """Non-numeric signal values should be converted to 0.0."""
        from bot.signals import calculate_weighted_signal

        bot = _make_bot()
        # _force_float will raise on "invalid" → caught and returned as 0.0
        bot._force_float = MagicMock(side_effect=lambda v: float(v) if isinstance(v, (int, float)) else (_ for _ in ()).throw(ValueError("bad")))
        signals = {"order_flow": "invalid", "volume": 0.3}

        # The function uses try/except, so this should handle gracefully
        result = calculate_weighted_signal(bot, signals)
        assert isinstance(result[0], float)

    def test_ml_package_preserved_in_signals(self):
        """ml_package key should be passed through to fusion without float conversion."""
        from bot.signals import calculate_weighted_signal

        bot = _make_bot()
        mock_ml = MagicMock()
        signals = {"order_flow": 0.5, "ml_package": mock_ml}

        calculate_weighted_signal(bot, signals)

        # Check that ml_package was passed in processed_signals
        call_args = bot.signal_fusion.fuse_signals_with_ml.call_args
        processed = call_args[0][0]
        assert processed["ml_package"] is mock_ml
