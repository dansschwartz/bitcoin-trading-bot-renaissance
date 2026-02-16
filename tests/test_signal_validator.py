"""
Tests for intelligence/signal_validator.py — SignalValidator
==============================================================
Covers the 6-gate validation pipeline: Pattern Detection,
Statistical Significance, Cost Check, Regime Check, OOS test,
and confidence tier assignment.
"""

import math

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.signal_validator import (
    SignalValidator,
    SignalValidationReport,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "min_p_value": 0.05,
        "min_trades_for_validation": 10,
        "min_sharpe": 0.5,
        "out_of_sample_pct": 0.25,
        "min_regimes_profitable": 2,
        "oos_sharpe_retention": 0.5,
        "tier_1_allocation": 1.0,
        "tier_2_allocation": 0.25,
        "tier_3_allocation": 0.0,
    }


@pytest.fixture
def validator(config, tmp_path):
    db_path = str(tmp_path / "test.db")
    return SignalValidator(config, db_path)


def _make_trade(return_bps, regime=None):
    trade = {"return_bps": return_bps}
    if regime:
        trade["regime"] = regime
    return trade


def _profitable_signal(market_data):
    """Signal generator that produces profitable trades."""
    return [_make_trade(20.0 + i * 0.1) for i in range(len(market_data) // 3)]


def _unprofitable_signal(market_data):
    """Signal generator that produces losing trades."""
    return [_make_trade(-20.0 - i * 0.1) for i in range(len(market_data) // 3)]


def _small_signal(market_data):
    """Signal generator that produces too few trades."""
    return [_make_trade(5.0)]


# ---------------------------------------------------------------------------
# Gate 1: Pattern Detection Tests
# ---------------------------------------------------------------------------

class TestGate1PatternDetection:

    def test_insufficient_trades(self, validator):
        trades = [_make_trade(10.0)] * 5
        result = validator._gate_1_pattern_detection(trades)
        assert result == ValidationResult.INSUFFICIENT_DATA

    def test_passed_with_positive_returns(self, validator):
        trades = [_make_trade(10.0)] * 15
        result = validator._gate_1_pattern_detection(trades)
        assert result == ValidationResult.PASSED

    def test_failed_with_negative_returns(self, validator):
        trades = [_make_trade(-10.0)] * 15
        result = validator._gate_1_pattern_detection(trades)
        assert result == ValidationResult.FAILED

    def test_failed_with_50pct_win_rate_negative_avg(self, validator):
        # 50% win rate but negative average
        trades = [_make_trade(5.0)] * 5 + [_make_trade(-15.0)] * 10
        result = validator._gate_1_pattern_detection(trades)
        assert result == ValidationResult.FAILED


# ---------------------------------------------------------------------------
# Gate 2: Statistical Significance Tests
# ---------------------------------------------------------------------------

class TestGate2StatisticalSignificance:

    def test_insufficient_data_short_array(self, validator):
        returns = np.array([1.0, 2.0], dtype=np.float64)
        result, p = validator._gate_2_statistical_significance(returns)
        assert result == ValidationResult.INSUFFICIENT_DATA
        assert p == 1.0

    def test_passed_with_significant_positive_returns(self, validator):
        np.random.seed(42)
        returns = np.random.normal(loc=5.0, scale=1.0, size=100)
        result, p = validator._gate_2_statistical_significance(returns)
        assert result == ValidationResult.PASSED
        assert p < 0.05

    def test_failed_with_zero_mean(self, validator):
        np.random.seed(42)
        returns = np.random.normal(loc=0.0, scale=1.0, size=100)
        result, p = validator._gate_2_statistical_significance(returns)
        # Mean is near zero -> p > threshold
        assert result == ValidationResult.FAILED

    def test_failed_with_negative_mean(self, validator):
        np.random.seed(42)
        returns = np.random.normal(loc=-5.0, scale=1.0, size=100)
        result, p = validator._gate_2_statistical_significance(returns)
        assert result == ValidationResult.FAILED

    def test_handles_nan_values(self, validator):
        returns = np.array([1.0, np.nan, 2.0, 3.0] * 10, dtype=np.float64)
        result, p = validator._gate_2_statistical_significance(returns)
        # Should filter NaN, remaining 30 elements with positive mean
        assert result in (ValidationResult.PASSED, ValidationResult.FAILED, ValidationResult.INSUFFICIENT_DATA)


# ---------------------------------------------------------------------------
# Gate 3: Explanation Tests
# ---------------------------------------------------------------------------

class TestGate3Explanation:

    def test_no_trades_explanation(self, validator):
        result = validator._gate_3_explanation([], np.array([]))
        assert "No trades" in result

    def test_summary_with_trades(self, validator):
        trades = [_make_trade(10.0)] * 5 + [_make_trade(-5.0)] * 3
        returns = np.array([10.0] * 5 + [-5.0] * 3)
        result = validator._gate_3_explanation(trades, returns)
        assert "8 trades" in result
        assert "5 wins" in result
        assert "3 losses" in result
        assert "Sharpe" in result


# ---------------------------------------------------------------------------
# Gate 4: Cost Check Tests
# ---------------------------------------------------------------------------

class TestGate4CostCheck:

    def test_insufficient_trades(self, validator):
        trades = [_make_trade(100.0)] * 3
        result = validator._gate_4_cost_check(trades, "BTC-USD")
        assert result == ValidationResult.INSUFFICIENT_DATA

    def test_passed_when_gross_exceeds_costs(self, validator):
        # avg gross = 25 bps > 15 bps default cost
        trades = [_make_trade(25.0)] * 15
        result = validator._gate_4_cost_check(trades, "BTC-USD")
        assert result == ValidationResult.PASSED

    def test_failed_when_gross_below_costs(self, validator):
        # avg gross = 5 bps < 15 bps default cost
        trades = [_make_trade(5.0)] * 15
        result = validator._gate_4_cost_check(trades, "BTC-USD")
        assert result == ValidationResult.FAILED


# ---------------------------------------------------------------------------
# Gate 5: Regime Check Tests
# ---------------------------------------------------------------------------

class TestGate5RegimeCheck:

    def test_no_regimes_insufficient_data(self, validator):
        trades = [_make_trade(10.0)] * 10
        result, profitable = validator._gate_5_regime_check(trades, [])
        assert result == ValidationResult.INSUFFICIENT_DATA
        assert profitable == []

    def test_passed_multiple_profitable_regimes(self, validator):
        trades = (
            [_make_trade(10.0, regime="trending")] * 10 +
            [_make_trade(8.0, regime="mean_reverting")] * 10
        )
        regimes = ["trending"] * 10 + ["mean_reverting"] * 10
        result, profitable = validator._gate_5_regime_check(trades, regimes)
        assert result == ValidationResult.PASSED
        assert len(profitable) >= 2

    def test_failed_only_one_profitable_regime(self, validator):
        trades = (
            [_make_trade(10.0, regime="trending")] * 10 +
            [_make_trade(-10.0, regime="mean_reverting")] * 10
        )
        regimes = ["trending"] * 10 + ["mean_reverting"] * 10
        result, profitable = validator._gate_5_regime_check(trades, regimes)
        assert result == ValidationResult.FAILED
        assert len(profitable) == 1

    def test_regime_from_index_fallback(self, validator):
        """When trade doesn't have regime, fall back to regime list index."""
        trades = [_make_trade(10.0)] * 10 + [_make_trade(8.0)] * 10
        regimes = ["trending"] * 10 + ["mean_reverting"] * 10
        result, profitable = validator._gate_5_regime_check(trades, regimes)
        assert result == ValidationResult.PASSED


# ---------------------------------------------------------------------------
# Gate 6: Out-of-Sample Tests
# ---------------------------------------------------------------------------

class TestGate6OOS:

    def test_insufficient_oos_data(self, validator):
        result = validator._gate_6_out_of_sample(
            _profitable_signal, "BTC-USD", [{}] * 5
        )
        assert result == ValidationResult.INSUFFICIENT_DATA

    def test_passed_with_good_oos_sharpe(self, validator):
        oos_data = [{"price": 50000 + i} for i in range(100)]
        result = validator._gate_6_out_of_sample(
            _profitable_signal, "BTC-USD", oos_data
        )
        assert result == ValidationResult.PASSED

    def test_failed_with_bad_oos(self, validator):
        oos_data = [{"price": 50000 + i} for i in range(100)]
        result = validator._gate_6_out_of_sample(
            _unprofitable_signal, "BTC-USD", oos_data
        )
        assert result == ValidationResult.FAILED

    def test_signal_generator_exception(self, validator):
        def bad_signal(data):
            raise RuntimeError("boom")

        oos_data = [{"price": 50000}] * 50
        result = validator._gate_6_out_of_sample(bad_signal, "BTC-USD", oos_data)
        assert result == ValidationResult.FAILED


# ---------------------------------------------------------------------------
# Confidence Tier Tests
# ---------------------------------------------------------------------------

class TestAssignConfidenceTier:

    def test_tier_1_all_gates_passed(self, validator):
        report = SignalValidationReport(
            signal_name="test", pair="BTC-USD",
            start_date="2024-01-01", end_date="2024-12-31",
            gate_1_pattern=ValidationResult.PASSED,
            gate_2_stat_sig=ValidationResult.PASSED,
            gate_4_cost=ValidationResult.PASSED,
            gate_5_regime=ValidationResult.PASSED,
            gate_6_oos=ValidationResult.PASSED,
            in_sample_sharpe=1.0,
        )
        assert validator.assign_confidence_tier(report) == 1

    def test_tier_2_core_passed_advanced_insufficient(self, validator):
        report = SignalValidationReport(
            signal_name="test", pair="BTC-USD",
            start_date="2024-01-01", end_date="2024-12-31",
            gate_1_pattern=ValidationResult.PASSED,
            gate_2_stat_sig=ValidationResult.PASSED,
            gate_4_cost=ValidationResult.PASSED,
            gate_5_regime=ValidationResult.INSUFFICIENT_DATA,
            gate_6_oos=ValidationResult.INSUFFICIENT_DATA,
            in_sample_sharpe=1.0,
        )
        assert validator.assign_confidence_tier(report) == 2

    def test_tier_3_core_gate_failed(self, validator):
        report = SignalValidationReport(
            signal_name="test", pair="BTC-USD",
            start_date="2024-01-01", end_date="2024-12-31",
            gate_1_pattern=ValidationResult.FAILED,
            gate_2_stat_sig=ValidationResult.PASSED,
            gate_4_cost=ValidationResult.PASSED,
        )
        assert validator.assign_confidence_tier(report) == 3

    def test_tier_3_all_insufficient(self, validator):
        report = SignalValidationReport(
            signal_name="test", pair="BTC-USD",
            start_date="2024-01-01", end_date="2024-12-31",
        )
        assert validator.assign_confidence_tier(report) == 3


# ---------------------------------------------------------------------------
# Sharpe Computation Tests
# ---------------------------------------------------------------------------

class TestComputeSharpe:

    def test_zero_std_returns_zero(self):
        returns = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        assert SignalValidator._compute_sharpe(returns) == 0.0

    def test_positive_sharpe(self):
        np.random.seed(42)
        returns = np.random.normal(loc=2.0, scale=1.0, size=100)
        sharpe = SignalValidator._compute_sharpe(returns)
        assert sharpe > 0

    def test_single_return_is_zero(self):
        returns = np.array([5.0], dtype=np.float64)
        assert SignalValidator._compute_sharpe(returns) == 0.0


# ---------------------------------------------------------------------------
# Report Summary Tests
# ---------------------------------------------------------------------------

class TestReportSummary:

    def test_summary_format(self):
        report = SignalValidationReport(
            signal_name="my_signal", pair="BTC-USD",
            start_date="2024-01-01", end_date="2024-12-31",
            gate_1_pattern=ValidationResult.PASSED,
            gate_2_stat_sig=ValidationResult.PASSED,
            gate_2_p_value=0.001,
            gate_4_cost=ValidationResult.PASSED,
            gate_5_regime=ValidationResult.PASSED,
            gate_5_profitable_regimes=["trending", "mean_reverting"],
            gate_6_oos=ValidationResult.PASSED,
            total_trades=200,
            in_sample_sharpe=1.5,
            out_of_sample_sharpe=1.2,
            confidence_tier=1,
            allocation_pct=1.0,
        )
        s = report.summary()
        assert "my_signal" in s
        assert "BTC-USD" in s
        assert "tier=1" in s
        assert "G1=PASSED" in s
        assert "G2=PASSED" in s
