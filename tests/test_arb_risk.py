"""
Tests for arbitrage/risk/arb_risk.py

Covers:
  - ArbitrageRiskEngine initialization with defaults and config
  - Position limits (max single arb)
  - Daily loss limit and halt
  - Trade rate limiting (per hour)
  - Consecutive losses tracking and halt
  - DO NOTHING default (halted state)
  - Total exposure limit
  - Confidence threshold
  - Funding arb approval gate
  - One-sided exposure tracking
  - Daily reset at midnight
  - Manual halt reset
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from arbitrage.risk.arb_risk import ArbitrageRiskEngine


def _make_signal(qty="0.01", buy_price="50000", sell_price="50050", confidence="0.7"):
    sig = MagicMock()
    sig.signal_id = "test_sig_001"
    sig.recommended_quantity = Decimal(qty)
    sig.buy_price = Decimal(buy_price)
    sig.sell_price = Decimal(sell_price)
    sig.confidence = Decimal(confidence)
    return sig


class TestArbitrageRiskEngine:
    @pytest.fixture
    def engine(self):
        return ArbitrageRiskEngine()

    @pytest.fixture
    def engine_custom(self):
        return ArbitrageRiskEngine({
            'max_single_arb_usd': 100,
            'max_total_exposure_usd': 1000,
            'max_daily_loss_usd': 50,
            'max_trades_per_hour': 10,
            'max_consecutive_losses': 3,
        })

    def test_default_initialization(self, engine):
        assert engine.max_single_arb_usd == Decimal('500')
        assert engine.max_total_exposure_usd == Decimal('5000')
        assert engine.max_daily_loss_usd == Decimal('100')
        assert engine.max_trades_per_hour == 100
        assert engine._halted is False

    def test_custom_config(self, engine_custom):
        assert engine_custom.max_single_arb_usd == Decimal('100')
        assert engine_custom.max_daily_loss_usd == Decimal('50')
        assert engine_custom.max_trades_per_hour == 10

    def test_approve_normal_signal(self, engine):
        # notional = 0.005 * ~50025 = ~$250 which is within $500 limit
        signal = _make_signal(qty="0.005")
        assert engine.approve_arbitrage(signal) is True

    def test_reject_when_halted(self, engine):
        engine._halted = True
        engine._halt_reason = "Test halt"
        signal = _make_signal()
        assert engine.approve_arbitrage(signal) is False

    def test_reject_exceeds_single_trade_limit(self, engine_custom):
        # $100 limit; signal notional = 0.01 * 50025 = $500.25
        signal = _make_signal(qty="0.01", buy_price="50000", sell_price="50050")
        assert engine_custom.approve_arbitrage(signal) is False

    def test_approve_within_single_trade_limit(self, engine_custom):
        # notional = 0.001 * 50025 = $50.025 < $100
        signal = _make_signal(qty="0.001")
        assert engine_custom.approve_arbitrage(signal) is True

    def test_daily_loss_triggers_halt(self, engine):
        engine._daily_pnl = Decimal('-101')
        signal = _make_signal()
        result = engine.approve_arbitrage(signal)
        assert result is False
        assert engine._halted is True
        assert "Daily loss" in engine._halt_reason

    def test_consecutive_losses_triggers_halt(self, engine):
        engine._consecutive_losses = 10  # Default max is 10
        signal = _make_signal()
        result = engine.approve_arbitrage(signal)
        assert result is False
        assert engine._halted is True

    def test_trade_rate_limit(self, engine_custom):
        # Fill up 10 trades in last hour
        now = datetime.utcnow()
        for _ in range(10):
            engine_custom._trade_times.append(now)

        signal = _make_signal(qty="0.001")
        assert engine_custom.approve_arbitrage(signal) is False

    def test_trade_rate_old_trades_ignored(self, engine_custom):
        # Old trades (>1 hour) should not count
        old_time = datetime.utcnow() - timedelta(hours=2)
        for _ in range(10):
            engine_custom._trade_times.append(old_time)

        signal = _make_signal(qty="0.001")
        assert engine_custom.approve_arbitrage(signal) is True

    def test_total_exposure_limit(self, engine_custom):
        engine_custom._current_exposure = {"mexc": Decimal('900'), "binance": Decimal('50')}
        # Total = 950, adding ~50 more would be ~1000 => at limit
        signal = _make_signal(qty="0.001")  # notional ~$50
        result = engine_custom.approve_arbitrage(signal)
        # 950 + 50.025 > 1000
        assert result is False

    def test_low_confidence_rejected(self, engine):
        signal = _make_signal(confidence="0.2")
        result = engine.approve_arbitrage(signal)
        assert result is False

    def test_record_trade_profit_resets_consecutive(self, engine):
        engine._consecutive_losses = 5
        engine.record_trade_result(Decimal('1.0'))  # Profit
        assert engine._consecutive_losses == 0
        assert engine._daily_pnl == Decimal('1.0')

    def test_record_trade_loss_increments_consecutive(self, engine):
        engine.record_trade_result(Decimal('-0.5'))
        assert engine._consecutive_losses == 1
        assert engine._daily_pnl == Decimal('-0.5')

    def test_record_one_sided_event(self, engine):
        engine.record_trade_result(Decimal('-10'), one_sided=True)
        assert engine._one_sided_exposure == Decimal('10')

    def test_one_sided_exposure_triggers_halt(self, engine):
        engine.max_one_sided_exposure_usd = Decimal('200')
        engine.record_trade_result(Decimal('-250'), one_sided=True)
        assert engine._halted is True
        assert "One-sided" in engine._halt_reason

    def test_reset_halt(self, engine):
        engine._halted = True
        engine._halt_reason = "Test"
        engine._consecutive_losses = 5

        engine.reset_halt()

        assert engine._halted is False
        assert engine._halt_reason == ""
        assert engine._consecutive_losses == 0

    def test_daily_reset_at_midnight(self, engine):
        # Simulate yesterday's PnL
        engine._daily_pnl = Decimal('-50')
        engine._daily_pnl_reset = datetime.utcnow() - timedelta(days=1)

        # Should reset on next check
        engine._check_daily_reset()

        assert engine._daily_pnl == Decimal('0')

    def test_daily_reset_auto_clears_daily_loss_halt(self, engine):
        engine._halted = True
        engine._halt_reason = "Daily loss limit exceeded"
        engine._daily_pnl_reset = datetime.utcnow() - timedelta(days=1)

        engine._check_daily_reset()

        assert engine._halted is False

    def test_approve_funding_arb(self, engine):
        result = engine.approve_funding_arb("BTC/USDT", Decimal('1500'))
        assert result is True

    def test_reject_funding_arb_exceeds_limit(self, engine):
        result = engine.approve_funding_arb("BTC/USDT", Decimal('3000'))
        assert result is False  # > max_funding_position_usd (2000)

    def test_reject_funding_arb_when_halted(self, engine):
        engine._halted = True
        result = engine.approve_funding_arb("BTC/USDT", Decimal('1000'))
        assert result is False

    def test_reject_funding_arb_exposure_limit(self, engine):
        engine._current_exposure = {"mexc": Decimal('4500')}
        result = engine.approve_funding_arb("BTC/USDT", Decimal('1000'))
        # 4500 + 1000 > 5000
        assert result is False

    def test_update_exposure(self, engine):
        engine.update_exposure("mexc", Decimal('500'))
        assert engine._current_exposure["mexc"] == Decimal('500')

    def test_get_status(self, engine):
        engine._daily_pnl = Decimal('-20')
        engine._consecutive_losses = 3
        status = engine.get_status()
        assert status["halted"] is False
        assert status["daily_pnl_usd"] == -20.0
        assert status["consecutive_losses"] == 3
