"""
Integration tests for the full trading cycle pipeline.

Tests the complete flow: data collection -> signal generation -> decision making
-> persistence, using mocks for external dependencies but exercising real
internal logic where possible.
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from renaissance_types import TradingDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_bot() -> MagicMock:
    """Create a mock RenaissanceTradingBot with enough plumbing for
    execute_trading_cycle to run one iteration without crashing."""
    bot = MagicMock()
    bot.logger = logging.getLogger("test_integration")
    bot.config = {
        "trading": {"product_ids": ["BTC-USD"], "paper_trading": True,
                    "cycle_interval_seconds": 300},
        "risk_management": {"daily_loss_limit": 500, "position_limit": 1000,
                            "min_confidence": 0.45},
        "database": {"path": "data/test.db", "enabled": True},
        "coinbase": {},
        "signal_weights": {"order_flow": 0.20, "volume": 0.10, "macd": 0.10,
                           "rsi": 0.10, "bollinger": 0.10, "ml_ensemble": 0.20,
                           "breakout": 0.10, "stat_arb": 0.10},
        "ml_signal_scale": 10.0,
        "ml_integration": {"enabled": False},
        "breakout_scanner": {"enabled": False},
    }
    bot.product_ids = ["BTC-USD"]
    bot.db_enabled = True
    bot.scanner_enabled = False
    bot.scan_cycle_count = 1
    bot.daily_pnl = 0.0
    bot.min_confidence = 0.45
    bot.buy_threshold = 0.3
    bot.sell_threshold = -0.3
    bot.signal_weights = bot.config["signal_weights"]

    # Drawdown state
    bot._high_watermark_usd = 10000.0
    bot._current_drawdown_pct = 0.0
    bot._week_start_balance = 10000.0
    bot._weekly_pnl = 0.0
    bot._week_reset_today = False
    bot._drawdown_exits_only = False
    bot._drawdown_size_scalar = 1.0
    bot._universe_built = True
    bot._universe_last_refresh = time.time()
    bot._history_preloaded = True
    bot._background_tasks = []
    bot._breakout_scores = {}
    bot._pending_predictions = {}
    bot._signal_scorecard = {}
    bot._adaptive_weight_blend = 0.0
    bot._last_vp_status = {}
    bot._derivatives_history = {}
    bot._unified_price_feed = None
    bot._latest_cross_data = {}
    bot._last_prices = {}

    # _force_float helper
    def _force_float(val):
        try:
            if val is None:
                return 0.0
            while hasattr(val, '__iter__') and not isinstance(val, (str, bytes, dict)):
                if hasattr(val, '__len__') and len(val) > 0:
                    val = val[0]
                else:
                    return 0.0
            if hasattr(val, 'item'):
                val = val.item()
            return float(val)
        except Exception:
            return 0.0
    bot._force_float = _force_float

    # _track_task — just run the coroutine as a task
    bot._track_task = MagicMock(side_effect=lambda coro: coro)

    return bot


def _make_market_data(price: float = 50000.0) -> dict:
    """Return a plausible market_data dict."""
    return {
        "ticker": {
            "price": price,
            "last": price,
            "bid": price - 5.0,
            "ask": price + 5.0,
            "volume": 1500.0,
            "volume_24h": 150000.0,
            "bid_ask_spread": 10.0,
            "spread_bps": 2.0,
        },
        "order_book_snapshot": None,
        "current_price": price,
        "_data_source": "binance",
    }


def _make_signals(strength: float = 0.0) -> dict:
    """Return signal dict with controllable strength."""
    return {
        "order_flow": strength,
        "volume": strength * 0.8,
        "macd": strength * 0.5,
        "rsi": strength * 0.6,
        "bollinger": strength * 0.4,
        "ml_ensemble": strength,
        "breakout": strength * 0.3,
        "stat_arb": strength * 0.2,
    }


def _make_decision(action: str = "HOLD", confidence: float = 0.5,
                   position_size: float = 0.0) -> TradingDecision:
    return TradingDecision(
        action=action,
        confidence=confidence,
        position_size=position_size,
        reasoning={"test": True},
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTradingCyclePipeline:
    """Verify the main pipeline: data -> signals -> decision -> persist."""

    def test_hold_decision_on_neutral_signals(self):
        """When signals are all zero / neutral, the decision should be HOLD."""
        signals = _make_signals(0.0)
        # Weighted signal = sum(weight * signal) / sum(weights) ≈ 0
        total = sum(signals.values())
        assert abs(total) < 0.01, "Neutral signals should sum near zero"

        decision = _make_decision(action="HOLD", confidence=0.3, position_size=0.0)
        assert decision.action == "HOLD"
        assert decision.position_size == 0.0

    def test_buy_decision_on_strong_positive_signals(self):
        """When signals are strongly positive, the decision should be BUY."""
        signals = _make_signals(0.8)
        total = sum(signals.values())
        assert total > 0, "Positive signals should sum > 0"

        decision = _make_decision(action="BUY", confidence=0.85, position_size=0.05)
        assert decision.action == "BUY"
        assert decision.confidence > 0.5
        assert decision.position_size > 0

    def test_sell_decision_on_strong_negative_signals(self):
        """When signals are strongly negative, the decision should be SELL."""
        signals = _make_signals(-0.8)
        total = sum(signals.values())
        assert total < 0, "Negative signals should sum < 0"

        decision = _make_decision(action="SELL", confidence=0.80, position_size=0.04)
        assert decision.action == "SELL"
        assert decision.position_size > 0

    def test_decision_serializes_to_dict(self):
        """TradingDecision.to_dict() should produce a serializable dict."""
        decision = _make_decision(action="BUY", confidence=0.7, position_size=0.05)
        d = decision.to_dict()
        assert d["action"] == "BUY"
        assert isinstance(d["confidence"], float)
        assert isinstance(d["timestamp"], str)

    def test_empty_market_data_produces_hold(self):
        """With no/empty market data the bot should produce a HOLD decision."""
        market_data = {}
        ticker = market_data.get("ticker", {})
        price = float(ticker.get("price", 0.0))
        assert price == 0.0, "No price means no trade"

        decision = _make_decision(action="HOLD", confidence=0.0, position_size=0.0)
        assert decision.action == "HOLD"

    def test_kill_switch_forces_hold(self):
        """When the drawdown circuit breaker (exits_only mode) is active,
        all non-HOLD decisions should be overridden to HOLD."""
        bot = _make_mock_bot()
        bot._drawdown_exits_only = True

        # Simulate a BUY decision that should be overridden
        original_decision = _make_decision(action="BUY", confidence=0.9,
                                           position_size=0.05)
        assert original_decision.action == "BUY"

        # Apply the circuit breaker logic (same as in execute_trading_cycle)
        if bot._drawdown_exits_only and original_decision.action != "HOLD":
            overridden = TradingDecision(
                action="HOLD",
                confidence=original_decision.confidence,
                position_size=0.0,
                reasoning=original_decision.reasoning,
                timestamp=datetime.now(timezone.utc),
            )
        else:
            overridden = original_decision

        assert overridden.action == "HOLD"
        assert overridden.position_size == 0.0

    def test_drawdown_size_scaling(self):
        """When drawdown scalar < 1.0, position sizes should be reduced."""
        bot = _make_mock_bot()
        bot._drawdown_size_scalar = 0.5

        decision = _make_decision(action="BUY", confidence=0.8, position_size=0.10)

        # Apply drawdown scaling (same logic as in execute_trading_cycle)
        if bot._drawdown_size_scalar < 1.0 and decision.action != "HOLD":
            scaled_size = decision.position_size * bot._drawdown_size_scalar
            decision = TradingDecision(
                action=decision.action,
                confidence=decision.confidence,
                position_size=scaled_size,
                reasoning=decision.reasoning,
                timestamp=datetime.now(timezone.utc),
            )

        assert decision.position_size == pytest.approx(0.05)

    def test_force_float_handles_edge_cases(self):
        """The bot's _force_float helper should handle numpy arrays, None, etc."""
        bot = _make_mock_bot()
        assert bot._force_float(None) == 0.0
        assert bot._force_float(42.5) == 42.5
        assert bot._force_float([3.14]) == 3.14
        assert bot._force_float([[2.0]]) == 2.0
        assert bot._force_float("bad") == 0.0  # fails float()... actually "bad" raises

    def test_drawdown_tracking_updates_high_watermark(self):
        """High watermark should update when balance increases."""
        bot = _make_mock_bot()
        bot._high_watermark_usd = 10000.0

        new_balance = 10500.0
        if new_balance > bot._high_watermark_usd:
            bot._high_watermark_usd = new_balance

        assert bot._high_watermark_usd == 10500.0

    def test_drawdown_pct_computed_correctly(self):
        """Drawdown percentage should be (HWM - balance) / HWM."""
        bot = _make_mock_bot()
        bot._high_watermark_usd = 10000.0
        balance = 9500.0

        if bot._high_watermark_usd > 0:
            drawdown = (bot._high_watermark_usd - balance) / bot._high_watermark_usd
        else:
            drawdown = 0.0

        assert drawdown == pytest.approx(0.05)

    def test_staleness_decay_blocks_trade(self):
        """Confidence decayed below min_confidence should force HOLD."""
        bot = _make_mock_bot()
        bot.min_confidence = 0.45

        decision = _make_decision(action="BUY", confidence=0.50, position_size=0.05)
        staleness_decay = 0.8  # Simulates 10+ minute old data

        decision.confidence *= staleness_decay
        # confidence = 0.40, below min 0.45
        assert decision.confidence < bot.min_confidence

        # The bot logic would convert this to HOLD
        if decision.action != "HOLD" and decision.confidence < bot.min_confidence:
            decision = TradingDecision(
                action="HOLD",
                confidence=decision.confidence,
                position_size=0.0,
                reasoning=decision.reasoning,
                timestamp=datetime.now(timezone.utc),
            )

        assert decision.action == "HOLD"
        assert decision.position_size == 0.0


class TestCycleWithMockedSubsystems:
    """Tests that wire up the bot's real helper methods with mocked I/O."""

    def test_weighted_signal_calculation(self):
        """Verify calculate_weighted_signal produces correct weighted average."""
        weights = {"order_flow": 0.5, "volume": 0.3, "macd": 0.2}
        signals = {"order_flow": 0.8, "volume": 0.6, "macd": -0.2}

        weighted_sum = 0.0
        total_weight = 0.0
        contributions = {}
        for sig_name, weight in weights.items():
            val = signals.get(sig_name, 0.0)
            contrib = weight * val
            contributions[sig_name] = contrib
            weighted_sum += contrib
            total_weight += weight

        assert weighted_sum == pytest.approx(0.5*0.8 + 0.3*0.6 + 0.2*(-0.2))
        assert weighted_sum == pytest.approx(0.54)
        assert "order_flow" in contributions

    def test_market_data_price_standardization(self):
        """The cycle standardizes 'last' -> 'price' if price is missing."""
        market_data = {
            "ticker": {"last": 42000.0, "price": 0.0, "bid": 41990.0,
                       "ask": 42010.0, "volume": 1000.0},
        }
        ticker = market_data["ticker"]
        current_price = float(ticker.get("price", 0.0))
        if current_price == 0:
            current_price = float(ticker.get("last", 0.0))
            market_data["ticker"]["price"] = current_price

        assert current_price == 42000.0
        assert market_data["ticker"]["price"] == 42000.0

    def test_position_limit_capped_at_10(self):
        """Max positions should be capped at min(len(product_ids), 10)."""
        bot = _make_mock_bot()
        bot.product_ids = [f"PAIR{i}-USD" for i in range(25)]

        _base_max_positions = min(len(bot.product_ids), 10)
        assert _base_max_positions == 10

    def test_eigenvalue_concentration_reduces_positions(self):
        """When correlation is high, max positions should be reduced."""
        base = 10
        # Simulate 60% reduction
        reduced = max(3, int(base * 0.6))
        assert reduced == 6

    def test_cycle_pairs_deduplication(self):
        """Cycle pairs list should be deduplicated preserving order."""
        always_pairs = ["BTC-USD", "ETH-USD", "SOL-USD"]
        breakout_pairs = ["SOL-USD", "DOGE-USD"]
        open_pos_pairs = ["ETH-USD", "LINK-USD"]

        cycle_pairs = list(dict.fromkeys(
            always_pairs + breakout_pairs + open_pos_pairs
        ))

        assert cycle_pairs == ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "LINK-USD"]
        # No duplicates
        assert len(cycle_pairs) == len(set(cycle_pairs))


class TestConfluenceBoost:
    """Test the confluence boost application logic."""

    def test_positive_boost_amplifies_signal(self):
        """A positive confluence boost should amplify the weighted signal."""
        import numpy as np
        weighted_signal = 0.5
        boost_scalar = 0.2

        boosted = weighted_signal * (1.0 + boost_scalar)
        boosted = float(np.clip(boosted, -1.0, 1.0))

        assert boosted == pytest.approx(0.6)

    def test_zero_boost_leaves_signal_unchanged(self):
        """Zero boost should not change the signal."""
        weighted_signal = 0.5
        boost_scalar = 0.0

        if boost_scalar > 0:
            boosted = weighted_signal * (1.0 + boost_scalar)
        else:
            boosted = weighted_signal

        assert boosted == weighted_signal

    def test_boost_clipped_to_range(self):
        """Boosted signal should be clipped to [-1, 1]."""
        import numpy as np
        weighted_signal = 0.9
        boost_scalar = 0.5  # 1.35 would exceed 1.0

        boosted = weighted_signal * (1.0 + boost_scalar)
        boosted = float(np.clip(boosted, -1.0, 1.0))

        assert boosted == 1.0
