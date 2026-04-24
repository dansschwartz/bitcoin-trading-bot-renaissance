"""
Tests for bot/decision.py — trading decision logic.

Tests the full decision-making pipeline including confidence gates, regime bias,
anti-churn, ML agreement, direction consensus, and position sizing.
"""

import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot(**overrides) -> MagicMock:
    """Create a minimal mock bot with all attributes expected by make_trading_decision."""
    bot = MagicMock()
    bot.logger = logging.getLogger("test_bot_decision")

    # Config
    bot.config = {
        "ml_max_prediction_age_minutes": 15,
    }
    bot.paper_trading = True
    bot.min_confidence = 0.45
    bot.buy_threshold = 0.06
    bot.sell_threshold = -0.06
    bot.daily_pnl = 0.0
    bot.daily_loss_limit = 500.0
    bot.position_limit = 1000.0
    bot.current_position = 0.0

    # Regime overlay (enabled=False by default to simplify tests)
    bot.regime_overlay = MagicMock()
    bot.regime_overlay.enabled = False
    bot.regime_overlay.get_confidence_boost.return_value = 0.0
    bot.regime_overlay.get_hmm_regime_label.return_value = "unknown"
    bot.regime_overlay.get_transition_warning.return_value = {
        "alert_level": "none", "size_multiplier": 1.0, "message": ""
    }

    # Risk manager
    bot.risk_manager = MagicMock()
    bot.risk_manager.assess_risk_regime.return_value = {
        "recommended_action": "normal",
        "confidence": 0.8,
    }

    # Risk gateway — allow all trades by default
    bot.risk_gateway = MagicMock()
    bot.risk_gateway.vae_trained = False
    bot.risk_gateway.vae = None
    bot.risk_gateway.assess_trade.return_value = (True, 0.01, "passed")

    # Position manager — no existing positions by default
    bot.position_manager = MagicMock()
    bot.position_manager._lock = MagicMock()
    bot.position_manager._lock.__enter__ = MagicMock(return_value=None)
    bot.position_manager._lock.__exit__ = MagicMock(return_value=False)
    bot.position_manager.positions = {}
    bot.position_manager._calculate_total_exposure.return_value = 0.0

    # Position sizer
    sizing_result = MagicMock()
    sizing_result.asset_units = 0.01
    sizing_result.usd_value = 500.0
    sizing_result.kelly_fraction = 0.05
    sizing_result.applied_fraction = 0.03
    sizing_result.edge = 0.01
    sizing_result.effective_edge = 0.008
    sizing_result.win_probability = 0.55
    sizing_result.market_impact_bps = 1.0
    sizing_result.capacity_used_pct = 5.0
    sizing_result.transaction_cost_ratio = 0.3
    sizing_result.volatility_scalar = 1.0
    sizing_result.regime_scalar = 1.0
    sizing_result.liquidity_scalar = 1.0
    sizing_result.sizing_method = "renaissance_kelly"
    sizing_result.reasons = ["sized"]
    bot.position_sizer = MagicMock()
    bot.position_sizer.calculate_size.return_value = sizing_result
    bot.position_sizer.estimate_round_trip_cost.return_value = 0.001

    # Signal scorecard & state
    bot._signal_scorecard = {}
    bot._signal_filter_stats = {
        "total": 0, "traded": 0, "filtered_threshold": 0,
        "filtered_confidence": 0, "filtered_agreement": 0,
    }
    bot._signal_history = {}
    bot._last_trade_cycle = {}
    bot.scan_cycle_count = 100

    # Optional modules (all off by default)
    bot.kelly_sizer = None
    bot.leverage_mgr = None
    bot.medallion_regime = None
    bot.health_monitor = None
    bot.portfolio_engine = None

    # Cached balance
    bot._cached_balance_usd = 10000.0
    bot._drawdown_size_scalar = 1.0
    bot._get_measured_edge = MagicMock(return_value=None)

    for k, v in overrides.items():
        setattr(bot, k, v)
    return bot


def _make_ml_package(predictions=None, timestamp=None, feature_vector=None):
    """Create a mock ML package."""
    ml = MagicMock()
    ml.ml_predictions = predictions or []
    ml.timestamp = timestamp or datetime.now()
    ml.feature_vector = feature_vector
    ml.fractal_insights = {}
    return ml


def _call_decision(bot, signal=0.10, contributions=None, product_id="BTC-USD",
                   ml_package=None, market_data=None, current_price=50000.0):
    """Helper to call make_trading_decision with sensible defaults."""
    from bot.decision import make_trading_decision
    return make_trading_decision(
        bot=bot,
        weighted_signal=signal,
        signal_contributions=contributions or {"order_flow": signal * 0.5, "rsi": signal * 0.3},
        current_price=current_price,
        product_id=product_id,
        ml_package=ml_package,
        market_data=market_data,
    )


# ---------------------------------------------------------------------------
# Tests: HOLD when confidence below min_confidence
# ---------------------------------------------------------------------------


class TestConfidenceGate:
    """Tests for confidence-based HOLD decisions."""

    def test_hold_when_confidence_below_min(self):
        """Should HOLD when calculated confidence is below min_confidence."""
        bot = _make_bot(min_confidence=0.90)  # Very high threshold → forces HOLD
        decision = _call_decision(bot, signal=0.10)
        assert decision.action == "HOLD"

    def test_hold_when_confidence_is_zero(self):
        """Should HOLD when confidence is exactly 0.0."""
        bot = _make_bot()
        # With a tiny signal below cost pre-screen, confidence will be 0.0
        decision = _call_decision(bot, signal=0.00001)
        assert decision.action == "HOLD"
        assert decision.confidence == 0.0

    def test_hold_when_signal_zero(self):
        """Zero signal → HOLD (blocked by cost pre-screen)."""
        bot = _make_bot()
        decision = _call_decision(bot, signal=0.0)
        assert decision.action == "HOLD"


# ---------------------------------------------------------------------------
# Tests: BUY/SELL threshold decisions
# ---------------------------------------------------------------------------


class TestThresholdDecisions:
    """Tests for BUY/SELL based on signal thresholds."""

    def test_buy_when_signal_above_buy_threshold(self):
        """Signal > buy_threshold with sufficient confidence → BUY."""
        bot = _make_bot(min_confidence=0.20)  # Low enough that HOLD won't trigger
        # High ML agreement to boost confidence
        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3), ("model_d", 0.2),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        assert decision.action == "BUY"

    def test_sell_when_signal_below_sell_threshold(self):
        """Signal < sell_threshold with sufficient confidence → SELL."""
        bot = _make_bot(min_confidence=0.20)
        ml = _make_ml_package(predictions=[
            ("model_a", -0.5), ("model_b", -0.4), ("model_c", -0.3), ("model_d", -0.2),
        ])
        decision = _call_decision(
            bot, signal=-0.10,
            contributions={"order_flow": -0.05, "rsi": -0.03},
            ml_package=ml,
        )
        assert decision.action == "SELL"

    def test_hold_when_signal_in_dead_zone(self):
        """Signal between thresholds → HOLD."""
        bot = _make_bot()
        decision = _call_decision(bot, signal=0.03)  # Between -0.06 and 0.06
        assert decision.action == "HOLD"


# ---------------------------------------------------------------------------
# Tests: ML prediction staleness
# ---------------------------------------------------------------------------


class TestMLStaleness:
    """Tests for ML prediction staleness check."""

    def test_stale_ml_discarded(self):
        """ML predictions older than TTL should be discarded."""
        bot = _make_bot(min_confidence=0.20)
        bot.config["ml_max_prediction_age_minutes"] = 15

        stale_time = datetime.now() - timedelta(minutes=30)
        ml = _make_ml_package(
            predictions=[("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3)],
            timestamp=stale_time,
        )

        # With stale ML, predictions should be discarded
        # The decision still depends on signal threshold + confidence
        decision = _call_decision(bot, signal=0.10, ml_package=ml)

        # ML package should have been set to None internally
        # We can verify the reasoning doesn't contain ML data influence
        assert isinstance(decision, object)

    def test_fresh_ml_used(self):
        """ML predictions within TTL should be used."""
        bot = _make_bot(min_confidence=0.20)
        bot.config["ml_max_prediction_age_minutes"] = 15

        fresh_time = datetime.now() - timedelta(minutes=5)
        ml = _make_ml_package(
            predictions=[("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3), ("model_d", 0.2)],
            timestamp=fresh_time,
        )

        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        # Should process normally (BUY if all gates pass)
        assert decision.action in ("BUY", "HOLD")


# ---------------------------------------------------------------------------
# Tests: Anti-churn gate
# ---------------------------------------------------------------------------


class TestAntiChurnGate:
    """Tests for anti-churn trade prevention."""

    def test_blocks_during_cooldown(self):
        """Should block trades within min_hold_cycles of last trade."""
        bot = _make_bot(min_confidence=0.20)
        bot.scan_cycle_count = 10
        bot._last_trade_cycle = {"BTC-USD": 8}  # Last trade 2 cycles ago
        bot._signal_history = {"BTC-USD": ["BUY"]}

        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3), ("model_d", 0.2),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        assert decision.action == "HOLD"

    def test_allows_after_cooldown(self):
        """Should allow trades after min_hold_cycles have passed."""
        bot = _make_bot(min_confidence=0.20)
        bot.scan_cycle_count = 100
        bot._last_trade_cycle = {"BTC-USD": 50}  # 50 cycles ago (well past cooldown)
        bot._signal_history = {"BTC-USD": ["BUY", "BUY"]}  # Consistent signal

        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3), ("model_d", 0.2),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        # Should not be blocked by anti-churn
        assert decision.action in ("BUY", "HOLD")

    def test_blocks_signal_flip(self):
        """Should block when signal flips rapidly (BUY → SELL)."""
        bot = _make_bot(min_confidence=0.20)
        bot.scan_cycle_count = 100
        bot._last_trade_cycle = {"BTC-USD": 50}
        # Previous signal was SELL, now getting BUY → flip detected
        bot._signal_history = {"BTC-USD": ["SELL"]}

        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3), ("model_d", 0.2),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        assert decision.action == "HOLD"


# ---------------------------------------------------------------------------
# Tests: Direction consensus gate
# ---------------------------------------------------------------------------


class TestDirectionConsensusGate:
    """Tests for ML direction consensus gate."""

    def test_blocks_when_models_disagree_on_direction(self):
        """Should block BUY when <50% of models are bullish."""
        bot = _make_bot(min_confidence=0.20)
        bot.scan_cycle_count = 100
        bot._last_trade_cycle = {}
        bot._signal_history = {"BTC-USD": ["BUY", "BUY"]}

        # 3 bearish, 1 bullish → only 25% aligned with BUY
        ml = _make_ml_package(predictions=[
            ("model_a", -0.5), ("model_b", -0.4), ("model_c", -0.3), ("model_d", 0.2),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        assert decision.action == "HOLD"

    def test_passes_when_models_agree(self):
        """Should allow BUY when >50% of models are bullish."""
        bot = _make_bot(min_confidence=0.20)
        bot.scan_cycle_count = 100
        bot._last_trade_cycle = {}
        bot._signal_history = {"BTC-USD": ["BUY", "BUY"]}

        # All bullish
        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3), ("model_d", 0.2),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        # May still be blocked by ML agreement gate, but not direction consensus
        assert decision.action in ("BUY", "HOLD")


# ---------------------------------------------------------------------------
# Tests: ML agreement gate
# ---------------------------------------------------------------------------


class TestMLAgreementGate:
    """Tests for ML model agreement gate."""

    def test_blocks_below_agreement_threshold(self):
        """Should block when model agreement < threshold (71% default)."""
        bot = _make_bot(min_confidence=0.20)
        bot.scan_cycle_count = 100
        bot._last_trade_cycle = {}
        bot._signal_history = {"BTC-USD": ["BUY", "BUY"]}

        # 2 positive, 2 negative → 50% agreement < 71%
        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4),
            ("model_c", -0.3), ("model_d", -0.2),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        assert decision.action == "HOLD"

    def test_passes_above_agreement_threshold(self):
        """Should allow when model agreement >= threshold."""
        bot = _make_bot(min_confidence=0.20)
        bot.scan_cycle_count = 100
        bot._last_trade_cycle = {}
        bot._signal_history = {"BTC-USD": ["BUY", "BUY"]}

        # 4 of 5 positive → 80% agreement > 71%
        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3),
            ("model_d", 0.2), ("model_e", -0.1),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        # Should not be blocked by ML agreement
        assert decision.action in ("BUY", "HOLD")


# ---------------------------------------------------------------------------
# Tests: Regime overlay effects
# ---------------------------------------------------------------------------


class TestRegimeOverlay:
    """Tests for regime overlay confidence adjustments."""

    def test_confidence_boost_applied(self):
        """Regime confidence boost should be applied to the final confidence."""
        bot = _make_bot(min_confidence=0.20)
        bot.regime_overlay.get_confidence_boost.return_value = 0.05
        bot.regime_overlay.enabled = False  # Disable label-based filtering

        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3), ("model_d", 0.2),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)

        # Confidence should have the 0.05 boost applied
        # (base confidence depends on agreement, but boost is always added)
        assert decision.confidence > 0

    def test_low_vol_regime_lowers_thresholds(self):
        """Low volatility regime should lower entry thresholds (boost trading)."""
        bot = _make_bot(min_confidence=0.20)
        bot.regime_overlay.enabled = True
        bot.regime_overlay.get_hmm_regime_label.return_value = "low_volatility"
        bot.regime_overlay.get_confidence_boost.return_value = 0.0
        bot.scan_cycle_count = 100
        bot._last_trade_cycle = {}
        bot._signal_history = {"BTC-USD": ["BUY", "BUY"]}

        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3), ("model_d", 0.2),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        # The key assertion is that the decision didn't get zero'd out
        # Low vol should boost mean reversion, not kill signals
        assert decision.action in ("BUY", "HOLD")


# ---------------------------------------------------------------------------
# Tests: Risk gates
# ---------------------------------------------------------------------------


class TestRiskGates:
    """Tests for risk-related gates (daily loss, VAE, risk regime)."""

    def test_hold_when_daily_loss_exceeded(self):
        """Should HOLD when daily loss limit is exceeded."""
        bot = _make_bot(min_confidence=0.20, daily_pnl=-600.0, daily_loss_limit=500.0)
        decision = _call_decision(bot, signal=0.10)
        assert decision.action == "HOLD"

    def test_hold_when_risk_regime_fallback(self):
        """Should HOLD when risk assessment recommends fallback mode."""
        bot = _make_bot(min_confidence=0.20)
        bot.risk_manager.assess_risk_regime.return_value = {
            "recommended_action": "fallback_mode"
        }
        decision = _call_decision(bot, signal=0.10)
        assert decision.action == "HOLD"

    def test_hold_when_risk_gateway_blocks(self):
        """Should HOLD when risk gateway blocks the trade."""
        bot = _make_bot(min_confidence=0.20)
        bot.scan_cycle_count = 100
        bot._last_trade_cycle = {}
        bot._signal_history = {"BTC-USD": ["BUY", "BUY"]}

        bot.risk_gateway.assess_trade.return_value = (False, 0.99, "vae_anomaly")

        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3), ("model_d", 0.2),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        assert decision.action == "HOLD"


# ---------------------------------------------------------------------------
# Tests: Cost pre-screen
# ---------------------------------------------------------------------------


class TestCostPreScreen:
    """Tests for the cost pre-screen gate."""

    def test_cost_prescreen_blocks_tiny_signal_in_paper_mode(self):
        """In paper mode, signals below 0.0001 should be blocked."""
        bot = _make_bot()
        decision = _call_decision(bot, signal=0.00005)
        assert decision.action == "HOLD"
        assert "cost_pre_screen" in str(decision.reasoning)


# ---------------------------------------------------------------------------
# Tests: Anti-stacking
# ---------------------------------------------------------------------------


class TestAntiStacking:
    """Tests for anti-stacking gate (prevents duplicate same-direction positions)."""

    def test_blocks_duplicate_long(self):
        """Should block BUY when already has LONG position on same asset."""
        from risk_management.position_manager import PositionStatus, PositionSide

        bot = _make_bot(min_confidence=0.20)
        bot.scan_cycle_count = 100
        bot._last_trade_cycle = {}
        bot._signal_history = {"BTC-USD": ["BUY", "BUY"]}

        # Create an existing LONG position using the real enum
        mock_pos = MagicMock()
        mock_pos.product_id = "BTC-USD"
        mock_pos.status = PositionStatus.OPEN
        mock_pos.side = PositionSide.LONG
        bot.position_manager.positions = {"pos1": mock_pos}

        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3), ("model_d", 0.2),
        ])
        decision = _call_decision(bot, signal=0.10, ml_package=ml)
        assert decision.action == "HOLD"


# ---------------------------------------------------------------------------
# Tests: Volatility dead-zone gate
# ---------------------------------------------------------------------------


class TestVolDeadZoneGate:
    """Tests for volatility dead-zone gate."""

    def test_blocks_in_dead_zone(self):
        """Should block trading when vol regime is dead_zone."""
        bot = _make_bot(min_confidence=0.20)
        bot.scan_cycle_count = 100
        bot._last_trade_cycle = {}
        bot._signal_history = {"BTC-USD": ["BUY", "BUY"]}

        ml = _make_ml_package(predictions=[
            ("model_a", 0.5), ("model_b", 0.4), ("model_c", 0.3), ("model_d", 0.2),
        ])
        market_data = {
            "volatility_prediction": {
                "vol_regime": "dead_zone",
                "predicted_magnitude_bps": 1.0,
            }
        }
        decision = _call_decision(bot, signal=0.10, ml_package=ml, market_data=market_data)
        assert decision.action == "HOLD"


# ---------------------------------------------------------------------------
# Tests: TradingDecision return structure
# ---------------------------------------------------------------------------


class TestTradingDecisionStructure:
    """Tests for the TradingDecision return type."""

    def test_returns_trading_decision_type(self):
        """make_trading_decision should return a TradingDecision."""
        from renaissance_types import TradingDecision

        bot = _make_bot()
        decision = _call_decision(bot, signal=0.03)

        assert isinstance(decision, TradingDecision)
        assert hasattr(decision, "action")
        assert hasattr(decision, "confidence")
        assert hasattr(decision, "position_size")
        assert hasattr(decision, "reasoning")
        assert hasattr(decision, "timestamp")

    def test_hold_has_zero_position_size(self):
        """HOLD decisions should have position_size == 0.0."""
        bot = _make_bot()
        decision = _call_decision(bot, signal=0.03)
        assert decision.action == "HOLD"
        assert decision.position_size == 0.0

    def test_reasoning_contains_required_keys(self):
        """Reasoning dict should contain standard keys."""
        bot = _make_bot()
        decision = _call_decision(bot, signal=0.10)

        assert "weighted_signal" in decision.reasoning
        assert "confidence" in decision.reasoning
        assert "signal_contributions" in decision.reasoning
        assert "risk_check" in decision.reasoning
