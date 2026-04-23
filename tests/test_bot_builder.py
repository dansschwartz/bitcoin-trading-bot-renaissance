"""
Tests for bot/builder.py — BotBuilder component initialization.

Verifies that each build phase correctly constructs and assigns bot attributes,
handles missing optional dependencies gracefully, and validates config.
"""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot(**overrides) -> MagicMock:
    """Create a minimal mock bot with attributes expected by BotBuilder."""
    bot = MagicMock()
    bot.logger = logging.getLogger("test_bot_builder")
    bot.config = {
        "trading": {
            "product_ids": ["BTC-USD", "ETH-USD"],
            "paper_trading": True,
            "sandbox": True,
            "cycle_interval_seconds": 300,
        },
        "database": {"path": "data/test.db", "enabled": True},
        "risk_management": {
            "daily_loss_limit": 500,
            "position_limit": 1000,
            "min_confidence": 0.45,
        },
        "coinbase": {},
        "signal_weights": {
            "order_flow": 0.14,
            "volume": 0.08,
            "macd": 0.05,
            "rsi": 0.05,
            "bollinger": 0.05,
            "alternative": 0.01,
            "stat_arb": 0.084,
            "volume_profile": 0.04,
            "fractal": 0.05,
            "entropy": 0.05,
            "quantum": 0.07,
            "lead_lag": 0.10,
            "garch_vol": 0.06,
            "ml_ensemble": 0.20,
            "ml_cnn": 0.0,
            "breakout": 0.08,
            "crash_regime": 0.15,
        },
        "ml_signal_scale": 10.0,
        "ml_integration": {"enabled": False},
    }
    bot.config_path = "config.yaml"
    bot.product_ids = ["BTC-USD", "ETH-USD"]
    bot.db_enabled = True
    # _force_float helper used by build_signal_weights
    bot._force_float = lambda v: float(v) if v is not None else 0.0
    # _track_task used by build_signal_weights
    bot._track_task = MagicMock(side_effect=lambda coro: coro)
    for k, v in overrides.items():
        setattr(bot, k, v)
    return bot


def _clear_rtb_flags():
    """Clear any availability flags on renaissance_trading_bot module."""
    rtb = sys.modules.get("renaissance_trading_bot")
    if rtb is None:
        return
    flag_names = [
        "ORCHESTRATOR_AVAILABLE", "ARBITRAGE_AVAILABLE", "RECOVERY_AVAILABLE",
        "MONITORING_AVAILABLE", "LIQUIDATION_DETECTOR_AVAILABLE",
        "SIGNAL_AGGREGATOR_AVAILABLE", "MULTI_EXCHANGE_BRIDGE_AVAILABLE",
        "DATA_VALIDATOR_AVAILABLE", "SIGNAL_THROTTLE_AVAILABLE",
        "SIGNAL_VALIDATION_AVAILABLE", "HEALTH_MONITOR_AVAILABLE",
        "MEDALLION_ANALOGS_AVAILABLE", "PORTFOLIO_ENGINE_AVAILABLE",
        "DEVIL_TRACKER_AVAILABLE", "KELLY_SIZER_AVAILABLE",
        "BREAKOUT_STRATEGY_AVAILABLE", "BAR_AGGREGATOR_AVAILABLE",
        "MEDALLION_REGIME_AVAILABLE", "HEARTBEAT_AVAILABLE",
        "HIERARCHICAL_REGIME_AVAILABLE", "BETA_MONITOR_AVAILABLE",
        "CAPACITY_MONITOR_AVAILABLE", "SHARPE_MONITOR_AVAILABLE",
        "INSURANCE_SCANNER_AVAILABLE", "SYNC_EXECUTOR_AVAILABLE",
        "TRADE_HIDER_AVAILABLE", "MHPE_AVAILABLE",
        "POSITION_REEVALUATOR_AVAILABLE", "FAST_REVERSION_AVAILABLE",
        "SPREAD_CAPTURE_AVAILABLE", "SUB_BAR_SCANNER_AVAILABLE",
        "AGENT_COORDINATOR_AVAILABLE", "MEDALLION_THROTTLE_AVAILABLE",
        "LEVERAGE_MANAGER_AVAILABLE", "MEDALLION_PORTFOLIO_ENGINE_AVAILABLE",
    ]
    for name in flag_names:
        if hasattr(rtb, name):
            setattr(rtb, name, False)


# ---------------------------------------------------------------------------
# Tests: build_data_layer
# ---------------------------------------------------------------------------


class TestBuildDataLayer:
    """Tests for BotBuilder.build_data_layer."""

    def test_creates_per_asset_tech_indicators(self):
        """build_data_layer should create a tech indicator instance per product_id."""
        bot = _make_bot()
        mock_eti = MagicMock()

        with patch.dict("sys.modules", {
            "enhanced_config_manager": MagicMock(),
            "microstructure_engine": MagicMock(),
            "enhanced_technical_indicators": MagicMock(EnhancedTechnicalIndicators=mock_eti),
            "market_data_provider": MagicMock(),
            "derivatives_data_provider": MagicMock(),
            "renaissance_signal_fusion": MagicMock(),
            "alternative_data_engine": MagicMock(),
            "renaissance_engine_core": MagicMock(),
        }):
            from bot.builder import BotBuilder
            BotBuilder.build_data_layer(bot)

        # Should have created indicators for each product_id
        assert hasattr(bot, "_tech_indicators")
        assert "BTC-USD" in bot._tech_indicators
        assert "ETH-USD" in bot._tech_indicators
        assert len(bot._tech_indicators) == 2

    def test_sets_signal_fusion_ml_scale(self):
        """build_data_layer should set ml_signal_scale on signal_fusion."""
        bot = _make_bot()
        mock_signal_fusion = MagicMock()

        with patch.dict("sys.modules", {
            "enhanced_config_manager": MagicMock(),
            "microstructure_engine": MagicMock(),
            "enhanced_technical_indicators": MagicMock(EnhancedTechnicalIndicators=MagicMock()),
            "market_data_provider": MagicMock(),
            "derivatives_data_provider": MagicMock(),
            "renaissance_signal_fusion": MagicMock(),
            "alternative_data_engine": MagicMock(),
            "renaissance_engine_core": MagicMock(SignalFusion=lambda: mock_signal_fusion),
        }):
            from bot.builder import BotBuilder
            BotBuilder.build_data_layer(bot)

        mock_signal_fusion.set_ml_signal_scale.assert_called_once_with(10.0)

    def test_initializes_derivatives_history(self):
        """build_data_layer should create empty _derivatives_history dict."""
        bot = _make_bot()

        with patch.dict("sys.modules", {
            "enhanced_config_manager": MagicMock(),
            "microstructure_engine": MagicMock(),
            "enhanced_technical_indicators": MagicMock(EnhancedTechnicalIndicators=MagicMock()),
            "market_data_provider": MagicMock(),
            "derivatives_data_provider": MagicMock(),
            "renaissance_signal_fusion": MagicMock(),
            "alternative_data_engine": MagicMock(),
            "renaissance_engine_core": MagicMock(),
        }):
            from bot.builder import BotBuilder
            BotBuilder.build_data_layer(bot)

        assert bot._derivatives_history == {}


# ---------------------------------------------------------------------------
# Tests: build_signal_layer
# ---------------------------------------------------------------------------


class TestBuildSignalLayer:
    """Tests for BotBuilder.build_signal_layer."""

    def _mock_signal_modules(self):
        """Return dict of mocked modules needed by build_signal_layer."""
        return {
            "volume_profile_engine": MagicMock(),
            "fractal_intelligence": MagicMock(),
            "market_entropy_engine": MagicMock(),
            "quantum_oscillator_engine": MagicMock(),
            "ghost_runner": MagicMock(),
            "self_reinforcing_learning": MagicMock(),
            "confluence_engine": MagicMock(),
            "basis_trading_engine": MagicMock(),
            "deep_nlp_bridge": MagicMock(),
            "market_making_engine": MagicMock(),
            "meta_strategy_selector": MagicMock(),
            "genetic_optimizer": MagicMock(),
            "cross_asset_engine": MagicMock(),
            "whale_activity_monitor": MagicMock(),
            "breakout_scanner": MagicMock(BreakoutScanner=MagicMock(), BreakoutSignal=MagicMock()),
            "advanced_mean_reversion_engine": MagicMock(),
            "correlation_network_engine": MagicMock(),
            "garch_volatility_engine": MagicMock(),
            "historical_data_cache": MagicMock(),
            "statistical_arbitrage_engine": MagicMock(),
            "random_baseline": MagicMock(),
        }

    def test_creates_signal_scorecard(self):
        """build_signal_layer should initialize an empty signal scorecard."""
        bot = _make_bot()
        _clear_rtb_flags()

        with patch.dict("sys.modules", self._mock_signal_modules()):
            from bot.builder import BotBuilder
            BotBuilder.build_signal_layer(bot)

        assert bot._signal_scorecard == {}

    def test_sets_ml_eval_params(self):
        """build_signal_layer should read ML evaluation config."""
        bot = _make_bot()
        bot.config["ml_evaluation"] = {
            "min_predictions_for_edge": 100,
            "edge_blend_measured": 0.7,
            "edge_blend_model": 0.3,
        }
        _clear_rtb_flags()

        with patch.dict("sys.modules", self._mock_signal_modules()):
            from bot.builder import BotBuilder
            BotBuilder.build_signal_layer(bot)

        assert bot._ml_eval_min_predictions == 100
        assert bot._ml_eval_blend_measured == pytest.approx(0.7)
        assert bot._ml_eval_blend_model == pytest.approx(0.3)

    def test_data_validator_none_when_flag_off(self):
        """build_signal_layer should set data_validator=None when flag is False."""
        bot = _make_bot()
        _clear_rtb_flags()

        with patch.dict("sys.modules", self._mock_signal_modules()):
            from bot.builder import BotBuilder
            BotBuilder.build_signal_layer(bot)

        assert bot.data_validator is None

    def test_portfolio_engine_none_when_flag_off(self):
        """build_signal_layer should set portfolio_engine=None when flag is False."""
        bot = _make_bot()
        _clear_rtb_flags()

        with patch.dict("sys.modules", self._mock_signal_modules()):
            from bot.builder import BotBuilder
            BotBuilder.build_signal_layer(bot)

        assert bot.portfolio_engine is None


# ---------------------------------------------------------------------------
# Tests: build_risk_layer
# ---------------------------------------------------------------------------


class TestBuildRiskLayer:
    """Tests for BotBuilder.build_risk_layer."""

    def test_sets_risk_parameters(self):
        """build_risk_layer should set daily_loss_limit, position_limit, min_confidence."""
        bot = _make_bot()

        with patch.dict("sys.modules", {
            "regime_overlay": MagicMock(),
            "risk_gateway": MagicMock(),
            "real_time_pipeline": MagicMock(),
            "renaissance_engine_core": MagicMock(),
        }):
            from bot.builder import BotBuilder
            BotBuilder.build_risk_layer(bot)

        assert bot.daily_loss_limit == 500.0
        assert bot.position_limit == 1000.0
        assert bot.min_confidence == 0.45

    def test_creates_regime_overlay(self):
        """build_risk_layer should create a regime_overlay."""
        bot = _make_bot()
        mock_regime = MagicMock()
        mock_regime_mod = MagicMock(RegimeOverlay=lambda *a, **kw: mock_regime)

        with patch.dict("sys.modules", {
            "regime_overlay": mock_regime_mod,
            "risk_gateway": MagicMock(),
            "real_time_pipeline": MagicMock(),
            "renaissance_engine_core": MagicMock(),
        }):
            from bot.builder import BotBuilder
            BotBuilder.build_risk_layer(bot)

        assert bot.regime_overlay is mock_regime


# ---------------------------------------------------------------------------
# Tests: build_execution_layer
# ---------------------------------------------------------------------------


class TestBuildExecutionLayer:
    """Tests for BotBuilder.build_execution_layer."""

    def _mock_execution_modules(self):
        return {
            "execution_algorithm_suite": MagicMock(),
            "slippage_protection_system": MagicMock(),
            "coinbase_client": MagicMock(
                EnhancedCoinbaseClient=MagicMock(),
                CoinbaseCredentials=MagicMock(),
            ),
            "position_manager": MagicMock(
                EnhancedPositionManager=MagicMock(),
                RiskLimits=MagicMock(),
            ),
            "position_sizer": MagicMock(),
            "alert_manager": MagicMock(),
            "database_manager": MagicMock(),
            "decision_audit_logger": MagicMock(),
            "performance_attribution_engine": MagicMock(),
            "coinbase_advanced_client": MagicMock(),
        }

    def test_sets_paper_trading_flag(self):
        """build_execution_layer should read paper_trading from config."""
        bot = _make_bot()
        bot.position_limit = 1000.0
        bot.daily_loss_limit = 500.0

        with patch.dict("sys.modules", self._mock_execution_modules()):
            from bot.builder import BotBuilder
            BotBuilder.build_execution_layer(bot)

        assert bot.paper_trading is True

    def test_initializes_background_tasks_list(self):
        """build_execution_layer should create empty _background_tasks list."""
        bot = _make_bot()
        bot.position_limit = 1000.0
        bot.daily_loss_limit = 500.0

        with patch.dict("sys.modules", self._mock_execution_modules()):
            from bot.builder import BotBuilder
            BotBuilder.build_execution_layer(bot)

        assert bot._background_tasks == []

    def test_high_watermark_initialized_to_zero(self):
        """build_execution_layer should start with high watermark at 0."""
        bot = _make_bot()
        bot.position_limit = 1000.0
        bot.daily_loss_limit = 500.0

        with patch.dict("sys.modules", self._mock_execution_modules()):
            from bot.builder import BotBuilder
            BotBuilder.build_execution_layer(bot)

        assert bot._high_watermark_usd == 0.0
        assert bot._current_drawdown_pct == 0.0

    def test_ws_client_none_on_import_failure(self):
        """build_execution_layer should set _ws_client=None if import fails."""
        bot = _make_bot()
        bot.position_limit = 1000.0
        bot.daily_loss_limit = 500.0

        modules = self._mock_execution_modules()
        # Make coinbase_advanced_client import raise
        del modules["coinbase_advanced_client"]

        with patch.dict("sys.modules", modules):
            # Patch the import to fail
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def failing_import(name, *args, **kwargs):
                if name == "coinbase_advanced_client":
                    raise ImportError("test: not available")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=failing_import):
                from bot.builder import BotBuilder
                BotBuilder.build_execution_layer(bot)

        assert bot._ws_client is None


# ---------------------------------------------------------------------------
# Tests: build_trading_state
# ---------------------------------------------------------------------------


class TestBuildTradingState:
    """Tests for BotBuilder.build_trading_state."""

    def test_initializes_state_variables(self):
        """build_trading_state should set position, P&L, and history defaults."""
        bot = _make_bot()

        with patch("bot.builder.BotBuilder.build_trading_state.__wrapped__", create=True):
            pass  # need to test the actual method

        mock_validate = MagicMock()
        with patch.dict("sys.modules", {
            "renaissance_trading_bot": MagicMock(validate_config=mock_validate),
        }):
            from bot.builder import BotBuilder
            BotBuilder.build_trading_state(bot)

        assert bot.current_position == 0.0
        assert bot.daily_pnl == 0.0
        assert bot.last_trade_time is None
        assert bot.decision_history == []

    def test_calls_validate_config(self):
        """build_trading_state should call validate_config."""
        bot = _make_bot()
        mock_validate = MagicMock()

        with patch.dict("sys.modules", {
            "renaissance_trading_bot": MagicMock(validate_config=mock_validate),
        }):
            from bot.builder import BotBuilder
            BotBuilder.build_trading_state(bot)

        mock_validate.assert_called_once_with(bot.config, bot.logger)


# ---------------------------------------------------------------------------
# Tests: build_signal_weights
# ---------------------------------------------------------------------------


class TestBuildSignalWeights:
    """Tests for BotBuilder.build_signal_weights."""

    def test_injects_missing_ml_weights(self):
        """build_signal_weights should inject ml_ensemble/ml_cnn/crash_regime if missing."""
        bot = _make_bot()
        bot.config["signal_weights"] = {"order_flow": 0.5, "volume": 0.5}
        bot.db_enabled = False

        mock_rtb = MagicMock()
        mock_rtb.MacroDataCache = MagicMock()
        with patch.dict("sys.modules", {
            "renaissance_trading_bot": mock_rtb,
        }):
            from bot.builder import BotBuilder
            BotBuilder.build_signal_weights(bot)

        assert "ml_ensemble" in bot.signal_weights
        assert "ml_cnn" in bot.signal_weights
        assert "crash_regime" in bot.signal_weights
        assert bot.signal_weights["ml_ensemble"] == 0.20

    def test_preserves_existing_ml_weights(self):
        """build_signal_weights should not overwrite existing ML weight values."""
        bot = _make_bot()
        bot.config["signal_weights"] = {
            "order_flow": 0.5,
            "ml_ensemble": 0.30,
            "ml_cnn": 0.05,
            "crash_regime": 0.10,
        }
        bot.db_enabled = False

        mock_rtb = MagicMock()
        mock_rtb.MacroDataCache = MagicMock()
        with patch.dict("sys.modules", {
            "renaissance_trading_bot": mock_rtb,
        }):
            from bot.builder import BotBuilder
            BotBuilder.build_signal_weights(bot)

        assert bot.signal_weights["ml_ensemble"] == 0.30
        assert bot.signal_weights["ml_cnn"] == 0.05
        assert bot.signal_weights["crash_regime"] == 0.10


# ---------------------------------------------------------------------------
# Tests: build_intelligence_layer (optional deps)
# ---------------------------------------------------------------------------


class TestBuildIntelligenceLayer:
    """Tests for BotBuilder.build_intelligence_layer — optional module handling."""

    def test_devil_tracker_none_when_flag_off(self):
        """Devil tracker should be None when flag is False."""
        bot = _make_bot()
        _clear_rtb_flags()

        from bot.builder import BotBuilder
        BotBuilder.build_intelligence_layer(bot)

        assert bot.devil_tracker is None

    def test_kelly_sizer_none_when_flag_off(self):
        """Kelly sizer should be None when flag is False."""
        bot = _make_bot()
        _clear_rtb_flags()

        from bot.builder import BotBuilder
        BotBuilder.build_intelligence_layer(bot)

        assert bot.kelly_sizer is None

    def test_bar_aggregator_none_when_flag_off(self):
        """Bar aggregator should be None when flag is False."""
        bot = _make_bot()
        _clear_rtb_flags()

        from bot.builder import BotBuilder
        BotBuilder.build_intelligence_layer(bot)

        assert bot.bar_aggregator is None

    def test_medallion_regime_none_when_flag_off(self):
        """Medallion regime detector should be None when flag is False."""
        bot = _make_bot()
        _clear_rtb_flags()

        from bot.builder import BotBuilder
        BotBuilder.build_intelligence_layer(bot)

        assert bot.medallion_regime is None

    def test_hierarchical_regime_detectors_none_when_flag_off(self):
        """Hierarchical regime detectors should be None when flag is False."""
        bot = _make_bot()
        _clear_rtb_flags()

        from bot.builder import BotBuilder
        BotBuilder.build_intelligence_layer(bot)

        assert bot._macro_regime_detector is None
        assert bot._crypto_regime_detector is None
        assert bot._model_router is None


# ---------------------------------------------------------------------------
# Tests: validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_missing_required_section_logged_as_error(self):
        """Missing required sections should produce errors."""
        mock_logger = MagicMock()
        config = {
            "trading": {"product_ids": ["BTC-USD"], "cycle_interval_seconds": 300},
            "risk_management": {"daily_loss_limit": 500, "position_limit": 1000, "min_confidence": 0.45},
            "signal_weights": {},
            "database": {"path": "data/test.db"},
            # missing "coinbase"
        }

        from renaissance_trading_bot import validate_config
        validate_config(config, mock_logger)

        error_calls = [
            str(c) for c in mock_logger.error.call_args_list
        ]
        assert any("coinbase" in c for c in error_calls)

    def test_signal_weight_sum_warning(self):
        """Signal weights that don't sum to ~1.0 should produce a warning."""
        mock_logger = MagicMock()
        config = {
            "trading": {"product_ids": ["BTC-USD"], "cycle_interval_seconds": 300},
            "risk_management": {"daily_loss_limit": 500, "position_limit": 1000, "min_confidence": 0.45},
            "signal_weights": {"order_flow": 0.5, "volume": 0.1},  # sum = 0.6
            "database": {"path": "data/test.db"},
            "coinbase": {},
        }

        from renaissance_trading_bot import validate_config
        validate_config(config, mock_logger)

        warning_calls = [
            str(c) for c in mock_logger.warning.call_args_list
        ]
        assert any("signal_weights" in c for c in warning_calls)

    def test_valid_config_no_errors(self):
        """Complete valid config should produce no errors."""
        mock_logger = MagicMock()
        config = {
            "trading": {"product_ids": ["BTC-USD"], "cycle_interval_seconds": 300},
            "risk_management": {"daily_loss_limit": 500, "position_limit": 1000, "min_confidence": 0.45},
            "signal_weights": {"order_flow": 0.5, "volume": 0.5},
            "database": {"path": "data/test.db"},
            "coinbase": {},
        }

        from renaissance_trading_bot import validate_config
        validate_config(config, mock_logger)

        mock_logger.error.assert_not_called()

    def test_range_violation_logged_as_error(self):
        """Out-of-range numeric values should produce errors."""
        mock_logger = MagicMock()
        config = {
            "trading": {"product_ids": ["BTC-USD"], "cycle_interval_seconds": 300},
            "risk_management": {
                "daily_loss_limit": -100,  # out of range
                "position_limit": 1000,
                "min_confidence": 0.45,
            },
            "signal_weights": {},
            "database": {"path": "data/test.db"},
            "coinbase": {},
        }

        from renaissance_trading_bot import validate_config
        validate_config(config, mock_logger)

        error_calls = [
            str(c) for c in mock_logger.error.call_args_list
        ]
        assert any("daily_loss_limit" in c for c in error_calls)

    def test_missing_required_key_in_section(self):
        """Missing required key within a section should produce error."""
        mock_logger = MagicMock()
        config = {
            "trading": {"product_ids": ["BTC-USD"]},  # missing cycle_interval_seconds
            "risk_management": {"daily_loss_limit": 500, "position_limit": 1000, "min_confidence": 0.45},
            "signal_weights": {},
            "database": {"path": "data/test.db"},
            "coinbase": {},
        }

        from renaissance_trading_bot import validate_config
        validate_config(config, mock_logger)

        error_calls = [
            str(c) for c in mock_logger.error.call_args_list
        ]
        assert any("cycle_interval_seconds" in c for c in error_calls)
