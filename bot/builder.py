"""
BotBuilder — Extracts all component initialization from RenaissanceTradingBot.__init__.

Each build_* method returns a dict of named components that get set as bot attributes.
The bot's __init__ calls BotBuilder.build_all(bot) to wire everything up.

Availability flags are accessed via sys.modules['renaissance_trading_bot'] to avoid
circular imports (this module is imported by renaissance_trading_bot).
"""
import os
import json
import queue
import logging
import time
import sys
import asyncio
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from typing import Dict, Any, Optional


def _flag(name: str) -> bool:
    """Read an availability flag from the renaissance_trading_bot module."""
    return getattr(sys.modules.get('renaissance_trading_bot'), name, False)


class BotBuilder:
    """Builds and wires all RenaissanceTradingBot subsystems.

    Usage from __init__:
        BotBuilder.build_all(self)
    """

    @staticmethod
    def build_all(bot: 'RenaissanceTradingBot') -> None:
        """Run every build phase on *bot*, setting attributes directly."""
        BotBuilder._log_module_status(bot)
        BotBuilder.build_data_layer(bot)
        BotBuilder.build_signal_layer(bot)
        BotBuilder.build_risk_layer(bot)
        BotBuilder.build_execution_layer(bot)
        BotBuilder.build_ml_layer(bot)
        BotBuilder.build_intelligence_layer(bot)
        BotBuilder.build_monitoring_layer(bot)
        BotBuilder.build_dashboard_layer(bot)
        BotBuilder.build_arbitrage_layer(bot)
        BotBuilder.build_universe_layer(bot)
        BotBuilder.build_signal_weights(bot)
        BotBuilder.build_trading_state(bot)
        BotBuilder.build_optional_engines(bot)

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _log_module_status(bot: 'RenaissanceTradingBot') -> None:
        _modules = {
            'Orchestrator': _flag('ORCHESTRATOR_AVAILABLE'), 'Arbitrage': _flag('ARBITRAGE_AVAILABLE'),
            'Recovery': _flag('RECOVERY_AVAILABLE'), 'Monitoring': _flag('MONITORING_AVAILABLE'),
            'LiquidationDetector': _flag('LIQUIDATION_DETECTOR_AVAILABLE'),
            'SignalAggregator': _flag('SIGNAL_AGGREGATOR_AVAILABLE'),
            'MultiExchangeBridge': _flag('MULTI_EXCHANGE_BRIDGE_AVAILABLE'),
            'DataValidator': _flag('DATA_VALIDATOR_AVAILABLE'),
            'SignalThrottle': _flag('SIGNAL_THROTTLE_AVAILABLE'),
            'SignalValidation': _flag('SIGNAL_VALIDATION_AVAILABLE'),
            'HealthMonitor': _flag('HEALTH_MONITOR_AVAILABLE'),
            'MedallionAnalogs': _flag('MEDALLION_ANALOGS_AVAILABLE'),
            'PortfolioEngine': _flag('PORTFOLIO_ENGINE_AVAILABLE'),
            'DevilTracker': _flag('DEVIL_TRACKER_AVAILABLE'),
            'KellySizer': _flag('KELLY_SIZER_AVAILABLE'),
        }
        active = [k for k, v in _modules.items() if v]
        missing = [k for k, v in _modules.items() if not v]
        bot.logger.info(
            f"Module status: {len(active)}/{len(_modules)} loaded | missing: {missing if missing else 'none'}"
        )

    # ─────────────────────────────────────────────────────────
    # 1. Data Layer
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_data_layer(bot: 'RenaissanceTradingBot') -> None:
        """Initialize data providers and market data components."""
        from enhanced_config_manager import EnhancedConfigManager
        from microstructure_engine import MicrostructureEngine
        from enhanced_technical_indicators import EnhancedTechnicalIndicators
        from market_data_provider import LiveMarketDataProvider
        from derivatives_data_provider import DerivativesDataProvider
        from renaissance_signal_fusion import RenaissanceSignalFusion
        from alternative_data_engine import AlternativeDataEngine
        from renaissance_engine_core import SignalFusion

        bot.product_ids = bot.config.get("trading", {}).get("product_ids", ["BTC-USD"])
        bot.config_manager = EnhancedConfigManager("config")

        bot.microstructure_engine = MicrostructureEngine()
        bot._tech_indicators: Dict[str, EnhancedTechnicalIndicators] = {
            pid: EnhancedTechnicalIndicators() for pid in bot.product_ids
        }
        bot.market_data_provider = LiveMarketDataProvider(bot.config, logger=bot.logger)
        bot.derivatives_provider = DerivativesDataProvider(cache_ttl_seconds=60)
        bot._derivatives_history: Dict[str, deque] = {}

        bot.signal_fusion = SignalFusion()
        bot.signal_fusion.set_ml_signal_scale(bot.config.get("ml_signal_scale", 10.0))
        bot.alternative_data_engine = AlternativeDataEngine(bot.config, logger=bot.logger)

    # ─────────────────────────────────────────────────────────
    # 2. Signal Layer
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_signal_layer(bot: 'RenaissanceTradingBot') -> None:
        """Initialize signal generation components."""
        from volume_profile_engine import VolumeProfileEngine
        from fractal_intelligence import FractalIntelligenceEngine
        from market_entropy_engine import MarketEntropyEngine
        from quantum_oscillator_engine import QuantumOscillatorEngine
        from ghost_runner import GhostRunner
        from self_reinforcing_learning import SelfReinforcingLearningEngine
        from confluence_engine import ConfluenceEngine
        from basis_trading_engine import BasisTradingEngine
        from deep_nlp_bridge import DeepNLPBridge
        from market_making_engine import MarketMakingEngine
        from meta_strategy_selector import MetaStrategySelector
        from genetic_optimizer import GeneticWeightOptimizer
        from cross_asset_engine import CrossAssetCorrelationEngine
        from whale_activity_monitor import WhaleActivityMonitor
        from breakout_scanner import BreakoutScanner, BreakoutSignal
        from advanced_mean_reversion_engine import AdvancedMeanReversionEngine
        from correlation_network_engine import CorrelationNetworkEngine
        from garch_volatility_engine import GARCHVolatilityEngine
        from historical_data_cache import HistoricalDataCache
        from statistical_arbitrage_engine import StatisticalArbitrageEngine
        from random_baseline import RandomEntryBaseline

        db_cfg = bot.config.get("database", {"path": "data/renaissance_bot.db", "enabled": True})
        db_path = db_cfg.get("path", "data/renaissance_bot.db")

        bot.volume_profile_engine = VolumeProfileEngine()
        bot.fractal_intelligence = FractalIntelligenceEngine(logger=bot.logger)
        bot.market_entropy = MarketEntropyEngine(logger=bot.logger)
        bot.quantum_oscillator = QuantumOscillatorEngine(logger=bot.logger)
        bot._last_vp_status = {}

        bot.ghost_runner = GhostRunner(bot, logger=bot.logger)
        bot.learning_engine = SelfReinforcingLearningEngine(db_path, logger=bot.logger)
        bot.confluence_engine = ConfluenceEngine(logger=bot.logger)
        bot.basis_engine = BasisTradingEngine(logger=bot.logger)
        bot.nlp_bridge = DeepNLPBridge(bot.config.get("nlp", {}), logger=bot.logger)
        bot.market_making = MarketMakingEngine(bot.config.get("market_making", {}), logger=bot.logger)
        bot.strategy_selector = MetaStrategySelector(bot.config.get("meta_strategy", {}), logger=bot.logger)

        bot.genetic_optimizer = GeneticWeightOptimizer(db_path, logger=bot.logger)
        bot.correlation_engine = CrossAssetCorrelationEngine(logger=bot.logger)
        bot.stat_arb_engine = StatisticalArbitrageEngine(logger=bot.logger)
        bot.whale_monitor = WhaleActivityMonitor(bot.config.get("whale_monitor", {}), logger=bot.logger)

        mr_cfg = bot.config.get("mean_reversion", {})
        bot.mean_reversion_engine = AdvancedMeanReversionEngine(mr_cfg, logger=bot.logger)

        corr_net_cfg = bot.config.get("correlation_network", {})
        bot.correlation_network = CorrelationNetworkEngine(corr_net_cfg, logger=bot.logger)

        garch_cfg = bot.config.get("garch_volatility", {})
        bot.garch_engine = GARCHVolatilityEngine(garch_cfg, logger=bot.logger)

        hist_cfg = bot.config.get("historical_data_cache", {})
        hist_cfg.setdefault("db_path", db_path)
        bot.historical_cache = HistoricalDataCache(hist_cfg, logger=bot.logger)
        if hist_cfg.get("enabled", False):
            bot.historical_cache.init_tables()

        # Breakout Scanner
        scanner_cfg = bot.config.get("breakout_scanner", {"enabled": True, "max_flagged": 30})
        bot.breakout_scanner = BreakoutScanner(
            max_flagged=scanner_cfg.get("max_flagged", 30),
            min_volume_usd=scanner_cfg.get("min_volume_usd", 500_000),
            min_breakout_score=scanner_cfg.get("min_breakout_score", 25.0),
            logger=bot.logger,
        )
        bot.scanner_enabled = scanner_cfg.get("enabled", True)
        bot._breakout_scores: Dict[str, BreakoutSignal] = {}

        # Breakout Strategy
        bot.breakout_strategy = None
        if _flag('BREAKOUT_STRATEGY_AVAILABLE'):
            try:
                from breakout_strategy import BreakoutStrategy
                _bst_db = db_path
                _bst_cfg = bot.config.get("breakout_strategy", {})
                bot.breakout_strategy = BreakoutStrategy(db_path=_bst_db, config=_bst_cfg, logger=bot.logger)
            except Exception as _bst_err:
                bot.logger.warning(f"Breakout Strategy init failed: {_bst_err}")

        # Random baseline
        bot.random_baseline = RandomEntryBaseline(db_path=db_path)

        # Data validator
        if _flag('DATA_VALIDATOR_AVAILABLE'):
            from data_validator import DataValidator
            bot.data_validator = DataValidator(logger=bot.logger)
        else:
            bot.data_validator = None

        # Signal auto-throttle
        if _flag('SIGNAL_THROTTLE_AVAILABLE'):
            from signal_auto_throttle import SignalAutoThrottle
            throttle_cfg = bot.config.get('signal_throttle', {})
            bot.signal_throttle = SignalAutoThrottle(throttle_cfg, logger=bot.logger)
        else:
            bot.signal_throttle = None

        # Signal validation gate
        if _flag('SIGNAL_VALIDATION_AVAILABLE'):
            from signal_validation_gate import SignalValidationGate
            bot.signal_validation_gate = SignalValidationGate(logger=bot.logger)
        else:
            bot.signal_validation_gate = None

        # Health monitor
        if _flag('HEALTH_MONITOR_AVAILABLE'):
            from portfolio_health_monitor import PortfolioHealthMonitor
            health_cfg = bot.config.get('health_monitor', {})
            bot.health_monitor = PortfolioHealthMonitor(health_cfg, logger=bot.logger)
        else:
            bot.health_monitor = None

        # Medallion signal analogs
        if _flag('MEDALLION_ANALOGS_AVAILABLE'):
            from medallion_signal_analogs import MedallionSignalAnalogs
            analog_cfg = bot.config.get('medallion_analogs', {})
            bot.medallion_analogs = MedallionSignalAnalogs(analog_cfg, logger=bot.logger)
        else:
            bot.medallion_analogs = None

        # Unified portfolio engine
        if _flag('PORTFOLIO_ENGINE_AVAILABLE'):
            from unified_portfolio_engine import UnifiedPortfolioEngine
            portfolio_cfg = bot.config.get('portfolio_engine', {})
            bot.portfolio_engine = UnifiedPortfolioEngine(portfolio_cfg, logger=bot.logger) if portfolio_cfg.get('enabled', False) else None
        else:
            bot.portfolio_engine = None

        # Signal scorecard & adaptive weights
        bot._signal_scorecard: Dict[str, Dict[str, Dict[str, int]]] = {}
        bot._pending_predictions: Dict[str, Dict] = {}
        bot._adaptive_weight_blend = 0.0
        bot._adaptive_min_samples = 15
        bot._ml_accuracy_cache: Dict[str, Dict[str, float]] = {}
        bot._ml_accuracy_cache_cycle = 0
        bot._ml_eval_min_predictions = int(bot.config.get('ml_evaluation', {}).get('min_predictions_for_edge', 50))
        bot._ml_eval_blend_measured = float(bot.config.get('ml_evaluation', {}).get('edge_blend_measured', 0.6))
        bot._ml_eval_blend_model = float(bot.config.get('ml_evaluation', {}).get('edge_blend_model', 0.4))

    # ─────────────────────────────────────────────────────────
    # 3. Risk Layer
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_risk_layer(bot: 'RenaissanceTradingBot') -> None:
        """Initialize risk management components."""
        from regime_overlay import RegimeOverlay
        from risk_gateway import RiskGateway
        from real_time_pipeline import RealTimePipeline
        from renaissance_engine_core import RiskManager

        _regime_db = bot.config.get('database', {}).get('path', 'data/renaissance_bot.db')
        bot.regime_overlay = RegimeOverlay(
            bot.config.get("regime_overlay", {}), logger=bot.logger, db_path=_regime_db
        )
        bot.risk_gateway = RiskGateway(bot.config.get("risk_gateway", {}), logger=bot.logger)
        bot.risk_manager = RiskManager(
            position_limit=bot.config.get("risk_management", {}).get("position_limit", 1000.0)
        )
        bot.real_time_pipeline = RealTimePipeline(
            bot.config.get("real_time_pipeline", {}), logger=bot.logger
        )

        risk_cfg = bot.config.get("risk_management", {})
        bot.daily_loss_limit = float(risk_cfg.get("daily_loss_limit", 500))
        bot.position_limit = float(risk_cfg.get("position_limit", 1000))
        bot.min_confidence = float(risk_cfg.get("min_confidence", 0.45))

    # ─────────────────────────────────────────────────────────
    # 4. Execution Layer
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_execution_layer(bot: 'RenaissanceTradingBot') -> None:
        """Initialize execution and position management components."""
        from execution_algorithm_suite import ExecutionAlgorithmSuite
        from slippage_protection_system import SlippageProtectionSystem
        from coinbase_client import EnhancedCoinbaseClient, CoinbaseCredentials
        from position_manager import EnhancedPositionManager, RiskLimits
        from position_sizer import RenaissancePositionSizer
        from alert_manager import AlertManager

        bot.execution_suite = ExecutionAlgorithmSuite()
        bot.slippage_protection = SlippageProtectionSystem()

        cb_config = bot.config.get("coinbase", {})
        paper_mode = bot.config.get("trading", {}).get("paper_trading", True)
        bot.paper_trading = paper_mode
        bot.coinbase_client = EnhancedCoinbaseClient(
            credentials=CoinbaseCredentials(
                api_key=os.environ.get(cb_config.get("api_key_env", "CB_API_KEY"), ""),
                api_secret=os.environ.get(cb_config.get("api_secret_env", "CB_API_SECRET"), ""),
                sandbox=bot.config.get("trading", {}).get("sandbox", True),
            ),
            paper_trading=paper_mode,
            logger=bot.logger,
        )
        bot.position_manager = EnhancedPositionManager(
            coinbase_client=bot.coinbase_client,
            risk_limits=RiskLimits(
                max_position_size_usd=bot.position_limit,
                max_daily_loss_usd=bot.daily_loss_limit,
            ),
            logger=bot.logger,
        )

        bot.position_sizer = RenaissancePositionSizer(
            config={
                "default_balance_usd": 10000.0,
                "max_position_pct": 10.0,
                "max_total_exposure_pct": 50.0,
                "kelly_fraction": 0.50,
                "min_edge": 0.001,
                "min_win_prob": 0.52,
                "taker_fee_bps": 5.0,
                "maker_fee_bps": 0.0,
                "spread_cost_bps": 2.0,
                "slippage_bps": 1.0,
                "cost_gate_ratio": 0.50,
                "target_vol": 0.02,
                "min_order_usd": 1.0,
            },
            logger=bot.logger,
        )
        bot._cached_balance_usd: float = 0.0
        bot._high_watermark_usd: float = 0.0
        bot._current_drawdown_pct: float = 0.0
        bot._weekly_pnl: float = 0.0
        bot._week_start_balance: float = 0.0
        bot._week_reset_today: bool = False

        bot._killed = False
        bot._start_time = datetime.now(timezone.utc)
        bot._background_tasks: list = []
        bot._weights_lock = asyncio.Lock()

        # Alert manager (legacy Slack)
        alert_cfg = bot.config.get("alerting", {})
        bot.alert_manager = AlertManager(alert_cfg, logger=bot.logger)

        # WebSocket feed
        bot._ws_queue: queue.Queue = queue.Queue(maxsize=1000)
        try:
            from coinbase_advanced_client import CoinbaseAdvancedClient
            ws_config = {
                'api_key': os.environ.get(cb_config.get("api_key_env", "CB_API_KEY"), ""),
                'api_secret': os.environ.get(cb_config.get("api_secret_env", "CB_API_SECRET"), ""),
                'passphrase': os.environ.get(cb_config.get("api_passphrase_env", ""), ""),
                'sandbox': bot.config.get("trading", {}).get("sandbox", True),
                'symbols': bot.product_ids,
                'websocket_channels': ["level2", "ticker", "matches"],
            }
            bot._ws_client = CoinbaseAdvancedClient(ws_config)
        except Exception as e:
            bot.logger.warning(f"WebSocket client init failed (will use REST fallback): {e}")
            bot._ws_client = None

        # Persistence & Attribution
        from database_manager import DatabaseManager
        from decision_audit_logger import DecisionAuditLogger
        from performance_attribution_engine import PerformanceAttributionEngine

        db_cfg = bot.config.get("database", {"path": "data/renaissance_bot.db", "enabled": True})
        bot.db_enabled = db_cfg.get("enabled", True)
        bot.db_manager = DatabaseManager(db_cfg)
        bot.audit_logger = DecisionAuditLogger(bot.db_manager)
        bot.attribution_engine = PerformanceAttributionEngine()

    # ─────────────────────────────────────────────────────────
    # 5. ML Layer
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_ml_layer(bot: 'RenaissanceTradingBot') -> None:
        """Initialize ML models and integration bridge."""
        from ml_integration_bridge import MLIntegrationBridge

        bot.ml_enabled = bot.config.get("ml_integration", {}).get("enabled", True)
        bot.ml_bridge = MLIntegrationBridge(bot.config)
        if bot.ml_enabled:
            bot.ml_bridge.initialize()
            loaded_models = list(bot.ml_bridge.model_manager.models.keys())
            if loaded_models:
                bot.logger.info(f"ML startup validation: {len(loaded_models)} trained models active: {loaded_models}")
            else:
                bot.logger.warning("ML startup validation: NO trained models loaded — ML predictions will be empty")

            metadata_path = os.path.join("models", "trained", "training_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path) as f:
                        training_meta = json.load(f)
                    for model_name, info in training_meta.items():
                        last_trained = datetime.fromisoformat(info["last_trained"].replace("Z", "+00:00"))
                        age_days = (datetime.now(timezone.utc) - last_trained).days
                        if age_days > 7:
                            bot.logger.warning(
                                f"ML staleness: {model_name} last trained {age_days} days ago — retraining recommended"
                            )
                except Exception as e:
                    bot.logger.debug(f"Could not check model staleness: {e}")

        bot.ml_performance_metrics = {
            'total_trades': 0, 'ml_enhanced_trades': 0,
            'avg_ml_processing_time': 0.0, 'ml_success_rate': 0.0,
        }
        bot.last_vpin = 0.5

    # ─────────────────────────────────────────────────────────
    # 6. Intelligence Layer
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_intelligence_layer(bot: 'RenaissanceTradingBot') -> None:
        """Initialize Medallion intelligence modules."""
        db_path = bot.config.get('database', {}).get('path', 'data/renaissance_bot.db')

        # Devil Tracker
        if _flag('DEVIL_TRACKER_AVAILABLE'):
            from core.devil_tracker import DevilTracker
            bot.devil_tracker = DevilTracker(db_path)
            bot.logger.info("DevilTracker: ACTIVE — tracking signal→fill execution quality")
        else:
            bot.devil_tracker = None

        # Kelly Sizer
        if _flag('KELLY_SIZER_AVAILABLE'):
            from core.kelly_position_sizer import KellyPositionSizer
            bot.kelly_sizer = KellyPositionSizer(bot.config, db_path)
            bot.logger.info("KellyPositionSizer: ACTIVE — optimal sizing from trade history")
        else:
            bot.kelly_sizer = None

        # Daily Signal Review
        if _flag('MEDALLION_THROTTLE_AVAILABLE'):
            from core.signal_throttle import SignalThrottle as MedallionSignalThrottle
            bot.daily_signal_review = MedallionSignalThrottle(bot.config, db_path)
            bot.logger.info("DailySignalReview: ACTIVE — end-of-day P&L throttling")
        else:
            bot.daily_signal_review = None

        # Leverage Manager
        if _flag('LEVERAGE_MANAGER_AVAILABLE'):
            from core.leverage_manager import LeverageManager
            bot.leverage_mgr = LeverageManager(bot.config, db_path)
            bot.logger.info("LeverageManager: ACTIVE — consistency-based leverage")
        else:
            bot.leverage_mgr = None

        # Medallion Regime Detector (observation only)
        if _flag('MEDALLION_REGIME_AVAILABLE'):
            from intelligence.regime_detector import RegimeDetector as MedallionRegimeDetector
            bot.medallion_regime = MedallionRegimeDetector(bot.config, db_path)
            bot.logger.info("MedallionRegimeDetector: OBSERVATION — logging alongside RegimeOverlay")
        else:
            bot.medallion_regime = None

        # Insurance Scanner
        if _flag('INSURANCE_SCANNER_AVAILABLE'):
            from intelligence.insurance_scanner import InsurancePremiumScanner
            bot.insurance_scanner = InsurancePremiumScanner(bot.config, db_path)
            bot.logger.info("InsurancePremiumScanner: OBSERVATION — periodic premium scanning")
        else:
            bot.insurance_scanner = None

        # Medallion Portfolio Engine
        if _flag('MEDALLION_PORTFOLIO_ENGINE_AVAILABLE'):
            from core.portfolio_engine import PortfolioEngine as MedallionPortfolioEngine
            bot.medallion_portfolio_engine = MedallionPortfolioEngine(
                config=bot.config,
                devil_tracker=bot.devil_tracker,
                position_manager=bot.position_manager,
            )
            bot.logger.info("MedallionPortfolioEngine: OBSERVATION — drift logging (no corrections)")
        else:
            bot.medallion_portfolio_engine = None

        # Bar Aggregator
        if _flag('BAR_AGGREGATOR_AVAILABLE'):
            from data_module.bar_aggregator import BarAggregator
            bot.bar_aggregator = BarAggregator(bot.config, db_path)
            bot.logger.info("BarAggregator: ACTIVE — 5-min bar aggregation")
        else:
            bot.bar_aggregator = None

        # Synchronized Executor
        if _flag('SYNC_EXECUTOR_AVAILABLE'):
            from execution.synchronized_executor import SynchronizedExecutor
            bot.sync_executor = SynchronizedExecutor(bot.config, bot.devil_tracker)
            bot.logger.info("SynchronizedExecutor: ACTIVE — cross-exchange execution")
        else:
            bot.sync_executor = None

        # Trade Hider
        if _flag('TRADE_HIDER_AVAILABLE'):
            from execution.trade_hider import TradeHider
            bot.trade_hider = TradeHider(bot.config)
            bot.logger.info("TradeHider: ACTIVE — execution obfuscation")
        else:
            bot.trade_hider = None

        # MHPE
        bot.mhpe = None
        if _flag('MHPE_AVAILABLE'):
            try:
                from intelligence.multi_horizon_estimator import MultiHorizonEstimator
                bot.mhpe = MultiHorizonEstimator(
                    config=bot.config.get('multi_horizon_estimator', {}),
                    regime_predictor=bot.medallion_regime,
                )
                bot.logger.info("MHPE: ACTIVE — 7-horizon probability cones")
            except Exception as _mhpe_err:
                bot.logger.warning(f"MHPE init failed: {_mhpe_err}")

        # Position Re-evaluator (Doc 10)
        bot.position_reevaluator = None
        if _flag('POSITION_REEVALUATOR_AVAILABLE'):
            try:
                from portfolio.position_reevaluator import PositionReEvaluator
                bot.position_reevaluator = PositionReEvaluator(
                    config=bot.config.get('reevaluation', {}),
                    kelly_sizer=bot.kelly_sizer,
                    regime_detector=bot.medallion_regime,
                    devil_tracker=bot.devil_tracker,
                    mhpe=bot.mhpe,
                    db_path=db_path,
                )
                bot.logger.info("PositionReEvaluator: ACTIVE — continuous position re-evaluation")
            except Exception as _re_err:
                bot.logger.warning(f"PositionReEvaluator init failed: {_re_err}")

        # Fast Mean Reversion Scanner
        bot.fast_reversion_scanner = None
        fmr_cfg = bot.config.get("fast_mean_reversion", {})
        if fmr_cfg.get("enabled", False) and _flag('FAST_REVERSION_AVAILABLE'):
            try:
                from intelligence.fast_mean_reversion import FastMeanReversionScanner
                bot.fast_reversion_scanner = FastMeanReversionScanner(fmr_cfg, bot.bar_aggregator)
                bot.logger.info("Fast Mean Reversion Scanner initialized")
            except Exception as e:
                bot.logger.warning(f"Fast reversion scanner init failed: {e}")

        # Heartbeat Writer
        bot.heartbeat_writer = None
        bot_id = bot.config.get("bot_id", "bot-01")
        orch_cfg = bot.config.get("orchestrator", {})
        if _flag('HEARTBEAT_AVAILABLE'):
            try:
                from orchestrator.heartbeat import HeartbeatWriter
                bot.heartbeat_writer = HeartbeatWriter(
                    bot_id=bot_id,
                    heartbeat_dir=orch_cfg.get("heartbeat_dir", "data/heartbeats"),
                )
                bot.logger.info(f"HeartbeatWriter initialized (bot_id={bot_id})")
            except Exception as e:
                bot.logger.warning(f"HeartbeatWriter init failed: {e}")

        # Hierarchical Regime Detection
        bot._macro_regime_detector = None
        bot._crypto_regime_detector = None
        bot._model_router = None
        if _flag('HIERARCHICAL_REGIME_AVAILABLE'):
            from macro_regime_detector import MacroRegimeDetector
            from crypto_regime_detector import CryptoRegimeDetector
            from model_router import ModelRouter
            bot._macro_regime_detector = MacroRegimeDetector(logger=bot.logger)
            bot._crypto_regime_detector = CryptoRegimeDetector(logger=bot.logger)
            bot._model_router = ModelRouter(observation_mode=True, logger=bot.logger)
            bot.logger.info(
                "Hierarchical Regime Detection initialized (OBSERVATION MODE): "
                "Macro(4-state) + Crypto(4-state) + ModelRouter(12-entry matrix)"
            )

    # ─────────────────────────────────────────────────────────
    # 7. Monitoring Layer
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_monitoring_layer(bot: 'RenaissanceTradingBot') -> None:
        """Initialize monitoring and alerting components."""
        db_path = bot.config.get('database', {}).get('path', 'data/renaissance_bot.db')

        # Beta Monitor
        if _flag('BETA_MONITOR_AVAILABLE'):
            from monitoring.beta_monitor import BetaMonitor
            bot.beta_monitor = BetaMonitor(bot.config, db_path)
            bot.logger.info("BetaMonitor: ACTIVE — portfolio beta tracking")
        else:
            bot.beta_monitor = None

        # Capacity Monitor
        if _flag('CAPACITY_MONITOR_AVAILABLE'):
            from monitoring.capacity_monitor import CapacityMonitor
            bot.capacity_monitor = CapacityMonitor(bot.config, db_path)
            bot.logger.info("CapacityMonitor: ACTIVE — capacity wall detection")
        else:
            bot.capacity_monitor = None

        # Sharpe Monitor
        if _flag('SHARPE_MONITOR_AVAILABLE'):
            from monitoring.sharpe_monitor import SharpeMonitor
            bot.sharpe_monitor_medallion = SharpeMonitor(bot.config, db_path)
            bot.logger.info("SharpeMonitor: ACTIVE — rolling Sharpe health")
        else:
            bot.sharpe_monitor_medallion = None

        # Recovery & State Manager
        bot.state_manager = None
        bot.shutdown_handler = None
        if _flag('RECOVERY_AVAILABLE'):
            try:
                from recovery.state_manager import StateManager
                from recovery.database import ensure_all_tables
                ensure_all_tables(db_path)
                bot.state_manager = StateManager()
                bot.logger.info("Recovery StateManager initialized")
            except Exception as e:
                bot.logger.warning(f"Recovery module init failed: {e}")

        # Monitoring Alert Manager (Telegram)
        bot.monitoring_alert_manager = None
        if _flag('MONITORING_AVAILABLE'):
            try:
                from monitoring.telegram_bot import TelegramAlerter
                from monitoring.alert_manager import AlertManager as MonitoringAlertManager
                telegram_cfg = bot.config.get("telegram", {})
                telegram_alerter = TelegramAlerter(config=telegram_cfg)
                bot.monitoring_alert_manager = MonitoringAlertManager(
                    telegram_alerter=telegram_alerter, db_path=db_path,
                )
                bot.logger.info(
                    "Monitoring AlertManager initialized (Telegram %s)",
                    "active" if telegram_alerter._bot_token else "console-only",
                )
            except Exception as e:
                bot.logger.warning(f"Monitoring module init failed: {e}")

        # Liquidation Detector
        bot.liquidation_detector = None
        if _flag('LIQUIDATION_DETECTOR_AVAILABLE'):
            try:
                from signals.liquidation_detector import LiquidationCascadeDetector
                liq_cfg = bot.config.get("liquidation_detector", {})
                bot.liquidation_detector = LiquidationCascadeDetector(config=liq_cfg)
                bot.logger.info("Liquidation Cascade Detector initialized")
            except Exception as e:
                bot.logger.warning(f"Liquidation detector init failed: {e}")

        # Signal Aggregator
        bot.signal_aggregator = None
        if _flag('SIGNAL_AGGREGATOR_AVAILABLE'):
            try:
                from signals.signal_aggregator import SignalAggregator
                micro_weights = bot.config.get("microstructure_signals", {}).get("weights", None)
                bot.signal_aggregator = SignalAggregator(weights=micro_weights)
                bot.logger.info("Advanced Signal Aggregator initialized")
            except Exception as e:
                bot.logger.warning(f"Signal aggregator init failed: {e}")

    # ─────────────────────────────────────────────────────────
    # 8. Dashboard Layer
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_dashboard_layer(bot: 'RenaissanceTradingBot') -> None:
        """Initialize dashboard components."""
        from dashboard.event_emitter import DashboardEventEmitter
        from institutional_dashboard import InstitutionalDashboard

        bot.dashboard_emitter = DashboardEventEmitter()
        bot.dashboard_emitter.clear_cache()

        # Legacy Flask dashboard
        bot.dashboard_enabled = bot.config.get("institutional_dashboard", {}).get("enabled", True)
        if bot.dashboard_enabled:
            try:
                _dash_port = int(bot.config.get("institutional_dashboard", {}).get("port", 5050))
                bot.dashboard = InstitutionalDashboard(bot, host="0.0.0.0", port=_dash_port)
                bot.dashboard.run()
            except Exception as e:
                bot.logger.warning(f"Failed to start dashboard (likely port conflict): {e}")
                bot.dashboard = None
        else:
            bot.dashboard = None

        # FastAPI Real-Time Dashboard
        bot._dashboard_server_task = None
        dash_cfg = bot.config.get("dashboard_config", {})
        if dash_cfg.get("enabled", True):
            try:
                from dashboard.server import create_app
                import threading
                import uvicorn
                bot._dashboard_app = create_app(
                    config_path=str(bot.config_path),
                    emitter=bot.dashboard_emitter,
                )
                dash_port = dash_cfg.get("port", 8080)
                dash_host = dash_cfg.get("host", "0.0.0.0")

                def _run_dashboard():
                    uvicorn.run(bot._dashboard_app, host=dash_host, port=dash_port, log_level="warning")
                threading.Thread(target=_run_dashboard, daemon=True).start()
                bot.logger.info(f"Real-time dashboard started on {dash_host}:{dash_port}")
            except Exception as e:
                bot.logger.warning(f"Failed to start real-time dashboard: {e}")

    # ─────────────────────────────────────────────────────────
    # 9. Arbitrage Layer
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_arbitrage_layer(bot: 'RenaissanceTradingBot') -> None:
        """Initialize arbitrage and multi-exchange bridge."""
        bot.arbitrage_enabled = bot.config.get("arbitrage", {}).get("enabled", False)
        bot.arbitrage_orchestrator = None
        if bot.arbitrage_enabled and _flag('ARBITRAGE_AVAILABLE'):
            try:
                from arbitrage.orchestrator import ArbitrageOrchestrator
                arb_config_path = bot.config.get("arbitrage", {}).get(
                    "config_path", "arbitrage/config/arbitrage.yaml"
                )
                bot.arbitrage_orchestrator = ArbitrageOrchestrator(config_path=arb_config_path)
                bot.logger.info("Arbitrage engine initialized (will start with trading loop)")
                if hasattr(bot, '_dashboard_app') and bot._dashboard_app:
                    bot._dashboard_app.state.arb_orchestrator = bot.arbitrage_orchestrator
            except Exception as e:
                bot.logger.warning(f"Arbitrage engine init failed: {e}")
                bot.arbitrage_orchestrator = None
        elif bot.arbitrage_enabled and not _flag('ARBITRAGE_AVAILABLE'):
            bot.logger.warning("Arbitrage enabled in config but module not available")

        bot._unified_price_feed = (
            getattr(bot.arbitrage_orchestrator, 'price_feed', None)
            if bot.arbitrage_orchestrator else None
        )

        # Wire liquidation feed
        if bot._unified_price_feed:
            if hasattr(bot, 'reversal_strategy') and bot.reversal_strategy:
                bot.reversal_strategy._price_feed = bot._unified_price_feed
                bot.logger.info("Reversal strategy: wired to liquidation feed")
            if hasattr(bot, 'timing_engine') and bot.timing_engine:
                bot.timing_engine._price_feed = bot._unified_price_feed

        # Multi-Exchange Signal Bridge
        bot.multi_exchange_bridge = None
        me_cfg = bot.config.get("multi_exchange_signals", {})
        if me_cfg.get("enabled", False) and _flag('MULTI_EXCHANGE_BRIDGE_AVAILABLE') and bot.arbitrage_orchestrator:
            try:
                from signals.multi_exchange_bridge import MultiExchangeBridge
                bot.multi_exchange_bridge = MultiExchangeBridge(
                    book_manager=bot.arbitrage_orchestrator.book_manager,
                    mexc_client=bot.arbitrage_orchestrator.mexc,
                    binance_client=bot.arbitrage_orchestrator.binance,
                )
                bot.logger.info("Multi-exchange signal bridge initialized")
            except Exception as e:
                bot.logger.warning(f"Multi-exchange bridge init failed: {e}")

    # ─────────────────────────────────────────────────────────
    # 10. Universe Layer
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_universe_layer(bot: 'RenaissanceTradingBot') -> None:
        """Initialize Binance spot provider and dynamic universe state."""
        from binance_spot_provider import BinanceSpotProvider

        bot.binance_spot = BinanceSpotProvider(logger=bot.logger)

        bot.trading_universe: list = []
        bot._pair_tiers: Dict[str, int] = {}
        bot._pair_binance_symbols: Dict[str, str] = {}
        bot._universe_built = False
        bot._universe_last_refresh: float = 0.0

        bot._pair_spread_history: Dict[str, list] = {}
        bot._spread_lookback = 100
        bot._max_spread_bps = float(bot.config.get('universe', {}).get('max_spread_bps', 3.0))

        bot._ml_pred_stats: Dict[str, Dict[str, float]] = {}
        bot._ml_zscore_lookback = int(bot.config.get('ml_ensemble', {}).get('rescale_lookback', 200))
        bot._ml_zscore_tanh_scale = float(bot.config.get('ml_ensemble', {}).get('rescale_tanh_scale', 0.5))

        bot.buy_threshold = float(bot.config.get('trading', {}).get('buy_threshold', 0.06))
        bot.sell_threshold = float(bot.config.get('trading', {}).get('sell_threshold', -0.06))
        bot.adaptive_thresholds = bot.config.get("adaptive_thresholds", True)
        bot.breakout_candidates = []
        bot.scan_cycle_count = 0
        bot._signal_filter_stats = {
            'total': 0, 'traded': 0, 'filtered_threshold': 0,
            'filtered_confidence': 0, 'filtered_agreement': 0,
        }

    # ─────────────────────────────────────────────────────────
    # 11. Signal Weights
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_signal_weights(bot: 'RenaissanceTradingBot') -> None:
        """Initialize signal weights and DB."""
        if bot.db_enabled:
            bot._track_task(bot.db_manager.init_database())
            bot._track_task(bot.audit_logger.self_test())

        raw_weights = bot.config.get("signal_weights", {
            'order_flow': 0.14, 'order_book': 0.0, 'volume': 0.08,
            'macd': 0.05, 'rsi': 0.05, 'bollinger': 0.05,
            'alternative': 0.01, 'stat_arb': 0.084,
            'volume_profile': 0.04, 'fractal': 0.05,
            'entropy': 0.05, 'quantum': 0.07,
            'lead_lag': 0.10, 'correlation_divergence': 0.0,
            'garch_vol': 0.06, 'ml_ensemble': 0.20,
            'ml_cnn': 0.0, 'breakout': 0.08, 'crash_regime': 0.15,
        })
        bot.signal_weights = {str(k): float(bot._force_float(v)) for k, v in raw_weights.items()}

        _ml_required = {'ml_ensemble': 0.20, 'ml_cnn': 0.0, 'crash_regime': 0.15}
        for k, v in _ml_required.items():
            if k not in bot.signal_weights:
                bot.signal_weights[k] = v
                bot.logger.info(f"Injected missing ML weight: {k}={v}")

        from bot.data_collection import MacroDataCache
        bot._macro_cache = MacroDataCache(logger=bot.logger)

    # ─────────────────────────────────────────────────────────
    # 12. Trading State
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_trading_state(bot: 'RenaissanceTradingBot') -> None:
        """Initialize trading state variables."""
        bot.current_position = 0.0
        bot.daily_pnl = 0.0
        bot.last_trade_time = None
        bot.decision_history = []

        from renaissance_trading_bot import validate_config
        validate_config(bot.config, bot.logger)

        # Disabled legacy strategies
        bot.polymarket_executor = None
        bot.polymarket_live_executor = None
        bot.reversal_strategy = None
        bot.simple_better = None
        bot.cascade_collector = None

    # ─────────────────────────────────────────────────────────
    # 13. Optional Engines (Agent Coordinator, Token Spray, Straddle, Oracle, Polymarket)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_optional_engines(bot: 'RenaissanceTradingBot') -> None:
        """Initialize optional feature engines."""
        db_path = bot.config.get('database', {}).get('path', 'data/renaissance_bot.db')

        # Polymarket Bridge & Scanner
        from polymarket_bridge import PolymarketBridge
        from polymarket_scanner import PolymarketScanner

        poly_cfg = bot.config.get("polymarket_bridge", {})
        bot.polymarket_bridge = PolymarketBridge(
            min_prediction=poly_cfg.get("min_prediction", 0.03),
            min_agreement=poly_cfg.get("min_agreement", 0.55),
            observation_mode=poly_cfg.get("observation_mode", True),
            logger=bot.logger,
        )
        poly_scanner_cfg = bot.config.get("polymarket_scanner", {})
        bot.polymarket_scanner = PolymarketScanner(
            db_path=db_path,
            cache_ttl=poly_scanner_cfg.get("cache_ttl", 300),
            logger=bot.logger,
        )
        bot._last_poly_scan = None
        bot._latest_scanner_opportunities = []

        # Spread Capture
        bot.rtds = None
        bot.spread_capture = None
        if _flag('SPREAD_CAPTURE_AVAILABLE'):
            try:
                from polymarket_rtds import PolymarketRTDS
                from polymarket_spread_capture import SpreadCaptureEngine, ASSETS as SC_ASSETS
                bot.rtds = PolymarketRTDS()
                bot.spread_capture = SpreadCaptureEngine(rtds=bot.rtds)
                bot.logger.info(
                    f"Spread Capture: initialized "
                    f"(0x8dxd strategy — {len(SC_ASSETS)} assets, 5m+15m, CLOB limit orders)"
                )
            except Exception as _sc_err:
                bot.logger.warning(f"Spread Capture init failed: {_sc_err}")

        # Sub-Bar Scanner
        bot.sub_bar_scanner = None
        _sub_bar_cfg = bot.config.get('sub_bar_scanner', {})
        if _flag('SUB_BAR_SCANNER_AVAILABLE') and _sub_bar_cfg.get('enabled', False):
            try:
                from sub_bar_scanner import SubBarScanner
                bot.sub_bar_scanner = SubBarScanner(config=_sub_bar_cfg, db_path=db_path)
                bot.logger.info(
                    f"Sub-bar scanner initialized "
                    f"(observation_mode={_sub_bar_cfg.get('observation_mode', True)})"
                )
            except Exception as _sbs_err:
                bot.logger.warning(f"Sub-bar scanner init failed: {_sbs_err}")

        # Agent Coordinator
        bot.agent_coordinator = None
        if _flag('AGENT_COORDINATOR_AVAILABLE'):
            try:
                from agents.coordinator import AgentCoordinator
                bot.agent_coordinator = AgentCoordinator(
                    bot=bot, db_path=db_path, config=bot.config, bot_logger=bot.logger,
                )
            except Exception as _ac_err:
                bot.logger.warning(f"AgentCoordinator init failed (trading unaffected): {_ac_err}")

        # Token Spray Engine
        bot.token_spray = None
        spray_config = bot.config.get('token_spray', {})
        if spray_config.get('enabled', False):
            try:
                from token_spray_engine import TokenSprayEngine
                bot.token_spray = TokenSprayEngine(config=spray_config, db_path=db_path, logger=bot.logger)
            except Exception as _spray_err:
                bot.logger.warning(f"TokenSprayEngine init failed: {_spray_err}")

        # Straddle Fleet
        bot.straddle_engine = None
        bot.straddle_engines: Dict[str, Any] = {}
        bot.straddle_fleet = None
        straddles_config = bot.config.get('straddles', {})
        if straddles_config.get('enabled', False):
            try:
                from straddle_engine import StraddleEngine, StraddleFleetController
                bot.straddle_fleet = StraddleFleetController(
                    fleet_daily_loss_limit=float(straddles_config.get('fleet_daily_loss_limit', 1500)),
                    fleet_max_deployed=float(straddles_config.get('fleet_max_deployed', 15000)),
                    logger=bot.logger,
                )
                for asset_key, asset_cfg in straddles_config.get('assets', {}).items():
                    if not asset_cfg.get('enabled', False):
                        continue
                    eng = StraddleEngine(
                        config=asset_cfg, db_path=db_path,
                        logger=bot.logger, fleet_controller=bot.straddle_fleet,
                    )
                    bot.straddle_fleet.register(eng)
                    bot.straddle_engines[asset_key] = eng
                if bot.straddle_engines:
                    bot.straddle_engine = next(iter(bot.straddle_engines.values()))
                bot.logger.info(f"StraddleFleet initialized: {list(bot.straddle_engines.keys())}")
            except Exception as _straddle_err:
                bot.logger.warning(f"StraddleFleet init failed: {_straddle_err}")
        # Legacy fallback
        if not bot.straddle_engines:
            old_straddle_config = bot.config.get('straddle', {})
            if old_straddle_config.get('enabled', False):
                try:
                    from straddle_engine import StraddleEngine
                    bot.straddle_engine = StraddleEngine(
                        config=old_straddle_config, db_path=db_path, logger=bot.logger,
                    )
                    bot.straddle_engines[bot.straddle_engine.asset] = bot.straddle_engine
                except Exception as _straddle_err:
                    bot.logger.warning(f"StraddleEngine init failed: {_straddle_err}")

        # Oracle
        bot.oracle = None
        try:
            from oracle.oracle_service import OracleService
            bot.oracle = OracleService(db_path=db_path)
            bot.logger.info(f"Oracle service initialized with {bot.oracle.model_count} models (lazy-load)")
        except Exception as _oracle_err:
            bot.logger.warning(f"Oracle init failed (will run without oracle): {_oracle_err}")

        # Oracle Trading Engine
        bot.oracle_trader = None
        _ot_config = bot.config.get('oracle_trading', {})
        if _ot_config.get('enabled', False) and bot.oracle:
            try:
                from oracle.oracle_trading_engine import OracleTradingEngine
                bot.oracle_trader = OracleTradingEngine(
                    config=_ot_config, oracle=bot.oracle, db_path=db_path,
                )
                bot.logger.info(
                    f"Oracle Trading Engine: {len(bot.oracle_trader.wallets)} pairs, "
                    f"${_ot_config.get('wallet_size', 5000):,}/pair"
                )
            except Exception as _ot_err:
                bot.logger.warning(f"Oracle Trading Engine init failed: {_ot_err}")

        bot.logger.info("Renaissance Trading Bot initialized with research-optimized weights")
        bot.logger.info(f"Signal weights: {bot.signal_weights}")
