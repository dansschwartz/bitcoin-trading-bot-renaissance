"""
Renaissance Technologies Bitcoin Trading Bot - Main Integration
Combines all components with research-optimized signal weights
"""

import asyncio
import logging
import logging.handlers
import json
import os
import queue
import signal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time

# Import all Renaissance components
from enhanced_config_manager import EnhancedConfigManager
from microstructure_engine import MicrostructureEngine, MicrostructureMetrics
from enhanced_technical_indicators import EnhancedTechnicalIndicators, IndicatorOutput
from data_providers.market_data_provider import LiveMarketDataProvider
from data_providers.derivatives_data_provider import DerivativesDataProvider
from renaissance_signal_fusion import RenaissanceSignalFusion
from data_providers.alternative_data_engine import AlternativeDataEngine, AlternativeSignal

# ── Regime Detection Hierarchy ──
# PRIMARY (drives all trading decisions):
#   regime_overlay.py → RegimeOverlay
#     Reads OHLCV bars from five_minute_bars table, runs Bootstrap (<200 bars)
#     or 5-state HMM (≥200 bars).  Every confidence boost, entry-threshold bias,
#     signal-weight adjustment, and position-sizing scalar reads from this class.
#     This is the ONLY regime source written to decisions.hmm_regime.
#
# OBSERVATION ONLY (logged but never used for decisions):
#   macro_regime_detector.py  → MacroRegimeDetector  (Dalio SPX/VIX/DXY, 4-state)
#   crypto_regime_detector.py → CryptoRegimeDetector  (EMA/funding/OI, 4-state)
#   model_router.py           → ModelRouter            (routes regime tuples → model config)
#   intelligence/regime_detector.py → MedallionRegimeDetector (3-state HMM, pickle model)
#
# INTERNAL (consumed inside RegimeOverlay, not called from this file):
#   advanced_regime_detector.py    → AdvancedRegimeDetector  (5-state HMM engine)
#   medallion_regime_predictor.py  → MedallionRegimePredictor (3-state, hmm_forecast field)
from regime_overlay import RegimeOverlay
from risk_gateway import RiskGateway
from real_time_pipeline import RealTimePipeline

# Step 10 Experimental Suite
# from renaissance_portfolio_optimizer import RenaissancePortfolioOptimizer
from execution_algorithm_suite import ExecutionAlgorithmSuite
from slippage_protection_system import SlippageProtectionSystem

# Persistence & Attribution
from database_manager import DatabaseManager
from performance_attribution_engine import PerformanceAttributionEngine

# Order Execution & Position Management
from data_providers.coinbase_client import EnhancedCoinbaseClient, CoinbaseCredentials
from position_manager import EnhancedPositionManager, RiskLimits, PositionStatus
# Legacy AlertManager (Slack webhook, always on) — lightweight fallback.
# The canonical alert system is monitoring.alert_manager.AlertManager (Telegram + persistence).
from alert_manager import AlertManager
from data_providers.coinbase_advanced_client import CoinbaseAdvancedClient
from logger import SecretMaskingFilter
from data_providers.binance_spot_provider import BinanceSpotProvider, to_binance_symbol, from_binance_symbol

# Step 14 & 16 & Deep Alternative
from genetic_optimizer import GeneticWeightOptimizer
from cross_asset_engine import CrossAssetCorrelationEngine
from data_providers.whale_activity_monitor import WhaleActivityMonitor
from strategies.breakout_scanner import BreakoutScanner, BreakoutSignal
try:
    from breakout_strategy import BreakoutStrategy
    BREAKOUT_STRATEGY_AVAILABLE = True
except ImportError as _bs_err:
    BREAKOUT_STRATEGY_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Breakout Strategy import failed: {_bs_err}")
from polymarket_bridge import PolymarketBridge
from polymarket_scanner import PolymarketScanner
try:
    from cascade_data_collector import CascadeDataCollector
    CASCADE_COLLECTOR_AVAILABLE = True
except ImportError:
    CASCADE_COLLECTOR_AVAILABLE = False
# DISABLED: Old Polymarket strategies — replaced by spread capture
# from polymarket_strategy_a import StrategyAExecutor
STRATEGY_A_AVAILABLE = False
# from polymarket_live_executor import PolymarketLiveExecutor
LIVE_EXECUTOR_AVAILABLE = False
# from polymarket_reversal import ReversalStrategy
REVERSAL_STRATEGY_AVAILABLE = False
# from simple_up_bet import SimpleUpBetter
SIMPLE_UP_AVAILABLE = False

# NEW: 0x8dxd-style spread capture v2 — passive limit orders on both sides
try:
    from polymarket_rtds import PolymarketRTDS
    from polymarket_spread_capture import SpreadCaptureEngine, ASSETS as SC_ASSETS
    SPREAD_CAPTURE_AVAILABLE = True
except ImportError as _sc_err:
    SPREAD_CAPTURE_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Spread Capture import failed: {_sc_err}")
try:
    from sub_bar_scanner import SubBarScanner
    SUB_BAR_SCANNER_AVAILABLE = True
except ImportError:
    SUB_BAR_SCANNER_AVAILABLE = False
from volume_profile_engine import VolumeProfileEngine
from fractal_intelligence import FractalIntelligenceEngine
from market_entropy_engine import MarketEntropyEngine
from quantum_oscillator_engine import QuantumOscillatorEngine
from ghost_runner import GhostRunner
from self_reinforcing_learning import SelfReinforcingLearningEngine
from confluence_engine import ConfluenceEngine
from strategies.basis_trading_engine import BasisTradingEngine
from deep_nlp_bridge import DeepNLPBridge
from strategies.market_making_engine import MarketMakingEngine
from strategies.meta_strategy_selector import MetaStrategySelector
from institutional_dashboard import InstitutionalDashboard
from dashboard.event_emitter import DashboardEventEmitter
from position_sizer import RenaissancePositionSizer
from random_baseline import RandomEntryBaseline

from renaissance_types import SignalType, OrderType, MLSignalPackage, TradingDecision
from ml.ml_integration_bridge import MLIntegrationBridge
from decision_audit_logger import DecisionAuditLogger

# Renaissance Medallion-Style Engines
from advanced_mean_reversion_engine import AdvancedMeanReversionEngine
from correlation_network_engine import CorrelationNetworkEngine
from garch_volatility_engine import GARCHVolatilityEngine
from data_providers.historical_data_cache import HistoricalDataCache

# Production Orchestrator (optional)
try:
    from production_trading_orchestrator import ProductionTradingOrchestrator, ProductionConfig
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

# Types moved to renaissance_types.py


from renaissance_engine_core import SignalFusion, RiskManager

# Multi-Exchange Arbitrage Engine
try:
    from arbitrage.orchestrator import ArbitrageOrchestrator
    ARBITRAGE_AVAILABLE = True
except ImportError:
    ARBITRAGE_AVAILABLE = False

# ── Operations & Intelligence Modules ──
try:
    from recovery.state_manager import StateManager, SystemState
    from recovery.shutdown import GracefulShutdownHandler
    from recovery.database import ensure_all_tables
    RECOVERY_AVAILABLE = True
except ImportError:
    RECOVERY_AVAILABLE = False

# Canonical alert system — Telegram routing, dedup, SQLite persistence (feature-gated)
try:
    from monitoring.telegram_bot import TelegramAlerter
    from monitoring.alert_manager import AlertManager as MonitoringAlertManager
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

try:
    from signals.liquidation_detector import LiquidationCascadeDetector
    LIQUIDATION_DETECTOR_AVAILABLE = True
except ImportError:
    LIQUIDATION_DETECTOR_AVAILABLE = False

try:
    from signals.signal_aggregator import SignalAggregator
    SIGNAL_AGGREGATOR_AVAILABLE = True
except ImportError:
    SIGNAL_AGGREGATOR_AVAILABLE = False

try:
    from signals.multi_exchange_bridge import MultiExchangeBridge
    MULTI_EXCHANGE_BRIDGE_AVAILABLE = True
except ImportError:
    MULTI_EXCHANGE_BRIDGE_AVAILABLE = False

try:
    from data_validator import DataValidator
    DATA_VALIDATOR_AVAILABLE = True
except ImportError:
    DATA_VALIDATOR_AVAILABLE = False

try:
    from signal_auto_throttle import SignalAutoThrottle
    SIGNAL_THROTTLE_AVAILABLE = True
except ImportError:
    SIGNAL_THROTTLE_AVAILABLE = False

try:
    from signal_validation_gate import SignalValidationGate
    SIGNAL_VALIDATION_AVAILABLE = True
except ImportError:
    SIGNAL_VALIDATION_AVAILABLE = False

try:
    from portfolio_health_monitor import PortfolioHealthMonitor
    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HEALTH_MONITOR_AVAILABLE = False

try:
    from medallion_signal_analogs import MedallionSignalAnalogs
    MEDALLION_ANALOGS_AVAILABLE = True
except ImportError:
    MEDALLION_ANALOGS_AVAILABLE = False

try:
    from unified_portfolio_engine import UnifiedPortfolioEngine
    PORTFOLIO_ENGINE_AVAILABLE = True
except ImportError:
    PORTFOLIO_ENGINE_AVAILABLE = False

# ── Medallion Intelligence Modules (Phase 2 Build) ──
try:
    from core.devil_tracker import DevilTracker
    DEVIL_TRACKER_AVAILABLE = True
except ImportError:
    DEVIL_TRACKER_AVAILABLE = False

try:
    from core.kelly_position_sizer import KellyPositionSizer
    KELLY_SIZER_AVAILABLE = True
except ImportError:
    KELLY_SIZER_AVAILABLE = False

try:
    from core.signal_throttle import SignalThrottle as MedallionSignalThrottle
    MEDALLION_THROTTLE_AVAILABLE = True
except ImportError:
    MEDALLION_THROTTLE_AVAILABLE = False

try:
    from core.leverage_manager import LeverageManager
    LEVERAGE_MANAGER_AVAILABLE = True
except ImportError:
    LEVERAGE_MANAGER_AVAILABLE = False

try:
    from core.portfolio_engine import PortfolioEngine as MedallionPortfolioEngine
    MEDALLION_PORTFOLIO_ENGINE_AVAILABLE = True
except ImportError:
    MEDALLION_PORTFOLIO_ENGINE_AVAILABLE = False

# OBSERVATION ONLY — logs alongside RegimeOverlay for comparison; not on decision path
try:
    from intelligence.regime_detector import RegimeDetector as MedallionRegimeDetector
    MEDALLION_REGIME_AVAILABLE = True
except ImportError:
    MEDALLION_REGIME_AVAILABLE = False

try:
    from intelligence.insurance_scanner import InsurancePremiumScanner
    INSURANCE_SCANNER_AVAILABLE = True
except ImportError:
    INSURANCE_SCANNER_AVAILABLE = False

try:
    from data_module.bar_aggregator import BarAggregator
    BAR_AGGREGATOR_AVAILABLE = True
except ImportError:
    BAR_AGGREGATOR_AVAILABLE = False

try:
    from execution.synchronized_executor import SynchronizedExecutor
    SYNC_EXECUTOR_AVAILABLE = True
except ImportError:
    SYNC_EXECUTOR_AVAILABLE = False

try:
    from execution.trade_hider import TradeHider
    TRADE_HIDER_AVAILABLE = True
except ImportError:
    TRADE_HIDER_AVAILABLE = False

try:
    from intelligence.fast_mean_reversion import FastMeanReversionScanner
    FAST_REVERSION_AVAILABLE = True
except ImportError:
    FAST_REVERSION_AVAILABLE = False

try:
    from orchestrator.heartbeat import HeartbeatWriter
    HEARTBEAT_AVAILABLE = True
except ImportError:
    HEARTBEAT_AVAILABLE = False

try:
    from monitoring.beta_monitor import BetaMonitor
    BETA_MONITOR_AVAILABLE = True
except ImportError:
    BETA_MONITOR_AVAILABLE = False

try:
    from monitoring.capacity_monitor import CapacityMonitor
    CAPACITY_MONITOR_AVAILABLE = True
except ImportError:
    CAPACITY_MONITOR_AVAILABLE = False

try:
    from monitoring.sharpe_monitor import SharpeMonitor
    SHARPE_MONITOR_AVAILABLE = True
except ImportError:
    SHARPE_MONITOR_AVAILABLE = False

try:
    from portfolio.position_reevaluator import PositionReEvaluator
    from core.data_structures import PositionContext, ReEvalResult
    POSITION_REEVALUATOR_AVAILABLE = True
except ImportError:
    POSITION_REEVALUATOR_AVAILABLE = False

try:
    from intelligence.multi_horizon_estimator import MultiHorizonEstimator
    MHPE_AVAILABLE = True
except ImportError:
    MHPE_AVAILABLE = False

# Doc 15: Agent Coordination System
try:
    from agents.coordinator import AgentCoordinator
    AGENT_COORDINATOR_AVAILABLE = True
except ImportError:
    AGENT_COORDINATOR_AVAILABLE = False

# Hierarchical Regime Detection (Dalio-inspired) — OBSERVATION ONLY
# These detectors log alongside RegimeOverlay but do NOT influence trading decisions.
# ModelRouter is in Phase 1 (observation mode): logs regime-to-model mapping, does not enforce.
try:
    from macro_regime_detector import MacroRegimeDetector, MacroRegime
    from crypto_regime_detector import CryptoRegimeDetector, CryptoRegime
    from model_router import ModelRouter
    HIERARCHICAL_REGIME_AVAILABLE = True
except ImportError as _hr_err:
    HIERARCHICAL_REGIME_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Hierarchical regime import failed: {_hr_err}")

# Types moved to renaissance_types.py

# Signal conversion helpers — canonical implementations in bot.helpers
from bot.helpers import (
    signed_strength as _signed_strength,
    continuous_rsi_signal as _continuous_rsi_signal,
    continuous_macd_signal as _continuous_macd_signal,
    continuous_bollinger_signal as _continuous_bollinger_signal,
    continuous_obv_signal as _continuous_obv_signal,
    convert_ws_orderbook_to_snapshot as _convert_ws_orderbook_to_snapshot,
)

def validate_config(config: Dict[str, Any], logger_inst: logging.Logger) -> bool:
    """Validate config at startup: check required keys, warn on ambiguous duplicates, validate ranges."""
    warnings = []
    errors = []

    # ── Required sections ──
    required_sections = ["trading", "risk_management", "signal_weights", "database", "coinbase"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required config section: '{section}'")

    # ── Required keys within sections ──
    required_keys = {
        "trading": ["product_ids", "cycle_interval_seconds"],
        "risk_management": ["daily_loss_limit", "position_limit", "min_confidence"],
        "database": ["path"],
    }
    for section, keys in required_keys.items():
        sec = config.get(section, {})
        for key in keys:
            if key not in sec:
                errors.append(f"Missing required key: '{section}.{key}'")

    # ── Ambiguous duplicate keys ──
    # kelly_fraction in leverage_manager vs kelly_sizer
    lm_kelly = config.get("leverage_manager", {}).get("kelly_fraction")
    ks_kelly = config.get("kelly_sizer", {}).get("kelly_fraction")
    if lm_kelly is not None and ks_kelly is not None and lm_kelly != ks_kelly:
        warnings.append(
            f"kelly_fraction differs: leverage_manager={lm_kelly} vs kelly_sizer={ks_kelly}"
        )

    # max_leverage in leverage_manager vs medallion_portfolio_engine
    lm_lev = config.get("leverage_manager", {}).get("max_leverage")
    pe_lev = config.get("medallion_portfolio_engine", {}).get("max_leverage")
    if lm_lev is not None and pe_lev is not None and lm_lev != pe_lev:
        warnings.append(
            f"max_leverage differs: leverage_manager={lm_lev} vs medallion_portfolio_engine={pe_lev}"
        )

    # short_window/long_window in signal_throttle vs health_monitor (different semantics)
    st_sw = config.get("signal_throttle", {}).get("short_window")
    hm_sw = config.get("health_monitor", {}).get("short_window")
    if st_sw is not None and hm_sw is not None:
        warnings.append(
            f"short_window appears in both signal_throttle ({st_sw} cycles) and "
            f"health_monitor ({hm_sw} trades) — different semantics"
        )

    # ── Numeric range validation ──
    range_checks = [
        ("risk_management.daily_loss_limit", config.get("risk_management", {}).get("daily_loss_limit"), 0, 1_000_000),
        ("risk_management.min_confidence", config.get("risk_management", {}).get("min_confidence"), 0.0, 1.0),
        ("kelly_sizer.kelly_fraction", ks_kelly, 0.0, 1.0),
        ("kelly_sizer.max_position_pct", config.get("kelly_sizer", {}).get("max_position_pct"), 0.0, 100.0),
        ("portfolio_engine.max_concentration", config.get("portfolio_engine", {}).get("max_concentration"), 0.0, 1.0),
        ("portfolio_engine.max_total_exposure_pct", config.get("portfolio_engine", {}).get("max_total_exposure_pct"), 0.0, 1.0),
        ("medallion_portfolio_engine.drift_threshold_pct", config.get("medallion_portfolio_engine", {}).get("drift_threshold_pct"), 0.0, 100.0),
    ]
    for name, val, lo, hi in range_checks:
        if val is not None:
            if val < lo or val > hi:
                errors.append(f"{name}={val} out of valid range [{lo}, {hi}]")

    # ── Signal weight sum check ──
    sw = config.get("signal_weights", {})
    if sw:
        total = sum(sw.values())
        if abs(total - 1.0) > 0.05:
            warnings.append(f"signal_weights sum to {total:.3f} (expected ~1.0)")

    # ── Log results ──
    for w in warnings:
        logger_inst.warning(f"CONFIG WARNING: {w}")
    for e in errors:
        logger_inst.error(f"CONFIG ERROR: {e}")

    # ── Active module summary ──
    modules = [
        ("RegimeOverlay", config.get("regime_overlay", {}).get("enabled", False)),
        ("PortfolioEngine", config.get("portfolio_engine", {}).get("enabled", False)),
        ("HealthMonitor", config.get("health_monitor", {}).get("enabled", False)),
        ("SignalThrottle", config.get("signal_throttle", {}).get("enabled", False)),
        ("MedallionAnalogs", config.get("medallion_analogs", {}).get("enabled", False)),
        ("Arbitrage", config.get("arbitrage", {}).get("enabled", False)),
        ("LiquidationDetector", config.get("liquidation_detector", {}).get("enabled", False)),
        ("InsuranceScanner", config.get("insurance_scanner", {}).get("enabled", False)),
        ("MedallionRegime", config.get("medallion_regime_detector", {}).get("enabled", False)),
        ("BetaMonitor", config.get("beta_monitor", {}).get("enabled", False)),
        ("SharpeMonitor", config.get("sharpe_monitor", {}).get("enabled", False)),
        ("CapacityMonitor", config.get("capacity_monitor", {}).get("enabled", False)),
    ]
    active = [m for m, e in modules if e]
    inactive = [m for m, e in modules if not e]
    logger_inst.info(f"CONFIG SUMMARY: {len(active)} active modules: {', '.join(active) or 'none'}")
    if inactive:
        logger_inst.info(f"CONFIG SUMMARY: {len(inactive)} inactive: {', '.join(inactive)}")

    if errors:
        logger_inst.error(f"Config validation found {len(errors)} error(s), {len(warnings)} warning(s)")
        return False
    if warnings:
        logger_inst.warning(f"Config validation passed with {len(warnings)} warning(s)")
    else:
        logger_inst.info("Config validation passed — no issues found")
    return True


# MacroDataCache — canonical implementation moved to bot/data_collection.py
from bot.data_collection import MacroDataCache


class RenaissanceTradingBot:
    """
    Main Renaissance Technologies-style Bitcoin trading bot
    Integrates all components with research-optimized weights
    """

    def _force_float(self, val: Any) -> float:
        """Recursive unpacking and float casting for paranoid scalar hardening."""
        try:
            temp = val
            if temp is None:
                return 0.0
            while hasattr(temp, '__iter__') and not isinstance(temp, (str, bytes, dict)):
                if hasattr(temp, '__len__') and len(temp) > 0:
                    temp = temp[0]
                else:
                    temp = 0.0
                    break
            if hasattr(temp, 'item'): 
                temp = temp.item()
            return float(temp)
        except Exception:
            return 0.0

    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the Renaissance trading bot.

        Config loading and logging are set up here; all subsystem initialization
        is delegated to bot.builder.BotBuilder.build_all(self).
        """
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = Path(__file__).resolve().parent / self.config_path

        self.config = self._load_config(self.config_path)
        self.logger = self._setup_logging(self.config)
        self._validate_config(self.config)

        # Delegate all subsystem initialization to BotBuilder
        from bot.builder import BotBuilder
        BotBuilder.build_all(self)

    def get_pairs_for_cycle(self, cycle_number: int) -> list:
        """Return pairs to scan this cycle based on 4-tier volume schedule."""
        from bot.data_collection import get_pairs_for_cycle as _get_pairs
        return _get_pairs(self, cycle_number)

    async def _build_and_apply_universe(self) -> None:
        """Build dynamic trading universe from Binance and apply it."""
        from bot.data_collection import build_and_apply_universe as _build
        return await _build(self)

    def _ml_zscore_rescale(self, pair: str, raw_pred: float) -> float:
        """Council #12: Per-pair Z-score normalization of ML predictions."""
        from bot.data_collection import ml_zscore_rescale as _rescale
        return _rescale(self, pair, raw_pred)

    async def _gap_fill_bars_on_startup(self) -> None:
        """Council proposal #1: Fill missing 5-min bars from Binance on startup."""
        from bot.data_collection import gap_fill_bars_on_startup as _gap_fill
        return await _gap_fill(self)

    async def _collect_from_binance(self, product_id: str) -> Dict[str, Any]:
        """Collect market data from Binance for a single pair."""
        from bot.data_collection import collect_from_binance as _collect
        return await _collect(self, product_id)

    def _get_tech(self, product_id: str) -> 'EnhancedTechnicalIndicators':
        """Get per-asset technical indicators instance (creates on-demand for new assets)."""
        from bot.data_collection import get_tech as _get_tech
        return _get_tech(self, product_id)

    def _load_price_df_from_db(self, product_id: str, limit: int = 300):
        """Load recent OHLCV bars from DB for ML inference."""
        from bot.data_collection import load_price_df_from_db as _load
        return _load(self, product_id, limit)

    def _load_candles_from_db(self, product_id: str, limit: int = 200) -> List:
        """Load historical bars from five_minute_bars as PriceData objects."""
        from bot.data_collection import load_candles_from_db as _load
        return _load(self, product_id, limit)

    def _setup_logging(self, config: Dict[str, Any]) -> logging.Logger:
        """Setup logging — delegates to bot.helpers."""
        from bot.helpers import setup_logging
        return setup_logging(self, config)

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load config — delegates to bot.helpers."""
        from bot.helpers import load_config
        return load_config(self, config_path)

    def _validate_config(self, config: Dict[str, Any]):
        """Validate config — delegates to bot.helpers."""
        from bot.helpers import validate_config
        return validate_config(self, config)

    def _track_task(self, coro) -> asyncio.Task:
        """Create and track an asyncio task for graceful shutdown."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        return task

    @staticmethod
    def _compute_realized_pnl(entry_price: float, close_price: float,
                               size: float, side: str) -> float:
        """Compute realized PnL from entry/close prices and position side."""
        from bot.position_ops import compute_realized_pnl
        return compute_realized_pnl(entry_price, close_price, size, side)

    async def _resolve_close_price(self, pos) -> float:
        """Resolve the best available market price for closing a position."""
        from bot.position_ops import resolve_close_price
        return await resolve_close_price(self, pos)

    async def _shutdown(self):
        """Cancel background tasks — delegates to bot.lifecycle."""
        from bot.lifecycle import shutdown
        return await shutdown(self)

    HEARTBEAT_FILE = Path("logs/heartbeat.json")

    def _write_heartbeat(self):
        """Write heartbeat — delegates to bot.helpers."""
        from bot.helpers import write_heartbeat
        return write_heartbeat(self)

    async def _log_ml_accuracy_summary(self) -> None:
        """Log per-model accuracy summary — delegates to bot.adaptive."""
        from bot.adaptive import log_ml_accuracy_summary
        return await log_ml_accuracy_summary(self)

    async def _log_kelly_calibration(self) -> None:
        """Log Kelly calibration — delegates to bot.adaptive."""
        from bot.adaptive import log_kelly_calibration
        return await log_kelly_calibration(self)

    async def _check_pipeline_health(self) -> None:
        """Pipeline watchdog — delegates to bot.adaptive."""
        from bot.adaptive import check_pipeline_health
        return await check_pipeline_health(self)

    async def collect_all_data(self, product_id: str = "BTC-USD") -> Dict[str, Any]:
        """Collect data from all sources for a specific product."""
        from bot.signals import collect_all_data as _collect_all_data
        return await _collect_all_data(self, product_id)

    async def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate signals from all components."""
        from bot.signals import generate_signals as _generate_signals
        return await _generate_signals(self, market_data)

    def calculate_weighted_signal(self, signals: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Calculate final weighted signal using Renaissance weights."""
        from bot.signals import calculate_weighted_signal as _calculate_weighted_signal
        return _calculate_weighted_signal(self, signals)

    def _calculate_dynamic_position_size(self, product_id: str, confidence: float, weighted_signal: float, current_price: float) -> float:
        """Calculate dynamic position size using Step 10 Portfolio Optimizer"""
        from bot.position_ops import calculate_dynamic_position_size
        return calculate_dynamic_position_size(self, product_id, confidence, weighted_signal, current_price)

    def make_trading_decision(self, weighted_signal: float, signal_contributions: Dict[str, float],
                              current_price: float = 0.0, real_time_result: Optional[Dict[str, Any]] = None,
                              product_id: str = "BTC-USD", ml_package: Optional[MLSignalPackage] = None,
                              market_data: Optional[Dict[str, Any]] = None,
                              drawdown_pct: float = 0.0,
                              audit_logger: Optional['DecisionAuditLogger'] = None) -> TradingDecision:
        """Make final trading decision with Renaissance methodology + Kelly position sizing."""
        from bot.decision import make_trading_decision as _make_trading_decision
        return _make_trading_decision(
            self, weighted_signal, signal_contributions,
            current_price=current_price, real_time_result=real_time_result,
            product_id=product_id, ml_package=ml_package,
            market_data=market_data, drawdown_pct=drawdown_pct,
            audit_logger=audit_logger,
        )

    async def _execute_smart_order(self, decision: TradingDecision, market_data: Dict[str, Any]):
        """Execute order through position manager — delegates to bot.position_ops."""
        from bot.position_ops import execute_smart_order
        return await execute_smart_order(self, decision, market_data)

    async def _run_adaptive_learning_cycle(self):
        """Adaptive learning — delegates to bot.adaptive."""
        from bot.adaptive import run_adaptive_learning_cycle
        return await run_adaptive_learning_cycle(self)

    def _save_optimized_weights(self, weights: Dict[str, float]):
        """Persist optimized weights — delegates to bot.adaptive."""
        from bot.adaptive import save_optimized_weights
        return save_optimized_weights(self, weights)

    async def _perform_attribution_analysis(self):
        """Performance attribution — delegates to bot.adaptive."""
        from bot.adaptive import perform_attribution_analysis
        return await perform_attribution_analysis(self)

    def _fetch_account_balance(self) -> float:
        """Fetch current USD account balance — delegates to bot.position_ops."""
        from bot.position_ops import fetch_account_balance
        return fetch_account_balance(self)

    def _check_bar_liveness(self) -> None:
        """Bar pipeline liveness check — delegates to bot.adaptive."""
        from bot.adaptive import check_bar_liveness
        return check_bar_liveness(self)

    async def execute_trading_cycle(self) -> TradingDecision:
        """Execute one complete trading cycle across all products"""
        cycle_start = time.time()
        decisions = []
        # Straddle opens are handled by each engine's _exit_loop (every 2s).
        # No need to open straddles in the main bot cycle.

        try:
            # Council S6: Check bar pipeline liveness at start of each cycle
            self._check_bar_liveness()

            # Fetch live account balance for dynamic position sizing
            account_balance = self._fetch_account_balance()
            self.logger.info(f"Account balance: ${account_balance:,.2f}")

            # Dynamically update position manager limits — selective: 1 per product, 10 max
            # Each trade is ~$1K (10% of $10K), 50% cap → ~5 simultaneous max from exposure
            self.position_manager.risk_limits.max_position_size_usd = account_balance * 0.10
            self.position_manager.risk_limits.max_total_exposure_usd = account_balance * 0.50
            _base_max_positions = min(len(self.product_ids), 10)
            # Council S3: Reduce max positions when eigenvalue concentration is high
            if (hasattr(self, 'correlation_network') and self.correlation_network.enabled
                    and self.correlation_network.should_reduce_positions()):
                _reduced = max(3, int(_base_max_positions * 0.6))
                self.logger.info(
                    f"EIGENVALUE CONCENTRATION: reducing max_positions "
                    f"{_base_max_positions} -> {_reduced} "
                    f"(ratio={self.correlation_network.get_eigenvalue_ratio():.3f})"
                )
                _base_max_positions = _reduced
            self.position_manager.risk_limits.max_total_positions = _base_max_positions
            self.position_manager.risk_limits.max_positions_per_product = 1

            # ── Drawdown tracking (Renaissance discipline) ──
            if account_balance > self._high_watermark_usd:
                self._high_watermark_usd = account_balance
            if self._high_watermark_usd > 0:
                self._current_drawdown_pct = (self._high_watermark_usd - account_balance) / self._high_watermark_usd
            else:
                self._current_drawdown_pct = 0.0

            # Weekly loss tracking — reset on Monday
            now = datetime.now(timezone.utc)
            if self._week_start_balance <= 0:
                self._week_start_balance = account_balance
            if now.weekday() == 0 and not getattr(self, '_week_reset_today', False):
                self._week_start_balance = account_balance
                self._weekly_pnl = 0.0
                self._week_reset_today = True
            elif now.weekday() != 0:
                self._week_reset_today = False
            self._weekly_pnl = account_balance - self._week_start_balance

            # ── Council S4: Periodic balance snapshot (every cycle) ──
            try:
                with self.position_manager._lock:
                    _open_count = len([p for p in self.position_manager.positions.values()
                                       if hasattr(p, 'status') and str(p.status).upper() != 'CLOSED'])
                _unrealized = sum(
                    getattr(p, 'unrealized_pnl', 0.0) or 0.0
                    for p in self.position_manager.positions.values()
                ) if hasattr(self.position_manager, 'positions') else 0.0
                self._track_task(self.db_manager.store_balance_snapshot(
                    total_equity=account_balance,
                    unrealized_pnl=_unrealized,
                    open_position_count=_open_count,
                    cash_balance=account_balance - _unrealized,
                    drawdown_pct=self._current_drawdown_pct,
                    high_watermark=self._high_watermark_usd,
                    daily_pnl=getattr(self, 'daily_pnl', 0.0),
                    source='periodic',
                ))
            except Exception as e:
                self.logger.debug(f"Balance snapshot failed: {e}")

            if self._current_drawdown_pct >= 0.03:
                self.logger.warning(
                    f"DRAWDOWN ALERT: {self._current_drawdown_pct:.1%} from HWM ${self._high_watermark_usd:,.2f} | "
                    f"Weekly P&L: ${self._weekly_pnl:,.2f}"
                )

            # ── Drawdown Circuit Breaker + Exposure Monitor ──
            from bot.cycle_ops import apply_drawdown_controls
            await apply_drawdown_controls(self, account_balance)

            # ── Build/refresh dynamic universe (weekly refresh) ──
            if not self._universe_built or (time.time() - self._universe_last_refresh > 86400):
                await self._build_and_apply_universe()

            # ══════════════════════════════════════════════════
            # PHASE 0: BREAKOUT SCAN (1 API call, ~1-2 seconds)
            # ══════════════════════════════════════════════════
            if self.scanner_enabled:
                try:
                    breakout_signals = await self.breakout_scanner.scan()
                    self._breakout_scores = {s.product_id: s for s in breakout_signals}
                except Exception as _bs_err:
                    self.logger.warning(f"Breakout scan failed: {_bs_err}")
                    breakout_signals = []
                    self._breakout_scores = {}

                # Build deep-scan list: always-scan majors + breakout flagged pairs
                always_pairs = self.breakout_scanner.get_always_scan_pairs()
                breakout_pairs = [s.product_id for s in breakout_signals]

                # Include pairs with open positions so exit engine always runs
                try:
                    with self.position_manager._lock:
                        open_pos_pairs = [p.product_id for p in self.position_manager.positions.values()]
                except Exception:
                    open_pos_pairs = []

                # Combine and deduplicate, preserving order (majors first, then breakouts, then open positions)
                cycle_pairs = list(dict.fromkeys(always_pairs + breakout_pairs + open_pos_pairs))

                n_open = len(set(open_pos_pairs) - set(always_pairs) - set(breakout_pairs))
                self.logger.info(
                    f"SCAN PLAN: {len(cycle_pairs)} pairs for deep scan "
                    f"({len(always_pairs)} majors + {len(breakout_signals)} breakouts + {n_open} open-pos)"
                )

                # Persist breakout signals to DB for dashboard consumption
                try:
                    _scan_time = datetime.now(timezone.utc).isoformat()
                    _total_scanned = getattr(self.breakout_scanner, 'last_scan_count', len(cycle_pairs))
                    _db_path = str(self.db_manager.db_path) if hasattr(self.db_manager, 'db_path') else "data/renaissance_bot.db"
                    import sqlite3 as _bo_sql
                    _bo_conn = _bo_sql.connect(_db_path)
                    _bo_conn.execute("""
                        CREATE TABLE IF NOT EXISTS breakout_scans (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            scan_time TEXT NOT NULL,
                            product_id TEXT NOT NULL,
                            symbol TEXT,
                            score REAL,
                            direction TEXT,
                            price REAL,
                            volume_24h_usd REAL,
                            price_change_pct REAL,
                            volume_score REAL,
                            price_score REAL,
                            momentum_score REAL,
                            volatility_score REAL,
                            divergence_score REAL,
                            total_scanned INTEGER,
                            UNIQUE(scan_time, product_id)
                        )
                    """)
                    _bo_conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_breakout_scan_time ON breakout_scans(scan_time)"
                    )
                    _bo_rows = [
                        (
                            _scan_time, s.product_id, s.symbol,
                            round(s.breakout_score, 1), s.direction, s.price,
                            round(s.volume_24h_usd, 2), round(s.price_change_pct, 2),
                            round(s.volume_score, 1), round(s.price_score, 1),
                            round(s.momentum_score, 1), round(s.volatility_score, 1),
                            round(s.divergence_score, 1), _total_scanned,
                        )
                        for s in breakout_signals
                    ]
                    _bo_conn.executemany("""
                        INSERT OR REPLACE INTO breakout_scans
                        (scan_time, product_id, symbol, score, direction, price,
                         volume_24h_usd, price_change_pct, volume_score, price_score,
                         momentum_score, volatility_score, divergence_score, total_scanned)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, _bo_rows)
                    # Prune scans older than 24h to keep DB lean
                    _bo_conn.execute(
                        "DELETE FROM breakout_scans WHERE scan_time < datetime('now', '-24 hours')"
                    )
                    _bo_conn.commit()
                    _bo_conn.close()
                except Exception as _be:
                    self.logger.debug(f"Breakout DB persist error: {_be}")

                # ── Run Breakout Strategy on signals ──
                if self.breakout_strategy and breakout_signals:
                    try:
                        _bst_prices = {s.product_id: s.price for s in breakout_signals}
                        # Merge with any cached prices from the main pipeline
                        if hasattr(self, '_last_prices') and self._last_prices:
                            _bst_prices.update(self._last_prices)
                        bst_result = self.breakout_strategy.execute_cycle(
                            breakout_signals, _bst_prices,
                        )
                        if bst_result.get("entries") or bst_result.get("exits"):
                            self.logger.info(
                                f"BREAKOUT STRATEGY: {len(bst_result.get('entries', []))} entries, "
                                f"{len(bst_result.get('exits', []))} exits"
                            )
                    except Exception as _bst_err:
                        self.logger.warning(f"Breakout strategy cycle error: {_bst_err}")

            else:
                # Fallback to tiered scanning if breakout scanner disabled
                cycle_pairs = self.get_pairs_for_cycle(self.scan_cycle_count)
                self.logger.info(
                    f"Cycle {self.scan_cycle_count}: scanning {len(cycle_pairs)}/{len(self.product_ids)} pairs"
                )

            # ── Preload candle history on first cycle (eliminates cold-start) ──
            if not getattr(self, '_history_preloaded', False):
                self._history_preloaded = True
                from bot.cycle_ops import preload_candle_history
                await preload_candle_history(self)

            # ══════════════════════════════════════════════════════
            # PHASE 1: PARALLEL DATA FETCH (all pairs concurrently)
            # ══════════════════════════════════════════════════════
            fetch_start = time.time()

            sem = asyncio.Semaphore(15)  # Max 15 concurrent Binance requests

            async def _fetch_one(pid: str):
                async with sem:
                    try:
                        return (pid, await self.collect_all_data(pid))
                    except Exception as e:
                        self.logger.debug(f"Parallel fetch failed for {pid}: {e}")
                        return (pid, {})

            fetch_results = await asyncio.gather(
                *[_fetch_one(pid) for pid in cycle_pairs],
                return_exceptions=True,
            )

            # Build dict of successful fetches
            market_data_all: Dict[str, Dict[str, Any]] = {}
            for result in fetch_results:
                if isinstance(result, Exception):
                    continue
                pid, data = result
                if data:
                    market_data_all[pid] = data

            fetch_elapsed = time.time() - fetch_start
            self.logger.info(
                f"PARALLEL FETCH: {len(market_data_all)}/{len(cycle_pairs)} pairs "
                f"in {fetch_elapsed:.1f}s"
            )

            # Council S2 P1: Record bar_aggregator heartbeat after data fetch
            self._track_task(self.db_manager.record_heartbeat(
                'bar_aggregator', items_processed=len(market_data_all),
                details={'fetch_secs': round(fetch_elapsed, 1), 'pairs_requested': len(cycle_pairs)}
            ))

            # ══════════════════════════════════════════════════════
            # PHASE 2: SEQUENTIAL PROCESSING (signals + decisions)
            # ══════════════════════════════════════════════════════

            # ── Build cross-asset data dict for ML cross-pair features ──
            cross_data = {}
            for _pid in cycle_pairs:
                try:
                    _tech = self._get_tech(_pid)
                    _cdf = _tech._to_dataframe()
                    if _cdf is None or len(_cdf) < 30:
                        _cdf = self._load_price_df_from_db(_pid, limit=300)
                    if _cdf is not None and len(_cdf) > 0:
                        cross_data[_pid] = _cdf
                except Exception as e:
                    self.logger.warning(f"Cross-asset data loading failed for {_pid}: {e}")
            # Save for Strategy A timing features (BTC lead-lag)
            self._latest_cross_data = cross_data

            # ── Hierarchical Regime Classification (once per cycle) ──
            from bot.cycle_ops import classify_hierarchical_regime
            _cycle_regime_state = classify_hierarchical_regime(self, cross_data, market_data_all)

            _ml_inference_count = 0  # Council S2 P1: track ML inferences for heartbeat

            for product_id in cycle_pairs:
                pair_start_time = time.time()

                # 1. Use pre-fetched market data from Phase 1
                market_data = market_data_all.get(product_id)
                if not market_data:
                    continue

                if not market_data:
                    self.logger.warning(f"No market data for {product_id}, skipping")
                    continue

                # ── Data Quality Gate ──
                if self.data_validator and not self.data_validator.validate_market_data(market_data, product_id):
                    self.logger.warning(f"DATA QUALITY GATE: {product_id} failed validation, skipping cycle")
                    continue

                # Standardize price key
                ticker = market_data.get('ticker', {})
                current_price = float(ticker.get('price', 0.0))
                if current_price == 0:
                    current_price = float(ticker.get('last', 0.0))
                    market_data['ticker']['price'] = current_price # Standardize

                # Wire current_price into market_data for all downstream consumers
                market_data['current_price'] = current_price

                # Feed price to portfolio engine for correlation tracking
                if self.portfolio_engine and current_price > 0:
                    self.portfolio_engine.update_price(product_id, current_price)

                # Feed price to paper trader so fills use correct price
                if current_price > 0 and hasattr(self.coinbase_client, 'paper_trader') and self.coinbase_client.paper_trader:
                    self.coinbase_client.paper_trader.update_price(product_id, current_price)

                # Update position manager with current price (required for PnL calc)
                if current_price > 0:
                    self.position_manager.update_positions({product_id: current_price})

                # ── Warmup Gate: new pairs need 30 bars before trading ──
                _tech_warmup = self._get_tech(product_id)
                if len(_tech_warmup.price_history) < 30:
                    # High-score breakout pairs get instant warmup via candle fetch
                    _bo_info = self._breakout_scores.get(product_id)
                    if _bo_info and _bo_info.breakout_score >= 50:
                        try:
                            _bo_sym = product_id.split('-')[0] + 'USDT'
                            _bo_candles = await self.binance_spot.fetch_candles(_bo_sym, '5m', 200)
                            if _bo_candles:
                                from enhanced_technical_indicators import PriceData
                                for c in _bo_candles:
                                    pd_obj = PriceData(
                                        timestamp=datetime.utcfromtimestamp(c['timestamp']),
                                        open=c['open'], high=c['high'],
                                        low=c['low'], close=c['close'],
                                        volume=c['volume'],
                                    )
                                    _tech_warmup.update_price_data(pd_obj)
                                    if c['close'] > 0:
                                        self.garch_engine.update_returns(product_id, c['close'])
                                        self.stat_arb_engine.update_price(product_id, c['close'])
                                self.logger.info(
                                    f"BREAKOUT WARMUP: {product_id} loaded {len(_bo_candles)} candles "
                                    f"(score={_bo_info.breakout_score:.0f})"
                                )
                                # Re-check after warmup
                                if len(_tech_warmup.price_history) < 30:
                                    continue
                            else:
                                continue
                        except Exception as _bw_err:
                            self.logger.debug(f"Breakout warmup failed for {product_id}: {_bw_err}")
                            continue
                    else:
                        self.logger.debug(
                            f"WARMUP: {product_id} has {len(_tech_warmup.price_history)} bars, need 30 — collecting data only"
                        )
                        # Still feed price engines so warmup progresses
                        if current_price > 0:
                            self.stat_arb_engine.update_price(product_id, current_price)
                            self.correlation_network.update_price(product_id, current_price)
                            self.garch_engine.update_returns(product_id, current_price)
                        pair_elapsed = time.time() - pair_start_time
                        if pair_elapsed > 10:
                            self.logger.warning(f"SLOW PAIR: {product_id} took {pair_elapsed:.1f}s (warmup)")
                        continue

                # ── Council S3: Feature Staleness Circuit Breaker ──
                # Decay confidence based on how old the pair's newest bar is
                _staleness_decay = 1.0
                try:
                    _tech = self._get_tech(product_id)
                    if _tech.price_history and hasattr(_tech.price_history[-1], 'timestamp'):
                        _last_bar_ts = _tech.price_history[-1].timestamp
                        _staleness_sec = (datetime.now() - _last_bar_ts).total_seconds()
                        _staleness_min = _staleness_sec / 60.0
                        if _staleness_min > 10:
                            import math as _math
                            _staleness_decay = _math.exp(-_staleness_min / 30.0)  # tau=30min
                            self.logger.warning(
                                f"STALENESS: {product_id} last bar {_staleness_min:.0f}min old — "
                                f"confidence decay={_staleness_decay:.2f}"
                            )
                        # Hard cutoff: skip trading entirely if data is too stale
                        if _staleness_min > 30:
                            self.logger.info(
                                f"STALE SKIP: {product_id} data is {_staleness_min:.0f}min old — skipping trade decisions"
                            )
                            market_data['_skip_stale'] = True
                        market_data['_staleness_minutes'] = _staleness_min
                        market_data['_staleness_decay'] = _staleness_decay
                except Exception as e:
                    self.logger.warning(f"Data staleness check failed for {product_id}: {e}")

                # ── Signal Scorecard: evaluate last cycle's predictions ──
                if product_id in self._pending_predictions and current_price > 0:
                    pred = self._pending_predictions[product_id]
                    prev_price = pred.get('price', 0)
                    if prev_price > 0:
                        actual_move = (current_price - prev_price) / prev_price
                        sc = self._signal_scorecard.setdefault(product_id, {})
                        for sig_name, sig_val in pred.get('signals', {}).items():
                            if abs(sig_val) < 0.01:
                                continue  # Skip near-zero signals
                            entry = sc.setdefault(sig_name, {"correct": 0, "total": 0})
                            entry["total"] += 1
                            # Signal was correct if direction matches price move
                            if (sig_val > 0 and actual_move > 0) or (sig_val < 0 and actual_move < 0):
                                entry["correct"] += 1

                    # Feed auto-throttle with performance data
                    if self.signal_throttle and prev_price > 0:
                        self.signal_throttle.update(product_id, pred.get('signals', {}), actual_move)

                    # Log scorecard every 20 cycles
                    cycle_num = getattr(self, 'scan_cycle_count', 0)
                    if cycle_num > 0 and cycle_num % 20 == 0 and product_id in self._signal_scorecard:
                        sc = self._signal_scorecard[product_id]
                        scored = {k: f"{v['correct']}/{v['total']} ({100*v['correct']/max(v['total'],1):.0f}%)"
                                  for k, v in sorted(sc.items(), key=lambda x: x[1]['correct']/max(x[1]['total'],1), reverse=True)
                                  if v['total'] >= 5}
                        if scored:
                            self.logger.info(f"SIGNAL SCORECARD [{product_id}]: {scored}")
                        me = self._get_measured_edge(product_id)
                        if me is not None:
                            self.logger.info(f"MEASURED EDGE [{product_id}]: {me:.4f} | Adaptive blend: {self._adaptive_weight_blend:.2f}")
                    # Log throttle status
                    if cycle_num > 0 and cycle_num % 20 == 0 and self.signal_throttle:
                        killed = self.signal_throttle.get_killed_count()
                        if killed > 0:
                            status = self.signal_throttle.get_status()
                            self.logger.info(f"SIGNAL THROTTLE: {killed} signals killed — {status['killed_signals']}")
                    # Log health monitor metrics
                    if cycle_num > 0 and cycle_num % 20 == 0 and self.health_monitor:
                        hm = self.health_monitor.get_metrics()
                        if hm.get('total_trades', 0) >= 5:
                            self.logger.info(
                                f"HEALTH MONITOR: Sharpe={hm.get('sharpe_medium', 0):.2f} "
                                f"WinRate={hm.get('win_rate', 0):.1%} "
                                f"Trades={hm['total_trades']} "
                                f"Multiplier={hm['size_multiplier']:.2f} "
                                f"Level={hm['alert_level']}"
                            )

                # 1.5 Persist Market Data for Step 19 feedback loop
                if self.db_enabled:
                    ticker = market_data.get('ticker', {})
                    from database_manager import MarketData as DBMarketData
                    md_persist = DBMarketData(
                        price=current_price,
                        volume=ticker.get('volume', 0.0),
                        bid=ticker.get('bid', 0.0),
                        ask=ticker.get('ask', 0.0),
                        spread=ticker.get('bid_ask_spread', 0.0),
                        timestamp=datetime.now(timezone.utc),
                        source=market_data.get('_data_source', 'Coinbase'),
                        product_id=product_id
                    )
                    self._track_task(self.db_manager.store_market_data(md_persist))

                # 1.55 Feed orderbook snapshot to BarAggregator (BUG 7 fix)
                if self.bar_aggregator and ticker.get('bid', 0) > 0:
                    try:
                        import time as _time
                        self.bar_aggregator.on_orderbook_snapshot(
                            pair=product_id,
                            exchange=market_data.get('_data_source', 'coinbase'),
                            best_bid=float(ticker.get('bid', 0)),
                            best_ask=float(ticker.get('ask', 0)),
                            timestamp=_time.time(),
                        )
                    except Exception as e:
                        self.logger.warning(f"BarAggregator orderbook snapshot feed failed for {product_id}: {e}")

                # 1.56 Feed fast signal layers with real-time price data
                if self.fast_reversion_scanner and current_price > 0:
                    try:
                        self.fast_reversion_scanner.on_price_update(
                            pair=product_id,
                            price=float(current_price),
                            volume=float(ticker.get('volume_24h', 0)),
                            timestamp=time.time(),
                        )
                    except Exception as e:
                        self.logger.warning(f"Fast reversion scanner price update failed for {product_id}: {e}")
                if self.liquidation_detector and hasattr(self.liquidation_detector, 'on_price_update'):
                    try:
                        self.liquidation_detector.on_price_update(
                            symbol=product_id.replace("-USD", "USDT").replace("-", ""),
                            price=float(current_price) if current_price else 0.0,
                            volume=float(ticker.get('volume_24h', 0)),
                            spread_bps=float(ticker.get('spread_bps', 0)),
                            timestamp=time.time(),
                        )
                    except Exception as e:
                        self.logger.warning(f"Liquidation detector price update failed for {product_id}: {e}")

                # 1.6 Update Medallion-style engines with current price
                if current_price > 0:
                    self.mean_reversion_engine.update_price(product_id, current_price)
                    self.correlation_network.update_price(product_id, current_price)
                    self.garch_engine.update_returns(product_id, current_price)
                    self.stat_arb_engine.update_price(product_id, current_price)

                    # Update correlation network for all tracked products
                    all_prices = {}
                    for pid in self.product_ids:
                        t = market_data.get('ticker', {})
                        p = float(t.get('price', 0.0)) if pid == product_id else 0.0
                        if p > 0:
                            all_prices[pid] = p
                    if all_prices:
                        self.correlation_network.update_prices(all_prices)

                    # GARCH model refit check
                    if self.garch_engine.should_refit(product_id):
                        self.garch_engine.fit_model(product_id)

                    # Correlation network full update
                    self.correlation_network.run_full_update(self.scan_cycle_count)

                # Inject cross-asset data into market_data for ML features
                market_data['_cross_data'] = cross_data
                market_data['_pair_name'] = product_id

                # Fetch derivatives snapshot (Binance futures + Fear & Greed) for ML features
                # Only fetch for core pairs + active positions to avoid API spam
                _CORE_DERIV_PAIRS = {'BTC-USD', 'ETH-USD', 'SOL-USD', 'LINK-USD',
                                     'AVAX-USD', 'DOGE-USD', 'XRP-USD'}
                try:
                    _active_pos = {p.product_id for p in self.position_manager.positions.values()}
                    _should_fetch_deriv = product_id in _CORE_DERIV_PAIRS or product_id in _active_pos
                    if _should_fetch_deriv:
                        deriv_snap = await self.derivatives_provider.get_derivatives_snapshot(product_id)
                        if deriv_snap:
                            if product_id not in self._derivatives_history:
                                self._derivatives_history[product_id] = deque(maxlen=500)
                            self._derivatives_history[product_id].append(deriv_snap)

                    # Build accumulated time series from history (for any pair with data)
                    if product_id in self._derivatives_history:
                        _hist = list(self._derivatives_history[product_id])
                        if _hist:
                            _deriv_series = {}
                            for _dk in ['funding_rate', 'open_interest', 'long_short_ratio',
                                        'taker_buy_vol', 'taker_sell_vol', 'fear_greed']:
                                _vals = [h.get(_dk, float('nan')) for h in _hist]
                                _deriv_series[_dk] = pd.Series(_vals)

                            # Inject real-time liquidation data (scalar values)
                            if self._unified_price_feed:
                                _liq_sym = product_id.replace("-USD", "/USDT")
                                _liq_stats = self._unified_price_feed.get_liquidation_stats(_liq_sym)
                                if _liq_stats:
                                    _deriv_series['liq_long_usd_5m'] = _liq_stats.get('long_usd_5m', 0)
                                    _deriv_series['liq_short_usd_5m'] = _liq_stats.get('short_usd_5m', 0)
                                    _deriv_series['liq_cascade_active'] = _liq_stats.get('cascade_active', False)

                            market_data['_derivatives_data'] = _deriv_series
                except Exception as _de:
                    self.logger.debug(f"Derivatives fetch skipped for {product_id}: {_de}")

                # 2. Generate signals from all components
                signals = await self.generate_signals(market_data)


                # HARDENING: Ensure all signals are floats
                signals = {k: self._force_float(v) for k, v in signals.items()}

                # 2.0b Breakout Scanner signal injection
                _bo_sig = self._breakout_scores.get(product_id)
                if _bo_sig and _bo_sig.breakout_score >= 25:
                    # Normalize to -1 to +1 range
                    _bo_norm = _bo_sig.breakout_score / 100.0
                    if _bo_sig.direction == 'bearish':
                        _bo_norm = -_bo_norm
                    signals['breakout'] = _bo_norm
                    self.logger.info(
                        f"BREAKOUT SIGNAL: {product_id} score={_bo_sig.breakout_score:.0f} "
                        f"dir={_bo_sig.direction} vol_24h=${_bo_sig.volume_24h_usd:,.0f} "
                        f"change={_bo_sig.price_change_pct:+.1f}%"
                    )

                # 2.0a-d Advanced signals (microstructure, liquidation, fast-MR, multi-exchange, medallion analogs)
                from bot.cycle_ops import inject_advanced_signals
                await inject_advanced_signals(self, product_id, signals, market_data, current_price)

                # Build OHLCV DataFrame for ML models (needed by bridge + RT pipeline)
                _tech_inst = self._get_tech(product_id)
                price_df = _tech_inst._to_dataframe()
                # Fallback: if tech indicators have <30 rows, load from DB bars
                if len(price_df) < 30:
                    price_df = self._load_price_df_from_db(product_id, limit=300)

                # 2.1 ML Enhanced Signal Fusion (Unified from Enhanced Bot)
                ml_package = None
                if self.ml_enabled:
                    # ML Bridge generates parallel model predictions using OHLCV data
                    _deriv = market_data.get('_derivatives_data')
                    ml_package = await self.ml_bridge.generate_ml_signals(
                        price_df, signals,
                        cross_data=cross_data, pair_name=product_id,
                        derivatives_data=_deriv,
                    )

                # 2.2 Volume Profile Intelligence (Institutional)
                vp_signal = 0.0
                vp_status = "No Profile"
                if not price_df.empty:
                    profile = self.volume_profile_engine.calculate_profile(price_df)
                    if profile:
                        vp_data = self.volume_profile_engine.get_profile_signal(current_price, profile)
                        vp_signal = vp_data['signal']
                        vp_status = vp_data['status']

                signals['volume_profile'] = vp_signal
                self._last_vp_status[product_id] = vp_status

                # 2.5 Update regime overlay BEFORE signal fusion (so regime weights apply)
                # NOTE: use regime_df (not price_df) to avoid clobbering the ML DataFrame
                try:
                    if self.regime_overlay.enabled:
                        if len(_tech_inst.price_history) > 0:
                            regime_df = _tech_inst._to_dataframe()
                        else:
                            import pandas as _pd
                            regime_df = _pd.DataFrame({'close': [current_price]})
                        self.regime_overlay.update(regime_df)
                except Exception as _e:
                    self.logger.debug(f"Regime overlay skipped: {_e}")

                # 2.6 Multi-pair mean reversion signal
                if len(self.product_ids) > 1:
                    mr_portfolio = self.mean_reversion_engine.generate_portfolio_signal(
                        self.product_ids, self.scan_cycle_count
                    )
                    if mr_portfolio.composite_signal != 0.0:
                        signals['stat_arb'] = self._force_float(mr_portfolio.composite_signal)
                        market_data['mean_reversion_portfolio'] = {
                            'composite_signal': mr_portfolio.composite_signal,
                            'n_active_pairs': mr_portfolio.n_active_pairs,
                            'best_pair': (mr_portfolio.best_pair.base_id + '/' + mr_portfolio.best_pair.target_id)
                                if mr_portfolio.best_pair else None,
                        }

                # 2.7 Apply regime-adjusted weights (transient per cycle, does not modify self.signal_weights)
                cycle_weights = self.signal_weights.copy()
                if self.regime_overlay.enabled and self.regime_overlay.current_regime:
                    cycle_weights = self.regime_overlay.get_regime_adjusted_weights(cycle_weights)

                # 2.7b Adaptive Weight Engine — blend measured signal accuracy into weights
                try:
                    cycle_weights = self._compute_adaptive_weights(product_id, cycle_weights)
                    if self._adaptive_weight_blend > 0.01 and self.scan_cycle_count % 50 == 0:
                        self.logger.info(f"ADAPTIVE WEIGHTS [{product_id}]: blend={self._adaptive_weight_blend:.2f}")
                except Exception as _aw_err:
                    self.logger.debug(f"Adaptive weights skipped: {_aw_err}")

                # 2.8 Apply GARCH position-size multiplier to dynamic thresholds
                # FIX: Reset to base thresholds per-product to prevent unbounded
                # accumulation across cycles. Use local vars so products don't
                # interfere with each other.
                buy_threshold = self.buy_threshold   # Base from _update_dynamic_thresholds()
                sell_threshold = self.sell_threshold
                garch_pos_mult = 1.0
                if self.garch_engine.is_available:
                    garch_pos_mult = self.garch_engine.get_position_size_multiplier(product_id)
                    buy_delta, sell_delta = self.garch_engine.get_dynamic_threshold_adjustment(product_id)
                    buy_threshold += buy_delta
                    sell_threshold += sell_delta

                # 2.9 Signal Validation Gate — clip anomalies, check regime consistency
                if self.signal_validation_gate:
                    regime_label = self.regime_overlay.get_hmm_regime_label() if self.regime_overlay.enabled else "unknown"
                    signals = self.signal_validation_gate.validate(signals, regime_label)

                # 2.10 Signal Auto-Throttle — zero killed signals before weighting
                if self.signal_throttle:
                    signals = self.signal_throttle.filter(signals, product_id)

                # 2.11 Daily Signal Review — apply end-of-day allocation multipliers (observation log)
                if self.daily_signal_review:
                    try:
                        for sig_name in list(signals.keys()):
                            alloc = self.daily_signal_review.get_allocation_multiplier(sig_name)
                            if alloc < 1.0:
                                self.logger.info(
                                    f"DAILY REVIEW (obs): {sig_name} allocation={alloc:.1f}x "
                                    f"(signal={signals[sig_name]:.4f})"
                                )
                    except Exception as e:
                        self.logger.warning(f"Daily signal review allocation check failed: {e}")

                # 3. Calculate Renaissance weighted signal with regime-adjusted weights
                # Temporarily swap weights for this cycle
                original_weights = self.signal_weights
                self.signal_weights = cycle_weights
                weighted_signal, contributions = self.calculate_weighted_signal(signals)
                self.signal_weights = original_weights

                # PARANOID SCALAR HARDENING: Ensure results are primitive floats
                weighted_signal = float(self._force_float(weighted_signal))
                contributions = {str(k): float(self._force_float(v)) for k, v in contributions.items()}

                # EXTRA HARDENING: Ensure signals dictionary is all floats for boost calculation
                signals = {str(k): float(self._force_float(v)) for k, v in signals.items()}
                # Record signals for scorecard evaluation next cycle
                _vol_mag_bps = 100.0  # default
                _vol_pred_inner = (market_data or {}).get('volatility_prediction')
                if _vol_pred_inner and isinstance(_vol_pred_inner, dict):
                    _vol_mag_bps = _vol_pred_inner.get('predicted_magnitude_bps', 100.0)
                self._pending_predictions[product_id] = {
                    'price': current_price,
                    'signals': {k: float(v) for k, v in signals.items()},
                    'cycle': getattr(self, 'scan_cycle_count', 0),
                    'predicted_magnitude_bps': _vol_mag_bps,
                }

                market_data['ml_package'] = ml_package

                # 3.1 Non-linear Confluence Boost (Step 20)
                confluence_data = self.confluence_engine.calculate_confluence_boost(signals)

                # Extract boost scalar with hardening
                boost_scalar_final = 0.0
                try:
                    raw_b_v_f = confluence_data.get('total_confluence_boost', 0.0)
                    boost_scalar_final = self._force_float(raw_b_v_f)
                except Exception:
                    boost_scalar_final = 0.0

                if boost_scalar_final > 0:
                    try:
                        b_sig_f = float(weighted_signal)
                        b_factor_f = float(1.0 + boost_scalar_final)
                        boosted_val_f = b_sig_f * b_factor_f
                        weighted_signal = float(np.clip(boosted_val_f, -1.0, 1.0))
                    except Exception as e:
                        self.logger.warning(f"Confluence boost application failed: {e}")
                else:
                    weighted_signal = self._force_float(weighted_signal)

                # Final check to ensure it's not a sequence before decision
                weighted_signal = float(weighted_signal)
                market_data['confluence_data'] = confluence_data
                market_data['garch_position_multiplier'] = garch_pos_mult
                market_data['regime_adjusted_weights'] = cycle_weights

                # Random baseline shadow entries
                if hasattr(self, 'random_baseline') and current_price > 0:
                    self.random_baseline.maybe_enter(product_id, current_price)

                # 3.15-3.16 Polymarket Bridge + Scanner
                from bot.cycle_ops import handle_polymarket_signals
                await handle_polymarket_signals(self, product_id, weighted_signal, ml_package, market_data)

                # 3.17 Strategy A — accumulate ML predictions per pair (multi-asset crash)
                from bot.cycle_ops import cache_strategy_a_predictions
                cache_strategy_a_predictions(self, product_id, market_data, price_df, ml_package)

                # 3.2 Update Dynamic Thresholds (Step 8)

                self._update_dynamic_thresholds(product_id, market_data)

                # 4. Real-time pipeline cycle (Step 12)
                rt_result = None
                if self.real_time_pipeline.enabled:
                    await self.real_time_pipeline.start()
                    raw_rt = await self.real_time_pipeline.run_cycle(price_df=price_df)
                    if raw_rt:
                        # Hardening real-time pipeline outputs
                        rt_result = {}
                        for k, v in raw_rt.items():
                            if k == 'predictions':
                                rt_result[k] = {mk: self._force_float(mv) for mk, mv in v.items()}
                            else:
                                try:
                                    rt_result[k] = self._force_float(v)
                                except Exception:
                                    rt_result[k] = v

                # 4.1 Build MLSignalPackage from real-time pipeline predictions
                if rt_result and rt_result.get('predictions') and (ml_package is None or ml_package.ensemble_score == 0.0):
                    preds = rt_result['predictions']
                    pred_values = [v for v in preds.values() if isinstance(v, (int, float))]
                    if pred_values:
                        ensemble_score = float(np.mean(pred_values))
                        # Confidence from model agreement: high agreement → high confidence
                        pred_std = float(np.std(pred_values)) if len(pred_values) > 1 else 0.5
                        agreement = max(0.0, 1.0 - pred_std / 0.1)
                        confidence_score = float(np.clip(agreement, 0.3, 0.95))
                        ml_package = MLSignalPackage(
                            primary_signals=[],
                            ml_predictions=list(preds.items()),
                            ensemble_score=ensemble_score,
                            confidence_score=confidence_score,
                            fractal_insights={},
                            processing_time_ms=0.0,
                        )
                        self.logger.info(
                            f"RT→ML bridge: ensemble={ensemble_score:+.4f}, "
                            f"confidence={confidence_score:.2f} ({len(pred_values)} models)"
                        )

                # 4.5 Statistical Arbitrage & Fractal Intelligence
                current_price = market_data.get('ticker', {}).get('price', 0.0)

                # (straddle open moved to pre-loop at line ~4303)

                self.stat_arb_engine.update_price(product_id, current_price)
                
                # 🏛️ Basis Trading Signal
                basis_signal = self._force_float(self.basis_engine.get_basis_signal(market_data))
                signals['basis'] = basis_signal
                
                stat_arb_data = {}
                if len(self.product_ids) > 1:
                    # Try stat arb vs BTC for all non-BTC assets
                    base = "BTC-USD"
                    if product_id == base:
                        # For BTC, find best pair from available assets
                        for target in self.product_ids:
                            if target != base:
                                try:
                                    pair_data = self.stat_arb_engine.calculate_pair_signal(base, target)
                                    if pair_data.get('status') == 'active' and abs(self._force_float(pair_data.get('signal', 0))) > abs(self._force_float(stat_arb_data.get('signal', 0))):
                                        stat_arb_data = pair_data
                                except Exception as e:
                                    self.logger.warning(f"Stat arb pair signal failed for {base}/{target}: {e}")
                    else:
                        # For non-BTC assets, compute pair vs BTC
                        try:
                            stat_arb_data = self.stat_arb_engine.calculate_pair_signal(base, product_id)
                            if 'signal' in stat_arb_data:
                                stat_arb_data['signal'] = -self._force_float(stat_arb_data['signal'])
                        except Exception as e:
                            self.logger.warning(f"Stat arb signal calculation failed for {base}/{product_id}: {e}")
                
                if stat_arb_data.get('status') == 'active':
                    signals['stat_arb'] = self._force_float(stat_arb_data['signal'])
                else:
                    signals['stat_arb'] = 0.0

                # ── Hard staleness gate: skip trading for pairs with data > 30min old ──
                if market_data.get('_skip_stale'):
                    pair_elapsed = time.time() - pair_start_time
                    if pair_elapsed > 10:
                        self.logger.warning(f"SLOW PAIR: {product_id} took {pair_elapsed:.1f}s (stale skip)")
                    continue

                # 5. Make trading decision
                ticker = market_data.get('ticker', {})
                current_price = self._force_float(ticker.get('price', 0.0))
                
                # ── Market Sanity Checks (pre-trade gates) ──
                from bot.cycle_ops import apply_market_sanity_checks
                _sanity_skip = apply_market_sanity_checks(self, product_id, market_data, current_price)
                if _sanity_skip:
                    continue

                # ── Token Spray Path (bypasses legacy decision) ──
                from bot.cycle_ops import execute_token_spray
                if await execute_token_spray(self, product_id, weighted_signal, contributions,
                                             ml_package, market_data, current_price, rt_result):
                    pair_elapsed = time.time() - pair_start_time
                    if pair_elapsed > 10:
                        self.logger.warning(f"SLOW PAIR: {product_id} took {pair_elapsed:.1f}s (spray)")
                    continue

                # 5.1 Meta-Strategy Selection
                regime_data = self.regime_overlay.current_regime or {}
                self.last_vpin = market_data.get('vpin', 0.5)
                execution_mode = self.strategy_selector.select_mode(market_data, regime_data)
                market_data['execution_mode'] = execution_mode

                # ── Audit Logger: collect pre-decision data ──
                _audit = self.audit_logger if self.db_enabled else None
                if _audit:
                    try:
                        _audit.start_decision(product_id, self.scan_cycle_count)
                        _ticker = market_data.get('ticker', {})
                        _ob = market_data.get('order_book_snapshot')
                        _ob_depth = 0.0
                        try:
                            if _ob and hasattr(_ob, 'bids') and hasattr(_ob, 'asks'):
                                _ob_depth = sum(float(getattr(lv, 'size', 0)) * float(getattr(lv, 'price', 0))
                                                for lv in (list(getattr(_ob, 'bids', []))[:10] + list(getattr(_ob, 'asks', []))[:10]))
                        except Exception:
                            _ob_depth = 0.0
                        _audit.record_market_snapshot(
                            price=current_price,
                            bid=self._force_float(_ticker.get('bid', 0)),
                            ask=self._force_float(_ticker.get('ask', 0)),
                            volume_24h=self._force_float(_ticker.get('volume', 0)),
                            ob_depth=_ob_depth,
                        )
                        _audit.record_raw_signals(signals)
                        _rl = self.regime_overlay.get_hmm_regime_label() if self.regime_overlay.enabled else "unknown"
                        _rc = 0.0
                        if self.regime_overlay.current_regime and isinstance(self.regime_overlay.current_regime, dict):
                            _rc = float(self.regime_overlay.current_regime.get('confidence', 0.0))
                        _audit.record_regime(_rl, _rc, 'hmm')
                        _audit.record_weights(
                            base_weights=original_weights,
                            cycle_weights=cycle_weights,
                            contributions=contributions,
                            weighted_signal=weighted_signal,
                        )
                        _audit.record_ml(ml_package, rt_result, self.config.get("ml_signal_scale", 10.0))
                        _audit.record_confluence(confluence_data)
                    except Exception as _audit_err:
                        self.logger.warning(f"Audit pre-decision collection failed: {_audit_err}")
                        _audit = None  # Disable audit for this decision

                decision = self.make_trading_decision(weighted_signal, contributions,
                                                    current_price=current_price,
                                                    real_time_result=rt_result,
                                                    product_id=product_id,
                                                    ml_package=ml_package,
                                                    market_data=market_data,
                                                    drawdown_pct=getattr(self, '_current_drawdown_pct', 0.0),
                                                    audit_logger=_audit)

                # Council S3: Apply staleness decay to confidence
                _staleness_decay = market_data.get('_staleness_decay', 1.0)
                if _staleness_decay < 1.0 and decision.confidence > 0:
                    decision.confidence *= _staleness_decay
                    # Post-decay safety: if confidence dropped below min_confidence, force HOLD
                    if decision.action != 'HOLD' and decision.confidence < self.min_confidence:
                        self.logger.warning(
                            f"STALENESS CONF GUARD: {product_id} {decision.action} blocked — "
                            f"confidence={decision.confidence:.4f} < min={self.min_confidence} after staleness decay={_staleness_decay:.4f}"
                        )
                        decision = TradingDecision(
                            action='HOLD', confidence=decision.confidence,
                            position_size=0.0, reasoning=decision.reasoning,
                            timestamp=datetime.now()
                        )

                # ── Audit Logger: record decision + finalize ──
                if _audit:
                    try:
                        _audit.record_decision(decision.action, decision.position_size)
                        _audit.record_execution(
                            mode=market_data.get('execution_mode', 'PAPER'),
                        )
                        if ml_package and hasattr(ml_package, 'feature_vector') and ml_package.feature_vector is not None:
                            _audit.record_feature_vector(ml_package.feature_vector)
                        _open_count = 0
                        try:
                            with self.position_manager._lock:
                                _open_count = sum(1 for p in self.position_manager.positions.values() if p.status == PositionStatus.OPEN)
                        except Exception as e:
                            self.logger.warning(f"Open position count for audit failed: {e}")
                        _audit.record_system_state(
                            drawdown_pct=getattr(self, '_current_drawdown_pct', 0.0),
                            daily_pnl=self.daily_pnl,
                            balance=self._high_watermark_usd if hasattr(self, '_high_watermark_usd') else 10000.0,
                            open_positions_count=_open_count,
                            scan_tier=getattr(self, '_current_scan_tier', 0),
                        )
                        await _audit.finalize()
                    except Exception as _audit_err:
                        self.logger.warning(f"Audit finalize failed: {_audit_err}")

                # 5.05 Devil Tracker — record signal detection price for cost tracking
                if self.devil_tracker and decision.action != 'HOLD':
                    try:
                        _exec_exchange = "mexc" if product_id in self._pair_binance_symbols else "coinbase"
                        _devil_trade_id = self.devil_tracker.record_signal_detection(
                            signal_type="combined",
                            pair=product_id,
                            exchange=_exec_exchange,
                            price=current_price,
                            side=decision.action,
                        )
                        if not hasattr(self, '_last_devil_trade_id'):
                            self._last_devil_trade_id = {}
                        self._last_devil_trade_id[product_id] = _devil_trade_id
                    except Exception as _dt_err:
                        self.logger.debug(f"Devil tracker signal record failed: {_dt_err}")

                # Drawdown circuit breaker: block new positions in exits-only mode
                if getattr(self, '_drawdown_exits_only', False) and decision.action != 'HOLD':
                    self.logger.warning(f"CIRCUIT BREAKER: blocking {decision.action} for {product_id} — exits only mode")
                    decision = TradingDecision(
                        action='HOLD', confidence=decision.confidence,
                        position_size=0.0, reasoning=decision.reasoning,
                        timestamp=datetime.now()
                    )
                # Drawdown size scaling
                elif getattr(self, '_drawdown_size_scalar', 1.0) < 1.0 and decision.action != 'HOLD':
                    scaled_size = decision.position_size * self._drawdown_size_scalar
                    decision = TradingDecision(
                        action=decision.action, confidence=decision.confidence,
                        position_size=scaled_size, reasoning=decision.reasoning,
                        timestamp=datetime.now()
                    )

                # Inject Meta-Strategy Execution Mode (Step 11/13)
                decision.reasoning['execution_mode'] = market_data.get('execution_mode', 'TAKER')

                decision.reasoning['product_id'] = product_id
                if rt_result:
                    decision.reasoning['real_time_pipeline'] = rt_result

                # Feed signal to Medallion Portfolio Engine for drift tracking
                if self.medallion_portfolio_engine and decision.action != 'HOLD':
                    try:
                        self.medallion_portfolio_engine.ingest_signal({
                            "pair": product_id,
                            "side": decision.action,
                            "strength": abs(weighted_signal),
                            "confidence": decision.confidence,
                            "signal_type": "combined",
                            "notional_usd": decision.position_size * current_price if current_price > 0 else 0,
                        })
                    except Exception as e:
                        self.logger.warning(f"Medallion portfolio engine signal ingest failed for {product_id}: {e}")

                # Retrieve lead_lag_alpha from market_data if it was calculated in generate_signals
                if 'lead_lag_alpha' in market_data:
                    decision.reasoning['lead_lag_alpha'] = market_data['lead_lag_alpha']

                if stat_arb_data:
                    decision.reasoning['stat_arb'] = stat_arb_data
                if 'whale_signals' in market_data:
                    decision.reasoning['whale_signals'] = market_data['whale_signals']
                if 'garch_forecast' in market_data:
                    decision.reasoning['garch_forecast'] = market_data['garch_forecast']
                if 'mean_reversion_portfolio' in market_data:
                    decision.reasoning['mean_reversion_portfolio'] = market_data['mean_reversion_portfolio']

                # Note: GARCH vol adjustment is now integrated into position_sizer
                # (no separate garch_pos_mult needed)

                # 5.1 Persist Decision & ML Predictions
                if self.db_enabled:
                    # Store decision with HMM regime
                    hmm_regime_label = self.regime_overlay.get_hmm_regime_label()
                    decision_persist = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'product_id': product_id,
                        'action': decision.action,
                        'confidence': decision.confidence,
                        'position_size': decision.position_size,
                        'weighted_signal': weighted_signal,
                        'reasoning': decision.reasoning,
                        'hmm_regime': hmm_regime_label,
                        'vae_loss': decision.reasoning.get('vae_loss'),
                    }
                    self._track_task(self.db_manager.store_decision(decision_persist))
                    
                    # Store ML predictions (prefer ml_bridge which has confidence)
                    _ml_preds = {}  # model_name -> (prediction, confidence)
                    if ml_package and ml_package.ml_predictions:
                        for mp in ml_package.ml_predictions:
                            if isinstance(mp, dict):
                                _ml_preds[mp.get('model', 'unknown')] = (
                                    mp.get('prediction', 0.0),
                                    mp.get('confidence'),
                                )
                            elif isinstance(mp, tuple) and len(mp) >= 2:
                                _ml_preds[mp[0]] = (mp[1], mp[2] if len(mp) > 2 else None)
                    elif rt_result and 'predictions' in rt_result:
                        for mn, pv in rt_result['predictions'].items():
                            _ml_preds[mn] = (pv, None)
                    for model_name, (pred, conf) in _ml_preds.items():
                        self._track_task(self.db_manager.store_ml_prediction({
                            'product_id': product_id,
                            'model_name': model_name,
                            'prediction': pred,
                            'confidence': conf,
                            'price_at_prediction': current_price,
                        }))
                    _ml_inference_count += len(_ml_preds)

                    # Periodic outcome evaluation (every 10 cycles)
                    if self.scan_cycle_count % 10 == 0:
                        self._track_task(self.db_manager.evaluate_ml_outcomes())
                        self._track_task(self.db_manager.evaluate_audit_outcomes())
                        # Council S3 #1/#5: Refresh ML accuracy cache for Kelly calibration
                        if self.scan_cycle_count != self._ml_accuracy_cache_cycle:
                            self._refresh_ml_accuracy_cache()
                            self._ml_accuracy_cache_cycle = self.scan_cycle_count

                # 5.5 Exit Engine — Monitor open positions for alpha decay
                from bot.cycle_ops import evaluate_exits
                evaluate_exits(self, product_id, current_price, market_data, decision)

                # 5.6 Continuous Position Re-evaluation (Doc 10)
                from bot.cycle_ops import reevaluate_positions
                reevaluate_positions(self, product_id, current_price, market_data, decision)

                # 5.7 Position stacking is prevented in make_trading_decision()
                # Same-direction positions are blocked; reversals close existing positions

                # 6. Smart Execution (Step 10)
                # Final pre-execution confidence guard (defense-in-depth)
                if decision.action != 'HOLD' and (
                    decision.confidence <= 0 or decision.confidence < self.min_confidence
                ):
                    self.logger.warning(
                        f"PRE-EXEC GUARD: {product_id} {decision.action} blocked — "
                        f"confidence={decision.confidence:.4f} < min={self.min_confidence}"
                    )
                    decision = TradingDecision(
                        action='HOLD', confidence=decision.confidence,
                        position_size=0.0, reasoning=decision.reasoning,
                        timestamp=datetime.now()
                    )
                if decision.action != 'HOLD':
                    if self.config.get("market_making", {}).get("enabled", False):
                        # ⚖️ Market Making Mode (Liquidity Provider)
                        quotes = self.market_making.calculate_quotes(
                            current_price, 
                            market_data.get('volatility', 0.02),
                            signals.get('order_book', 0.0),
                            vpin=market_data.get('vpin', 0.5)
                        )
                        self.logger.info(f"⚖️ MARKET MAKING QUOTES: Bid {quotes['bid']:.2f} | Ask {quotes['ask']:.2f} (Skew: {quotes['skew']:.4f})")
                        # In production, send limit orders here
                    else:
                        # Standard Sniper/TWAP/VWAP Taker execution
                        await self._execute_smart_order(decision, market_data)

                # 7. Consciousness Dashboard (The "Inner Thoughts")
                self._log_consciousness_dashboard(product_id, decision, rt_result)

                # 8. Adaptive Learning Cycle (Step 15) & Evolutionary Step (Step 14)
                # Periodically calibrate models (e.g., every 10 cycles)
                self.decision_history.append(decision)
                if len(self.decision_history) % 10 == 0:
                    self._track_task(self._run_adaptive_learning_cycle())
                    self._track_task(self._perform_attribution_analysis())

                    # Update multi-exchange funding rates (every 10 cycles)
                    if self.multi_exchange_bridge:
                        self._track_task(self.multi_exchange_bridge.update_funding_rates())

                    # Periodic position reconciliation (skip in paper trading — no real exchange positions)
                    if not self.paper_trading:
                        recon = self.position_manager.reconcile_with_exchange()
                        if recon.get("status") == "MISMATCH":
                            self._track_task(
                                self.alert_manager.send_alert("CRITICAL", "Position Mismatch",
                                    f"{len(recon['discrepancies'])} discrepancies detected")
                            )
                    
                    # Run Self-Reinforcing Learning (Step 19)
                    if self.real_time_pipeline.enabled:
                        self._track_task(self.learning_engine.run_learning_cycle(
                            self.real_time_pipeline.processor.models
                        ))
                    
                    # Run Genetic Weight Optimization (Step 14) — skip if locked
                    if not self.config.get('weight_lock', False):
                        async def run_evo():
                            new_weights = await self.genetic_optimizer.run_optimization_cycle(self.signal_weights)
                            if new_weights != self.signal_weights:
                                self.logger.info("Evolutionary Step (Step 14): Weights updated.")
                                async with self._weights_lock:
                                    self.signal_weights = new_weights

                        self._track_task(run_evo())

                    # ── Medallion Monitors (periodic health checks) ──
                    try:
                        if self.sharpe_monitor_medallion:
                            sharpe_report = self.sharpe_monitor_medallion.get_report()
                            sharpe_val = sharpe_report.get("rolling_sharpe_30d", 0)
                            if self.sharpe_monitor_medallion.should_reduce_exposure():
                                self.logger.warning(
                                    f"SHARPE MONITOR: reduce exposure (Sharpe={sharpe_val:.2f})"
                                )
                            else:
                                self.logger.info(f"SHARPE MONITOR: Sharpe={sharpe_val:.2f}")
                    except Exception as e:
                        self.logger.warning(f"Sharpe monitor check failed: {e}")
                    try:
                        if self.beta_monitor:
                            beta_report = self.beta_monitor.get_report()
                            if self.beta_monitor.should_alert():
                                rec = self.beta_monitor.get_hedge_recommendation()
                                self.logger.warning(
                                    f"BETA MONITOR: beta={beta_report.get('beta', 0):.2f} — "
                                    f"hedge: {rec.get('action', 'none')}"
                                )
                    except Exception as e:
                        self.logger.warning(f"Beta monitor check failed: {e}")
                    try:
                        if self.capacity_monitor:
                            caps = self.capacity_monitor.get_all_capacities()
                            for _cpair, _cdata in caps.items():
                                if _cdata.get("at_capacity_wall"):
                                    self.logger.warning(
                                        f"CAPACITY MONITOR: {_cpair} at capacity wall"
                                    )
                    except Exception as e:
                        self.logger.warning(f"Capacity monitor check failed: {e}")

                # DEPRECATED: Old breakout scan (replaced by Phase 0 breakout_scanner)
                # if self.scan_cycle_count % 10 == 0:
                #     self._track_task(self._run_breakout_scan())

                decisions.append(decision)

                pair_elapsed = time.time() - pair_start_time
                if pair_elapsed > 10:
                    self.logger.warning(f"SLOW PAIR: {product_id} took {pair_elapsed:.1f}s")
                self.logger.info(f"[{product_id}] Decision: {decision.action} "
                               f"(Conf: {decision.confidence:.3f}, Size: {decision.position_size:.3f}, "
                               f"time: {pair_elapsed:.1f}s)")

                # 📊 Emit dashboard events
                from bot.cycle_ops import emit_dashboard_events
                emit_dashboard_events(self, product_id, decision, weighted_signal, current_price, market_data)

            # Increment cycle counter ONCE per cycle (not per pair)
            self.scan_cycle_count += 1

            # Council S2 P1: Record ml_inference + decision_engine heartbeats after per-pair loop
            self._track_task(self.db_manager.record_heartbeat(
                'ml_inference', items_processed=_ml_inference_count,
                details={'cycle': self.scan_cycle_count, 'pairs': len(cycle_pairs)}
            ))
            _n_decisions = len(decisions) if decisions else 0
            _n_trades = sum(1 for d in (decisions or []) if getattr(d, 'action', 'HOLD') != 'HOLD')
            self._track_task(self.db_manager.record_heartbeat(
                'decision_engine', items_processed=_n_decisions,
                details={'cycle': self.scan_cycle_count, 'pairs': len(cycle_pairs), 'trades': _n_trades}
            ))
            self._track_task(self.db_manager.record_heartbeat(
                'trade_executor', items_processed=_n_trades,
                details={'cycle': self.scan_cycle_count}
            ))

            # Council S2 P1: Pipeline watchdog check every 30 cycles (~30 min)
            if self.scan_cycle_count % 30 == 0:
                self._track_task(self._check_pipeline_health())

            # ── Periodic DB pruning (every 100 cycles) ──
            if self.scan_cycle_count % 100 == 0:
                self._prune_old_data()
                # Council S2 P3: Log ML accuracy summary every 100 cycles
                self._track_task(self._log_ml_accuracy_summary())
                # Council S3: Compute model accuracy scorecard
                self._track_task(self.db_manager.compute_model_scorecard(window_days=7))
                # Council S3: Log spread stats for slippage calibration
                if hasattr(self, '_pair_spread_history') and self._pair_spread_history:
                    _spread_parts = []
                    for _sp_pair, _sp_hist in sorted(self._pair_spread_history.items()):
                        if _sp_hist:
                            _avg = sum(_sp_hist) / len(_sp_hist)
                            _short = _sp_pair.replace('USDT', '').replace('-USD', '')
                            _spread_parts.append(f"{_short}={_avg:.2f}bps")
                    if _spread_parts:
                        self.logger.info(f"SPREAD STATS: {', '.join(_spread_parts[:15])}")

            # Council S3: Kelly calibration check every 500 cycles (~8 hours)
            if self.scan_cycle_count % 500 == 0 and self.scan_cycle_count > 0:
                self._track_task(self._log_kelly_calibration())

            # Random baseline position management
            if hasattr(self, 'random_baseline'):
                _rb_prices = {}
                for _pid in cycle_pairs:
                    _md = market_data_all.get(_pid, {})
                    _tk = _md.get('ticker', {})
                    _px = float(_tk.get('price', 0)) if _tk else 0
                    if _px > 0:
                        _rb_prices[_pid] = _px
                if _rb_prices:
                    _rb_exits = self.random_baseline.update_positions(_rb_prices)
                    if _rb_exits:
                        self.logger.info(f"RANDOM BASELINE: {len(_rb_exits)} shadow exits")

            # ── Strategy A: now runs on its own 60s background loop (_run_strategy_a_loop) ──
            # ML predictions (_sa_ml_cache) are still populated here by the main cycle
            # and consumed by the independent Strategy A loop.

            cycle_time = time.time() - cycle_start
            self.logger.info(
                f"Cycle {self.scan_cycle_count} complete: "
                f"{len(cycle_pairs)} pairs, "
                f"fetch={fetch_elapsed:.1f}s, total={cycle_time:.1f}s"
            )
            if cycle_time > 240:
                self.logger.warning(
                    f"CYCLE OVERRUN RISK: {cycle_time:.0f}s for {len(cycle_pairs)} pairs"
                )

            # ── Risk Alert Evaluation (emit to dashboard) ──
            try:
                from dashboard.db_queries import evaluate_risk_alerts
                db_path = str(self.db_manager.db_path) if hasattr(self.db_manager, 'db_path') else "data/renaissance_bot.db"
                risk_alerts = evaluate_risk_alerts(db_path)
                for alert in risk_alerts:
                    self._track_task(self.dashboard_emitter.emit("risk.alert", alert))
                    self.logger.warning(f"RISK ALERT [{alert['severity']}]: {alert['message']}")
            except Exception as _ra:
                self.logger.debug(f"Risk alert evaluation error: {_ra}")

            # ── Doc 15: Agent cycle hook ──
            if self.agent_coordinator:
                try:
                    first_dec = decisions[0] if decisions else None
                    self.agent_coordinator.on_cycle_complete({
                        "action": first_dec.action if first_dec else "HOLD",
                        "confidence": first_dec.confidence if first_dec else 0.0,
                        "product_id": first_dec.reasoning.get("product_id", "") if first_dec else "",
                        "regime": first_dec.reasoning.get("hmm_regime", "unknown") if first_dec else "unknown",
                        "cycle_time": cycle_time,
                    })
                except Exception as e:
                    self.logger.warning(f"Agent coordinator cycle complete notification failed: {e}")

            # Return the first decision or a HOLD if none
            return decisions[0] if decisions else TradingDecision('HOLD', 0.0, 0.0, {}, datetime.now())

        except Exception as e:
            import traceback
            self.logger.error(f"Trading cycle failed: {e}")
            self.logger.error(traceback.format_exc())
            return TradingDecision('HOLD', 0.0, 0.0, {'error': str(e)}, datetime.now())

    def _log_consciousness_dashboard(self, product_id: str, decision: TradingDecision, rt_result: Optional[Dict[str, Any]]):
        """Consciousness dashboard — delegates to bot.helpers."""
        from bot.helpers import log_consciousness_dashboard
        return log_consciousness_dashboard(self, product_id, decision, rt_result)

    # ──────────────────────────────────────────────
    #  Kill Switch
    # ──────────────────────────────────────────────
    KILL_FILE = Path("KILL_SWITCH")

    def trigger_kill_switch(self, reason: str):
        """Kill switch — delegates to bot.lifecycle."""
        from bot.lifecycle import trigger_kill_switch
        return trigger_kill_switch(self, reason)

    def _check_kill_file(self):
        """Check kill file — delegates to bot.lifecycle."""
        from bot.lifecycle import check_kill_file
        return check_kill_file(self)

    # ── Background Loops — delegates to bot.lifecycle ──

    async def _run_websocket_feed(self):
        from bot.lifecycle import run_websocket_feed
        return await run_websocket_feed(self)

    async def _run_arbitrage_engine(self):
        from bot.lifecycle import run_arbitrage_engine
        return await run_arbitrage_engine(self)

    async def _run_btc_price_relay(self):
        from bot.lifecycle import run_btc_price_relay
        return await run_btc_price_relay(self)

    async def _run_strategy_a_loop(self):
        from bot.lifecycle import run_strategy_a_loop
        return await run_strategy_a_loop(self)

    # ──────────────────────────────────────────────
    #  Liquidation Cascade Detector (Module D)
    async def _run_liquidation_detector(self):
        from bot.lifecycle import run_liquidation_detector
        return await run_liquidation_detector(self)

    async def _run_fast_reversion_scanner(self):
        from bot.lifecycle import run_fast_reversion_scanner
        return await run_fast_reversion_scanner(self)

    async def _run_sub_bar_scanner(self):
        from bot.lifecycle import run_sub_bar_scanner
        return await run_sub_bar_scanner(self)

    async def _run_heartbeat_writer(self, interval: float = 5.0):
        from bot.lifecycle import run_heartbeat_writer
        return await run_heartbeat_writer(self, interval)

    async def _run_portfolio_drift_logger(self):
        from bot.lifecycle import run_portfolio_drift_logger
        return await run_portfolio_drift_logger(self)

    async def _run_insurance_scanner_loop(self):
        from bot.lifecycle import run_insurance_scanner_loop
        return await run_insurance_scanner_loop(self)

    async def _run_daily_signal_review_loop(self):
        from bot.lifecycle import run_daily_signal_review_loop
        return await run_daily_signal_review_loop(self)

    async def _run_beta_monitor_loop(self):
        from bot.lifecycle import run_beta_monitor_loop
        return await run_beta_monitor_loop(self)

    async def _run_sharpe_monitor_loop(self):
        from bot.lifecycle import run_sharpe_monitor_loop
        return await run_sharpe_monitor_loop(self)

    async def _run_capacity_monitor_loop(self):
        from bot.lifecycle import run_capacity_monitor_loop
        return await run_capacity_monitor_loop(self)

    async def _run_regime_detector_loop(self):
        from bot.lifecycle import run_regime_detector_loop
        return await run_regime_detector_loop(self)

    async def _run_telegram_report_loop(self):
        from bot.lifecycle import run_telegram_report_loop
        return await run_telegram_report_loop(self)

    # ──────────────────────────────────────────────
    #  State Recovery
    # ──────────────────────────────────────────────
    async def _restore_state(self):
        """Restore positions and daily PnL — delegates to bot.position_ops."""
        from bot.position_ops import restore_state
        return await restore_state(self)

    def _prune_old_data(self) -> None:
        """Prune old data — delegates to bot.lifecycle."""
        from bot.lifecycle import prune_old_data
        return prune_old_data(self)

    async def _get_spray_prices(self, pairs: List[str]) -> Dict[str, float]:
        """Fetch current prices for token spray exit checks — delegates to bot.position_ops."""
        from bot.position_ops import get_spray_prices
        return await get_spray_prices(self, pairs)

    async def _get_straddle_price(self, pair: str = '') -> Dict[str, float]:
        """Fetch current price for straddle exit checks — delegates to bot.position_ops."""
        from bot.position_ops import get_straddle_price
        return await get_straddle_price(self, pair)

    async def _deduplicate_positions_on_startup(self) -> None:
        """Close duplicate and opposing positions — delegates to bot.position_ops."""
        from bot.position_ops import deduplicate_positions_on_startup
        return await deduplicate_positions_on_startup(self)

    async def run_continuous_trading(self, cycle_interval: int = 300):
        """Run continuous trading — delegates to bot.lifecycle."""
        from bot.lifecycle import run_continuous_trading
        return await run_continuous_trading(self, cycle_interval)

    def _compute_adaptive_weights(self, product_id: str, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Adaptive weights — delegates to bot.adaptive."""
        from bot.adaptive import compute_adaptive_weights
        return compute_adaptive_weights(self, product_id, base_weights)

    def _get_measured_edge(self, product_id: str) -> Optional[float]:
        """Measured edge — delegates to bot.adaptive."""
        from bot.adaptive import get_measured_edge
        return get_measured_edge(self, product_id)

    def _refresh_ml_accuracy_cache(self) -> None:
        """ML accuracy cache — delegates to bot.adaptive."""
        from bot.adaptive import refresh_ml_accuracy_cache
        return refresh_ml_accuracy_cache(self)

    def _update_dynamic_thresholds(self, product_id: str, market_data: Dict[str, Any]):
        """Dynamic thresholds — delegates to bot.adaptive."""
        from bot.adaptive import update_dynamic_thresholds
        return update_dynamic_thresholds(self, product_id, market_data)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Performance summary — delegates to bot.helpers."""
        from bot.helpers import get_performance_summary
        return get_performance_summary(self)
