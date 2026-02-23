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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time

# Import all Renaissance components
from enhanced_config_manager import EnhancedConfigManager
from microstructure_engine import MicrostructureEngine, MicrostructureMetrics
from enhanced_technical_indicators import EnhancedTechnicalIndicators, IndicatorOutput
from market_data_provider import LiveMarketDataProvider
from derivatives_data_provider import DerivativesDataProvider
from renaissance_signal_fusion import RenaissanceSignalFusion
from alternative_data_engine import AlternativeDataEngine, AlternativeSignal

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
from coinbase_client import EnhancedCoinbaseClient, CoinbaseCredentials
from position_manager import EnhancedPositionManager, RiskLimits, PositionStatus
from alert_manager import AlertManager
from coinbase_advanced_client import CoinbaseAdvancedClient
from logger import SecretMaskingFilter
from binance_spot_provider import BinanceSpotProvider, to_binance_symbol, from_binance_symbol

# Step 14 & 16 & Deep Alternative
from genetic_optimizer import GeneticWeightOptimizer
from cross_asset_engine import CrossAssetCorrelationEngine
from whale_activity_monitor import WhaleActivityMonitor
from breakout_scanner import BreakoutScanner, BreakoutSignal
from polymarket_bridge import PolymarketBridge
from polymarket_scanner import PolymarketScanner
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
from institutional_dashboard import InstitutionalDashboard
from dashboard.event_emitter import DashboardEventEmitter
from position_sizer import RenaissancePositionSizer

from renaissance_types import SignalType, OrderType, MLSignalPackage, TradingDecision
from ml_integration_bridge import MLIntegrationBridge

# Renaissance Medallion-Style Engines
from advanced_mean_reversion_engine import AdvancedMeanReversionEngine
from correlation_network_engine import CorrelationNetworkEngine
from garch_volatility_engine import GARCHVolatilityEngine
from historical_data_cache import HistoricalDataCache

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

# Types moved to renaissance_types.py

def _signed_strength(signal: IndicatorOutput) -> float:
    """Convert a IndicatorOutput into a signed strength value."""
    if not signal:
        return 0.0
    direction = str(signal.signal).upper()
    strength = abs(float(signal.strength))
    if direction == "SELL":
        return -strength
    if direction == "BUY":
        return strength
    return 0.0


def _continuous_rsi_signal(signal: IndicatorOutput) -> float:
    """Convert RSI to continuous signal: oversold(+1) ↔ neutral(0) ↔ overbought(-1)."""
    if not signal:
        return 0.0
    rsi_value = float(signal.value) if signal.value is not None else 50.0
    if np.isnan(rsi_value) or np.isinf(rsi_value):
        return 0.0
    # RSI 0→+1 (oversold=BUY), RSI 50→0, RSI 100→-1 (overbought=SELL)
    return float(np.clip(-(rsi_value - 50.0) / 50.0, -1.0, 1.0))


def _continuous_macd_signal(signal: IndicatorOutput) -> float:
    """Convert MACD histogram to continuous signal using metadata."""
    if not signal or not signal.metadata:
        return 0.0
    hist = signal.metadata.get('histogram', 0.0)
    if hist is None or (hasattr(hist, '__float__') and (np.isnan(float(hist)) or np.isinf(float(hist)))):
        return 0.0
    hist = float(hist)
    # Normalize histogram by a reasonable scale (price-relative)
    # Use signal line as normalizer, fallback to raw clip
    signal_line = abs(float(signal.metadata.get('signal_line', 1.0) or 1.0))
    if signal_line > 0:
        normalized = hist / signal_line
    else:
        normalized = hist
    return float(np.clip(normalized, -1.0, 1.0))


def _continuous_bollinger_signal(signal: IndicatorOutput) -> float:
    """Convert Bollinger position to continuous signal: lower_band(+1) ↔ mid(0) ↔ upper_band(-1)."""
    if not signal:
        return 0.0
    position = float(signal.value) if signal.value is not None else 0.5
    if np.isnan(position) or np.isinf(position):
        return 0.0
    # position 0→+1 (at lower band=BUY), 0.5→0, 1→-1 (at upper band=SELL)
    return float(np.clip(-(position - 0.5) * 2.0, -1.0, 1.0))


def _continuous_obv_signal(signal: IndicatorOutput) -> float:
    """Convert OBV momentum to continuous signal using metadata instead of binary BUY/SELL."""
    if not signal or not signal.metadata:
        return _signed_strength(signal)  # fallback
    obv_momentum = signal.metadata.get('obv_momentum', 0.0)
    obv_change = signal.metadata.get('obv_change', 0.0)
    divergence = signal.metadata.get('divergence', 0)
    if obv_momentum is None:
        obv_momentum = 0.0
    if obv_change is None:
        obv_change = 0.0
    try:
        obv_momentum = float(obv_momentum)
        obv_change = float(obv_change)
        divergence = float(divergence)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(obv_momentum) or np.isinf(obv_momentum):
        return 0.0
    # Blend momentum and change, boost by divergence
    raw = obv_momentum * 3.0 + obv_change * 2.0 + divergence * 0.3
    return float(np.clip(raw, -1.0, 1.0))


def _convert_ws_orderbook_to_snapshot(ws_ob: dict, last_price: float = 0.0) -> 'OrderBookSnapshot':
    """Convert WebSocket order_book dict {bids: {price: size}, asks: {price: size}} to OrderBookSnapshot."""
    from microstructure_engine import OrderBookSnapshot, OrderBookLevel
    bids_dict = ws_ob.get('bids', {})
    asks_dict = ws_ob.get('asks', {})
    bid_levels = [OrderBookLevel(price=p, size=s) for p, s in sorted(bids_dict.items(), reverse=True)[:20]]
    ask_levels = [OrderBookLevel(price=p, size=s) for p, s in sorted(asks_dict.items())[:20]]
    return OrderBookSnapshot(
        timestamp=datetime.now(),
        bids=bid_levels,
        asks=ask_levels,
        last_price=last_price,
        last_size=0.0,
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
        """Initialize the Renaissance trading bot"""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = Path(__file__).resolve().parent / self.config_path

        self.config = self._load_config(self.config_path)
        self.logger = self._setup_logging(self.config)
        self._validate_config(self.config)

        # Log optional module availability at startup
        _modules = {
            'Orchestrator': ORCHESTRATOR_AVAILABLE, 'Arbitrage': ARBITRAGE_AVAILABLE,
            'Recovery': RECOVERY_AVAILABLE, 'Monitoring': MONITORING_AVAILABLE,
            'LiquidationDetector': LIQUIDATION_DETECTOR_AVAILABLE,
            'SignalAggregator': SIGNAL_AGGREGATOR_AVAILABLE,
            'MultiExchangeBridge': MULTI_EXCHANGE_BRIDGE_AVAILABLE,
            'DataValidator': DATA_VALIDATOR_AVAILABLE,
            'SignalThrottle': SIGNAL_THROTTLE_AVAILABLE,
            'SignalValidation': SIGNAL_VALIDATION_AVAILABLE,
            'HealthMonitor': HEALTH_MONITOR_AVAILABLE,
            'MedallionAnalogs': MEDALLION_ANALOGS_AVAILABLE,
            'PortfolioEngine': PORTFOLIO_ENGINE_AVAILABLE,
            'DevilTracker': DEVIL_TRACKER_AVAILABLE,
            'KellySizer': KELLY_SIZER_AVAILABLE,
        }
        active = [k for k, v in _modules.items() if v]
        missing = [k for k, v in _modules.items() if not v]
        self.logger.info(f"Module status: {len(active)}/{len(_modules)} loaded | missing: {missing if missing else 'none'}")

        # Multi-Asset Support
        self.product_ids = self.config.get("trading", {}).get("product_ids", ["BTC-USD"])
        self.config_manager = EnhancedConfigManager("config")

        # Initialize all components
        self.microstructure_engine = MicrostructureEngine()
        # Per-asset technical indicators — prevents signal bleed across products
        self._tech_indicators: Dict[str, EnhancedTechnicalIndicators] = {
            pid: EnhancedTechnicalIndicators() for pid in self.product_ids
        }
        self.market_data_provider = LiveMarketDataProvider(self.config, logger=self.logger)
        self.derivatives_provider = DerivativesDataProvider(cache_ttl_seconds=60)
        
        # Unified Signal Fusion (Step 16+)
        self.signal_fusion = SignalFusion()
        self.signal_fusion.set_ml_signal_scale(self.config.get("ml_signal_scale", 10.0))
        
        self.alternative_data_engine = AlternativeDataEngine(self.config, logger=self.logger)

        # Initialize Advanced Adapters (Step 7 & 9)
        _regime_db = self.config.get('database', {}).get('path', 'data/renaissance_bot.db')
        self.regime_overlay = RegimeOverlay(self.config.get("regime_overlay", {}), logger=self.logger, db_path=_regime_db)
        self.risk_gateway = RiskGateway(self.config.get("risk_gateway", {}), logger=self.logger)
        
        # Initialize the core risk manager (moved from gateway integration logic)
        self.risk_manager = RiskManager(
            position_limit=self.config.get("risk_management", {}).get("position_limit", 1000.0)
        )
        
        self.real_time_pipeline = RealTimePipeline(self.config.get("real_time_pipeline", {}), logger=self.logger)

        # Initialize Step 10 Components
        # self.portfolio_optimizer = RenaissancePortfolioOptimizer()
        self.execution_suite = ExecutionAlgorithmSuite()
        self.slippage_protection = SlippageProtectionSystem()

        # Risk management (from original bot) - needed before position_manager init
        risk_cfg = self.config.get("risk_management", {})
        self.daily_loss_limit = float(risk_cfg.get("daily_loss_limit", 500))
        self.position_limit = float(risk_cfg.get("position_limit", 1000))
        self.min_confidence = float(risk_cfg.get("min_confidence", 0.50))

        # Initialize Coinbase Client & Position Manager
        cb_config = self.config.get("coinbase", {})
        paper_mode = self.config.get("trading", {}).get("paper_trading", True)
        self.paper_trading = paper_mode
        self.coinbase_client = EnhancedCoinbaseClient(
            credentials=CoinbaseCredentials(
                api_key=os.environ.get(cb_config.get("api_key_env", "CB_API_KEY"), ""),
                api_secret=os.environ.get(cb_config.get("api_secret_env", "CB_API_SECRET"), ""),
                sandbox=self.config.get("trading", {}).get("sandbox", True),
            ),
            paper_trading=paper_mode,
            logger=self.logger,
        )
        self.position_manager = EnhancedPositionManager(
            coinbase_client=self.coinbase_client,
            risk_limits=RiskLimits(
                max_position_size_usd=self.position_limit,
                max_daily_loss_usd=self.daily_loss_limit,
            ),
            logger=self.logger,
        )

        # Renaissance-inspired position sizer (Kelly + cost gate + vol normalization)
        # "Small bets, many times. We are the casino, not the gambler."
        self.position_sizer = RenaissancePositionSizer(
            config={
                "default_balance_usd": 50000.0,    # Fallback if balance fetch fails
                "max_position_pct": 10.0,          # Max 10% of balance per position
                "max_total_exposure_pct": 50.0,    # Max 50% total exposure
                "kelly_fraction": 0.50,            # Half-Kelly for drawdown control
                "min_edge": 0.001,                 # 0.1% minimum edge
                "min_win_prob": 0.52,              # Need > 52% to trade
                "taker_fee_bps": 5.0,              # MEXC taker 0.05% (backup, rarely used)
                "maker_fee_bps": 0.0,              # MEXC maker = FREE (our default execution)
                "spread_cost_bps": 2.0,            # Tight for MEXC top pairs
                "slippage_bps": 1.0,               # Limit orders = minimal slippage
                "cost_gate_ratio": 0.50,           # Block if cost > 50% of expected profit
                "target_vol": 0.02,                # Target 2% daily vol
                "min_order_usd": 1.0,              # MEXC min order ~$1
            },
            logger=self.logger,
        )
        self._cached_balance_usd: float = 0.0   # Updated each cycle
        self._high_watermark_usd: float = 0.0  # Track peak balance for drawdown
        self._current_drawdown_pct: float = 0.0 # Current drawdown from HWM
        self._weekly_pnl: float = 0.0           # Track weekly P&L
        self._week_start_balance: float = 0.0   # Balance at start of week
        self._week_reset_today: bool = False     # Weekly reset flag

        # ── Data Validator (Medallion: trust but verify all inputs) ──
        self.data_validator = DataValidator(logger=self.logger) if DATA_VALIDATOR_AVAILABLE else None

        # ── Signal Auto-Throttle (Medallion: kill losers fast) ──
        throttle_cfg = self.config.get('signal_throttle', {})
        self.signal_throttle = SignalAutoThrottle(throttle_cfg, logger=self.logger) if SIGNAL_THROTTLE_AVAILABLE else None

        # ── Signal Validation Gate (Medallion: every signal earns its place) ──
        self.signal_validation_gate = SignalValidationGate(logger=self.logger) if SIGNAL_VALIDATION_AVAILABLE else None

        # ── Portfolio Health Monitor (Medallion: rolling Sharpe as health metric) ──
        health_cfg = self.config.get('health_monitor', {})
        self.health_monitor = PortfolioHealthMonitor(health_cfg, logger=self.logger) if HEALTH_MONITOR_AVAILABLE else None

        # ── Medallion Signal Analogs (sharp move reversion, seasonality, funding timing) ──
        analog_cfg = self.config.get('medallion_analogs', {})
        self.medallion_analogs = MedallionSignalAnalogs(analog_cfg, logger=self.logger) if MEDALLION_ANALOGS_AVAILABLE else None

        # ── Unified Portfolio Engine (Medallion: all products as one portfolio) ──
        portfolio_cfg = self.config.get('portfolio_engine', {})
        self.portfolio_engine = UnifiedPortfolioEngine(portfolio_cfg, logger=self.logger) if PORTFOLIO_ENGINE_AVAILABLE and portfolio_cfg.get('enabled', False) else None

        # ── Medallion Intelligence Modules (Phase 2) ──
        db_path = self.config.get('database', {}).get('path', 'data/renaissance_bot.db')

        self.devil_tracker = DevilTracker(db_path) if DEVIL_TRACKER_AVAILABLE else None
        if self.devil_tracker:
            self.logger.info("DevilTracker: ACTIVE — tracking signal→fill execution quality")

        self.kelly_sizer = KellyPositionSizer(self.config, db_path) if KELLY_SIZER_AVAILABLE else None
        if self.kelly_sizer:
            self.logger.info("KellyPositionSizer: ACTIVE — optimal sizing from trade history")

        # Daily Signal Review — end-of-day P&L audit per signal type (distinct from intra-day signal_throttle)
        self.daily_signal_review = MedallionSignalThrottle(self.config, db_path) if MEDALLION_THROTTLE_AVAILABLE else None
        if self.daily_signal_review:
            self.logger.info("DailySignalReview: ACTIVE — end-of-day P&L throttling")

        self.leverage_mgr = LeverageManager(self.config, db_path) if LEVERAGE_MANAGER_AVAILABLE else None
        if self.leverage_mgr:
            self.logger.info("LeverageManager: ACTIVE — consistency-based leverage")

        self.medallion_regime = MedallionRegimeDetector(self.config, db_path) if MEDALLION_REGIME_AVAILABLE else None
        if self.medallion_regime:
            self.logger.info("MedallionRegimeDetector: OBSERVATION — logging alongside RegimeOverlay")

        self.insurance_scanner = InsurancePremiumScanner(self.config, db_path) if INSURANCE_SCANNER_AVAILABLE else None
        if self.insurance_scanner:
            self.logger.info("InsurancePremiumScanner: OBSERVATION — periodic premium scanning")

        # Medallion Portfolio Engine — target/actual reconciliation in observation mode (log drift, don't execute)
        if MEDALLION_PORTFOLIO_ENGINE_AVAILABLE:
            pe_cfg = self.config
            self.medallion_portfolio_engine = MedallionPortfolioEngine(
                config=pe_cfg,
                devil_tracker=self.devil_tracker,
                position_manager=self.position_manager,
            )
            self.logger.info("MedallionPortfolioEngine: OBSERVATION — drift logging (no corrections)")
        else:
            self.medallion_portfolio_engine = None

        self.bar_aggregator = BarAggregator(self.config, db_path) if BAR_AGGREGATOR_AVAILABLE else None
        if self.bar_aggregator:
            self.logger.info("BarAggregator: ACTIVE — 5-min bar aggregation")

        self.sync_executor = SynchronizedExecutor(self.config, self.devil_tracker) if SYNC_EXECUTOR_AVAILABLE else None
        if self.sync_executor:
            self.logger.info("SynchronizedExecutor: ACTIVE — cross-exchange execution")

        self.trade_hider = TradeHider(self.config) if TRADE_HIDER_AVAILABLE else None
        if self.trade_hider:
            self.logger.info("TradeHider: ACTIVE — execution obfuscation")

        self.beta_monitor = BetaMonitor(self.config, db_path) if BETA_MONITOR_AVAILABLE else None
        if self.beta_monitor:
            self.logger.info("BetaMonitor: ACTIVE — portfolio beta tracking")

        self.capacity_monitor = CapacityMonitor(self.config, db_path) if CAPACITY_MONITOR_AVAILABLE else None
        if self.capacity_monitor:
            self.logger.info("CapacityMonitor: ACTIVE — capacity wall detection")

        self.sharpe_monitor_medallion = SharpeMonitor(self.config, db_path) if SHARPE_MONITOR_AVAILABLE else None
        if self.sharpe_monitor_medallion:
            self.logger.info("SharpeMonitor: ACTIVE — rolling Sharpe health")

        # ── Multi-Horizon Probability Estimator (Doc 11) ──
        self.mhpe = None
        if MHPE_AVAILABLE:
            try:
                self.mhpe = MultiHorizonEstimator(
                    config=self.config.get('multi_horizon_estimator', {}),
                    regime_predictor=self.medallion_regime,
                )
                self.logger.info("MHPE: ACTIVE — 7-horizon probability cones")
            except Exception as _mhpe_err:
                self.logger.warning(f"MHPE init failed: {_mhpe_err}")

        # ── Position Re-evaluator (Doc 10) ──
        self.position_reevaluator = None
        if POSITION_REEVALUATOR_AVAILABLE:
            try:
                self.position_reevaluator = PositionReEvaluator(
                    config=self.config.get('reevaluation', {}),
                    kelly_sizer=self.kelly_sizer,
                    regime_detector=self.medallion_regime,
                    devil_tracker=self.devil_tracker,
                    mhpe=self.mhpe,
                )
                self.logger.info("PositionReEvaluator: ACTIVE — continuous position re-evaluation")
            except Exception as _re_err:
                self.logger.warning(f"PositionReEvaluator init failed: {_re_err}")

        # ── Signal Scorecard (Renaissance: measure everything) ──
        # Records {product_id: {signal_name: {"correct": N, "total": N}}}
        self._signal_scorecard: Dict[str, Dict[str, Dict[str, int]]] = {}
        # Pending predictions: {product_id: {"price": float, "signals": {name: value}, "cycle": int}}
        self._pending_predictions: Dict[str, Dict] = {}
        # Adaptive weight engine: how much to blend scorecard-based weights vs config
        self._adaptive_weight_blend = 0.0  # starts at 0 (pure config), ramps to 0.5 max
        self._adaptive_min_samples = 15    # need this many observations before adjusting

        self._killed = False
        self._start_time = datetime.now(timezone.utc)
        self._background_tasks: list = []
        self._weights_lock = asyncio.Lock()

        # Initialize Alert Manager
        alert_cfg = self.config.get("alerting", {})
        self.alert_manager = AlertManager(alert_cfg, logger=self.logger)

        # Initialize WebSocket feed (real-time market data)
        self._ws_queue: queue.Queue = queue.Queue(maxsize=1000)
        try:
            ws_config = {
                'api_key': os.environ.get(cb_config.get("api_key_env", "CB_API_KEY"), ""),
                'api_secret': os.environ.get(cb_config.get("api_secret_env", "CB_API_SECRET"), ""),
                'passphrase': os.environ.get(cb_config.get("api_passphrase_env", ""), ""),
                'sandbox': self.config.get("trading", {}).get("sandbox", True),
                'symbols': self.product_ids,
                'websocket_channels': ["level2", "ticker", "matches"],
            }
            self._ws_client = CoinbaseAdvancedClient(ws_config)
        except Exception as e:
            self.logger.warning(f"WebSocket client init failed (will use REST fallback): {e}")
            self._ws_client = None

        # Initialize Persistence & Attribution
        db_cfg = self.config.get("database", {"path": "data/renaissance_bot.db", "enabled": True})
        self.db_enabled = db_cfg.get("enabled", True)
        self.db_manager = DatabaseManager(db_cfg)
        self.attribution_engine = PerformanceAttributionEngine()
        
        # Initialize Evolutionary & Global Intelligence
        self.genetic_optimizer = GeneticWeightOptimizer(db_cfg.get("path", "data/renaissance_bot.db"), logger=self.logger)
        self.correlation_engine = CrossAssetCorrelationEngine(logger=self.logger)
        from statistical_arbitrage_engine import StatisticalArbitrageEngine
        self.stat_arb_engine = StatisticalArbitrageEngine(logger=self.logger)
        self.whale_monitor = WhaleActivityMonitor(self.config.get("whale_monitor", {}), logger=self.logger)

        # Renaissance Medallion-Style Engines
        mr_cfg = self.config.get("mean_reversion", {})
        self.mean_reversion_engine = AdvancedMeanReversionEngine(mr_cfg, logger=self.logger)

        corr_net_cfg = self.config.get("correlation_network", {})
        self.correlation_network = CorrelationNetworkEngine(corr_net_cfg, logger=self.logger)

        garch_cfg = self.config.get("garch_volatility", {})
        self.garch_engine = GARCHVolatilityEngine(garch_cfg, logger=self.logger)

        hist_cfg = self.config.get("historical_data_cache", {})
        hist_cfg.setdefault("db_path", db_cfg.get("path", "data/renaissance_bot.db"))
        self.historical_cache = HistoricalDataCache(hist_cfg, logger=self.logger)
        if hist_cfg.get("enabled", False):
            self.historical_cache.init_tables()

        # Initialize Breakout Scanner — scans ALL 600+ Binance pairs in 1 API call
        scanner_cfg = self.config.get("breakout_scanner", {"enabled": True, "max_flagged": 30})
        self.breakout_scanner = BreakoutScanner(
            max_flagged=scanner_cfg.get("max_flagged", 30),
            min_volume_usd=scanner_cfg.get("min_volume_usd", 500_000),
            min_breakout_score=scanner_cfg.get("min_breakout_score", 25.0),
            logger=self.logger,
        )
        self.scanner_enabled = scanner_cfg.get("enabled", True)
        self._breakout_scores: Dict[str, BreakoutSignal] = {}  # Current cycle's breakout signals

        # Initialize Polymarket Bridge — converts ML signals to binary bet signals
        poly_cfg = self.config.get("polymarket_bridge", {})
        self.polymarket_bridge = PolymarketBridge(
            min_prediction=poly_cfg.get("min_prediction", 0.03),
            min_agreement=poly_cfg.get("min_agreement", 0.55),
            observation_mode=poly_cfg.get("observation_mode", True),
            logger=self.logger,
        )

        # Initialize Polymarket Scanner — discovers all crypto prediction markets
        scanner_db = db_cfg.get("path", "data/renaissance_bot.db")
        poly_scanner_cfg = self.config.get("polymarket_scanner", {})
        self.polymarket_scanner = PolymarketScanner(
            db_path=scanner_db,
            cache_ttl=poly_scanner_cfg.get("cache_ttl", 300),
            logger=self.logger,
        )
        self._last_poly_scan: Optional[datetime] = None
        self._latest_scanner_opportunities: List[dict] = []

        self.volume_profile_engine = VolumeProfileEngine()
        self.fractal_intelligence = FractalIntelligenceEngine(logger=self.logger)
        self.market_entropy = MarketEntropyEngine(logger=self.logger)
        self.quantum_oscillator = QuantumOscillatorEngine(logger=self.logger)
        self._last_vp_status = {} # product_id -> status
        
        # Initialize Ghost Runner (Step 18)
        self.ghost_runner = GhostRunner(self, logger=self.logger)
        
        # Initialize Self-Reinforcing Learning Engine (Step 19)
        from self_reinforcing_learning import SelfReinforcingLearningEngine
        self.learning_engine = SelfReinforcingLearningEngine(db_cfg.get("path", "data/renaissance_bot.db"), logger=self.logger)
        
        # Initialize Confluence Engine (Step 20 - Non-linear Meta-Learning)
        from confluence_engine import ConfluenceEngine
        self.confluence_engine = ConfluenceEngine(logger=self.logger)
        
        # 🏛️ Basis Trading Engine
        self.basis_engine = BasisTradingEngine(logger=self.logger)
        
        # 🧠 Deep NLP Bridge
        self.nlp_bridge = DeepNLPBridge(self.config.get("nlp", {}), logger=self.logger)
        
        # ⚖️ Market Making Engine
        self.market_making = MarketMakingEngine(self.config.get("market_making", {}), logger=self.logger)

        # 🚀 Meta-Strategy Selector (Step 11/13 Refinement)
        self.strategy_selector = MetaStrategySelector(self.config.get("meta_strategy", {}), logger=self.logger)
        
        # 🤖 ML Integration Bridge (Unified from Enhanced Bot)
        self.ml_enabled = self.config.get("ml_integration", {}).get("enabled", True)
        self.ml_bridge = MLIntegrationBridge(self.config)
        if self.ml_enabled:
            self.ml_bridge.initialize()
            # Validate trained models loaded correctly
            loaded_models = list(self.ml_bridge.model_manager.models.keys())
            if loaded_models:
                self.logger.info(f"ML startup validation: {len(loaded_models)} trained models active: {loaded_models}")
            else:
                self.logger.warning("ML startup validation: NO trained models loaded — ML predictions will be empty")

            # Check model staleness
            metadata_path = os.path.join("models", "trained", "training_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path) as f:
                        training_meta = json.load(f)
                    for model_name, info in training_meta.items():
                        last_trained = datetime.fromisoformat(info["last_trained"].replace("Z", "+00:00"))
                        age_days = (datetime.now(timezone.utc) - last_trained).days
                        if age_days > 7:
                            self.logger.warning(
                                f"ML staleness: {model_name} last trained {age_days} days ago — retraining recommended"
                            )
                except Exception as e:
                    self.logger.debug(f"Could not check model staleness: {e}")

        # Performance Tracking
        self.ml_performance_metrics = {
            'total_trades': 0,
            'ml_enhanced_trades': 0,
            'avg_ml_processing_time': 0.0,
            'ml_success_rate': 0.0
        }

        # State tracking for Dashboard
        self.last_vpin = 0.5
        
        # 📊 Dashboard Event Emitter (real-time dashboard integration)
        self.dashboard_emitter = DashboardEventEmitter()
        self.dashboard_emitter.clear_cache()  # Flush stale data from prior session

        # ── Operations & Intelligence Modules ──────────────────────────
        # Module A: Disaster Recovery — State Manager & Graceful Shutdown
        self.state_manager = None
        self.shutdown_handler = None
        if RECOVERY_AVAILABLE:
            try:
                db_path = self.config.get("database", {}).get("path", "data/renaissance_bot.db")
                # Run database migration (idempotent — adds new tables if missing)
                ensure_all_tables(db_path)
                self.state_manager = StateManager()
                self.logger.info("Recovery StateManager initialized")
            except Exception as e:
                self.logger.warning(f"Recovery module init failed: {e}")

        # Module C: Monitoring — Telegram Alerter & Enhanced Alert Manager
        self.monitoring_alert_manager = None
        if MONITORING_AVAILABLE:
            try:
                telegram_cfg = self.config.get("telegram", {})
                telegram_alerter = TelegramAlerter(config=telegram_cfg)
                db_path = self.config.get("database", {}).get("path", "data/renaissance_bot.db")
                self.monitoring_alert_manager = MonitoringAlertManager(
                    telegram_alerter=telegram_alerter,
                    db_path=db_path,
                )
                self.logger.info("Monitoring AlertManager initialized (Telegram %s)",
                    "active" if telegram_alerter._bot_token else "console-only")
            except Exception as e:
                self.logger.warning(f"Monitoring module init failed: {e}")

        # Module D: Liquidation Cascade Detector
        self.liquidation_detector = None
        if LIQUIDATION_DETECTOR_AVAILABLE:
            try:
                liq_cfg = self.config.get("liquidation_detector", {})
                self.liquidation_detector = LiquidationCascadeDetector(config=liq_cfg)
                self.logger.info("Liquidation Cascade Detector initialized")
            except Exception as e:
                self.logger.warning(f"Liquidation detector init failed: {e}")

        # Fast Mean Reversion Scanner (1s evaluation)
        self.fast_reversion_scanner = None
        fmr_cfg = self.config.get("fast_mean_reversion", {})
        if fmr_cfg.get("enabled", False) and FAST_REVERSION_AVAILABLE:
            try:
                self.fast_reversion_scanner = FastMeanReversionScanner(
                    fmr_cfg, self.bar_aggregator
                )
                self.logger.info("Fast Mean Reversion Scanner initialized")
            except Exception as e:
                self.logger.warning(f"Fast reversion scanner init failed: {e}")

        # Heartbeat Writer (multi-bot coordination)
        self.heartbeat_writer = None
        bot_id = self.config.get("bot_id", "bot-01")
        orch_cfg = self.config.get("orchestrator", {})
        if HEARTBEAT_AVAILABLE:
            try:
                self.heartbeat_writer = HeartbeatWriter(
                    bot_id=bot_id,
                    heartbeat_dir=orch_cfg.get("heartbeat_dir", "data/heartbeats"),
                )
                self.logger.info(f"HeartbeatWriter initialized (bot_id={bot_id})")
            except Exception as e:
                self.logger.warning(f"HeartbeatWriter init failed: {e}")

        # Module F: Advanced Microstructure Signal Aggregator
        self.signal_aggregator = None
        if SIGNAL_AGGREGATOR_AVAILABLE:
            try:
                micro_weights = self.config.get("microstructure_signals", {}).get("weights", None)
                self.signal_aggregator = SignalAggregator(weights=micro_weights)
                self.logger.info("Advanced Signal Aggregator initialized")
            except Exception as e:
                self.logger.warning(f"Signal aggregator init failed: {e}")

        # 📊 Institutional Dashboard (legacy Flask stub)
        self.dashboard_enabled = self.config.get("institutional_dashboard", {}).get("enabled", True)
        if self.dashboard_enabled:
            try:
                _dash_port = int(self.config.get("institutional_dashboard", {}).get("port", 5050))
                self.dashboard = InstitutionalDashboard(self, host="0.0.0.0", port=_dash_port)
                self.dashboard.run()
            except Exception as e:
                self.logger.warning(f"Failed to start dashboard (likely port conflict): {e}")
                self.dashboard = None
        else:
            self.dashboard = None

        # 📊 FastAPI Real-Time Dashboard
        self._dashboard_server_task = None
        dash_cfg = self.config.get("dashboard_config", {})
        if dash_cfg.get("enabled", True):
            try:
                from dashboard.server import create_app
                import threading
                import uvicorn
                self._dashboard_app = create_app(
                    config_path=str(self.config_path),
                    emitter=self.dashboard_emitter,
                )
                dash_port = dash_cfg.get("port", 8080)
                dash_host = dash_cfg.get("host", "0.0.0.0")
                def _run_dashboard():
                    uvicorn.run(self._dashboard_app, host=dash_host, port=dash_port, log_level="warning")
                threading.Thread(target=_run_dashboard, daemon=True).start()
                self.logger.info(f"Real-time dashboard started on {dash_host}:{dash_port}")
            except Exception as e:
                self.logger.warning(f"Failed to start real-time dashboard: {e}")

        # Multi-Exchange Arbitrage Engine
        self.arbitrage_enabled = self.config.get("arbitrage", {}).get("enabled", False)
        self.arbitrage_orchestrator = None
        if self.arbitrage_enabled and ARBITRAGE_AVAILABLE:
            try:
                arb_config_path = self.config.get("arbitrage", {}).get(
                    "config_path", "arbitrage/config/arbitrage.yaml"
                )
                self.arbitrage_orchestrator = ArbitrageOrchestrator(config_path=arb_config_path)
                self.logger.info("Arbitrage engine initialized (will start with trading loop)")
                # Wire orchestrator to dashboard for API access
                if hasattr(self, '_dashboard_app') and self._dashboard_app:
                    self._dashboard_app.state.arb_orchestrator = self.arbitrage_orchestrator
            except Exception as e:
                self.logger.warning(f"Arbitrage engine init failed: {e}")
                self.arbitrage_orchestrator = None
        elif self.arbitrage_enabled and not ARBITRAGE_AVAILABLE:
            self.logger.warning("Arbitrage enabled in config but module not available")

        # ── Multi-Exchange Signal Bridge ──
        self.multi_exchange_bridge = None
        me_cfg = self.config.get("multi_exchange_signals", {})
        if me_cfg.get("enabled", False) and MULTI_EXCHANGE_BRIDGE_AVAILABLE and self.arbitrage_orchestrator:
            try:
                self.multi_exchange_bridge = MultiExchangeBridge(
                    book_manager=self.arbitrage_orchestrator.book_manager,
                    mexc_client=self.arbitrage_orchestrator.mexc,
                    binance_client=self.arbitrage_orchestrator.binance,
                )
                self.logger.info("Multi-exchange signal bridge initialized")
            except Exception as e:
                self.logger.warning(f"Multi-exchange bridge init failed: {e}")

        # ── Binance Spot Provider (primary data source for expanded universe) ──
        self.binance_spot = BinanceSpotProvider(logger=self.logger)

        # Dynamic universe state (populated async in run_continuous_trading)
        self.trading_universe: list = []
        self._pair_tiers: Dict[str, int] = {}
        self._pair_binance_symbols: Dict[str, str] = {}
        self._universe_built = False
        self._universe_last_refresh: float = 0.0

        # Step 8: Dynamic Thresholds (calibrated to actual signal distribution)
        # Backtest proof: |prediction| < 0.06 is noise (51% accuracy).
        # Only trade strong ML signals — configurable via config.json.
        self.buy_threshold = float(self.config.get('trading', {}).get('buy_threshold', 0.06))
        self.sell_threshold = float(self.config.get('trading', {}).get('sell_threshold', -0.06))
        self.adaptive_thresholds = self.config.get("adaptive_thresholds", True)
        self.breakout_candidates = []
        self.scan_cycle_count = 0
        # Signal filter stats — tracks how selective the pipeline is
        self._signal_filter_stats = {
            'total': 0, 'traded': 0, 'filtered_threshold': 0,
            'filtered_confidence': 0, 'filtered_agreement': 0,
        }

        if self.db_enabled:
            self._track_task(self.db_manager.init_database())

        # Renaissance Research-Optimized Signal Weights (17 signals — ML included)
        raw_weights = self.config.get("signal_weights", {
            'order_flow': 0.14,               # Institutional Flow (reduced from 0.18)
            'order_book': 0.12,               # Microstructure
            'volume': 0.08,                   # Volume
            'macd': 0.05,                     # Momentum
            'rsi': 0.05,                      # Mean Reversion (technical)
            'bollinger': 0.05,                # Volatility Bands
            'alternative': 0.01,              # Sentiment/Whales (reduced — often zero)
            'stat_arb': 0.12,                 # Multi-pair Mean Reversion
            'volume_profile': 0.04,           # Volume Profile
            'fractal': 0.05,                  # Fractal Intelligence
            'entropy': 0.04,                  # Market Entropy
            'quantum': 0.02,                  # Quantum Oscillator (reduced — heuristic)
            'lead_lag': 0.03,                 # Cross-Asset Lead-Lag
            'correlation_divergence': 0.06,   # Correlation Network Divergence
            'garch_vol': 0.06,                # GARCH Volatility Signal
            'ml_ensemble': 0.05,              # ML 7-model ensemble prediction
            'ml_cnn': 0.03,                   # ML CNN model prediction
            'breakout': 0.08,                 # Breakout scanner signal (Binance-wide)
        })
        self.signal_weights = {str(k): float(self._force_float(v)) for k, v in raw_weights.items()}

        # Ensure ML weights always present (genetic optimizer may drop them)
        _ml_required = {'ml_ensemble': 0.05, 'ml_cnn': 0.03}
        for k, v in _ml_required.items():
            if k not in self.signal_weights:
                self.signal_weights[k] = v
                self.logger.info(f"Injected missing ML weight: {k}={v}")

        # Trading state
        self.current_position = 0.0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self.decision_history = []

        # ── Enhanced Config Validation (Audit 4) ──
        validate_config(self.config, self.logger)

        # ── Doc 15: Agent Coordination System ──
        self.agent_coordinator = None
        if AGENT_COORDINATOR_AVAILABLE:
            try:
                _agent_db = self.config.get('database', {}).get('path', 'data/renaissance_bot.db')
                self.agent_coordinator = AgentCoordinator(
                    bot=self, db_path=_agent_db, config=self.config, bot_logger=self.logger,
                )
            except Exception as _ac_err:
                self.logger.warning(f"AgentCoordinator init failed (trading unaffected): {_ac_err}")

        self.logger.info("Renaissance Trading Bot initialized with research-optimized weights")
        self.logger.info(f"Signal weights: {self.signal_weights}")

    def get_pairs_for_cycle(self, cycle_number: int) -> list:
        """Return pairs to scan this cycle based on 4-tier volume schedule.

        Tier 1 (top 15 by volume):   every cycle
        Tier 2 (16-50):              every 2nd cycle
        Tier 3 (51-100):             every 3rd cycle
        Tier 4 (101-150):            every 4th cycle
        """
        if not self._pair_tiers:
            # Fallback: scan all product_ids if universe not built yet
            return list(self.product_ids)

        pairs = []
        for pid in self.product_ids:
            tier = self._pair_tiers.get(pid, 1)
            if tier == 1:
                pairs.append(pid)
            elif tier == 2 and cycle_number % 2 == 0:
                pairs.append(pid)
            elif tier == 3 and cycle_number % 3 == 0:
                pairs.append(pid)
            elif tier == 4 and cycle_number % 4 == 0:
                pairs.append(pid)
        return pairs

    async def _build_and_apply_universe(self) -> None:
        """Build dynamic trading universe from Binance and apply it."""
        try:
            universe_cfg = self.config.get('universe', {})
            min_vol = float(universe_cfg.get('min_volume_usd', 2_000_000))
            max_pairs = int(universe_cfg.get('max_pairs', 150))

            universe = await self.binance_spot.build_trading_universe(
                min_volume_usd=min_vol, max_pairs=max_pairs,
            )
            if not universe:
                self.logger.warning("UNIVERSE: Binance returned empty — keeping existing product_ids")
                return

            self.trading_universe = universe
            self.product_ids = [c['product_id'] for c in universe]
            self._pair_tiers = {c['product_id']: c['tier'] for c in universe}
            self._pair_binance_symbols = {
                c['product_id']: c['binance_symbol'] for c in universe
            }
            self._universe_built = True
            self._universe_last_refresh = time.time()
            self.logger.info(f"UNIVERSE BUILT: {len(self.product_ids)} pairs")
        except Exception as e:
            self.logger.error(f"UNIVERSE BUILD FAILED: {e} — keeping existing product_ids")

    async def _collect_from_binance(self, product_id: str) -> Dict[str, Any]:
        """Collect market data from Binance for a single pair.

        Returns a market_data dict compatible with the existing pipeline.
        """
        binance_sym = self._pair_binance_symbols.get(product_id)
        if not binance_sym:
            binance_sym = to_binance_symbol(product_id)

        try:
            ticker = await self.binance_spot.fetch_ticker(binance_sym)
            if not ticker or ticker.get('price', 0) <= 0:
                return {}

            # Build market_data dict compatible with existing pipeline
            tech = self._get_tech(product_id)

            # Fetch latest candle for tech indicator feed
            candles = await self.binance_spot.fetch_candles(binance_sym, '5m', 2)
            if candles:
                from enhanced_technical_indicators import PriceData
                latest = candles[-1]
                price_data = PriceData(
                    timestamp=datetime.utcfromtimestamp(latest['timestamp']),
                    open=latest['open'],
                    high=latest['high'],
                    low=latest['low'],
                    close=latest['close'],
                    volume=latest['volume'],
                )
                tech.update_price_data(price_data)
            else:
                price_data = None

            technical_signals = tech.get_latest_signals()

            # Build orderbook snapshot for microstructure signals
            order_book_snapshot = None
            try:
                ob = await self.binance_spot.fetch_orderbook(binance_sym, 20)
                if ob and ob.get('bids') and ob.get('asks'):
                    from microstructure_engine import OrderBookSnapshot, OrderBookLevel
                    bids = [OrderBookLevel(price=p, size=s) for p, s in ob['bids']]
                    asks = [OrderBookLevel(price=p, size=s) for p, s in ob['asks']]
                    order_book_snapshot = OrderBookSnapshot(
                        timestamp=datetime.utcnow(),
                        bids=bids,
                        asks=asks,
                        last_price=ticker['price'],
                        last_size=0.0,
                    )
            except Exception:
                pass

            # Compute bid_ask_spread
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            spread = ask - bid if bid > 0 and ask > 0 else 0.0

            return {
                'order_book_snapshot': order_book_snapshot,
                'price_data': price_data,
                'technical_signals': technical_signals,
                'alternative_signals': {},  # Filled later in sequential phase
                'ticker': {
                    'price': ticker['price'],
                    'bid': bid,
                    'ask': ask,
                    'best_bid': bid,
                    'best_ask': ask,
                    'volume': ticker.get('volume_24h', 0),
                    'volume_24h': ticker.get('volume_24h', 0),
                    'quote_volume_24h': ticker.get('quote_volume_24h', 0),
                    'bid_ask_spread': spread,
                },
                'product_id': product_id,
                'timestamp': datetime.now(),
                'recent_trades': [],
                '_data_source': 'binance',
            }
        except Exception as e:
            self.logger.debug(f"Binance collect failed for {product_id}: {e}")
            return {}

    def _get_tech(self, product_id: str) -> 'EnhancedTechnicalIndicators':
        """Get per-asset technical indicators instance (creates on-demand for new assets)."""
        if product_id not in self._tech_indicators:
            self._tech_indicators[product_id] = EnhancedTechnicalIndicators()
        return self._tech_indicators[product_id]

    def _load_price_df_from_db(self, product_id: str, limit: int = 100):
        """Load recent OHLCV bars from DB for ML inference when tech indicators are sparse."""
        try:
            import pandas as _pd
            import sqlite3
            db_path = self.config.get('database', {}).get('path', 'data/renaissance_bot.db')
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            rows = conn.execute(
                "SELECT bar_start, open, high, low, close, volume "
                "FROM five_minute_bars WHERE pair=? ORDER BY bar_start DESC LIMIT ?",
                (product_id, limit)
            ).fetchall()
            conn.close()
            if len(rows) < 30:
                return _pd.DataFrame()
            df = _pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        except Exception as e:
            self.logger.debug(f"DB bar load failed for {product_id}: {e}")
            import pandas as _pd
            return _pd.DataFrame()

    def _load_candles_from_db(self, product_id: str, limit: int = 200) -> List:
        """Load historical bars from five_minute_bars as PriceData objects for tech indicator bootstrap."""
        try:
            import sqlite3
            from enhanced_technical_indicators import PriceData
            db_path = self.config.get('database', {}).get('path', 'data/renaissance_bot.db')
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            rows = conn.execute(
                "SELECT bar_start, open, high, low, close, volume "
                "FROM five_minute_bars WHERE pair=? ORDER BY bar_start ASC LIMIT ?",
                (product_id, limit)
            ).fetchall()
            conn.close()
            if not rows:
                return []
            candles = []
            for row in rows:
                ts, o, h, l, c, v = row
                candles.append(PriceData(
                    timestamp=datetime.fromtimestamp(ts),
                    open=float(o), high=float(h), low=float(l),
                    close=float(c), volume=float(v or 0),
                ))
            self.logger.info(f"Loaded {len(candles)} bars from DB for {product_id}")
            return candles
        except Exception as e:
            self.logger.warning(f"DB candle load failed for {product_id}: {e}")
            return []

    def _setup_logging(self, config: Dict[str, Any]) -> logging.Logger:
        """Setup comprehensive logging"""
        log_cfg = config.get("logging", {})
        log_file = log_cfg.get("file", "logs/renaissance_bot.log")
        log_level = log_cfg.get("level", "INFO")

        log_path = (Path(__file__).resolve().parent / log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use force=True to override any handlers set by imports
        logging.basicConfig(
            level=getattr(logging, str(log_level).upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.handlers.RotatingFileHandler(
                    log_path, maxBytes=50 * 1024 * 1024, backupCount=5
                ),
                logging.StreamHandler()
            ],
            force=True,
        )

        # Apply secret masking to all handlers
        masking_filter = SecretMaskingFilter()
        for handler in logging.getLogger().handlers:
            handler.addFilter(masking_filter)

        return logging.getLogger(__name__)

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load bot configuration from JSON file."""
        default_config = {
            "trading": {
                "product_id": "BTC-USD",
                "cycle_interval_seconds": 300,
                "paper_trading": True,
                "sandbox": True
            },
            "risk_management": {
                "daily_loss_limit": 500,
                "position_limit": 1000,
                "min_confidence": 0.50
            },
            "signal_weights": {
                "order_flow": 0.32,
                "order_book": 0.21,
                "volume": 0.14,
                "macd": 0.105,
                "rsi": 0.115,
                "bollinger": 0.095,
                "alternative": 0.045
            },
            "data": {
                "candle_granularity": "ONE_MINUTE",
                "candle_lookback_minutes": 120,
                "order_book_depth": 10
            },
            "logging": {
                "file": "logs/renaissance_bot.log",
                "level": "INFO"
            }
        }

        if not config_path.exists():
            return default_config

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                return {**default_config, **loaded}
        except Exception as exc:
            print(f"Warning: failed to load config at {config_path}: {exc}")
            return default_config

    def _validate_config(self, config: Dict[str, Any]):
        """Validate critical config values at startup."""
        errors = []

        risk = config.get("risk_management", {})
        dl = risk.get("daily_loss_limit", 500)
        if not (0 < dl <= 100000):
            errors.append(f"daily_loss_limit={dl} out of range (0, 100000]")
        pl = risk.get("position_limit", 1000)
        if not (0 < pl <= 1000000):
            errors.append(f"position_limit={pl} out of range (0, 1000000]")
        mc = risk.get("min_confidence", 0.65)
        if not (0.0 < mc <= 1.0):
            errors.append(f"min_confidence={mc} out of range (0, 1.0]")

        trading = config.get("trading", {})
        interval = trading.get("cycle_interval_seconds", 300)
        if not (10 <= interval <= 3600):
            errors.append(f"cycle_interval_seconds={interval} out of range [10, 3600]")

        # Auto-normalize signal weights
        weights = config.get("signal_weights", {})
        if weights:
            total = sum(float(v) for v in weights.values())
            if total > 0 and abs(total - 1.0) > 0.01:
                self.logger.warning(f"Signal weights sum to {total:.3f}, normalizing to 1.0")
                for k in weights:
                    weights[k] = float(weights[k]) / total

        if errors:
            for e in errors:
                self.logger.error(f"CONFIG ERROR: {e}")
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

    def _track_task(self, coro) -> asyncio.Task:
        """Create and track an asyncio task for graceful shutdown."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        return task

    @staticmethod
    def _compute_realized_pnl(entry_price: float, close_price: float,
                               size: float, side: str) -> float:
        """Compute realized PnL from entry/close prices and position side."""
        if entry_price <= 0 or close_price <= 0 or size <= 0:
            return 0.0
        side_upper = side.upper() if isinstance(side, str) else str(side).upper()
        if side_upper in ("LONG", "BUY"):
            return (close_price - entry_price) * size
        elif side_upper in ("SHORT", "SELL"):
            return (entry_price - close_price) * size
        return 0.0

    async def _shutdown(self):
        """Cancel background tasks and cleanup resources."""
        self.logger.info("Shutting down - cancelling background tasks...")
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        self.logger.info("Shutdown complete.")

    HEARTBEAT_FILE = Path("logs/heartbeat.json")

    def _write_heartbeat(self):
        """Write heartbeat file for external monitoring."""
        try:
            self.HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            heartbeat = {
                "alive": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle_count": len(self.decision_history),
                "killed": self._killed,
                "paper_mode": self.coinbase_client.paper_trading,
            }
            with open(self.HEARTBEAT_FILE, 'w') as f:
                json.dump(heartbeat, f)
        except Exception:
            pass

    async def collect_all_data(self, product_id: str = "BTC-USD") -> Dict[str, Any]:
        """Collect data from all sources for a specific product.

        Routes to Binance for expanded-universe pairs, Coinbase for legacy pairs.
        """
        # Use Binance as primary data source when universe is built
        # Also use Binance for breakout-flagged pairs not in the original universe
        _is_breakout_pair = product_id in getattr(self, '_breakout_scores', {})
        if (self._universe_built and product_id in self._pair_binance_symbols) or _is_breakout_pair:
            data = await self._collect_from_binance(product_id)
            if data:
                return data
            # Fall through to Coinbase if Binance fails

        try:
            # Try WebSocket data first (sub-100ms latency)
            # Drain the queue but only use data matching this product_id
            latest_ws = None
            requeue = []
            while not self._ws_queue.empty():
                try:
                    msg = self._ws_queue.get_nowait()
                    msg_pid = getattr(msg, 'product_id', None) or getattr(msg, 'symbol', None) or ''
                    if msg_pid == product_id:
                        latest_ws = msg
                    else:
                        requeue.append(msg)
                except queue.Empty:
                    break
            # Put back messages for other products
            for msg in requeue:
                try:
                    self._ws_queue.put_nowait(msg)
                except queue.Full:
                    pass

            # Check WebSocket data freshness
            MAX_DATA_AGE_SECONDS = 30
            if latest_ws and hasattr(latest_ws, 'timestamp') and latest_ws.timestamp:
                data_age = (datetime.now() - latest_ws.timestamp).total_seconds()
                if data_age > MAX_DATA_AGE_SECONDS:
                    self.logger.warning(f"WebSocket data stale ({data_age:.1f}s old), falling back to REST")
                    latest_ws = None

            if latest_ws and hasattr(latest_ws, 'price') and latest_ws.price > 0:
                # Use real-time WebSocket data
                ticker = {
                    'price': latest_ws.price,
                    'volume': latest_ws.volume,
                    'bid': latest_ws.bid,
                    'ask': latest_ws.ask,
                    'bid_ask_spread': latest_ws.spread,
                }
                order_book_snapshot = getattr(latest_ws, 'order_book', None)
                self.logger.debug(f"Using WebSocket data for {product_id} @ ${latest_ws.price:.2f}")
            else:
                ticker = None
                order_book_snapshot = None

            # Always fetch REST snapshot for candle/price history (needed for technicals)
            snapshot = await asyncio.to_thread(self.market_data_provider.fetch_snapshot, product_id)
            tech = self._get_tech(product_id)
            if snapshot.price_data:
                tech.update_price_data(snapshot.price_data)

            # Prefer WS ticker if available, otherwise use REST
            if ticker is None:
                ticker = snapshot.ticker
                order_book_snapshot = snapshot.order_book_snapshot

            technical_signals = tech.get_latest_signals()
            alt_signals = await self.alternative_data_engine.get_alternative_signals()

            return {
                'order_book_snapshot': order_book_snapshot or snapshot.order_book_snapshot,
                'price_data': snapshot.price_data,
                'technical_signals': technical_signals,
                'alternative_signals': alt_signals,
                'ticker': ticker or snapshot.ticker,
                'product_id': product_id,
                'timestamp': datetime.now(),
                'recent_trades': getattr(latest_ws, 'recent_trades', []) if latest_ws else [],
            }
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return {}

    async def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate signals from all components"""
        signals = {}

        try:
            # 1. Microstructure signals (Order Flow + Order Book = 53% total weight)
            # Feed recent trades into microstructure engine for flow/VPIN analysis
            try:
                _recent = market_data.get('recent_trades', []) or []
                if _recent:
                    from microstructure_engine import TradeData as MSTradeData
                    for t in _recent[-20:]:  # last 20 trades
                        if isinstance(t, dict) and t.get('price', 0) > 0:
                            self.microstructure_engine.update_trade(MSTradeData(
                                timestamp=datetime.now(),
                                price=float(t['price']),
                                size=float(t.get('size', 0)),
                                side=str(t.get('side', 'unknown')),
                                trade_id=str(t.get('trade_id', '')),
                            ))
            except Exception as _tf_err:
                self.logger.debug(f"Trade feed to microstructure failed: {_tf_err}")

            order_book_snapshot = market_data.get('order_book_snapshot')
            # Convert WebSocket dict format to OrderBookSnapshot if needed
            if order_book_snapshot and isinstance(order_book_snapshot, dict) and ('bids' in order_book_snapshot or 'asks' in order_book_snapshot):
                try:
                    current_px = float(market_data.get('current_price', 0) or market_data.get('ticker', {}).get('price', 0))
                    order_book_snapshot = _convert_ws_orderbook_to_snapshot(order_book_snapshot, current_px)
                    market_data['order_book_snapshot'] = order_book_snapshot  # update for downstream
                except Exception as _conv_err:
                    self.logger.debug(f"Order book conversion failed: {_conv_err}")
                    order_book_snapshot = None
            if order_book_snapshot:
                microstructure_signal = self.microstructure_engine.update_order_book(order_book_snapshot)
                signals['order_flow'] = microstructure_signal.large_trade_flow
                signals['order_book'] = microstructure_signal.order_book_imbalance
            else:
                signals['order_flow'] = 0.0
                signals['order_book'] = 0.0

            # 2. Technical indicators (38% total weight)
            _pid = market_data.get('product_id', 'BTC-USD')
            technical_signal = market_data.get('technical_signals') or self._get_tech(_pid).get_latest_signals()
            if technical_signal:
                signals['volume'] = _continuous_obv_signal(technical_signal.obv_momentum)
                signals['macd'] = _continuous_macd_signal(technical_signal.quick_macd)
                signals['rsi'] = _continuous_rsi_signal(technical_signal.fast_rsi)
                signals['bollinger'] = _continuous_bollinger_signal(technical_signal.dynamic_bollinger)
            else:
                signals['volume'] = 0.0
                signals['macd'] = 0.0
                signals['rsi'] = 0.0
                signals['bollinger'] = 0.0

                # 3. Alternative data signals (4.5% total weight)
            if market_data.get('alternative_signals'):
                alt_signal = market_data['alternative_signals']
                
                # Fetch whale signal
                whale_data = await self.whale_monitor.get_whale_signals()
                whale_pressure = float(whale_data.get("whale_pressure", 0.0))
                
                # 🧠 Deep NLP Reasoning (New)
                news_text = " ".join([str(n) for n in market_data.get('news', [])])
                if news_text.strip():
                    nlp_result = await self.nlp_bridge.analyze_sentiment_with_reasoning(news_text)
                    nlp_sentiment = float(nlp_result.get('sentiment', 0.0))
                    market_data['nlp_reasoning'] = nlp_result.get('reasoning', 'No deep context')
                else:
                    nlp_sentiment = 0.0
                    market_data['nlp_reasoning'] = 'No news data available'
                
                # Combine all alternative signals into one composite score
                # 20% Reddit, 15% News, 15% Twitter, 15% Fear/Greed, 10% Whale, 25% Deep NLP
                alternative_composite = (
                    float(alt_signal.reddit_sentiment or 0.0) * 0.20 +
                    float(alt_signal.news_sentiment or 0.0) * 0.15 +
                    float(alt_signal.social_sentiment or 0.0) * 0.15 +
                    float(alt_signal.market_psychology or 0.0) * 0.15 +
                    whale_pressure * 0.10 +
                    nlp_sentiment * 0.25
                )
                signals['alternative'] = float(alternative_composite)
                market_data['whale_signals'] = whale_data # Pass along for dashboard
            else:
                signals['alternative'] = 0.0

            self.logger.info(f"Generated signals: {signals}")

            # 4. Institutional & High-Dimensional Intelligence (Step 16+)
            # Each signal group is isolated so one failure doesn't kill the rest
            p_id = market_data.get('product_id', 'Unknown')
            df = self._get_tech(p_id)._to_dataframe()
            cur_price = market_data.get('current_price', 0.0)

            # Volume Profile
            try:
                if not df.empty and cur_price > 0:
                    profile = self.volume_profile_engine.calculate_profile(df)
                    if profile:
                        vp_signal = self.volume_profile_engine.get_profile_signal(cur_price, profile)
                        signals['volume_profile'] = vp_signal['signal']
                        self._last_vp_status[p_id] = vp_signal['status']
            except Exception as e:
                self.logger.debug(f"Volume profile signal failed: {e}")

            # Statistical Arbitrage (BTC vs ETH)
            try:
                if p_id in ["BTC-USD", "ETH-USD"]:
                    other_id = "ETH-USD" if p_id == "BTC-USD" else "BTC-USD"
                    sa_signal = self.stat_arb_engine.calculate_pair_signal(p_id, other_id)
                    signals['stat_arb'] = sa_signal.get('signal', 0.0)
            except Exception as e:
                self.logger.debug(f"Stat arb signal failed: {e}")

            # Cross-Asset Lead-Lag Alpha (Step 16)
            try:
                if len(self.product_ids) > 1 and cur_price > 0:
                    self.correlation_engine.update_price(p_id, cur_price)
                    base = self.product_ids[0]
                    target = p_id
                    if base != target:
                        ll_data = self.correlation_engine.calculate_lead_lag(base, target)
                        signals['lead_lag'] = ll_data.get('directional_signal', 0.0)
                        market_data['lead_lag_alpha'] = ll_data
            except Exception as e:
                self.logger.debug(f"Lead-lag signal failed: {e}")

            # Fractal Intelligence (DTW)
            try:
                if not df.empty and len(df) >= 10:
                    prices = df['close'].values
                    fractal_result = self.fractal_intelligence.find_best_match(prices)
                    signals['fractal'] = fractal_result['signal']
                    market_data['fractal_intelligence'] = fractal_result
            except Exception as e:
                self.logger.debug(f"Fractal signal failed: {e}")

            # Market Entropy (Shannon/ApEn)
            try:
                if not df.empty and len(df) >= 20:
                    prices = df['close'].values
                    entropy_result = self.market_entropy.calculate_entropy(prices)
                    signals['entropy'] = 0.5 * (entropy_result['predictability'] - 0.5)
                    market_data['market_entropy'] = entropy_result
            except Exception as e:
                self.logger.debug(f"Entropy signal failed: {e}")

            # ML Feature Pipeline & Real-Time Intelligence (Step 12/16 Bridge)
            try:
                if not df.empty and len(df) >= 18 and self.real_time_pipeline.enabled:
                    _cross = market_data.get('_cross_data')
                    _pair = market_data.get('_pair_name')
                    _deriv = market_data.get('_derivatives_data')
                    rt_result = await self.real_time_pipeline.processor.process_all_models(
                        {'price_df': df},
                        cross_data=_cross, pair_name=_pair,
                        derivatives_data=_deriv,
                    )
                    market_data['real_time_predictions'] = rt_result
                    _ml_scale = self.config.get("ml_signal_scale", 10.0)
                    # MetaEnsemble is the key from real_time_pipeline name_map
                    _ens_val = rt_result.get('MetaEnsemble') or rt_result.get('Ensemble') or 0.0
                    if _ens_val:
                        signals['ml_ensemble'] = float(np.clip(_ens_val * _ml_scale, -1.0, 1.0))
                    if 'CNN' in rt_result:
                        signals['ml_cnn'] = float(np.clip(rt_result['CNN'] * _ml_scale, -1.0, 1.0))
                    self.logger.info(
                        f"ML SIGNALS: ensemble={signals.get('ml_ensemble', 0):.4f}, "
                        f"cnn={signals.get('ml_cnn', 0):.4f} (raw: E={_ens_val:.4f}, "
                        f"C={rt_result.get('CNN', 0):.4f}, scale={_ml_scale})"
                    )
            except Exception as e:
                self.logger.warning(f"ML RT pipeline failed: {e}")

            # Quantum Oscillator (QHO)
            try:
                if not df.empty and len(df) >= 30:
                    prices = df['close'].values
                    quantum_result = self.quantum_oscillator.calculate_quantum_levels(prices)
                    signals['quantum'] = quantum_result['signal']
                    market_data['quantum_oscillator'] = quantum_result
            except Exception as e:
                self.logger.debug(f"Quantum signal failed: {e}")

            # Correlation Network Divergence Signal
            try:
                if self.correlation_network.enabled:
                    div_signal = self.correlation_network.get_correlation_divergence_signal(p_id)
                    signals['correlation_divergence'] = div_signal
            except Exception as e:
                self.logger.debug(f"Correlation divergence signal failed: {e}")

            # GARCH Volatility Signal (vol_ratio as directional bias)
            try:
                if self.garch_engine.is_available:
                    forecast = self.garch_engine.forecast_volatility(p_id)
                    vol_ratio = forecast.get('vol_ratio', 1.0)
                    if vol_ratio < 0.8:
                        signals['garch_vol'] = min((1.0 - vol_ratio) * 0.5, 0.5)
                    elif vol_ratio > 1.2:
                        signals['garch_vol'] = max(-(vol_ratio - 1.0) * 0.5, -0.5)
                    else:
                        signals['garch_vol'] = 0.0
                    market_data['garch_forecast'] = forecast
            except Exception as e:
                self.logger.debug(f"GARCH vol signal failed: {e}")

            return signals

        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return {key: 0.0 for key in self.signal_weights.keys()}

    def calculate_weighted_signal(self, signals: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Calculate final weighted signal using Renaissance weights (Institutional hardening)"""
        
        # We redirect to the new ML-enhanced fusion if possible, or use standard
        # For backward compatibility with tests/backtests that call this directly
        ml_package = signals.get('ml_package') # Might be injected in some contexts
        
        # 🛡️ PURE SCALAR TYPE GUARD for all signal inputs
        processed_signals = {}
        for k, v in signals.items():
            if k == 'ml_package':
                processed_signals[k] = v
                continue
            try:
                processed_signals[k] = self._force_float(v)
            except Exception:
                processed_signals[k] = 0.0
        
        weighted_signal, confidence, fusion_metadata = self.signal_fusion.fuse_signals_with_ml(
            processed_signals, self.signal_weights, ml_package
        )
        
        # Ensure contributions are also hardened
        contributions = fusion_metadata.get('contributions', {})
        hardened_contribs = {}
        for k, v in contributions.items():
            try:
                hardened_contribs[k] = self._force_float(v)
            except Exception:
                hardened_contribs[k] = 0.0

        return float(self._force_float(weighted_signal)), hardened_contribs

    def _calculate_dynamic_position_size(self, product_id: str, confidence: float, weighted_signal: float, current_price: float) -> float:
        """Calculate dynamic position size using Step 10 Portfolio Optimizer"""
        try:
            # Prepare minimal data for optimizer
            # For a single asset, we optimize between cash and the asset
            universe_data = {
                'returns': np.array([weighted_signal * 0.01]), # Expected return based on signal
                'market_cap': np.array([1.0]),
                'assets': [product_id]
            }
            
            market_data = {
                'bid_ask_spread': np.array([0.0005]),
                'market_impact': np.array([0.0002])
            }
            
            opt_result = self.portfolio_optimizer.optimize_portfolio(universe_data, market_data)
            
            if 'weights' in opt_result:
                # Weight for the asset (index 0)
                optimized_weight = float(opt_result['weights'][0])
                # Scale by confidence
                final_size = optimized_weight * confidence
                return float(np.clip(final_size, 0.0, 0.3)) # Cap at 30%
            
            # Fallback to standard sizing
            return min(confidence * 0.5, 0.3)
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization sizing failed: {e}")
            return min(confidence * 0.5, 0.3)

    def make_trading_decision(self, weighted_signal: float, signal_contributions: Dict[str, float],
                              current_price: float = 0.0, real_time_result: Optional[Dict[str, Any]] = None,
                              product_id: str = "BTC-USD", ml_package: Optional[MLSignalPackage] = None,
                              market_data: Optional[Dict[str, Any]] = None,
                              drawdown_pct: float = 0.0) -> TradingDecision:
        """Make final trading decision with Renaissance methodology + Kelly position sizing"""

        # ── COST PRE-SCREEN: "The edge must exceed the vig" — Medallion Principle ──
        # In paper trading mode, use much lower cost threshold since fees are simulated.
        # The full cost pre-screen runs later in live mode but should not block paper signals.
        try:
            if self.paper_trading:
                # Paper mode: use minimal cost threshold to avoid blocking valid signals
                min_viable_signal = 0.001
            else:
                round_trip_cost = self.position_sizer.estimate_round_trip_cost()
                min_viable_signal = round_trip_cost * 1.0
            if abs(weighted_signal) < min_viable_signal:
                self.logger.debug(
                    f"COST PRE-SCREEN: {product_id} signal {weighted_signal:.4f} < "
                    f"min viable {min_viable_signal:.4f}"
                )
                return TradingDecision(
                    action='HOLD', confidence=0.0, position_size=0.0,
                    reasoning={'blocked_by': 'cost_pre_screen',
                               'signal': weighted_signal,
                               'min_viable': min_viable_signal},
                    timestamp=datetime.now()
                )
        except Exception:
            pass  # Don't let cost pre-screen crash the decision pipeline

        # Calculate confidence based on signal strength and directional consensus
        # signal_strength: rescale so that a strong signal (0.02) maps to 1.0
        # Typical weighted signals are 0.001-0.03; old denominator 0.05 wasted
        # the top 60% of the confidence range.
        signal_strength = min(abs(weighted_signal) / 0.02, 1.0)

        # Directional consensus: fraction of non-trivial signals agreeing on direction
        raw_contribs = [v for v in signal_contributions.values() if abs(v) > 0.0001]
        if raw_contribs and weighted_signal != 0:
            agreeing = sum(1 for v in raw_contribs if np.sign(v) == np.sign(weighted_signal))
            signal_consensus = agreeing / len(raw_contribs)
        else:
            signal_consensus = 0.5

        # Geometric mean: both strength AND consensus must be present
        confidence = float(np.sqrt(signal_strength * signal_consensus))

        # Apply regime-derived confidence boost (max +/-5%)
        confidence = float(np.clip(confidence + self.regime_overlay.get_confidence_boost(), 0.0, 1.0))

        # ML Enhanced Confidence
        if ml_package:
            direction_match = np.sign(weighted_signal) == np.sign(ml_package.ensemble_score)
            overlay = 0.05 if direction_match else -0.05
            consciousness_factor = ml_package.confidence_score
            confidence = float(np.clip(confidence + (overlay * consciousness_factor), 0.0, 1.0))
            self.logger.info(f"ML confidence adjustment: {(overlay * consciousness_factor):+.4f} (Consciousness: {consciousness_factor:.2f})")

        # ── Regime-biased entry thresholds ──
        # In bearish regimes, require higher conviction for longs (and vice versa).
        # Trading WITH the regime: lower bar. AGAINST: higher bar.
        _regime_label = None
        try:
            if self.regime_overlay.enabled:
                _regime_label = self.regime_overlay.get_hmm_regime_label()
        except Exception:
            pass

        _BEARISH = {'bear_trending', 'bear_mean_reverting', 'high_volatility'}
        _BULLISH = {'bull_trending', 'bull_mean_reverting'}

        # Start with base thresholds
        _pred_thresh = abs(self.buy_threshold)  # Usually 0.06
        _agree_thresh = 0.71

        # Determine action direction
        if confidence < self.min_confidence:
            action = 'HOLD'
        elif weighted_signal > self.buy_threshold:
            action = 'BUY'
        elif weighted_signal < self.sell_threshold:
            action = 'SELL'
        else:
            action = 'HOLD'

        # Apply regime bias to thresholds based on trade direction
        if action != 'HOLD' and _regime_label and _regime_label not in ('neutral_sideways', 'unknown', 'low_volatility'):
            _is_bearish = _regime_label in _BEARISH
            _is_bullish = _regime_label in _BULLISH
            _counter_trend = (_is_bearish and action == 'BUY') or (_is_bullish and action == 'SELL')
            _with_trend = (_is_bearish and action == 'SELL') or (_is_bullish and action == 'BUY')

            if _counter_trend:
                # Swimming upstream — require much higher conviction
                _pred_thresh = 0.10
                _agree_thresh = 0.80
                # Re-check prediction threshold with raised bar
                if abs(weighted_signal) < _pred_thresh:
                    self.logger.info(
                        f"REGIME FILTER: {product_id} {action} in {_regime_label} — "
                        f"|signal|={abs(weighted_signal):.4f} < {_pred_thresh} (counter-trend blocked)"
                    )
                    action = 'HOLD'
                else:
                    self.logger.info(
                        f"REGIME FILTER: {product_id} {action} in {_regime_label} — "
                        f"raised thresholds to pred>{_pred_thresh} agree>{_agree_thresh}"
                    )
            elif _with_trend:
                # Trading with regime — lower the bar
                _pred_thresh = 0.05
                _agree_thresh = 0.65
                self.logger.info(
                    f"REGIME BOOST: {product_id} {action} in {_regime_label} — "
                    f"lowered thresholds to pred>{_pred_thresh} agree>{_agree_thresh}"
                )

        # ── Signal filter stats tracking ──
        self._signal_filter_stats['total'] += 1
        if action == 'HOLD' and abs(weighted_signal) > 0.001:
            self._signal_filter_stats['filtered_threshold'] += 1

        # ── ML Agreement Gate: only trade when models agree strongly ──
        # Threshold is regime-adjusted: 0.65 with-trend, 0.71 neutral, 0.80 counter-trend
        if action != 'HOLD' and ml_package and ml_package.ml_predictions:
            pred_values = []
            for mp in ml_package.ml_predictions:
                if isinstance(mp, dict):
                    v = mp.get('prediction', 0.0)
                    if isinstance(v, (int, float)):
                        pred_values.append(float(v))
                elif isinstance(mp, (tuple, list)) and len(mp) >= 2:
                    v = mp[1]
                    if isinstance(v, (int, float)):
                        pred_values.append(float(v))
            if len(pred_values) >= 3:
                signs = [1 if p > 0 else (-1 if p < 0 else 0) for p in pred_values]
                nonzero_signs = [s for s in signs if s != 0]
                if nonzero_signs:
                    agreement = max(nonzero_signs.count(1), nonzero_signs.count(-1)) / len(nonzero_signs)
                    if agreement < _agree_thresh:
                        self.logger.info(
                            f"ML AGREEMENT GATE: {product_id} blocked — "
                            f"only {agreement:.0%} model agreement (need >{_agree_thresh:.0%})"
                        )
                        self._signal_filter_stats['filtered_agreement'] += 1
                        action = 'HOLD'

        # Track traded signals
        if action != 'HOLD':
            self._signal_filter_stats['traded'] += 1

        # Log filter stats every 20 cycles
        cycle_num = getattr(self, 'scan_cycle_count', 0)
        if cycle_num > 0 and cycle_num % 20 == 0 and self._signal_filter_stats['total'] > 0:
            stats = self._signal_filter_stats
            self.logger.info(
                f"SIGNAL FILTER STATS: {stats['traded']}/{stats['total']} traded "
                f"({100*stats['traded']/max(stats['total'],1):.0f}%), "
                f"filtered: threshold={stats['filtered_threshold']}, "
                f"confidence={stats['filtered_confidence']}, "
                f"agreement={stats['filtered_agreement']}"
            )

        # ── Anti-Churn Gate (Renaissance: conviction before action) ──
        if not hasattr(self, '_signal_history'):
            self._signal_history = {}  # product_id -> list of recent actions
            self._last_trade_cycle = {}  # product_id -> cycle number when last traded

        # Track signal direction history per asset
        hist = self._signal_history.setdefault(product_id, [])
        hist.append(action)
        if len(hist) > 10:
            hist.pop(0)

        if action != 'HOLD':
            cycle_num = getattr(self, 'scan_cycle_count', 0)
            last_trade = self._last_trade_cycle.get(product_id, -999)
            min_hold_cycles = 12  # Must wait 12 cycles (~60min) between trades on same asset

            # 1. Minimum hold period — don't trade if we just traded
            if cycle_num - last_trade < min_hold_cycles:
                self.logger.info(
                    f"ANTI-CHURN: {product_id} cooldown — {cycle_num - last_trade}/{min_hold_cycles} cycles since last trade"
                )
                action = 'HOLD'

            # 2. Signal persistence — require 2 consecutive signals in same direction
            elif len(hist) >= 2 and hist[-2] != action and hist[-2] != 'HOLD':
                self.logger.info(
                    f"ANTI-CHURN: {product_id} signal flip ({hist[-2]} -> {action}) — waiting for persistence"
                )
                action = 'HOLD'

            # 3. Signal reversal on open position — close the existing position
            # instead of blocking the signal (the old behavior trapped losing positions)
            if action != 'HOLD':
                try:
                    with self.position_manager._lock:
                        matching_positions = [
                            pos for pos in self.position_manager.positions.values()
                            if pos.product_id == product_id and pos.status == PositionStatus.OPEN
                        ]
                    for pos in matching_positions:
                        pos_side = pos.side.value.upper()
                        if (pos_side == 'LONG' and action == 'SELL') or \
                           (pos_side == 'SHORT' and action == 'BUY'):
                            # Close the existing position — the signal is telling us to exit
                            close_ok, close_msg = self.position_manager.close_position(
                                pos.position_id, reason=f"Signal reversal: {pos_side} -> {action}"
                            )
                            if close_ok:
                                _cpx = current_price
                                _rpnl = self._compute_realized_pnl(
                                    pos.entry_price, _cpx, pos.size, pos_side
                                )
                                self._track_task(
                                    self.db_manager.close_position_record(
                                        pos.position_id,
                                        close_price=float(_cpx),
                                        realized_pnl=float(_rpnl),
                                        exit_reason="signal_reversal",
                                    )
                                )
                            self.logger.info(
                                f"SIGNAL REVERSAL: {product_id} closed {pos_side} position — {close_msg}"
                            )
                            # Don't also open a new opposing position — just exit
                            action = 'HOLD'
                            break
                except Exception as e:
                    self.logger.error(f"NETTING CHECK FAILED for {product_id}: {e} — blocking trade for safety")
                    action = 'HOLD'

            # 4. Already positioned — don't stack same-direction positions
            if action != 'HOLD':
                try:
                    with self.position_manager._lock:
                        same_dir = [
                            pos for pos in self.position_manager.positions.values()
                            if pos.product_id == product_id
                            and pos.status == PositionStatus.OPEN
                            and (
                                (pos.side.value.upper() == 'LONG' and action == 'BUY') or
                                (pos.side.value.upper() == 'SHORT' and action == 'SELL')
                            )
                        ]
                    if same_dir:
                        self.logger.info(
                            f"ALREADY POSITIONED: {product_id} already has {len(same_dir)} "
                            f"{same_dir[0].side.value} position(s) — holding"
                        )
                        action = 'HOLD'
                except Exception as e:
                    self.logger.error(f"ANTI-STACK CHECK FAILED for {product_id}: {e} — blocking trade for safety")
                    action = 'HOLD'

        # ML-Enhanced Risk Assessment (Regime Gate)
        risk_assessment = self.risk_manager.assess_risk_regime(ml_package)
        if risk_assessment['recommended_action'] == 'fallback_mode':
            self.logger.warning("ML Risk assessment triggered FALLBACK MODE - halting trades")
            action = 'HOLD'

        # Daily loss limit check
        if abs(self.daily_pnl) >= self.daily_loss_limit:
            action = 'HOLD'
            self.logger.warning(f"Daily loss limit reached: ${self.daily_pnl}")

        # Gate through VAE Anomaly Detection
        vae_loss = 0.0
        gate_reason = "not_evaluated"
        feature_vector = ml_package.feature_vector if ml_package else None

        # Always compute VAE loss for monitoring (even on HOLD)
        if feature_vector is not None and self.risk_gateway.vae_trained and self.risk_gateway.vae is not None:
            try:
                _, vae_loss = self.risk_gateway._check_anomaly(feature_vector)
            except Exception:
                pass

        if action != 'HOLD':
            portfolio_data = {
                'total_value': self.position_limit,
                'daily_pnl': self.daily_pnl,
                'positions': {'BTC': self.current_position},
                'current_price': current_price
            }
            is_allowed, vae_loss, gate_reason = self.risk_gateway.assess_trade(
                action=action,
                amount=0,  # We don't know size yet; gate on action only
                current_price=current_price,
                portfolio_data=portfolio_data,
                feature_vector=feature_vector
            )
            if not is_allowed:
                self.logger.warning(f"Risk Gateway BLOCKED {action} order (reason={gate_reason}, vae_loss={vae_loss:.4f})")
                action = 'HOLD'

        # ── Renaissance Position Sizing (Kelly + cost gate + vol normalization) ──
        position_size = 0.0
        sizing_result = None
        if action != 'HOLD':
            # Gather volatility data
            mkt = market_data or {}
            garch_forecast = mkt.get('garch_forecast', {})
            volatility = garch_forecast.get('forecast_vol', None)
            vol_regime = garch_forecast.get('vol_regime', None)

            # Gather regime data
            fractal_regime = None
            if ml_package:
                fractal_regime = ml_package.fractal_insights.get('regime_detection', None)

            # Current exposure from position manager
            current_exposure = self.position_manager._calculate_total_exposure()

            # Order book depth for liquidity constraint
            order_book_depth = None
            ob = mkt.get('order_book_snapshot')
            if ob:
                try:
                    if hasattr(ob, 'bids'):
                        # OrderBookSnapshot dataclass
                        bid_depth = sum(lv.price * lv.size for lv in ob.bids[:10]) if ob.bids else 0
                        ask_depth = sum(lv.price * lv.size for lv in ob.asks[:10]) if ob.asks else 0
                    else:
                        # Dict fallback
                        bids = ob.get('bids', [])
                        asks = ob.get('asks', [])
                        bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:10]) if bids else 0
                        ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:10]) if asks else 0
                    order_book_depth = bid_depth + ask_depth
                except Exception:
                    pass

            # Extract daily volume from market data if available
            daily_volume_usd = None
            try:
                ticker = mkt.get('ticker', {})
                vol_24h = ticker.get('volume_24h') or ticker.get('volume')
                if vol_24h and current_price > 0:
                    daily_volume_usd = float(vol_24h) * current_price
            except Exception:
                pass

            # Use measured edge from signal scorecard when available
            _measured_edge = self._get_measured_edge(product_id)

            # ── Signal Confidence Tier — size by conviction level ──
            # Tier 1 (Proven >55%, 100+ samples): full size (1.0x)
            # Tier 2 (50-55% or <100 samples): half size (0.5x)
            # Tier 3 (<50%): quarter size (0.25x)
            _tier_multiplier = 1.0
            # Track individual multipliers for sizing chain log (Audit 3)
            _chain = {"regime": 1.0, "corr": 1.0, "health": 1.0, "tier": 1.0}

            # ── Regime Transition Warning: reduce size on adverse transition risk ──
            if self.regime_overlay.enabled:
                try:
                    transition = self.regime_overlay.get_transition_warning()
                    if transition["alert_level"] != "none":
                        _chain["regime"] = transition["size_multiplier"]
                        _tier_multiplier *= transition["size_multiplier"]
                        self.logger.info(f"REGIME TRANSITION: {transition['message']}")
                except Exception:
                    pass

            # ── Portfolio Engine: correlation-aware sizing ──
            if self.portfolio_engine and action != 'HOLD':
                try:
                    # Get current positions exposure
                    current_positions = {}
                    with self.position_manager._lock:
                        for pos in self.position_manager.positions.values():
                            pid = pos.product_id
                            current_positions[pid] = current_positions.get(pid, 0.0) + (pos.size * pos.entry_price)
                    # Current product signal
                    product_signals = {product_id: (weighted_signal, confidence)}
                    # Add other products' last known signals (approximate)
                    for pid in self.product_ids:
                        if pid != product_id and pid not in product_signals:
                            product_signals[pid] = (0.0, 0.0)  # neutral for unprocessed

                    port_result = self.portfolio_engine.optimize(
                        product_signals, current_positions,
                        self._cached_balance_usd or 10000.0,
                        cycle_count=getattr(self, 'scan_cycle_count', 0),
                    )
                    port_adj = port_result.get(product_id, {})
                    port_mult = port_adj.get("size_multiplier", 1.0)
                    _chain["corr"] = port_mult
                    if port_mult < 1.0:
                        _tier_multiplier *= port_mult
                        self.logger.info(f"PORTFOLIO ENGINE: {product_id} sized to {port_mult:.0%} — {port_adj.get('reason', '')}")
                except Exception:
                    pass

            # ── Health Monitor: apply rolling Sharpe-based size scaling ──
            if self.health_monitor:
                health_mult = self.health_monitor.get_size_multiplier()
                _chain["health"] = health_mult
                if health_mult < 1.0:
                    self.logger.info(f"HEALTH MONITOR: Sizing at {health_mult:.0%} (Sharpe-based)")
                _tier_multiplier *= health_mult
                if self.health_monitor.is_exits_only():
                    action = 'HOLD'
                    self.logger.warning("HEALTH MONITOR: EXITS-ONLY mode — blocking new entries")
            sc = self._signal_scorecard.get(product_id, {})
            if sc:
                # Find the dominant signal contributors
                top_signals = sorted(
                    [(k, v) for k, v in signal_contributions.items() if abs(v) > 0.01],
                    key=lambda x: abs(x[1]), reverse=True
                )[:3]  # Top 3 contributors
                tier_scores = []
                for sig_name, _ in top_signals:
                    stats = sc.get(sig_name, {})
                    total = stats.get('total', 0)
                    correct = stats.get('correct', 0)
                    if total >= 100 and correct / total > 0.55:
                        tier_scores.append(1.0)    # Tier 1
                    elif total >= 100 and correct / total >= 0.50:
                        tier_scores.append(0.5)    # Tier 2
                    elif total < 100:
                        tier_scores.append(0.5)    # Tier 2 (insufficient data)
                    else:
                        tier_scores.append(0.25)   # Tier 3
                if tier_scores:
                    _tier_multiplier = sum(tier_scores) / len(tier_scores)

            sizing_result = self.position_sizer.calculate_size(
                signal_strength=weighted_signal,
                confidence=confidence,
                current_price=current_price,
                product_id=product_id,
                volatility=volatility,
                vol_regime=vol_regime,
                fractal_regime=fractal_regime,
                order_book_depth_usd=order_book_depth,
                current_exposure_usd=current_exposure,
                ml_package=ml_package,
                account_balance_usd=self._cached_balance_usd or None,
                daily_volume_usd=daily_volume_usd,
                drawdown_pct=drawdown_pct,
                measured_edge=_measured_edge,
                tier_size_multiplier=_tier_multiplier,
            )
            position_size = sizing_result.asset_units

            if position_size <= 0:
                action = 'HOLD'
                self.logger.info(f"Position sizer returned 0: {sizing_result.reasons[-1] if sizing_result.reasons else 'no edge'}")
            else:
                self.logger.info(
                    f"POSITION SIZED: {action} {position_size:.8f} {product_id} "
                    f"(${sizing_result.usd_value:.2f}) | "
                    f"Kelly={sizing_result.kelly_fraction:.4f} -> {sizing_result.applied_fraction:.4f} | "
                    f"Edge={sizing_result.edge:.4f} EffEdge={sizing_result.effective_edge:.4f} "
                    f"P(w)={sizing_result.win_probability:.3f} | "
                    f"Impact={sizing_result.market_impact_bps:.1f}bps "
                    f"Capacity={sizing_result.capacity_used_pct:.1f}% | "
                    f"CostRatio={sizing_result.transaction_cost_ratio:.2f} "
                    f"VolScalar={sizing_result.volatility_scalar:.2f} "
                    f"RegimeScalar={sizing_result.regime_scalar:.2f}"
                )

            # ── SIZING CHAIN SUMMARY (Audit 3) ──
            _chain["tier"] = _tier_multiplier
            kelly_f = sizing_result.kelly_fraction if sizing_result else 0.0
            final_usd = sizing_result.usd_value if sizing_result else 0.0
            self.logger.info(
                f"SIZING CHAIN {product_id}: "
                f"regime={_chain['regime']:.2f} x corr={_chain['corr']:.2f} x "
                f"health={_chain['health']:.2f} x tier={_chain['tier']:.2f} x "
                f"kelly={kelly_f:.4f} -> final=${final_usd:.2f}"
            )

            # ── Kelly Sizer ACTIVE — adjust position size via Kelly optimal sizing ──
            if self.kelly_sizer and position_size > 0 and current_price > 0:
                try:
                    dominant_sig = max(signal_contributions, key=lambda k: abs(signal_contributions[k]), default="combined")
                    kelly_usd = self.kelly_sizer.get_position_size(
                        signal_dict={"signal_type": dominant_sig, "pair": product_id, "confidence": confidence},
                        equity=self._cached_balance_usd or 10000.0,
                    )
                    if kelly_usd > 0:
                        base_usd = position_size * current_price
                        # Blend: use minimum of existing size and Kelly recommendation
                        # This prevents over-sizing beyond what Kelly says is optimal
                        kelly_capped_usd = min(base_usd, kelly_usd)
                        kelly_ratio = kelly_capped_usd / base_usd if base_usd > 0 else 1.0
                        if kelly_ratio < 0.95:  # Only adjust if Kelly says significantly less
                            position_size = kelly_capped_usd / current_price
                            self.logger.info(
                                f"KELLY SIZER: {product_id} sized to {kelly_ratio:.0%} of base "
                                f"(Kelly=${kelly_usd:.2f}, base=${base_usd:.2f}, final=${kelly_capped_usd:.2f})"
                            )
                        else:
                            self.logger.info(
                                f"KELLY SIZER: {product_id} Kelly=${kelly_usd:.2f} >= base=${base_usd:.2f} — no reduction"
                            )
                    elif kelly_usd == 0:
                        # Kelly says don't trade — but only if we have sufficient data
                        kelly_stats = self.kelly_sizer.get_statistics(dominant_sig, product_id)
                        if kelly_stats.get("sufficient_data") and kelly_stats.get("expectancy_per_trade_bps", 0) <= 0:
                            self.logger.warning(
                                f"KELLY SIZER: {product_id} negative expectancy — blocking trade"
                            )
                            position_size = 0.0
                            action = 'HOLD'
                except Exception as _kelly_err:
                    self.logger.debug(f"Kelly sizer failed: {_kelly_err}")

            # ── Leverage Manager ACTIVE — apply consistency-based leverage multiplier ──
            if self.leverage_mgr and position_size > 0:
                try:
                    max_safe_lev = self.leverage_mgr.compute_max_safe_leverage()
                    if max_safe_lev > 0 and max_safe_lev < 1.0:
                        # Reduce size if leverage headroom is limited
                        position_size *= max_safe_lev
                        self.logger.info(
                            f"LEVERAGE MGR: {product_id} sized to {max_safe_lev:.0%} "
                            f"(consistency-based leverage cap)"
                        )
                    elif max_safe_lev >= 1.0:
                        self.logger.debug(
                            f"LEVERAGE MGR: {product_id} leverage headroom OK ({max_safe_lev:.2f}x)"
                        )
                    # If max_safe_lev == 0, block trade
                    if max_safe_lev == 0 and self.leverage_mgr.should_reduce_leverage():
                        self.logger.warning(
                            f"LEVERAGE MGR: {product_id} no leverage headroom — blocking"
                        )
                        position_size = 0.0
                        action = 'HOLD'
                except Exception as _lev_err:
                    self.logger.debug(f"Leverage manager failed: {_lev_err}")

            # ── Medallion Regime observation (Audit 1/2: log alongside RegimeOverlay) ──
            if self.medallion_regime:
                try:
                    med_regime = self.medallion_regime.predict_current_regime()
                    overlay_regime = self.regime_overlay.get_hmm_regime_label() if self.regime_overlay.enabled else "unknown"
                    self.logger.info(
                        f"REGIME COMPARE (obs) {product_id}: "
                        f"overlay={overlay_regime} vs medallion={med_regime.get('regime_name', 'unknown')} "
                        f"(conf={med_regime.get('confidence', 0):.2f})"
                    )
                except Exception:
                    pass

            # ── FINAL SIZE NORMALIZATION — predictable sizing from signal quality ──
            if action != 'HOLD' and position_size > 0 and current_price > 0:
                balance = self._cached_balance_usd or 50000.0
                base_usd = balance * 0.03                                 # 3% of equity
                sig_scalar = min(abs(weighted_signal) / 0.02, 2.0)        # signal quality
                sig_scalar = max(sig_scalar, 0.5)
                conf_scalar = max(0.5, min((confidence - 0.3) * 3.0, 2.0))  # confidence quality
                dd_scalar = getattr(self, '_drawdown_size_scalar', 1.0)
                normalized_usd = base_usd * sig_scalar * conf_scalar * dd_scalar
                normalized_usd = max(100.0, min(normalized_usd, balance * 0.099))  # $100 floor, 9.9% ceiling (0.1% buffer for rounding)
                normalized_size = normalized_usd / current_price
                original_usd = position_size * current_price
                if abs(normalized_usd - original_usd) > 10:
                    self.logger.info(
                        f"SIZE NORMALIZATION: {product_id} "
                        f"chain=${original_usd:.0f} → normalized=${normalized_usd:.0f} "
                        f"(sig={sig_scalar:.2f}x, conf={conf_scalar:.2f}x, dd={dd_scalar:.2f}x)"
                    )
                position_size = normalized_size

        reasoning = {
            'weighted_signal': weighted_signal,
            'confidence': confidence,
            'signal_contributions': signal_contributions,
            'current_price': current_price,
            'ml_risk_assessment': risk_assessment,
            'vae_loss': vae_loss,
            'risk_gateway_reason': gate_reason,
            'risk_check': {
                'daily_pnl': self.daily_pnl,
                'daily_limit': self.daily_loss_limit,
                'position_limit': self.position_limit
            },
        }
        if sizing_result:
            reasoning['position_sizing'] = {
                'method': sizing_result.sizing_method,
                'kelly_fraction': sizing_result.kelly_fraction,
                'applied_fraction': sizing_result.applied_fraction,
                'edge': sizing_result.edge,
                'effective_edge': sizing_result.effective_edge,
                'market_impact_bps': sizing_result.market_impact_bps,
                'capacity_used_pct': sizing_result.capacity_used_pct,
                'win_probability': sizing_result.win_probability,
                'cost_ratio': sizing_result.transaction_cost_ratio,
                'vol_scalar': sizing_result.volatility_scalar,
                'regime_scalar': sizing_result.regime_scalar,
                'liquidity_scalar': sizing_result.liquidity_scalar,
                'usd_value': sizing_result.usd_value,
                'reasons': sizing_result.reasons,
            }

        decision = TradingDecision(
            action=action,
            confidence=confidence,
            position_size=position_size,
            reasoning=reasoning,
            timestamp=datetime.now()
        )

        return decision

    async def _execute_smart_order(self, decision: TradingDecision, market_data: Dict[str, Any]):
        """Execute order through position manager (real or paper) with slippage analysis.

        Routes through MEXC (0% maker) for Binance-sourced pairs, Coinbase for legacy.
        """
        try:
            product_id = market_data.get('product_id', 'BTC-USD')
            current_price = decision.reasoning.get('current_price', 0.0)

            # Determine execution venue
            is_mexc_execution = (
                self._universe_built
                and product_id in self._pair_binance_symbols
            )

            # For MEXC execution: use limit order at best bid/ask for 0% maker fee
            if is_mexc_execution:
                ticker = market_data.get('ticker', {})
                if decision.action == 'BUY':
                    limit_price = float(ticker.get('bid', current_price))
                    limit_price *= 1.0001  # Tiny premium for fill probability
                else:
                    limit_price = float(ticker.get('ask', current_price))
                    limit_price *= 0.9999  # Tiny discount
                # In paper mode, limit_price ≈ current_price (negligible difference)
                current_price = limit_price if limit_price > 0 else current_price
                order_type = 'LIMIT_MAKER'
                execution_exchange = 'mexc'
            else:
                order_type = 'MARKET'
                execution_exchange = 'coinbase'

            order_details = {
                'product_id': product_id,
                'side': decision.action,
                'size': decision.position_size,
                'price': current_price,
                'type': order_type,
                'exchange': execution_exchange,
            }

            # 1. Analyze Slippage Risk
            slippage_risk = self.slippage_protection.analyze_slippage_risk(order_details, market_data)
            self.logger.info(f"Slippage risk for {product_id}: {slippage_risk.get('risk_level', 'UNKNOWN')}")

            # 2. Map action to position side
            side = "LONG" if decision.action == "BUY" else "SHORT"

            # 3. Execute through position manager (risk checks -> API call -> position tracking)
            success, message, position = self.position_manager.open_position(
                product_id=product_id,
                side=side,
                size=decision.position_size,
                entry_price=current_price,
            )

            exec_result = {
                'status': 'EXECUTED' if success else 'REJECTED',
                'message': message,
                'position_id': position.position_id if position else None,
                'execution_price': current_price,
                'slippage': slippage_risk.get('predicted_slippage', 0.0),
                'exchange': execution_exchange,
                'order_type': order_type,
            }

            # Record trade cycle for anti-churn cooldown
            if success:
                self._last_trade_cycle[product_id] = getattr(self, 'scan_cycle_count', 0)
                # Log MEXC maker order
                if is_mexc_execution:
                    self.logger.info(
                        f"MEXC LIMIT ORDER: {decision.action} {decision.position_size:.8f} "
                        f"{product_id} @ ${current_price:.2f} (maker, 0% fee)"
                    )

            # Devil Tracker — record fill (actual execution price vs signal price)
            if success and self.devil_tracker:
                try:
                    _dtid = getattr(self, '_last_devil_trade_id', {}).get(product_id)
                    if _dtid:
                        self.devil_tracker.record_order_submission(_dtid, current_price)
                        self.devil_tracker.record_fill(
                            _dtid,
                            fill_price=current_price,
                            fill_quantity=decision.position_size,
                            fill_fee=slippage_risk.get('predicted_slippage', 0.0) * decision.position_size * current_price / 10000,
                        )
                except Exception as _dt_err:
                    self.logger.debug(f"Devil tracker fill record failed: {_dt_err}")

            # 4. Persist Trade
            if success and self.db_enabled:
                trade_data = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'product_id': product_id,
                    'side': decision.action,
                    'size': decision.position_size,
                    'price': current_price,
                    'status': 'EXECUTED',
                    'algo_used': f'POSITION_MANAGER_{execution_exchange.upper()}',
                    'slippage': slippage_risk.get('predicted_slippage', 0.0) if not is_mexc_execution else 0.0,
                    'execution_time': 0.0,
                }
                self._track_task(self.db_manager.store_trade(trade_data))
                # Persist position to DB for state recovery
                if position:
                    self._track_task(self.db_manager.save_position({
                        'position_id': position.position_id,
                        'product_id': product_id,
                        'side': side,
                        'size': decision.position_size,
                        'entry_price': current_price,
                        'stop_loss_price': position.stop_loss_price,
                        'take_profit_price': position.take_profit_price,
                        'opened_at': position.entry_time.isoformat(),
                        'status': 'OPEN',
                    }))

            # 4.3 Feed trade to BarAggregator (BUG 7 fix — was never called)
            if success and self.bar_aggregator:
                try:
                    import time as _time
                    self.bar_aggregator.on_trade(
                        pair=product_id,
                        exchange=execution_exchange,
                        price=current_price,
                        quantity=decision.position_size,
                        side=decision.action.lower(),
                        timestamp=_time.time(),
                    )
                except Exception:
                    pass

            # 4.5 Send monitoring alert for executed trade (Module C)
            if success and self.monitoring_alert_manager:
                try:
                    self._track_task(self.monitoring_alert_manager.send_trade_alert({
                        'product_id': product_id,
                        'side': decision.action,
                        'size': decision.position_size,
                        'price': current_price,
                        'confidence': decision.confidence,
                        'slippage': slippage_risk.get('predicted_slippage', 0.0),
                    }))
                except Exception:
                    pass

            # 5. Check daily loss after trade
            if self.position_manager.daily_pnl < -self.daily_loss_limit:
                self.trigger_kill_switch(
                    f"Daily loss limit breached: ${abs(self.position_manager.daily_pnl):.2f}"
                )

            if exec_result['status'] == 'REJECTED':
                n_pos = len(self.position_manager.positions)
                exp = self.position_manager._calculate_total_exposure()
                lim = self.position_manager.risk_limits.max_total_exposure_usd
                self.logger.info(
                    f"Smart execution complete: REJECTED ({product_id}) | {message} "
                    f"| positions={n_pos}, exposure=${exp:.0f}, limit=${lim:.0f}, "
                    f"trade_usd=${decision.position_size * current_price:.0f}"
                )
            else:
                self.logger.info(f"Smart execution complete: {exec_result['status']} | {message}")
            return exec_result

        except Exception as e:
            self.logger.error(f"Smart execution failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}

    async def _run_adaptive_learning_cycle(self):
        """Step 15: Online model calibration and attribution analysis"""
        # Specific log for verification
        with open("logs/adaptive_learning.log", "a") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} - Adaptive Learning Cycle triggered.\n")
            
        self.logger.info("Adaptive Learning Cycle triggered.")
        if not self.db_enabled:
            return

        try:
            # 1. Fetch recent decisions
            recent_decisions = await self.db_manager.get_recent_data('decisions', hours=24)
            if len(recent_decisions) < 5:
                self.logger.info("Insufficient data for adaptive learning. Need at least 5 decisions.")
                return

            self.logger.info(f"Starting Adaptive Learning Cycle with {len(recent_decisions)} data points.")

            # 2. Run Genetic Weight Optimization (Step 14)
            # Skip genetic optimization when weights are locked
            if self.config.get('weight_lock', False):
                self.logger.info("Weight lock enabled — skipping genetic optimization")
            else:
                optimized_weights = await self.genetic_optimizer.run_optimization_cycle(self.signal_weights)

                if optimized_weights != self.signal_weights:
                    self.logger.info("New optimized weights discovered via Evolution!")
                    async with self._weights_lock:
                        old_weights = self.signal_weights.copy()
                        self.signal_weights = optimized_weights

                    # Log the change
                    for k, v in optimized_weights.items():
                        diff = v - old_weights.get(k, 0)
                        if abs(diff) > 0.001:
                            self.logger.info(f"  {k}: {old_weights.get(k,0):.3f} -> {v:.3f} ({diff:+.3f})")

                    # 3. Persist to config.json to close the loop
                    self._save_optimized_weights(optimized_weights)
            
            # 4. Run Self-Reinforcing Learning Cycle (Step 19)
            if self.real_time_pipeline.enabled:
                await self.learning_engine.run_learning_cycle(
                    self.real_time_pipeline.processor.models
                )

            # 5. Trigger meta-learner training if we have an Ensemble model
            processor = self.real_time_pipeline.processor
            if "Ensemble" in processor.models:
                ensemble = processor.models["Ensemble"]
                self.logger.info("Calibrating Quantum Ensemble meta-learner with recent experience.")

            self.logger.info(f"Adaptive calibration complete. Analyzed {len(recent_decisions)} recent data points.")

        except Exception as e:
            self.logger.error(f"Adaptive learning cycle failed: {e}")

    def _save_optimized_weights(self, weights: Dict[str, float]):
        """Persist optimized weights back to config/config.json"""
        try:
            if not self.config_path.exists():
                return

            with open(self.config_path, 'r') as f:
                config_data = json.load(f)

            # Respect weight_lock — never overwrite locked weights
            if config_data.get('weight_lock', False):
                self.logger.info("Weight lock enabled — skipping weight persistence")
                return

            # Ensure ML weights survive genetic optimization
            _ml_required = {'ml_ensemble': 0.05, 'ml_cnn': 0.03}
            for k, v in _ml_required.items():
                if k not in weights:
                    weights[k] = v

            config_data['signal_weights'] = weights

            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=4)

            self.logger.info(f"Optimized weights persisted to {self.config_path}")

        except Exception as e:
            self.logger.error(f"Failed to persist optimized weights: {e}")

    async def _perform_attribution_analysis(self):
        """Step 11/13: Comprehensive performance attribution with Factor Analysis"""
        if not self.db_enabled:
            return

        try:
            # 1. Fetch labels (Realized outcomes) from DB
            # This uses the ground truth created in Step 19
            labels = await self.db_manager.get_recent_data('labels', hours=72)
            if not labels:
                self.logger.info("Attribution Analysis: No recent labels available yet.")
                return

            self.logger.info(f"🏛️ RENAISSANCE ATTRIBUTION: Analyzing {len(labels)} realized outcomes.")

            # 2. Prepare Factor Exposures from signal contributions
            # We use the signal_contributions stored in the decisions table (via reasoning JSON)
            decisions = await self.db_manager.get_recent_data('decisions', hours=72)
            if not decisions:
                self.logger.info("Attribution Analysis: No recent decisions available.")
                return
            
            # Map labels to decisions
            label_map = {l['decision_id']: l for l in labels}
            
            portfolio_returns = []
            benchmark_returns = []
            
            # Use current signal weights to define factors
            current_factors = list(self.signal_weights.keys())
            factor_exposures = {k: [] for k in current_factors}
            
            for d in decisions:
                if d['id'] in label_map:
                    l = label_map[d['id']]
                    # Portfolio return is based on decision and actual price change
                    side_mult = 1.0 if d['action'] == 'BUY' else -1.0 if d['action'] == 'SELL' else 0.0
                    portfolio_returns.append(l['ret_pct'] * side_mult)
                    
                    # Benchmark (Buy and Hold)
                    benchmark_returns.append(l['ret_pct'])
                    
                    # Factors (Normalized contributions)
                    reasoning = json.loads(d['reasoning'])
                    contributions = reasoning.get('signal_contributions', {})
                    for k in factor_exposures.keys():
                        factor_exposures[k].append(contributions.get(k, 0.0))

            if len(portfolio_returns) < 5:
                self.logger.info("Attribution Analysis: Insufficient data samples for factor regression.")
                return

            # Execute Attribution
            attribution = self.attribution_engine.analyze_performance_attribution(
                pd.Series(portfolio_returns),
                pd.Series(benchmark_returns),
                factor_exposures,
                {'factor_returns': pd.DataFrame()} # Market data can be enhanced later
            )
            
            if 'error' not in attribution:
                summary = attribution.get('performance_summary', {})
                self.logger.info(f"✅ ATTRIBUTION COMPLETE: Alpha: {summary.get('alpha', 0):+.4f} | Beta: {summary.get('beta', 0):.4f}")
                
                # Identify Top Alpha Drivers
                factor_attr = attribution.get('factor_attribution', {})
                if factor_attr:
                    # Sort factors by their contribution to return
                    drivers = sorted(factor_attr.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
                    top_driver = drivers[0][0] if drivers else "None"
                    self.logger.info(f"🚀 TOP ALPHA DRIVER: {top_driver}")
            
        except Exception as e:
            self.logger.error(f"Performance attribution failed: {e}")

    def _fetch_account_balance(self) -> float:
        """Fetch current USD account balance from Coinbase (or paper trader).

        Includes sanity cap: paper trading balance cannot exceed 2x initial
        to prevent phantom inflation from short-sell accounting.
        """
        try:
            portfolio = self.coinbase_client.get_portfolio_breakdown()
            if "error" not in portfolio:
                balance = portfolio.get("total_balance_usd", 0.0)
                if balance > 0:
                    # Safety cap for paper trading
                    if getattr(self.coinbase_client, 'paper_trading', False):
                        initial = getattr(
                            getattr(self.coinbase_client, 'paper_trader', None),
                            'INITIAL_BALANCE_USD', 10000.0
                        )
                        balance = min(balance, initial * 2.0)
                    self._cached_balance_usd = balance
                    return balance
        except Exception as e:
            self.logger.debug(f"Balance fetch failed: {e}")
        # Return cached or default
        return self._cached_balance_usd if self._cached_balance_usd > 0 else 50000.0

    async def execute_trading_cycle(self) -> TradingDecision:
        """Execute one complete trading cycle across all products"""
        cycle_start = time.time()
        decisions = []

        try:
            # Fetch live account balance for dynamic position sizing
            account_balance = self._fetch_account_balance()
            self.logger.info(f"Account balance: ${account_balance:,.2f}")

            # Dynamically update position manager limits — selective: 1 per product, 15 max
            # With 43 pairs we scan widely but trade selectively (max 15 simultaneous)
            # Each trade is $1K-$5K (3-10% of equity), so 50% cap ≈ 5-25 simultaneous trades
            self.position_manager.risk_limits.max_position_size_usd = account_balance * 0.10
            self.position_manager.risk_limits.max_total_exposure_usd = account_balance * 0.50
            self.position_manager.risk_limits.max_total_positions = min(len(self.product_ids), 15)
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

            if self._current_drawdown_pct >= 0.03:
                self.logger.warning(
                    f"DRAWDOWN ALERT: {self._current_drawdown_pct:.1%} from HWM ${self._high_watermark_usd:,.2f} | "
                    f"Weekly P&L: ${self._weekly_pnl:,.2f}"
                )

            # ── Drawdown Circuit Breaker ──
            self._drawdown_size_scalar = 1.0
            self._drawdown_exits_only = False
            if self._current_drawdown_pct >= 0.15:
                # 15%+ drawdown: close ALL positions immediately
                self.logger.warning("CIRCUIT BREAKER: 15% drawdown — closing all positions")
                try:
                    with self.position_manager._lock:
                        all_positions = list(self.position_manager.positions.values())
                    for pos in all_positions:
                        ok, _ = self.position_manager.close_position(pos.position_id, reason="Circuit breaker: 15% drawdown")
                        if ok:
                            _cpx = getattr(pos, 'current_price', 0.0) or 0.0
                            _side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                            _rpnl = self._compute_realized_pnl(
                                pos.entry_price, _cpx, pos.size, _side
                            )
                            self._track_task(self.db_manager.close_position_record(
                                pos.position_id,
                                close_price=float(_cpx),
                                realized_pnl=float(_rpnl),
                                exit_reason="circuit_breaker",
                            ))
                except Exception as cb_err:
                    self.logger.error(f"Circuit breaker close-all failed: {cb_err}")
                self._drawdown_exits_only = True
            elif self._current_drawdown_pct >= 0.10:
                # 10%+ drawdown: exits only, no new positions
                self.logger.warning("CIRCUIT BREAKER: 10% drawdown — exits only mode")
                self._drawdown_exits_only = True
            elif self._current_drawdown_pct >= 0.05:
                # 5%+ drawdown: reduce position sizes by 50%
                self._drawdown_size_scalar = 0.5
                self.logger.info(f"DRAWDOWN SCALING: {self._current_drawdown_pct:.1%} — 50% position sizes")

            # ── Continuous Exposure Monitor ──
            try:
                total_exposure = self.position_manager._calculate_total_exposure()
                max_exposure = account_balance * 0.50
                if total_exposure > max_exposure:
                    self.logger.warning(
                        f"EXPOSURE LIMIT: ${total_exposure:,.2f} > ${max_exposure:,.2f} — force-closing worst position"
                    )
                    with self.position_manager._lock:
                        open_positions = list(self.position_manager.positions.values())
                    if open_positions:
                        # Find worst-performing position
                        worst_pos = None
                        worst_pnl = float('inf')
                        for pos in open_positions:
                            if hasattr(pos, 'unrealized_pnl'):
                                if pos.unrealized_pnl < worst_pnl:
                                    worst_pnl = pos.unrealized_pnl
                                    worst_pos = pos
                            elif hasattr(pos, 'entry_price') and pos.entry_price > 0:
                                worst_pos = worst_pos or pos  # fallback: pick first
                        if worst_pos:
                            ok, _ = self.position_manager.close_position(
                                worst_pos.position_id, reason="Exposure limit exceeded"
                            )
                            if ok:
                                _cpx = getattr(worst_pos, 'current_price', 0.0) or 0.0
                                _side = worst_pos.side.value if hasattr(worst_pos.side, 'value') else str(worst_pos.side)
                                _rpnl = self._compute_realized_pnl(
                                    worst_pos.entry_price, _cpx, worst_pos.size, _side
                                )
                                self._track_task(self.db_manager.close_position_record(
                                    worst_pos.position_id,
                                    close_price=float(_cpx),
                                    realized_pnl=float(_rpnl),
                                    exit_reason="exposure_limit",
                                ))
                            self.logger.info(f"EXPOSURE CLOSE: {worst_pos.position_id}")
            except Exception as exp_err:
                self.logger.debug(f"Exposure monitor error: {exp_err}")

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

            else:
                # Fallback to tiered scanning if breakout scanner disabled
                cycle_pairs = self.get_pairs_for_cycle(self.scan_cycle_count)
                self.logger.info(
                    f"Cycle {self.scan_cycle_count}: scanning {len(cycle_pairs)}/{len(self.product_ids)} pairs"
                )

            # ── Preload candle history on first cycle (eliminates cold-start) ──
            if not getattr(self, '_history_preloaded', False):
                self._history_preloaded = True
                # Preload always-scan majors from Binance (or Coinbase fallback)
                preload_pairs = self.breakout_scanner.get_always_scan_pairs()[:15] if self.scanner_enabled else \
                    [p for p in self.product_ids if self._pair_tiers.get(p, 1) == 1][:15]
                for pid in preload_pairs:
                    try:
                        bsym = self._pair_binance_symbols.get(pid, to_binance_symbol(pid))
                        raw_candles = await self.binance_spot.fetch_candles(bsym, '5m', 200)
                        if not raw_candles:
                            # Fallback to Coinbase
                            raw_candles_cb = await asyncio.to_thread(
                                self.market_data_provider.fetch_candle_history, pid
                            )
                            if raw_candles_cb:
                                pid_tech = self._get_tech(pid)
                                for candle in raw_candles_cb:
                                    pid_tech.update_price_data(candle)
                                    if candle.close > 0:
                                        self.garch_engine.update_returns(pid, candle.close)
                                        self.stat_arb_engine.update_price(pid, candle.close)
                                        self.correlation_network.update_price(pid, candle.close)
                                        self.mean_reversion_engine.update_price(pid, candle.close)
                                        self.correlation_engine.update_price(pid, candle.close)
                                self.logger.info(f"Preloaded {len(raw_candles_cb)} candles for {pid} (Coinbase)")
                            continue

                        from enhanced_technical_indicators import PriceData
                        pid_tech = self._get_tech(pid)
                        for c in raw_candles:
                            pd_obj = PriceData(
                                timestamp=datetime.utcfromtimestamp(c['timestamp']),
                                open=c['open'], high=c['high'],
                                low=c['low'], close=c['close'],
                                volume=c['volume'],
                            )
                            pid_tech.update_price_data(pd_obj)
                            if c['close'] > 0:
                                self.garch_engine.update_returns(pid, c['close'])
                                self.stat_arb_engine.update_price(pid, c['close'])
                                self.correlation_network.update_price(pid, c['close'])
                                self.mean_reversion_engine.update_price(pid, c['close'])
                                self.correlation_engine.update_price(pid, c['close'])
                        self.logger.info(
                            f"Preloaded {len(raw_candles)} candles for {pid} (Binance) — "
                            f"price_history={len(pid_tech.price_history)}"
                        )
                        if self.garch_engine.should_refit(pid):
                            self.garch_engine.fit_model(pid)
                    except Exception as e:
                        self.logger.warning(f"History preload failed for {pid}: {e}")

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
                        _cdf = self._load_price_df_from_db(_pid, limit=100)
                    if _cdf is not None and len(_cdf) > 0:
                        cross_data[_pid] = _cdf
                except Exception:
                    pass

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
                    except Exception:
                        pass

                # 1.56 Feed fast signal layers with real-time price data
                if self.fast_reversion_scanner and current_price > 0:
                    try:
                        self.fast_reversion_scanner.on_price_update(
                            pair=product_id,
                            price=float(current_price),
                            volume=float(ticker.get('volume_24h', 0)),
                            timestamp=time.time(),
                        )
                    except Exception:
                        pass
                if self.liquidation_detector and hasattr(self.liquidation_detector, 'on_price_update'):
                    try:
                        self.liquidation_detector.on_price_update(
                            symbol=product_id.replace("-USD", "USDT").replace("-", ""),
                            price=float(current_price) if current_price else 0.0,
                            volume=float(ticker.get('volume_24h', 0)),
                            spread_bps=float(ticker.get('spread_bps', 0)),
                            timestamp=time.time(),
                        )
                    except Exception:
                        pass

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
                try:
                    deriv_snap = await self.derivatives_provider.get_derivatives_snapshot(product_id)
                    if deriv_snap:
                        market_data['_derivatives_data'] = {
                            k: pd.Series([v]) for k, v in deriv_snap.items()
                            if not (isinstance(v, float) and np.isnan(v))
                        }
                except Exception as _de:
                    self.logger.debug(f"Derivatives fetch skipped for {product_id}: {_de}")

                # 2. Generate signals from all components
                signals = await self.generate_signals(market_data)
                
                # HARDENING: Ensure all signals are floats
                signals = {k: self._force_float(v) for k, v in signals.items()}

                # 2.0b Breakout Scanner signal injection
                _bo_sig = self._breakout_scores.get(product_id)
                if _bo_sig and _bo_sig.breakout_score >= 40:
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

                # 2.0a Advanced Microstructure Signals (Module F)
                if self.signal_aggregator:
                    try:
                        ob_snap = market_data.get('order_book_snapshot')
                        bids_list, asks_list = [], []
                        if ob_snap is not None:
                            # OrderBookSnapshot object with .bids / .asks lists of OrderBookLevel
                            if hasattr(ob_snap, 'bids') and hasattr(ob_snap, 'asks'):
                                bids_list = [(float(l.price), float(l.size)) for l in ob_snap.bids[:20]
                                             if hasattr(l, 'price')]
                                asks_list = [(float(l.price), float(l.size)) for l in ob_snap.asks[:20]
                                             if hasattr(l, 'price')]
                            elif isinstance(ob_snap, dict):
                                # Raw dict format {bids: {price: size}, asks: {price: size}}
                                bids_raw = ob_snap.get('bids', {})
                                asks_raw = ob_snap.get('asks', {})
                                if isinstance(bids_raw, dict):
                                    bids_list = sorted(
                                        [(float(p), float(s)) for p, s in bids_raw.items()],
                                        reverse=True
                                    )[:20]
                                    asks_list = sorted(
                                        [(float(p), float(s)) for p, s in asks_raw.items()]
                                    )[:20]

                        if bids_list and asks_list:
                            self.signal_aggregator.update_book(bids_list, asks_list)
                            micro_entry = self.signal_aggregator.get_signal_dict_entry()
                            micro_score = self._force_float(micro_entry.get('microstructure_advanced', 0.0))
                            if abs(micro_score) > 0.001:
                                signals['microstructure_advanced'] = micro_score
                    except Exception as _micro_err:
                        self.logger.debug(f"Advanced microstructure signal failed: {_micro_err}")

                # 2.0b Liquidation Cascade Signal (Module D)
                if self.liquidation_detector:
                    try:
                        # Map product_id to Binance symbol (BTC-USD → BTCUSDT)
                        binance_sym = product_id.replace("-USD", "USDT").replace("-", "")
                        current_risk = await self.liquidation_detector.get_current_risk()
                        sym_risk = current_risk.get(binance_sym, {})
                        risk_score = float(sym_risk.get('risk_score', 0.0))
                        if risk_score > 0.3:
                            direction = sym_risk.get('direction', 'long_liquidation')
                            direction_mult = 1.0 if direction == "short_squeeze" else -1.0
                            signals['liquidation_cascade'] = self._force_float(
                                direction_mult * risk_score
                            )
                            market_data['cascade_risk'] = {
                                'symbol': binance_sym,
                                'direction': direction,
                                'risk_score': risk_score,
                                'funding_rate': sym_risk.get('funding_rate', 0.0),
                            }
                    except Exception as _liq_err:
                        self.logger.debug(f"Liquidation cascade signal failed: {_liq_err}")

                # 2.0d Fast Mean Reversion Signal
                if self.fast_reversion_scanner:
                    try:
                        fmr_signal = self.fast_reversion_scanner.get_latest_signal(product_id)
                        if fmr_signal and fmr_signal.confidence > 0.52:
                            direction_mult = 1.0 if fmr_signal.direction == "long" else -1.0
                            signals['fast_mean_reversion'] = self._force_float(
                                direction_mult * fmr_signal.confidence
                            )
                    except Exception as _fmr_err:
                        self.logger.debug(f"Fast mean reversion signal failed: {_fmr_err}")

                # 2.0c Multi-Exchange Signal Bridge (MEXC + Binance consensus)
                if self.multi_exchange_bridge:
                    try:
                        # Extract Coinbase order book volumes for aggregation
                        cb_bid_vol = 0.0
                        cb_ask_vol = 0.0
                        cb_bid = 0.0
                        cb_ask = 0.0
                        ob_snap = market_data.get('order_book_snapshot')
                        if ob_snap is not None:
                            if hasattr(ob_snap, 'bids') and ob_snap.bids:
                                cb_bid = float(ob_snap.bids[0].price) if hasattr(ob_snap.bids[0], 'price') else 0.0
                                cb_bid_vol = sum(float(lv.size) for lv in ob_snap.bids[:10] if hasattr(lv, 'size'))
                            if hasattr(ob_snap, 'asks') and ob_snap.asks:
                                cb_ask = float(ob_snap.asks[0].price) if hasattr(ob_snap.asks[0], 'price') else 0.0
                                cb_ask_vol = sum(float(lv.size) for lv in ob_snap.asks[:10] if hasattr(lv, 'size'))

                        me_signals = self.multi_exchange_bridge.get_signals(
                            product_id=product_id,
                            coinbase_bid=cb_bid,
                            coinbase_ask=cb_ask,
                            coinbase_bid_vol=cb_bid_vol,
                            coinbase_ask_vol=cb_ask_vol,
                        )
                        # Inject as a single weighted composite signal
                        me_cfg = self.config.get("multi_exchange_signals", {})
                        me_weights = me_cfg.get("weights", {})
                        me_composite = 0.0
                        me_weight_sum = 0.0
                        for sig_name, sig_val in me_signals.items():
                            w = me_weights.get(sig_name, 0.025)
                            me_composite += sig_val * w
                            me_weight_sum += w
                        if me_weight_sum > 0:
                            me_composite /= me_weight_sum
                        signals['multi_exchange'] = self._force_float(me_composite)
                        if abs(me_composite) > 0.01:
                            self.logger.info(
                                f"MULTI-EXCHANGE [{product_id}]: composite={me_composite:+.4f} "
                                f"momentum={me_signals['cross_exchange_momentum']:+.4f} "
                                f"dispersion={me_signals['price_dispersion']:+.4f} "
                                f"imbalance={me_signals['aggregated_book_imbalance']:+.4f} "
                                f"funding={me_signals['funding_rate_signal']:+.4f}"
                            )
                    except Exception as _me_err:
                        self.logger.debug(f"Multi-exchange bridge error: {_me_err}")

                # 2.0b Medallion Signal Analogs (sharp move reversion, seasonality, funding timing)
                if self.medallion_analogs:
                    try:
                        _tech = self._get_tech(product_id)
                        price_hist = list(_tech.price_history) if hasattr(_tech, 'price_history') else []
                        # Get funding rate from multi-exchange bridge cache if available
                        _funding = 0.0
                        if hasattr(self, 'multi_exchange_bridge') and self.multi_exchange_bridge:
                            usdt_sym = product_id.split("-")[0] + "/USDT"
                            _funding = self.multi_exchange_bridge._funding_cache.get(usdt_sym, 0.0)

                        analog_signals = self.medallion_analogs.get_signals(
                            product_id=product_id,
                            current_price=current_price,
                            price_history=price_hist,
                            funding_rate=_funding,
                        )
                        # Inject as weighted sub-signals
                        analog_weights = self.config.get('medallion_analogs', {}).get('weights', {})
                        analog_composite = 0.0
                        analog_w_sum = 0.0
                        for sig_name, sig_val in analog_signals.items():
                            w = analog_weights.get(sig_name, 0.01)
                            analog_composite += sig_val * w
                            analog_w_sum += w
                        if analog_w_sum > 0:
                            analog_composite /= analog_w_sum
                        if abs(analog_composite) > 0.001:
                            signals['medallion_analog'] = self._force_float(analog_composite)
                    except Exception as _ma_err:
                        self.logger.debug(f"Medallion analogs error: {_ma_err}")

                # Build OHLCV DataFrame for ML models (needed by bridge + RT pipeline)
                _tech_inst = self._get_tech(product_id)
                price_df = _tech_inst._to_dataframe()
                # Fallback: if tech indicators have <30 rows, load from DB bars
                if len(price_df) < 30:
                    price_df = self._load_price_df_from_db(product_id, limit=100)

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
                    except Exception:
                        pass

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
                self._pending_predictions[product_id] = {
                    'price': current_price,
                    'signals': {k: float(v) for k, v in signals.items()},
                    'cycle': getattr(self, 'scan_cycle_count', 0),
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

                # 3.15 Polymarket Bridge — emit BTC signal for binary bet markets
                if product_id == 'BTC-USD':
                    try:
                        _pm_model_preds = {}
                        if ml_package and ml_package.ml_predictions:
                            for mp in ml_package.ml_predictions:
                                if isinstance(mp, (tuple, list)) and len(mp) >= 2:
                                    _pm_model_preds[str(mp[0])] = float(mp[1]) if isinstance(mp[1], (int, float)) else 0.0
                                elif isinstance(mp, dict):
                                    _pm_model_preds[mp.get('name', 'unknown')] = float(mp.get('prediction', 0.0))

                        # Compute agreement from model predictions
                        _pm_agreement = 0.5
                        if _pm_model_preds:
                            _pm_signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in _pm_model_preds.values()]
                            _pm_nonzero = [s for s in _pm_signs if s != 0]
                            if _pm_nonzero:
                                _pm_agreement = max(_pm_nonzero.count(1), _pm_nonzero.count(-1)) / len(_pm_nonzero)

                        _pm_regime = regime_label if 'regime_label' in dir() else (
                            self.regime_overlay.get_hmm_regime_label() if self.regime_overlay.enabled else "unknown"
                        )

                        _pm_btc_breakout = 0.0
                        _pm_bo = self._breakout_scores.get('BTC-USD')
                        if _pm_bo:
                            _pm_btc_breakout = _pm_bo.breakout_score

                        _pm_price = float(market_data.get('ticker', {}).get('price', 0.0))

                        self.polymarket_bridge.generate_signal(
                            prediction=weighted_signal,
                            agreement=_pm_agreement,
                            regime=_pm_regime,
                            breakout_score=_pm_btc_breakout,
                            btc_price=_pm_price,
                            model_confidences=_pm_model_preds,
                            scanner_opportunities=self._latest_scanner_opportunities,
                        )
                    except Exception as _pm_err:
                        self.logger.debug(f"Polymarket bridge error: {_pm_err}")

                    # 3.16 Polymarket Scanner — discover all crypto prediction markets (every 5 min)
                    try:
                        _scan_due = self._last_poly_scan is None or \
                            (datetime.now() - self._last_poly_scan).total_seconds() >= 300
                        if _scan_due:
                            _scan_preds: Dict[str, float] = {}
                            _scan_prices: Dict[str, float] = {}
                            # Map BTC-USD prediction to BTC asset key
                            if ml_package and ml_package.ensemble_score != 0.0:
                                _scan_preds['BTC'] = float(weighted_signal)
                            # Always include current BTC price (we're in the BTC-USD block)
                            if _pm_price and _pm_price > 0:
                                _scan_prices['BTC'] = float(_pm_price)
                            # Collect latest prices for all traded assets
                            if hasattr(self, '_last_prices'):
                                for _pid, _px in self._last_prices.items():
                                    if _px > 0 and '-' in _pid:
                                        _asset_key = _pid.split('-')[0]
                                        _scan_prices[_asset_key] = float(_px)
                            _scan_opps = await self.polymarket_scanner.scan(
                                ml_predictions=_scan_preds,
                                agreement=_pm_agreement,
                                regime=_pm_regime,
                                current_prices=_scan_prices,
                            )
                            self._last_poly_scan = datetime.now()
                            # Include top scanner opportunities in next bridge signal
                            if _scan_opps:
                                self._latest_scanner_opportunities = [
                                    {
                                        "condition_id": o.market.condition_id,
                                        "question": o.market.question[:120],
                                        "market_type": o.market.market_type,
                                        "asset": o.market.asset,
                                        "direction": o.direction,
                                        "edge": o.edge,
                                        "confidence": o.confidence,
                                        "our_probability": o.our_probability,
                                        "yes_price": o.market.yes_price,
                                        "target_price": o.market.target_price,
                                        "timeframe_minutes": o.market.timeframe_minutes,
                                        "source": o.source,
                                    }
                                    for o in _scan_opps[:10]
                                ]
                            else:
                                self._latest_scanner_opportunities = []
                    except Exception as _ps_err:
                        self.logger.debug(f"Polymarket scanner error: {_ps_err}")

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
                                except Exception:
                                    pass
                    else:
                        # For non-BTC assets, compute pair vs BTC
                        try:
                            stat_arb_data = self.stat_arb_engine.calculate_pair_signal(base, product_id)
                            if 'signal' in stat_arb_data:
                                stat_arb_data['signal'] = -self._force_float(stat_arb_data['signal'])
                        except Exception:
                            pass
                
                if stat_arb_data.get('status') == 'active':
                    signals['stat_arb'] = self._force_float(stat_arb_data['signal'])
                else:
                    signals['stat_arb'] = 0.0

                # 5. Make trading decision
                ticker = market_data.get('ticker', {})
                current_price = self._force_float(ticker.get('price', 0.0))
                
                # ── Market Sanity Checks (pre-trade gates) ──
                # 1. Stale data check
                data_ts = market_data.get('timestamp')
                if data_ts:
                    if isinstance(data_ts, str):
                        data_ts = datetime.fromisoformat(data_ts)
                    data_age = (datetime.now() - data_ts).total_seconds()
                    if data_age > 60:
                        self.logger.warning(f"SANITY: Market data {data_age:.0f}s old - holding")
                        continue

                # 2. Flash crash / price spike detection
                if hasattr(self, '_last_prices') and product_id in self._last_prices:
                    last_px = self._last_prices[product_id]
                    if last_px > 0 and current_price > 0:
                        pct_change = abs(current_price - last_px) / last_px
                        if pct_change > 0.05:  # 5% move in one cycle
                            self.logger.warning(
                                f"SANITY: Flash move {pct_change:.1%} on {product_id} "
                                f"(${last_px:,.2f} -> ${current_price:,.2f}) — skipping cycle"
                            )
                            continue
                if not hasattr(self, '_last_prices'):
                    self._last_prices = {}
                self._last_prices[product_id] = current_price

                # 3. Abnormal spread detection
                ticker_data = market_data.get('ticker', {})
                bid = self._force_float(ticker_data.get('bid', 0))
                ask = self._force_float(ticker_data.get('ask', 0))
                if bid > 0 and ask > 0:
                    spread_bps = ((ask - bid) / ((ask + bid) / 2)) * 10000
                    if spread_bps > 50:  # >50bps spread = illiquid
                        self.logger.warning(
                            f"SANITY: Wide spread {spread_bps:.0f}bps on {product_id} — reducing confidence"
                        )
                        market_data['_sanity_spread_penalty'] = True

                # 4. Weekly loss limit check
                weekly_loss_limit = self._high_watermark_usd * 0.20 if self._high_watermark_usd > 0 else 2000
                if self._weekly_pnl < -weekly_loss_limit:
                    self.logger.warning(
                        f"SANITY: Weekly loss ${self._weekly_pnl:,.2f} exceeds limit ${-weekly_loss_limit:,.2f} — holding"
                    )
                    continue

                # 5.1 Meta-Strategy Selection
                regime_data = self.regime_overlay.current_regime or {}
                self.last_vpin = market_data.get('vpin', 0.5)
                execution_mode = self.strategy_selector.select_mode(market_data, regime_data)
                market_data['execution_mode'] = execution_mode

                decision = self.make_trading_decision(weighted_signal, contributions,
                                                    current_price=current_price,
                                                    real_time_result=rt_result,
                                                    product_id=product_id,
                                                    ml_package=ml_package,
                                                    market_data=market_data,
                                                    drawdown_pct=getattr(self, '_current_drawdown_pct', 0.0))

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
                    except Exception:
                        pass

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
                        }))

                # 5.5 Exit Engine — Monitor open positions for alpha decay
                try:
                    with self.position_manager._lock:
                        open_positions = list(self.position_manager.positions.values())
                    for pos in open_positions:
                        if pos.product_id != product_id:
                            continue
                        holding_periods = int(
                            (datetime.now() - pos.entry_time).total_seconds()
                            / max(60, self.config.get("trading", {}).get("cycle_interval_seconds", 300))
                        )
                        garch_forecast = market_data.get('garch_forecast', {})
                        _pos_side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                        exit_decision = self.position_sizer.calculate_exit_size(
                            position_size=pos.size,
                            entry_price=pos.entry_price,
                            current_price=current_price,
                            holding_periods=holding_periods,
                            confidence=decision.confidence,
                            volatility=garch_forecast.get('forecast_vol'),
                            regime=self.regime_overlay.get_current_regime() if hasattr(self.regime_overlay, 'get_current_regime') else None,
                            side=_pos_side,
                        )
                        if exit_decision['exit_fraction'] > 0:
                            self.logger.info(
                                f"EXIT ENGINE [{pos.position_id}]: {exit_decision['reason']} — "
                                f"fraction={exit_decision['exit_fraction']:.0%}, urgency={exit_decision['urgency']}"
                            )
                            close_ok, close_msg = self.position_manager.close_position(
                                pos.position_id, reason=f"Exit engine: {exit_decision['reason']}"
                            )
                            if close_ok:
                                _side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                                _rpnl = self._compute_realized_pnl(
                                    pos.entry_price, current_price, pos.size, _side
                                )
                                self._track_task(self.db_manager.close_position_record(
                                    pos.position_id,
                                    close_price=float(current_price),
                                    realized_pnl=float(_rpnl),
                                    exit_reason=f"exit_engine:{exit_decision['reason']}",
                                ))
                                _hold_min = (datetime.now() - pos.entry_time).total_seconds() / 60
                                self.logger.info(
                                    f"TRADE CLOSED: {product_id} held {_hold_min:.1f} min | "
                                    f"reason=exit_engine:{exit_decision['reason']} | P&L=${float(_rpnl):.2f}"
                                )
                                # Record trade PnL for health monitor
                                if self.health_monitor and pos.entry_price > 0:
                                    trade_pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                                    if pos.side.value.upper() == 'SHORT':
                                        trade_pnl_pct = -trade_pnl_pct
                                    self.health_monitor.record_trade(trade_pnl_pct, product_id)
                                self._track_task(self.dashboard_emitter.emit("trade", {
                                    "product_id": product_id,
                                    "side": "EXIT",
                                    "price": float(current_price),
                                    "size": float(pos.size * exit_decision['exit_fraction']),
                                    "reason": exit_decision['reason'],
                                }))
                except Exception as exit_err:
                    self.logger.debug(f"Exit engine error: {exit_err}")

                # 5.6 Continuous Position Re-evaluation (Doc 10)
                if self.position_reevaluator:
                    try:
                        with self.position_manager._lock:
                            _reeval_positions = list(self.position_manager.positions.values())
                        _reeval_positions_for_pid = [p for p in _reeval_positions if p.product_id == product_id]
                        if _reeval_positions_for_pid:
                            from decimal import Decimal as _D
                            _regime_label = self.regime_overlay.get_hmm_regime_label() if self.regime_overlay.enabled else "unknown"
                            _contexts = []
                            for _pos in _reeval_positions_for_pid:
                                _age_s = (datetime.now() - _pos.entry_time).total_seconds()
                                _side = _pos.side.value.lower() if hasattr(_pos.side, 'value') else str(_pos.side).lower()
                                _pnl_bps = 0.0
                                if _pos.entry_price > 0 and current_price > 0:
                                    _move = (current_price - _pos.entry_price) / _pos.entry_price * 10000
                                    _pnl_bps = _move if _side == "long" else -_move
                                _signal_ttl = self.config.get('reevaluation', {}).get('signal_ttl_seconds', 3600)
                                _ctx = PositionContext(
                                    position_id=_pos.position_id,
                                    pair=product_id,
                                    exchange="mexc" if product_id in self._pair_binance_symbols else "coinbase",
                                    side=_side,
                                    strategy="combined",
                                    entry_price=_D(str(_pos.entry_price)),
                                    entry_size=_D(str(_pos.size)),
                                    entry_size_usd=_D(str(_pos.size * _pos.entry_price)),
                                    entry_timestamp=_pos.entry_time.timestamp(),
                                    entry_confidence=decision.confidence,
                                    entry_expected_move_bps=10.0,
                                    entry_cost_estimate_bps=2.0,
                                    entry_net_edge_bps=8.0,
                                    entry_regime=_regime_label,
                                    entry_volatility=0.02,
                                    entry_book_depth_usd=_D("50000"),
                                    entry_spread_bps=1.0,
                                    signal_ttl_seconds=_signal_ttl,
                                    current_size=_D(str(_pos.size)),
                                    current_size_usd=_D(str(_pos.size * current_price)),
                                    current_price=_D(str(current_price)),
                                    unrealized_pnl_bps=_pnl_bps,
                                    current_confidence=decision.confidence,
                                    current_regime=_regime_label,
                                )
                                _contexts.append(_ctx)
                            _reeval_results = self.position_reevaluator.reevaluate_all(
                                _contexts,
                                portfolio_state={"equity": self._cached_balance_usd or 10000.0},
                                market_state={"regime": _regime_label, "price": current_price},
                            )
                            for _rr in _reeval_results:
                                if _rr.action == "close":
                                    _close_ok, _close_msg = self.position_manager.close_position(
                                        _rr.position_id, reason=f"ReEval: {_rr.reason_code}"
                                    )
                                    if _close_ok:
                                        # Compute realized PnL from entry/exit prices
                                        _rr_pos = next((p for p in _reeval_positions_for_pid if p.position_id == _rr.position_id), None)
                                        if _rr_pos:
                                            _rr_side = _rr_pos.side.value if hasattr(_rr_pos.side, 'value') else str(_rr_pos.side)
                                            _rr_rpnl = self._compute_realized_pnl(
                                                _rr_pos.entry_price, current_price, _rr_pos.size, _rr_side
                                            )
                                        else:
                                            _rr_rpnl = 0.0
                                        self._track_task(
                                            self.db_manager.close_position_record(
                                                _rr.position_id,
                                                close_price=float(current_price),
                                                realized_pnl=float(_rr_rpnl),
                                                exit_reason=f"reeval:{_rr.reason_code}",
                                            )
                                        )
                                        _rr_hold_min = (datetime.now() - _rr_pos.entry_time).total_seconds() / 60 if _rr_pos else 0.0
                                        self.logger.warning(
                                            f"REEVAL CLOSE: {_rr.position_id} — {_rr.reason_code} "
                                            f"(edge={_rr.remaining_edge_bps:.1f}bps, urgency={_rr.urgency})"
                                        )
                                        self.logger.info(
                                            f"TRADE CLOSED: {product_id} held {_rr_hold_min:.1f} min | "
                                            f"reason=reeval:{_rr.reason_code} | P&L=${float(_rr_rpnl):.2f}"
                                        )
                                        if self.devil_tracker:
                                            self.devil_tracker.record_exit(
                                                _rr.position_id, "reeval", _rr.reason_code
                                            )
                                    else:
                                        self.logger.warning(
                                            f"REEVAL CLOSE FAILED: {_rr.position_id} — {_close_msg}"
                                        )
                                elif _rr.action == "trim" and _rr.trim_to_usd > 0:
                                    self.logger.warning(
                                        f"REEVAL TRIM: {_rr.position_id} to ${_rr.trim_to_usd:.2f} "
                                        f"— {_rr.reason_code}"
                                    )
                                elif _rr.action != "hold":
                                    self.logger.warning(
                                        f"REEVAL {_rr.action.upper()}: {_rr.position_id} — {_rr.reason_code}"
                                    )
                    except Exception as _reeval_err:
                        self.logger.warning(f"Position re-evaluation failed: {_reeval_err}")

                # 5.7 Position stacking is prevented in make_trading_decision()
                # Same-direction positions are blocked; reversals close existing positions

                # 6. Smart Execution (Step 10)
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
                    except Exception:
                        pass
                    try:
                        if self.beta_monitor:
                            beta_report = self.beta_monitor.get_report()
                            if self.beta_monitor.should_alert():
                                rec = self.beta_monitor.get_hedge_recommendation()
                                self.logger.warning(
                                    f"BETA MONITOR: beta={beta_report.get('beta', 0):.2f} — "
                                    f"hedge: {rec.get('action', 'none')}"
                                )
                    except Exception:
                        pass
                    try:
                        if self.capacity_monitor:
                            caps = self.capacity_monitor.get_all_capacities()
                            for _cpair, _cdata in caps.items():
                                if _cdata.get("at_capacity_wall"):
                                    self.logger.warning(
                                        f"CAPACITY MONITOR: {_cpair} at capacity wall"
                                    )
                    except Exception:
                        pass

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
                try:
                    self._track_task(self.dashboard_emitter.emit("cycle", {
                        "product_id": product_id,
                        "action": decision.action,
                        "confidence": float(decision.confidence),
                        "position_size": float(decision.position_size),
                        "weighted_signal": float(weighted_signal),
                        "hmm_regime": self.regime_overlay.get_hmm_regime_label() if self.regime_overlay.enabled else None,
                        "price": float(current_price),
                    }))
                    # Emit live regime state (classifier, bar count, confidence)
                    if self.regime_overlay.enabled and self.regime_overlay.current_regime:
                        _regime = self.regime_overlay.current_regime
                        self._track_task(self.dashboard_emitter.emit("regime", {
                            "hmm_regime": _regime.get("hmm_regime", "unknown"),
                            "confidence": float(_regime.get("hmm_confidence", 0.0)),
                            "classifier": self.regime_overlay.active_classifier,
                            "bar_count": self.regime_overlay.bar_count,
                            "trend_persistence": float(_regime.get("trend_persistence", 0.0)),
                            "volatility_acceleration": float(_regime.get("volatility_acceleration", 1.0)),
                            "details": _regime.get("bootstrap_details", ""),
                        }))
                    if decision.action != 'HOLD':
                        self._track_task(self.dashboard_emitter.emit("trade", {
                            "product_id": product_id,
                            "side": decision.action,
                            "price": float(current_price),
                            "size": float(decision.position_size),
                        }))
                    if hasattr(self, 'risk_gateway') and self.risk_gateway:
                        self._track_task(self.dashboard_emitter.emit("risk.gateway", {
                            "product_id": product_id,
                            "action": decision.action,
                            "vae_loss": float(decision.reasoning.get('vae_loss', 0.0) or 0.0),
                            "verdict": decision.reasoning.get('risk_gateway_reason', 'unknown'),
                            "pass_count": self.risk_gateway.pass_count,
                            "reject_count": self.risk_gateway.reject_count,
                        }))
                    if market_data.get('confluence_data'):
                        self._track_task(self.dashboard_emitter.emit("confluence", market_data['confluence_data']))
                except Exception as _de:
                    self.logger.debug(f"Dashboard emit error: {_de}")

            # Increment cycle counter ONCE per cycle (not per pair)
            self.scan_cycle_count += 1

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
                except Exception:
                    pass

            # Return the first decision or a HOLD if none
            return decisions[0] if decisions else TradingDecision('HOLD', 0.0, 0.0, {}, datetime.now())

        except Exception as e:
            import traceback
            self.logger.error(f"Trading cycle failed: {e}")
            self.logger.error(traceback.format_exc())
            return TradingDecision('HOLD', 0.0, 0.0, {'error': str(e)}, datetime.now())

    def _log_consciousness_dashboard(self, product_id: str, decision: TradingDecision, rt_result: Optional[Dict[str, Any]]):
        """Displays the bot's 'Inner Thoughts' and ML consensus in a rich format"""
        self.logger.info(f"\n" + "="*60 + f"\n🧠 CONSCIOUSNESS DASHBOARD: {product_id}\n" + "="*60)
        
        # 1. Decision Summary
        action_emoji = "🚀" if decision.action == "BUY" else "🔻" if decision.action == "SELL" else "⚖️"
        mode = decision.reasoning.get('execution_mode', 'TAKER')
        self.logger.info(f"ACTION: {action_emoji} {decision.action} | MODE: {mode} | CONFIDENCE: {decision.confidence:.2%} | SIZE: {decision.position_size:.2%}")
        
        # 2. Market Regime & Global Intelligence
        regime = self.regime_overlay.get_current_regime() if hasattr(self.regime_overlay, 'get_current_regime') else "NORMAL"
        boost = self.regime_overlay.get_confidence_boost()
        self.logger.info(f"MARKET REGIME: {regime} | REGIME BOOST: {boost:+.4f}")
        
        # Whale & Lead-Lag Signals
        whale = decision.reasoning.get('whale_signals', {})
        w_pressure = whale.get('whale_pressure', 0.0)
        w_count = whale.get('whale_count', 0)
        w_emoji = "🐋" if abs(w_pressure) > 0.1 else "🌊"
        
        lead_lag = decision.reasoning.get('lead_lag_alpha', {})
        corr = lead_lag.get('correlation', 0.0)
        lag = lead_lag.get('lag_periods', 0)
        ll_emoji = "🔗" if abs(corr) > 0.7 else "⛓️"
        
        self.logger.info(f"WHALE PRESSURE: {w_emoji} {w_pressure:+.4f} ({w_count} alerts) | LEAD-LAG: {ll_emoji} Corr:{corr:.2f} Lag:{lag}")
        
        # 2.5 Market Microstructure & Fractal Intelligence
        ms_metrics = self.microstructure_engine.get_latest_metrics()
        vpin = ms_metrics.vpin if ms_metrics else 0.5
        v_emoji = "⚠️" if vpin > 0.7 else "⚖️"
        
        tech_signals = self._get_tech(product_id).get_latest_signals()
        hurst = tech_signals.hurst_exponent if tech_signals else 0.5
        h_emoji = "📈" if hurst > 0.6 else "📉" if hurst < 0.4 else "↔️"
        h_status = "Trending" if hurst > 0.6 else "Mean-Rev" if hurst < 0.4 else "Random"
        
        self.logger.info(f"VPIN TOXICITY: {v_emoji} {vpin:.4f} | HURST EXP: {h_emoji} {hurst:.4f} ({h_status})")
        
        # 2.6 Statistical Arbitrage Signal
        stat_arb = decision.reasoning.get('stat_arb', {})
        sa_signal = stat_arb.get('signal', 0.0)
        sa_z = stat_arb.get('z_score', 0.0)
        sa_emoji = "🎯" if abs(sa_signal) > 0.3 else "⚖️"
        self.logger.info(f"STAT ARB SIGNAL: {sa_emoji} {sa_signal:+.4f} (Z-Score: {sa_z:+.2f})")
        
        # 2.7 Volume Profile Signal
        vp_signal = decision.reasoning.get('volume_profile_signal', 0.0)
        vp_status = decision.reasoning.get('volume_profile_status', 'Unknown')
        vp_emoji = "📊" if abs(vp_signal) > 0.3 else "⚖️"
        self.logger.info(f"VOLUME PROFILE: {vp_emoji} {vp_signal:+.4f} ({vp_status})")
        
        # 2.8 High-Dimensional Intelligence (Fractal, Entropy, Quantum)
        fractal = decision.reasoning.get('fractal_intelligence', {})
        f_pattern = fractal.get('best_pattern', 'None')
        f_sim = fractal.get('similarity', 0.0)
        f_emoji = "🧬" if f_sim > 0.7 else "🧩"
        
        entropy = decision.reasoning.get('market_entropy', {})
        e_pred = entropy.get('predictability', 0.5)
        e_emoji = "🔮" if e_pred > 0.7 else "🌀"
        
        quantum = decision.reasoning.get('quantum_oscillator', {})
        q_state = quantum.get('current_energy_state', 0)
        q_prob = quantum.get('tunneling_probability', 0.0)
        q_emoji = "⚛️" if q_prob > 0.8 else "🔋"
        
        self.logger.info(f"FRACTAL PATTERN: {f_emoji} {f_pattern} ({f_sim:.2%}) | ENTROPY PRED: {e_emoji} {e_pred:.4f}")
        self.logger.info(f"QUANTUM STATE: {q_emoji} Level {q_state} | TUNNELING PROB: {q_prob:.2%}")
        
        # 3. ML Consensus (Step 12 Feature Fan-Out)
        if rt_result and 'predictions' in rt_result:
            self.logger.info("-"*60 + "\n🤖 ML FEATURE FAN-OUT CONSENSUS\n" + "-"*60)
            preds = rt_result['predictions']
            for model, val in preds.items():
                m_emoji = "📈" if val > 0.1 else "📉" if val < -0.1 else "↔️"
                self.logger.info(f"   {model:20} : {m_emoji} {val:+.4f}")
            
            # Aggregate Consensus
            model_values = list(preds.values())
            consensus = sum(model_values) / len(model_values) if model_values else 0
            c_emoji = "🔥" if abs(consensus) > 0.5 else "✅"
            self.logger.info(f"AGGREGATE CONSENSUS: {c_emoji} {consensus:+.4f}")
        
        # 4. Step 9 Risk Check
        risk_check = decision.reasoning.get('risk_check', {})
        self.logger.info("-"*60 + f"\n🛡️ RISK GATEWAY STATUS: {'ALLOWED' if decision.action != 'HOLD' or decision.reasoning.get('weighted_signal', 0) < 0.1 else 'BLOCKED'}\n" + "-"*60)
        self.logger.info(f"Daily PnL: ${risk_check.get('daily_pnl', 0):.2f} / Limit: ${risk_check.get('daily_limit', 0):.2f}")
        self.logger.info(
            f"Drawdown: {self._current_drawdown_pct:.1%} from HWM ${self._high_watermark_usd:,.2f} | "
            f"Weekly PnL: ${self._weekly_pnl:,.2f}"
        )
        
        # 5. Persistence & Attribution (Step 13)
        if self.db_enabled:
            self.logger.info("-" * 60 + "\n💾 PERSISTENCE & ANALYTICS\n" + "-" * 60)
            self.logger.info(f"Database: {self.db_manager.db_path} | STATUS: ACTIVE")
            self.logger.info(f"Historical Decisions: {len(self.decision_history)}")

        # 6. Global Breakout Intelligence
        if self.breakout_candidates:
            self.logger.info("-" * 60 + "\n🚀 GLOBAL BREAKOUT INTELLIGENCE\n" + "-" * 60)
            for r in self.breakout_candidates[:5]:
                b_emoji = "🔥" if r['breakout_score'] >= 80 else "✨"
                self.logger.info(f"   {r['symbol']:15} : {b_emoji} Score {r['breakout_score']} | Vol Surge: {r['volume_surge']:.2f}x | {r['exchange']}")

        self.logger.info("="*60 + "\n")

    # DEPRECATED: Old breakout scan replaced by Phase 0 breakout_scanner in execute_trading_cycle
    # async def _run_breakout_scan(self):
    #     """Step 16+: Renaissance Global Scanner for breakout opportunities"""
    #     pass

    # ──────────────────────────────────────────────
    #  Kill Switch
    # ──────────────────────────────────────────────
    KILL_FILE = Path("KILL_SWITCH")

    def trigger_kill_switch(self, reason: str):
        """Activate kill switch: close all positions, halt trading loop."""
        self.logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        self._killed = True
        try:
            self.position_manager.set_emergency_stop(True, reason)
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
        # Fire alert asynchronously (best-effort)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self.alert_manager.send_alert("CRITICAL", "Kill Switch", reason)
                )
        except Exception:
            pass

    def _check_kill_file(self):
        """Check for file-based kill switch (touch KILL_SWITCH to halt)."""
        if self.KILL_FILE.exists():
            reason = self.KILL_FILE.read_text().strip() or "Kill file detected"
            self.trigger_kill_switch(reason)

    # ──────────────────────────────────────────────
    #  WebSocket Feed
    # ──────────────────────────────────────────────
    async def _run_websocket_feed(self):
        """Background WebSocket feed for real-time market data."""
        while not self._killed:
            try:
                await self._ws_client.connect_websocket()
                await self._ws_client.listen_for_messages(self._ws_queue)
            except Exception as e:
                self.logger.warning(f"WebSocket reconnecting: {e}")
                await asyncio.sleep(5)

    # ──────────────────────────────────────────────
    #  Multi-Exchange Arbitrage Engine
    # ──────────────────────────────────────────────
    async def _run_arbitrage_engine(self):
        """Run the arbitrage engine as a background task."""
        try:
            self.logger.info("Arbitrage engine starting...")
            await self.arbitrage_orchestrator.start()
        except asyncio.CancelledError:
            self.logger.info("Arbitrage engine cancelled — shutting down")
            await self.arbitrage_orchestrator.stop()
        except Exception as e:
            self.logger.error(f"Arbitrage engine error: {e}")
            try:
                await self.arbitrage_orchestrator.stop()
            except Exception:
                pass

    # ──────────────────────────────────────────────
    #  Liquidation Cascade Detector (Module D)
    # ──────────────────────────────────────────────
    async def _run_liquidation_detector(self):
        """Run the liquidation cascade detector as a background task."""
        try:
            self.logger.info("Liquidation cascade detector starting...")
            await self.liquidation_detector.start()
        except asyncio.CancelledError:
            self.logger.info("Liquidation detector cancelled — shutting down")
            await self.liquidation_detector.stop()
        except Exception as e:
            self.logger.error(f"Liquidation detector error: {e}")
            try:
                await self.liquidation_detector.stop()
            except Exception:
                pass

    # ──────────────────────────────────────────────
    #  Fast Mean Reversion Scanner
    # ──────────────────────────────────────────────
    async def _run_fast_reversion_scanner(self):
        """Run the fast mean reversion scanner as a background task."""
        try:
            await self.fast_reversion_scanner.run_loop()
        except asyncio.CancelledError:
            self.fast_reversion_scanner.stop()
        except Exception as e:
            self.logger.error(f"Fast reversion scanner error: {e}")

    # ──────────────────────────────────────────────
    #  Heartbeat Writer (Multi-Bot Coordination)
    # ──────────────────────────────────────────────
    async def _run_heartbeat_writer(self, interval: float = 5.0):
        """Run the heartbeat writer as a background task."""
        try:
            await self.heartbeat_writer.start(self, interval=interval)
        except asyncio.CancelledError:
            self.heartbeat_writer.stop()
        except Exception as e:
            self.logger.error(f"Heartbeat writer error: {e}")

    # ──────────────────────────────────────────────
    #  Phase 2 Observation Loops
    # ──────────────────────────────────────────────

    async def _run_portfolio_drift_logger(self):
        """Log target vs actual portfolio drift every 60s (observation mode — no corrections)."""
        try:
            while not self._killed:
                try:
                    engine = self.medallion_portfolio_engine
                    drift = engine.compute_drift()
                    if drift:
                        pairs = ", ".join(f"{p}={d:+.0f}$" for p, d in drift.items())
                        self.logger.info(f"PORTFOLIO DRIFT (obs): {pairs}")
                except Exception as e:
                    self.logger.debug(f"Portfolio drift logger error: {e}")
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass

    async def _run_insurance_scanner_loop(self):
        """Scan for insurance premiums every 30 minutes (observation mode)."""
        try:
            while not self._killed:
                for pair in self.product_ids[:3]:  # Top 3 products only
                    try:
                        result = self.insurance_scanner.get_all_premiums(pair)
                        if result.get("any_premium_detected"):
                            count = result.get("total_premiums_found", 0)
                            rec = result.get("combined_recommendation", "none")
                            self.logger.info(
                                f"INSURANCE PREMIUM (obs): {pair} — {count} premiums detected, rec={rec}"
                            )
                    except Exception as e:
                        self.logger.debug(f"Insurance scanner error for {pair}: {e}")
                await asyncio.sleep(1800)  # 30 minutes
        except asyncio.CancelledError:
            pass

    async def _run_daily_signal_review_loop(self):
        """Run daily signal P&L review at midnight UTC."""
        try:
            while not self._killed:
                now = datetime.now(timezone.utc)
                # Sleep until next midnight UTC
                tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
                wait_seconds = (tomorrow - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                if self._killed:
                    break
                try:
                    summary = self.daily_signal_review.update_daily()
                    if summary:
                        for sig_type, stats in summary.items():
                            status = stats.get("status", "active")
                            pnl = stats.get("pnl", 0)
                            self.logger.info(
                                f"DAILY SIGNAL REVIEW: {sig_type} — P&L=${pnl:.2f}, status={status}"
                            )
                except Exception as e:
                    self.logger.error(f"Daily signal review error: {e}")
        except asyncio.CancelledError:
            pass

    # ──────────────────────────────────────────────
    #  Phase 2 Monitor Loops (BUG 6 fix — orphaned monitors)
    # ──────────────────────────────────────────────

    async def _run_beta_monitor_loop(self):
        """Periodic beta computation (every 60 min, observation mode)."""
        try:
            while not self._killed:
                try:
                    report = self.beta_monitor.get_report()
                    beta = report.get("current_beta", 0.0)
                    status = report.get("current_status", "ok")
                    trend = report.get("trend", "unknown")
                    self.logger.info(
                        f"BETA MONITOR (obs): beta={beta:+.4f} status={status} trend={trend}"
                    )
                    if self.beta_monitor.should_alert() and self.monitoring_alert_manager:
                        hedge = self.beta_monitor.get_hedge_recommendation()
                        self._track_task(self.monitoring_alert_manager.send_warning(
                            f"Beta alert: {hedge.get('rationale', 'high beta deviation')}"
                        ))
                except Exception as e:
                    self.logger.debug(f"Beta monitor loop error: {e}")
                await asyncio.sleep(3600)  # 60 minutes
        except asyncio.CancelledError:
            pass

    async def _run_sharpe_monitor_loop(self):
        """Periodic Sharpe health check (every 60 min, observation mode)."""
        try:
            while not self._killed:
                try:
                    report = self.sharpe_monitor_medallion.get_report()
                    s7 = report.get("sharpe_7d", 0.0)
                    s30 = report.get("sharpe_30d", 0.0)
                    status = report.get("status", "unknown")
                    mult = report.get("exposure_multiplier", 1.0)
                    self.logger.info(
                        f"SHARPE MONITOR (obs): 7d={s7:.2f} 30d={s30:.2f} "
                        f"status={status} exposure_mult={mult:.2f}"
                    )
                except Exception as e:
                    self.logger.debug(f"Sharpe monitor loop error: {e}")
                await asyncio.sleep(3600)  # 60 minutes
        except asyncio.CancelledError:
            pass

    async def _run_capacity_monitor_loop(self):
        """Periodic capacity analysis (every 60 min, observation mode)."""
        try:
            while not self._killed:
                try:
                    caps = self.capacity_monitor.get_all_capacities()
                    constrained = [p for p, r in caps.items() if r.get("capacity_status") == "constrained"]
                    warning = [p for p, r in caps.items() if r.get("capacity_status") == "warning"]
                    self.logger.info(
                        f"CAPACITY MONITOR (obs): {len(caps)} pairs analysed, "
                        f"{len(constrained)} constrained, {len(warning)} warning"
                    )
                    if constrained and self.monitoring_alert_manager:
                        self._track_task(self.monitoring_alert_manager.send_warning(
                            f"Capacity constrained pairs: {', '.join(constrained)}"
                        ))
                except Exception as e:
                    self.logger.debug(f"Capacity monitor loop error: {e}")
                await asyncio.sleep(3600)  # 60 minutes
        except asyncio.CancelledError:
            pass

    async def _run_regime_detector_loop(self):
        """Periodic regime retraining + prediction (every 5 min, observation mode)."""
        try:
            while not self._killed:
                try:
                    if self.medallion_regime.needs_retrain():
                        trained = self.medallion_regime.train()
                        if trained:
                            self.medallion_regime.save_model()
                            self.logger.info("REGIME DETECTOR (obs): Model retrained and saved")
                    pred = self.medallion_regime.predict_current_regime()
                    regime = pred.get("regime_name", "unknown")
                    conf = pred.get("confidence", 0.0)
                    self.logger.info(
                        f"REGIME DETECTOR (obs): regime={regime} confidence={conf:.2f}"
                    )
                except Exception as e:
                    self.logger.debug(f"Regime detector loop error: {e}")
                await asyncio.sleep(300)  # 5 minutes
        except asyncio.CancelledError:
            pass

    # ──────────────────────────────────────────────
    #  Unified Telegram Reporting (Gap 5 fix)
    # ──────────────────────────────────────────────

    async def _run_telegram_report_loop(self):
        """Send a consolidated hourly status report via Telegram."""
        try:
            await asyncio.sleep(300)  # Wait 5 min after startup before first report
            while not self._killed:
                try:
                    stats = {
                        "uptime": str(datetime.now(timezone.utc) - self._start_time).split('.')[0],
                        "trades_1h": 0,
                        "pnl_1h": 0.0,
                        "open_positions": len(self.position_manager.positions),
                        "exchanges_healthy": "coinbase",
                    }
                    # Count recent trades from DB
                    if self.db_enabled:
                        try:
                            import sqlite3
                            cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
                            conn = sqlite3.connect(self.db_path, timeout=5.0)
                            row = conn.execute(
                                "SELECT COUNT(*), COALESCE(SUM(CASE WHEN UPPER(side)='SELL' "
                                "THEN size*price WHEN UPPER(side)='BUY' THEN -size*price ELSE 0 END), 0) "
                                "FROM trades WHERE timestamp >= ? AND status != 'FAILED'",
                                (cutoff,)
                            ).fetchone()
                            conn.close()
                            if row:
                                stats["trades_1h"] = row[0] or 0
                                stats["pnl_1h"] = float(row[1] or 0)
                        except Exception:
                            pass

                    await self.monitoring_alert_manager._telegram.send_hourly_heartbeat(stats)
                except Exception as e:
                    self.logger.debug(f"Telegram report loop error: {e}")
                await asyncio.sleep(3600)  # Every hour
        except asyncio.CancelledError:
            pass

    # ──────────────────────────────────────────────
    #  State Recovery
    # ──────────────────────────────────────────────
    async def _restore_state(self):
        """Restore positions and daily PnL from the database after restart."""
        try:
            # Restore open positions
            open_positions = await self.db_manager.get_open_positions()
            restored = 0
            net_position = 0.0
            for row in open_positions:
                from position_manager import Position, PositionSide, PositionStatus
                pos = Position(
                    position_id=row['position_id'],
                    product_id=row['product_id'],
                    side=PositionSide(row['side']),
                    size=row['size'],
                    entry_price=row['entry_price'],
                    current_price=row['entry_price'],
                    stop_loss_price=row.get('stop_loss_price'),
                    take_profit_price=row.get('take_profit_price'),
                    status=PositionStatus.OPEN,
                    entry_time=datetime.fromisoformat(row['opened_at']),
                )
                self.position_manager.positions[pos.position_id] = pos
                sign = 1.0 if pos.side == PositionSide.LONG else -1.0
                net_position += sign * pos.size
                restored += 1

            # Restore daily PnL from today's trades
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            daily_pnl = await self.db_manager.get_daily_pnl(today)
            self.position_manager.daily_pnl = daily_pnl
            self.daily_pnl = daily_pnl
            self.current_position = net_position

            if restored > 0 or daily_pnl != 0:
                self.logger.info(
                    f"State restored: {restored} open positions, "
                    f"net_position={net_position:.6f}, daily_pnl=${daily_pnl:.2f}"
                )

            # Reconcile with exchange after restoring state (skip in paper trading)
            if not self.paper_trading:
                recon = self.position_manager.reconcile_with_exchange()
                if recon.get("status") == "MISMATCH":
                    asyncio.ensure_future(
                        self.alert_manager.send_alert("CRITICAL", "Position Mismatch",
                            f"{len(recon['discrepancies'])} discrepancies found on startup")
                    )
        except Exception as e:
            self.logger.warning(f"State recovery skipped: {e}")

    async def _deduplicate_positions_on_startup(self) -> None:
        """Close duplicate and opposing positions found after DB restore.

        Rules:
        1. If multiple same-side positions exist for a product, keep the newest, close the rest.
        2. If opposing positions exist for a product (LONG + SHORT), close both and go flat.
        """
        from position_manager import PositionSide, PositionStatus
        from collections import defaultdict

        # Group open positions by product_id
        by_product: Dict[str, List] = defaultdict(list)
        with self.position_manager._lock:
            for pos in list(self.position_manager.positions.values()):
                if pos.status == PositionStatus.OPEN:
                    by_product[pos.product_id].append(pos)

        closed_count = 0
        for product_id, positions in by_product.items():
            longs = [p for p in positions if p.side == PositionSide.LONG]
            shorts = [p for p in positions if p.side == PositionSide.SHORT]

            # Rule 2: Opposing positions — close ALL (go flat)
            if longs and shorts:
                self.logger.warning(
                    f"STARTUP DEDUP: {product_id} has {len(longs)} LONG + {len(shorts)} SHORT — closing all (go flat)"
                )
                for pos in longs + shorts:
                    ok, msg = self.position_manager.close_position(
                        pos.position_id, reason="startup_dedup_opposing"
                    )
                    if ok:
                        self._track_task(
                            self.db_manager.close_position_record(
                                pos.position_id,
                                close_price=float(pos.current_price),
                                realized_pnl=0.0,
                                exit_reason="startup_dedup_opposing",
                            )
                        )
                        closed_count += 1
                continue

            # Rule 1: Duplicate same-side — keep newest, close rest
            for group in [longs, shorts]:
                if len(group) > 1:
                    # Sort by entry_time descending (newest first)
                    group.sort(key=lambda p: p.entry_time, reverse=True)
                    keep = group[0]
                    dupes = group[1:]
                    self.logger.warning(
                        f"STARTUP DEDUP: {product_id} has {len(group)} {keep.side.value} positions — "
                        f"keeping {keep.position_id}, closing {len(dupes)} duplicates"
                    )
                    for pos in dupes:
                        ok, msg = self.position_manager.close_position(
                            pos.position_id, reason="startup_dedup_duplicate"
                        )
                        if ok:
                            self._track_task(
                                self.db_manager.close_position_record(
                                    pos.position_id,
                                    close_price=float(pos.current_price),
                                    realized_pnl=0.0,
                                    exit_reason="startup_dedup_duplicate",
                                )
                            )
                            closed_count += 1

        if closed_count > 0:
            self.logger.info(f"STARTUP DEDUP: closed {closed_count} duplicate/opposing positions")

    async def run_continuous_trading(self, cycle_interval: int = 300):
        """Run continuous Renaissance trading (default 5-minute cycles)"""
        self.logger.info(f"Starting continuous Renaissance trading with {cycle_interval}s cycles")

        # ── Module A: Graceful Shutdown Handler ──
        if self.state_manager and RECOVERY_AVAILABLE:
            try:
                loop = asyncio.get_event_loop()
                self.shutdown_handler = GracefulShutdownHandler(
                    state_manager=self.state_manager,
                    coinbase_client=self.coinbase_client,
                    alert_manager=self.monitoring_alert_manager,
                    drain_timeout_seconds=30.0,
                )
                self.shutdown_handler.install(loop=loop)
                await self.state_manager.aset_system_state(SystemState.STARTING, "bot starting")
                self.logger.info("Graceful shutdown handlers installed")
            except Exception as e:
                self.logger.warning(f"Graceful shutdown setup failed: {e}")
                # Fallback to basic signal handlers
                def _handle_shutdown(signum, frame):
                    self.trigger_kill_switch(f"Signal {signum} received")
                signal.signal(signal.SIGINT, _handle_shutdown)
                signal.signal(signal.SIGTERM, _handle_shutdown)
        else:
            # Fallback signal handlers when recovery module is not available
            def _handle_shutdown(signum, frame):
                self.trigger_kill_switch(f"Signal {signum} received")
            signal.signal(signal.SIGINT, _handle_shutdown)
            signal.signal(signal.SIGTERM, _handle_shutdown)

        # Restore positions from DB so anti-stacking logic works across restarts.
        # In paper mode: restore positions but reset daily PnL (balances reset each start).
        paper_mode = self.config.get("trading", {}).get("paper_trading", True)
        if not paper_mode:
            await self._restore_state()
        else:
            # Restore positions only (for anti-stacking), reset PnL
            try:
                open_positions = await self.db_manager.get_open_positions()
                restored = 0
                for row in open_positions:
                    from position_manager import Position, PositionSide, PositionStatus
                    pos = Position(
                        position_id=row['position_id'],
                        product_id=row['product_id'],
                        side=PositionSide(row['side']),
                        size=row['size'],
                        entry_price=row['entry_price'],
                        current_price=row['entry_price'],
                        stop_loss_price=row.get('stop_loss_price'),
                        take_profit_price=row.get('take_profit_price'),
                        status=PositionStatus.OPEN,
                        entry_time=datetime.fromisoformat(row['opened_at']),
                    )
                    self.position_manager.positions[pos.position_id] = pos
                    restored += 1
                self.logger.info(f"Paper mode: restored {restored} positions from DB (anti-stacking)")
            except Exception as e:
                self.logger.warning(f"Paper mode position restore failed: {e}")
            self.position_manager.daily_pnl = 0.0
            self.daily_pnl = 0.0

        # ── Startup deduplication: close duplicate/opposing positions from DB ──
        await self._deduplicate_positions_on_startup()

        # ── Module A: Set RUNNING state ──
        if self.state_manager:
            try:
                await self.state_manager.aset_system_state(SystemState.RUNNING, "trading loop started")
            except Exception:
                pass

        # ── Module C: Startup alert ──
        if self.monitoring_alert_manager:
            try:
                await self.monitoring_alert_manager.send_system_event(
                    "Bot Started",
                    f"Renaissance bot starting with {len(self.product_ids)} products, "
                    f"{'paper' if paper_mode else 'live'} mode"
                )
            except Exception:
                pass

        # ── Build dynamic trading universe from Binance ──
        self.logger.info("Building dynamic trading universe from Binance...")
        await self._build_and_apply_universe()
        if self._universe_built:
            # Re-send startup alert with actual pair count
            if self.monitoring_alert_manager:
                try:
                    await self.monitoring_alert_manager.send_system_event(
                        "Universe Built",
                        f"Dynamic universe: {len(self.product_ids)} pairs from Binance"
                    )
                except Exception:
                    pass

        # Start real-time pipeline if enabled
        if self.real_time_pipeline.enabled:
            await self.real_time_pipeline.start()

        # Start Ghost Runner Loop (Step 18)
        self._track_task(self.ghost_runner.start_ghost_loop(interval=cycle_interval * 2))

        # Start WebSocket feed for real-time data
        if self._ws_client:
            self._track_task(self._run_websocket_feed())

        # Start Multi-Exchange Arbitrage Engine (runs independently alongside main loop)
        if self.arbitrage_orchestrator:
            self.logger.info("Launching arbitrage engine...")
            self._track_task(self._run_arbitrage_engine())

        # ── Module D: Start Liquidation Cascade Detector ──
        if self.liquidation_detector:
            self.logger.info("Launching liquidation cascade detector...")
            self._track_task(self._run_liquidation_detector())

        # ── Fast Mean Reversion Scanner (1s eval) ──
        if self.fast_reversion_scanner:
            self.logger.info("Launching fast mean reversion scanner (1s eval)...")
            self._track_task(self._run_fast_reversion_scanner())

        # ── Heartbeat Writer (multi-bot coordination) ──
        if self.heartbeat_writer:
            hb_interval = self.config.get("orchestrator", {}).get(
                "heartbeat_interval_seconds", 5
            )
            self.logger.info(f"Launching heartbeat writer (every {hb_interval}s)...")
            self._track_task(self._run_heartbeat_writer(hb_interval))

        # ── Phase 2 Observation Loops ──
        if self.medallion_portfolio_engine:
            self.logger.info("Launching medallion portfolio drift logger (observation mode)...")
            self._track_task(self._run_portfolio_drift_logger())

        if self.insurance_scanner:
            self.logger.info("Launching insurance premium scanner (every 30 min)...")
            self._track_task(self._run_insurance_scanner_loop())

        if self.daily_signal_review:
            self.logger.info("Launching daily signal review (midnight UTC)...")
            self._track_task(self._run_daily_signal_review_loop())

        # ── Phase 2 Monitor Loops (BUG 6 fix) ──
        if self.beta_monitor:
            self.logger.info("Launching beta monitor loop (every 60 min, observation mode)...")
            self._track_task(self._run_beta_monitor_loop())

        if self.sharpe_monitor_medallion:
            self.logger.info("Launching sharpe monitor loop (every 60 min, observation mode)...")
            self._track_task(self._run_sharpe_monitor_loop())

        if self.capacity_monitor:
            self.logger.info("Launching capacity monitor loop (every 60 min, observation mode)...")
            self._track_task(self._run_capacity_monitor_loop())

        if self.medallion_regime:
            self.logger.info("Launching regime detector loop (every 5 min, observation mode)...")
            self._track_task(self._run_regime_detector_loop())

        # ── Doc 15: Agent weekly research loop + deployment loop ──
        if self.agent_coordinator:
            self.logger.info("Launching agent weekly research check loop...")
            self._track_task(self.agent_coordinator.run_weekly_check_loop())
            self.logger.info("Launching agent deployment loop...")
            self._track_task(self.agent_coordinator.run_deployment_loop())

        # ── Gap 5 fix: Unified Telegram Reporting ──
        if self.monitoring_alert_manager:
            self.logger.info("Launching unified Telegram hourly report loop...")
            self._track_task(self._run_telegram_report_loop())

        while not self._killed:
            try:
                # Check file-based kill switch
                self._check_kill_file()
                if self._killed:
                    break

                # Execute trading cycle
                decision = await self.execute_trading_cycle()

                self.logger.info(f"{'LIVE' if not self.coinbase_client.paper_trading else 'PAPER'} TRADE: "
                               f"{decision.action} - "
                               f"Confidence: {decision.confidence:.3f} - "
                               f"Position Size: {decision.position_size:.3f}")

                # Write heartbeat after each successful cycle
                self._write_heartbeat()
                # Recovery module heartbeat (file-based for watchdog)
                if self.state_manager:
                    try:
                        await self.state_manager.asend_heartbeat()
                    except Exception:
                        pass

                # Wait for next cycle
                await asyncio.sleep(cycle_interval)

            except KeyboardInterrupt:
                self.trigger_kill_switch("KeyboardInterrupt")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in trading loop: {e}")
                await asyncio.sleep(60)

        self.logger.info("Trading loop exited. Shutting down background tasks...")
        await self._shutdown()

    def _compute_adaptive_weights(self, product_id: str, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adaptive Weight Engine — Renaissance-style Bayesian signal weight updating.

        Uses the signal scorecard (measured accuracy per signal) to adjust weights:
        - Signals with >55% accuracy get upweighted
        - Signals with <48% accuracy get downweighted
        - Signals with too few samples keep config weights (prior)
        - Blend between config (prior) and measured (posterior) ramps up as data accumulates

        Returns adjusted weights dict (does NOT mutate self.signal_weights).
        """
        sc = self._signal_scorecard.get(product_id, {})
        if not sc:
            return base_weights

        # Aggregate across all products for more data
        agg_sc: Dict[str, Dict[str, int]] = {}
        for pid, signals in self._signal_scorecard.items():
            for sig_name, stats in signals.items():
                entry = agg_sc.setdefault(sig_name, {"correct": 0, "total": 0})
                entry["correct"] += stats["correct"]
                entry["total"] += stats["total"]

        # Find signals with enough data
        eligible = {}
        max_total = 0
        for sig_name, stats in agg_sc.items():
            if stats["total"] >= self._adaptive_min_samples:
                accuracy = stats["correct"] / stats["total"]
                eligible[sig_name] = accuracy
                max_total = max(max_total, stats["total"])

        if not eligible:
            return base_weights

        # Ramp blend factor: 0 at min_samples, 0.5 at 100+ samples
        blend = min(0.5, (max_total - self._adaptive_min_samples) / 170.0)
        self._adaptive_weight_blend = blend

        # Compute accuracy-derived weights
        # Transform accuracy to weight multiplier:
        # 50% (random) → 0.5x, 55% → 1.0x, 60% → 1.5x, 65%+ → 2.0x
        # <48% (anti-predictive) → 0.1x
        multipliers = {}
        for sig_name in base_weights:
            if sig_name in eligible:
                acc = eligible[sig_name]
                if acc < 0.48:
                    multipliers[sig_name] = 0.1  # actively wrong — near zero
                elif acc < 0.52:
                    multipliers[sig_name] = 0.5  # noise
                elif acc < 0.55:
                    multipliers[sig_name] = 0.8  # weak
                elif acc < 0.60:
                    multipliers[sig_name] = 1.2  # good
                elif acc < 0.65:
                    multipliers[sig_name] = 1.5  # strong
                else:
                    multipliers[sig_name] = 2.0  # excellent
            else:
                multipliers[sig_name] = 1.0  # no data → keep as-is

        # Blend: final = (1 - blend) * config_weight + blend * (config_weight * multiplier)
        # Simplifies to: final = config_weight * (1 - blend + blend * multiplier)
        adapted = {}
        for sig_name, w in base_weights.items():
            m = multipliers.get(sig_name, 1.0)
            adapted[sig_name] = w * (1.0 - blend + blend * m)

        # Renormalize so weights sum to 1.0
        total = sum(adapted.values())
        if total > 0:
            adapted = {k: v / total for k, v in adapted.items()}

        return adapted

    def _get_measured_edge(self, product_id: str) -> Optional[float]:
        """
        Compute realized edge from signal scorecard instead of fabricating it.
        Returns None if insufficient data, else a float [0, 0.15].
        """
        sc = self._signal_scorecard.get(product_id, {})
        if not sc:
            return None
        total_correct = sum(s["correct"] for s in sc.values())
        total_total = sum(s["total"] for s in sc.values())
        if total_total < 20:
            return None  # not enough data
        # Aggregate accuracy across all signals
        accuracy = total_correct / total_total
        # Edge = accuracy - 0.5 (above random), capped at 0.15
        edge = max(0.0, accuracy - 0.5)
        return min(edge, 0.15)

    def _update_dynamic_thresholds(self, product_id: str, market_data: Dict[str, Any]):
        """Adjusts BUY/SELL thresholds based on volatility and confidence (Step 8)"""
        if not self.adaptive_thresholds:
            return

        try:
            # Use technical indicators volatility regime
            latest_tech = self._get_tech(product_id).get_latest_signals()
            vol_regime = latest_tech.volatility_regime if latest_tech else None
            
            # Base thresholds — from config (default 0.06 after backtest analysis).
            # Backtest proved: only |prediction| > 0.06 has >53% accuracy.
            base_buy = float(self.config.get('trading', {}).get('buy_threshold', 0.06))
            base_sell = float(self.config.get('trading', {}).get('sell_threshold', -0.06))
            self.buy_threshold = base_buy
            self.sell_threshold = base_sell

            # Adjust based on volatility (scale from higher base)
            if vol_regime == "high_volatility" or vol_regime == "extreme_volatility":
                # Increase thresholds in high volatility to avoid fakeouts
                self.buy_threshold = base_buy * 1.5
                self.sell_threshold = base_sell * 1.5
            elif vol_regime == "low_volatility":
                # Decrease thresholds in low volatility to catch smaller moves
                self.buy_threshold = base_buy * 0.7
                self.sell_threshold = base_sell * 0.7
                
            self.logger.info(f"Dynamic Thresholds updated: Buy {self.buy_threshold:.2f}, Sell {self.sell_threshold:.2f} (Regime: {vol_regime})")
        except Exception as e:
            self.logger.error(f"Failed to update dynamic thresholds: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the Renaissance bot"""
        if not self.decision_history:
            return {"message": "No trading decisions yet"}

        recent_decisions = self.decision_history[-20:]  # Last 20 decisions

        summary = {
            'total_decisions': len(self.decision_history),
            'recent_decisions': len(recent_decisions),
            'action_distribution': {},
            'average_confidence': 0.0,
            'average_position_size': 0.0,
            'signal_weight_distribution': self.signal_weights
        }

        # Calculate distributions
        actions = [d.action for d in recent_decisions]
        confidences = [d.confidence for d in recent_decisions]
        position_sizes = [d.position_size for d in recent_decisions]

        for action in ['BUY', 'SELL', 'HOLD']:
            summary['action_distribution'][action] = actions.count(action)

        if confidences:
            summary['average_confidence'] = np.mean(confidences)
        if position_sizes:
            summary['average_position_size'] = np.mean(position_sizes)

        return summary

class MicrostructureAnalyzer:
    """Microstructure analysis component"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, data: Dict) -> Dict[str, float]:
        return {'order_flow': 0.0, 'spread': 0.0, 'depth': 0.0}


class TechnicalAnalyzer:
    """Technical analysis component"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, data: Dict) -> Dict[str, float]:
        return {'rsi': 0.0, 'macd': 0.0, 'bollinger': 0.0}


class AlternativeDataAnalyzer:
    """Alternative data analysis component"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, data: Dict) -> Dict[str, float]:
        return {'sentiment': 0.0, 'social': 0.0, 'news': 0.0}

# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize Renaissance bot
        bot = RenaissanceTradingBot()

        # Run a few test cycles
        print("Running Renaissance Trading Bot Test...")
        for i in range(3):
            print(f"\n--- Cycle {i+1} ---")
            decision = await bot.execute_trading_cycle()
            print(f"Decision: {decision.action}")
            print(f"Confidence: {decision.confidence:.3f}")
            print(f"Position Size: {decision.position_size:.3f}")

            await asyncio.sleep(2)  # Short delay for testing

        # Show performance summary
        summary = bot.get_performance_summary()
        print(f"\nPerformance Summary: {json.dumps(summary, indent=2)}")

    # Run the test
    asyncio.run(main())
