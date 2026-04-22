"""
Enhanced Configuration Management System
Dynamic parameter management with regime-based switching and real-time adaptation.
Renaissance Technologies-style configuration management with safety controls.
"""

import json
import yaml
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import time
from pathlib import Path
import copy

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications for dynamic configuration"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS_RANGE = "sideways_range"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class ConfigurationPriority(Enum):
    """Configuration priority levels"""
    EMERGENCY = "emergency"
    SAFETY = "safety"
    REGIME = "regime"
    PERFORMANCE = "performance"
    DEFAULT = "default"


@dataclass
class SafetyLimits:
    """Safety limit parameters"""
    max_position_size: float = 1000.0
    max_daily_loss: float = 500.0
    max_drawdown: float = 0.15
    max_volatility_threshold: float = 0.08
    min_confidence_threshold: float = 0.5
    emergency_stop_loss: float = 0.20
    max_leverage: float = 1.0
    max_trades_per_hour: int = 60
    min_account_balance: float = 1000.0


@dataclass
class SignalWeights:
    """Dynamic signal weight configuration"""
    order_flow: float = 0.32
    order_book: float = 0.22
    volume_analysis: float = 0.16
    macd: float = 0.10
    rsi: float = 0.08
    bollinger: float = 0.07
    alternative_data: float = 0.05

    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = sum(asdict(self).values())
        if total > 0:
            for key in asdict(self):
                setattr(self, key, getattr(self, key) / total)


@dataclass
class TradingThresholds:
    """Trading decision thresholds"""
    buy_threshold: float = 0.65
    sell_threshold: float = -0.65
    confidence_threshold: float = 0.60
    strong_signal_threshold: float = 0.80
    very_strong_signal_threshold: float = 0.90
    position_size_multiplier: float = 1.0
    risk_adjustment_factor: float = 1.0


@dataclass
class RegimeConfiguration:
    """Configuration for specific market regime"""
    regime: MarketRegime
    signal_weights: SignalWeights
    trading_thresholds: TradingThresholds
    safety_limits: SafetyLimits
    execution_frequency: int = 60  # seconds
    lookback_period: int = 100  # bars
    description: str = ""
    active: bool = True


class EnhancedConfigManager:
    """
    Enhanced Configuration Management System
    Provides dynamic parameter management with regime-based switching
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.current_regime = MarketRegime.UNKNOWN
        self.active_config: Optional[RegimeConfiguration] = None
        self.regime_configs: Dict[MarketRegime, RegimeConfiguration] = {}
        self.safety_overrides: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}

        # Configuration monitoring
        self.config_lock = threading.Lock()
        self.last_config_update = datetime.now()
        self.config_version = 1
        self.config_history: List[Dict] = []

        # Performance tracking
        self.parameter_performance: Dict[str, Dict[str, float]] = {}
        self.regime_performance: Dict[MarketRegime, Dict[str, float]] = {}

        # Initialize default configurations
        self._initialize_default_configs()
        self._start_monitoring_thread()

        self.logger.info("✅ Enhanced Configuration Manager initialized")

    def _initialize_default_configs(self):
        """Initialize default regime-based configurations"""

        # Bull Market Configuration
        bull_config = RegimeConfiguration(
            regime=MarketRegime.BULL_TRENDING,
            signal_weights=SignalWeights(
                order_flow=0.35,
                order_book=0.25,
                volume_analysis=0.18,
                macd=0.08,
                rsi=0.06,
                bollinger=0.05,
                alternative_data=0.03
            ),
            trading_thresholds=TradingThresholds(
                buy_threshold=0.60,
                sell_threshold=-0.70,
                confidence_threshold=0.55,
                strong_signal_threshold=0.75,
                position_size_multiplier=1.2
            ),
            safety_limits=SafetyLimits(
                max_position_size=1200.0,
                max_daily_loss=600.0,
                max_drawdown=0.18
            ),
            description="Optimized for trending bull markets"
        )

        # Bear Market Configuration
        bear_config = RegimeConfiguration(
            regime=MarketRegime.BEAR_TRENDING,
            signal_weights=SignalWeights(
                order_flow=0.30,
                order_book=0.20,
                volume_analysis=0.15,
                macd=0.12,
                rsi=0.10,
                bollinger=0.08,
                alternative_data=0.05
            ),
            trading_thresholds=TradingThresholds(
                buy_threshold=0.70,
                sell_threshold=-0.60,
                confidence_threshold=0.65,
                strong_signal_threshold=0.85,
                position_size_multiplier=0.8
            ),
            safety_limits=SafetyLimits(
                max_position_size=800.0,
                max_daily_loss=400.0,
                max_drawdown=0.12
            ),
            description="Conservative approach for bear markets"
        )

        # High Volatility Configuration
        high_vol_config = RegimeConfiguration(
            regime=MarketRegime.HIGH_VOLATILITY,
            signal_weights=SignalWeights(
                order_flow=0.40,
                order_book=0.25,
                volume_analysis=0.20,
                macd=0.05,
                rsi=0.05,
                bollinger=0.03,
                alternative_data=0.02
            ),
            trading_thresholds=TradingThresholds(
                buy_threshold=0.75,
                sell_threshold=-0.75,
                confidence_threshold=0.70,
                strong_signal_threshold=0.90,
                position_size_multiplier=0.6
            ),
            safety_limits=SafetyLimits(
                max_position_size=600.0,
                max_daily_loss=300.0,
                max_drawdown=0.10,
                max_volatility_threshold=0.12
            ),
            description="High-frequency microstructure focus for volatile markets"
        )

        # Sideways Market Configuration
        sideways_config = RegimeConfiguration(
            regime=MarketRegime.SIDEWAYS_RANGE,
            signal_weights=SignalWeights(
                order_flow=0.25,
                order_book=0.15,
                volume_analysis=0.12,
                macd=0.15,
                rsi=0.15,
                bollinger=0.12,
                alternative_data=0.06
            ),
            trading_thresholds=TradingThresholds(
                buy_threshold=0.65,
                sell_threshold=-0.65,
                confidence_threshold=0.60,
                strong_signal_threshold=0.80,
                position_size_multiplier=1.0
            ),
            safety_limits=SafetyLimits(),
            description="Balanced approach for range-bound markets"
        )

        # Store configurations
        self.regime_configs = {
            MarketRegime.BULL_TRENDING: bull_config,
            MarketRegime.BEAR_TRENDING: bear_config,
            MarketRegime.HIGH_VOLATILITY: high_vol_config,
            MarketRegime.SIDEWAYS_RANGE: sideways_config
        }

        # Set default active configuration
        self.active_config = sideways_config
        self.current_regime = MarketRegime.SIDEWAYS_RANGE

        self.logger.info(f"Initialized {len(self.regime_configs)} regime configurations")

    def get_current_config(self) -> RegimeConfiguration:
        """Get the current active configuration"""
        with self.config_lock:
            return copy.deepcopy(self.active_config)

    def switch_regime(self, new_regime: MarketRegime, reason: str = "Manual switch") -> bool:
        """Switch to a different market regime configuration"""
        try:
            with self.config_lock:
                if new_regime not in self.regime_configs:
                    self.logger.warning(f"Regime {new_regime} not configured, staying with {self.current_regime}")
                    return False

                old_regime = self.current_regime
                old_config = self.active_config

                # Switch to new regime
                self.current_regime = new_regime
                self.active_config = self.regime_configs[new_regime]
                self.last_config_update = datetime.now()
                self.config_version += 1

                # Log the switch
                self.config_history.append({
                    'timestamp': self.last_config_update.isoformat(),
                    'old_regime': old_regime.value,
                    'new_regime': new_regime.value,
                    'reason': reason,
                    'version': self.config_version
                })

                self.logger.info(f"🔄 Regime switched: {old_regime.value} → {new_regime.value} ({reason})")
                self.logger.info(f"📊 New weights - Order Flow: {self.active_config.signal_weights.order_flow:.2%}")
                self.logger.info(
                    f"🎯 New thresholds - Buy: {self.active_config.trading_thresholds.buy_threshold}, Confidence: {self.active_config.trading_thresholds.confidence_threshold}")

                return True

        except Exception as e:
            self.logger.error(f"Error switching regime: {e}")
            return False

    def update_signal_weights(self, new_weights: Dict[str, float], reason: str = "Manual update") -> bool:
        """Update signal weights dynamically"""
        try:
            with self.config_lock:
                old_weights = asdict(self.active_config.signal_weights)

                # Update weights
                for signal, weight in new_weights.items():
                    if hasattr(self.active_config.signal_weights, signal):
                        setattr(self.active_config.signal_weights, signal, weight)

                # Normalize weights
                self.active_config.signal_weights.normalize()

                self.last_config_update = datetime.now()
                self.config_version += 1

                self.logger.info(f"🔧 Signal weights updated ({reason})")
                for signal, weight in new_weights.items():
                    old_weight = old_weights.get(signal, 0)
                    self.logger.info(f"   {signal}: {old_weight:.2%} → {weight:.2%}")

                return True

        except Exception as e:
            self.logger.error(f"Error updating signal weights: {e}")
            return False

    def update_safety_limits(self, new_limits: Dict[str, Any], emergency: bool = False) -> bool:
        """Update safety limits with optional emergency override"""
        try:
            with self.config_lock:
                if emergency:
                    self.safety_overrides.update(new_limits)
                    self.logger.warning(f"🚨 Emergency safety override applied: {new_limits}")
                else:
                    # Update normal safety limits
                    for param, value in new_limits.items():
                        if hasattr(self.active_config.safety_limits, param):
                            setattr(self.active_config.safety_limits, param, value)

                self.last_config_update = datetime.now()
                self.config_version += 1

                self.logger.info(f"🛡️ Safety limits updated: {new_limits}")
                return True

        except Exception as e:
            self.logger.error(f"Error updating safety limits: {e}")
            return False

    def get_effective_safety_limits(self) -> SafetyLimits:
        """Get effective safety limits, including any emergency overrides"""
        with self.config_lock:
            limits = copy.deepcopy(self.active_config.safety_limits)

            # Apply emergency overrides
            for param, value in self.safety_overrides.items():
                if hasattr(limits, param):
                    setattr(limits, param, value)

            return limits

    def clear_emergency_overrides(self) -> bool:
        """Clear all emergency safety overrides"""
        try:
            with self.config_lock:
                old_overrides = self.safety_overrides.copy()
                self.safety_overrides.clear()

                self.logger.info(f"✅ Emergency overrides cleared: {old_overrides}")
                return True

        except Exception as e:
            self.logger.error(f"Error clearing emergency overrides: {e}")
            return False

    def record_performance_metric(self, metric_name: str, value: float):
        """Record performance metric for current configuration"""
        with self.config_lock:
            self.performance_metrics[metric_name] = value

            # Track regime-specific performance
            if self.current_regime not in self.regime_performance:
                self.regime_performance[self.current_regime] = {}

            self.regime_performance[self.current_regime][metric_name] = value

    def get_regime_performance(self, regime: MarketRegime) -> Dict[str, float]:
        """Get performance metrics for a specific regime"""
        return self.regime_performance.get(regime, {})

    def save_configuration(self, filename: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        try:
            if filename is None:
                filename = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            config_data = {
                'current_regime': self.current_regime.value,
                'config_version': self.config_version,
                'last_update': self.last_config_update.isoformat(),
                'active_config': {
                    'regime': self.active_config.regime.value,
                    'signal_weights': asdict(self.active_config.signal_weights),
                    'trading_thresholds': asdict(self.active_config.trading_thresholds),
                    'safety_limits': asdict(self.active_config.safety_limits),
                    'execution_frequency': self.active_config.execution_frequency,
                    'lookback_period': self.active_config.lookback_period,
                    'description': self.active_config.description
                },
                'safety_overrides': self.safety_overrides,
                'performance_metrics': self.performance_metrics,
                'config_history': self.config_history[-10:]  # Last 10 changes
            }

            filepath = self.config_dir / filename
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.logger.info(f"📁 Configuration saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def load_configuration(self, filename: str) -> bool:
        """Load configuration from file"""
        try:
            filepath = self.config_dir / filename
            with open(filepath, 'r') as f:
                config_data = json.load(f)

            # Restore configuration
            regime = MarketRegime(config_data['current_regime'])
            if regime in self.regime_configs:
                self.switch_regime(regime, f"Loaded from {filename}")

            self.safety_overrides = config_data.get('safety_overrides', {})
            self.performance_metrics = config_data.get('performance_metrics', {})

            self.logger.info(f"📂 Configuration loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False

    def _start_monitoring_thread(self):
        """Start background monitoring thread"""

        def monitor():
            while True:
                try:
                    # Monitor configuration performance
                    self._evaluate_config_performance()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    self.logger.error(f"Configuration monitoring error: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def _evaluate_config_performance(self):
        """Evaluate current configuration performance"""
        try:
            with self.config_lock:
                # Calculate performance metrics
                if len(self.performance_metrics) > 0:
                    avg_confidence = self.performance_metrics.get('avg_confidence', 0)
                    win_rate = self.performance_metrics.get('win_rate', 0)
                    sharpe_ratio = self.performance_metrics.get('sharpe_ratio', 0)

                    # Log performance summary
                    if datetime.now().minute % 15 == 0:  # Every 15 minutes
                        self.logger.info(
                            f"📊 Config Performance - Confidence: {avg_confidence:.2%}, Win Rate: {win_rate:.2%}, Sharpe: {sharpe_ratio:.2f}")

        except Exception as e:
            self.logger.error(f"Error evaluating config performance: {e}")

    def get_config_summary(self) -> Dict[str, Any]:
        """Get comprehensive configuration summary"""
        with self.config_lock:
            return {
                'current_regime': self.current_regime.value,
                'config_version': self.config_version,
                'last_update': self.last_config_update.isoformat(),
                'signal_weights': asdict(self.active_config.signal_weights),
                'trading_thresholds': asdict(self.active_config.trading_thresholds),
                'safety_limits': asdict(self.get_effective_safety_limits()),
                'emergency_overrides_active': len(self.safety_overrides) > 0,
                'performance_metrics': self.performance_metrics.copy(),
                'recent_changes': len(self.config_history)
            }



# Global configuration manager instance
enhanced_config_manager = EnhancedConfigManager()
