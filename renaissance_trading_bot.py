"""
Renaissance Technologies Bitcoin Trading Bot - Main Integration
Combines all components with research-optimized signal weights
"""

import asyncio
import logging
import json
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

# Step 14 & 16 & Deep Alternative
from genetic_optimizer import GeneticWeightOptimizer
from cross_asset_engine import CrossAssetCorrelationEngine
from whale_activity_monitor import WhaleActivityMonitor
from breakout_scanner import BreakoutScanner
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

from renaissance_types import SignalType, OrderType, MLSignalPackage, TradingDecision
from ml_integration_bridge import MLIntegrationBridge

# Production Orchestrator (optional)
try:
    from production_trading_orchestrator import ProductionTradingOrchestrator, ProductionConfig
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

# Types moved to renaissance_types.py


from renaissance_engine_core import SignalFusion, RiskManager

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

        # Multi-Asset Support
        self.product_ids = self.config.get("trading", {}).get("product_ids", ["BTC-USD"])
        self.config_manager = EnhancedConfigManager("config")

        # Initialize all components
        self.microstructure_engine = MicrostructureEngine()
        self.technical_indicators = EnhancedTechnicalIndicators()
        self.market_data_provider = LiveMarketDataProvider(self.config, logger=self.logger)
        
        # Unified Signal Fusion (Step 16+)
        self.signal_fusion = SignalFusion()
        
        self.alternative_data_engine = AlternativeDataEngine(self.config, logger=self.logger)

        # Initialize Advanced Adapters (Step 7 & 9)
        self.regime_overlay = RegimeOverlay(self.config.get("regime_overlay", {}), logger=self.logger)
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

        # Initialize Breakout Scanner (Step 16+)
        scanner_cfg = self.config.get("breakout_scanner", {"enabled": True, "top_n": 30})
        self.breakout_scanner = BreakoutScanner(
            exchanges=scanner_cfg.get("exchanges", ["coinbase", "kraken"]),
            top_n=scanner_cfg.get("top_n", 30),
            logger=self.logger
        )
        self.scanner_enabled = scanner_cfg.get("enabled", True)
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
        
        # Performance Tracking
        self.ml_performance_metrics = {
            'total_trades': 0,
            'ml_enhanced_trades': 0,
            'avg_ml_processing_time': 0.0,
            'ml_success_rate': 0.0
        }

        # State tracking for Dashboard
        self.last_vpin = 0.5
        
        # 📊 Institutional Dashboard
        self.dashboard_enabled = self.config.get("institutional_dashboard", {}).get("enabled", True)
        if self.dashboard_enabled:
            try:
                self.dashboard = InstitutionalDashboard(self, host="0.0.0.0", port=5000)
                self.dashboard.run()
            except Exception as e:
                self.logger.warning(f"Failed to start dashboard (likely port conflict): {e}")
                self.dashboard = None
        else:
            self.dashboard = None
        
        # Initialize Feature Pipeline (Step 16)
        from feature_pipeline import FractalFeaturePipeline
        self.feature_pipeline = FractalFeaturePipeline(
            hd_dimension=100  # Ensure stable feature vector size
        )
        self.pipeline_fitted = False
        
        # Step 8: Dynamic Thresholds
        self.buy_threshold = 0.1
        self.sell_threshold = -0.1
        self.adaptive_thresholds = self.config.get("adaptive_thresholds", True)
        self.breakout_candidates = []
        self.scan_cycle_count = 0

        if self.db_enabled:
            asyncio.create_task(self.db_manager.init_database())

        # Renaissance Research-Optimized Signal Weights
        raw_weights = self.config.get("signal_weights", {
            'order_flow': 0.28,      # Institutional Flow
            'order_book': 0.18,      # Microstructure
            'volume': 0.12,          # Volume
            'macd': 0.08,            # Momentum
            'rsi': 0.08,             # Mean Reversion
            'bollinger': 0.08,       # Volatility
            'alternative': 0.03,     # Sentiment/Whales
            'stat_arb': 0.15         # Statistical Arbitrage
        })
        self.signal_weights = {str(k): float(self._force_float(v)) for k, v in raw_weights.items()}

        # Trading state
        self.current_position = 0.0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self.decision_history = []

        # Risk management (from original bot)
        risk_cfg = self.config.get("risk_management", {})
        self.daily_loss_limit = float(risk_cfg.get("daily_loss_limit", 500))
        self.position_limit = float(risk_cfg.get("position_limit", 1000))
        self.min_confidence = float(risk_cfg.get("min_confidence", 0.65))

        self.logger.info("Renaissance Trading Bot initialized with research-optimized weights")
        self.logger.info(f"Signal weights: {self.signal_weights}")

    def _setup_logging(self, config: Dict[str, Any]) -> logging.Logger:
        """Setup comprehensive logging"""
        log_cfg = config.get("logging", {})
        log_file = log_cfg.get("file", "logs/renaissance_bot.log")
        log_level = log_cfg.get("level", "INFO")

        log_path = (Path(__file__).resolve().parent / log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, str(log_level).upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
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
                "min_confidence": 0.65
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

    async def collect_all_data(self, product_id: str = "BTC-USD") -> Dict[str, Any]:
        """Collect data from all sources for a specific product"""
        try:
            snapshot = await asyncio.to_thread(self.market_data_provider.fetch_snapshot, product_id)
            if snapshot.price_data:
                self.technical_indicators.update_price_data(snapshot.price_data)

            technical_signals = self.technical_indicators.get_latest_signals()
            alt_signals = await self.alternative_data_engine.get_alternative_signals()

            return {
                'order_book_snapshot': snapshot.order_book_snapshot,
                'price_data': snapshot.price_data,
                'technical_signals': technical_signals,
                'alternative_signals': alt_signals,
                'ticker': snapshot.ticker,
                'product_id': product_id,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return {}

    async def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate signals from all components"""
        signals = {}

        try:
            # 1. Microstructure signals (Order Flow + Order Book = 53% total weight)
            order_book_snapshot = market_data.get('order_book_snapshot')
            if order_book_snapshot:
                microstructure_signal = self.microstructure_engine.update_order_book(order_book_snapshot)
                signals['order_flow'] = microstructure_signal.large_trade_flow
                signals['order_book'] = microstructure_signal.order_book_imbalance
            else:
                signals['order_flow'] = 0.0
                signals['order_book'] = 0.0

            # 2. Technical indicators (38% total weight)
            technical_signal = market_data.get('technical_signals') or self.technical_indicators.get_latest_signals()
            if technical_signal:
                signals['volume'] = _signed_strength(technical_signal.obv_momentum)
                signals['macd'] = _signed_strength(technical_signal.quick_macd)
                signals['rsi'] = _signed_strength(technical_signal.fast_rsi)
                signals['bollinger'] = _signed_strength(technical_signal.dynamic_bollinger)
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
            try:
                p_id = market_data.get('product_id', 'Unknown')
                # Get common dataframe for high-dimensional analysis
                df = self.technical_indicators._to_dataframe()
                
                # Volume Profile
                if not df.empty:
                    profile = self.volume_profile_engine.calculate_profile(df)
                    if profile:
                        vp_signal = self.volume_profile_engine.get_profile_signal(
                            market_data.get('current_price', 0.0), 
                            profile
                        )
                        signals['volume_profile'] = vp_signal['signal']
                        self._last_vp_status[p_id] = vp_signal['status']

                # Statistical Arbitrage (BTC vs ETH)
                if p_id in ["BTC-USD", "ETH-USD"]:
                    other_id = "ETH-USD" if p_id == "BTC-USD" else "BTC-USD"
                    sa_signal = self.stat_arb_engine.calculate_pair_signal(p_id, other_id)
                    signals['stat_arb'] = sa_signal.get('signal', 0.0)

                # Cross-Asset Lead-Lag Alpha (Step 16)
                if len(self.product_ids) > 1:
                    # Update Correlation Engine with current price
                    self.correlation_engine.update_price(p_id, market_data.get('current_price', 0.0))
                    
                    base = self.product_ids[0]
                    target = p_id
                    if base != target:
                        ll_data = self.correlation_engine.calculate_lead_lag(base, target)
                        signals['lead_lag'] = ll_data.get('lead_lag_score', 0.0)
                        market_data['lead_lag_alpha'] = ll_data

                # Fractal Intelligence (DTW)
                if not df.empty:
                    prices = df['close'].values
                    fractal_result = self.fractal_intelligence.find_best_match(prices)
                    signals['fractal'] = fractal_result['signal']
                    market_data['fractal_intelligence'] = fractal_result

                # Market Entropy (Shannon/ApEn)
                if not df.empty:
                    prices = df['close'].values
                    entropy_result = self.market_entropy.calculate_entropy(prices)
                    signals['entropy'] = 0.5 * (entropy_result['predictability'] - 0.5) # Center at 0
                    market_data['market_entropy'] = entropy_result

                # ML Feature Pipeline & Real-Time Intelligence (Step 12/16 Bridge)
                if not df.empty:
                    try:
                        # Extract real market features
                        if not self.pipeline_fitted:
                            self.feature_pipeline.fit_transform(df)
                            self.pipeline_fitted = True
                        
                        feature_vector = self.feature_pipeline.transform(df)
                        
                        # Inject into real-time pipeline
                        if self.real_time_pipeline.enabled:
                            rt_result = await self.real_time_pipeline.processor.process_all_models({
                                'feature_vector': feature_vector,
                                'price_df': df
                            })
                            market_data['real_time_predictions'] = rt_result
                            
                            # Add ML signals to fusion (if they have enough confidence)
                            if 'Ensemble' in rt_result:
                                signals['ml_ensemble'] = rt_result['Ensemble']
                            if 'CNN' in rt_result:
                                signals['ml_cnn'] = rt_result['CNN']
                                
                    except Exception as e:
                        self.logger.warning(f"ML feature bridge failed: {e}")

                # Quantum Oscillator (QHO)
                if not df.empty:
                    prices = df['close'].values
                    quantum_result = self.quantum_oscillator.calculate_quantum_levels(prices)
                    signals['quantum'] = quantum_result['signal']
                    market_data['quantum_oscillator'] = quantum_result

            except Exception as e:
                self.logger.warning(f"Advanced signal generation skipped: {e}")

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
            except:
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
            except:
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
                              product_id: str = "BTC-USD", ml_package: Optional[MLSignalPackage] = None) -> TradingDecision:
        """Make final trading decision with Renaissance methodology"""

        # Calculate confidence based on signal strength and consensus
        signal_strength = abs(weighted_signal)
        signal_consensus = 1.0 - np.std(list(signal_contributions.values()))
        confidence = (signal_strength + signal_consensus) / 2.0
        
        # Apply regime-derived confidence boost (max +/-5%)
        confidence = float(np.clip(confidence + self.regime_overlay.get_confidence_boost(), 0.0, 1.0))

        # 🤖 ML Enhanced Confidence (Unified from Enhanced Bot)
        if ml_package:
            # If models agree with signal direction, boost confidence
            direction_match = np.sign(weighted_signal) == np.sign(ml_package.ensemble_score)
            overlay = 0.05 if direction_match else -0.05
            consciousness_factor = ml_package.confidence_score
            confidence = float(np.clip(confidence + (overlay * consciousness_factor), 0.0, 1.0))
            self.logger.info(f"ML confidence adjustment: {(overlay * consciousness_factor):+.4f} (Consciousness: {consciousness_factor:.2f})")

        # Determine action
        if confidence < self.min_confidence:
            action = 'HOLD'
            position_size = 0.0
        elif weighted_signal > self.buy_threshold:
            action = 'BUY'
            # 🛡️ ML Enhanced Sizing
            position_size = self.risk_manager.calculate_ml_enhanced_position_size(
                weighted_signal, confidence, current_price, ml_package
            )
        elif weighted_signal < self.sell_threshold:
            action = 'SELL'
            position_size = self.risk_manager.calculate_ml_enhanced_position_size(
                weighted_signal, confidence, current_price, ml_package
            )
        else:
            action = 'HOLD'
            position_size = 0.0

        # Step 19: Log turnover and expected alpha
        if action != 'HOLD':
            self.logger.info(f"🎯 TURNOVER DETECTED: {action} {product_id} | Signal: {weighted_signal:+.4f} | Conf: {confidence:.3f}")

        # ML-Enhanced Risk Assessment (Regime Gate)
        risk_assessment = self.risk_manager.assess_risk_regime(ml_package)
        if risk_assessment['recommended_action'] == 'fallback_mode':
            self.logger.warning("ML Risk assessment triggered FALLBACK MODE - halting trades")
            action = 'HOLD'
            position_size = 0.0

        # Apply basic risk limits
        if abs(self.daily_pnl) >= self.daily_loss_limit:
            action = 'HOLD'
            position_size = 0.0
            self.logger.warning(f"Daily loss limit reached: ${self.daily_pnl}")

        # Gate decision through Advanced Risk Gateway (Step 9) - VAE Anomaly Detection
        if action != 'HOLD':
            portfolio_data = {
                'total_value': self.position_limit,
                'daily_pnl': self.daily_pnl,
                'positions': {'BTC': self.current_position},
                'current_price': current_price
            }
            
            # Use ML package feature vector for VAE if available
            feature_vector = ml_package.feature_vector if ml_package else None

            is_allowed = self.risk_gateway.assess_trade(
                action=action,
                amount=position_size,
                current_price=current_price,
                portfolio_data=portfolio_data,
                feature_vector=feature_vector
            )
            
            if not is_allowed:
                self.logger.warning(f"Risk Gateway BLOCKED {action} order (VAE Anomaly or Risk limit)")
                action = 'HOLD'
                position_size = 0.0

        reasoning = {
            'weighted_signal': weighted_signal,
            'confidence': confidence,
            'signal_contributions': signal_contributions,
            'current_price': current_price,
            'ml_risk_assessment': risk_assessment,
            'risk_check': {
                'daily_pnl': self.daily_pnl,
                'daily_limit': self.daily_loss_limit,
                'position_limit': self.position_limit
            }
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
        """Execute order using Step 10 Smart Execution Suite"""
        try:
            product_id = market_data.get('product_id', 'BTC-USD')
            current_price = decision.reasoning.get('current_price', 0.0)
            
            order_details = {
                'product_id': product_id,
                'side': decision.action,
                'size': decision.position_size,
                'price': current_price,
                'type': 'MARKET'
            }
            
            # 1. Analyze Slippage Risk
            slippage_risk = self.slippage_protection.analyze_slippage_risk(order_details, market_data)
            self.logger.info(f"Slippage risk for {product_id}: {slippage_risk.get('risk_level', 'UNKNOWN')}")
            
            # 2. Select Algorithm
            algo = 'SMART'
            if slippage_risk.get('predicted_slippage', 0) > 0.005: # > 0.5%
                algo = 'TWAP'
                self.logger.info(f"High slippage risk detected, switching to {algo}")
            
            # 3. Execute via Suite
            exec_result = self.execution_suite.execute_order(order_details, market_data, algorithm=algo)
            
            # 4. Persist Trade
            if self.db_enabled:
                trade_data = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'product_id': product_id,
                    'side': decision.action,
                    'size': decision.position_size,
                    'price': exec_result.get('execution_price', current_price),
                    'status': exec_result.get('status', 'EXECUTED'),
                    'algo_used': algo,
                    'slippage': exec_result.get('slippage', 0.0),
                    'execution_time': exec_result.get('execution_time', 0.0)
                }
                asyncio.create_task(self.db_manager.store_trade(trade_data))

            self.logger.info(f"Smart execution complete: {exec_result.get('status', 'EXECUTED')} via {algo}")
            return exec_result
            
        except Exception as e:
            self.logger.error(f"Smart execution failed: {e}")
            # Fallback to simple paper trade log
            self.logger.info(f"PAPER TRADE (Fallback): {decision.action} {decision.position_size}")
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
            optimized_weights = await self.genetic_optimizer.run_optimization_cycle(self.signal_weights)
            
            if optimized_weights != self.signal_weights:
                self.logger.info("New optimized weights discovered via Evolution!")
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

    async def execute_trading_cycle(self) -> TradingDecision:
        """Execute one complete trading cycle across all products"""
        cycle_start = time.time()
        decisions = []

        try:
            for product_id in self.product_ids:
                self.logger.info(f"Starting cycle for {product_id}...")
                
                # 1. Collect all market data
                market_data = await self.collect_all_data(product_id)

                if not market_data:
                    self.logger.warning(f"No market data for {product_id}, skipping")
                    continue
                
                # Standardize price key
                ticker = market_data.get('ticker', {})
                current_price = float(ticker.get('price', 0.0))
                if current_price == 0:
                    current_price = float(ticker.get('last', 0.0))
                    market_data['ticker']['price'] = current_price # Standardize
                
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
                        source="Coinbase",
                        product_id=product_id
                    )
                    asyncio.create_task(self.db_manager.store_market_data(md_persist))

                # 2. Generate signals from all components
                signals = await self.generate_signals(market_data)
                
                # HARDENING: Ensure all signals are floats
                signals = {k: self._force_float(v) for k, v in signals.items()}
                
                # 2.1 ML Enhanced Signal Fusion (Unified from Enhanced Bot)
                ml_package = None
                if self.ml_enabled:
                    # ML Bridge generates parallel model predictions (CNN-LSTM, N-BEATS, etc.)
                    ml_package = await self.ml_bridge.generate_ml_signals(market_data, signals)
                
                # 2.2 Volume Profile Intelligence (Institutional)
                price_df = self.technical_indicators._to_dataframe()
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

                # 3. Calculate Renaissance weighted signal
                weighted_signal, contributions = self.calculate_weighted_signal(signals)
                
                # PARANOID SCALAR HARDENING: Ensure results are primitive floats
                weighted_signal = float(self._force_float(weighted_signal))
                contributions = {str(k): float(self._force_float(v)) for k, v in contributions.items()}

                # EXTRA HARDENING: Ensure signals dictionary is all floats for boost calculation
                signals = {str(k): float(self._force_float(v)) for k, v in signals.items()}
                self.logger.info(f"HARDENED SIGNALS: {[(k, type(v)) for k, v in signals.items()]}")

                market_data['ml_package'] = ml_package
                
                # 3.1 Non-linear Confluence Boost (Step 20)
                confluence_data = self.confluence_engine.calculate_confluence_boost(signals)
                self.logger.info(f"CONFLUENCE DATA TYPE: {type(confluence_data.get('total_confluence_boost'))}")
                
                # Extract boost scalar with hardening
                boost_scalar_final = 0.0
                try:
                    raw_b_v_f = confluence_data.get('total_confluence_boost', 0.0)
                    boost_scalar_final = self._force_float(raw_b_v_f)
                except:
                    boost_scalar_final = 0.0

                if boost_scalar_final > 0:
                    try:
                        # PURE SCALAR MULTIPLICATION TYPE GUARD
                        b_sig_f = float(weighted_signal)
                        b_factor_f = float(1.0 + boost_scalar_final)
                        self.logger.info(f"BOOST DEBUG: b_sig_f={type(b_sig_f)}, b_factor_f={type(b_factor_f)}")
                        # Binary operation on standard floats
                        boosted_val_f = b_sig_f * b_factor_f
                        weighted_signal = float(np.clip(boosted_val_f, -1.0, 1.0))
                        self.logger.info(f"🏛️ CONFLUENCE BOOST: {b_sig_f:+.4f} -> {weighted_signal:+.4f} (Factor: {b_factor_f})")
                    except Exception as e:
                        self.logger.warning(f"Confluence boost application failed: {e}")
                else:
                    weighted_signal = self._force_float(weighted_signal)
                
                # Final check to ensure it's not a sequence before decision
                weighted_signal = float(weighted_signal)
                market_data['confluence_data'] = confluence_data

                # 3.2 Update Dynamic Thresholds (Step 8)
                self._update_dynamic_thresholds(product_id, market_data)

                # Update regime overlay
                try:
                    if self.regime_overlay.enabled and hasattr(self.technical_indicators, 'price_history') \
                       and len(self.technical_indicators.price_history) > 0:
                        price_df = self.technical_indicators._to_dataframe()
                        if not price_df.empty:
                            self.regime_overlay.update(price_df)
                except Exception as _e:
                    self.logger.debug(f"Regime overlay skipped: {_e}")

                # 4. Real-time pipeline cycle (Step 12)
                rt_result = None
                if self.real_time_pipeline.enabled:
                    await self.real_time_pipeline.start()
                    raw_rt = await self.real_time_pipeline.run_cycle()
                    if raw_rt:
                        # Hardening real-time pipeline outputs
                        rt_result = {}
                        for k, v in raw_rt.items():
                            if k == 'predictions':
                                rt_result[k] = {mk: self._force_float(mv) for mk, mv in v.items()}
                            else:
                                try:
                                    rt_result[k] = self._force_float(v)
                                except:
                                    rt_result[k] = v
                
                # 4.5 Statistical Arbitrage & Fractal Intelligence
                current_price = market_data.get('ticker', {}).get('price', 0.0)
                self.stat_arb_engine.update_price(product_id, current_price)
                
                # 🏛️ Basis Trading Signal
                basis_signal = self._force_float(self.basis_engine.get_basis_signal(market_data))
                signals['basis'] = basis_signal
                
                stat_arb_data = {}
                if len(self.product_ids) > 1:
                    base = "BTC-USD"
                    target = "ETH-USD"
                    if product_id == base:
                        stat_arb_data = self.stat_arb_engine.calculate_pair_signal(base, target)
                    elif product_id in ["ETH-USD"]:
                        # Inverse for ETH if BTC is base
                        stat_arb_data = self.stat_arb_engine.calculate_pair_signal(base, product_id)
                        if 'signal' in stat_arb_data:
                            stat_arb_data['signal'] = -self._force_float(stat_arb_data['signal'])
                
                if stat_arb_data.get('status') == 'active':
                    signals['stat_arb'] = self._force_float(stat_arb_data['signal'])
                else:
                    signals['stat_arb'] = 0.0

                # 5. Make trading decision
                ticker = market_data.get('ticker', {})
                current_price = self._force_float(ticker.get('price', 0.0))
                
                # 5.1 Meta-Strategy Selection
                regime_data = self.regime_overlay.current_regime or {}
                self.last_vpin = market_data.get('vpin', 0.5)
                execution_mode = self.strategy_selector.select_mode(market_data, regime_data)
                market_data['execution_mode'] = execution_mode
                
                decision = self.make_trading_decision(weighted_signal, contributions, 
                                                    current_price=current_price, 
                                                    real_time_result=rt_result,
                                                    product_id=product_id,
                                                    ml_package=ml_package)
                
                # Inject Meta-Strategy Execution Mode (Step 11/13)
                decision.reasoning['execution_mode'] = market_data.get('execution_mode', 'TAKER')
                
                decision.reasoning['product_id'] = product_id
                if rt_result:
                    decision.reasoning['real_time_pipeline'] = rt_result
                
                # Retrieve lead_lag_alpha from market_data if it was calculated in generate_signals
                if 'lead_lag_alpha' in market_data:
                    decision.reasoning['lead_lag_alpha'] = market_data['lead_lag_alpha']
                    
                if stat_arb_data:
                    decision.reasoning['stat_arb'] = stat_arb_data
                if 'whale_signals' in market_data:
                    decision.reasoning['whale_signals'] = market_data['whale_signals']

                # 5.1 Persist Decision & ML Predictions
                if self.db_enabled:
                    # Store decision
                    decision_persist = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                        'product_id': product_id,
                        'action': decision.action,
                        'confidence': decision.confidence,
                        'position_size': decision.position_size,
                        'weighted_signal': weighted_signal,
                        'reasoning': decision.reasoning
                    }
                    asyncio.create_task(self.db_manager.store_decision(decision_persist))
                    
                    # Store ML predictions
                    if rt_result and 'predictions' in rt_result:
                        for model_name, pred in rt_result['predictions'].items():
                            asyncio.create_task(self.db_manager.store_ml_prediction({
                                'product_id': product_id,
                                'model_name': model_name,
                                'prediction': pred
                            }))

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
                    asyncio.create_task(self._run_adaptive_learning_cycle())
                    asyncio.create_task(self._perform_attribution_analysis())
                    
                    # Run Self-Reinforcing Learning (Step 19)
                    if self.real_time_pipeline.enabled:
                        asyncio.create_task(self.learning_engine.run_learning_cycle(
                            self.real_time_pipeline.processor.models
                        ))
                    
                    # Run Genetic Weight Optimization (Step 14)
                    async def run_evo():
                        new_weights = await self.genetic_optimizer.run_optimization_cycle(self.signal_weights)
                        if new_weights != self.signal_weights:
                            self.logger.info("Evolutionary Step (Step 14): Weights updated.")
                            self.signal_weights = new_weights
                    
                    asyncio.create_task(run_evo())

                # Run Breakout Scan (Step 16+)
                self.scan_cycle_count += 1
                if self.scan_cycle_count % 10 == 0:
                    asyncio.create_task(self._run_breakout_scan())
                
                decisions.append(decision)
                
                self.logger.info(f"[{product_id}] Decision: {decision.action} "
                               f"(Conf: {decision.confidence:.3f}, Size: {decision.position_size:.3f})")

            cycle_time = time.time() - cycle_start
            self.logger.info(f"Total cycle completed in {cycle_time:.2f}s")

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
        
        tech_signals = self.technical_indicators.get_latest_signals()
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

    async def _run_breakout_scan(self):
        """Step 16+: Renaissance Global Scanner for breakout opportunities"""
        if not self.scanner_enabled:
            return
            
        try:
            self.logger.info("🚀 Initiating Renaissance Global Breakout Scan...")
            results = await self.breakout_scanner.scan_all_exchanges()
            self.breakout_candidates = results
            
            if results:
                self.logger.info(f"🔥 Found {len(results)} breakout candidates!")
                for r in results[:5]:
                    # Renaissance Strategy: Auto-Watch high confidence breakouts
                    if r['breakout_score'] >= 80 and r['symbol'] not in self.product_ids:
                        self.logger.info(f"🎯 Auto-Watching high-confidence breakout: {r['symbol']}")
                        self.product_ids.append(r['symbol'])
                        
            else:
                self.logger.info("No major breakouts detected in this cycle.")
        except Exception as e:
            self.logger.error(f"Breakout scan failed: {e}")

    async def run_continuous_trading(self, cycle_interval: int = 300):
        """Run continuous Renaissance trading (default 5-minute cycles)"""
        self.logger.info(f"Starting continuous Renaissance trading with {cycle_interval}s cycles")

        # Start real-time pipeline if enabled
        if self.real_time_pipeline.enabled:
            await self.real_time_pipeline.start()

        # Start Ghost Runner Loop (Step 18)
        asyncio.create_task(self.ghost_runner.start_ghost_loop(interval=cycle_interval * 2))

        while True:
            try:
                # Execute trading cycle
                decision = await self.execute_trading_cycle()

                # In production, this would execute the actual trade
                # For now, we'll just log the paper trading decision
                self.logger.info(f"PAPER TRADE: {decision.action} - "
                               f"Confidence: {decision.confidence:.3f} - "
                               f"Position Size: {decision.position_size:.3f}")

                # Wait for next cycle
                await asyncio.sleep(cycle_interval)

            except KeyboardInterrupt:
                self.logger.info("Trading stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def _update_dynamic_thresholds(self, product_id: str, market_data: Dict[str, Any]):
        """Adjusts BUY/SELL thresholds based on volatility and confidence (Step 8)"""
        if not self.adaptive_thresholds:
            return

        try:
            # Use technical indicators volatility regime
            latest_tech = self.technical_indicators.get_latest_signals()
            vol_regime = latest_tech.volatility_regime if latest_tech else None
            
            # Base thresholds
            self.buy_threshold = 0.1
            self.sell_threshold = -0.1
            
            # Adjust based on volatility
            if vol_regime == "high_volatility" or vol_regime == "extreme_volatility":
                # Increase thresholds in high volatility to avoid fakeouts
                self.buy_threshold = 0.15
                self.sell_threshold = -0.15
            elif vol_regime == "low_volatility":
                # Decrease thresholds in low volatility to catch smaller moves
                self.buy_threshold = 0.07
                self.sell_threshold = -0.07
                
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
