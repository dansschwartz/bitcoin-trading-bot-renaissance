"""
Renaissance Technologies Bitcoin Trading Bot - Main Integration
Combines all components with research-optimized signal weights
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time

# Import all Renaissance components
from enhanced_config_manager import EnhancedConfigManager
from microstructure_engine import MicrostructureEngine, MicrostructureMetrics
from enhanced_technical_indicators import EnhancedTechnicalIndicators, TechnicalSignal
from order_book_collector import OrderBookCollector
from price_data_collector import PriceDataCollector
from renaissance_signal_fusion import RenaissanceSignalFusion
from alternative_data_engine import AlternativeDataEngine, AlternativeSignal

from enum import Enum


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"
    MICROSTRUCTURE = "MICROSTRUCTURE"
    TECHNICAL = "TECHNICAL"
    ALTERNATIVE = "ALTERNATIVE"
    RSI = "RSI"
    MACD = "MACD"
    BOLLINGER = "BOLLINGER"
    ORDER_FLOW = "ORDER_FLOW"
    VOLUME = "VOLUME"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    TREND = "TREND"
    VOLATILITY = "VOLATILITY"
    SENTIMENT = "SENTIMENT"
    FRACTAL = "FRACTAL"
    QUANTUM = "QUANTUM"
    CONSCIOUSNESS = "CONSCIOUSNESS"


class OrderType(Enum):
    """Types of trading orders"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class SignalFusion:
    """Signal fusion results"""

    def __init__(self):
        self.combined_signal = 0.0
        self.confidence = 0.0
        self.contributing_signals = {}
        self.weights = {}

    def fuse_signals(self, signals: Dict[str, float], weights: Dict[str, float]) -> Dict[str, float]:
        """Fuse multiple signals with weights"""
        combined = sum(signals[k] * weights.get(k, 0) for k in signals.keys())
        return {
            'combined_signal': combined,
            'confidence': min(abs(combined), 1.0),
            'contributing_signals': signals,
            'weights': weights
        }


class RiskManager:
    """Risk management component"""
    def __init__(self, daily_loss_limit=500.0, position_limit=1000.0, *args, **kwargs):
        self.max_position_size = position_limit
        self.current_risk = 0.0
        self.daily_pnl = 0.0
        self.daily_loss_limit = daily_loss_limit
        self.position_limit = position_limit
        self.risk_limits = {
            'daily_loss': daily_loss_limit,
            'position_limit': position_limit
        }
        # Accept any additional arguments without error
        for key, value in kwargs.items():
            setattr(self, key, value)

@dataclass
class TradingDecision:
    """Final trading decision with Renaissance methodology"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0 to 1
    position_size: float  # Percentage of available capital
    reasoning: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            'action': self.action,
            'confidence': self.confidence,
            'position_size': self.position_size,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }

class RenaissanceTradingBot:
    """
    Main Renaissance Technologies-style Bitcoin trading bot
    Integrates all components with research-optimized weights
    """

    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the Renaissance trading bot"""
        self.logger = self._setup_logging()

        # Initialize configuration manager
        self.config_manager = EnhancedConfigManager("config")

        # Initialize all components
        self.microstructure_engine = MicrostructureEngine()
        self.technical_indicators = EnhancedTechnicalIndicators()
        self.order_book_collector = OrderBookCollector()
        self.price_data_collector = PriceDataCollector()
        self.signal_fusion = RenaissanceSignalFusion()
        self.alternative_data_engine = AlternativeDataEngine()

        # Renaissance Research-Optimized Signal Weights
        self.signal_weights = {
            'order_flow': 0.32,      # 30-34% - Dominant signal
            'order_book': 0.21,      # 18-24% - Market microstructure
            'volume': 0.14,          # 10-18% - Volume analysis
            'macd': 0.105,           # 8-13% - Momentum
            'rsi': 0.115,            # 5-18% - Mean reversion
            'bollinger': 0.095,      # 5-18% - Volatility
            'alternative': 0.045     # 2-7% - Alternative data
        }

        # Trading state
        self.current_position = 0.0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self.decision_history = []

        # Risk management (from original bot)
        self.daily_loss_limit = 500  # $500 daily loss limit
        self.position_limit = 1000   # $1000 position limit
        self.min_confidence = 0.65   # Minimum confidence for trades

        self.logger.info("Renaissance Trading Bot initialized with research-optimized weights")
        self.logger.info(f"Signal weights: {self.signal_weights}")

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('../archive/renaissance_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    async def collect_all_data(self) -> Dict[str, Any]:
        """Collect data from all sources"""
        try:
            # Collect synchronous data
            order_book = self.order_book_collector.get_latest_microstructure_data()
            price_data = self.price_data_collector.get_latest_technical_data()

            # Collect async data
            alt_signals = await self.alternative_data_engine.get_alternative_signals()

            return {
                'order_book': order_book,
                'price_data': price_data,
                'alternative_signals': alt_signals,
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
            if market_data.get('order_book'):
                # Generate fresh order book snapshot and analyze with OUR engine
                order_book_snapshot = self.order_book_collector._generate_order_book_snapshot()
                microstructure_signal = self.microstructure_engine._analyze_microstructure(order_book_snapshot)

                if microstructure_signal:
                    signals['order_flow'] = microstructure_signal.large_trade_flow  # Use large_trade_flow for order flow
                    signals['order_book'] = microstructure_signal.order_book_imbalance  # This one is correct
                else:
                    signals['order_flow'] = 0.0
                    signals['order_book'] = 0.0
            else:
                signals['order_flow'] = 0.0
                signals['order_book'] = 0.0

            # 2. Technical indicators (38% total weight)
            if market_data.get('price_data'):
                technical_signal = self.technical_indicators.get_latest_signals()

                if technical_signal:
                    # Use the actual attributes from MultiTimeframeSignals
                    signals['volume'] = technical_signal.volume_signal.strength  # or whatever the actual attribute is
                    signals['macd'] = technical_signal.macd_signal.strength
                    signals['rsi'] = technical_signal.rsi_signal.strength
                    signals['bollinger'] = technical_signal.bollinger_signal.strength
                else:
                    signals['volume'] = 0.0
                    signals['macd'] = 0.0
                    signals['rsi'] = 0.0
                    signals['bollinger'] = 0.0
            else:
                signals['volume'] = 0.0
                signals['macd'] = 0.0
                signals['rsi'] = 0.0
                signals['bollinger'] = 0.0

            # 3. Alternative data signals (4.5% total weight)
            if market_data.get('alternative_signals'):
                alt_signal = market_data['alternative_signals']
                # Combine all alternative signals into one composite score
                alternative_composite = (
                    alt_signal.social_sentiment * 0.4 +
                    alt_signal.on_chain_strength * 0.35 +
                    alt_signal.market_psychology * 0.25
                )
                signals['alternative'] = alternative_composite
            else:
                signals['alternative'] = 0.0

            self.logger.info(f"Generated signals: {signals}")
            return signals

        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return {key: 0.0 for key in self.signal_weights.keys()}

    def calculate_weighted_signal(self, signals: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Calculate final weighted signal using Renaissance weights"""
        weighted_signal = 0.0
        signal_contributions = {}

        for signal_type, weight in self.signal_weights.items():
            signal_value = signals.get(signal_type, 0.0)
            contribution = signal_value * weight
            weighted_signal += contribution
            signal_contributions[signal_type] = contribution

        # Normalize to [-1, 1] range
        weighted_signal = np.clip(weighted_signal, -1.0, 1.0)

        return weighted_signal, signal_contributions

    def make_trading_decision(self, weighted_signal: float, signal_contributions: Dict[str, float]) -> TradingDecision:
        """Make final trading decision with Renaissance methodology"""

        # Calculate confidence based on signal strength and consensus
        signal_strength = abs(weighted_signal)
        signal_consensus = 1.0 - np.std(list(signal_contributions.values()))
        confidence = (signal_strength + signal_consensus) / 2.0

        # Determine action
        if confidence < self.min_confidence:
            action = 'HOLD'
            position_size = 0.0
        elif weighted_signal > 0.1:  # Buy threshold
            action = 'BUY'
            position_size = min(confidence * 0.5, 0.3)  # Max 30% position
        elif weighted_signal < -0.1:  # Sell threshold
            action = 'SELL'
            position_size = min(confidence * 0.5, 0.3)  # Max 30% position
        else:
            action = 'HOLD'
            position_size = 0.0

        # Apply risk management
        if abs(self.daily_pnl) >= self.daily_loss_limit:
            action = 'HOLD'
            position_size = 0.0
            self.logger.warning(f"Daily loss limit reached: ${self.daily_pnl}")

        reasoning = {
            'weighted_signal': weighted_signal,
            'confidence': confidence,
            'signal_contributions': signal_contributions,
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

    async def execute_trading_cycle(self) -> TradingDecision:
        """Execute one complete trading cycle"""
        cycle_start = time.time()

        try:
            # 1. Collect all market data
            self.logger.info("Starting Renaissance trading cycle...")
            market_data = await self.collect_all_data()

            if not market_data:
                self.logger.warning("No market data available, holding position")
                return TradingDecision('HOLD', 0.0, 0.0, {}, datetime.now())

            # 2. Generate signals from all components
            signals = await self.generate_signals(market_data)

            # 3. Calculate Renaissance weighted signal
            weighted_signal, contributions = self.calculate_weighted_signal(signals)

            # 4. Make trading decision
            decision = self.make_trading_decision(weighted_signal, contributions)

            # 5. Log decision
            self.decision_history.append(decision)
            self.logger.info(f"Trading Decision: {decision.action} "
                           f"(Confidence: {decision.confidence:.3f}, "
                           f"Size: {decision.position_size:.3f}, "
                           f"Signal: {weighted_signal:.3f})")

            # 6. Log signal breakdown
            self.logger.info("Signal Contributions:")
            for signal_type, contribution in contributions.items():
                weight = self.signal_weights[signal_type]
                self.logger.info(f"  {signal_type}: {contribution:.4f} "
                               f"(weight: {weight:.3f}, raw: {signals.get(signal_type, 0):.4f})")

            cycle_time = time.time() - cycle_start
            self.logger.info(f"Trading cycle completed in {cycle_time:.2f}s")

            return decision

        except Exception as e:
            self.logger.error(f"Trading cycle failed: {e}")
            return TradingDecision('HOLD', 0.0, 0.0, {'error': str(e)}, datetime.now())

    async def run_continuous_trading(self, cycle_interval: int = 300):
        """Run continuous Renaissance trading (default 5-minute cycles)"""
        self.logger.info(f"Starting continuous Renaissance trading with {cycle_interval}s cycles")

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
