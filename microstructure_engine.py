"""
Microstructure Signal Engine
Renaissance Technologies-style order book analysis and institutional flow detection.
Advanced market microstructure analytics for high-frequency trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque, defaultdict
from enum import Enum
import threading
import time
import json

logger = logging.getLogger(__name__)

class FlowDirection(Enum):
    """Trade flow direction classification"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class LiquidityRegime(Enum):
    """Market liquidity regime classification"""
    HIGH_LIQUIDITY = "high_liquidity"
    NORMAL_LIQUIDITY = "normal_liquidity"
    LOW_LIQUIDITY = "low_liquidity"
    STRESS_LIQUIDITY = "stress_liquidity"

@dataclass
class OrderBookLevel:
    """Single order book level"""
    price: float
    size: float
    orders: int = 1
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_price: float
    last_size: float
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return 0.0
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2.0
        return self.last_price

@dataclass
class TradeData:
    """Individual trade data"""
    timestamp: datetime
    price: float
    size: float
    side: str  # 'buy', 'sell', 'unknown'
    trade_id: str = ""
    
class MicrostructureMetrics(NamedTuple):
    """Microstructure analysis results"""
    order_book_imbalance: float
    depth_pressure: float
    volume_spike_score: float
    large_trade_flow: float
    spread_regime_score: float
    vpin: float
    overall_signal: float
    confidence: float
    regime: LiquidityRegime

class MicrostructureSignal(NamedTuple):
    """Lightweight microstructure signal container used in tests"""
    order_flow_strength: float
    order_book_imbalance: float
    volume_pressure: float
    confidence: float
    timestamp: datetime

class MicrostructureEngine:
    """
    Advanced Microstructure Signal Engine
    Implements Renaissance Technologies-style order book and flow analysis
    """
    
    def __init__(self, max_history: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.max_history = max_history
        
        # Data storage
        self.order_book_history: deque = deque(maxlen=max_history)
        self.trade_history: deque = deque(maxlen=max_history)
        self.metrics_history: deque = deque(maxlen=max_history)
        
        # Analysis parameters
        self.depth_levels = 10  # Order book levels to analyze
        self.volume_lookback = 100  # Trades for volume analysis
        self.flow_lookback = 50  # Trades for flow analysis
        
        # Calibration parameters
        self.imbalance_threshold = 0.6
        self.volume_spike_threshold = 2.5
        self.large_trade_threshold = 1000.0  # USD value
        self.spread_percentiles = {'tight': 0.25, 'wide': 0.75}
        
        # VPIN (Volume-Synchronized Probability of Informed Trading)
        self.vpin_bucket_size = 10.0  # Base volume units
        self.vpin_buckets_count = 50
        self.vpin_buckets = deque(maxlen=self.vpin_buckets_count)
        self.current_bucket_buy_vol = 0.0
        self.current_bucket_sell_vol = 0.0
        
        # Threading for real-time analysis
        self.analysis_lock = threading.Lock()
        self.last_analysis = datetime.now()
        
        # Performance tracking
        self.signal_performance: Dict[str, float] = {}
        
        self.logger.info("✅ Microstructure Engine initialized")
    
    def update_order_book(self, snapshot: OrderBookSnapshot) -> MicrostructureMetrics:
        """
        Update order book and generate microstructure signals
        
        Args:
            snapshot: Current order book snapshot
            
        Returns:
            MicrostructureMetrics with all signal components
        """
        try:
            with self.analysis_lock:
                # Store snapshot
                self.order_book_history.append(snapshot)
                
                # Generate all microstructure signals
                metrics = self._analyze_microstructure(snapshot)
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                self.last_analysis = datetime.now()
                
                # Log significant signals
                if metrics.confidence > 0.7:
                    self.logger.info(f"🔬 Strong microstructure signal: {metrics.overall_signal:.3f} "
                                   f"(confidence: {metrics.confidence:.2%})")
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error updating order book: {e}")
            return self._default_metrics()
    
    def update_trade(self, trade: TradeData) -> None:
        """Update trade data for flow analysis and VPIN"""
        try:
            with self.analysis_lock:
                self.trade_history.append(trade)
                
                # Update VPIN bucket
                if trade.side == 'buy':
                    self.current_bucket_buy_vol += trade.size
                elif trade.side == 'sell':
                    self.current_bucket_sell_vol += trade.size
                
                # Check if bucket is full
                total_vol = self.current_bucket_buy_vol + self.current_bucket_sell_vol
                if total_vol >= self.vpin_bucket_size:
                    imbalance = abs(self.current_bucket_buy_vol - self.current_bucket_sell_vol)
                    self.vpin_buckets.append(imbalance)
                    
                    # Reset current bucket
                    self.current_bucket_buy_vol = 0.0
                    self.current_bucket_sell_vol = 0.0
                
        except Exception as e:
            self.logger.error(f"Error updating trade: {e}")
    
    async def analyze_microstructure(self, order_book: Any) -> MicrostructureSignal:
        """Compatibility stub for tests; returns lightweight microstructure signal."""
        # Minimal placeholder; tests patch this method.
        return MicrostructureSignal(
            order_flow_strength=0.0,
            order_book_imbalance=0.0,
            volume_pressure=0.0,
            confidence=0.0,
            timestamp=datetime.now()
        )
    
    def _analyze_microstructure(self, current_snapshot: OrderBookSnapshot) -> MicrostructureMetrics:
        """
        Comprehensive microstructure analysis
        
        Returns all Renaissance Technologies-style microstructure signals
        """
        try:
            # 1. Order Book Imbalance (30% weight)
            imbalance_score = self._calculate_order_book_imbalance(current_snapshot)
            
            # 2. Depth Pressure Analysis (20% weight)
            depth_score = self._calculate_depth_pressure(current_snapshot)
            
            # 3. Volume Spike Detection (15% weight)
            volume_score = self._calculate_volume_spike()
            
            # 4. Large Trade Flow Analysis
            flow_score = self._calculate_large_trade_flow()
            
            # 5. Spread Regime Classification (5% weight)
            spread_score = self._calculate_spread_regime(current_snapshot)
            
            # 6. VPIN Analysis (10% weight)
            vpin_score = self._calculate_vpin()
            
            # Combine signals with Renaissance Technologies weights
            overall_signal = (
                imbalance_score * 0.25 +
                depth_score * 0.15 +
                volume_score * 0.10 +
                flow_score * 0.25 +
                spread_score * 0.05 +
                (vpin_score - 0.5) * 2.0 * 0.20
            )
            
            # Bound overall signal to [-1, 1]
            overall_signal = max(min(overall_signal, 1.0), -1.0)
            
            # Calculate confidence based on signal consistency
            confidence = self._calculate_signal_confidence([
                imbalance_score, depth_score, volume_score, flow_score, spread_score, vpin_score
            ])
            
            # Determine liquidity regime
            regime = self._classify_liquidity_regime(current_snapshot)
            
            return MicrostructureMetrics(
                order_book_imbalance=imbalance_score,
                depth_pressure=depth_score,
                volume_spike_score=volume_score,
                large_trade_flow=flow_score,
                spread_regime_score=spread_score,
                vpin=vpin_score,
                overall_signal=overall_signal,
                confidence=confidence,
                regime=regime
            )
            
        except Exception as e:
            self.logger.error(f"Error in microstructure analysis: {e}")
            return self._default_metrics()
    
    def _calculate_order_book_imbalance(self, snapshot: OrderBookSnapshot) -> float:
        """
        Calculate order book imbalance - Renaissance Technologies core signal
        
        Measures the relative strength of bid vs ask pressure across multiple levels
        """
        try:
            if not snapshot.bids or not snapshot.asks:
                return 0.0
            
            # Calculate depth-weighted imbalance across multiple levels
            total_bid_value = 0.0
            total_ask_value = 0.0
            
            # Analyze up to depth_levels on each side
            max_levels = min(self.depth_levels, len(snapshot.bids), len(snapshot.asks))
            
            for i in range(max_levels):
                # Weight diminishes with distance from best price
                level_weight = 1.0 / (1.0 + i * 0.5)
                
                bid_level = snapshot.bids[i]
                ask_level = snapshot.asks[i]
                
                # Dollar-weighted values
                bid_value = bid_level.price * bid_level.size * level_weight
                ask_value = ask_level.price * ask_level.size * level_weight
                
                total_bid_value += bid_value
                total_ask_value += ask_value
            
            # Calculate imbalance ratio
            if total_bid_value + total_ask_value > 0:
                imbalance = (total_bid_value - total_ask_value) / (total_bid_value + total_ask_value)
            else:
                imbalance = 0.0
            
            # Apply non-linear transformation for sensitivity
            return np.tanh(imbalance * 3.0)  # Scale and apply tanh
            
        except Exception as e:
            self.logger.error(f"Error calculating order book imbalance: {e}")
            return 0.0
    
    def _calculate_depth_pressure(self, snapshot: OrderBookSnapshot) -> float:
        """
        Calculate depth pressure - institutional order detection
        
        Analyzes the distribution and size of orders at different price levels
        """
        try:
            if len(self.order_book_history) < 2:
                return 0.0
            
            # Compare current vs historical depth
            current_bid_depth = sum(level.size for level in snapshot.bids[:self.depth_levels])
            current_ask_depth = sum(level.size for level in snapshot.asks[:self.depth_levels])
            
            # Historical average depth
            if len(self.order_book_history) > 10:
                recent_snapshots = list(self.order_book_history)[-10:]
                avg_bid_depth = np.mean([
                    sum(level.size for level in s.bids[:self.depth_levels])
                    for s in recent_snapshots if s.bids
                ])
                avg_ask_depth = np.mean([
                    sum(level.size for level in s.asks[:self.depth_levels])
                    for s in recent_snapshots if s.asks
                ])
            else:
                avg_bid_depth = current_bid_depth
                avg_ask_depth = current_ask_depth
            
            # Calculate depth pressure
            if avg_bid_depth > 0 and avg_ask_depth > 0:
                bid_pressure = (current_bid_depth - avg_bid_depth) / avg_bid_depth
                ask_pressure = (current_ask_depth - avg_ask_depth) / avg_ask_depth
                
                # Net depth pressure
                depth_pressure = bid_pressure - ask_pressure
            else:
                depth_pressure = 0.0
            
            # Apply sigmoid transformation
            return np.tanh(depth_pressure * 2.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating depth pressure: {e}")
            return 0.0
    
    def _calculate_volume_spike(self) -> float:
        """
        Detect volume spikes - abnormal trading activity
        
        Identifies sudden increases in trading volume that may indicate institutional activity
        """
        try:
            if len(self.trade_history) < self.volume_lookback:
                return 0.0
            
            recent_trades = list(self.trade_history)[-self.volume_lookback:]
            
            # Calculate recent volume metrics
            recent_volumes = [trade.size for trade in recent_trades[-10:]]  # Last 10 trades
            historical_volumes = [trade.size for trade in recent_trades[:-10]]  # Earlier trades
            
            if not recent_volumes or not historical_volumes:
                return 0.0
            
            recent_avg_volume = np.mean(recent_volumes)
            historical_avg_volume = np.mean(historical_volumes)
            historical_std_volume = np.std(historical_volumes)
            
            if historical_std_volume > 0 and historical_avg_volume > 0:
                # Z-score of recent volume vs historical
                volume_z_score = (recent_avg_volume - historical_avg_volume) / historical_std_volume
                
                # Convert to signal between -1 and 1
                return np.tanh(volume_z_score / 2.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating volume spike: {e}")
            return 0.0
    
    def _calculate_large_trade_flow(self) -> float:
        """
        Analyze large trade flow - institutional vs retail detection
        
        Identifies patterns in large trades that indicate institutional activity
        """
        try:
            if len(self.trade_history) < self.flow_lookback:
                return 0.0
            
            recent_trades = list(self.trade_history)[-self.flow_lookback:]
            
            # Classify trades by size
            large_buy_volume = 0.0
            large_sell_volume = 0.0
            total_buy_volume = 0.0
            total_sell_volume = 0.0
            
            for trade in recent_trades:
                trade_value = trade.price * trade.size
                
                if trade.side == 'buy':
                    total_buy_volume += trade.size
                    if trade_value > self.large_trade_threshold:
                        large_buy_volume += trade.size
                elif trade.side == 'sell':
                    total_sell_volume += trade.size
                    if trade_value > self.large_trade_threshold:
                        large_sell_volume += trade.size
            
            # Calculate large trade flow imbalance
            total_large_volume = large_buy_volume + large_sell_volume
            if total_large_volume > 0:
                large_flow_imbalance = (large_buy_volume - large_sell_volume) / total_large_volume
            else:
                large_flow_imbalance = 0.0
            
            # Weight by proportion of large trades
            total_volume = total_buy_volume + total_sell_volume
            if total_volume > 0:
                large_trade_proportion = total_large_volume / total_volume
                weighted_flow = large_flow_imbalance * min(large_trade_proportion * 2.0, 1.0)
            else:
                weighted_flow = 0.0
            
            return np.tanh(weighted_flow * 2.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating large trade flow: {e}")
            return 0.0
    
    def _calculate_spread_regime(self, snapshot: OrderBookSnapshot) -> float:
        """
        Classify spread regime - market making vs taking analysis
        
        Analyzes bid-ask spread behavior to determine market liquidity conditions
        """
        try:
            current_spread = snapshot.spread
            if current_spread <= 0:
                return 0.0
            
            # Calculate spread as percentage of mid price
            mid_price = snapshot.mid_price
            if mid_price > 0:
                spread_pct = current_spread / mid_price
            else:
                return 0.0
            
            # Historical spread analysis
            if len(self.order_book_history) > 20:
                recent_spreads = []
                for s in list(self.order_book_history)[-20:]:
                    if s.mid_price > 0 and s.spread > 0:
                        recent_spreads.append(s.spread / s.mid_price)
                
                if recent_spreads:
                    spread_percentiles = np.percentile(recent_spreads, [25, 75])
                    spread_25th, spread_75th = spread_percentiles
                    
                    # Classify spread regime
                    if spread_pct < spread_25th:
                        # Tight spread - good liquidity, positive for trading
                        regime_score = 0.5
                    elif spread_pct > spread_75th:
                        # Wide spread - poor liquidity, negative for trading
                        regime_score = -0.5
                    else:
                        # Normal spread
                        regime_score = 0.0
                else:
                    regime_score = 0.0
            else:
                regime_score = 0.0
            
            return regime_score
            
        except Exception as e:
            self.logger.error(f"Error calculating spread regime: {e}")
            return 0.0
    
    def _calculate_vpin(self) -> float:
        """
        Calculates Volume-Synchronized Probability of Informed Trading.
        Higher VPIN indicates higher toxicity and potential for price moves.
        """
        if len(self.vpin_buckets) < 10:
            return 0.5
        
        # VPIN = sum(abs(buy - sell)) / (n * bucket_size)
        total_imbalance = sum(self.vpin_buckets)
        vpin = total_imbalance / (len(self.vpin_buckets) * self.vpin_bucket_size)
        return float(min(max(vpin, 0.0), 1.0))

    def _calculate_signal_confidence(self, signals: List[float]) -> float:
        """
        Calculate overall signal confidence based on consistency
        
        Higher confidence when signals agree, lower when they conflict
        """
        try:
            if not signals:
                return 0.0
            
            # Remove zero signals for consistency calculation
            non_zero_signals = [s for s in signals if abs(s) > 0.01]
            
            if len(non_zero_signals) < 2:
                return 0.3  # Low confidence with few signals
            
            # Calculate signal consistency
            signal_signs = [1 if s > 0 else -1 for s in non_zero_signals]
            sign_consistency = abs(sum(signal_signs)) / len(signal_signs)
            
            # Calculate signal magnitude consistency
            signal_magnitudes = [abs(s) for s in non_zero_signals]
            magnitude_std = np.std(signal_magnitudes)
            magnitude_mean = np.mean(signal_magnitudes)
            
            if magnitude_mean > 0:
                magnitude_consistency = 1.0 - min(magnitude_std / magnitude_mean, 1.0)
            else:
                magnitude_consistency = 0.0
            
            # Combined confidence
            confidence = (sign_consistency * 0.6 + magnitude_consistency * 0.4)
            
            # Boost confidence for strong signals
            max_signal_strength = max(abs(s) for s in signals)
            strength_boost = min(max_signal_strength, 0.3)
            
            final_confidence = min(confidence + strength_boost, 1.0)
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {e}")
            return 0.0
    
    def _classify_liquidity_regime(self, snapshot: OrderBookSnapshot) -> LiquidityRegime:
        """Classify current market liquidity regime"""
        try:
            # Simple classification based on spread and depth
            spread_pct = snapshot.spread / snapshot.mid_price if snapshot.mid_price > 0 else 0
            
            total_depth = (
                sum(level.size for level in snapshot.bids[:5]) +
                sum(level.size for level in snapshot.asks[:5])
            )
            
            if spread_pct < 0.001 and total_depth > 1000:
                return LiquidityRegime.HIGH_LIQUIDITY
            elif spread_pct > 0.005 or total_depth < 100:
                return LiquidityRegime.LOW_LIQUIDITY
            elif spread_pct > 0.01:
                return LiquidityRegime.STRESS_LIQUIDITY
            else:
                return LiquidityRegime.NORMAL_LIQUIDITY
                
        except Exception as e:
            self.logger.error(f"Error classifying liquidity regime: {e}")
            return LiquidityRegime.NORMAL_LIQUIDITY
    
    def _default_metrics(self) -> MicrostructureMetrics:
        """Return default metrics in case of errors"""
        return MicrostructureMetrics(
            order_book_imbalance=0.0,
            depth_pressure=0.0,
            volume_spike_score=0.0,
            large_trade_flow=0.0,
            spread_regime_score=0.0,
            vpin=0.5,
            overall_signal=0.0,
            confidence=0.0,
            regime=LiquidityRegime.NORMAL_LIQUIDITY
        )
    
    def get_latest_metrics(self) -> Optional[MicrostructureMetrics]:
        """Get the most recent microstructure metrics"""
        with self.analysis_lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            return None
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get comprehensive microstructure signal summary"""
        try:
            latest = self.get_latest_metrics()
            if not latest:
                return {'status': 'no_data'}
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_signal': latest.overall_signal,
                'confidence': latest.confidence,
                'regime': latest.regime.value,
                'components': {
                    'order_book_imbalance': latest.order_book_imbalance,
                    'depth_pressure': latest.depth_pressure,
                    'volume_spike': latest.volume_spike_score,
                    'large_trade_flow': latest.large_trade_flow,
                    'spread_regime': latest.spread_regime_score
                },
                'data_points': {
                    'order_book_history': len(self.order_book_history),
                    'trade_history': len(self.trade_history),
                    'metrics_history': len(self.metrics_history)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal summary: {e}")
            return {'status': 'error', 'message': str(e)}

# Global microstructure engine instance
microstructure_engine = MicrostructureEngine()
