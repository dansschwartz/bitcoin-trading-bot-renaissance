"""
Order Book Data Collector & Real-Time Analyzer
Collects real-time order book data for microstructure analysis
"""

import asyncio
import json
import logging
import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import random

from microstructure_engine import (
    OrderBookSnapshot as _MicrostructureSnapshot,
    OrderBookLevel,
    TradeData,
    microstructure_engine,
)

logger = logging.getLogger(__name__)


# ── Existing OrderBookCollector (unchanged) ───────────────────────────

class OrderBookCollector:
    """
    Collects order book data for microstructure analysis
    Simulates real-time order book updates for paper trading
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.collection_thread = None

        # Simulation parameters for paper trading
        self.base_price = 59000.0  # BTC base price
        self.price_volatility = 0.002  # 0.2% volatility
        self.spread_base = 10.0  # Base spread in USD

        self.logger.info("Order Book Collector initialized")

    def start_collection(self):
        """Start collecting order book data"""
        if self.is_running:
            return

        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()

        self.logger.info("Order book collection started")

    def stop_collection(self):
        """Stop collecting order book data"""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)

        self.logger.info("Order book collection stopped")

    def _collection_loop(self):
        """Main collection loop"""
        while self.is_running:
            try:
                # Generate simulated order book
                snapshot = self._generate_order_book_snapshot()

                # Update microstructure engine
                metrics = microstructure_engine.update_order_book(snapshot)

                # Generate simulated trade
                trade = self._generate_simulated_trade()
                microstructure_engine.update_trade(trade)

                # Log microstructure signals periodically
                if datetime.now().second % 30 == 0:  # Every 30 seconds
                    self._log_microstructure_signals(metrics)

                # Wait before next update
                time.sleep(1)  # 1 second updates

            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(5)

    def _generate_order_book_snapshot(self) -> _MicrostructureSnapshot:
        """Generate realistic order book snapshot for simulation"""
        try:
            # Current price with random walk
            price_change = random.gauss(0, self.price_volatility)
            current_price = self.base_price * (1 + price_change)

            # Update base price slowly
            self.base_price = self.base_price * 0.999 + current_price * 0.001

            # Generate spread
            spread = self.spread_base * random.uniform(0.5, 2.0)

            # Best bid/ask
            best_bid_price = current_price - spread / 2
            best_ask_price = current_price + spread / 2

            # Generate bid levels
            bids = []
            for i in range(10):
                level_price = best_bid_price - (i * spread * 0.5)
                level_size = random.uniform(0.1, 5.0) * (1 + random.uniform(0, 2))
                bids.append(OrderBookLevel(price=level_price, size=level_size))

            # Generate ask levels
            asks = []
            for i in range(10):
                level_price = best_ask_price + (i * spread * 0.5)
                level_size = random.uniform(0.1, 5.0) * (1 + random.uniform(0, 2))
                asks.append(OrderBookLevel(price=level_price, size=level_size))

            return _MicrostructureSnapshot(
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                last_price=current_price,
                last_size=random.uniform(0.1, 2.0)
            )

        except Exception as e:
            self.logger.error(f"Error generating order book snapshot: {e}")
            return _MicrostructureSnapshot(
                timestamp=datetime.now(),
                bids=[OrderBookLevel(price=59000, size=1.0)],
                asks=[OrderBookLevel(price=59010, size=1.0)],
                last_price=59005,
                last_size=0.5
            )

    def _generate_simulated_trade(self) -> TradeData:
        """Generate simulated trade data"""
        side = random.choice(['buy', 'sell'])
        price = self.base_price * (1 + random.gauss(0, 0.001))
        size = random.uniform(0.01, 2.0)

        # Occasionally generate large trades
        if random.random() < 0.1:  # 10% chance
            size *= random.uniform(5, 20)

        return TradeData(
            timestamp=datetime.now(),
            price=price,
            size=size,
            side=side,
            trade_id=f"sim_{int(time.time())}_{random.randint(1000, 9999)}"
        )

    def _log_microstructure_signals(self, metrics):
        """Log microstructure signals"""
        try:
            self.logger.info(f"Microstructure Signals:")
            self.logger.info(f"   Order Book Imbalance: {metrics.order_book_imbalance:+.3f}")
            self.logger.info(f"   Depth Pressure: {metrics.depth_pressure:+.3f}")
            self.logger.info(f"   Volume Spike: {metrics.volume_spike_score:+.3f}")
            self.logger.info(f"   Large Trade Flow: {metrics.large_trade_flow:+.3f}")
            self.logger.info(f"   Spread Regime: {metrics.spread_regime_score:+.3f}")
            self.logger.info(f"   Overall Signal: {metrics.overall_signal:+.3f} (confidence: {metrics.confidence:.2%})")
            self.logger.info(f"   Liquidity Regime: {metrics.regime.value}")

        except Exception as e:
            self.logger.error(f"Error logging microstructure signals: {e}")

    def get_latest_microstructure_data(self) -> Dict:
        """Get latest microstructure analysis"""
        return microstructure_engine.get_signal_summary()

# Global order book collector instance
order_book_collector = OrderBookCollector()


# ── Real-Time Order Book Analyzer (Step 11) ──────────────────────────

class LiquidityRegime(Enum):
    """Liquidity regime classification"""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    CRISIS = "crisis"


@dataclass
class BookImbalance:
    """Order book imbalance metrics"""
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    ratio: float = 0.0
    direction: str = "neutral"


@dataclass
class OrderBookSnapshot:
    """Analyzed order book snapshot (Step 11)"""
    mid_price: float = 0.0
    spread: float = 0.0
    imbalance_ratio: float = 0.0
    liquidity_score: float = 0.0
    microstructure_signal: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderBookMetrics:
    """Advanced order book metrics"""
    effective_spread: float = 0.0
    resilience_score: float = 0.0
    toxicity_score: float = 0.0
    consciousness_enhancement: float = 0.0
    depth_imbalance: float = 0.0
    price_impact: float = 0.0


class RealTimeOrderBookAnalyzer:
    """
    Real-time order book analyzer with consciousness enhancement.

    Processes raw order book updates (bids/asks as price-size tuples),
    calculates microstructure metrics, and generates trading signals.
    """

    def __init__(self, max_levels: int = 20, consciousness_boost: float = 1.0,
                 analysis_window: int = 100):
        self.max_levels = max_levels
        self.consciousness_boost = consciousness_boost
        self.analysis_window = analysis_window
        self.logger = logging.getLogger(__name__)

        self.current_snapshot: Optional[OrderBookSnapshot] = None
        self.current_metrics: Optional[OrderBookMetrics] = None
        self.history: List[OrderBookSnapshot] = []
        self.metrics_history: List[OrderBookMetrics] = []
        self.start_time = time.time()
        self.updates_processed = 0

    def process_order_book_update(self, bids: List[Tuple], asks: List[Tuple]) -> OrderBookSnapshot:
        """Process an order book update and return analyzed snapshot"""
        consciousness = self.consciousness_boost

        # Extract best bid/ask
        if not bids or not asks:
            return self.current_snapshot or OrderBookSnapshot()

        best_bid_price = bids[0][0]
        best_ask_price = asks[0][0]
        mid_price = (best_bid_price + best_ask_price) / 2.0
        spread = best_ask_price - best_bid_price

        # Calculate volume imbalance
        bid_volume = sum(size for _, size in bids[:self.max_levels])
        ask_volume = sum(size for _, size in asks[:self.max_levels])
        total_volume = bid_volume + ask_volume

        if total_volume > 0:
            imbalance_ratio = (bid_volume - ask_volume) / total_volume
        else:
            imbalance_ratio = 0.0

        # Liquidity score based on depth
        depth_score = min(1.0, total_volume / 20.0)
        spread_score = max(0.0, 1.0 - (spread / mid_price) * 1000)
        liquidity_score = (depth_score * 0.6 + spread_score * 0.4)

        # Microstructure signal with consciousness enhancement
        flow_signal = imbalance_ratio * 0.5
        spread_signal = -spread / mid_price * 100
        microstructure_signal = (flow_signal + spread_signal * 0.3) * consciousness

        snapshot = OrderBookSnapshot(
            mid_price=mid_price,
            spread=spread,
            imbalance_ratio=imbalance_ratio,
            liquidity_score=liquidity_score,
            microstructure_signal=microstructure_signal,
            timestamp=time.time(),
        )

        self.current_snapshot = snapshot
        self.history.append(snapshot)
        if len(self.history) > self.analysis_window:
            self.history = self.history[-self.analysis_window:]

        # Calculate metrics
        self._calculate_metrics(bids, asks, snapshot)

        self.updates_processed += 1
        return snapshot

    def _calculate_metrics(self, bids: List[Tuple], asks: List[Tuple],
                           snapshot: OrderBookSnapshot):
        """Calculate advanced order book metrics"""
        consciousness = self.consciousness_boost

        # Effective spread (relative to mid)
        if snapshot.mid_price > 0:
            effective_spread = snapshot.spread / snapshot.mid_price * 10000  # in bps
        else:
            effective_spread = 0.0

        # Resilience score — how stable is the book
        if len(self.history) >= 3:
            recent_spreads = [h.spread for h in self.history[-5:]]
            spread_stability = 1.0 / (1.0 + np.std(recent_spreads) / max(np.mean(recent_spreads), 0.01))
            resilience_score = min(1.0, spread_stability * snapshot.liquidity_score)
        else:
            resilience_score = 0.5

        # Toxicity score — adverse selection risk
        if len(self.history) >= 3:
            recent_imbalances = [abs(h.imbalance_ratio) for h in self.history[-5:]]
            avg_imbalance = np.mean(recent_imbalances)
            toxicity_score = min(1.0, avg_imbalance * 1.5)
        else:
            toxicity_score = 0.3

        # Consciousness enhancement effect
        consciousness_enhancement = consciousness - 1.0

        # Depth imbalance
        bid_vol = sum(s for _, s in bids[:self.max_levels])
        ask_vol = sum(s for _, s in asks[:self.max_levels])
        depth_imbalance = (bid_vol - ask_vol) / max(bid_vol + ask_vol, 1.0)

        # Price impact estimate
        total_depth = bid_vol + ask_vol
        if total_depth > 0:
            price_impact = snapshot.spread / snapshot.mid_price / max(total_depth, 0.01)
        else:
            price_impact = 0.0

        self.current_metrics = OrderBookMetrics(
            effective_spread=effective_spread,
            resilience_score=resilience_score,
            toxicity_score=toxicity_score,
            consciousness_enhancement=consciousness_enhancement,
            depth_imbalance=depth_imbalance,
            price_impact=price_impact,
        )

        self.metrics_history.append(self.current_metrics)
        if len(self.metrics_history) > self.analysis_window:
            self.metrics_history = self.metrics_history[-self.analysis_window:]

    def get_current_analysis(self) -> Dict[str, Any]:
        """Get comprehensive current analysis report"""
        snapshot = self.current_snapshot
        metrics = self.current_metrics

        if snapshot is None:
            return {}

        # Classification
        if metrics and metrics.resilience_score > 0.7:
            liquidity_regime = LiquidityRegime.HIGH
        elif metrics and metrics.resilience_score > 0.4:
            liquidity_regime = LiquidityRegime.NORMAL
        elif metrics and metrics.resilience_score > 0.2:
            liquidity_regime = LiquidityRegime.LOW
        else:
            liquidity_regime = LiquidityRegime.CRISIS

        # Consciousness effectiveness
        if len(self.history) > 1:
            effectiveness = min(1.0, abs(self.history[-1].microstructure_signal) * 2)
        else:
            effectiveness = 0.0

        uptime = (time.time() - self.start_time)

        return {
            'snapshot': {
                'mid_price': snapshot.mid_price,
                'spread': snapshot.spread,
                'imbalance_ratio': snapshot.imbalance_ratio,
                'liquidity_score': snapshot.liquidity_score,
                'microstructure_signal': snapshot.microstructure_signal,
            },
            'metrics': {
                'effective_spread': metrics.effective_spread if metrics else 0.0,
                'resilience_score': metrics.resilience_score if metrics else 0.0,
                'toxicity_score': metrics.toxicity_score if metrics else 0.0,
            },
            'classification': {
                'liquidity_regime': liquidity_regime.value,
                'updates_processed': self.updates_processed,
            },
            'consciousness': {
                'factor': self.consciousness_boost,
                'effectiveness': effectiveness,
                'enhancement': self.consciousness_boost - 1.0,
            },
            'performance': {
                'uptime_seconds': uptime,
                'total_updates': self.updates_processed,
                'history_length': len(self.history),
            },
        }

    def predict_short_term_movement(self, horizon_seconds: float = 5.0) -> Dict[str, Any]:
        """Predict short-term price movement from order book dynamics"""
        consciousness = self.consciousness_boost

        if len(self.history) < 3:
            return {
                'prediction': 'NEUTRAL',
                'probability': 0.5,
                'confidence': 0.0,
                'consciousness_enhanced': True,
            }

        # Analyze recent imbalance trend
        recent = self.history[-min(10, len(self.history)):]
        imbalances = [h.imbalance_ratio for h in recent]

        avg_imbalance = np.mean(imbalances)
        imbalance_trend = imbalances[-1] - imbalances[0] if len(imbalances) > 1 else 0.0

        # Microstructure signal trend
        signals = [h.microstructure_signal for h in recent]
        avg_signal = np.mean(signals)

        # Combined directional score
        direction_score = (
            avg_imbalance * 0.4 +
            imbalance_trend * 0.3 +
            avg_signal * 0.3
        ) * consciousness

        # Determine prediction
        if direction_score > 0.05:
            prediction = 'UP'
            probability = min(0.95, 0.5 + abs(direction_score))
        elif direction_score < -0.05:
            prediction = 'DOWN'
            probability = min(0.95, 0.5 + abs(direction_score))
        else:
            prediction = 'NEUTRAL'
            probability = 0.5

        # Confidence based on data quality and consistency
        data_quality = min(1.0, len(self.history) / 20.0)
        consistency = 1.0 / (1.0 + np.std(imbalances) * 5)
        confidence = min(1.0, data_quality * 0.5 + consistency * 0.5) * min(consciousness, 1.5)
        confidence = min(1.0, max(0.0, confidence))

        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'consciousness_enhanced': True,
            'direction_score': direction_score,
            'horizon_seconds': horizon_seconds,
        }
