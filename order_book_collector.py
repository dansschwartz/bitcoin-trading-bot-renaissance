"""
Order Book Data Collector
Collects real-time order book data for microstructure analysis
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import threading
import time
import random

from microstructure_engine import OrderBookSnapshot, OrderBookLevel, TradeData, microstructure_engine

logger = logging.getLogger(__name__)

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
        
        self.logger.info("✅ Order Book Collector initialized")
    
    def start_collection(self):
        """Start collecting order book data"""
        if self.is_running:
            return
        
        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.logger.info("📊 Order book collection started")
    
    def stop_collection(self):
        """Stop collecting order book data"""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        self.logger.info("🛑 Order book collection stopped")
    
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
    
    def _generate_order_book_snapshot(self) -> OrderBookSnapshot:
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
                level_size = random.uniform(0.1, 5.0) * (1 + random.uniform(0, 2))  # Varying sizes
                bids.append(OrderBookLevel(price=level_price, size=level_size))
            
            # Generate ask levels
            asks = []
            for i in range(10):
                level_price = best_ask_price + (i * spread * 0.5)
                level_size = random.uniform(0.1, 5.0) * (1 + random.uniform(0, 2))  # Varying sizes
                asks.append(OrderBookLevel(price=level_price, size=level_size))
            
            return OrderBookSnapshot(
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                last_price=current_price,
                last_size=random.uniform(0.1, 2.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error generating order book snapshot: {e}")
            # Return minimal snapshot
            return OrderBookSnapshot(
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
            size *= random.uniform(5, 20)  # Large trade
        
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
            self.logger.info(f"🔬 Microstructure Signals:")
            self.logger.info(f"   📊 Order Book Imbalance: {metrics.order_book_imbalance:+.3f}")
            self.logger.info(f"   📈 Depth Pressure: {metrics.depth_pressure:+.3f}")
            self.logger.info(f"   ⚡ Volume Spike: {metrics.volume_spike_score:+.3f}")
            self.logger.info(f"   🌊 Large Trade Flow: {metrics.large_trade_flow:+.3f}")
            self.logger.info(f"   📉 Spread Regime: {metrics.spread_regime_score:+.3f}")
            self.logger.info(f"   🎯 Overall Signal: {metrics.overall_signal:+.3f} (confidence: {metrics.confidence:.2%})")
            self.logger.info(f"   💧 Liquidity Regime: {metrics.regime.value}")
            
        except Exception as e:
            self.logger.error(f"Error logging microstructure signals: {e}")
    
    def get_latest_microstructure_data(self) -> Dict:
        """Get latest microstructure analysis"""
        return microstructure_engine.get_signal_summary()

# Global order book collector instance
order_book_collector = OrderBookCollector()
