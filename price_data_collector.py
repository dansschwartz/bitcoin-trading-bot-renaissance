"""
Price Data Collector
Collects OHLCV price data for enhanced technical indicators analysis
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import time
import random

from enhanced_technical_indicators import enhanced_technical_indicators, PriceData, MultiTimeframeSignals

logger = logging.getLogger(__name__)

class PriceDataCollector:
    """
    Collects price data for technical indicators analysis
    Generates realistic OHLCV data for paper trading simulation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.collection_thread = None
        
        # Simulation parameters
        self.base_price = 59000.0
        self.volatility = 0.01  # 1% volatility
        self.volume_base = 100.0
        
        # Current candle data
        self.current_candle = {
            'open': self.base_price,
            'high': self.base_price,
            'low': self.base_price,
            'close': self.base_price,
            'volume': 0.0,
            'start_time': datetime.now()
        }
        
        # Candle interval (1 minute for fast response)
        self.candle_interval = 60  # seconds
        
        self.logger.info("✅ Price Data Collector initialized")
    
    def start_collection(self):
        """Start collecting price data"""
        if self.is_running:
            return
        
        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.logger.info("📊 Price data collection started")
    
    def stop_collection(self):
        """Stop collecting price data"""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        self.logger.info("🛑 Price data collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check if we need to complete current candle
                time_elapsed = (current_time - self.current_candle['start_time']).total_seconds()
                
                if time_elapsed >= self.candle_interval:
                    # Complete current candle and start new one
                    completed_candle = self._complete_candle()
                    
                    # Update technical indicators
                    signals = enhanced_technical_indicators.update_price_data(completed_candle)
                    
                    # Log technical signals periodically
                    if current_time.second % 60 == 0:  # Every minute
                        self._log_technical_signals(signals)
                    
                    # Start new candle
                    self._start_new_candle()
                
                # Update current candle with tick data
                self._update_current_candle()
                
                # Wait before next tick
                time.sleep(5)  # 5 second ticks
                
            except Exception as e:
                self.logger.error(f"Error in price collection loop: {e}")
                time.sleep(10)
    
    def _update_current_candle(self):
        """Update current candle with new tick"""
        try:
            # Generate price tick with random walk
            price_change = random.gauss(0, self.volatility / 100)  # Small tick change
            new_price = self.current_candle['close'] * (1 + price_change)
            
            # Update candle OHLC
            self.current_candle['high'] = max(self.current_candle['high'], new_price)
            self.current_candle['low'] = min(self.current_candle['low'], new_price)
            self.current_candle['close'] = new_price
            
            # Add volume
            tick_volume = random.uniform(0.1, 5.0)
            self.current_candle['volume'] += tick_volume
            
            # Update base price slowly (trend simulation)
            trend_factor = random.gauss(0, 0.0001)  # Very small trend
            self.base_price *= (1 + trend_factor)
            
        except Exception as e:
            self.logger.error(f"Error updating current candle: {e}")
    
    def _complete_candle(self) -> PriceData:
        """Complete current candle and return PriceData"""
        try:
            candle_data = PriceData(
                timestamp=self.current_candle['start_time'],
                open=self.current_candle['open'],
                high=self.current_candle['high'],
                low=self.current_candle['low'],
                close=self.current_candle['close'],
                volume=self.current_candle['volume']
            )
            
            return candle_data
            
        except Exception as e:
            self.logger.error(f"Error completing candle: {e}")
            # Return default candle
            return PriceData(
                timestamp=datetime.now(),
                open=self.base_price,
                high=self.base_price,
                low=self.base_price,
                close=self.base_price,
                volume=100.0
            )
    
    def _start_new_candle(self):
        """Start a new candle"""
        current_price = self.current_candle['close']
        
        self.current_candle = {
            'open': current_price,
            'high': current_price,
            'low': current_price,
            'close': current_price,
            'volume': 0.0,
            'start_time': datetime.now()
        }
    
    def _log_technical_signals(self, signals: MultiTimeframeSignals):
        """Log technical indicators signals"""
        try:
            self.logger.info(f"📈 Enhanced Technical Indicators:")
            self.logger.info(f"   🔄 Fast RSI (7): {signals.fast_rsi.value:.1f} - {signals.fast_rsi.signal} ({signals.fast_rsi.strength:.2f})")
            self.logger.info(f"   📊 Quick MACD (5/13): {signals.quick_macd.value:+.3f} - {signals.quick_macd.signal} ({signals.quick_macd.strength:.2f})")
            self.logger.info(f"   📉 Dynamic Bollinger: {signals.dynamic_bollinger.value:.3f} - {signals.dynamic_bollinger.signal} ({signals.dynamic_bollinger.strength:.2f})")
            self.logger.info(f"   📈 OBV Momentum: {signals.obv_momentum.value:+.3f} - {signals.obv_momentum.signal} ({signals.obv_momentum.strength:.2f})")
            self.logger.info(f"   🎯 Combined Signal: {signals.combined_signal:+.3f} (confidence: {signals.confidence:.2%})")
            self.logger.info(f"   📈 Trend: {signals.trend_direction.value}")
            self.logger.info(f"   📊 Volatility: {signals.volatility_regime.value}")
            
        except Exception as e:
            self.logger.error(f"Error logging technical signals: {e}")
    
    def get_latest_technical_data(self) -> Dict:
        """Get latest technical analysis"""
        return enhanced_technical_indicators.get_signals_summary()
    
    def get_current_price(self) -> float:
        """Get current price"""
        return self.current_candle['close']

# Global price data collector instance
price_data_collector = PriceDataCollector()
