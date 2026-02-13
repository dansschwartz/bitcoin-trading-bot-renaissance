"""
Enhanced Technical Indicators System
Renaissance Technologies-style fast-response indicators with adaptive parameters.
Advanced technical analysis with multi-timeframe fusion and dynamic optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
from enum import Enum
import threading
import math

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction classification"""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW_VOLATILITY = "low_volatility"
    NORMAL_VOLATILITY = "normal_volatility"
    HIGH_VOLATILITY = "high_volatility"
    EXTREME_VOLATILITY = "extreme_volatility"

@dataclass
class PriceData:
    """OHLCV price data point"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class IndicatorOutput(NamedTuple):
    """Technical indicator signal result"""
    value: float
    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]

class MultiTimeframeSignals(NamedTuple):
    """Multi-timeframe combined signals"""
    fast_rsi: IndicatorOutput
    quick_macd: IndicatorOutput
    dynamic_bollinger: IndicatorOutput
    obv_momentum: IndicatorOutput
    hurst_exponent: float
    combined_signal: float
    confidence: float
    trend_direction: TrendDirection
    volatility_regime: VolatilityRegime

class TechnicalSignal(NamedTuple):
    """Aggregated indicator strengths used in tests"""
    rsi_strength: float
    macd_strength: float
    bollinger_strength: float
    volume_strength: float
    confidence: float
    timestamp: datetime

class EnhancedTechnicalIndicators:
    """
    Enhanced Technical Indicators System
    Renaissance Technologies-style fast-response technical analysis
    """
    
    def __init__(self, max_history: int = 500):
        self.logger = logging.getLogger(__name__)
        self.max_history = max_history
        
        # Price data storage
        self.price_history: deque = deque(maxlen=max_history)
        self.signals_history: deque = deque(maxlen=100)
        
        # Indicator parameters - Renaissance Technologies optimized
        self.rsi_period = 7  # Fast RSI
        self.macd_fast = 5   # Quick MACD
        self.macd_slow = 13
        self.macd_signal = 4
        self.bb_period = 14  # Dynamic Bollinger Bands
        self.bb_std_dev = 2.0
        self.obv_lookback = 20
        
        # Adaptive parameters
        self.volatility_lookback = 20
        self.trend_lookback = 30
        self.adaptive_mode = True
        
        # Multi-timeframe weights
        self.timeframe_weights = {
            'short': 0.4,   # 1-5 periods
            'medium': 0.4,  # 5-20 periods  
            'long': 0.2     # 20+ periods
        }
        
        # Threading
        self.analysis_lock = threading.Lock()
        
        # Performance tracking
        self.signal_performance: Dict[str, List[float]] = {
            'rsi': [], 'macd': [], 'bollinger': [], 'obv': []
        }
        
        self.logger.info("✅ Enhanced Technical Indicators initialized")
    
    def update_price_data(self, price_data: PriceData) -> MultiTimeframeSignals:
        """
        Update with new price data and generate enhanced technical signals
        
        Args:
            price_data: New OHLCV price data
            
        Returns:
            MultiTimeframeSignals with all enhanced indicators
        """
        try:
            with self.analysis_lock:
                # Store price data
                self.price_history.append(price_data)
                
                # Generate all technical signals
                signals = self._generate_all_signals()
                
                # Store signals
                self.signals_history.append(signals)
                
                # Log significant signals
                if signals.confidence > 0.7:
                    self.logger.info(f"📈 Strong technical signal: {signals.combined_signal:+.3f} "
                                   f"(confidence: {signals.confidence:.2%})")
                
                return signals
                
        except Exception as e:
            self.logger.error(f"Error updating technical indicators: {e}")
            return self._default_signals()
    
    def _generate_all_signals(self) -> MultiTimeframeSignals:
        """Generate all enhanced technical indicators"""
        try:
            if len(self.price_history) < self.macd_slow + 5:
                return self._default_signals()
            
            # Convert to pandas for easier calculation
            df = self._to_dataframe()
            
            # 1. Fast Adaptive RSI (7-period)
            rsi_signal = self._calculate_fast_adaptive_rsi(df)
            
            # 2. Quick MACD (5/13 periods)
            macd_signal = self._calculate_quick_macd(df)
            
            # 3. Dynamic Bollinger Bands
            bb_signal = self._calculate_dynamic_bollinger_bands(df)
            
            # 4. OBV Momentum
            obv_signal = self._calculate_obv_momentum(df)
            
            # 5. Hurst Exponent (Fractal Regime)
            hurst_exp = self._calculate_hurst_exponent(df)
            
            # 6. Multi-timeframe fusion
            combined_signal, confidence = self._fuse_multitimeframe_signals([
                rsi_signal, macd_signal, bb_signal, obv_signal
            ])
            
            # Adjust combined signal based on Hurst Exponent
            # H < 0.5: Mean-reverting -> favor RSI/BB
            # H > 0.5: Trending -> favor MACD/OBV
            if hurst_exp > 0.55:
                # Trending: Amplify MACD/OBV, suppress RSI/BB
                combined_signal = (macd_signal.strength * 0.4 + obv_signal.strength * 0.4 + combined_signal * 0.2)
            elif hurst_exp < 0.45:
                # Mean-reverting: Amplify RSI/BB, suppress MACD/OBV
                combined_signal = (rsi_signal.strength * 0.4 + bb_signal.strength * 0.4 + combined_signal * 0.2)
            
            # 7. Trend and volatility classification
            trend_direction = self._classify_trend_direction(df)
            volatility_regime = self._classify_volatility_regime(df)
            
            return MultiTimeframeSignals(
                fast_rsi=rsi_signal,
                quick_macd=macd_signal,
                dynamic_bollinger=bb_signal,
                obv_momentum=obv_signal,
                hurst_exponent=hurst_exp,
                combined_signal=combined_signal,
                confidence=confidence,
                trend_direction=trend_direction,
                volatility_regime=volatility_regime
            )
            
        except Exception as e:
            self.logger.error(f"Error generating technical signals: {e}")
            return self._default_signals()
    
    def _calculate_fast_adaptive_rsi(self, df: pd.DataFrame) -> IndicatorOutput:
        """
        Calculate fast adaptive RSI with Renaissance Technologies enhancements
        
        Uses adaptive period based on market volatility
        """
        try:
            # Calculate volatility-adjusted period
            if self.adaptive_mode and len(df) > self.volatility_lookback:
                volatility = df['close'].rolling(self.volatility_lookback).std().iloc[-1]
                vol_percentile = self._calculate_volatility_percentile(df, volatility)
                
                # Adjust RSI period based on volatility
                if vol_percentile > 0.8:  # High volatility - shorter period
                    rsi_period = max(5, self.rsi_period - 2)
                elif vol_percentile < 0.2:  # Low volatility - longer period
                    rsi_period = min(14, self.rsi_period + 2)
                else:
                    rsi_period = self.rsi_period
            else:
                rsi_period = self.rsi_period
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Enhanced signal generation
            if current_rsi > 75:  # Overbought - tighter than traditional 70
                signal_type = 'SELL'
                strength = min((current_rsi - 75) / 25, 1.0)
            elif current_rsi < 25:  # Oversold - tighter than traditional 30
                signal_type = 'BUY' 
                strength = min((25 - current_rsi) / 25, 1.0)
            elif current_rsi > 65:  # Moderate overbought
                signal_type = 'SELL'
                strength = (current_rsi - 65) / 10 * 0.5
            elif current_rsi < 35:  # Moderate oversold
                signal_type = 'BUY'
                strength = (35 - current_rsi) / 10 * 0.5
            else:
                signal_type = 'HOLD'
                strength = 0.0
            
            # Calculate confidence based on RSI momentum
            rsi_momentum = rsi.diff().iloc[-1] if len(rsi) > 1 else 0
            confidence = min(abs(current_rsi - 50) / 50 + abs(rsi_momentum) / 10, 1.0)
            
            return IndicatorOutput(
                value=current_rsi,
                signal=signal_type,
                strength=strength,
                confidence=confidence,
                metadata={
                    'period': rsi_period,
                    'momentum': rsi_momentum,
                    'adaptive': self.adaptive_mode
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating fast adaptive RSI: {e}")
            return IndicatorOutput(50.0, 'HOLD', 0.0, 0.0, {})
    
    def _calculate_quick_macd(self, df: pd.DataFrame) -> IndicatorOutput:
        """
        Calculate quick MACD with Renaissance Technologies parameters
        
        Uses faster periods (5/13 vs traditional 12/26) for reduced lag
        """
        try:
            # Calculate MACD with quick parameters
            ema_fast = df['close'].ewm(span=self.macd_fast).mean()
            ema_slow = df['close'].ewm(span=self.macd_slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.macd_signal).mean()
            histogram = macd_line - signal_line
            
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_hist = histogram.iloc[-1]
            
            # Enhanced signal generation with trend confirmation
            if len(histogram) > 1:
                hist_momentum = histogram.diff().iloc[-1]
                macd_momentum = macd_line.diff().iloc[-1]
            else:
                hist_momentum = 0
                macd_momentum = 0
            
            # Multi-condition signal
            if (current_macd > current_signal and 
                current_hist > 0 and 
                hist_momentum > 0):
                signal_type = 'BUY'
                strength = min(abs(current_hist) / df['close'].std() + hist_momentum / 10, 1.0)
            elif (current_macd < current_signal and 
                  current_hist < 0 and 
                  hist_momentum < 0):
                signal_type = 'SELL'
                strength = min(abs(current_hist) / df['close'].std() + abs(hist_momentum) / 10, 1.0)
            else:
                signal_type = 'HOLD'
                strength = 0.0
            
            # Confidence based on signal clarity and momentum
            signal_clarity = abs(current_hist) / df['close'].std() if df['close'].std() > 0 else 0
            momentum_strength = abs(hist_momentum) / 10
            confidence = min(signal_clarity + momentum_strength, 1.0)
            
            return IndicatorOutput(
                value=current_macd,
                signal=signal_type,
                strength=strength,
                confidence=confidence,
                metadata={
                    'signal_line': current_signal,
                    'histogram': current_hist,
                    'histogram_momentum': hist_momentum,
                    'macd_momentum': macd_momentum,
                    'fast_period': self.macd_fast,
                    'slow_period': self.macd_slow
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating quick MACD: {e}")
            return IndicatorOutput(0.0, 'HOLD', 0.0, 0.0, {})
    
    def _calculate_dynamic_bollinger_bands(self, df: pd.DataFrame) -> IndicatorOutput:
        """
        Calculate dynamic Bollinger Bands with adaptive parameters
        
        Adjusts standard deviation multiplier based on market volatility
        """
        try:
            # Calculate base Bollinger Bands
            sma = df['close'].rolling(window=self.bb_period).mean()
            std = df['close'].rolling(window=self.bb_period).std()
            
            # Dynamic standard deviation multiplier
            if self.adaptive_mode and len(df) > self.volatility_lookback:
                current_vol = std.iloc[-1]
                vol_percentile = self._calculate_volatility_percentile(df, current_vol)
                
                # Adjust standard deviation multiplier
                if vol_percentile > 0.8:  # High volatility - wider bands
                    std_multiplier = self.bb_std_dev + 0.5
                elif vol_percentile < 0.2:  # Low volatility - tighter bands
                    std_multiplier = max(1.5, self.bb_std_dev - 0.5)
                else:
                    std_multiplier = self.bb_std_dev
            else:
                std_multiplier = self.bb_std_dev
            
            upper_band = sma + (std * std_multiplier)
            lower_band = sma - (std * std_multiplier)
            
            current_price = df['close'].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_sma = sma.iloc[-1]
            
            # Calculate position within bands
            band_width = current_upper - current_lower
            if band_width > 0:
                position = (current_price - current_lower) / band_width
            else:
                position = 0.5
            
            # Enhanced signal generation with squeeze detection
            band_squeeze = self._detect_bollinger_squeeze(std, self.bb_period)
            
            if position > 0.85:  # Near upper band
                signal_type = 'SELL'
                strength = min((position - 0.85) / 0.15, 1.0)
            elif position < 0.15:  # Near lower band
                signal_type = 'BUY'
                strength = min((0.15 - position) / 0.15, 1.0)
            elif band_squeeze and position > 0.6:  # Squeeze breakout up
                signal_type = 'BUY'
                strength = 0.6
            elif band_squeeze and position < 0.4:  # Squeeze breakout down
                signal_type = 'SELL'
                strength = 0.6
            else:
                signal_type = 'HOLD'
                strength = 0.0
            
            # Confidence based on band position and volatility
            position_clarity = abs(position - 0.5) * 2  # 0 to 1
            squeeze_confidence = 0.3 if band_squeeze else 0.0
            confidence = min(position_clarity + squeeze_confidence, 1.0)
            
            return IndicatorOutput(
                value=position,
                signal=signal_type,
                strength=strength,
                confidence=confidence,
                metadata={
                    'upper_band': current_upper,
                    'lower_band': current_lower,
                    'sma': current_sma,
                    'band_width': band_width,
                    'std_multiplier': std_multiplier,
                    'squeeze': band_squeeze,
                    'adaptive': self.adaptive_mode
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic Bollinger Bands: {e}")
            return IndicatorOutput(0.5, 'HOLD', 0.0, 0.0, {})
    
    def _calculate_obv_momentum(self, df: pd.DataFrame) -> IndicatorOutput:
        """
        Calculate On-Balance Volume momentum with Renaissance Technologies enhancements
        
        Analyzes volume-price relationship for institutional flow detection
        """
        try:
            # Calculate On-Balance Volume
            obv = [0]
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.append(obv[-1] + df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.append(obv[-1] - df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            
            obv_series = pd.Series(obv, index=df.index)
            
            # Calculate OBV momentum
            obv_sma = obv_series.rolling(window=self.obv_lookback).mean()
            current_obv = obv_series.iloc[-1]
            current_obv_sma = obv_sma.iloc[-1]
            
            # OBV relative position
            if current_obv_sma != 0:
                obv_position = (current_obv - current_obv_sma) / abs(current_obv_sma)
            else:
                obv_position = 0
            
            # OBV momentum (rate of change)
            if len(obv_series) > 5:
                obv_momentum = (obv_series.iloc[-1] - obv_series.iloc[-6]) / abs(obv_series.iloc[-6])
            else:
                obv_momentum = 0
            
            # Price-OBV divergence analysis
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-self.obv_lookback]) / df['close'].iloc[-self.obv_lookback]
            obv_change = (obv_series.iloc[-1] - obv_series.iloc[-self.obv_lookback]) / abs(obv_series.iloc[-self.obv_lookback])
            
            # Detect divergence
            divergence = 0
            if price_change > 0.02 and obv_change < -0.02:  # Bearish divergence
                divergence = -1
            elif price_change < -0.02 and obv_change > 0.02:  # Bullish divergence
                divergence = 1
            
            # Generate signal
            if obv_momentum > 0.05 and obv_position > 0.1:
                signal_type = 'BUY'
                strength = min(obv_momentum + obv_position, 1.0)
            elif obv_momentum < -0.05 and obv_position < -0.1:
                signal_type = 'SELL'
                strength = min(abs(obv_momentum) + abs(obv_position), 1.0)
            elif divergence != 0:
                signal_type = 'BUY' if divergence > 0 else 'SELL'
                strength = 0.7  # Divergence signals are strong
            else:
                signal_type = 'HOLD'
                strength = 0.0
            
            # Confidence based on momentum clarity and volume consistency
            momentum_clarity = min(abs(obv_momentum) * 5, 1.0)
            position_clarity = min(abs(obv_position), 1.0)
            divergence_boost = 0.3 if divergence != 0 else 0.0
            confidence = min(momentum_clarity + position_clarity + divergence_boost, 1.0)
            
            return IndicatorOutput(
                value=obv_position,
                signal=signal_type,
                strength=strength,
                confidence=confidence,
                metadata={
                    'obv': current_obv,
                    'obv_sma': current_obv_sma,
                    'obv_momentum': obv_momentum,
                    'divergence': divergence,
                    'price_change': price_change,
                    'obv_change': obv_change
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating OBV momentum: {e}")
            return IndicatorOutput(0.0, 'HOLD', 0.0, 0.0, {})
    
    def _calculate_hurst_exponent(self, df: pd.DataFrame, window: int = 100) -> float:
        """
        Calculates Hurst Exponent to identify fractal regimes.
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        try:
            if len(df) < 20:
                return 0.5
            
            actual_window = min(window, len(df))
            prices = df['close'].values[-actual_window:]
            
            # Simple R/S analysis approximation
            lags = range(2, actual_window // 2)
            tau = []
            for lag in lags:
                diffs = np.subtract(prices[lag:], prices[:-lag])
                if len(diffs) > 0:
                    tau.append(np.std(diffs))
            
            if len(tau) < 2:
                return 0.5
                
            # Log-log linear regression
            m = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
            hurst = m[0]
            
            return float(np.clip(hurst, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating Hurst Exponent: {e}")
            return 0.5

    def _fuse_multitimeframe_signals(self, signals: List[IndicatorOutput]) -> Tuple[float, float]:
        """
        Fuse multiple technical signals with Renaissance Technologies weighting
        
        Combines signals from different indicators with adaptive weighting
        """
        try:
            if not signals:
                return 0.0, 0.0
            
            # Renaissance Technologies signal weights
            weights = {
                'rsi': 0.25,      # Fast RSI
                'macd': 0.35,     # Quick MACD (higher weight for trend)
                'bollinger': 0.25, # Dynamic Bollinger
                'obv': 0.15       # OBV momentum
            }
            
            # Convert signals to numerical values
            signal_values = []
            confidence_values = []
            
            for i, signal in enumerate(signals):
                # Convert signal to numerical value
                if signal.signal == 'BUY':
                    signal_value = signal.strength
                elif signal.signal == 'SELL':
                    signal_value = -signal.strength
                else:
                    signal_value = 0.0
                
                signal_values.append(signal_value)
                confidence_values.append(signal.confidence)
            
            # Weighted combination
            weight_list = list(weights.values())
            combined_signal = sum(sv * w for sv, w in zip(signal_values, weight_list))
            
            # Adaptive confidence calculation
            avg_confidence = sum(cv * w for cv, w in zip(confidence_values, weight_list))
            
            # Signal agreement boost
            positive_signals = sum(1 for sv in signal_values if sv > 0.1)
            negative_signals = sum(1 for sv in signal_values if sv < -0.1)
            agreement = max(positive_signals, negative_signals) / len(signal_values)
            
            # Final confidence with agreement boost
            final_confidence = min(avg_confidence + (agreement - 0.5) * 0.3, 1.0)
            
            return combined_signal, final_confidence
            
        except Exception as e:
            self.logger.error(f"Error fusing multitimeframe signals: {e}")
            return 0.0, 0.0
    
    def _classify_trend_direction(self, df: pd.DataFrame) -> TrendDirection:
        """Classify overall trend direction"""
        try:
            if len(df) < self.trend_lookback:
                return TrendDirection.SIDEWAYS
            
            # Calculate trend metrics
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-self.trend_lookback]) / df['close'].iloc[-self.trend_lookback]
            
            # Simple moving averages for trend
            sma_short = df['close'].rolling(5).mean().iloc[-1]
            sma_long = df['close'].rolling(20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Classify trend
            if price_change > 0.05 and current_price > sma_short > sma_long:
                return TrendDirection.STRONG_UPTREND
            elif price_change > 0.02 and current_price > sma_short:
                return TrendDirection.UPTREND
            elif price_change < -0.05 and current_price < sma_short < sma_long:
                return TrendDirection.STRONG_DOWNTREND
            elif price_change < -0.02 and current_price < sma_short:
                return TrendDirection.DOWNTREND
            else:
                return TrendDirection.SIDEWAYS
                
        except Exception as e:
            self.logger.error(f"Error classifying trend direction: {e}")
            return TrendDirection.SIDEWAYS
    
    def _classify_volatility_regime(self, df: pd.DataFrame) -> VolatilityRegime:
        """Classify volatility regime"""
        try:
            if len(df) < self.volatility_lookback:
                return VolatilityRegime.NORMAL_VOLATILITY
            
            # Calculate current volatility
            returns = df['close'].pct_change()
            current_vol = returns.rolling(self.volatility_lookback).std().iloc[-1]
            
            # Historical volatility percentiles
            vol_percentile = self._calculate_volatility_percentile(df, current_vol)
            
            if vol_percentile > 0.9:
                return VolatilityRegime.EXTREME_VOLATILITY
            elif vol_percentile > 0.7:
                return VolatilityRegime.HIGH_VOLATILITY
            elif vol_percentile < 0.3:
                return VolatilityRegime.LOW_VOLATILITY
            else:
                return VolatilityRegime.NORMAL_VOLATILITY
                
        except Exception as e:
            self.logger.error(f"Error classifying volatility regime: {e}")
            return VolatilityRegime.NORMAL_VOLATILITY
    
    def _calculate_volatility_percentile(self, df: pd.DataFrame, current_vol: float) -> float:
        """Calculate volatility percentile"""
        try:
            returns = df['close'].pct_change().dropna()
            if len(returns) < 50:
                return 0.5
            
            # Rolling volatilities
            rolling_vols = returns.rolling(self.volatility_lookback).std().dropna()
            
            if len(rolling_vols) == 0:
                return 0.5
            
            percentile = (rolling_vols < current_vol).mean()
            return percentile
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility percentile: {e}")
            return 0.5
    
    def _detect_bollinger_squeeze(self, std_series: pd.Series, period: int) -> bool:
        """Detect Bollinger Band squeeze (low volatility)"""
        try:
            if len(std_series) < period * 2:
                return False
            
            current_std = std_series.iloc[-1]
            avg_std = std_series.rolling(period * 2).mean().iloc[-1]
            
            # Squeeze when current volatility is significantly below average
            return current_std < avg_std * 0.7
            
        except Exception as e:
            self.logger.error(f"Error detecting Bollinger squeeze: {e}")
            return False
    
    def _to_dataframe(self) -> pd.DataFrame:
        """Convert price history to pandas DataFrame"""
        if not self.price_history:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
        data = []
        for price_data in self.price_history:
            data.append({
                'timestamp': price_data.timestamp,
                'open': price_data.open,
                'high': price_data.high,
                'low': price_data.low,
                'close': price_data.close,
                'volume': price_data.volume
            })
        
        df = pd.DataFrame(data)
        if not df.empty and 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        return df
    
    def _default_signals(self) -> MultiTimeframeSignals:
        """Return default signals in case of errors"""
        default_signal = IndicatorOutput(0.0, 'HOLD', 0.0, 0.0, {})
        
        return MultiTimeframeSignals(
            fast_rsi=default_signal,
            quick_macd=default_signal,
            dynamic_bollinger=default_signal,
            obv_momentum=default_signal,
            hurst_exponent=0.5,
            combined_signal=0.0,
            confidence=0.0,
            trend_direction=TrendDirection.SIDEWAYS,
            volatility_regime=VolatilityRegime.NORMAL_VOLATILITY
        )
    
    def get_latest_signals(self) -> Optional[MultiTimeframeSignals]:
        """Get the most recent technical signals"""
        with self.analysis_lock:
            if self.signals_history:
                return self.signals_history[-1]
            return None
    
    async def calculate_enhanced_signals(self, df: pd.DataFrame) -> TechnicalSignal:
        """Compatibility stub for tests; returns aggregated strengths.
        In production, use `update_price_data` and `get_latest_signals`.
        """
        # Minimal placeholder; tests patch this method.
        from datetime import timezone
        return TechnicalSignal(
            rsi_strength=0.0,
            macd_strength=0.0,
            bollinger_strength=0.0,
            volume_strength=0.0,
            confidence=0.0,
            timestamp=datetime.now(timezone.utc)
        )
    
    def get_signals_summary(self) -> Dict[str, Any]:
        """Get comprehensive technical signals summary"""
        try:
            latest = self.get_latest_signals()
            if not latest:
                return {'status': 'no_data'}
            
            return {
                'timestamp': datetime.now().isoformat(),
                'combined_signal': latest.combined_signal,
                'confidence': latest.confidence,
                'hurst_exponent': latest.hurst_exponent,
                'trend_direction': latest.trend_direction.value,
                'volatility_regime': latest.volatility_regime.value,
                'indicators': {
                    'fast_rsi': {
                        'value': latest.fast_rsi.value,
                        'signal': latest.fast_rsi.signal,
                        'strength': latest.fast_rsi.strength,
                        'confidence': latest.fast_rsi.confidence
                    },
                    'quick_macd': {
                        'value': latest.quick_macd.value,
                        'signal': latest.quick_macd.signal,
                        'strength': latest.quick_macd.strength,
                        'confidence': latest.quick_macd.confidence
                    },
                    'dynamic_bollinger': {
                        'value': latest.dynamic_bollinger.value,
                        'signal': latest.dynamic_bollinger.signal,
                        'strength': latest.dynamic_bollinger.strength,
                        'confidence': latest.dynamic_bollinger.confidence
                    },
                    'obv_momentum': {
                        'value': latest.obv_momentum.value,
                        'signal': latest.obv_momentum.signal,
                        'strength': latest.obv_momentum.strength,
                        'confidence': latest.obv_momentum.confidence
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signals summary: {e}")
            return {'status': 'error', 'message': str(e)}

# Global enhanced technical indicators instance
enhanced_technical_indicators = EnhancedTechnicalIndicators()
