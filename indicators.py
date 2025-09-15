"""
Enhanced Technical Indicators Module for Renaissance Technologies Trading Bot
Production-ready technical analysis with advanced order flow and regime detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"


@dataclass
class IndicatorResult:
    """Container for indicator calculation results"""
    value: float
    signal: str
    strength: float
    confidence: float
    metadata: Dict[str, Any] = None


class TechnicalIndicators:
    """
    Renaissance Technologies Enhanced Technical Indicators
    Includes traditional indicators with advanced order flow analysis
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> IndicatorResult:
        """
        Calculate Relative Strength Index with Renaissance enhancements
        """
        try:
            if len(prices) < period + 1:
                return IndicatorResult(50.0, "HOLD", 0.0, 0.1)

            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            current_rsi = rsi.iloc[-1]

            # Renaissance-style signal interpretation
            if current_rsi > 70:
                signal = "SELL"
                strength = min((current_rsi - 70) / 20, 1.0)
            elif current_rsi < 30:
                signal = "BUY"
                strength = min((30 - current_rsi) / 20, 1.0)
            else:
                signal = "HOLD"
                strength = 0.0

            # Confidence based on RSI momentum
            rsi_momentum = rsi.diff().iloc[-1]
            confidence = min(abs(rsi_momentum) / 5.0, 1.0)

            return IndicatorResult(
                value=current_rsi,
                signal=signal,
                strength=strength,
                confidence=confidence,
                metadata={'rsi_momentum': rsi_momentum}
            )

        except Exception as e:
            self.logger.error(f"RSI calculation error: {e}")
            return IndicatorResult(50.0, "HOLD", 0.0, 0.1)

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26,
                       signal_period: int = 9) -> IndicatorResult:
        """
        Calculate MACD with Renaissance Technologies enhancements
        """
        try:
            if len(prices) < slow + signal_period:
                return IndicatorResult(0.0, "HOLD", 0.0, 0.1)

            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period).mean()
            histogram = macd_line - signal_line

            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_hist = histogram.iloc[-1]

            # Renaissance-style signal interpretation
            if current_macd > current_signal and current_hist > 0:
                signal = "BUY"
                strength = min(abs(current_hist) * 1000, 1.0)
            elif current_macd < current_signal and current_hist < 0:
                signal = "SELL"
                strength = min(abs(current_hist) * 1000, 1.0)
            else:
                signal = "HOLD"
                strength = 0.0

            # Confidence based on histogram momentum
            hist_momentum = histogram.diff().iloc[-1]
            confidence = min(abs(hist_momentum) * 500, 1.0)

            return IndicatorResult(
                value=current_macd,
                signal=signal,
                strength=strength,
                confidence=confidence,
                metadata={
                    'signal_line': current_signal,
                    'histogram': current_hist,
                    'momentum': hist_momentum
                }
            )

        except Exception as e:
            self.logger.error(f"MACD calculation error: {e}")
            return IndicatorResult(0.0, "HOLD", 0.0, 0.1)

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> IndicatorResult:
        """
        Calculate Bollinger Bands with Renaissance regime detection
        """
        try:
            if len(prices) < period:
                return IndicatorResult(0.0, "HOLD", 0.0, 0.1)

            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            current_price = prices.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_middle = sma.iloc[-1]

            # Calculate %B (position within bands)
            percent_b = (current_price - current_lower) / (current_upper - current_lower)

            # Renaissance-style signal interpretation
            if percent_b > 0.8:
                signal = "SELL"
                strength = min((percent_b - 0.8) / 0.2, 1.0)
            elif percent_b < 0.2:
                signal = "BUY"
                strength = min((0.2 - percent_b) / 0.2, 1.0)
            else:
                signal = "HOLD"
                strength = 0.0

            # Confidence based on band width (volatility)
            band_width = (current_upper - current_lower) / current_middle
            confidence = min(band_width * 10, 1.0)

            return IndicatorResult(
                value=percent_b,
                signal=signal,
                strength=strength,
                confidence=confidence,
                metadata={
                    'upper_band': current_upper,
                    'lower_band': current_lower,
                    'middle_band': current_middle,
                    'band_width': band_width
                }
            )

        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation error: {e}")
            return IndicatorResult(0.0, "HOLD", 0.0, 0.1)

    def calculate_order_flow(self, market_data: pd.DataFrame) -> IndicatorResult:
        """
        Renaissance Technologies Order Flow Analysis
        Advanced volume-price relationship analysis
        """
        try:
            if len(market_data) < 10:
                return IndicatorResult(0.0, "HOLD", 0.0, 0.1)

            # Calculate VWAP (Volume Weighted Average Price)
            vwap = (market_data['close'] * market_data['volume']).cumsum() / market_data['volume'].cumsum()

            # Order flow imbalance
            typical_price = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            money_flow = typical_price * market_data['volume']

            # Positive and negative money flow
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()

            # Money Flow Index
            mfi = 100 - (100 / (1 + positive_flow / negative_flow))

            current_price = market_data['close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            current_mfi = mfi.iloc[-1]

            # Renaissance-style order flow signal
            price_vwap_ratio = (current_price - current_vwap) / current_vwap

            if price_vwap_ratio > 0.002 and current_mfi > 60:
                signal = "BUY"
                strength = min(abs(price_vwap_ratio) * 100, 1.0)
            elif price_vwap_ratio < -0.002 and current_mfi < 40:
                signal = "SELL"
                strength = min(abs(price_vwap_ratio) * 100, 1.0)
            else:
                signal = "HOLD"
                strength = 0.0

            # Confidence based on volume consistency
            volume_trend = market_data['volume'].rolling(5).mean().iloc[-1] / \
                           market_data['volume'].rolling(20).mean().iloc[-1]
            confidence = min(volume_trend, 1.0)

            return IndicatorResult(
                value=price_vwap_ratio,
                signal=signal,
                strength=strength,
                confidence=confidence,
                metadata={
                    'vwap': current_vwap,
                    'mfi': current_mfi,
                    'volume_trend': volume_trend
                }
            )

        except Exception as e:
            self.logger.error(f"Order flow calculation error: {e}")
            return IndicatorResult(0.0, "HOLD", 0.0, 0.1)

    def calculate_volume_profile(self, market_data: pd.DataFrame) -> IndicatorResult:
        """
        Volume Profile Analysis for Renaissance Technologies
        """
        try:
            if len(market_data) < 20:
                return IndicatorResult(0.0, "HOLD", 0.0, 0.1)

            # On-Balance Volume
            obv = (market_data['volume'] *
                   np.where(market_data['close'] > market_data['close'].shift(1), 1,
                            np.where(market_data['close'] < market_data['close'].shift(1), -1, 0))).cumsum()

            # Volume Rate of Change
            volume_roc = market_data['volume'].pct_change(periods=10)

            # Price Volume Trend
            pvt = ((market_data['close'] - market_data['close'].shift(1)) /
                   market_data['close'].shift(1) * market_data['volume']).cumsum()

            current_obv = obv.iloc[-1]
            current_volume_roc = volume_roc.iloc[-1]
            current_pvt = pvt.iloc[-1]

            # Renaissance volume signal
            obv_momentum = obv.diff().iloc[-1]
            pvt_momentum = pvt.diff().iloc[-1]

            if obv_momentum > 0 and pvt_momentum > 0 and current_volume_roc > 0.1:
                signal = "BUY"
                strength = min(abs(obv_momentum) / 1000000, 1.0)
            elif obv_momentum < 0 and pvt_momentum < 0 and current_volume_roc > 0.1:
                signal = "SELL"
                strength = min(abs(obv_momentum) / 1000000, 1.0)
            else:
                signal = "HOLD"
                strength = 0.0

            confidence = min(abs(current_volume_roc), 1.0)

            return IndicatorResult(
                value=current_volume_roc,
                signal=signal,
                strength=strength,
                confidence=confidence,
                metadata={
                    'obv': current_obv,
                    'pvt': current_pvt,
                    'obv_momentum': obv_momentum,
                    'pvt_momentum': pvt_momentum
                }
            )

        except Exception as e:
            self.logger.error(f"Volume profile calculation error: {e}")
            return IndicatorResult(0.0, "HOLD", 0.0, 0.1)

    def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Renaissance Technologies Market Regime Detection
        """
        try:
            if len(market_data) < 50:
                return MarketRegime.SIDEWAYS

            # Calculate various regime indicators
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(20).std()

            # Trend strength (ADX-like)
            high_low = market_data['high'] - market_data['low']
            high_close_prev = abs(market_data['high'] - market_data['close'].shift(1))
            low_close_prev = abs(market_data['low'] - market_data['close'].shift(1))

            tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            atr = tr.rolling(14).mean()

            # Price momentum
            momentum_5 = market_data['close'].pct_change(5)
            momentum_20 = market_data['close'].pct_change(20)

            # Current values
            current_vol = volatility.iloc[-1]
            current_momentum_5 = momentum_5.iloc[-1]
            current_momentum_20 = momentum_20.iloc[-1]

            # Regime classification logic
            vol_threshold = volatility.quantile(0.7)

            if current_vol > vol_threshold:
                if abs(current_momentum_5) > 0.02:
                    return MarketRegime.BREAKOUT
                else:
                    return MarketRegime.HIGH_VOLATILITY
            else:
                if current_momentum_20 > 0.05:
                    return MarketRegime.BULL
                elif current_momentum_20 < -0.05:
                    return MarketRegime.BEAR
                else:
                    return MarketRegime.SIDEWAYS

        except Exception as e:
            self.logger.error(f"Regime detection error: {e}")
            return MarketRegime.SIDEWAYS

    def calculate_all_indicators(self, market_data: pd.DataFrame) -> Dict[str, IndicatorResult]:
        """
        Calculate all Renaissance Technologies indicators
        """
        try:
            if market_data.empty:
                return {}

            prices = market_data['close']

            results = {
                'rsi': self.calculate_rsi(prices),
                'macd': self.calculate_macd(prices),
                'bollinger': self.calculate_bollinger_bands(prices),
                'order_flow': self.calculate_order_flow(market_data),
                'volume_profile': self.calculate_volume_profile(market_data)
            }

            # Add regime detection
            regime = self.detect_regime(market_data)

            self.logger.info(f"All indicators calculated successfully. Regime: {regime.value}")

            return results

        except Exception as e:
            self.logger.error(f"Error calculating all indicators: {e}")
            return {}


# Global instance for easy import
indicators = TechnicalIndicators()


def get_enhanced_indicators(market_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function for getting all Renaissance Technologies indicators
    """
    return indicators.calculate_all_indicators(market_data)


def get_market_regime(market_data: pd.DataFrame) -> MarketRegime:
    """
    Convenience function for regime detection
    """
    return indicators.detect_regime(market_data)