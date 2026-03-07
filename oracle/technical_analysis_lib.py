"""
Technical Analysis Library — Feature Computation for Oracle
============================================================
Computes the exact 36 features used by the research paper's MLP models.

Features:
  6 oscillators: Z_score, RSI, boll, ULTOSC, pct_change, zsVol
  4 moving-average ratios: PR_MA_Ratio_short, MA_Ratio_short, MA_Ratio, PR_MA_Ratio
  23 candlestick patterns: CDL2CROWS through CDLUPSIDEGAP2CROWS
  3 time features: DayOfWeek, Month, Hourly

Requires: talib (TA-Lib)
Input DataFrame must have: Date, Open, High, Low, Close, Volume
"""

import numpy as np
import pandas as pd
import talib


class TecnicalAnalysis:
    """Static methods matching the research paper's feature pipeline."""

    @staticmethod
    def compute_oscillators(data: pd.DataFrame) -> pd.DataFrame:
        """Compute oscillator and moving-average features."""
        df = data.copy()
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values

        # Z-score of close price (20-period)
        ma20 = talib.SMA(close, timeperiod=20)
        std20 = talib.STDDEV(close, timeperiod=20)
        df['Z_score'] = (close - ma20) / np.where(std20 > 0, std20, 1.0)

        # RSI (14-period), normalized to 0-1
        df['RSI'] = talib.RSI(close, timeperiod=14) / 100.0

        # Bollinger Band position (0-1 range)
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        band_width = upper - lower
        df['boll'] = np.where(band_width > 0, (close - lower) / band_width, 0.5)

        # Ultimate Oscillator, normalized to 0-1
        df['ULTOSC'] = talib.ULTOSC(high, low, close,
                                     timeperiod1=7, timeperiod2=14,
                                     timeperiod3=28) / 100.0

        # Percentage change
        df['pct_change'] = df['Close'].pct_change()

        # Z-score of volume (20-period)
        vol_ma20 = talib.SMA(volume, timeperiod=20)
        vol_std20 = talib.STDDEV(volume, timeperiod=20)
        df['zsVol'] = (volume - vol_ma20) / np.where(vol_std20 > 0, vol_std20, 1.0)

        # Price/MA ratios
        sma10 = talib.SMA(close, timeperiod=10)
        sma50 = talib.SMA(close, timeperiod=50)
        sma100 = talib.SMA(close, timeperiod=100)

        # Short-term price-to-MA ratio
        df['PR_MA_Ratio_short'] = np.where(sma10 > 0, (close / sma10) - 1.0, 0.0)

        # Short MA ratio (10/50)
        df['MA_Ratio_short'] = np.where(sma50 > 0, (sma10 / sma50) - 1.0, 0.0)

        # Long MA ratio (50/100)
        df['MA_Ratio'] = np.where(sma100 > 0, (sma50 / sma100) - 1.0, 0.0)

        # Price-to-long-MA ratio
        df['PR_MA_Ratio'] = np.where(sma100 > 0, (close / sma100) - 1.0, 0.0)

        return df

    @staticmethod
    def find_patterns(data: pd.DataFrame) -> pd.DataFrame:
        """Compute 23 candlestick pattern indicators."""
        df = data.copy()
        o = df['Open'].values
        h = df['High'].values
        l = df['Low'].values
        c = df['Close'].values

        # All 23 candlestick patterns used in the paper
        # Each returns -100, 0, or +100 — we normalize to -1, 0, 1
        patterns = {
            'CDL2CROWS': talib.CDL2CROWS,
            'CDL3BLACKCROWS': talib.CDL3BLACKCROWS,
            'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS,
            'CDLABANDONEDBABY': talib.CDLABANDONEDBABY,
            'CDLBELTHOLD': talib.CDLBELTHOLD,
            'CDLCOUNTERATTACK': talib.CDLCOUNTERATTACK,
            'CDLDARKCLOUDCOVER': talib.CDLDARKCLOUDCOVER,
            'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI,
            'CDLENGULFING': talib.CDLENGULFING,
            'CDLEVENINGDOJISTAR': talib.CDLEVENINGDOJISTAR,
            'CDLEVENINGSTAR': talib.CDLEVENINGSTAR,
            'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI,
            'CDLHANGINGMAN': talib.CDLHANGINGMAN,
            'CDLHARAMICROSS': talib.CDLHARAMICROSS,
            'CDLINVERTEDHAMMER': talib.CDLINVERTEDHAMMER,
            'CDLMARUBOZU': talib.CDLMARUBOZU,
            'CDLMORNINGDOJISTAR': talib.CDLMORNINGDOJISTAR,
            'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
            'CDLPIERCING': talib.CDLPIERCING,
            'CDLRISEFALL3METHODS': talib.CDLRISEFALL3METHODS,
            'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
            'CDLSPINNINGTOP': talib.CDLSPINNINGTOP,
            'CDLUPSIDEGAP2CROWS': talib.CDLUPSIDEGAP2CROWS,
        }

        for name, func in patterns.items():
            df[name] = func(o, h, l, c) / 100.0

        return df

    @staticmethod
    def add_timely_data(data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = data.copy()
        dt = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = dt.dt.dayofweek
        df['Month'] = dt.dt.month
        df['Hourly'] = dt.dt.hour / 4.0  # Normalize: 0,4,8,12,16,20 → 0,1,2,3,4,5
        return df
