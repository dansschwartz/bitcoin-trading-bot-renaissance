"""Macro data fetcher — daily SPX, VIX, DXY, 10Y yield via yfinance.

Standalone module with hourly caching. Used by crash-regime models
for features 15-24 (daily macro indicators).
"""

import logging
import time
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MacroDataFetcher:
    """Cache daily macro data for crash-regime model features.

    Refreshes every 3600 seconds via yfinance. Returns zeros on failure.
    Thread-safe for concurrent bot access.
    """

    REFRESH_INTERVAL = 3600  # 1 hour

    TICKERS = {
        'spx': '^GSPC',
        'vix': '^VIX',
        'dxy': 'DX-Y.NYB',
        'us10y': '^TNX',
    }

    def __init__(self, logger_: Optional[logging.Logger] = None):
        self.logger = logger_ or logger
        self._cache: Dict[str, float] = {}
        self._last_refresh: float = 0.0

    def get(self) -> Dict[str, float]:
        """Return cached macro data, refreshing if stale."""
        now = time.time()
        if now - self._last_refresh > self.REFRESH_INTERVAL:
            self._refresh()
        return self._cache

    def _refresh(self) -> None:
        """Fetch daily macro data via yfinance."""
        self._last_refresh = time.time()
        try:
            import yfinance as yf
        except ImportError:
            self.logger.warning("yfinance not installed — macro features will be zeros")
            self._cache = {}
            return

        result: Dict[str, float] = {}
        for name, symbol in self.TICKERS.items():
            try:
                tk = yf.Ticker(symbol)
                hist = tk.history(period='5d')
                if hist.empty or len(hist) < 2:
                    continue
                close = hist['Close'].values
                current = float(close[-1])
                prev = float(close[-2])

                if name == 'spx':
                    result['spx_return_1d'] = (current / prev - 1.0) if prev > 0 else 0.0
                    sma = float(np.mean(close)) if len(close) >= 3 else current
                    result['spx_vs_sma'] = (current / sma - 1.0) if sma > 0 else 0.0
                elif name == 'vix':
                    result['vix_norm'] = (current - 20.0) / 20.0
                    result['vix_change'] = (current / prev - 1.0) if prev > 0 else 0.0
                    result['vix_extreme'] = 1.0 if current > 30.0 else 0.0
                elif name == 'dxy':
                    result['dxy_return_1d'] = (current / prev - 1.0) if prev > 0 else 0.0
                    sma = float(np.mean(close)) if len(close) >= 3 else current
                    result['dxy_trend'] = (current / sma - 1.0) if sma > 0 else 0.0
                elif name == 'us10y':
                    result['yield_level'] = (current - 4.0) / 2.0
                    result['yield_change'] = (current - prev) if prev > 0 else 0.0
            except Exception as e:
                self.logger.debug(f"Macro fetch failed for {symbol}: {e}")

        # FNG placeholder (reused from derivatives_data_provider in bot)
        result.setdefault('fng_norm', 0.0)

        self._cache = result
        self.logger.info(f"Macro cache refreshed: {len(result)} features")
