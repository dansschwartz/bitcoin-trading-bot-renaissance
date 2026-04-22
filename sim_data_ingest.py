"""Data ingestion: fetch real OHLCV data via ccxt / yfinance, clean, and align."""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import ccxt as _ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from sim_config import DEFAULT_CONFIG


class SimDataIngest:
    """Fetch, clean, and present OHLCV data for the simulation system.

    Priority: ccxt first (crypto-native), yfinance as fallback.
    Runs a cleaning pipeline: NaN interpolation, outlier clipping,
    volume normalisation, UTC timezone alignment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        cfg = config or DEFAULT_CONFIG.get("data", {})
        self.logger = logger or logging.getLogger(__name__)

        self.lookback_days = cfg.get("lookback_days", 730)
        self.source_priority = cfg.get("source_priority", ["ccxt", "yfinance"])
        self.outlier_sigma = cfg.get("outlier_sigma", 3.0)
        self.nan_interpolation = cfg.get("nan_interpolation", "linear")
        self.volume_norm_window = cfg.get("volume_norm_window", 20)

        self._exchange = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(self, symbol: str,
                    lookback_days: Optional[int] = None) -> pd.DataFrame:
        """Fetch daily OHLCV for *symbol*, cleaned and UTC-aligned.

        Returns a DataFrame with columns ``[open, high, low, close, volume]``
        and a ``DatetimeIndex``.  Returns an empty DataFrame on failure.
        """
        days = lookback_days or self.lookback_days
        df = pd.DataFrame()

        for source in self.source_priority:
            if source == "ccxt" and CCXT_AVAILABLE:
                df = self._fetch_via_ccxt(symbol, days)
            elif source == "yfinance" and YFINANCE_AVAILABLE:
                df = self._fetch_via_yfinance(symbol, days)
            if not df.empty:
                break

        if df.empty:
            self.logger.warning(f"No data fetched for {symbol}")
            return df

        df = self.clean_ohlcv(df)
        return df

    def clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full cleaning pipeline to an OHLCV DataFrame.

        Steps:
        1. Sort by date, drop duplicate dates.
        2. Interpolate NaN values (linear).
        3. Clip outliers beyond ``outlier_sigma`` * rolling std.
        4. Normalise volume by 20-day rolling mean.
        5. Ensure UTC DatetimeIndex.
        """
        if df.empty:
            return df

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # 1 — NaN interpolation
        df = df.interpolate(method=self.nan_interpolation, limit_direction="both")
        df = df.ffill().bfill()  # catch edges

        # 2 — Outlier clipping on price columns
        price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        window = min(60, max(5, len(df) // 4))
        for col in price_cols:
            rolling_mean = df[col].rolling(window, min_periods=1).mean()
            rolling_std = df[col].rolling(window, min_periods=1).std().clip(lower=1e-9)
            upper = rolling_mean + self.outlier_sigma * rolling_std
            lower = rolling_mean - self.outlier_sigma * rolling_std
            mask = (df[col] > upper) | (df[col] < lower)
            if mask.any():
                median_fill = df[col].rolling(5, min_periods=1, center=True).median()
                df.loc[mask, col] = median_fill[mask]

        # 3 — Volume normalisation
        if "volume" in df.columns:
            vol_mean = df["volume"].rolling(self.volume_norm_window,
                                            min_periods=1).mean().clip(lower=1e-9)
            df["volume_norm"] = df["volume"] / vol_mean

        # 4 — UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        return df

    def get_log_returns(self, df: pd.DataFrame) -> np.ndarray:
        """Compute daily log returns from *close* column."""
        if df.empty or "close" not in df.columns or len(df) < 2:
            return np.array([])
        prices = df["close"].values.astype(float)
        return np.diff(np.log(np.maximum(prices, 1e-9)))

    def get_multi_asset_data(self, symbols: List[str],
                             lookback_days: Optional[int] = None,
                             ) -> Dict[str, pd.DataFrame]:
        """Fetch and clean OHLCV for every symbol."""
        results: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self.fetch_ohlcv(sym, lookback_days)
            if not df.empty:
                results[sym] = df
        return results

    def get_aligned_returns(self, symbols: List[str],
                            lookback_days: Optional[int] = None,
                            ) -> pd.DataFrame:
        """Aligned daily log returns for multiple assets.

        Inner-joins on date so every row has returns for *all* assets.
        """
        data = self.get_multi_asset_data(symbols, lookback_days)
        if len(data) < 2:
            return pd.DataFrame()

        returns_dict: Dict[str, pd.Series] = {}
        for sym, df in data.items():
            if len(df) < 2:
                continue
            log_ret = np.log(df["close"] / df["close"].shift(1))
            log_ret.name = sym
            returns_dict[sym] = log_ret

        if len(returns_dict) < 2:
            return pd.DataFrame()

        aligned = pd.DataFrame(returns_dict).dropna()
        return aligned

    # ------------------------------------------------------------------
    # Private fetchers
    # ------------------------------------------------------------------

    def _fetch_via_ccxt(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Fetch daily candles via ccxt."""
        try:
            exchange = self._get_exchange()
            if exchange is None:
                return pd.DataFrame()

            ccxt_symbol = symbol.replace("-", "/")
            since_ms = int(
                (datetime.now(timezone.utc) - timedelta(days=lookback_days))
                .timestamp() * 1000
            )

            all_candles: list = []
            fetch_since = since_ms
            while True:
                candles = exchange.fetch_ohlcv(
                    ccxt_symbol, timeframe="1d", since=fetch_since, limit=500
                )
                if not candles:
                    break
                all_candles.extend(candles)
                fetch_since = candles[-1][0] + 1
                if len(candles) < 500:
                    break

            if not all_candles:
                return pd.DataFrame()

            df = pd.DataFrame(
                all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("date", inplace=True)
            df.drop(columns=["timestamp"], inplace=True)
            return df[["open", "high", "low", "close", "volume"]]

        except Exception as e:
            self.logger.warning(f"ccxt fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_via_yfinance(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Fetch daily candles via yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=lookback_days)
            df = ticker.history(start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"),
                                interval="1d")
            if df.empty:
                return pd.DataFrame()

            # Normalise column names
            rename_map = {}
            for col in df.columns:
                lc = col.lower()
                if lc in ("open", "high", "low", "close", "volume"):
                    rename_map[col] = lc
            df = df.rename(columns=rename_map)
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            return df[keep]

        except Exception as e:
            self.logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def _get_exchange(self):
        """Lazily init ccxt exchange."""
        if self._exchange is None and CCXT_AVAILABLE:
            try:
                self._exchange = _ccxt.coinbase({"enableRateLimit": True})
            except Exception as e:
                self.logger.error(f"ccxt exchange init failed: {e}")
        return self._exchange
