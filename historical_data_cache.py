"""
Historical Data Cache — Daily Candle Storage & Multi-Asset Universe
Fetches and caches daily OHLCV candles for 1+ year via ccxt (Coinbase),
provides get_daily_returns(), get_daily_candles(), get_top_assets_by_volume().
Supports the AdvancedRegimeDetector, AdvancedMeanReversionEngine, and
CorrelationNetworkEngine with deep historical data.
"""

import numpy as np
import pandas as pd
import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


class HistoricalDataCache:
    """
    Fetches and caches daily OHLCV candles in SQLite.
    Provides aligned return matrices for multi-asset engines.
    """

    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get("enabled", False)
        self.db_path = config.get("db_path", "data/renaissance_bot.db")
        self.exchange_id = config.get("exchange", "coinbase")
        self.default_lookback_days = config.get("lookback_days", 365)
        self.refresh_interval_hours = config.get("refresh_interval_hours", 6)
        self.max_assets = config.get("max_assets", 30)

        self._exchange = None
        self._initialized = False

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_tables(self):
        """Create daily_candles and data_refresh_log tables if needed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS daily_candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(product_id, date)
            )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS data_refresh_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                last_refresh TEXT NOT NULL,
                rows_fetched INTEGER NOT NULL
            )''')
            conn.commit()
        self._initialized = True

    def _get_exchange(self):
        """Lazily initialize ccxt exchange."""
        if self._exchange is None and CCXT_AVAILABLE:
            try:
                exchange_class = getattr(ccxt, self.exchange_id, None)
                if exchange_class:
                    self._exchange = exchange_class({"enableRateLimit": True})
                else:
                    self.logger.warning(f"Exchange {self.exchange_id} not found in ccxt, using coinbase")
                    self._exchange = ccxt.coinbase({"enableRateLimit": True})
            except Exception as e:
                self.logger.error(f"Failed to initialize exchange: {e}")
        return self._exchange

    def needs_refresh(self, product_id: str) -> bool:
        """Check if daily candles for this product need refreshing."""
        if not self._initialized:
            self.init_tables()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_refresh FROM data_refresh_log WHERE product_id = ? ORDER BY id DESC LIMIT 1",
                (product_id,)
            )
            row = cursor.fetchone()
            if not row:
                return True

            last_refresh = datetime.fromisoformat(row[0])
            hours_since = (datetime.now(timezone.utc) - last_refresh).total_seconds() / 3600.0
            return hours_since >= self.refresh_interval_hours

    def fetch_and_cache_candles(self, product_id: str, lookback_days: Optional[int] = None) -> int:
        """
        Fetch daily OHLCV candles from exchange and cache in SQLite.
        Returns number of rows inserted/updated.
        """
        if not CCXT_AVAILABLE:
            self.logger.warning("ccxt not available, cannot fetch candles")
            return 0

        if not self._initialized:
            self.init_tables()

        exchange = self._get_exchange()
        if exchange is None:
            return 0

        days = lookback_days or self.default_lookback_days
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        # Map product_id format (BTC-USD -> BTC/USD)
        symbol = product_id.replace("-", "/")

        try:
            all_candles = []
            fetch_since = since_ms
            while True:
                candles = exchange.fetch_ohlcv(symbol, timeframe="1d", since=fetch_since, limit=500)
                if not candles:
                    break
                all_candles.extend(candles)
                # Move to next batch
                fetch_since = candles[-1][0] + 1
                if len(candles) < 500:
                    break
                time.sleep(exchange.rateLimit / 1000.0)

            if not all_candles:
                self.logger.warning(f"No candles fetched for {product_id}")
                return 0

            with self._get_connection() as conn:
                cursor = conn.cursor()
                rows_inserted = 0
                for candle in all_candles:
                    ts, o, h, l, c, v = candle[:6]
                    date_str = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
                    cursor.execute(
                        "INSERT OR REPLACE INTO daily_candles (product_id, date, open, high, low, close, volume) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (product_id, date_str, o, h, l, c, v or 0.0)
                    )
                    rows_inserted += 1

                # Log refresh
                cursor.execute(
                    "INSERT INTO data_refresh_log (product_id, last_refresh, rows_fetched) VALUES (?, ?, ?)",
                    (product_id, datetime.now(timezone.utc).isoformat(), rows_inserted)
                )
                conn.commit()

            self.logger.info(f"Cached {rows_inserted} daily candles for {product_id}")
            return rows_inserted

        except Exception as e:
            self.logger.error(f"Failed to fetch candles for {product_id}: {e}")
            return 0

    def get_daily_candles(self, product_id: str, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        Get cached daily candles as a DataFrame.
        Auto-refreshes if stale.
        """
        if not self._initialized:
            self.init_tables()

        if self.enabled and self.needs_refresh(product_id):
            self.fetch_and_cache_candles(product_id, lookback_days)

        days = lookback_days or self.default_lookback_days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM daily_candles "
                "WHERE product_id = ? AND date >= ? ORDER BY date ASC",
                conn,
                params=(product_id, cutoff),
            )

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        return df

    def get_daily_returns(self, product_id: str, lookback_days: Optional[int] = None) -> np.ndarray:
        """Get log returns from daily close prices."""
        df = self.get_daily_candles(product_id, lookback_days)
        if df.empty or len(df) < 2:
            return np.array([])
        prices = df["close"].values
        return np.diff(np.log(np.maximum(prices, 1e-9)))

    def get_multi_asset_returns(self, product_ids: List[str],
                                lookback_days: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get aligned daily log returns for multiple assets.
        Returns a DataFrame with product_ids as columns.
        """
        returns_dict = {}
        for pid in product_ids:
            df = self.get_daily_candles(pid, lookback_days)
            if df.empty or len(df) < 2:
                continue
            log_ret = np.log(df["close"] / df["close"].shift(1))
            returns_dict[pid] = log_ret

        if len(returns_dict) < 2:
            return None

        returns_df = pd.DataFrame(returns_dict).dropna()
        if len(returns_df) < 30:
            return None

        return returns_df

    def get_top_assets_by_volume(self, n: int = 20) -> List[str]:
        """
        Get top N assets by total cached volume.
        Useful for selecting the correlation universe.
        """
        if not self._initialized:
            self.init_tables()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT product_id, SUM(volume) as total_vol FROM daily_candles "
                "GROUP BY product_id ORDER BY total_vol DESC LIMIT ?",
                (n,)
            )
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def refresh_all(self, product_ids: List[str]) -> Dict[str, int]:
        """Refresh candles for all given product_ids. Returns {pid: rows_fetched}."""
        results = {}
        for pid in product_ids:
            if self.needs_refresh(pid):
                results[pid] = self.fetch_and_cache_candles(pid)
            else:
                results[pid] = 0
        return results
