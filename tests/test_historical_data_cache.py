"""Tests for HistoricalDataCache."""
import unittest
import tempfile
import os
import numpy as np
from historical_data_cache import HistoricalDataCache


class TestHistoricalDataCache(unittest.TestCase):

    def _make_cache(self, **overrides):
        cfg = {
            "enabled": False,  # Don't auto-fetch in tests
            "db_path": self._db_path,
        }
        cfg.update(overrides)
        return HistoricalDataCache(cfg)

    def setUp(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db_path = self._tmpfile.name
        self._tmpfile.close()

    def tearDown(self):
        try:
            os.unlink(self._db_path)
        except OSError:
            pass

    def test_init_tables(self):
        cache = self._make_cache()
        cache.init_tables()
        self.assertTrue(cache._initialized)

    def test_needs_refresh_true_when_no_data(self):
        cache = self._make_cache()
        cache.init_tables()
        self.assertTrue(cache.needs_refresh("BTC-USD"))

    def test_get_daily_candles_empty(self):
        cache = self._make_cache()
        cache.init_tables()
        df = cache.get_daily_candles("BTC-USD")
        self.assertTrue(df.empty)

    def test_get_daily_returns_empty(self):
        cache = self._make_cache()
        cache.init_tables()
        returns = cache.get_daily_returns("BTC-USD")
        self.assertEqual(len(returns), 0)

    def test_manual_insert_and_retrieve(self):
        """Directly insert candles and verify retrieval."""
        cache = self._make_cache()
        cache.init_tables()

        # Insert test candles with recent dates
        import sqlite3
        from datetime import datetime, timedelta, timezone
        conn = sqlite3.connect(self._db_path)
        base_date = datetime.now(timezone.utc) - timedelta(days=29)
        for i in range(30):
            date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute(
                "INSERT INTO daily_candles (product_id, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("BTC-USD", date_str, 100+i, 102+i, 99+i, 101+i, 1000+i)
            )
        conn.commit()
        conn.close()

        df = cache.get_daily_candles("BTC-USD", lookback_days=365)
        self.assertEqual(len(df), 30)
        self.assertIn("close", df.columns)

    def test_get_daily_returns_from_candles(self):
        """Insert candles, retrieve log returns."""
        cache = self._make_cache()
        cache.init_tables()

        import sqlite3
        from datetime import datetime, timedelta, timezone
        conn = sqlite3.connect(self._db_path)
        base_date = datetime.now(timezone.utc) - timedelta(days=49)
        prices = np.linspace(100, 200, 50)
        for i, p in enumerate(prices):
            date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute(
                "INSERT INTO daily_candles (product_id, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("BTC-USD", date_str, p, p+1, p-1, p, 1000)
            )
        conn.commit()
        conn.close()

        returns = cache.get_daily_returns("BTC-USD", lookback_days=365)
        self.assertEqual(len(returns), 49)  # n-1 returns for n prices
        self.assertTrue(np.all(np.isfinite(returns)))

    def test_get_top_assets_by_volume(self):
        cache = self._make_cache()
        cache.init_tables()

        import sqlite3
        from datetime import datetime, timedelta, timezone
        conn = sqlite3.connect(self._db_path)
        base_date = datetime.now(timezone.utc) - timedelta(days=9)
        # BTC has higher volume than ETH
        for i in range(10):
            d = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute(
                "INSERT INTO daily_candles (product_id, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("BTC-USD", d, 100, 101, 99, 100, 10000)
            )
            conn.execute(
                "INSERT INTO daily_candles (product_id, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("ETH-USD", d, 50, 51, 49, 50, 5000)
            )
        conn.commit()
        conn.close()

        top = cache.get_top_assets_by_volume(n=5)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0], "BTC-USD")  # Higher total volume

    def test_get_multi_asset_returns_insufficient(self):
        cache = self._make_cache()
        cache.init_tables()
        result = cache.get_multi_asset_returns(["BTC-USD", "ETH-USD"])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
