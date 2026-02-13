"""Tests for SimDataIngest cleaning pipeline."""
import unittest
import numpy as np
import pandas as pd
from sim_data_ingest import SimDataIngest


class TestSimDataIngest(unittest.TestCase):

    def _make_ingest(self, **overrides):
        cfg = {"lookback_days": 365, "outlier_sigma": 3.0,
               "source_priority": [], "nan_interpolation": "linear"}
        cfg.update(overrides)
        return SimDataIngest(cfg)

    def _make_ohlcv(self, n=100, with_nan=False, with_outlier=False):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = 100.0 + np.cumsum(rng.normal(0, 1, n))
        close = np.maximum(close, 10)
        df = pd.DataFrame({
            "open": close - rng.uniform(0, 1, n),
            "high": close + rng.uniform(0, 2, n),
            "low": close - rng.uniform(0, 2, n),
            "close": close,
            "volume": rng.uniform(1e6, 5e6, n),
        }, index=dates)
        if with_nan:
            df.iloc[10, df.columns.get_loc("close")] = np.nan
            df.iloc[20, df.columns.get_loc("volume")] = np.nan
        if with_outlier:
            df.iloc[50, df.columns.get_loc("close")] = close[50] * 10  # 10x spike
        return df

    def test_clean_removes_nan(self):
        ingest = self._make_ingest()
        df = self._make_ohlcv(with_nan=True)
        cleaned = ingest.clean_ohlcv(df)
        self.assertFalse(cleaned["close"].isna().any())
        self.assertFalse(cleaned["volume"].isna().any())

    def test_clean_clips_outliers(self):
        ingest = self._make_ingest()
        df = self._make_ohlcv(with_outlier=True)
        original_spike = df.iloc[50]["close"]
        cleaned = ingest.clean_ohlcv(df)
        self.assertLess(cleaned.iloc[50]["close"], original_spike)

    def test_clean_adds_volume_norm(self):
        ingest = self._make_ingest()
        df = self._make_ohlcv()
        cleaned = ingest.clean_ohlcv(df)
        self.assertIn("volume_norm", cleaned.columns)

    def test_clean_utc_timezone(self):
        ingest = self._make_ingest()
        df = self._make_ohlcv()
        cleaned = ingest.clean_ohlcv(df)
        self.assertIsNotNone(cleaned.index.tz)
        self.assertEqual(str(cleaned.index.tz), "UTC")

    def test_clean_sorted_no_duplicates(self):
        ingest = self._make_ingest()
        df = self._make_ohlcv()
        # Add duplicate
        df = pd.concat([df, df.iloc[:1]])
        cleaned = ingest.clean_ohlcv(df)
        self.assertFalse(cleaned.index.duplicated().any())
        # Sorted
        self.assertTrue((cleaned.index == cleaned.index.sort_values()).all())

    def test_get_log_returns_length(self):
        ingest = self._make_ingest()
        df = self._make_ohlcv(n=50)
        cleaned = ingest.clean_ohlcv(df)
        returns = ingest.get_log_returns(cleaned)
        self.assertEqual(len(returns), 49)

    def test_get_log_returns_empty(self):
        ingest = self._make_ingest()
        returns = ingest.get_log_returns(pd.DataFrame())
        self.assertEqual(len(returns), 0)

    def test_clean_empty_df(self):
        ingest = self._make_ingest()
        df = ingest.clean_ohlcv(pd.DataFrame())
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
