"""
Tests for scripts/training/fetch_training_data.py — Data fetching.

Covers CSV saving/loading, incremental update logic, and batch download.
All HTTP requests to Coinbase API are mocked.
"""

import json
import os
import time
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.training.fetch_training_data import (
    fetch_candles_batch,
    download_pair,
    load_cached_data,
    download_all,
    GRANULARITY,
    BATCH_SIZE,
    DEFAULT_PAIRS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle_data(n: int = 10, start_ts: int = 1700000000) -> list:
    """Create synthetic Coinbase-format candle data.
    Coinbase returns: [timestamp, low, high, open, close, volume]
    """
    candles = []
    for i in range(n):
        ts = start_ts + i * GRANULARITY
        candles.append([
            ts,           # timestamp
            100.0 - i,    # low
            110.0 + i,    # high
            105.0,        # open
            106.0 + i * 0.1,  # close
            50.0 + i,     # volume
        ])
    return candles


def _make_existing_csv(path: str, n: int = 5, start_ts: int = 1700000000) -> pd.DataFrame:
    """Create an existing CSV file with OHLCV data."""
    rows = []
    for i in range(n):
        rows.append({
            "timestamp": start_ts + i * GRANULARITY,
            "open": 105.0,
            "high": 110.0,
            "low": 100.0,
            "close": 106.0,
            "volume": 50.0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Tests — fetch_candles_batch
# ---------------------------------------------------------------------------

class TestFetchCandlesBatch:
    @patch("urllib.request.urlopen")
    def test_returns_parsed_data(self, mock_urlopen):
        candles = _make_candle_data(5)
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(candles).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_candles_batch("BTC-USD", 1700000000, 1700001500)
        assert len(result) == 5
        assert result[0][0] == 1700000000

    @patch("urllib.request.urlopen")
    def test_url_construction(self, mock_urlopen):
        candles = _make_candle_data(1)
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(candles).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        fetch_candles_batch("ETH-USD", 100, 200)
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        url = request_obj.full_url
        assert "ETH-USD" in url
        assert "granularity=300" in url

    @patch("urllib.request.urlopen")
    def test_empty_response(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps([]).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_candles_batch("BTC-USD", 100, 200)
        assert result == []


# ---------------------------------------------------------------------------
# Tests — download_pair
# ---------------------------------------------------------------------------

class TestDownloadPair:
    @patch("scripts.training.fetch_training_data.fetch_candles_batch")
    @patch("scripts.training.fetch_training_data.time.sleep")
    def test_creates_csv(self, mock_sleep, mock_fetch, tmp_path):
        candles = _make_candle_data(10, start_ts=int(time.time()) - 3600)
        mock_fetch.return_value = candles

        output_dir = str(tmp_path / "training")
        df = download_pair("BTC-USD", days=1, output_dir=output_dir)

        csv_path = os.path.join(output_dir, "BTC-USD.csv")
        assert os.path.exists(csv_path)
        assert df is not None
        assert len(df) > 0
        assert "open" in df.columns
        assert "close" in df.columns

    @patch("scripts.training.fetch_training_data.fetch_candles_batch")
    @patch("scripts.training.fetch_training_data.time.sleep")
    def test_incremental_update(self, mock_sleep, mock_fetch, tmp_path):
        output_dir = str(tmp_path / "training")
        os.makedirs(output_dir)

        # Create existing CSV with old timestamps
        old_ts = int(time.time()) - 86400 * 2  # 2 days ago
        csv_path = os.path.join(output_dir, "BTC-USD.csv")
        _make_existing_csv(csv_path, n=5, start_ts=old_ts)

        # New candles have timestamps after the existing ones
        new_ts = old_ts + 5 * GRANULARITY + GRANULARITY
        new_candles = _make_candle_data(3, start_ts=new_ts)
        mock_fetch.return_value = new_candles

        df = download_pair("BTC-USD", days=3, output_dir=output_dir)
        assert df is not None
        assert len(df) >= 5  # At least the existing data

    @patch("scripts.training.fetch_training_data.fetch_candles_batch")
    @patch("scripts.training.fetch_training_data.time.sleep")
    def test_deduplication(self, mock_sleep, mock_fetch, tmp_path):
        output_dir = str(tmp_path / "training")
        os.makedirs(output_dir)

        # Create existing data
        ts = int(time.time()) - 86400
        csv_path = os.path.join(output_dir, "BTC-USD.csv")
        _make_existing_csv(csv_path, n=5, start_ts=ts)

        # Return overlapping candles
        overlap_candles = _make_candle_data(5, start_ts=ts)
        mock_fetch.return_value = overlap_candles

        df = download_pair("BTC-USD", days=2, output_dir=output_dir)
        # After dedup, should have same count (no duplicates)
        assert df is not None
        assert len(df["timestamp"].unique()) == len(df)

    @patch("scripts.training.fetch_training_data.fetch_candles_batch")
    @patch("scripts.training.fetch_training_data.time.sleep")
    def test_already_up_to_date(self, mock_sleep, mock_fetch, tmp_path):
        """If existing data is newer than target start, minimal fetch needed."""
        output_dir = str(tmp_path / "training")
        os.makedirs(output_dir)

        # Existing data is very recent
        recent_ts = int(time.time()) - 60
        csv_path = os.path.join(output_dir, "BTC-USD.csv")
        _make_existing_csv(csv_path, n=5, start_ts=recent_ts)

        df = download_pair("BTC-USD", days=1, output_dir=output_dir)
        # Should return existing data without fetching
        assert df is not None
        mock_fetch.assert_not_called()

    @patch("scripts.training.fetch_training_data.fetch_candles_batch")
    @patch("scripts.training.fetch_training_data.time.sleep")
    def test_fetch_failure_returns_existing(self, mock_sleep, mock_fetch, tmp_path):
        """If fetch fails, should return existing data."""
        output_dir = str(tmp_path / "training")
        os.makedirs(output_dir)

        ts = int(time.time()) - 86400 * 5
        csv_path = os.path.join(output_dir, "BTC-USD.csv")
        existing_df = _make_existing_csv(csv_path, n=5, start_ts=ts)

        mock_fetch.side_effect = ConnectionError("API down")

        df = download_pair("BTC-USD", days=10, output_dir=output_dir)
        # Should return existing data when fetch fails
        assert df is not None
        assert len(df) == 5

    @patch("scripts.training.fetch_training_data.fetch_candles_batch")
    @patch("scripts.training.fetch_training_data.time.sleep")
    def test_no_existing_no_fetch_returns_empty(self, mock_sleep, mock_fetch, tmp_path):
        """If no existing data and fetch returns nothing, return empty."""
        output_dir = str(tmp_path / "training")
        mock_fetch.return_value = []

        df = download_pair("BTC-USD", days=1, output_dir=output_dir)
        assert df is not None and len(df) == 0 or df is None


# ---------------------------------------------------------------------------
# Tests — load_cached_data
# ---------------------------------------------------------------------------

class TestLoadCachedData:
    def test_loads_existing_csv(self, tmp_path):
        output_dir = str(tmp_path / "training")
        os.makedirs(output_dir)
        csv_path = os.path.join(output_dir, "BTC-USD.csv")
        _make_existing_csv(csv_path, n=10)

        df = load_cached_data("BTC-USD", output_dir=output_dir)
        assert df is not None
        assert len(df) == 10

    def test_returns_none_if_missing(self, tmp_path):
        output_dir = str(tmp_path / "training")
        os.makedirs(output_dir)
        assert load_cached_data("MISSING-PAIR", output_dir=output_dir) is None

    def test_returns_none_if_empty(self, tmp_path):
        output_dir = str(tmp_path / "training")
        os.makedirs(output_dir)
        csv_path = os.path.join(output_dir, "EMPTY-PAIR.csv")
        pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]).to_csv(
            csv_path, index=False
        )
        assert load_cached_data("EMPTY-PAIR", output_dir=output_dir) is None


# ---------------------------------------------------------------------------
# Tests — download_all
# ---------------------------------------------------------------------------

class TestDownloadAll:
    @patch("scripts.training.fetch_training_data.download_pair")
    def test_downloads_all_pairs(self, mock_download):
        mock_download.return_value = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [10, 20, 30],
        })

        results = download_all(pairs=["BTC-USD", "ETH-USD"], days=1)
        assert len(results) == 2
        assert "BTC-USD" in results
        assert "ETH-USD" in results
        assert mock_download.call_count == 2

    @patch("scripts.training.fetch_training_data.download_pair")
    def test_skips_empty_results(self, mock_download):
        mock_download.return_value = pd.DataFrame()

        results = download_all(pairs=["BTC-USD"], days=1)
        assert len(results) == 0

    @patch("scripts.training.fetch_training_data.download_pair")
    def test_default_pairs_used(self, mock_download):
        mock_download.return_value = pd.DataFrame({
            "timestamp": [1], "open": [100], "high": [101],
            "low": [99], "close": [100], "volume": [10],
        })

        results = download_all(days=1)
        assert mock_download.call_count == len(DEFAULT_PAIRS)

    @patch("scripts.training.fetch_training_data.download_pair")
    def test_handles_none_return(self, mock_download):
        mock_download.return_value = None
        results = download_all(pairs=["BTC-USD"], days=1)
        assert len(results) == 0
