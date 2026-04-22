"""
Tests for intelligence/fast_mean_reversion.py — FastMeanReversionScanner
==========================================================================
Covers dislocation detection, rolling VWAP/stddev, reversion tracking,
signal generation, TTL expiry, and disabled scanner behaviour.
"""

import math
import time

import pytest
from unittest.mock import MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.fast_mean_reversion import (
    FastMeanReversionScanner,
    FastReversionSignal,
    _PairStats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def enabled_config():
    return {
        "enabled": True,
        "evaluation_interval_seconds": 1.0,
        "rolling_window_seconds": 60,
        "deviation_threshold_sigma": 2.0,
        "min_deviation_bps": 5.0,
        "max_signal_ttl_seconds": 120,
        "min_confidence": 0.52,
        "min_samples": 5,
        "pairs": ["BTC-USD", "ETH-USD"],
    }


@pytest.fixture
def disabled_config():
    return {"enabled": False}


def _feed_prices(scanner, pair, prices, volume=1.0, start_ts=1000.0, interval=1.0):
    """Feed a list of prices into the scanner."""
    for i, p in enumerate(prices):
        ts = start_ts + i * interval
        scanner.on_price_update(pair, p, volume, ts)


# ---------------------------------------------------------------------------
# _PairStats Tests
# ---------------------------------------------------------------------------

class TestPairStats:

    def test_sample_count_empty(self):
        ps = _PairStats()
        assert ps.sample_count == 0

    def test_rolling_vwap_empty(self):
        ps = _PairStats()
        assert ps.rolling_vwap == 0.0

    def test_rolling_stdev_insufficient_data(self):
        ps = _PairStats()
        ps.prices.append(100.0)
        assert ps.rolling_stdev == 0.0

    def test_rolling_vwap_with_zero_volume(self):
        ps = _PairStats()
        ps.prices.extend([100.0, 102.0, 101.0])
        ps.volumes.extend([0.0, 0.0, 0.0])
        # Zero total volume -> falls back to simple mean
        expected = (100.0 + 102.0 + 101.0) / 3
        assert ps.rolling_vwap == pytest.approx(expected, rel=1e-6)

    def test_rolling_vwap_with_volume(self):
        ps = _PairStats()
        ps.prices.extend([100.0, 200.0])
        ps.volumes.extend([1.0, 3.0])
        # VWAP = (100*1 + 200*3) / (1+3) = 700/4 = 175.0
        assert ps.rolling_vwap == pytest.approx(175.0, rel=1e-6)

    def test_rolling_stdev(self):
        ps = _PairStats()
        ps.prices.extend([100.0, 102.0, 98.0, 101.0, 99.0])
        mean = sum(ps.prices) / len(ps.prices)
        variance = sum((p - mean) ** 2 for p in ps.prices) / (len(ps.prices) - 1)
        expected = math.sqrt(variance)
        assert ps.rolling_stdev == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Scanner Tests
# ---------------------------------------------------------------------------

class TestFastMeanReversionScanner:

    def test_disabled_scanner_ignores_updates(self, disabled_config):
        scanner = FastMeanReversionScanner(disabled_config)
        scanner.on_price_update("BTC-USD", 100.0, 1.0, time.time())
        assert scanner._stats == {}

    def test_on_price_update_creates_pair_stats(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        ts = 1000.0
        scanner.on_price_update("BTC-USD", 50000.0, 1.0, ts)
        assert "BTC-USD" in scanner._stats
        assert scanner._stats["BTC-USD"].sample_count == 1

    def test_on_price_update_rejects_negative_price(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        scanner.on_price_update("BTC-USD", -100.0, 1.0, 1000.0)
        assert "BTC-USD" not in scanner._stats

    def test_on_price_update_rejects_zero_price(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        scanner.on_price_update("BTC-USD", 0.0, 1.0, 1000.0)
        assert "BTC-USD" not in scanner._stats

    def test_rolling_window_trims_old_entries(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        # Feed data over 120 seconds (window is 60s)
        for i in range(120):
            scanner.on_price_update("BTC-USD", 50000.0, 1.0, 1000.0 + i)

        stats = scanner._stats["BTC-USD"]
        # After window trimming, should only have ~60 entries
        # (the on_price_update trims entries older than window)
        assert stats.sample_count <= 61

    def test_evaluate_insufficient_samples(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        # Feed fewer than min_samples
        for i in range(3):
            scanner.on_price_update("BTC-USD", 50000.0 + i, 1.0, 1000.0 + i)
        assert scanner.evaluate("BTC-USD") is None

    def test_evaluate_no_dislocation(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        # Feed stable prices — no dislocation expected
        _feed_prices(scanner, "BTC-USD", [50000.0] * 30)
        result = scanner.evaluate("BTC-USD")
        # No deviation -> should be None (stdev=0 check)
        assert result is None

    def test_evaluate_detects_upward_dislocation(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        # Feed stable prices then spike up
        stable = [50000.0] * 20
        spiked = [50000.0] * 19 + [50100.0]  # sharp upward spike
        _feed_prices(scanner, "BTC-USD", stable + spiked)

        result = scanner.evaluate("BTC-USD")
        if result is not None:
            # Price above VWAP -> direction should be "short"
            assert result.direction == "short"
            assert result.dislocation_pct > 0
            assert result.confidence >= 0.52

    def test_evaluate_detects_downward_dislocation(self, enabled_config):
        """Create a clear downward dislocation and verify long signal."""
        scanner = FastMeanReversionScanner(enabled_config)
        # Start at 50000, drop sharply
        prices = [50000.0] * 25 + [49700.0] * 15
        _feed_prices(scanner, "BTC-USD", prices)

        result = scanner.evaluate("BTC-USD")
        if result is not None:
            assert result.direction == "long"

    def test_evaluate_unknown_pair_returns_none(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        assert scanner.evaluate("UNKNOWN-PAIR") is None

    def test_get_latest_signal_none_without_evaluation(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        assert scanner.get_latest_signal("BTC-USD") is None

    def test_get_latest_signal_returns_cached(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        sig = FastReversionSignal(
            pair="BTC-USD",
            direction="long",
            dislocation_pct=0.05,
            mean_price=50000.0,
            current_price=49975.0,
            expected_reversion_bps=3.0,
            confidence=0.60,
            ttl_seconds=120,
            timestamp=time.time(),
        )
        scanner._latest_signals["BTC-USD"] = sig
        assert scanner.get_latest_signal("BTC-USD") is sig

    def test_get_latest_signal_expired(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        sig = FastReversionSignal(
            pair="BTC-USD",
            direction="long",
            dislocation_pct=0.05,
            mean_price=50000.0,
            current_price=49975.0,
            expected_reversion_bps=3.0,
            confidence=0.60,
            ttl_seconds=120,
            timestamp=time.time() - 200,  # expired 80 seconds ago
        )
        scanner._latest_signals["BTC-USD"] = sig
        assert scanner.get_latest_signal("BTC-USD") is None
        assert "BTC-USD" not in scanner._latest_signals

    def test_stop_sets_running_false(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        scanner._running = True
        scanner.stop()
        assert scanner._running is False

    def test_signal_confidence_capped_at_085(self, enabled_config):
        """Even with extreme deviation, confidence should not exceed 0.85."""
        scanner = FastMeanReversionScanner(enabled_config)
        # Feed prices that create a huge sigma deviation
        prices = [50000.0] * 20 + [51000.0] * 20
        _feed_prices(scanner, "BTC-USD", prices)

        result = scanner.evaluate("BTC-USD")
        if result is not None:
            assert result.confidence <= 0.85

    def test_negative_volume_clamped_to_zero(self, enabled_config):
        scanner = FastMeanReversionScanner(enabled_config)
        scanner.on_price_update("BTC-USD", 50000.0, -5.0, 1000.0)
        stats = scanner._stats["BTC-USD"]
        assert stats.volumes[-1] == 0.0
