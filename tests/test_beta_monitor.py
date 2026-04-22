"""
Tests for monitoring/beta_monitor.py — BetaMonitor
===================================================
Covers: initialization, beta computation via OLS, alert triggering, threshold
checks, hedge recommendations, data retrieval, and edge cases.
All database access is mocked via in-memory SQLite.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from monitoring.beta_monitor import BetaMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_db() -> str:
    """Create an in-memory-like temp SQLite database with test tables and return path."""
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE market_data (
            product_id TEXT,
            price REAL,
            timestamp TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE trades (
            side TEXT,
            size REAL,
            price REAL,
            timestamp TEXT,
            status TEXT DEFAULT 'FILLED'
        )
    """)
    conn.execute("""
        CREATE TABLE open_positions (
            size REAL,
            entry_price REAL,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()
    return path


def _seed_btc_prices(db_path: str, hours: int = 48, base_price: float = 50000.0):
    """Insert hourly BTC price data for the specified number of hours."""
    rng = np.random.RandomState(42)
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc)
    for i in range(hours):
        ts = (now - timedelta(hours=hours - i)).isoformat()
        # Add some variation
        price_start = base_price + i * 10 + rng.randn() * 50
        price_end = price_start + rng.randn() * 100
        # Two prices per hour to have open and close
        conn.execute(
            "INSERT INTO market_data (product_id, price, timestamp) VALUES (?, ?, ?)",
            ("BTC-USD", price_start, ts),
        )
        ts2 = (now - timedelta(hours=hours - i) + timedelta(minutes=30)).isoformat()
        conn.execute(
            "INSERT INTO market_data (product_id, price, timestamp) VALUES (?, ?, ?)",
            ("BTC-USD", price_end, ts2),
        )
    conn.commit()
    conn.close()


def _seed_trades(db_path: str, hours: int = 48, base_price: float = 50000.0):
    """Insert hourly trade data."""
    rng = np.random.RandomState(123)
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc)
    for i in range(hours):
        ts = (now - timedelta(hours=hours - i)).isoformat()
        side = "BUY" if i % 3 != 0 else "SELL"
        price = base_price + i * 10 + rng.randn() * 50
        size = 0.01 + rng.random() * 0.05
        conn.execute(
            "INSERT INTO trades (side, size, price, timestamp, status) VALUES (?, ?, ?, ?, ?)",
            (side, size, price, ts, "FILLED"),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path():
    path = _create_test_db()
    yield path
    import os
    os.unlink(path)


@pytest.fixture
def seeded_db_path(db_path):
    _seed_btc_prices(db_path, hours=48)
    _seed_trades(db_path, hours=48)
    return db_path


@pytest.fixture
def monitor(seeded_db_path):
    config = {
        "beta_monitor": {
            "target_beta": 0.0,
            "alert_threshold": 0.3,
            "window_hours": 24,
        }
    }
    return BetaMonitor(config, seeded_db_path)


@pytest.fixture
def empty_monitor(db_path):
    config = {"beta_monitor": {}}
    return BetaMonitor(config, db_path)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_values(self, db_path):
        config = {}
        bm = BetaMonitor(config, db_path)
        assert bm._target_beta == 0.0
        assert bm._alert_threshold == 0.3
        assert bm._window_hours == 24
        assert bm._last_result is None

    def test_custom_config(self, db_path):
        config = {
            "beta_monitor": {
                "target_beta": 0.5,
                "alert_threshold": 0.1,
                "window_hours": 12,
            }
        }
        bm = BetaMonitor(config, db_path)
        assert bm._target_beta == 0.5
        assert bm._alert_threshold == 0.1
        assert bm._window_hours == 12


# ---------------------------------------------------------------------------
# compute_beta tests
# ---------------------------------------------------------------------------

class TestComputeBeta:
    def test_returns_dict_with_expected_keys(self, monitor):
        result = monitor.compute_beta()
        for key in ("beta", "alpha", "r_squared", "p_value", "status",
                     "message", "window_hours", "n_observations"):
            assert key in result

    def test_beta_is_finite(self, monitor):
        result = monitor.compute_beta()
        assert math.isfinite(result["beta"])
        assert math.isfinite(result["alpha"])

    def test_r_squared_in_range(self, monitor):
        result = monitor.compute_beta()
        if result["n_observations"] >= 3:
            assert 0.0 <= result["r_squared"] <= 1.0

    def test_empty_data_returns_default(self, empty_monitor):
        result = empty_monitor.compute_beta()
        assert result["beta"] == 0.0
        assert result["n_observations"] == 0
        assert "Insufficient data" in result["message"]

    def test_custom_window_hours(self, monitor):
        result = monitor.compute_beta(window_hours=6)
        assert result["window_hours"] == 6

    def test_caches_last_result(self, monitor):
        result = monitor.compute_beta()
        assert monitor._last_result is not None
        assert monitor._last_result["beta"] == result["beta"]

    def test_status_ok_within_threshold(self, seeded_db_path):
        """If threshold is extremely wide, status should be 'ok'."""
        config = {
            "beta_monitor": {
                "alert_threshold": 1e6,  # Impossibly generous threshold
            }
        }
        bm = BetaMonitor(config, seeded_db_path)
        result = bm.compute_beta()
        if result["n_observations"] >= 3:
            assert result["status"] == "ok"

    def test_status_alert_when_exceeds_threshold(self, seeded_db_path):
        """If threshold is tiny, any nonzero beta should trigger alert."""
        config = {
            "beta_monitor": {
                "alert_threshold": 0.0001,  # Impossibly tight
            }
        }
        bm = BetaMonitor(config, seeded_db_path)
        result = bm.compute_beta()
        if result["n_observations"] >= 3 and abs(result["beta"]) > 0.0001:
            assert result["status"] in ("alert", "warning")


# ---------------------------------------------------------------------------
# should_alert tests
# ---------------------------------------------------------------------------

class TestShouldAlert:
    def test_no_alert_on_empty_data(self, empty_monitor):
        assert empty_monitor.should_alert() is False

    def test_alert_triggers_on_extreme_beta(self, seeded_db_path):
        config = {
            "beta_monitor": {
                "alert_threshold": 0.0001,
            }
        }
        bm = BetaMonitor(config, seeded_db_path)
        bm.compute_beta()
        if bm._last_result and bm._last_result["n_observations"] >= 3:
            # With such a tight threshold, most non-trivial betas trigger alert
            if abs(bm._last_result["beta"]) > 0.0001:
                assert bm.should_alert() is True

    def test_should_alert_computes_if_no_cached_result(self, monitor):
        assert monitor._last_result is None
        # should_alert will call compute_beta internally
        result = monitor.should_alert()
        assert isinstance(result, bool)
        assert monitor._last_result is not None


# ---------------------------------------------------------------------------
# get_report tests
# ---------------------------------------------------------------------------

class TestGetReport:
    def test_report_has_expected_keys(self, monitor):
        report = monitor.get_report()
        expected_keys = [
            "current_beta", "current_alpha", "current_r_squared",
            "current_p_value", "current_status", "current_message",
            "beta_7d_rolling", "trend", "target_beta", "alert_threshold",
        ]
        for key in expected_keys:
            assert key in report

    def test_trend_detection(self, monitor):
        report = monitor.get_report()
        assert report["trend"] in ("increasing_exposure", "decreasing_exposure", "stable")

    def test_report_returns_fallback_on_error(self, db_path):
        """If DB is missing tables, report should still return a dict."""
        # Create a DB without the required tables
        import tempfile, os
        fd, bad_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(bad_path)
        conn.close()

        config = {"beta_monitor": {}}
        bm = BetaMonitor(config, bad_path)
        report = bm.get_report()
        assert report["current_beta"] == 0.0
        os.unlink(bad_path)


# ---------------------------------------------------------------------------
# get_hedge_recommendation tests
# ---------------------------------------------------------------------------

class TestHedgeRecommendation:
    def test_no_hedge_when_close_to_target(self, seeded_db_path):
        config = {
            "beta_monitor": {
                "alert_threshold": 1e6,  # impossibly wide
            }
        }
        bm = BetaMonitor(config, seeded_db_path)
        bm.compute_beta()
        rec = bm.get_hedge_recommendation()
        # With such a wide threshold, no hedge needed
        assert rec["needs_hedge"] is False

    def test_hedge_direction_sell_for_positive_beta(self, seeded_db_path):
        config = {
            "beta_monitor": {
                "alert_threshold": 0.0001,
            }
        }
        bm = BetaMonitor(config, seeded_db_path)
        result = bm.compute_beta()
        rec = bm.get_hedge_recommendation()
        if rec["needs_hedge"] and result["beta"] > 0:
            assert rec["direction"] == "SELL"

    def test_hedge_recommendation_returns_dict(self, monitor):
        rec = monitor.get_hedge_recommendation()
        assert "needs_hedge" in rec
        assert "direction" in rec
        assert "recommended_size_usd" in rec
        assert "rationale" in rec

    def test_hedge_on_empty_data(self, empty_monitor):
        rec = empty_monitor.get_hedge_recommendation()
        assert rec["needs_hedge"] is False


# ---------------------------------------------------------------------------
# Internal helpers tests
# ---------------------------------------------------------------------------

class TestEstimatePortfolioNotional:
    def test_falls_back_to_default(self, empty_monitor):
        notional = empty_monitor._estimate_portfolio_notional()
        assert notional == 1000.0  # absolute fallback

    def test_uses_open_positions_if_available(self, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO open_positions (size, entry_price, status) VALUES (?, ?, ?)",
            (0.5, 50000.0, "OPEN"),
        )
        conn.commit()
        conn.close()

        config = {"beta_monitor": {}}
        bm = BetaMonitor(config, db_path)
        notional = bm._estimate_portfolio_notional()
        assert notional == 25000.0  # 0.5 * 50000
