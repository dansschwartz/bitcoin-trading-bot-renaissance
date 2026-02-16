"""
Tests for monitoring/capacity_monitor.py — CapacityMonitor
==========================================================
Covers: initialization, slippage vs trade size regression, capacity wall
detection, headroom calculation, recommended max size, bulk analysis, and
edge cases. Database access uses temp SQLite files with synthetic data.
"""

from __future__ import annotations

import math
import os
import sqlite3
import tempfile

import numpy as np
import pytest

from monitoring.capacity_monitor import CapacityMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_db() -> str:
    """Create a temp SQLite database with devil_tracker table."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE devil_tracker (
            pair TEXT,
            fill_price REAL,
            fill_quantity REAL,
            slippage_bps REAL,
            signal_timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()
    return path


def _seed_linear_slippage(
    db_path: str,
    pair: str = "BTC-USD",
    n_trades: int = 50,
    base_slippage: float = 0.5,
    slope_per_usd: float = 0.001,
    noise_std: float = 0.1,
):
    """Insert trades where slippage = base + slope * size + noise.

    This creates a clear linear relationship between trade size and slippage.
    """
    conn = sqlite3.connect(db_path)
    np.random.seed(42)
    for i in range(n_trades):
        # Trade sizes from 100 to 10000 USD
        quantity = 0.01 + np.random.random() * 0.1
        price = 50000.0 + np.random.randn() * 100
        size_usd = price * quantity
        slippage = base_slippage + slope_per_usd * size_usd + np.random.randn() * noise_std
        slippage = max(0.0, slippage)  # slippage can't be negative in practice
        ts = f"2026-01-{min(i + 1, 28):02d}T12:00:00"
        conn.execute(
            "INSERT INTO devil_tracker (pair, fill_price, fill_quantity, slippage_bps, signal_timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (pair, price, quantity, slippage, ts),
        )
    conn.commit()
    conn.close()


def _seed_high_slippage(db_path: str, pair: str = "BTC-USD", n_trades: int = 50):
    """Insert trades where slippage is already above the wall."""
    conn = sqlite3.connect(db_path)
    np.random.seed(42)
    for i in range(n_trades):
        quantity = 0.01 + np.random.random() * 0.1
        price = 50000.0
        slippage = 10.0 + np.random.randn() * 0.5  # well above 5 bps wall
        ts = f"2026-01-{min(i + 1, 28):02d}T12:00:00"
        conn.execute(
            "INSERT INTO devil_tracker (pair, fill_price, fill_quantity, slippage_bps, signal_timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (pair, price, quantity, slippage, ts),
        )
    conn.commit()
    conn.close()


def _seed_low_slippage(db_path: str, pair: str = "BTC-USD", n_trades: int = 50):
    """Insert trades with very low, flat slippage."""
    conn = sqlite3.connect(db_path)
    np.random.seed(42)
    for i in range(n_trades):
        quantity = 0.01 + np.random.random() * 0.1
        price = 50000.0
        slippage = 0.5 + np.random.randn() * 0.1  # well below 5 bps wall
        slippage = max(0.01, slippage)
        ts = f"2026-01-{min(i + 1, 28):02d}T12:00:00"
        conn.execute(
            "INSERT INTO devil_tracker (pair, fill_price, fill_quantity, slippage_bps, signal_timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (pair, price, quantity, slippage, ts),
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
    os.unlink(path)


@pytest.fixture
def linear_db(db_path):
    _seed_linear_slippage(db_path, n_trades=50)
    return db_path


@pytest.fixture
def high_slip_db(db_path):
    _seed_high_slippage(db_path, n_trades=50)
    return db_path


@pytest.fixture
def low_slip_db(db_path):
    _seed_low_slippage(db_path, n_trades=50)
    return db_path


@pytest.fixture
def linear_monitor(linear_db):
    return CapacityMonitor({"capacity_monitor": {}}, linear_db)


@pytest.fixture
def high_slip_monitor(high_slip_db):
    return CapacityMonitor({"capacity_monitor": {}}, high_slip_db)


@pytest.fixture
def low_slip_monitor(low_slip_db):
    return CapacityMonitor({"capacity_monitor": {}}, low_slip_db)


@pytest.fixture
def empty_monitor(db_path):
    return CapacityMonitor({"capacity_monitor": {}}, db_path)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_values(self, db_path):
        cm = CapacityMonitor({}, db_path)
        assert cm._slippage_wall_bps == 5.0
        assert cm._min_trades == 20
        assert cm._cache == {}

    def test_custom_config(self, db_path):
        config = {
            "capacity_monitor": {
                "slippage_wall_bps": 10.0,
                "min_trades_for_analysis": 50,
            }
        }
        cm = CapacityMonitor(config, db_path)
        assert cm._slippage_wall_bps == 10.0
        assert cm._min_trades == 50


# ---------------------------------------------------------------------------
# analyze_capacity tests
# ---------------------------------------------------------------------------

class TestAnalyzeCapacity:
    def test_returns_dict_with_expected_keys(self, linear_monitor):
        result = linear_monitor.analyze_capacity("BTC-USD")
        for key in ("pair", "slope_bps_per_1000usd", "intercept_bps",
                     "capacity_status", "estimated_max_size_usd",
                     "current_avg_size_usd", "headroom_pct", "n_trades",
                     "r_squared", "message"):
            assert key in result

    def test_linear_slippage_detected(self, linear_monitor):
        result = linear_monitor.analyze_capacity("BTC-USD")
        assert result["n_trades"] > 0
        # With slope_per_usd=0.001, slope_per_1000 ~ 1.0 bps
        assert result["slope_bps_per_1000usd"] > 0.0

    def test_r_squared_in_valid_range(self, linear_monitor):
        result = linear_monitor.analyze_capacity("BTC-USD")
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_high_slippage_is_constrained(self, high_slip_monitor):
        result = high_slip_monitor.analyze_capacity("BTC-USD")
        assert result["capacity_status"] == "constrained"
        assert result["headroom_pct"] == 0.0 or result["estimated_max_size_usd"] == 0.0

    def test_low_slippage_is_ok(self, low_slip_monitor):
        result = low_slip_monitor.analyze_capacity("BTC-USD")
        assert result["capacity_status"] == "ok"
        assert result["headroom_pct"] > 30.0

    def test_insufficient_trades(self, db_path):
        """Fewer trades than min_trades_for_analysis -> insufficient."""
        conn = sqlite3.connect(db_path)
        for i in range(5):  # less than default 20
            conn.execute(
                "INSERT INTO devil_tracker (pair, fill_price, fill_quantity, slippage_bps, signal_timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                ("BTC-USD", 50000.0, 0.01, 1.0, f"2026-01-0{i + 1}T12:00:00"),
            )
        conn.commit()
        conn.close()
        cm = CapacityMonitor({}, db_path)
        result = cm.analyze_capacity("BTC-USD")
        assert result["capacity_status"] == "ok"
        assert "Insufficient" in result["message"] or "Only" in result["message"]

    def test_unknown_pair_returns_insufficient(self, empty_monitor):
        result = empty_monitor.analyze_capacity("NONEXISTENT-PAIR")
        assert result["n_trades"] == 0

    def test_caches_result(self, linear_monitor):
        linear_monitor.analyze_capacity("BTC-USD")
        assert "BTC-USD" in linear_monitor._cache

    def test_estimated_max_size_positive(self, linear_monitor):
        result = linear_monitor.analyze_capacity("BTC-USD")
        assert result["estimated_max_size_usd"] >= 0.0

    def test_headroom_pct_bounded(self, linear_monitor):
        result = linear_monitor.analyze_capacity("BTC-USD")
        assert 0.0 <= result["headroom_pct"] <= 100.0


# ---------------------------------------------------------------------------
# get_all_capacities tests
# ---------------------------------------------------------------------------

class TestGetAllCapacities:
    def test_empty_db_returns_empty(self, empty_monitor):
        result = empty_monitor.get_all_capacities()
        assert result == {}

    def test_multiple_pairs(self, db_path):
        _seed_linear_slippage(db_path, pair="BTC-USD", n_trades=30)
        _seed_low_slippage(db_path, pair="ETH-USD", n_trades=30)
        cm = CapacityMonitor({}, db_path)
        results = cm.get_all_capacities()
        assert "BTC-USD" in results
        assert "ETH-USD" in results
        assert len(results) == 2

    def test_returns_dict_of_dicts(self, linear_db):
        cm = CapacityMonitor({}, linear_db)
        results = cm.get_all_capacities()
        for pair, analysis in results.items():
            assert isinstance(analysis, dict)
            assert "capacity_status" in analysis


# ---------------------------------------------------------------------------
# get_recommended_max_size tests
# ---------------------------------------------------------------------------

class TestRecommendedMaxSize:
    def test_unconstrained_returns_inf(self, low_slip_monitor):
        max_size = low_slip_monitor.get_recommended_max_size("BTC-USD")
        # Low slippage with flat slope -> estimated_max is inf
        # Depends on regression outcome; may be inf or a large number
        assert max_size > 0.0

    def test_constrained_returns_small_or_zero(self, high_slip_monitor):
        max_size = high_slip_monitor.get_recommended_max_size("BTC-USD")
        # Already above wall -> should be 0 or very small
        assert max_size >= 0.0

    def test_includes_safety_margin(self, linear_db):
        """Recommended size should be 80% of the theoretical max."""
        cm = CapacityMonitor({}, linear_db)
        result = cm.analyze_capacity("BTC-USD")
        theoretical_max = result["estimated_max_size_usd"]
        recommended = cm.get_recommended_max_size("BTC-USD")
        if theoretical_max != float("inf") and theoretical_max > 0:
            assert abs(recommended - theoretical_max * 0.8) < 0.01

    def test_unknown_pair_triggers_analysis(self, linear_db):
        cm = CapacityMonitor({}, linear_db)
        assert "BTC-USD" not in cm._cache
        max_size = cm.get_recommended_max_size("BTC-USD")
        assert "BTC-USD" in cm._cache

    def test_unknown_pair_no_data_returns_inf(self, empty_monitor):
        max_size = empty_monitor.get_recommended_max_size("NOPE-USD")
        assert max_size == float("inf")


# ---------------------------------------------------------------------------
# Data retrieval tests
# ---------------------------------------------------------------------------

class TestDataRetrieval:
    def test_fetch_pair_trades_returns_tuples(self, linear_db):
        cm = CapacityMonitor({}, linear_db)
        trades = cm._fetch_pair_trades("BTC-USD")
        assert len(trades) > 0
        assert isinstance(trades[0], tuple)
        assert len(trades[0]) == 2  # (size_usd, slippage_bps)

    def test_fetch_pair_trades_empty_pair(self, linear_db):
        cm = CapacityMonitor({}, linear_db)
        trades = cm._fetch_pair_trades("NONEXISTENT")
        assert trades == []

    def test_fetch_all_pairs(self, db_path):
        _seed_linear_slippage(db_path, pair="BTC-USD")
        _seed_linear_slippage(db_path, pair="ETH-USD")
        cm = CapacityMonitor({}, db_path)
        pairs = cm._fetch_all_pairs()
        assert "BTC-USD" in pairs
        assert "ETH-USD" in pairs


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_null_values_excluded(self, db_path):
        """Rows with NULL fill_price/quantity/slippage should be skipped."""
        conn = sqlite3.connect(db_path)
        for i in range(25):
            conn.execute(
                "INSERT INTO devil_tracker (pair, fill_price, fill_quantity, slippage_bps, signal_timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                ("BTC-USD", None, 0.01, 1.0, f"2026-01-{min(i + 1, 28):02d}T12:00:00"),
            )
        conn.commit()
        conn.close()
        cm = CapacityMonitor({}, db_path)
        result = cm.analyze_capacity("BTC-USD")
        assert result["n_trades"] == 0

    def test_zero_fill_price_excluded(self, db_path):
        conn = sqlite3.connect(db_path)
        for i in range(25):
            conn.execute(
                "INSERT INTO devil_tracker (pair, fill_price, fill_quantity, slippage_bps, signal_timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                ("BTC-USD", 0.0, 0.01, 1.0, f"2026-01-{min(i + 1, 28):02d}T12:00:00"),
            )
        conn.commit()
        conn.close()
        cm = CapacityMonitor({}, db_path)
        result = cm.analyze_capacity("BTC-USD")
        assert result["n_trades"] == 0

    def test_outlier_filtering(self, db_path):
        """Extreme outliers should be filtered via 3-sigma rule."""
        np.random.seed(42)
        conn = sqlite3.connect(db_path)
        # Normal data
        for i in range(30):
            conn.execute(
                "INSERT INTO devil_tracker (pair, fill_price, fill_quantity, slippage_bps, signal_timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                ("BTC-USD", 50000.0, 0.01, 1.0 + np.random.randn() * 0.1,
                 f"2026-01-{min(i + 1, 28):02d}T12:00:00"),
            )
        # Extreme outlier
        conn.execute(
            "INSERT INTO devil_tracker (pair, fill_price, fill_quantity, slippage_bps, signal_timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            ("BTC-USD", 50000.0, 0.01, 1000.0, "2026-01-29T12:00:00"),
        )
        conn.commit()
        conn.close()
        cm = CapacityMonitor({}, db_path)
        result = cm.analyze_capacity("BTC-USD")
        # Outlier should be filtered; slippage stats should not be crazy
        assert result["intercept_bps"] < 100.0

    def test_missing_db_file_handled(self):
        cm = CapacityMonitor({}, "/nonexistent/db.db")
        result = cm.analyze_capacity("BTC-USD")
        assert result["n_trades"] == 0

    def test_warning_status_near_wall(self, db_path):
        """Create data where predicted slippage is 70-99% of wall."""
        conn = sqlite3.connect(db_path)
        np.random.seed(42)
        for i in range(30):
            # Slippage around 3.5-4 bps with some variation (wall is 5)
            slippage = 3.5 + np.random.random() * 0.5 + i * 0.01
            conn.execute(
                "INSERT INTO devil_tracker (pair, fill_price, fill_quantity, slippage_bps, signal_timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                ("BTC-USD", 50000.0, 0.01 + i * 0.001, slippage,
                 f"2026-01-{min(i + 1, 28):02d}T12:00:00"),
            )
        conn.commit()
        conn.close()
        cm = CapacityMonitor({}, db_path)
        result = cm.analyze_capacity("BTC-USD")
        # Status should be warning or constrained (slippage is near wall)
        assert result["capacity_status"] in ("warning", "constrained")
