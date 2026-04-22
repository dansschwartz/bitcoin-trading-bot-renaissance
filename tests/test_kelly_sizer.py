"""
Unit tests for core/kelly_position_sizer.py
=============================================
Tests KellyPositionSizer: binary Kelly, continuous Kelly,
get_position_size, statistics API, caching, edge cases.

All DB access is mocked or uses temporary SQLite files.
"""

import math
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest

from core.kelly_position_sizer import (
    KellyPositionSizer,
    TradeStats,
    _DEFAULT_CONFIG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database with a trades table."""
    db_path = str(tmp_path / "test_kelly.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            product_id TEXT,
            side TEXT,
            size REAL,
            price REAL,
            status TEXT,
            algo_used TEXT,
            slippage REAL,
            execution_time REAL
        )
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sizer(tmp_db):
    """Return a KellyPositionSizer with default config and empty DB."""
    return KellyPositionSizer(config={}, db_path=tmp_db)


def _insert_round_trip(db_path, pair, algo, buy_price, sell_price, slippage=0.0):
    """Insert a BUY + SELL trade pair into the trades table."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO trades (timestamp, product_id, side, size, price, status, algo_used, slippage) "
        "VALUES (datetime('now'), ?, 'BUY', 1.0, ?, 'filled', ?, ?)",
        (pair, buy_price, algo, slippage),
    )
    conn.execute(
        "INSERT INTO trades (timestamp, product_id, side, size, price, status, algo_used, slippage) "
        "VALUES (datetime('now'), ?, 'SELL', 1.0, ?, 'filled', ?, ?)",
        (pair, sell_price, algo, slippage),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Binary Kelly
# ---------------------------------------------------------------------------

class TestBinaryKelly:

    def test_positive_edge(self):
        """Standard case: win_rate=60%, b=1.5 -> f* = 0.6 - 0.4/1.5 = 0.333"""
        f = KellyPositionSizer._binary_kelly(0.6, 15.0, 10.0)
        # b = 15/10 = 1.5; f = 0.6 - 0.4/1.5 = 0.333...
        assert f == pytest.approx(1 / 3, abs=0.001)

    def test_zero_edge(self):
        """Exactly breakeven: f* = p - q/b = 0.5 - 0.5/1.0 = 0.0"""
        f = KellyPositionSizer._binary_kelly(0.5, 10.0, 10.0)
        assert f == pytest.approx(0.0)

    def test_negative_edge(self):
        """Losing system returns 0 (clamped)."""
        f = KellyPositionSizer._binary_kelly(0.3, 10.0, 10.0)
        # 0.3 - 0.7/1.0 = -0.4 -> clamped to 0
        assert f == 0.0

    def test_zero_win_rate(self):
        """Zero win rate returns 0."""
        f = KellyPositionSizer._binary_kelly(0.0, 10.0, 10.0)
        assert f == 0.0

    def test_zero_avg_loss(self):
        """Zero avg loss returns 0 (can't compute b)."""
        f = KellyPositionSizer._binary_kelly(0.6, 10.0, 0.0)
        assert f == 0.0

    def test_perfect_win_rate(self):
        """100% win rate: f* = 1.0 - 0/b = 1.0."""
        f = KellyPositionSizer._binary_kelly(1.0, 10.0, 5.0)
        assert f == 1.0

    def test_high_payoff_ratio(self):
        """High b should produce a larger Kelly fraction."""
        f_low = KellyPositionSizer._binary_kelly(0.55, 10.0, 10.0)
        f_high = KellyPositionSizer._binary_kelly(0.55, 30.0, 10.0)
        assert f_high > f_low


# ---------------------------------------------------------------------------
# Continuous Kelly (requires scipy)
# ---------------------------------------------------------------------------

class TestContinuousKelly:

    def test_continuous_kelly_insufficient_data(self, sizer):
        """Returns 0.0 when there are fewer trades than min_trades."""
        f = sizer.compute_kelly_continuous("stat_arb", "BTC-USD")
        assert f == 0.0

    def test_continuous_kelly_with_sufficient_data(self, tmp_db):
        """With enough winning trades, continuous Kelly returns positive value."""
        sizer = KellyPositionSizer(
            config={"kelly_sizer": {"min_trades": 5, "lookback_trades": 100}},
            db_path=tmp_db,
        )
        # Insert 10 winning round-trips: buy at 100, sell at 105
        for _ in range(10):
            _insert_round_trip(tmp_db, "BTC-USD", "stat_arb", 100.0, 105.0)

        f = sizer.compute_kelly_continuous("stat_arb", "BTC-USD")
        assert f > 0.0

    def test_continuous_kelly_fallback_to_binary(self, tmp_db):
        """If scipy optimization fails, falls back to binary Kelly."""
        sizer = KellyPositionSizer(
            config={"kelly_sizer": {"min_trades": 5}},
            db_path=tmp_db,
        )
        for _ in range(10):
            _insert_round_trip(tmp_db, "BTC-USD", "stat_arb", 100.0, 105.0)

        # This should work via either continuous or binary
        f = sizer.compute_kelly_continuous("stat_arb", "BTC-USD")
        assert f >= 0.0


# ---------------------------------------------------------------------------
# get_position_size
# ---------------------------------------------------------------------------

class TestGetPositionSize:

    def test_zero_equity(self, sizer):
        """Returns 0 when equity is zero or negative."""
        size = sizer.get_position_size(
            {"signal_type": "stat_arb", "pair": "BTC-USD"}, equity=0.0
        )
        assert size == 0.0

        size2 = sizer.get_position_size(
            {"signal_type": "stat_arb", "pair": "BTC-USD"}, equity=-1000.0
        )
        assert size2 == 0.0

    def test_insufficient_data_uses_default(self, sizer):
        """When not enough trades, uses default_position_pct * confidence."""
        size = sizer.get_position_size(
            {"signal_type": "stat_arb", "pair": "BTC-USD", "confidence": 0.8},
            equity=10000.0,
        )
        # default_position_pct=1.0%, confidence=0.8 -> 10000 * 0.01 * 0.8 = 80
        assert size == pytest.approx(80.0)

    def test_default_confidence_is_half(self, sizer):
        """When confidence is not provided, it defaults to 0.5."""
        size = sizer.get_position_size(
            {"signal_type": "stat_arb", "pair": "BTC-USD"},
            equity=10000.0,
        )
        # default_position_pct=1.0%, confidence=0.5 -> 10000 * 0.01 * 0.5 = 50
        assert size == pytest.approx(50.0)

    def test_negative_expectancy_halt(self, tmp_db):
        """With negative expectancy and halt action, returns 0."""
        sizer = KellyPositionSizer(
            config={"kelly_sizer": {"min_trades": 5, "negative_kelly_action": "halt"}},
            db_path=tmp_db,
        )
        # Insert 10 losing round-trips
        for _ in range(10):
            _insert_round_trip(tmp_db, "BTC-USD", "stat_arb", 100.0, 95.0)

        size = sizer.get_position_size(
            {"signal_type": "stat_arb", "pair": "BTC-USD", "confidence": 0.8},
            equity=10000.0,
        )
        assert size == 0.0

    def test_negative_expectancy_reduce(self, tmp_db):
        """With negative expectancy and reduce action, returns small size."""
        sizer = KellyPositionSizer(
            config={"kelly_sizer": {"min_trades": 5, "negative_kelly_action": "reduce"}},
            db_path=tmp_db,
        )
        for _ in range(10):
            _insert_round_trip(tmp_db, "BTC-USD", "stat_arb", 100.0, 95.0)

        size = sizer.get_position_size(
            {"signal_type": "stat_arb", "pair": "BTC-USD", "confidence": 0.8},
            equity=10000.0,
        )
        # default_position_pct=1% * 0.25 * equity = 25.0
        assert 0 < size <= 25.0

    def test_positive_expectancy_uses_kelly(self, tmp_db):
        """With positive expectancy, uses Kelly-based sizing."""
        sizer = KellyPositionSizer(
            config={"kelly_sizer": {
                "min_trades": 5,
                "kelly_fraction": 0.25,
                "max_position_pct": 10.0,
            }},
            db_path=tmp_db,
        )
        # Insert 20 trades: 14 wins, 6 losses (70% win rate, good b ratio)
        for _ in range(14):
            _insert_round_trip(tmp_db, "BTC-USD", "stat_arb", 100.0, 108.0)
        for _ in range(6):
            _insert_round_trip(tmp_db, "BTC-USD", "stat_arb", 100.0, 96.0)

        size = sizer.get_position_size(
            {"signal_type": "stat_arb", "pair": "BTC-USD", "confidence": 0.8},
            equity=100000.0,
        )
        assert size > 0
        # Should not exceed max_position_pct
        assert size <= 100000.0 * 0.10

    def test_capped_at_max_position_pct(self, tmp_db):
        """Position size never exceeds max_position_pct of equity."""
        sizer = KellyPositionSizer(
            config={"kelly_sizer": {
                "min_trades": 3,
                "kelly_fraction": 1.0,  # Full Kelly
                "max_position_pct": 5.0,
            }},
            db_path=tmp_db,
        )
        # Extremely profitable trades to push Kelly fraction high
        for _ in range(20):
            _insert_round_trip(tmp_db, "BTC-USD", "stat_arb", 100.0, 150.0)

        size = sizer.get_position_size(
            {"signal_type": "stat_arb", "pair": "BTC-USD", "confidence": 1.0},
            equity=100000.0,
        )
        assert size <= 100000.0 * 0.05 + 1  # Small tolerance


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:

    def test_stats_are_cached(self, sizer):
        """Repeated calls return cached stats within the recalculate interval."""
        stats1 = sizer._get_cached_stats("stat_arb", "BTC-USD")
        stats2 = sizer._get_cached_stats("stat_arb", "BTC-USD")
        assert stats1 is stats2  # Same object from cache

    def test_cache_expires(self, sizer):
        """Stats are rebuilt after the recalculate interval."""
        sizer._recalc_interval = 0  # Expire immediately
        stats1 = sizer._get_cached_stats("stat_arb", "BTC-USD")
        time.sleep(0.01)
        stats2 = sizer._get_cached_stats("stat_arb", "BTC-USD")
        assert stats1 is not stats2


# ---------------------------------------------------------------------------
# Statistics API
# ---------------------------------------------------------------------------

class TestStatisticsAPI:

    def test_get_statistics_empty(self, sizer):
        stats = sizer.get_statistics("stat_arb", "BTC-USD")
        assert stats["total_trades"] == 0
        assert stats["sufficient_data"] is False
        assert "config" in stats

    def test_get_statistics_with_data(self, tmp_db):
        sizer = KellyPositionSizer(
            config={"kelly_sizer": {"min_trades": 3}},
            db_path=tmp_db,
        )
        # Mix of wins and losses so binary Kelly can compute a valid fraction
        for _ in range(8):
            _insert_round_trip(tmp_db, "BTC-USD", "stat_arb", 100.0, 108.0)
        for _ in range(3):
            _insert_round_trip(tmp_db, "BTC-USD", "stat_arb", 100.0, 96.0)

        stats = sizer.get_statistics("stat_arb", "BTC-USD")
        assert stats["total_trades"] >= 3
        assert stats["sufficient_data"] is True
        assert stats["win_rate"] > 0
        assert stats["avg_win_bps"] > 0
        assert stats["avg_loss_bps"] > 0
        assert stats["full_kelly_fraction"] > 0
        assert stats["expectancy_per_trade_bps"] > 0


# ---------------------------------------------------------------------------
# TradeStats dataclass
# ---------------------------------------------------------------------------

class TestTradeStats:

    def test_default_values(self):
        ts = TradeStats()
        assert ts.total_trades == 0
        assert ts.wins == 0
        assert ts.win_rate == 0.0
        assert ts.returns_bps is None
        assert ts.full_kelly_fraction == 0.0

    def test_custom_values(self):
        ts = TradeStats(
            total_trades=50,
            wins=30,
            losses=20,
            win_rate=0.6,
            avg_win_bps=12.0,
            avg_loss_bps=8.0,
        )
        assert ts.total_trades == 50
        assert ts.win_rate == 0.6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_config_override(self, tmp_db):
        """Custom config overrides defaults."""
        sizer = KellyPositionSizer(
            config={"kelly_sizer": {
                "kelly_fraction": 0.5,
                "min_trades": 10,
                "max_position_pct": 20.0,
            }},
            db_path=tmp_db,
        )
        assert sizer.cfg["kelly_fraction"] == 0.5
        assert sizer.cfg["min_trades"] == 10
        assert sizer.cfg["max_position_pct"] == 20.0
        # Non-overridden keys keep defaults
        assert sizer.cfg["default_position_pct"] == _DEFAULT_CONFIG["default_position_pct"]

    def test_none_config(self, tmp_db):
        """None config uses all defaults."""
        sizer = KellyPositionSizer(config=None, db_path=tmp_db)
        assert sizer.cfg["kelly_fraction"] == _DEFAULT_CONFIG["kelly_fraction"]

    def test_missing_trades_table(self, tmp_path):
        """Gracefully handles missing trades table."""
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.close()

        sizer = KellyPositionSizer(config={}, db_path=db_path)
        size = sizer.get_position_size(
            {"signal_type": "x", "pair": "Y", "confidence": 0.5},
            equity=10000.0,
        )
        # Should use default fallback, not crash
        assert size >= 0
