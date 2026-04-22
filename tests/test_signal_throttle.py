"""
Unit tests for core/signal_throttle.py
========================================
Tests SignalThrottle: daily P&L tracking, consecutive losing day counting,
signal health checks, throttle/disable/re-enable logic, allocation multipliers,
and end-of-day aggregation.

All DB access uses temporary SQLite files.
"""

import sqlite3
from datetime import datetime, timezone, timedelta

import pytest

from core.signal_throttle import SignalThrottle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database path."""
    return str(tmp_path / "test_throttle.db")


@pytest.fixture
def throttle(tmp_db):
    """Return a SignalThrottle with default config."""
    return SignalThrottle(config={}, db_path=tmp_db)


def _insert_daily_pnl(db_path, signal_type, date_str, pnl, num_trades=5, win_rate=0.5):
    """Insert a record into signal_daily_pnl."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO signal_daily_pnl
           (signal_type, date, pnl, num_trades, win_rate)
           VALUES (?, ?, ?, ?, ?)""",
        (signal_type, date_str, pnl, num_trades, win_rate),
    )
    conn.commit()
    conn.close()


def _insert_pnl_sequence(db_path, signal_type, pnl_sequence, days_back=None):
    """Insert a sequence of daily P&L values, most recent last."""
    if days_back is None:
        days_back = len(pnl_sequence)
    now = datetime.now(timezone.utc)
    for i, pnl in enumerate(pnl_sequence):
        date = now - timedelta(days=days_back - i - 1)
        date_str = date.strftime("%Y-%m-%d")
        _insert_daily_pnl(db_path, signal_type, date_str, pnl)


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

class TestTableCreation:

    def test_table_created_on_init(self, tmp_db):
        st = SignalThrottle(config={}, db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "signal_daily_pnl" in table_names
        conn.close()

    def test_idempotent_init(self, tmp_db):
        """Multiple inits on same DB do not error."""
        st1 = SignalThrottle(config={}, db_path=tmp_db)
        st2 = SignalThrottle(config={}, db_path=tmp_db)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:

    def test_default_config(self, tmp_db):
        st = SignalThrottle(config={}, db_path=tmp_db)
        assert st._throttle_after == 3
        assert st._disable_after == 5
        assert st._re_enable_after == 3
        assert st._throttle_reduction == 0.5

    def test_custom_config(self, tmp_db):
        st = SignalThrottle(
            config={"medallion_signal_throttle": {
                "throttle_after_consecutive_losing_days": 2,
                "disable_after_consecutive_losing_days": 4,
                "re_enable_after_profitable_days": 2,
                "throttle_reduction": 0.3,
            }},
            db_path=tmp_db,
        )
        assert st._throttle_after == 2
        assert st._disable_after == 4
        assert st._re_enable_after == 2
        assert st._throttle_reduction == 0.3


# ---------------------------------------------------------------------------
# Consecutive day counting (static methods)
# ---------------------------------------------------------------------------

class TestConsecutiveDayCounting:

    def test_count_consecutive_losing_empty(self):
        assert SignalThrottle._count_consecutive_losing_days([]) == 0

    def test_count_consecutive_losing_all_positive(self):
        data = [("2024-01-01", 10.0), ("2024-01-02", 20.0), ("2024-01-03", 5.0)]
        assert SignalThrottle._count_consecutive_losing_days(data) == 0

    def test_count_consecutive_losing_all_negative(self):
        data = [("2024-01-01", -10.0), ("2024-01-02", -20.0), ("2024-01-03", -5.0)]
        assert SignalThrottle._count_consecutive_losing_days(data) == 3

    def test_count_consecutive_losing_mixed(self):
        data = [
            ("2024-01-01", 10.0),
            ("2024-01-02", -5.0),
            ("2024-01-03", -3.0),
        ]
        assert SignalThrottle._count_consecutive_losing_days(data) == 2

    def test_count_consecutive_losing_interrupted(self):
        data = [
            ("2024-01-01", -10.0),
            ("2024-01-02", 5.0),
            ("2024-01-03", -3.0),
        ]
        assert SignalThrottle._count_consecutive_losing_days(data) == 1

    def test_count_consecutive_profitable_empty(self):
        assert SignalThrottle._count_consecutive_profitable_days([]) == 0

    def test_count_consecutive_profitable_all_positive(self):
        data = [("2024-01-01", 10.0), ("2024-01-02", 20.0)]
        assert SignalThrottle._count_consecutive_profitable_days(data) == 2

    def test_count_consecutive_profitable_zero_counts(self):
        """Zero P&L is considered profitable (pnl >= 0)."""
        data = [("2024-01-01", 0.0), ("2024-01-02", 0.0)]
        assert SignalThrottle._count_consecutive_profitable_days(data) == 2

    def test_count_consecutive_profitable_mixed(self):
        data = [
            ("2024-01-01", -5.0),
            ("2024-01-02", 10.0),
            ("2024-01-03", 20.0),
        ]
        assert SignalThrottle._count_consecutive_profitable_days(data) == 2


# ---------------------------------------------------------------------------
# Signal health check
# ---------------------------------------------------------------------------

class TestCheckSignalHealth:

    def test_no_data_returns_active(self, throttle):
        """When no data exists, signal is active with full allocation."""
        health = throttle.check_signal_health("stat_arb")
        assert health["status"] == "active"
        assert health["allocation_multiplier"] == 1.0
        assert health["consecutive_losing_days"] == 0

    def test_active_signal(self, throttle, tmp_db):
        """Signal with recent profits is active."""
        _insert_pnl_sequence(tmp_db, "stat_arb", [10.0, 20.0, 5.0])
        health = throttle.check_signal_health("stat_arb")
        assert health["status"] == "active"
        assert health["allocation_multiplier"] == 1.0

    def test_throttled_signal(self, throttle, tmp_db):
        """3+ consecutive losing days triggers throttle."""
        _insert_pnl_sequence(tmp_db, "stat_arb", [10.0, -5.0, -3.0, -8.0])
        health = throttle.check_signal_health("stat_arb")
        assert health["status"] == "throttled"
        assert health["allocation_multiplier"] == 0.5
        assert health["consecutive_losing_days"] == 3

    def test_disabled_signal(self, throttle, tmp_db):
        """5+ consecutive losing days triggers disable."""
        _insert_pnl_sequence(tmp_db, "stat_arb",
                             [10.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        health = throttle.check_signal_health("stat_arb")
        assert health["status"] == "disabled"
        assert health["allocation_multiplier"] == 0.0
        assert health["consecutive_losing_days"] == 5

    def test_re_enabled_after_profitable_days(self, throttle, tmp_db):
        """
        Signal that was disabled gets re-enabled after 3 consecutive
        profitable days.

        Note: The re-enable check in the code looks at consecutive profitable
        days at the END of the sequence. If there are 5 losing + 3 profitable,
        the consecutive_losing count will be 0 (since they're not at the end
        anymore). But the consecutive_profitable count will be 3, which
        triggers re-enable.
        """
        # 5 losing days followed by 3 profitable
        _insert_pnl_sequence(tmp_db, "stat_arb",
                             [-1.0, -2.0, -3.0, -4.0, -5.0, 10.0, 20.0, 30.0])
        health = throttle.check_signal_health("stat_arb")
        # consecutive_losing = 0 (last 3 are profitable)
        # so status would already be "active"
        assert health["status"] == "active"
        assert health["allocation_multiplier"] == 1.0

    def test_action_taken_message(self, throttle, tmp_db):
        """When throttled, action_taken contains descriptive message."""
        _insert_pnl_sequence(tmp_db, "stat_arb", [10.0, -5.0, -3.0, -8.0])
        health = throttle.check_signal_health("stat_arb")
        assert health["action_taken"] is not None
        assert "throttled" in health["action_taken"].lower()


# ---------------------------------------------------------------------------
# is_signal_allowed
# ---------------------------------------------------------------------------

class TestIsSignalAllowed:

    def test_allowed_when_no_data(self, throttle):
        assert throttle.is_signal_allowed("stat_arb") is True

    def test_allowed_when_active(self, throttle, tmp_db):
        _insert_pnl_sequence(tmp_db, "stat_arb", [10.0, 20.0])
        assert throttle.is_signal_allowed("stat_arb") is True

    def test_allowed_when_throttled(self, throttle, tmp_db):
        """Throttled signals are still allowed (just reduced)."""
        _insert_pnl_sequence(tmp_db, "stat_arb", [10.0, -5.0, -3.0, -8.0])
        assert throttle.is_signal_allowed("stat_arb") is True

    def test_not_allowed_when_disabled(self, throttle, tmp_db):
        _insert_pnl_sequence(tmp_db, "stat_arb",
                             [10.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        assert throttle.is_signal_allowed("stat_arb") is False


# ---------------------------------------------------------------------------
# get_allocation_multiplier
# ---------------------------------------------------------------------------

class TestGetAllocationMultiplier:

    def test_full_allocation_when_active(self, throttle):
        assert throttle.get_allocation_multiplier("stat_arb") == 1.0

    def test_half_allocation_when_throttled(self, throttle, tmp_db):
        _insert_pnl_sequence(tmp_db, "stat_arb", [10.0, -5.0, -3.0, -8.0])
        assert throttle.get_allocation_multiplier("stat_arb") == 0.5

    def test_zero_allocation_when_disabled(self, throttle, tmp_db):
        _insert_pnl_sequence(tmp_db, "stat_arb",
                             [10.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        assert throttle.get_allocation_multiplier("stat_arb") == 0.0


# ---------------------------------------------------------------------------
# get_all_signal_health
# ---------------------------------------------------------------------------

class TestGetAllSignalHealth:

    def test_empty_db(self, throttle):
        result = throttle.get_all_signal_health()
        assert result == {}

    def test_multiple_signals(self, throttle, tmp_db):
        _insert_pnl_sequence(tmp_db, "stat_arb", [10.0, 20.0])
        _insert_pnl_sequence(tmp_db, "momentum", [-5.0, -3.0, -8.0])
        result = throttle.get_all_signal_health()
        assert "stat_arb" in result
        assert "momentum" in result
        assert result["stat_arb"]["status"] == "active"
        assert result["momentum"]["status"] == "throttled"


# ---------------------------------------------------------------------------
# update_daily
# ---------------------------------------------------------------------------

class TestUpdateDaily:

    def test_update_daily_no_trades_table(self, throttle):
        """Gracefully handles missing trades table."""
        summary = throttle.update_daily("2024-01-15")
        assert summary == {}

    def test_update_daily_with_trades(self, throttle, tmp_db):
        """Aggregates trades by signal type from trades table."""
        # Create the trades table
        conn = sqlite3.connect(tmp_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
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
        # Insert trades for 2024-01-15
        conn.execute(
            "INSERT INTO trades (timestamp, product_id, side, size, price, status, algo_used, slippage) "
            "VALUES ('2024-01-15 10:00:00', 'BTC-USD', 'SELL', 1.0, 100.0, 'filled', 'stat_arb', 0.0)"
        )
        conn.execute(
            "INSERT INTO trades (timestamp, product_id, side, size, price, status, algo_used, slippage) "
            "VALUES ('2024-01-15 11:00:00', 'BTC-USD', 'BUY', 1.0, 80.0, 'filled', 'stat_arb', 0.0)"
        )
        conn.commit()
        conn.close()

        summary = throttle.update_daily("2024-01-15")
        assert "stat_arb" in summary
        assert summary["stat_arb"]["num_trades"] == 2

    def test_update_daily_skips_null_algo(self, throttle, tmp_db):
        """Trades with NULL algo_used are skipped."""
        conn = sqlite3.connect(tmp_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
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
        conn.execute(
            "INSERT INTO trades (timestamp, product_id, side, size, price, status, algo_used) "
            "VALUES ('2024-01-15 10:00:00', 'BTC-USD', 'SELL', 1.0, 100.0, 'filled', NULL)"
        )
        conn.commit()
        conn.close()

        summary = throttle.update_daily("2024-01-15")
        assert summary == {}

    def test_update_daily_defaults_to_yesterday(self, throttle, tmp_db):
        """When target_date is None, uses yesterday."""
        # Just verifies it doesn't crash
        summary = throttle.update_daily()
        assert isinstance(summary, dict)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_daily_pnl_lookback(self, throttle, tmp_db):
        """Only looks back 7 days by default."""
        now = datetime.now(timezone.utc)
        # Insert old data (15 days ago)
        old_date = (now - timedelta(days=15)).strftime("%Y-%m-%d")
        _insert_daily_pnl(tmp_db, "stat_arb", old_date, -100.0)
        # Insert recent data
        recent_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        _insert_daily_pnl(tmp_db, "stat_arb", recent_date, 50.0)

        health = throttle.check_signal_health("stat_arb")
        assert health["consecutive_losing_days"] == 0
        assert len(health["daily_pnl_history"]) == 1  # Only recent day

    def test_different_signal_types_independent(self, throttle, tmp_db):
        """Each signal type is tracked independently."""
        _insert_pnl_sequence(tmp_db, "stat_arb", [10.0, 20.0])
        _insert_pnl_sequence(tmp_db, "momentum", [-5.0, -3.0, -8.0, -2.0, -1.0])

        assert throttle.is_signal_allowed("stat_arb") is True
        assert throttle.is_signal_allowed("momentum") is False
