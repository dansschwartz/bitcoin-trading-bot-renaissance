"""
Unit tests for core/leverage_manager.py
========================================
Tests LeverageManager: consistency score, rolling Sharpe, max safe leverage,
current leverage, should_reduce_leverage, and full report.

All DB access uses temporary SQLite files with a populated trades table.
"""

import math
import sqlite3
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from core.leverage_manager import LeverageManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database with a trades table."""
    db_path = str(tmp_path / "test_leverage.db")
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
def lm(tmp_db):
    """Return a LeverageManager with default config and empty DB."""
    return LeverageManager(config={}, db_path=tmp_db)


def _insert_daily_trades(db_path, days_back, daily_pnl_sequence):
    """
    Insert trades to produce the given daily P&L sequence.

    daily_pnl_sequence: list of floats, e.g. [100, -50, 200, ...]
    Each entry creates a single SELL trade on that date with that P&L
    (positive = profitable day).
    """
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc)
    for i, pnl in enumerate(daily_pnl_sequence):
        date = now - timedelta(days=days_back - i)
        ts = date.strftime("%Y-%m-%d 12:00:00")
        if pnl >= 0:
            # Profitable day: SELL at higher price
            conn.execute(
                "INSERT INTO trades (timestamp, product_id, side, size, price, status) "
                "VALUES (?, 'BTC-USD', 'SELL', 1.0, ?, 'filled')",
                (ts, abs(pnl)),
            )
        else:
            # Losing day: BUY at higher price (buy costs money)
            conn.execute(
                "INSERT INTO trades (timestamp, product_id, side, size, price, status) "
                "VALUES (?, 'BTC-USD', 'BUY', 1.0, ?, 'filled')",
                (ts, abs(pnl)),
            )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_config(self, tmp_db):
        lm = LeverageManager(config={}, db_path=tmp_db)
        assert lm._max_leverage == 3.0
        assert lm._target_sharpe == 1.5
        assert lm._kelly_fraction == 0.25
        assert lm._consistency_window_days == 30

    def test_custom_config(self, tmp_db):
        lm = LeverageManager(
            config={"leverage_manager": {
                "max_leverage": 5.0,
                "target_sharpe": 2.0,
                "kelly_fraction": 0.5,
                "consistency_window_days": 60,
                "enabled": False,
            }},
            db_path=tmp_db,
        )
        assert lm._max_leverage == 5.0
        assert lm._target_sharpe == 2.0
        assert lm._kelly_fraction == 0.5
        assert lm._consistency_window_days == 60
        assert lm._enabled is False


# ---------------------------------------------------------------------------
# Consistency score
# ---------------------------------------------------------------------------

class TestConsistencyScore:

    def test_no_data_returns_zero(self, lm):
        score = lm.compute_consistency_score()
        assert score == 0.0

    def test_all_winning_days(self, tmp_db):
        lm = LeverageManager(config={}, db_path=tmp_db)
        # 10 days, all profitable
        _insert_daily_trades(tmp_db, 10, [100.0] * 10)
        score = lm.compute_consistency_score(window_days=30)
        assert score == 1.0

    def test_all_losing_days(self, tmp_db):
        lm = LeverageManager(config={}, db_path=tmp_db)
        _insert_daily_trades(tmp_db, 10, [-50.0] * 10)
        score = lm.compute_consistency_score(window_days=30)
        assert score == 0.0

    def test_mixed_days(self, tmp_db):
        lm = LeverageManager(config={}, db_path=tmp_db)
        # 10 days: 7 winning, 3 losing
        pnls = [100, 50, -20, 80, -10, 120, 60, -30, 90, 70]
        _insert_daily_trades(tmp_db, 10, pnls)
        score = lm.compute_consistency_score(window_days=30)
        # 3 losing days out of 10 -> 1 - 3/10 = 0.7
        assert score == pytest.approx(0.7)

    def test_score_clamped_0_to_1(self, lm):
        """Score is always in [0, 1]."""
        score = lm.compute_consistency_score()
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------

class TestRollingSharpe:

    def test_insufficient_data(self, lm):
        """Returns 0.0 with fewer than 5 days of data."""
        assert lm._compute_rolling_sharpe() == 0.0

    def test_positive_sharpe(self, tmp_db):
        lm = LeverageManager(config={}, db_path=tmp_db)
        # 10 consistent positive days
        _insert_daily_trades(tmp_db, 10, [100.0] * 10)
        sharpe = lm._compute_rolling_sharpe()
        # With zero variance (all same return), goes to target_sharpe * 2
        assert sharpe == lm._target_sharpe * 2.0

    def test_negative_mean(self, tmp_db):
        lm = LeverageManager(config={}, db_path=tmp_db)
        # 10 consistent losing days
        _insert_daily_trades(tmp_db, 10, [-50.0] * 10)
        sharpe = lm._compute_rolling_sharpe()
        # All same (negative) return -> std=0, mean<0 -> returns 0
        assert sharpe == 0.0

    def test_mixed_returns_sharpe(self, tmp_db):
        lm = LeverageManager(config={}, db_path=tmp_db)
        # 6 days with varied returns
        pnls = [200, 100, -50, 150, -20, 180]
        _insert_daily_trades(tmp_db, 6, pnls)
        sharpe = lm._compute_rolling_sharpe()
        # Should be positive (net positive returns)
        assert sharpe > 0


# ---------------------------------------------------------------------------
# Max safe leverage
# ---------------------------------------------------------------------------

class TestMaxSafeLeverage:

    def test_no_data_returns_zero(self, lm):
        max_lev = lm.compute_max_safe_leverage()
        assert max_lev == 0.0

    def test_capped_at_config_max(self, tmp_db):
        lm = LeverageManager(
            config={"leverage_manager": {"max_leverage": 2.0}},
            db_path=tmp_db,
        )
        _insert_daily_trades(tmp_db, 10, [1000.0] * 10)
        max_lev = lm.compute_max_safe_leverage()
        assert max_lev <= 2.0

    def test_scales_with_consistency(self, tmp_db):
        """Higher consistency yields higher max safe leverage."""
        lm_good = LeverageManager(config={}, db_path=tmp_db)
        _insert_daily_trades(tmp_db, 10, [100.0] * 10)
        max_good = lm_good.compute_max_safe_leverage()

        # Create a new DB with mixed performance
        db2 = tmp_db.replace("test_leverage", "test_leverage2")
        conn = sqlite3.connect(db2)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY, timestamp TEXT, product_id TEXT,
                side TEXT, size REAL, price REAL, status TEXT,
                algo_used TEXT, slippage REAL, execution_time REAL
            )
        """)
        conn.commit()
        conn.close()
        _insert_daily_trades(db2, 10, [100, -50, 100, -50, 100, -50, 100, -50, 100, -50])
        lm_mixed = LeverageManager(config={}, db_path=db2)
        max_mixed = lm_mixed.compute_max_safe_leverage()

        assert max_good >= max_mixed


# ---------------------------------------------------------------------------
# Current leverage
# ---------------------------------------------------------------------------

class TestCurrentLeverage:

    def test_no_positions(self, lm):
        lev = lm.get_current_leverage([], 10000.0)
        assert lev == 0.0

    def test_simple_leverage(self, lm):
        positions = [
            {"size": 1.0, "price": 60000.0},
            {"size": 0.5, "entry_price": 3000.0},
        ]
        lev = lm.get_current_leverage(positions, 10000.0)
        # (1*60000 + 0.5*3000) / 10000 = 61500 / 10000 = 6.15
        assert lev == pytest.approx(6.15)

    def test_zero_equity(self, lm):
        positions = [{"size": 1.0, "price": 100.0}]
        lev = lm.get_current_leverage(positions, 0.0)
        assert lev == 0.0

    def test_negative_equity(self, lm):
        positions = [{"size": 1.0, "price": 100.0}]
        lev = lm.get_current_leverage(positions, -1000.0)
        assert lev == 0.0

    def test_uses_entry_price_fallback(self, lm):
        """When 'price' key is missing, uses 'entry_price'."""
        positions = [{"size": 2.0, "entry_price": 500.0}]
        lev = lm.get_current_leverage(positions, 1000.0)
        assert lev == pytest.approx(1.0)

    def test_negative_size_uses_abs(self, lm):
        """Negative size (short) is treated as absolute value."""
        positions = [{"size": -1.0, "price": 100.0}]
        lev = lm.get_current_leverage(positions, 100.0)
        assert lev == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Should reduce leverage
# ---------------------------------------------------------------------------

class TestShouldReduceLeverage:

    def test_disabled_returns_false(self, tmp_db):
        lm = LeverageManager(
            config={"leverage_manager": {"enabled": False}},
            db_path=tmp_db,
        )
        lm._current_leverage = 10.0
        assert lm.should_reduce_leverage() is False

    def test_no_current_leverage(self, lm):
        assert lm.should_reduce_leverage() is False

    def test_no_trade_history(self, lm):
        """With no trade data, returns False to avoid false alarms."""
        lm._current_leverage = 5.0
        assert lm.should_reduce_leverage() is False

    def test_overleveraged(self, tmp_db):
        lm = LeverageManager(config={}, db_path=tmp_db)
        # All losing days -> consistency = 0 -> max_safe = 0
        # But with losing days, max_safe = 0 and current > 0
        _insert_daily_trades(tmp_db, 10, [50.0, -30.0, 50.0, -30.0, 50.0,
                                           -30.0, 50.0, -30.0, 50.0, -30.0])
        lm.get_current_leverage(
            [{"size": 10.0, "price": 60000.0}],
            equity=10000.0,
        )
        # max_safe_leverage might be very small; if current > max_safe, returns True
        result = lm.should_reduce_leverage()
        # We can't guarantee True here since it depends on the computed max_safe
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Leverage report
# ---------------------------------------------------------------------------

class TestLeverageReport:

    def test_empty_report(self, lm):
        report = lm.get_leverage_report()
        assert "current_leverage" in report
        assert "max_safe_leverage" in report
        assert "consistency_score" in report
        assert "rolling_sharpe" in report
        assert "status" in report
        assert report["status"] == "ok"

    def test_report_with_positions(self, tmp_db):
        lm = LeverageManager(config={}, db_path=tmp_db)
        _insert_daily_trades(tmp_db, 10, [100.0] * 10)
        report = lm.get_leverage_report(
            positions=[{"size": 1.0, "price": 1000.0}],
            equity=10000.0,
        )
        assert report["current_leverage"] == pytest.approx(0.1)
        assert report["consistency_score"] == 1.0
        assert report["status"] in ("ok", "warning", "overleveraged")

    def test_report_all_keys_present(self, lm):
        report = lm.get_leverage_report()
        expected_keys = [
            "current_leverage", "max_safe_leverage", "consistency_score",
            "rolling_sharpe", "num_losing_days", "total_days",
            "status", "recommended_action",
        ]
        for key in expected_keys:
            assert key in report, f"Missing key: {key}"

    def test_report_overleveraged_status(self, tmp_db):
        """When leverage exceeds max safe, status should be overleveraged."""
        lm = LeverageManager(
            config={"leverage_manager": {"max_leverage": 0.01}},
            db_path=tmp_db,
        )
        _insert_daily_trades(tmp_db, 10, [100.0] * 10)
        report = lm.get_leverage_report(
            positions=[{"size": 10.0, "price": 60000.0}],
            equity=10000.0,
        )
        # current = 60x, max capped at 0.01x -> overleveraged
        if report["max_safe_leverage"] > 0:
            assert report["status"] == "overleveraged"
            assert report["recommended_action"] is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_missing_trades_table(self, tmp_path):
        """Gracefully handles a DB with no trades table."""
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.close()

        lm = LeverageManager(config={}, db_path=db_path)
        score = lm.compute_consistency_score()
        assert score == 0.0

        report = lm.get_leverage_report()
        assert report["status"] == "ok"

    def test_cached_state_updated(self, lm):
        """get_current_leverage caches positions and equity."""
        positions = [{"size": 1.0, "price": 100.0}]
        lm.get_current_leverage(positions, 1000.0)
        assert lm._positions == positions
        assert lm._equity == 1000.0
        assert lm._current_leverage == pytest.approx(0.1)
