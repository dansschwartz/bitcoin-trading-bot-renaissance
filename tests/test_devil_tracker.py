"""
Unit tests for core/devil_tracker.py
======================================
Tests DevilTracker: signal detection, order submission, fill recording,
round-trip devil computation, summary analytics, re-evaluation tracking,
and alert generation.

All tests use a temporary in-memory or tmpdir SQLite database.
"""

import os
import sqlite3
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from core.devil_tracker import DevilTracker, TradeExecution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database path."""
    return str(tmp_path / "test_devil.db")


@pytest.fixture
def tracker(tmp_db):
    """Return a fresh DevilTracker instance."""
    return DevilTracker(tmp_db)


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

class TestTableCreation:

    def test_table_created_on_init(self, tmp_db):
        """DevilTracker creates its table and indexes on construction."""
        dt = DevilTracker(tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "devil_tracker" in table_names
        conn.close()

    def test_idempotent_table_creation(self, tmp_db):
        """Creating two trackers on the same DB does not error."""
        dt1 = DevilTracker(tmp_db)
        dt2 = DevilTracker(tmp_db)
        # Should not raise
        assert dt1.db_path == dt2.db_path


# ---------------------------------------------------------------------------
# Signal detection (Phase 1)
# ---------------------------------------------------------------------------

class TestSignalDetection:

    def test_record_signal_detection_returns_trade_id(self, tracker):
        tid = tracker.record_signal_detection(
            signal_type="stat_arb",
            pair="BTC-USD",
            exchange="mexc",
            price=64350.0,
        )
        assert tid is not None
        assert isinstance(tid, str)
        assert len(tid) == 32  # uuid4 hex

    def test_record_signal_detection_default_side_is_buy(self, tracker):
        tid = tracker.record_signal_detection(
            "stat_arb", "ETH-USD", "mexc", 3200.0
        )
        trade = tracker.get_trade(tid)
        assert trade is not None
        assert trade.side == "BUY"

    def test_record_signal_detection_custom_side(self, tracker):
        tid = tracker.record_signal_detection(
            "stat_arb", "ETH-USD", "mexc", 3200.0, side="SELL"
        )
        trade = tracker.get_trade(tid)
        assert trade.side == "SELL"

    def test_record_signal_detection_persists(self, tracker, tmp_db):
        tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 64350.0
        )
        # Open a separate connection to verify persistence
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT pair, signal_price FROM devil_tracker WHERE trade_id=?",
            (tid,),
        ).fetchone()
        assert row is not None
        assert row[0] == "BTC-USD"
        assert row[1] == 64350.0
        conn.close()


# ---------------------------------------------------------------------------
# Order submission (Phase 2)
# ---------------------------------------------------------------------------

class TestOrderSubmission:

    def test_record_order_submission_success(self, tracker):
        tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 64350.0
        )
        result = tracker.record_order_submission(tid, 64352.0)
        assert result is True

        trade = tracker.get_trade(tid)
        assert trade.order_price == 64352.0
        assert trade.order_timestamp is not None

    def test_record_order_submission_computes_latency(self, tracker):
        tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 64350.0
        )
        # Small sleep to ensure non-zero latency
        time.sleep(0.01)
        tracker.record_order_submission(tid, 64352.0)
        trade = tracker.get_trade(tid)
        assert trade.latency_signal_to_order_ms is not None
        assert trade.latency_signal_to_order_ms >= 0

    def test_record_order_submission_missing_trade_id(self, tracker):
        result = tracker.record_order_submission("nonexistent_id", 64352.0)
        assert result is False


# ---------------------------------------------------------------------------
# Fill recording (Phase 3)
# ---------------------------------------------------------------------------

class TestFillRecording:

    def test_record_fill_success_buy(self, tracker):
        tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 64350.0, side="BUY"
        )
        tracker.record_order_submission(tid, 64352.0)
        result = tracker.record_fill(tid, 64355.0, 0.01, 0.12)
        assert result is True

        trade = tracker.get_trade(tid)
        assert trade.fill_price == 64355.0
        assert trade.fill_quantity == 0.01
        assert trade.fill_fee == 0.12
        assert trade.slippage_bps is not None
        assert trade.slippage_bps > 0  # fill > signal

    def test_record_fill_computes_slippage_bps(self, tracker):
        tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 10000.0, side="BUY"
        )
        tracker.record_fill(tid, 10010.0, 1.0, 0.0)
        trade = tracker.get_trade(tid)
        # |10010 - 10000| / 10000 * 10000 = 10 bps
        assert trade.slippage_bps == pytest.approx(10.0)

    def test_record_fill_computes_devil_buy(self, tracker):
        """For a BUY, lower fill is better. Devil = theoretical - actual."""
        tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 100.0, side="BUY"
        )
        # Fill at 102 (worse than signal 100), qty=1, fee=0.5
        tracker.record_fill(tid, 102.0, 1.0, 0.5)
        trade = tracker.get_trade(tid)
        # actual_pnl = (100 - 102) * 1 - 0.5 = -2.5
        # theoretical_pnl = 0 (baseline)
        # devil = 0 - (-2.5) = 2.5
        assert trade.actual_pnl == pytest.approx(-2.5)
        assert trade.theoretical_pnl == pytest.approx(0.0)
        assert trade.devil == pytest.approx(2.5)

    def test_record_fill_computes_devil_sell(self, tracker):
        """For a SELL, higher fill is better."""
        tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 100.0, side="SELL"
        )
        # Fill at 98 (worse than signal 100), qty=1, fee=0.5
        tracker.record_fill(tid, 98.0, 1.0, 0.5)
        trade = tracker.get_trade(tid)
        # actual_pnl = (98 - 100)*1 - 0.5 = -2.5
        # devil = 0 - (-2.5) = 2.5
        assert trade.actual_pnl == pytest.approx(-2.5)
        assert trade.devil == pytest.approx(2.5)

    def test_record_fill_missing_trade_id(self, tracker):
        result = tracker.record_fill("nonexistent_id", 64355.0, 0.01, 0.12)
        assert result is False


# ---------------------------------------------------------------------------
# Round-trip Devil
# ---------------------------------------------------------------------------

class TestRoundTripDevil:

    def test_compute_round_trip_long(self, tracker):
        """Round trip for a long trade: BUY entry, SELL exit."""
        entry_tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 100.0, side="BUY"
        )
        tracker.record_fill(entry_tid, 101.0, 1.0, 0.5)

        exit_tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 110.0, side="SELL"
        )
        tracker.record_fill(exit_tid, 109.0, 1.0, 0.5)

        devil = tracker.compute_devil_for_round_trip(entry_tid, exit_tid)
        assert devil is not None
        # theoretical = (110 - 100) * 1 = 10
        # actual = (109 - 101) * 1 - 0.5 - 0.5 = 7.0
        # devil = 10 - 7 = 3.0
        assert devil == pytest.approx(3.0)

    def test_compute_round_trip_missing_leg(self, tracker):
        entry_tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 100.0, side="BUY"
        )
        tracker.record_fill(entry_tid, 101.0, 1.0, 0.5)

        devil = tracker.compute_devil_for_round_trip(entry_tid, "nonexistent_id")
        assert devil is None

    def test_compute_round_trip_incomplete_data(self, tracker):
        """Return None when fills are missing (e.g., no fill on exit)."""
        entry_tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 100.0, side="BUY"
        )
        tracker.record_fill(entry_tid, 101.0, 1.0, 0.5)

        exit_tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 110.0, side="SELL"
        )
        # No fill recorded for exit
        devil = tracker.compute_devil_for_round_trip(entry_tid, exit_tid)
        assert devil is None


# ---------------------------------------------------------------------------
# Summary analytics
# ---------------------------------------------------------------------------

class TestDevilSummary:

    def test_summary_empty_db(self, tracker):
        summary = tracker.get_devil_summary(window_hours=24)
        assert summary["trade_count"] == 0
        assert summary["total_devil"] == 0.0
        assert summary["by_pair"] == {}

    def test_summary_with_data(self, tracker):
        """Summary aggregates correctly over filled trades."""
        for i in range(3):
            tid = tracker.record_signal_detection(
                "stat_arb", "BTC-USD", "mexc", 100.0, side="BUY"
            )
            tracker.record_fill(tid, 100.5, 1.0, 0.1)

        summary = tracker.get_devil_summary(window_hours=1)
        assert summary["trade_count"] == 3
        assert "BTC-USD" in summary["by_pair"]
        assert summary["avg_slippage_bps"] > 0

    def test_summary_respects_window(self, tracker, tmp_db):
        """Trades outside the window are excluded."""
        tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 100.0
        )
        tracker.record_fill(tid, 100.5, 1.0, 0.1)

        # Manually backdate the signal_timestamp to 48 hours ago
        old_ts = (
            datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        )
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "UPDATE devil_tracker SET signal_timestamp=? WHERE trade_id=?",
            (old_ts, tid),
        )
        conn.commit()
        conn.close()

        summary = tracker.get_devil_summary(window_hours=24)
        assert summary["trade_count"] == 0


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

class TestAlerts:

    def test_should_alert_empty_db(self, tracker):
        assert tracker.should_alert() is None

    def test_should_alert_below_threshold(self, tracker):
        # Record a trade with minimal slippage
        tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 10000.0
        )
        tracker.record_fill(tid, 10000.01, 1.0, 0.0)
        alert = tracker.should_alert(threshold_bps=100.0)
        assert alert is None


# ---------------------------------------------------------------------------
# Re-evaluation tracking (Doc 10)
# ---------------------------------------------------------------------------

class TestReevalTracking:

    def test_record_trim(self, tracker):
        """record_trim inserts a row into reeval_events."""
        tracker.record_trim(
            position_id="pos-001",
            pair="BTC-USD",
            trim_size=0.005,
            trim_price=64000.0,
            actual_cost_bps=1.5,
        )
        # Verify
        conn = sqlite3.connect(tracker.db_path)
        row = conn.execute(
            "SELECT * FROM reeval_events WHERE event_type='trim'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_record_exit(self, tracker):
        """record_exit inserts a row into reeval_events."""
        tracker.record_exit(
            position_id="pos-002",
            pair="ETH-USD",
            side="long",
            entry_price=3200.0,
            exit_price=3250.0,
            size=0.5,
            estimated_cost_bps=2.0,
            actual_cost_bps=2.5,
            reason_code="EDGE_CONSUMED",
            hold_time_seconds=180.0,
            adjustments=2,
            pnl_bps=15.0,
        )
        conn = sqlite3.connect(tracker.db_path)
        row = conn.execute(
            "SELECT * FROM reeval_events WHERE event_type='exit'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_reeval_effectiveness_report_empty(self, tracker):
        """Report returns zeroed metrics when no reeval_events exist."""
        report = tracker.get_reeval_effectiveness_report()
        assert report["total_exits"] == 0
        assert report["total_trims"] == 0
        assert report["avg_pnl_reeval_bps"] == 0.0

    def test_reeval_effectiveness_report_with_data(self, tracker):
        """Report correctly splits hard vs reeval exits."""
        # Reeval exit
        tracker.record_exit(
            position_id="pos-001",
            reason_code="EDGE_CONSUMED",
            pnl_bps=10.0,
            hold_time_seconds=60.0,
            adjustments=1,
        )
        # Hard exit
        tracker.record_exit(
            position_id="pos-002",
            reason_code="HARD_TIME_EXPIRED",
            pnl_bps=-5.0,
            hold_time_seconds=300.0,
            adjustments=0,
        )
        # Trim
        tracker.record_trim(
            position_id="pos-001",
            actual_cost_bps=1.2,
        )

        report = tracker.get_reeval_effectiveness_report()
        assert report["total_exits"] == 2
        assert report["reeval_exits"] == 1
        assert report["hard_exits"] == 1
        assert report["total_trims"] == 1
        assert report["avg_pnl_reeval_bps"] == 10.0
        assert report["avg_pnl_hard_bps"] == -5.0
        assert report["total_trim_cost_bps"] == 1.2


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class TestGetTrade:

    def test_get_trade_returns_trade_execution(self, tracker):
        tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 64350.0
        )
        trade = tracker.get_trade(tid)
        assert isinstance(trade, TradeExecution)
        assert trade.trade_id == tid
        assert trade.pair == "BTC-USD"
        assert trade.signal_price == 64350.0

    def test_get_trade_nonexistent(self, tracker):
        assert tracker.get_trade("nonexistent") is None

    def test_get_trade_includes_fill_data(self, tracker):
        tid = tracker.record_signal_detection(
            "stat_arb", "BTC-USD", "mexc", 100.0
        )
        tracker.record_fill(tid, 100.5, 1.0, 0.1)
        trade = tracker.get_trade(tid)
        assert trade.fill_price == 100.5
        assert trade.fill_quantity == 1.0
        assert trade.fill_fee == 0.1
