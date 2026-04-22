"""
Tests for arbitrage/tracking/performance.py

Covers:
  - PerformanceTracker initialization and DB schema creation
  - Trade recording and in-memory aggregation
  - P&L tracking per strategy and per pair
  - Signal recording
  - Hourly PnL bucketing
  - Summary reporting (win rate, avg profit, by_strategy)
  - SQLite persistence
"""
import os
import pytest
import sqlite3
import tempfile
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

from arbitrage.tracking.performance import PerformanceTracker


def _make_execution_result(
    trade_id="test_001", status="filled",
    profit=Decimal('0.50'), strategy="cross_exchange",
    symbol="BTC/USDT", buy_exchange="mexc", sell_exchange="binance",
):
    signal = MagicMock()
    signal.signal_id = trade_id
    signal.signal_type = strategy
    signal.symbol = symbol
    signal.buy_exchange = buy_exchange
    signal.sell_exchange = sell_exchange
    signal.buy_price = Decimal('50000')
    signal.sell_price = Decimal('50050')
    signal.recommended_quantity = Decimal('0.01')
    signal.gross_spread_bps = Decimal('10')
    signal.net_spread_bps = Decimal('7')
    signal.expected_profit_usd = Decimal('0.35')
    signal.total_cost_bps = Decimal('3')
    signal.confidence = Decimal('0.7')

    result = MagicMock()
    result.trade_id = trade_id
    result.signal = signal
    result.status = status
    result.actual_profit_usd = profit
    result.realized_cost_bps = Decimal('3.2')

    buy_result = MagicMock()
    buy_result.fee_amount = Decimal('0')
    sell_result = MagicMock()
    sell_result.fee_amount = Decimal('0.375')

    result.buy_result = buy_result
    result.sell_result = sell_result

    return result


class TestPerformanceTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        db_path = str(tmp_path / "test_arb.db")
        return PerformanceTracker(db_path=db_path)

    def test_initialization_creates_tables(self, tracker):
        conn = sqlite3.connect(tracker.db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        conn.close()

        assert "arb_trades" in table_names
        assert "arb_signals" in table_names
        assert "arb_daily_summary" in table_names

    def test_record_filled_trade(self, tracker):
        result = _make_execution_result(profit=Decimal('0.50'))
        tracker.record_trade(result)

        assert tracker._total_trades == 1
        assert tracker._total_profit == Decimal('0.50')
        assert tracker._by_strategy["cross_exchange"]["trades"] == 1
        assert tracker._by_strategy["cross_exchange"]["wins"] == 1

    def test_record_losing_trade(self, tracker):
        result = _make_execution_result(profit=Decimal('-0.30'))
        tracker.record_trade(result)

        assert tracker._total_profit == Decimal('-0.30')
        assert tracker._by_strategy["cross_exchange"]["losses"] == 1

    def test_record_unfilled_trade(self, tracker):
        result = _make_execution_result(status="no_fill", profit=Decimal('0'))
        tracker.record_trade(result)

        assert tracker._total_trades == 1
        # Unfilled trades don't add to profit or strategy stats
        assert tracker._by_strategy["cross_exchange"]["trades"] == 0

    def test_multiple_strategies_tracked(self, tracker):
        cross = _make_execution_result(
            trade_id="t1", strategy="cross_exchange", profit=Decimal('0.50')
        )
        funding = _make_execution_result(
            trade_id="t2", strategy="funding_rate", profit=Decimal('0.30')
        )
        tri = _make_execution_result(
            trade_id="t3", strategy="triangular", profit=Decimal('0.10')
        )

        tracker.record_trade(cross)
        tracker.record_trade(funding)
        tracker.record_trade(tri)

        assert tracker._by_strategy["cross_exchange"]["trades"] == 1
        assert tracker._by_strategy["funding_rate"]["trades"] == 1
        assert tracker._by_strategy["triangular"]["trades"] == 1
        assert tracker._total_profit == Decimal('0.90')

    def test_per_pair_tracking(self, tracker):
        t1 = _make_execution_result(trade_id="t1", symbol="BTC/USDT", profit=Decimal('0.50'))
        t2 = _make_execution_result(trade_id="t2", symbol="ETH/USDT", profit=Decimal('0.20'))
        t3 = _make_execution_result(trade_id="t3", symbol="BTC/USDT", profit=Decimal('-0.10'))

        tracker.record_trade(t1)
        tracker.record_trade(t2)
        tracker.record_trade(t3)

        assert tracker._by_pair["BTC/USDT"]["trades"] == 2
        assert tracker._by_pair["BTC/USDT"]["profit_usd"] == pytest.approx(0.40)
        assert tracker._by_pair["ETH/USDT"]["trades"] == 1

    def test_hourly_pnl_bucketing(self, tracker):
        result = _make_execution_result(profit=Decimal('0.50'))
        tracker.record_trade(result)

        hour_key = datetime.utcnow().strftime("%Y-%m-%d %H:00")
        assert hour_key in tracker._hourly_pnl

    def test_record_signal(self, tracker):
        signal = MagicMock()
        signal.signal_id = "sig_001"
        signal.signal_type = "cross_exchange"
        signal.symbol = "BTC/USDT"
        signal.gross_spread_bps = Decimal('10')
        signal.net_spread_bps = Decimal('7')

        tracker.record_signal(signal, approved=True, executed=True)

        # Verify in DB
        conn = sqlite3.connect(tracker.db_path)
        rows = conn.execute("SELECT * FROM arb_signals").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_persist_trade_to_db(self, tracker):
        result = _make_execution_result()
        tracker.record_trade(result)

        conn = sqlite3.connect(tracker.db_path)
        rows = conn.execute("SELECT * FROM arb_trades").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_get_summary_empty(self, tracker):
        summary = tracker.get_summary()
        assert summary["total_trades"] == 0
        assert summary["total_fills"] == 0
        assert summary["total_profit_usd"] == 0.0
        assert summary["win_rate"] == 0

    def test_get_summary_with_data(self, tracker):
        tracker.record_trade(_make_execution_result(trade_id="t1", profit=Decimal('0.50')))
        tracker.record_trade(_make_execution_result(trade_id="t2", profit=Decimal('-0.10')))
        tracker.record_trade(_make_execution_result(trade_id="t3", profit=Decimal('0.30')))

        summary = tracker.get_summary()
        assert summary["total_trades"] == 3
        assert summary["total_fills"] == 3
        assert summary["total_profit_usd"] == pytest.approx(0.70)
        assert summary["win_rate"] == pytest.approx(2 / 3)

    def test_get_summary_top_pairs(self, tracker):
        for i in range(3):
            tracker.record_trade(_make_execution_result(
                trade_id=f"btc_{i}", symbol="BTC/USDT", profit=Decimal('0.50')
            ))
        tracker.record_trade(_make_execution_result(
            trade_id="eth_1", symbol="ETH/USDT", profit=Decimal('0.20')
        ))

        summary = tracker.get_summary()
        assert "BTC/USDT" in summary["top_pairs"]

    def test_empty_stats_helper(self, tracker):
        stats = tracker._empty_stats()
        assert stats == {'trades': 0, 'wins': 0, 'losses': 0, 'profit_usd': 0.0}
