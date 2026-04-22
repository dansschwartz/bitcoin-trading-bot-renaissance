"""Tests for agents.observation_collector.ObservationCollector."""

import json
import os
import sqlite3
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.db_schema import ensure_agent_tables
from agents.observation_collector import ObservationCollector


@pytest.fixture
def test_db():
    """Create a temp DB with all required tables + sample data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS five_minute_bars (
            pair TEXT, bar_start TEXT, bar_end TEXT,
            open REAL, high REAL, low REAL, close REAL, volume REAL
        );
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY, timestamp TEXT, product_id TEXT,
            action TEXT, confidence REAL, position_size REAL,
            weighted_signal REAL, reasoning TEXT, hmm_regime TEXT
        );
        CREATE TABLE IF NOT EXISTS signal_daily_pnl (
            id INTEGER PRIMARY KEY, date TEXT, signal_type TEXT, pnl REAL,
            num_trades INTEGER, win_rate REAL
        );
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY, timestamp TEXT, product_id TEXT,
            side TEXT, size REAL, price REAL, status TEXT,
            algo_used TEXT, slippage REAL, execution_time REAL
        );
        CREATE TABLE IF NOT EXISTS open_positions (
            position_id TEXT, product_id TEXT, side TEXT, size REAL,
            entry_price REAL, stop_loss_price REAL, take_profit_price REAL,
            opened_at TEXT, status TEXT
        );
        CREATE TABLE IF NOT EXISTS devil_tracker (
            trade_id TEXT PRIMARY KEY, signal_type TEXT, pair TEXT,
            side TEXT, exchange TEXT, signal_timestamp TEXT,
            signal_price REAL, fill_price REAL, slippage_bps REAL,
            latency_signal_to_fill_ms REAL, devil REAL
        );
        CREATE TABLE IF NOT EXISTS daily_performance (
            date TEXT, total_trades INTEGER, winning_trades INTEGER,
            losing_trades INTEGER, failed_trades INTEGER,
            gross_profit_usd REAL, total_fees_usd REAL, net_profit_usd REAL,
            cross_exchange_pnl REAL, funding_rate_pnl REAL,
            triangular_pnl REAL, directional_pnl REAL,
            max_drawdown_usd REAL, max_one_sided_usd REAL,
            emergency_closes INTEGER, avg_estimated_cost REAL,
            avg_realized_cost REAL, cost_estimation_error REAL,
            total_equity_usd REAL, coinbase_equity_usd REAL,
            mexc_equity_usd REAL, binance_equity_usd REAL
        );
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY, timestamp TEXT, model_name TEXT,
            confidence REAL
        );

        -- Insert sample data
        INSERT INTO decisions (timestamp, product_id, action, confidence, position_size,
                              weighted_signal, reasoning, hmm_regime)
        VALUES (datetime('now'), 'BTC-USD', 'BUY', 0.75, 100, 0.5, '{}', 'trending');

        INSERT INTO trades (timestamp, product_id, side, size, price, status)
        VALUES (datetime('now'), 'BTC-USD', 'BUY', 0.01, 50000, 'EXECUTED');
        INSERT INTO trades (timestamp, product_id, side, size, price, status)
        VALUES (datetime('now'), 'ETH-USD', 'SELL', 0.1, 2000, 'EXECUTED');

        INSERT INTO open_positions (position_id, product_id, side, size, entry_price, status)
        VALUES ('pos-1', 'BTC-USD', 'BUY', 0.01, 50000, 'open');

        INSERT INTO signal_daily_pnl (date, signal_type, pnl)
        VALUES (date('now'), 'rsi', 5.0);
        INSERT INTO signal_daily_pnl (date, signal_type, pnl)
        VALUES (date('now'), 'macd', -2.0);

        INSERT INTO daily_performance (date, net_profit_usd, total_trades, winning_trades, losing_trades)
        VALUES (date('now'), 7.50, 2, 1, 1);
        INSERT INTO daily_performance (date, net_profit_usd, total_trades, winning_trades, losing_trades)
        VALUES (date('now', '-1 day'), 3.00, 3, 2, 1);
    """)
    conn.commit()
    conn.close()
    ensure_agent_tables(path)
    yield path
    os.unlink(path)


class TestObservationCollector:
    def test_compile_report_structure(self, test_db):
        collector = ObservationCollector(db_path=test_db, config={})
        report = collector.compile_weekly_report()
        assert "meta" in report
        assert "portfolio" in report
        assert "signals" in report
        assert "regimes" in report
        assert "execution" in report
        assert "data_quality" in report
        assert "config_snapshot" in report
        assert "summary" in report

    def test_portfolio_section(self, test_db):
        collector = ObservationCollector(db_path=test_db, config={})
        report = collector.compile_weekly_report()
        portfolio = report["portfolio"]
        # Aggregated from daily_performance: 2+3=5 trades, 1+2=3 wins
        assert portfolio["total_trades"] == 5
        assert portfolio["wins"] == 3
        assert portfolio["win_rate"] == 0.6
        assert portfolio["total_pnl"] == 10.5  # 7.50 + 3.00
        assert portfolio["open_positions"] == 1

    def test_signals_section(self, test_db):
        collector = ObservationCollector(db_path=test_db, config={})
        report = collector.compile_weekly_report()
        signals = report["signals"]
        assert len(signals["per_signal_pnl"]) == 2
        # Best signal first
        assert signals["per_signal_pnl"][0]["signal"] == "rsi"

    def test_regimes_section(self, test_db):
        collector = ObservationCollector(db_path=test_db, config={})
        report = collector.compile_weekly_report()
        regimes = report["regimes"]
        assert len(regimes["regime_distribution"]) >= 1
        assert regimes["regime_distribution"][0]["regime"] == "trending"

    def test_summary_computed(self, test_db):
        collector = ObservationCollector(db_path=test_db, config={})
        report = collector.compile_weekly_report()
        summary = report["summary"]
        assert summary["total_pnl"] == 10.5
        assert summary["total_trades"] == 5
        assert summary["best_signal"] == "rsi"
        assert summary["worst_signal"] == "macd"

    def test_config_snapshot(self, test_db):
        config = {
            "signal_weights": {"rsi": 0.1, "macd": 0.15},
            "risk_management": {"position_limit": 1000},
        }
        collector = ObservationCollector(db_path=test_db, config=config)
        report = collector.compile_weekly_report()
        snap = report["config_snapshot"]
        assert snap["signal_weights"]["rsi"] == 0.1

    def test_save_report(self, test_db):
        collector = ObservationCollector(db_path=test_db, config={})
        report = collector.compile_weekly_report()
        filepath = collector.save_report(report)
        # Check DB entry
        conn = sqlite3.connect(test_db)
        row = conn.execute("SELECT COUNT(*) FROM weekly_reports").fetchone()
        assert row[0] == 1
        conn.close()

    def test_report_is_json_serializable(self, test_db):
        collector = ObservationCollector(db_path=test_db, config={})
        report = collector.compile_weekly_report()
        # Should not raise
        serialized = json.dumps(report, default=str)
        assert len(serialized) > 100
