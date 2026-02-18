"""Tests for agent wrappers — get_status(), get_observations(), on_cycle_complete()."""

import os
import sqlite3
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.event_bus import EventBus
from agents.db_schema import ensure_agent_tables
from agents.data_agent import DataCollectionAgent
from agents.signal_agent import SignalAgent
from agents.risk_agent import RiskAgent
from agents.execution_agent import ExecutionAgent
from agents.portfolio_agent import PortfolioAgent
from agents.monitoring_agent import MonitoringAgent
from agents.meta_agent import MetaAgent


@pytest.fixture
def test_db():
    """Create a temp DB with agent tables + minimal trading tables."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    # Create minimal tables agents query
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
            id INTEGER PRIMARY KEY, date TEXT, signal_type TEXT, daily_pnl REAL
        );
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY, close_time TEXT, pnl REAL
        );
        CREATE TABLE IF NOT EXISTS open_positions (
            id INTEGER PRIMARY KEY, status TEXT
        );
        CREATE TABLE IF NOT EXISTS devil_tracker (
            trade_id TEXT PRIMARY KEY, signal_type TEXT, pair TEXT,
            side TEXT, exchange TEXT, signal_timestamp TEXT,
            signal_price REAL, fill_price REAL, slippage_bps REAL,
            latency_signal_to_fill_ms REAL, devil REAL
        );
        CREATE TABLE IF NOT EXISTS daily_performance (
            id INTEGER PRIMARY KEY, date TEXT, total_pnl REAL,
            sharpe_ratio REAL, total_trades INTEGER, win_rate REAL
        );
        CREATE TABLE IF NOT EXISTS balance_snapshots (
            id INTEGER PRIMARY KEY, timestamp TEXT, total_balance_usd REAL
        );
        CREATE TABLE IF NOT EXISTS signal_throttle_log (
            id INTEGER PRIMARY KEY, timestamp TEXT, signal_type TEXT,
            action TEXT, reason TEXT
        );
        CREATE TABLE IF NOT EXISTS reeval_events (
            id INTEGER PRIMARY KEY, timestamp TEXT
        );
    """)
    conn.commit()
    conn.close()
    ensure_agent_tables(path)
    yield path
    os.unlink(path)


@pytest.fixture
def bus():
    return EventBus()


class TestDataCollectionAgent:
    def test_get_status(self, bus, test_db):
        agent = DataCollectionAgent(event_bus=bus, db_path=test_db, config={})
        status = agent.get_status()
        assert status["name"] == "data"
        assert "total_bars" in status
        assert status["total_bars"] == 0

    def test_get_observations(self, bus, test_db):
        agent = DataCollectionAgent(event_bus=bus, db_path=test_db, config={})
        obs = agent.get_observations()
        assert obs["agent"] == "data"
        assert "bar_completeness" in obs

    def test_on_cycle_complete_emits(self, bus, test_db):
        agent = DataCollectionAgent(event_bus=bus, db_path=test_db, config={})
        emitted = []
        bus.subscribe("data.*", lambda ch, d: emitted.append(ch))
        agent.on_cycle_complete({"bar_count": 100})
        assert "data.bars_updated" in emitted


class TestSignalAgent:
    def test_get_status_with_scorecard(self, bus, test_db):
        scorecard = {"BTC-USD": {"rsi": {"correct": 7, "total": 10}}}
        agent = SignalAgent(
            event_bus=bus, db_path=test_db, config={},
            signal_scorecard=scorecard,
        )
        status = agent.get_status()
        assert status["total_predictions"] == 10
        assert status["accuracy"] == 0.7

    def test_get_observations(self, bus, test_db):
        agent = SignalAgent(event_bus=bus, db_path=test_db, config={})
        obs = agent.get_observations()
        assert "signal_pnl" in obs

    def test_on_cycle_emits_trade_signal(self, bus, test_db):
        agent = SignalAgent(event_bus=bus, db_path=test_db, config={})
        emitted = []
        bus.subscribe("signal.*", lambda ch, d: emitted.append(d))
        agent.on_cycle_complete({"action": "BUY", "confidence": 0.8, "product_id": "BTC-USD"})
        assert len(emitted) == 1
        assert emitted[0]["action"] == "BUY"


class TestRiskAgent:
    def test_get_status(self, bus, test_db):
        agent = RiskAgent(event_bus=bus, db_path=test_db, config={})
        status = agent.get_status()
        assert status["name"] == "risk"

    def test_circuit_breaker_event(self, bus, test_db):
        agent = RiskAgent(event_bus=bus, db_path=test_db, config={})
        emitted = []
        bus.subscribe("risk.*", lambda ch, d: emitted.append(ch))
        agent.on_cycle_complete({"circuit_breaker_active": True, "circuit_breaker_reason": "drawdown"})
        assert "risk.circuit_breaker" in emitted


class TestExecutionAgent:
    def test_get_status(self, bus, test_db):
        agent = ExecutionAgent(event_bus=bus, db_path=test_db, config={})
        status = agent.get_status()
        assert status["devil_tracker_entries"] == 0

    def test_get_observations(self, bus, test_db):
        agent = ExecutionAgent(event_bus=bus, db_path=test_db, config={})
        obs = agent.get_observations()
        assert "execution_quality" in obs


class TestPortfolioAgent:
    def test_get_status(self, bus, test_db):
        agent = PortfolioAgent(event_bus=bus, db_path=test_db, config={})
        status = agent.get_status()
        assert status["open_positions"] == 0

    def test_get_observations(self, bus, test_db):
        agent = PortfolioAgent(event_bus=bus, db_path=test_db, config={})
        obs = agent.get_observations()
        assert "trades" in obs


class TestMonitoringAgent:
    def test_get_status(self, bus, test_db):
        agent = MonitoringAgent(event_bus=bus, db_path=test_db, config={})
        status = agent.get_status()
        assert status["name"] == "monitoring"

    def test_get_observations(self, bus, test_db):
        agent = MonitoringAgent(event_bus=bus, db_path=test_db, config={})
        obs = agent.get_observations()
        assert "daily_performance" in obs


class TestMetaAgent:
    def test_get_status_with_weights(self, bus, test_db):
        weights = {"rsi": 0.1, "macd": 0.15, "bollinger": 0.08}
        agent = MetaAgent(
            event_bus=bus, db_path=test_db, config={},
            signal_weights=weights,
        )
        status = agent.get_status()
        assert status["signal_weight_count"] == 3
        assert "top_weights" in status

    def test_get_observations(self, bus, test_db):
        agent = MetaAgent(event_bus=bus, db_path=test_db, config={})
        obs = agent.get_observations()
        assert obs["agent"] == "meta"
