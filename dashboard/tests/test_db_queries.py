"""Tests for dashboard database query functions."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest

from dashboard import db_queries


@pytest.fixture
def test_db():
    """Create a temporary SQLite database with test data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    c = conn.cursor()

    # Create tables matching the bot's schema
    c.execute("""CREATE TABLE decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
        product_id TEXT NOT NULL, action TEXT NOT NULL, confidence REAL NOT NULL,
        position_size REAL NOT NULL, weighted_signal REAL NOT NULL,
        reasoning TEXT NOT NULL, feature_vector TEXT, vae_loss REAL, hmm_regime TEXT)""")

    c.execute("""CREATE TABLE trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
        product_id TEXT NOT NULL, side TEXT NOT NULL, size REAL NOT NULL,
        price REAL NOT NULL, status TEXT NOT NULL, algo_used TEXT,
        slippage REAL, execution_time REAL)""")

    c.execute("""CREATE TABLE open_positions (
        position_id TEXT PRIMARY KEY, product_id TEXT NOT NULL,
        side TEXT NOT NULL, size REAL NOT NULL, entry_price REAL NOT NULL,
        stop_loss_price REAL, take_profit_price REAL,
        opened_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'OPEN')""")

    c.execute("""CREATE TABLE market_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT, price REAL NOT NULL,
        volume REAL NOT NULL, bid REAL NOT NULL, ask REAL NOT NULL,
        spread REAL NOT NULL, timestamp TEXT NOT NULL, source TEXT NOT NULL,
        product_id TEXT DEFAULT 'BTC-USD')""")

    c.execute("""CREATE TABLE ml_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
        product_id TEXT NOT NULL, model_name TEXT NOT NULL, prediction REAL NOT NULL)""")

    # Insert test data
    now = datetime.now(timezone.utc).isoformat()

    c.execute(
        "INSERT INTO decisions VALUES (NULL,?,?,?,?,?,?,?,NULL,?,?)",
        (now, "BTC-USD", "BUY", 0.75, 0.1, 0.23, json.dumps({"reason": "test"}), 0.05, "bullish"),
    )
    c.execute(
        "INSERT INTO decisions VALUES (NULL,?,?,?,?,?,?,?,NULL,?,?)",
        (now, "BTC-USD", "HOLD", 0.45, 0.0, 0.01, json.dumps({"reason": "low_signal"}), 0.12, "neutral"),
    )
    c.execute(
        "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?)",
        (now, "BTC-USD", "BUY", 0.001, 50000.0, "FILLED", "TWAP", 0.001, 1.5),
    )
    c.execute(
        "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?)",
        (now, "BTC-USD", "SELL", 0.001, 51000.0, "FILLED", "SNIPER", 0.0005, 0.8),
    )
    c.execute(
        "INSERT INTO open_positions VALUES (?,?,?,?,?,?,?,?,?)",
        ("pos_1", "BTC-USD", "BUY", 0.002, 49500.0, 48000.0, 52000.0, now, "OPEN"),
    )
    c.execute(
        "INSERT INTO market_data VALUES (NULL,?,?,?,?,?,?,?,?)",
        (50500.0, 100.0, 50490.0, 50510.0, 20.0, now, "Coinbase", "BTC-USD"),
    )
    c.execute(
        "INSERT INTO ml_predictions VALUES (NULL,?,?,?,?)",
        (now, "BTC-USD", "CNN-LSTM", 0.65),
    )
    c.execute(
        "INSERT INTO ml_predictions VALUES (NULL,?,?,?,?)",
        (now, "BTC-USD", "N-BEATS", 0.72),
    )

    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


def test_get_cycle_count(test_db):
    assert db_queries.get_cycle_count(test_db) == 2


def test_get_trade_count(test_db):
    assert db_queries.get_trade_count(test_db) == 2


def test_get_recent_decisions(test_db):
    decisions = db_queries.get_recent_decisions(test_db, limit=10)
    assert len(decisions) == 2
    assert decisions[0]["action"] in ("BUY", "HOLD", "SELL")
    assert isinstance(decisions[0]["reasoning"], dict)


def test_get_decision_by_id(test_db):
    d = db_queries.get_decision_by_id(test_db, 1)
    assert d is not None
    assert d["action"] == "BUY"
    assert d["confidence"] == 0.75


def test_get_open_positions(test_db):
    positions = db_queries.get_open_positions(test_db)
    assert len(positions) == 1
    assert positions[0]["position_id"] == "pos_1"


def test_get_closed_trades(test_db):
    trades = db_queries.get_closed_trades(test_db, limit=10)
    assert len(trades) == 2


def test_get_equity_curve(test_db):
    curve = db_queries.get_equity_curve(test_db, hours=24)
    assert len(curve) == 2
    assert "cumulative_pnl" in curve[-1]


def test_get_pnl_summary(test_db):
    summary = db_queries.get_pnl_summary(test_db, hours=24)
    assert "total_trades" in summary
    assert "realized_pnl" in summary
    assert summary["total_trades"] == 2


def test_get_performance_by_regime(test_db):
    data = db_queries.get_performance_by_regime(test_db)
    assert len(data) >= 1
    assert "regime" in data[0]


def test_get_ml_predictions_history(test_db):
    preds = db_queries.get_ml_predictions_history(test_db, hours=24)
    assert len(preds) == 2


def test_get_regime_history(test_db):
    history = db_queries.get_regime_history(test_db, limit=10)
    assert len(history) == 2


def test_get_vae_history(test_db):
    history = db_queries.get_vae_history(test_db, limit=10)
    assert len(history) == 2


def test_get_risk_metrics(test_db):
    metrics = db_queries.get_risk_metrics(test_db)
    assert "max_drawdown" in metrics
    assert "cumulative_pnl" in metrics


def test_get_exposure(test_db):
    exp = db_queries.get_exposure(test_db)
    assert "net_exposure" in exp
    assert "gross_exposure" in exp
    assert exp["position_count"] == 1


def test_get_latest_price(test_db):
    lp = db_queries.get_latest_price(test_db, "BTC-USD")
    assert lp is not None
    assert lp["price"] == 50500.0


def test_get_calendar_pnl(test_db):
    cal = db_queries.get_calendar_pnl(test_db)
    assert len(cal) >= 1
    assert "daily_pnl" in cal[0]


def test_get_return_distribution(test_db):
    dist = db_queries.get_return_distribution(test_db)
    assert len(dist) == 2
