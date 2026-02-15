"""Tests for FastAPI router endpoints."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from dashboard.server import create_app


@pytest.fixture
def test_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute("""CREATE TABLE decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, product_id TEXT,
        action TEXT, confidence REAL, position_size REAL, weighted_signal REAL,
        reasoning TEXT, feature_vector TEXT, vae_loss REAL, hmm_regime TEXT)""")
    c.execute("""CREATE TABLE trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, product_id TEXT,
        side TEXT, size REAL, price REAL, status TEXT, algo_used TEXT,
        slippage REAL, execution_time REAL)""")
    c.execute("""CREATE TABLE open_positions (
        position_id TEXT PRIMARY KEY, product_id TEXT, side TEXT, size REAL,
        entry_price REAL, stop_loss_price REAL, take_profit_price REAL,
        opened_at TEXT, status TEXT DEFAULT 'OPEN')""")
    c.execute("""CREATE TABLE market_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT, price REAL, volume REAL,
        bid REAL, ask REAL, spread REAL, timestamp TEXT, source TEXT,
        product_id TEXT DEFAULT 'BTC-USD')""")
    c.execute("""CREATE TABLE ml_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, product_id TEXT,
        model_name TEXT, prediction REAL)""")

    now = datetime.now(timezone.utc).isoformat()
    c.execute("INSERT INTO decisions VALUES (NULL,?,?,?,?,?,?,?,NULL,?,?)",
              (now, "BTC-USD", "BUY", 0.8, 0.1, 0.3, json.dumps({"r": "t"}), 0.04, "bull"))
    c.execute("INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?)",
              (now, "BTC-USD", "BUY", 0.001, 50000, "FILLED", "TWAP", 0.001, 1.0))
    c.execute("INSERT INTO market_data VALUES (NULL,?,?,?,?,?,?,?,?)",
              (50000, 100, 49990, 50010, 20, now, "Coinbase", "BTC-USD"))

    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture
def test_config(test_db, tmp_path):
    cfg = {
        "trading": {"product_ids": ["BTC-USD"], "paper_trading": True},
        "database": {"path": test_db, "enabled": True},
        "signal_weights": {"order_flow": 0.18, "rsi": 0.05},
        "regime_overlay": {"enabled": True},
        "risk_gateway": {"enabled": False},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    return str(cfg_path)


@pytest.fixture
def client(test_config):
    app = create_app(config_path=test_config)
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_system_status(client):
    r = client.get("/api/system/status")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "OPERATIONAL"
    assert "cycle_count" in data


def test_system_config(client):
    r = client.get("/api/system/config")
    assert r.status_code == 200
    data = r.json()
    assert "flags" in data
    assert "signal_weights" in data


def test_recent_decisions(client):
    r = client.get("/api/decisions/recent?limit=10")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1


def test_decision_detail(client):
    r = client.get("/api/decisions/1")
    assert r.status_code == 200
    assert r.json()["action"] == "BUY"


def test_open_positions(client):
    r = client.get("/api/positions/open")
    assert r.status_code == 200


def test_closed_trades(client):
    r = client.get("/api/trades/closed?limit=10")
    assert r.status_code == 200


def test_equity_curve(client):
    r = client.get("/api/analytics/equity?range=1D")
    assert r.status_code == 200


def test_pnl_summary(client):
    r = client.get("/api/analytics/pnl?range=1D")
    assert r.status_code == 200
    assert "total_trades" in r.json()


def test_brain_ensemble(client):
    r = client.get("/api/brain/ensemble")
    assert r.status_code == 200


def test_brain_regime(client):
    r = client.get("/api/brain/regime")
    assert r.status_code == 200


def test_risk_metrics(client):
    r = client.get("/api/risk/metrics")
    assert r.status_code == 200


def test_risk_exposure(client):
    r = client.get("/api/risk/exposure")
    assert r.status_code == 200
