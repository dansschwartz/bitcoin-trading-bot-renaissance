"""
Tests for database_manager.py — CRUD operations with in-memory SQLite.

Tests init_database, store/read operations for decisions, market_data,
trades, positions, and balance snapshots.
"""

import asyncio
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from database_manager import DatabaseManager, MarketData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """Provide a DatabaseManager backed by a temporary file."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_crud.db")
    manager = DatabaseManager({"path": db_path})
    asyncio.run(manager.init_database())
    yield manager
    # Cleanup
    try:
        os.unlink(db_path)
        os.rmdir(tmpdir)
    except OSError:
        pass


@pytest.fixture
def db_path(db) -> str:
    return db.db_path


# ---------------------------------------------------------------------------
# Tests: Init
# ---------------------------------------------------------------------------

class TestInitDatabase:
    def test_creates_tables(self, db, db_path):
        """init_database should create all required tables."""
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()

        expected_tables = [
            "market_data", "decisions", "trades", "open_positions",
            "ml_predictions", "sentiment_data",
        ]
        for t in expected_tables:
            assert t in tables, f"Missing table: {t}"

    def test_wal_mode_enabled(self, db_path):
        """Database should use WAL journal mode."""
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_idempotent_init(self, db):
        """Calling init_database twice should not raise."""
        asyncio.run(db.init_database())
        # Should complete without error


# ---------------------------------------------------------------------------
# Tests: Market Data CRUD
# ---------------------------------------------------------------------------

class TestMarketDataCRUD:
    def test_store_market_data(self, db, db_path):
        md = MarketData(
            price=50000.0, volume=1500.0, bid=49990.0, ask=50010.0,
            spread=20.0, timestamp=datetime.now(timezone.utc),
            source="binance", product_id="BTC-USD",
        )
        asyncio.run(db.store_market_data(md))

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        conn.close()
        assert count == 1

    def test_store_multiple_market_data(self, db, db_path):
        for i in range(5):
            md = MarketData(
                price=50000.0 + i, volume=1500.0, bid=49990.0, ask=50010.0,
                spread=20.0, timestamp=datetime.now(timezone.utc),
                source="binance", product_id="BTC-USD",
            )
            asyncio.run(db.store_market_data(md))

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        conn.close()
        assert count == 5


# ---------------------------------------------------------------------------
# Tests: Decision CRUD
# ---------------------------------------------------------------------------

class TestDecisionCRUD:
    def test_store_decision(self, db, db_path):
        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "product_id": "BTC-USD",
            "action": "BUY",
            "confidence": 0.85,
            "position_size": 0.05,
            "weighted_signal": 0.6,
            "reasoning": '{"test": true}',
            "hmm_regime": "trending",
        }
        asyncio.run(db.store_decision(decision))

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM decisions").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_decision_fields_persisted(self, db, db_path):
        decision = {
            "timestamp": "2024-01-01T00:00:00",
            "product_id": "ETH-USD",
            "action": "SELL",
            "confidence": 0.75,
            "position_size": 0.03,
            "weighted_signal": -0.5,
            "reasoning": '{"signal": "bearish"}',
            "hmm_regime": "mean_reverting",
            "vae_loss": 0.15,
        }
        asyncio.run(db.store_decision(decision))

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        assert row["product_id"] == "ETH-USD"
        assert row["action"] == "SELL"
        assert row["hmm_regime"] == "mean_reverting"


# ---------------------------------------------------------------------------
# Tests: Trade CRUD
# ---------------------------------------------------------------------------

class TestTradeCRUD:
    def test_store_trade(self, db, db_path):
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "product_id": "BTC-USD",
            "side": "BUY",
            "size": 0.001,
            "price": 50000.0,
            "status": "FILLED",
        }
        asyncio.run(db.store_trade(trade))

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# Tests: Position CRUD
# ---------------------------------------------------------------------------

class TestPositionCRUD:
    def test_save_position(self, db, db_path):
        position = {
            "position_id": "pos_001",
            "product_id": "BTC-USD",
            "side": "BUY",
            "size": 0.001,
            "entry_price": 50000.0,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN",
        }
        asyncio.run(db.save_position(position))

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM open_positions WHERE position_id = ?", ("pos_001",)
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["product_id"] == "BTC-USD"
        assert row["status"] == "OPEN"

    def test_get_open_positions(self, db):
        position = {
            "position_id": "pos_002",
            "product_id": "ETH-USD",
            "side": "SELL",
            "size": 0.01,
            "entry_price": 3000.0,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN",
        }
        asyncio.run(db.save_position(position))
        positions = asyncio.run(db.get_open_positions())
        assert len(positions) >= 1
        # Should find our position
        found = any(p.get("position_id") == "pos_002" or
                    p.get("product_id") == "ETH-USD" for p in positions)
        assert found

    def test_close_position_record(self, db, db_path):
        # First open a position
        position = {
            "position_id": "pos_003",
            "product_id": "SOL-USD",
            "side": "BUY",
            "size": 1.0,
            "entry_price": 100.0,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN",
        }
        asyncio.run(db.save_position(position))

        # Close it
        asyncio.run(db.close_position_record(
            position_id="pos_003",
            close_price=110.0,
            realized_pnl=10.0,
            exit_reason="take_profit",
        ))

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM open_positions WHERE position_id = ?", ("pos_003",)
        ).fetchone()
        conn.close()

        assert row["status"] == "CLOSED"
        assert row["close_price"] == 110.0
        assert row["realized_pnl"] == 10.0
        assert row["exit_reason"] == "take_profit"


# ---------------------------------------------------------------------------
# Tests: Balance Snapshots
# ---------------------------------------------------------------------------

class TestBalanceSnapshots:
    def test_store_balance_snapshot(self, db, db_path):
        asyncio.run(db.store_balance_snapshot(
            total_equity=10500.0,
            unrealized_pnl=200.0,
            open_position_count=3,
            cash_balance=10300.0,
            drawdown_pct=0.02,
            high_watermark=10700.0,
            daily_pnl=150.0,
            source="periodic",
        ))

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM portfolio_snapshots").fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# Tests: Pipeline Heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_record_heartbeat(self, db, db_path):
        asyncio.run(db.record_heartbeat(
            component="bar_aggregator",
            items_processed=50,
            details={"fetch_secs": 2.1, "pairs_requested": 70},
        ))

        conn = sqlite3.connect(db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM pipeline_heartbeat"
        ).fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# Tests: Connection Safety
# ---------------------------------------------------------------------------

class TestConnectionSafety:
    def test_context_manager_closes_connection(self, db):
        """Connection should be closed after context manager exits."""
        with db._get_connection() as conn:
            conn.execute("SELECT 1")
        # Connection should be closed now
        with pytest.raises(Exception):
            conn.execute("SELECT 1")

    def test_rollback_on_exception(self, db, db_path):
        """Data should not be committed if an exception occurs."""
        try:
            with db._get_connection() as conn:
                conn.execute(
                    "INSERT INTO market_data (price, volume, bid, ask, spread, "
                    "timestamp, source) VALUES (1,1,1,1,1,'2024-01-01','test')"
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass

        conn2 = sqlite3.connect(db_path)
        count = conn2.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        conn2.close()
        assert count == 0

    def test_concurrent_reads(self, db, db_path):
        """WAL mode should allow concurrent readers."""
        conn1 = sqlite3.connect(db_path, timeout=5.0)
        conn1.execute("PRAGMA journal_mode=WAL")
        conn2 = sqlite3.connect(db_path, timeout=5.0)
        conn2.execute("PRAGMA journal_mode=WAL")

        # Both should be able to read simultaneously
        r1 = conn1.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        r2 = conn2.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

        conn1.close()
        conn2.close()
        assert r1 == r2
