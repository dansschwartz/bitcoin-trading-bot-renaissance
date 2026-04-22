"""Unit tests for state recovery (DB persistence and restore)."""

import asyncio
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone

from database_manager import DatabaseManager


def run(coro):
    """Helper to run async coroutines in sync tests (Python 3.12+ safe)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestPositionPersistence(unittest.TestCase):
    """Test saving and restoring positions via DatabaseManager."""

    def setUp(self):
        self.tmp = tempfile.mktemp(suffix=".db")
        self.db = DatabaseManager({"path": self.tmp})
        run(self.db.init_database())

    def tearDown(self):
        try:
            os.unlink(self.tmp)
        except OSError:
            pass

    def test_save_and_get_open_position(self):
        pos = {
            "position_id": "BTC-USD_LONG_123",
            "product_id": "BTC-USD",
            "side": "LONG",
            "size": 0.01,
            "entry_price": 50000.0,
            "stop_loss_price": 49000.0,
            "take_profit_price": 52000.0,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN",
        }
        run(self.db.save_position(pos))

        positions = run(self.db.get_open_positions())
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["position_id"], "BTC-USD_LONG_123")
        self.assertEqual(positions[0]["side"], "LONG")
        self.assertAlmostEqual(positions[0]["entry_price"], 50000.0)

    def test_closed_positions_not_restored(self):
        pos = {
            "position_id": "BTC-USD_LONG_456",
            "product_id": "BTC-USD",
            "side": "LONG",
            "size": 0.01,
            "entry_price": 50000.0,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN",
        }
        run(self.db.save_position(pos))
        run(self.db.close_position_record("BTC-USD_LONG_456"))

        positions = run(self.db.get_open_positions())
        self.assertEqual(len(positions), 0)

    def test_upsert_overwrites(self):
        pos = {
            "position_id": "BTC-USD_LONG_789",
            "product_id": "BTC-USD",
            "side": "LONG",
            "size": 0.01,
            "entry_price": 50000.0,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN",
        }
        run(self.db.save_position(pos))

        # Update size
        pos["size"] = 0.02
        run(self.db.save_position(pos))

        positions = run(self.db.get_open_positions())
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["size"], 0.02)


class TestDailyPnlRecovery(unittest.TestCase):
    """Test recovering daily PnL from trade records."""

    def setUp(self):
        self.tmp = tempfile.mktemp(suffix=".db")
        self.db = DatabaseManager({"path": self.tmp})
        run(self.db.init_database())

    def tearDown(self):
        try:
            os.unlink(self.tmp)
        except OSError:
            pass

    def test_daily_pnl_from_trades(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ts = datetime.now(timezone.utc).isoformat()

        # Simulate a buy then a sell
        run(self.db.store_trade({
            "timestamp": ts, "product_id": "BTC-USD", "side": "BUY",
            "size": 0.01, "price": 50000.0, "status": "EXECUTED",
        }))
        run(self.db.store_trade({
            "timestamp": ts, "product_id": "BTC-USD", "side": "SELL",
            "size": 0.01, "price": 51000.0, "status": "EXECUTED",
        }))

        pnl = run(self.db.get_daily_pnl(today))
        # PnL = SELL revenue - BUY cost = 0.01*51000 - 0.01*50000 = 10
        self.assertAlmostEqual(pnl, 10.0, places=2)

    def test_no_trades_returns_zero(self):
        pnl = run(self.db.get_daily_pnl("2099-01-01"))
        self.assertEqual(pnl, 0.0)


class TestOpenPositionsTable(unittest.TestCase):
    """Test the open_positions table is created on init."""

    def test_table_exists_after_init(self):
        tmp = tempfile.mktemp(suffix=".db")
        db = DatabaseManager({"path": tmp})
        run(db.init_database())

        conn = sqlite3.connect(tmp)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='open_positions'")
        result = cursor.fetchone()
        conn.close()
        os.unlink(tmp)

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "open_positions")


if __name__ == "__main__":
    unittest.main()
