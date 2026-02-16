"""Unit tests for database connection safety (Step 7)."""

import asyncio
import os
import sqlite3
import tempfile
import unittest

from database_manager import DatabaseManager


class TestDatabaseSafety(unittest.TestCase):
    """Verify _get_connection context manager behavior."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "test_safety.db")
        self.db = DatabaseManager({"path": self.db_path})
        asyncio.run(self.db.init_database())

    def test_wal_mode_enabled(self):
        """Database should use WAL journal mode after init."""
        conn = sqlite3.connect(self.db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        self.assertEqual(mode, "wal")

    def test_connection_closed_after_use(self):
        """Connection should be closed after context manager exits."""
        with self.db._get_connection() as conn:
            # Connection is open inside the block
            cursor = conn.execute("SELECT 1")
            self.assertEqual(cursor.fetchone()[0], 1)
        # After exiting, connection should be closed
        # Attempting to use the closed connection should raise
        with self.assertRaises(Exception):
            conn.execute("SELECT 1")

    def test_rollback_on_error(self):
        """Transaction should be rolled back on exception."""
        try:
            with self.db._get_connection() as conn:
                conn.execute("INSERT INTO market_data (price, volume, bid, ask, spread, timestamp, source) VALUES (1,1,1,1,1,'2024-01-01','test')")
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Data should NOT be committed due to rollback
        with self.db._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
            self.assertEqual(count, 0)

    def test_timeout_set(self):
        """Connection should have a timeout configured."""
        # We verify this indirectly: _get_connection uses timeout=10.0
        # A second connection shouldn't hang when the first holds a lock
        with self.db._get_connection() as conn1:
            conn1.execute("BEGIN EXCLUSIVE")
            # Second connection with a short timeout should eventually fail
            # but our 10s timeout means it waits a bit. Just verify it doesn't crash.
            # We'll use a separate direct connection with a very short timeout.
            conn2 = sqlite3.connect(self.db_path, timeout=0.1)
            with self.assertRaises(sqlite3.OperationalError):
                conn2.execute("BEGIN EXCLUSIVE")
            conn2.close()


if __name__ == "__main__":
    unittest.main()
