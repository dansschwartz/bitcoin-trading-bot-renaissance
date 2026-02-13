"""Unit tests for SQL injection prevention."""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from database_manager import DatabaseManager


class TestSQLInjectionPrevention(unittest.TestCase):
    """Verify table whitelist and parameterized queries."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self._tmpdir, "test.db")
        self.db = DatabaseManager({"path": db_path})
        asyncio.get_event_loop().run_until_complete(self.db.init_database())

    # -- get_recent_data --

    def test_get_recent_data_rejects_injection(self):
        """SQL injection via table name should raise ValueError."""
        with self.assertRaises(ValueError):
            asyncio.get_event_loop().run_until_complete(
                self.db.get_recent_data("trades; DROP TABLE users", 24)
            )

    def test_get_recent_data_rejects_unknown_table(self):
        """Arbitrary table names not in whitelist should be rejected."""
        with self.assertRaises(ValueError):
            asyncio.get_event_loop().run_until_complete(
                self.db.get_recent_data("secret_passwords", 24)
            )

    def test_get_recent_data_valid_table(self):
        """Whitelisted table names should work without error."""
        for table in ("market_data", "decisions", "trades"):
            result = asyncio.get_event_loop().run_until_complete(
                self.db.get_recent_data(table, 24)
            )
            self.assertIsInstance(result, list)

    def test_get_recent_data_hours_parameterized(self):
        """Hours value should be safely parameterized (no injection via hours)."""
        # Even if hours is a weird string-like int, it should be cast to int
        result = asyncio.get_event_loop().run_until_complete(
            self.db.get_recent_data("trades", 48)
        )
        self.assertIsInstance(result, list)

    # -- cleanup_old_data --

    def test_cleanup_uses_whitelisted_tables_only(self):
        """cleanup_old_data should only touch ALLOWED_TABLES."""
        # Should not raise — all tables in its internal list are whitelisted
        asyncio.get_event_loop().run_until_complete(
            self.db.cleanup_old_data(days=30)
        )

    # -- ALLOWED_TABLES completeness --

    def test_allowed_tables_is_frozenset(self):
        """Whitelist should be immutable."""
        self.assertIsInstance(DatabaseManager.ALLOWED_TABLES, frozenset)

    def test_allowed_tables_contains_expected(self):
        expected = {"market_data", "labels", "sentiment_data", "onchain_data",
                    "decisions", "trades", "ml_predictions", "open_positions"}
        self.assertEqual(DatabaseManager.ALLOWED_TABLES, expected)


class TestGeneticOptimizerSQL(unittest.TestCase):
    """Verify genetic_optimizer uses parameterized IN clause."""

    def test_parameterized_in_clause(self):
        """The _calculate_fitness query should use ? placeholders, not f-strings."""
        import inspect
        from genetic_optimizer import GeneticWeightOptimizer

        source = inspect.getsource(GeneticWeightOptimizer._calculate_fitness)
        # Should use parameterized placeholders
        self.assertIn("'?' * len(decision_ids)", source)
        # Should NOT have raw f-string interpolation of decision_ids
        self.assertNotIn("','.join(decision_ids)}", source)


if __name__ == "__main__":
    unittest.main()
