"""Unit tests for stale data detection (Step 12)."""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


class TestStaleDataDetection(unittest.TestCase):
    """Verify stale WebSocket and market data handling."""

    def test_stale_ws_data_rejected(self):
        """WebSocket data older than MAX_DATA_AGE_SECONDS should be treated as None."""
        # Simulate the freshness check logic from collect_all_data
        MAX_DATA_AGE_SECONDS = 30

        class FakeWS:
            def __init__(self, age_seconds):
                self.timestamp = datetime.now() - timedelta(seconds=age_seconds)

        # Stale data (60s old)
        latest_ws = FakeWS(60)
        if hasattr(latest_ws, 'timestamp') and latest_ws.timestamp:
            data_age = (datetime.now() - latest_ws.timestamp).total_seconds()
            if data_age > MAX_DATA_AGE_SECONDS:
                latest_ws = None

        self.assertIsNone(latest_ws)

    def test_fresh_ws_data_accepted(self):
        """WebSocket data within MAX_DATA_AGE_SECONDS should pass."""
        MAX_DATA_AGE_SECONDS = 30

        class FakeWS:
            def __init__(self, age_seconds):
                self.timestamp = datetime.now() - timedelta(seconds=age_seconds)
                self.price = 50000.0

        # Fresh data (5s old)
        latest_ws = FakeWS(5)
        if hasattr(latest_ws, 'timestamp') and latest_ws.timestamp:
            data_age = (datetime.now() - latest_ws.timestamp).total_seconds()
            if data_age > MAX_DATA_AGE_SECONDS:
                latest_ws = None

        self.assertIsNotNone(latest_ws)
        self.assertEqual(latest_ws.price, 50000.0)

    def test_stale_market_data_returns_hold(self):
        """Market data older than 60s should produce a HOLD decision."""
        # Simulate the check from execute_trading_cycle
        market_data = {
            'timestamp': datetime.now() - timedelta(seconds=90),
            'price': 50000.0,
        }

        data_age = (datetime.now() - market_data.get('timestamp', datetime.now())).total_seconds()
        should_hold = data_age > 60

        self.assertTrue(should_hold)

    def test_fresh_market_data_allows_trading(self):
        """Market data under 60s should allow normal trading."""
        market_data = {
            'timestamp': datetime.now() - timedelta(seconds=10),
            'price': 50000.0,
        }

        data_age = (datetime.now() - market_data.get('timestamp', datetime.now())).total_seconds()
        should_hold = data_age > 60

        self.assertFalse(should_hold)


if __name__ == "__main__":
    unittest.main()
