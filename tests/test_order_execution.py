"""Unit tests for order execution routing (paper vs real) and circuit breaker."""

import unittest
from coinbase_client import (
    EnhancedCoinbaseClient,
    CoinbaseCredentials,
    CircuitBreaker,
    CircuitBreakerError,
)


def _make_client(paper: bool = True) -> EnhancedCoinbaseClient:
    """Helper to create a client with dummy credentials."""
    creds = CoinbaseCredentials(api_key="test-key", api_secret="test-secret", sandbox=True)
    return EnhancedCoinbaseClient(credentials=creds, paper_trading=paper)


class TestPaperTradingRouting(unittest.TestCase):
    """Verify paper vs real routing logic."""

    def test_paper_trading_routes_to_simulator(self):
        client = _make_client(paper=True)
        self.assertTrue(client.paper_trading)
        self.assertIsNotNone(client.paper_trader)

        result = client.create_market_order("BTC-USD", "BUY", size=0.001)
        # Paper trader returns a simulated fill — no real API call made
        self.assertIn("order", result)

    def test_real_trading_flag(self):
        client = _make_client(paper=False)
        self.assertFalse(client.paper_trading)
        self.assertIsNone(client.paper_trader)


class TestCircuitBreaker(unittest.TestCase):
    """Tests for the circuit breaker pattern."""

    def test_starts_closed(self):
        cb = CircuitBreaker()
        self.assertEqual(cb.state, "CLOSED")

    def test_opens_after_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb._on_failure()
        self.assertEqual(cb.state, "OPEN")

    def test_open_breaker_raises(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb._on_failure()
        self.assertEqual(cb.state, "OPEN")
        # The call() method raises when breaker is open
        with self.assertRaises(CircuitBreakerError):
            cb.call(lambda: None)

    def test_success_resets_failures(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb._on_failure()
        cb._on_failure()
        self.assertEqual(cb.failure_count, 2)
        cb._on_success()
        self.assertEqual(cb.failure_count, 0)
        self.assertEqual(cb.state, "CLOSED")


if __name__ == "__main__":
    unittest.main()
