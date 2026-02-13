"""Tests for AdvancedMeanReversionEngine."""
import unittest
import numpy as np
from advanced_mean_reversion_engine import (
    AdvancedMeanReversionEngine, PairState, MeanReversionPortfolio,
)


class TestAdvancedMeanReversion(unittest.TestCase):

    def _make_engine(self, **overrides):
        cfg = {"min_history": 30, "max_half_life": 120, "min_half_life": 1}
        cfg.update(overrides)
        return AdvancedMeanReversionEngine(cfg)

    def _feed_cointegrated_prices(self, engine, n=200):
        """Feed two assets with a cointegrated relationship."""
        np.random.seed(42)
        # BTC random walk
        btc = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        # ETH = 0.5 * BTC + noise (cointegrated)
        eth = 0.5 * btc + np.random.randn(n) * 0.2
        for i in range(n):
            engine.update_price("BTC-USD", float(max(btc[i], 1.0)))
            engine.update_price("ETH-USD", float(max(eth[i], 1.0)))

    def test_update_price_stores_history(self):
        engine = self._make_engine()
        engine.update_price("BTC-USD", 100.0)
        engine.update_price("BTC-USD", 101.0)
        self.assertEqual(len(engine.history["BTC-USD"]), 2)

    def test_invalid_price_ignored(self):
        engine = self._make_engine()
        engine.update_price("BTC-USD", -5.0)
        engine.update_price("BTC-USD", 0.0)
        self.assertNotIn("BTC-USD", engine.history)

    def test_half_life_calculation(self):
        engine = self._make_engine()
        # Mean-reverting spread: AR(1) with negative coefficient
        np.random.seed(42)
        spread = np.zeros(100)
        for i in range(1, 100):
            spread[i] = -0.1 * spread[i - 1] + np.random.randn() * 0.1
        hl = engine._calculate_half_life(spread)
        self.assertGreater(hl, 0)
        self.assertLess(hl, 50)

    def test_half_life_non_mean_reverting(self):
        engine = self._make_engine()
        # Strong trending spread (positive theta → no mean reversion)
        np.random.seed(42)
        spread = np.arange(100, dtype=float)  # Pure uptrend
        hl = engine._calculate_half_life(spread)
        self.assertEqual(hl, float("inf"))

    def test_kalman_update_initializes(self):
        engine = self._make_engine()
        beta, P = engine._kalman_update("test_pair", 100.0, 50.0)
        self.assertAlmostEqual(beta, 2.0, places=1)
        self.assertEqual(P, 1.0)

    def test_kalman_update_converges(self):
        engine = self._make_engine()
        # Feed multiple observations with true beta=0.5
        for i in range(50):
            x = 100.0 + i * 0.1
            y = 0.5 * x + np.random.randn() * 0.01
            beta, _ = engine._kalman_update("test", y, x)
        self.assertAlmostEqual(beta, 0.5, places=1)

    def test_pair_signal_insufficient_data(self):
        engine = self._make_engine(min_history=50)
        engine.update_price("BTC-USD", 100.0)
        engine.update_price("ETH-USD", 50.0)
        ps = engine.calculate_pair_signal("BTC-USD", "ETH-USD")
        self.assertIsInstance(ps, PairState)
        self.assertEqual(ps.signal, 0.0)

    def test_generate_portfolio_signal(self):
        engine = self._make_engine(min_history=30, retest_interval=1)
        self._feed_cointegrated_prices(engine, n=200)
        portfolio = engine.generate_portfolio_signal(["BTC-USD", "ETH-USD"], cycle_count=0)
        self.assertIsInstance(portfolio, MeanReversionPortfolio)
        self.assertGreaterEqual(portfolio.n_active_pairs, 0)
        self.assertGreaterEqual(portfolio.composite_signal, -1.0)
        self.assertLessEqual(portfolio.composite_signal, 1.0)

    def test_discover_pairs_caching(self):
        engine = self._make_engine(min_history=30, retest_interval=100)
        self._feed_cointegrated_prices(engine, n=200)
        pairs1 = engine.discover_pairs(["BTC-USD", "ETH-USD"], cycle_count=0)
        # Second call within retest interval should use cache
        pairs2 = engine.discover_pairs(["BTC-USD", "ETH-USD"], cycle_count=10)
        self.assertEqual(pairs1, pairs2)


if __name__ == "__main__":
    unittest.main()
