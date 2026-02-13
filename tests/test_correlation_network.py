"""Tests for CorrelationNetworkEngine."""
import unittest
import numpy as np
from correlation_network_engine import CorrelationNetworkEngine


class TestCorrelationNetwork(unittest.TestCase):

    def _make_engine(self, **overrides):
        cfg = {"enabled": True, "min_history_length": 20, "update_interval_cycles": 1}
        cfg.update(overrides)
        return CorrelationNetworkEngine(cfg)

    def _feed_prices(self, engine, n=100, n_assets=5):
        """Feed correlated price series for multiple assets."""
        np.random.seed(42)
        base = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        for i in range(n_assets):
            pid = f"ASSET-{i}"
            noise = np.random.randn(n) * 0.3
            prices = base + noise * (i + 1)
            for p in prices:
                engine.update_price(pid, float(max(p, 1.0)))

    def test_update_price_single(self):
        engine = self._make_engine()
        engine.update_price("BTC-USD", 100.0)
        engine.update_price("BTC-USD", 101.0)
        self.assertEqual(len(engine._price_history["BTC-USD"]), 2)

    def test_update_prices_batch(self):
        engine = self._make_engine()
        engine.update_prices({"BTC-USD": 100.0, "ETH-USD": 50.0})
        self.assertIn("BTC-USD", engine._price_history)
        self.assertIn("ETH-USD", engine._price_history)

    def test_invalid_price_ignored(self):
        engine = self._make_engine()
        engine.update_price("BTC-USD", -10.0)
        engine.update_price("BTC-USD", 0.0)
        self.assertNotIn("BTC-USD", engine._price_history)

    def test_disabled_engine_no_ops(self):
        engine = CorrelationNetworkEngine({"enabled": False})
        engine.update_price("BTC-USD", 100.0)
        self.assertNotIn("BTC-USD", engine._price_history)
        self.assertEqual(engine.get_correlation_divergence_signal("BTC-USD"), 0.0)

    def test_correlation_matrix_requires_3_assets(self):
        engine = self._make_engine(min_history_length=5)
        np.random.seed(42)
        # Only 2 assets - should return None
        for i in range(10):
            engine.update_price("A", float(100 + i))
            engine.update_price("B", float(50 + i))
        matrix = engine.compute_correlation_matrix()
        self.assertIsNone(matrix)

    def test_correlation_matrix_computed(self):
        engine = self._make_engine(min_history_length=10)
        self._feed_prices(engine, n=50, n_assets=4)
        matrix = engine.compute_correlation_matrix()
        self.assertIsNotNone(matrix)
        self.assertEqual(matrix.shape[0], 4)
        self.assertEqual(matrix.shape[1], 4)
        # Diagonal should be ~1.0
        for i in range(4):
            self.assertAlmostEqual(matrix.iloc[i, i], 1.0, places=1)

    def test_identify_clusters(self):
        engine = self._make_engine(min_history_length=10)
        self._feed_prices(engine, n=50, n_assets=5)
        matrix = engine.compute_correlation_matrix()
        if matrix is not None:
            clusters = engine.identify_clusters(matrix)
            self.assertIsInstance(clusters, dict)
            # All assets should be assigned to a cluster
            all_members = [m for members in clusters.values() for m in members]
            self.assertEqual(len(all_members), len(matrix))

    def test_divergence_detection(self):
        engine = self._make_engine(min_history_length=10, divergence_zscore_threshold=1.0)
        self._feed_prices(engine, n=50, n_assets=5)
        matrix = engine.compute_correlation_matrix()
        if matrix is not None:
            clusters = engine.identify_clusters(matrix)
            divergences = engine.detect_divergences(matrix, clusters)
            self.assertIsInstance(divergences, dict)
            for v in divergences.values():
                self.assertGreaterEqual(v, -1.0)
                self.assertLessEqual(v, 1.0)

    def test_run_full_update(self):
        engine = self._make_engine(min_history_length=10, update_interval_cycles=1)
        self._feed_prices(engine, n=50, n_assets=4)
        engine.run_full_update(cycle_count=1)
        summary = engine.get_network_summary()
        self.assertGreaterEqual(summary["tracked_assets"], 4)

    def test_should_recompute(self):
        engine = self._make_engine(update_interval_cycles=5)
        self.assertTrue(engine.should_recompute(5))
        engine._last_compute_cycle = 5
        self.assertFalse(engine.should_recompute(7))
        self.assertTrue(engine.should_recompute(10))


if __name__ == "__main__":
    unittest.main()
