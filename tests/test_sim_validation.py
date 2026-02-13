"""Tests for SimValidationSuite."""
import unittest
import numpy as np
from sim_validation import SimValidationSuite


class TestSimValidation(unittest.TestCase):

    def _make_suite(self):
        return SimValidationSuite()

    def _identical_data(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0003, 0.02, n)
        prices = 100.0 * np.exp(np.cumsum(returns))
        # Simulate paths that match empirical
        paths = np.vstack([prices] * 10)  # 10 identical paths
        return returns, paths

    def test_ks_test_identical(self):
        suite = self._make_suite()
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        result = suite.ks_test(data, data)
        self.assertEqual(result["ks_stat"], 0.0)
        self.assertEqual(result["ks_pvalue"], 1.0)

    def test_ks_test_different(self):
        suite = self._make_suite()
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(5, 1, 1000)
        result = suite.ks_test(a, b)
        self.assertGreater(result["ks_stat"], 0.5)

    def test_acf_comparison_identical(self):
        suite = self._make_suite()
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        rmse = suite.acf_comparison(data, data)
        self.assertAlmostEqual(rmse, 0.0, places=5)

    def test_scorecard_range(self):
        suite = self._make_suite()
        returns, paths = self._identical_data()
        score = suite.compute_scorecard("TestModel", "TEST", returns, paths)
        self.assertGreaterEqual(score.composite_score, 0)
        self.assertLessEqual(score.composite_score, 10)

    def test_scorecard_good_match(self):
        """Paths that exactly match empirical should score reasonably."""
        suite = self._make_suite()
        returns, paths = self._identical_data()
        score = suite.compute_scorecard("TestModel", "TEST", returns, paths)
        # Should score at least moderately (>2 given GARCH comparison noise)
        self.assertGreater(score.composite_score, 2.0)

    def test_scorecard_bad_match(self):
        """Very different simulated data should score poorly."""
        suite = self._make_suite()
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 500)
        # Paths with completely different distribution
        bad_paths = np.ones((10, 501)) * 100
        score = suite.compute_scorecard("BadModel", "TEST", returns, bad_paths)
        self.assertLess(score.composite_score, 5.0)

    def test_vol_clustering_identical(self):
        suite = self._make_suite()
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        rmse = suite.vol_clustering_comparison(data, data)
        self.assertAlmostEqual(rmse, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
