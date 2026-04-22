"""Tests for SimStatistics."""
import unittest
import numpy as np
from sim_statistics import SimStatistics


class TestSimStatistics(unittest.TestCase):

    def _make_stats(self):
        return SimStatistics()

    def _normal_returns(self, n=1000, seed=42):
        return np.random.default_rng(seed).normal(0.0, 0.02, n)

    def test_distribution_stats_normal(self):
        stats = self._make_stats()
        returns = self._normal_returns()
        result = stats.distribution_stats(returns)
        self.assertAlmostEqual(result["mean"], 0.0, places=2)
        self.assertAlmostEqual(result["std"], 0.02, places=2)
        # Normal: skewness ≈ 0, kurtosis ≈ 0
        self.assertAlmostEqual(result["skewness"], 0.0, delta=0.3)
        self.assertAlmostEqual(result["kurtosis"], 0.0, delta=0.5)

    def test_volatility_properties(self):
        stats = self._make_stats()
        returns = self._normal_returns()
        result = stats.volatility_properties(returns)
        self.assertGreater(result["annualized_vol"], 0)
        ann_vol = 0.02 * np.sqrt(252)
        self.assertAlmostEqual(result["annualized_vol"], ann_vol, delta=0.05)

    def test_tail_properties_var(self):
        stats = self._make_stats()
        returns = self._normal_returns()
        result = stats.tail_properties(returns)
        # VaR should be negative (5th percentile of returns)
        self.assertLess(result["var_5pct"], 0)
        # CVaR should be <= VaR
        self.assertLessEqual(result["cvar_5pct"], result["var_5pct"])

    def test_tail_properties_max_drawdown(self):
        stats = self._make_stats()
        returns = self._normal_returns()
        result = stats.tail_properties(returns)
        self.assertLessEqual(result["max_drawdown"], 0)

    def test_autocorrelation_structure(self):
        stats = self._make_stats()
        returns = self._normal_returns()
        result = stats.autocorrelation_structure(returns)
        acf = result["returns_acf"]
        if acf:
            self.assertAlmostEqual(acf[0], 1.0, places=2)
            self.assertEqual(len(acf), 21)  # lag 0 to 20

    def test_full_analysis_keys(self):
        stats = self._make_stats()
        returns = self._normal_returns()
        result = stats.full_analysis(returns)
        self.assertIn("distribution", result)
        self.assertIn("volatility", result)
        self.assertIn("tail", result)
        self.assertIn("autocorrelation", result)

    def test_hill_estimator_positive(self):
        stats = self._make_stats()
        returns = self._normal_returns(n=2000)
        result = stats.tail_properties(returns)
        self.assertGreater(result["hill_tail_index"], 0)


if __name__ == "__main__":
    unittest.main()
