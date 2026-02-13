"""Tests for SimPortfolioSimulator."""
import unittest
import numpy as np
import pandas as pd
from sim_portfolio import SimPortfolioSimulator
from sim_model_gbm import GBMSimulator


class TestSimPortfolio(unittest.TestCase):

    def _make_sim(self):
        return SimPortfolioSimulator()

    def _make_aligned_returns(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        # Correlated returns
        base = rng.normal(0, 0.02, n)
        btc = base + rng.normal(0, 0.005, n)
        eth = 0.7 * base + rng.normal(0, 0.01, n)
        sol = 0.5 * base + rng.normal(0, 0.015, n)
        return pd.DataFrame({"BTC": btc, "ETH": eth, "SOL": sol}, index=dates)

    def test_correlation_matrix_psd(self):
        sim = self._make_sim()
        returns = self._make_aligned_returns()
        corr = sim.compute_correlation_matrix(returns)
        # Must be PSD: all eigenvalues >= 0
        eigenvalues = np.linalg.eigvalsh(corr)
        self.assertTrue(np.all(eigenvalues >= -1e-7))

    def test_correlation_matrix_shape(self):
        sim = self._make_sim()
        returns = self._make_aligned_returns()
        corr = sim.compute_correlation_matrix(returns)
        self.assertEqual(corr.shape, (3, 3))
        # Diagonal should be 1
        for i in range(3):
            self.assertAlmostEqual(corr[i, i], 1.0, places=5)

    def test_generate_correlated_paths_shape(self):
        sim = self._make_sim()
        returns = self._make_aligned_returns()
        corr = sim.compute_correlation_matrix(returns)

        models = {}
        for asset in ["BTC", "ETH", "SOL"]:
            m = GBMSimulator({})
            r = returns[asset].values
            p = 100 * np.exp(np.cumsum(r))
            m.calibrate(r, p)
            models[asset] = m

        paths = sim.generate_correlated_paths(
            models, {"BTC": 100, "ETH": 100, "SOL": 100},
            corr, n_steps=50, n_simulations=20, seed=42,
        )
        self.assertEqual(len(paths), 3)
        for asset, p in paths.items():
            self.assertEqual(p.shape, (20, 51))
            self.assertTrue(np.all(p > 0))

    def test_portfolio_equity(self):
        sim = self._make_sim()
        paths = {
            "A": np.ones((5, 10)) * 100,
            "B": np.ones((5, 10)) * 200,
        }
        eq = sim.portfolio_equity(paths, {"A": 0.5, "B": 0.5})
        self.assertEqual(len(eq), 10)
        # Equal weight of normalised paths should be ~1.0 everywhere
        self.assertAlmostEqual(eq[0], 1.0, places=3)

    def test_portfolio_equity_empty(self):
        sim = self._make_sim()
        eq = sim.portfolio_equity({}, {})
        self.assertEqual(len(eq), 0)


if __name__ == "__main__":
    unittest.main()
