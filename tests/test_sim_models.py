"""Tests for all 5 simulation models."""
import unittest
import numpy as np

from sim_model_monte_carlo import MonteCarloSimulator
from sim_model_gbm import GBMSimulator
from sim_model_heston import HestonSimulator
from sim_model_hmm_regime import HMMRegimeSimulator
from sim_model_ngram import NGramSimulator


class SimModelTestBase:
    """Mixin providing common test data and assertions."""

    def _make_synthetic_data(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0003, 0.02, n)
        prices = 100.0 * np.exp(np.cumsum(returns))
        return returns, prices


class TestMonteCarloSimulator(unittest.TestCase, SimModelTestBase):

    def _make_model(self, **overrides):
        cfg = {"parametric": True}
        cfg.update(overrides)
        return MonteCarloSimulator(cfg)

    def test_calibrate_sets_params(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        params = model.calibrate(returns, prices)
        self.assertIn("mu", params)
        self.assertIn("sigma", params)
        self.assertTrue(model.is_calibrated)

    def test_simulate_shape(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=252, n_simulations=50, seed=42)
        self.assertEqual(paths.shape, (50, 253))

    def test_simulate_positive_prices(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=252, n_simulations=50, seed=42)
        self.assertTrue(np.all(paths > 0))

    def test_bootstrap_mode(self):
        model = self._make_model(parametric=False)
        returns, prices = self._make_synthetic_data()
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=100, n_simulations=20, seed=42)
        self.assertEqual(paths.shape, (20, 101))
        self.assertTrue(np.all(paths > 0))

    def test_not_calibrated_raises(self):
        model = self._make_model()
        with self.assertRaises(RuntimeError):
            model.simulate(S0=100, n_steps=10, n_simulations=5)


class TestGBMSimulator(unittest.TestCase, SimModelTestBase):

    def _make_model(self, **overrides):
        return GBMSimulator(overrides)

    def test_calibrate_annualised(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        params = model.calibrate(returns, prices)
        self.assertAlmostEqual(params["mu"], params["mu_daily"] * 252, places=5)

    def test_simulate_shape(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=252, n_simulations=100, seed=42)
        self.assertEqual(paths.shape, (100, 253))

    def test_all_positive(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=252, n_simulations=100, seed=42)
        self.assertTrue(np.all(paths > 0))

    def test_starts_at_S0(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        model.calibrate(returns, prices)
        paths = model.simulate(S0=42.0, n_steps=100, n_simulations=10, seed=42)
        self.assertTrue(np.allclose(paths[:, 0], 42.0))


class TestHestonSimulator(unittest.TestCase, SimModelTestBase):

    def _make_model(self, **overrides):
        cfg = {"kappa": 2.0, "xi": 0.5, "rho": -0.7}
        cfg.update(overrides)
        return HestonSimulator(cfg)

    def test_calibrate_feller_check(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        params = model.calibrate(returns, prices)
        self.assertIn("feller_satisfied", params)

    def test_simulate_shape(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=252, n_simulations=50, seed=42)
        self.assertEqual(paths.shape, (50, 253))

    def test_all_positive(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=252, n_simulations=50, seed=42)
        self.assertTrue(np.all(paths > 0))

    def test_starts_at_S0(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        model.calibrate(returns, prices)
        paths = model.simulate(S0=200.0, n_steps=50, n_simulations=10, seed=42)
        self.assertTrue(np.allclose(paths[:, 0], 200.0))


class TestHMMRegimeSimulator(unittest.TestCase, SimModelTestBase):

    def _make_model(self, **overrides):
        cfg = {"n_regimes": 3, "n_iter": 50}
        cfg.update(overrides)
        return HMMRegimeSimulator(cfg)

    def test_calibrate_stores_regimes(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data(n=300)
        params = model.calibrate(returns, prices)
        self.assertIn("regime_params", params)
        self.assertIn("transition_matrix", params)
        self.assertTrue(model.is_calibrated)

    def test_simulate_shape(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data(n=300)
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=252, n_simulations=30, seed=42)
        self.assertEqual(paths.shape, (30, 253))

    def test_all_positive(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data(n=300)
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=252, n_simulations=30, seed=42)
        self.assertTrue(np.all(paths > 0))

    def test_fallback_with_short_data(self):
        model = self._make_model()
        returns = np.array([0.01, -0.01, 0.005])
        prices = np.array([100, 101, 100, 100.5])
        params = model.calibrate(returns, prices)
        self.assertTrue(model.is_calibrated)
        # Fallback to single regime
        paths = model.simulate(S0=100, n_steps=50, n_simulations=5, seed=42)
        self.assertEqual(paths.shape, (5, 51))

    def test_regime_labels(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data(n=300)
        model.calibrate(returns, prices)
        labels = model.get_regime_labels()
        self.assertIsInstance(labels, dict)


class TestNGramSimulator(unittest.TestCase, SimModelTestBase):

    def _make_model(self, **overrides):
        cfg = {"n": 3, "n_bins": 20}
        cfg.update(overrides)
        return NGramSimulator(cfg)

    def test_calibrate_stores_contexts(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        params = model.calibrate(returns, prices)
        self.assertIn("n_unique_contexts", params)
        self.assertGreater(params["n_unique_contexts"], 0)

    def test_simulate_shape(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=252, n_simulations=30, seed=42)
        self.assertEqual(paths.shape, (30, 253))

    def test_all_positive(self):
        model = self._make_model()
        returns, prices = self._make_synthetic_data()
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=252, n_simulations=30, seed=42)
        self.assertTrue(np.all(paths > 0))

    def test_fallback_with_short_data(self):
        model = self._make_model()
        returns = np.array([0.01, -0.01])
        prices = np.array([100, 101, 100])
        model.calibrate(returns, prices)
        paths = model.simulate(S0=100, n_steps=50, n_simulations=5, seed=42)
        self.assertEqual(paths.shape, (5, 51))

    def test_different_n_values(self):
        for n in (2, 3, 5):
            model = self._make_model(n=n)
            returns, prices = self._make_synthetic_data()
            model.calibrate(returns, prices)
            paths = model.simulate(S0=100, n_steps=50, n_simulations=5, seed=42)
            self.assertEqual(paths.shape, (5, 51))


if __name__ == "__main__":
    unittest.main()
