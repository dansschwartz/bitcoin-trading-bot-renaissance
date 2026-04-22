"""Tests for SimBayesianUncertainty."""
import unittest
import numpy as np
from sim_bayesian_uncertainty import SimBayesianUncertainty
from sim_model_gbm import GBMSimulator


class TestSimBayesian(unittest.TestCase):

    def _make_uncertainty(self, **overrides):
        cfg = {"n_bootstrap": 20, "block_size": 10}
        cfg.update(overrides)
        return SimBayesianUncertainty(cfg)

    def _make_data(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0003, 0.02, n)
        prices = 100.0 * np.exp(np.cumsum(returns))
        return returns, prices

    def test_block_bootstrap_preserves_length(self):
        bu = self._make_uncertainty()
        data = np.arange(100, dtype=float)
        samples = bu.block_bootstrap(data, n_bootstrap=10, block_size=10)
        self.assertEqual(len(samples), 10)
        for s in samples:
            self.assertEqual(len(s), 100)

    def test_block_bootstrap_different_samples(self):
        bu = self._make_uncertainty()
        data = np.random.default_rng(42).normal(0, 1, 200)
        samples = bu.block_bootstrap(data, n_bootstrap=5, seed=42)
        # Not all identical
        self.assertFalse(np.allclose(samples[0], samples[1]))

    def test_parameter_distributions_shape(self):
        bu = self._make_uncertainty(n_bootstrap=15)
        model = GBMSimulator({})
        returns, prices = self._make_data()
        dists = bu.estimate_parameter_distributions(model, returns, prices)
        self.assertIn("mu", dists)
        self.assertIn("sigma", dists)
        for name, pd in dists.items():
            self.assertEqual(pd.param_name, name)
            self.assertGreater(len(pd.samples), 0)
            self.assertLessEqual(pd.ci_lower, pd.ci_upper)

    def test_simulation_fan_shape(self):
        bu = self._make_uncertainty(n_bootstrap=5)
        model = GBMSimulator({})
        returns, prices = self._make_data()
        model.calibrate(returns, prices)
        fan = bu.simulation_fan(
            model, S0=prices[-1], n_steps=50,
            returns=returns, prices=prices,
            n_sims_per_param=3, n_bootstrap=5, seed=42,
        )
        # Should have 5 bootstrap * 3 sims = 15 paths
        self.assertEqual(fan.shape[0], 15)
        self.assertEqual(fan.shape[1], 51)

    def test_model_restored_after_estimation(self):
        bu = self._make_uncertainty(n_bootstrap=5)
        model = GBMSimulator({})
        returns, prices = self._make_data()
        model.calibrate(returns, prices)
        original_mu = model.parameters["mu"]
        bu.estimate_parameter_distributions(model, returns, prices)
        self.assertAlmostEqual(model.parameters["mu"], original_mu, places=6)


if __name__ == "__main__":
    unittest.main()
