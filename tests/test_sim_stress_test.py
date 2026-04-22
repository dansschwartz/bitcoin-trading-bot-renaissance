"""Tests for SimStressTest."""
import unittest
import numpy as np
from sim_stress_test import SimStressTest


class TestSimStressTest(unittest.TestCase):

    def _make_tester(self, **overrides):
        cfg = {"flash_crash_pct": -0.30, "covid_decline_days": 30,
               "covid_total_decline": -0.50, "death_spiral_feedback": 0.02}
        cfg.update(overrides)
        return SimStressTest(cfg)

    def _make_paths(self, n_sims=10, n_steps=100, S0=100.0, seed=42):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0003, 0.02, (n_sims, n_steps))
        log_paths = np.cumsum(returns, axis=1)
        paths = S0 * np.exp(np.column_stack([np.zeros(n_sims), log_paths]))
        return paths

    def test_flash_crash_reduces_prices(self):
        tester = self._make_tester()
        paths = self._make_paths()
        crash_day = 50
        stressed = tester.inject_flash_crash(paths, crash_day=crash_day, crash_pct=-0.30)
        # Prices at crash_day and after should be lower
        self.assertTrue(np.all(stressed[:, crash_day] < paths[:, crash_day]))
        # Before crash day: unchanged
        self.assertTrue(np.allclose(stressed[:, :crash_day], paths[:, :crash_day]))

    def test_covid_decline_reduces_end_price(self):
        tester = self._make_tester()
        paths = self._make_paths()
        stressed = tester.inject_covid_decline(paths, start_day=30, duration=20, total_decline=-0.50)
        # End prices should be much lower
        self.assertTrue(np.mean(stressed[:, -1]) < np.mean(paths[:, -1]))

    def test_death_spiral_accelerates(self):
        tester = self._make_tester()
        paths = self._make_paths()
        stressed = tester.inject_death_spiral(paths, start_day=50, feedback=0.05, duration=10)
        # End prices should be significantly lower
        ratio = np.mean(stressed[:, -1]) / np.mean(paths[:, -1])
        self.assertLess(ratio, 0.8)

    def test_stress_correlation(self):
        tester = self._make_tester()
        paths_a = self._make_paths(seed=42)
        paths_b = self._make_paths(seed=123)
        multi = {"A": paths_a, "B": paths_b}
        stressed = tester.stress_correlation(multi, crisis_start=30, crisis_end=60)
        self.assertIn("A", stressed)
        self.assertIn("B", stressed)
        # Shapes preserved
        self.assertEqual(stressed["A"].shape, paths_a.shape)

    def test_liquidity_crisis_scales_costs(self):
        tester = self._make_tester()
        costs = {"maker_fee": 0.001, "taker_fee": 0.002, "base_slippage_bps": 5.0}
        scaled = tester.inject_liquidity_crisis(costs, multiplier=5.0)
        self.assertAlmostEqual(scaled["maker_fee"], 0.005)
        self.assertAlmostEqual(scaled["taker_fee"], 0.010)
        self.assertAlmostEqual(scaled["base_slippage_bps"], 25.0)

    def test_flash_crash_out_of_bounds(self):
        tester = self._make_tester()
        paths = self._make_paths()
        # Crash day out of bounds → no change
        stressed = tester.inject_flash_crash(paths, crash_day=999)
        self.assertTrue(np.allclose(stressed, paths))


if __name__ == "__main__":
    unittest.main()
