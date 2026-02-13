"""Tests for SimTransactionCostModel."""
import unittest
import numpy as np
from sim_transaction_costs import SimTransactionCostModel


class TestSimTransactionCosts(unittest.TestCase):

    def _make_model(self, **overrides):
        cfg = {"maker_fee": 0.001, "taker_fee": 0.002, "base_slippage_bps": 5.0,
               "vol_slippage_coeff": 0.1, "volume_slippage_coeff": 0.05,
               "half_spread_bps": 3.0, "funding_rate_daily": 0.0001}
        cfg.update(overrides)
        return SimTransactionCostModel(cfg)

    def test_cost_positive(self):
        model = self._make_model()
        cost = model.calculate_cost(trade_size_usd=10000, price=50000,
                                    volatility=0.02, daily_volume=1e9)
        self.assertGreater(cost.total, 0)

    def test_maker_cheaper_than_taker(self):
        model = self._make_model()
        maker = model.calculate_cost(10000, 50000, 0.02, 1e9, is_maker=True)
        taker = model.calculate_cost(10000, 50000, 0.02, 1e9, is_maker=False)
        self.assertLess(maker.total, taker.total)

    def test_regime_multiplier_volatile(self):
        model = self._make_model()
        normal = model.calculate_cost(10000, 50000, 0.02, 1e9, regime="normal")
        volatile = model.calculate_cost(10000, 50000, 0.02, 1e9, regime="volatile")
        self.assertAlmostEqual(volatile.total, normal.total * 2.0, places=2)

    def test_regime_multiplier_crisis(self):
        model = self._make_model()
        normal = model.calculate_cost(10000, 50000, 0.02, 1e9, regime="normal")
        crisis = model.calculate_cost(10000, 50000, 0.02, 1e9, regime="crisis")
        self.assertAlmostEqual(crisis.total, normal.total * 3.0, places=2)

    def test_cost_scales_with_size(self):
        model = self._make_model()
        small = model.calculate_cost(1000, 50000, 0.02, 1e9)
        large = model.calculate_cost(100000, 50000, 0.02, 1e9)
        self.assertGreater(large.total, small.total)

    def test_funding_cost(self):
        model = self._make_model()
        no_hold = model.calculate_cost(10000, 50000, 0.02, 1e9, holding_days=0)
        hold_30 = model.calculate_cost(10000, 50000, 0.02, 1e9, holding_days=30)
        self.assertGreater(hold_30.total, no_hold.total)

    def test_cost_in_bps(self):
        model = self._make_model()
        cost = model.calculate_cost(10000, 50000, 0.02, 1e9)
        bps = model.cost_in_bps(cost, 10000)
        self.assertGreater(bps, 0)

    def test_apply_costs_to_returns(self):
        model = self._make_model()
        returns = np.array([0.01, 0.02, -0.01, 0.005, -0.02])
        trade_mask = np.array([True, False, True, False, False])
        prices = np.array([100, 101, 103, 102, 102.5])
        vols = np.full(5, 0.02)
        volumes = np.full(5, 1e9)
        adj = model.apply_costs_to_returns(returns, trade_mask, prices, vols, volumes)
        # Adjusted returns should be lower at trade points
        self.assertLess(adj[0], returns[0])
        self.assertAlmostEqual(adj[1], returns[1])  # no trade


if __name__ == "__main__":
    unittest.main()
