"""Tests for sim_config dataclasses and defaults."""
import unittest
import numpy as np
from sim_config import (
    AssetConfig, SimulationResult, BacktestResult, ValidationScore,
    TradeCost, ParameterDistribution, Trade, DEFAULT_CONFIG, merge_config,
)


class TestSimConfig(unittest.TestCase):

    def test_default_config_has_required_sections(self):
        for key in ("assets", "data", "simulation", "models",
                     "transaction_costs", "strategies", "bootstrap",
                     "stress_test", "output", "backtest"):
            self.assertIn(key, DEFAULT_CONFIG)

    def test_default_config_assets(self):
        assets = DEFAULT_CONFIG["assets"]
        self.assertEqual(len(assets), 3)
        symbols = [a["symbol"] for a in assets]
        self.assertIn("BTC-USD", symbols)

    def test_asset_config_auto_ticker(self):
        ac = AssetConfig(symbol="BTC-USD")
        self.assertEqual(ac.yfinance_ticker, "BTC-USD")

    def test_simulation_result_properties(self):
        paths = np.ones((10, 253))
        sr = SimulationResult(model_name="GBM", asset="BTC-USD", paths=paths)
        self.assertEqual(sr.n_simulations, 10)
        self.assertEqual(sr.n_steps, 252)

    def test_simulation_result_mean_path(self):
        paths = np.ones((5, 10)) * 100
        sr = SimulationResult(model_name="T", asset="A", paths=paths)
        mp = sr.mean_path()
        self.assertEqual(len(mp), 10)
        self.assertAlmostEqual(mp[0], 100.0)

    def test_validation_score_composite_range(self):
        vs = ValidationScore(
            model_name="M", asset="A",
            ks_stat=0.05, ks_pvalue=0.5,
            ad_stat=1.0, ad_pvalue=0.3,
            acf_rmse=0.1, garch_param_distance=0.2,
            vol_clustering_score=0.15, composite_score=7.5,
        )
        self.assertGreaterEqual(vs.composite_score, 0)
        self.assertLessEqual(vs.composite_score, 10)

    def test_trade_cost_fields(self):
        tc = TradeCost(maker_fee=1, taker_fee=0, slippage=0.5,
                       half_spread=0.3, funding_cost=0.1, total=1.9)
        self.assertAlmostEqual(tc.total, 1.9)

    def test_merge_config_override(self):
        merged = merge_config({"simulation": {"n_simulations": 500}})
        self.assertEqual(merged["simulation"]["n_simulations"], 500)
        # Other defaults preserved
        self.assertEqual(merged["simulation"]["n_steps"], 252)

    def test_merge_config_none(self):
        merged = merge_config(None)
        self.assertEqual(merged, DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
