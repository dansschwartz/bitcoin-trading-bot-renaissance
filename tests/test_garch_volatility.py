"""Tests for GARCHVolatilityEngine."""
import unittest
import numpy as np
from garch_volatility_engine import GARCHVolatilityEngine


class TestGARCHVolatilityEngine(unittest.TestCase):

    def _make_engine(self, **overrides):
        cfg = {"enabled": True, "min_observations": 50, "refit_interval_cycles": 10,
               "historical_vol_window": 20}
        cfg.update(overrides)
        return GARCHVolatilityEngine(cfg)

    def _feed_returns(self, engine, product_id="BTC-USD", n=200, seed=42):
        """Feed synthetic prices to generate returns."""
        np.random.seed(seed)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        prices = np.maximum(prices, 10.0)
        for p in prices:
            engine.update_returns(product_id, float(p))

    def test_update_returns_stores_data(self):
        engine = self._make_engine()
        engine.update_returns("BTC-USD", 100.0)
        engine.update_returns("BTC-USD", 101.0)
        engine.update_returns("BTC-USD", 102.0)
        self.assertEqual(len(engine._returns["BTC-USD"]), 2)  # log returns = n-1 prices

    def test_disabled_engine(self):
        engine = GARCHVolatilityEngine({"enabled": False})
        engine.update_returns("BTC-USD", 100.0)
        self.assertNotIn("BTC-USD", engine._returns)
        self.assertEqual(engine.get_position_size_multiplier("BTC-USD"), 1.0)

    def test_invalid_price_ignored(self):
        engine = self._make_engine()
        engine.update_returns("BTC-USD", -5.0)
        engine.update_returns("BTC-USD", 0.0)
        self.assertNotIn("BTC-USD", engine._returns)

    def test_should_refit_false_insufficient_data(self):
        engine = self._make_engine(min_observations=100)
        self._feed_returns(engine, n=50)
        self.assertFalse(engine.should_refit("BTC-USD"))

    def test_should_refit_true(self):
        engine = self._make_engine(min_observations=50, refit_interval_cycles=5)
        self._feed_returns(engine, n=100)
        # After 100 price updates, cycle_count is high enough
        self.assertTrue(engine.should_refit("BTC-USD"))

    def test_fit_model_ewma_fallback(self):
        """Even without arch library, EWMA fallback should work."""
        engine = self._make_engine(min_observations=50)
        self._feed_returns(engine, n=100)
        result = engine.fit_model("BTC-USD")
        self.assertTrue(result)
        self.assertIn("BTC-USD", engine._models)

    def test_forecast_volatility_default(self):
        engine = self._make_engine()
        forecast = engine.forecast_volatility("BTC-USD")
        self.assertEqual(forecast["vol_ratio"], 1.0)
        self.assertEqual(forecast["vol_regime"], "stable")

    def test_forecast_volatility_with_model(self):
        engine = self._make_engine(min_observations=50, historical_vol_window=20)
        self._feed_returns(engine, n=100)
        engine.fit_model("BTC-USD")
        forecast = engine.forecast_volatility("BTC-USD")
        self.assertGreater(forecast["forecast_vol"], 0.0)
        self.assertGreater(forecast["historical_vol"], 0.0)
        self.assertIn(forecast["vol_regime"], ["expanding", "contracting", "stable"])

    def test_position_size_multiplier_range(self):
        engine = self._make_engine(min_observations=50, historical_vol_window=20)
        self._feed_returns(engine, n=100)
        engine.fit_model("BTC-USD")
        mult = engine.get_position_size_multiplier("BTC-USD")
        self.assertGreaterEqual(mult, 0.5)
        self.assertLessEqual(mult, 1.5)

    def test_dynamic_threshold_adjustment(self):
        engine = self._make_engine(min_observations=50, historical_vol_window=20)
        self._feed_returns(engine, n=100)
        engine.fit_model("BTC-USD")
        buy_delta, sell_delta = engine.get_dynamic_threshold_adjustment("BTC-USD")
        # Deltas should be small
        self.assertGreaterEqual(buy_delta, -0.05)
        self.assertLessEqual(buy_delta, 0.05)
        self.assertGreaterEqual(sell_delta, -0.05)
        self.assertLessEqual(sell_delta, 0.05)

    def test_vol_ratio_clipped(self):
        """Vol ratio should be clipped to [0.2, 5.0]."""
        engine = self._make_engine(min_observations=50, historical_vol_window=20)
        self._feed_returns(engine, n=100)
        engine.fit_model("BTC-USD")
        forecast = engine.forecast_volatility("BTC-USD")
        self.assertGreaterEqual(forecast["vol_ratio"], 0.2)
        self.assertLessEqual(forecast["vol_ratio"], 5.0)


if __name__ == "__main__":
    unittest.main()
