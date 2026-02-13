"""Tests for AdvancedRegimeDetector (5-state HMM)."""
import unittest
import numpy as np
import pandas as pd
from advanced_regime_detector import (
    AdvancedRegimeDetector, MarketRegime, RegimeState,
    REGIME_ALPHA_WEIGHTS, ALPHA_TO_SIGNAL_MAP,
)


def _make_price_df(n=500, seed=42):
    """Generate synthetic OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    close = np.maximum(close, 10.0)
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    open_ = close + rng.randn(n) * 0.3
    volume = rng.uniform(100, 10000, n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    })


class TestAdvancedRegimeDetector(unittest.TestCase):

    def test_fit_with_sufficient_data(self):
        detector = AdvancedRegimeDetector({"min_samples": 100})
        df = _make_price_df(n=400)
        result = detector.fit(df)
        self.assertTrue(result)
        self.assertTrue(detector.is_fitted)

    def test_fit_with_insufficient_data(self):
        detector = AdvancedRegimeDetector({"min_samples": 200})
        df = _make_price_df(n=50)
        result = detector.fit(df)
        self.assertFalse(result)
        self.assertFalse(detector.is_fitted)

    def test_predict_returns_regime_state(self):
        detector = AdvancedRegimeDetector({"min_samples": 100})
        df = _make_price_df(n=400)
        detector.fit(df)
        state = detector.predict(df)
        self.assertIsNotNone(state)
        self.assertIsInstance(state, RegimeState)
        self.assertIsInstance(state.current_regime, MarketRegime)
        self.assertGreater(state.confidence, 0.0)
        self.assertLessEqual(state.confidence, 1.0)

    def test_regime_probabilities_sum_to_one(self):
        detector = AdvancedRegimeDetector({"min_samples": 100})
        df = _make_price_df(n=400)
        detector.fit(df)
        state = detector.predict(df)
        if state and state.regime_probabilities:
            total = sum(state.regime_probabilities.values())
            self.assertAlmostEqual(total, 1.0, places=2)

    def test_alpha_weights_structure(self):
        """All 5 regimes have the 4 expected alpha weight keys."""
        for regime in MarketRegime:
            weights = REGIME_ALPHA_WEIGHTS[regime]
            self.assertIn("momentum_boost", weights)
            self.assertIn("mean_rev_boost", weights)
            self.assertIn("volatility_boost", weights)
            self.assertIn("flow_boost", weights)

    def test_alpha_to_signal_map_keys(self):
        """ALPHA_TO_SIGNAL_MAP covers all alpha weight keys."""
        expected_keys = {"momentum_boost", "mean_rev_boost", "volatility_boost", "flow_boost"}
        self.assertEqual(set(ALPHA_TO_SIGNAL_MAP.keys()), expected_keys)

    def test_maybe_refit_triggers(self):
        detector = AdvancedRegimeDetector({"min_samples": 100, "refit_interval": 10})
        df = _make_price_df(n=400)
        detector.fit(df)
        # Should not refit at cycle 5
        result5 = detector.maybe_refit(df, 5)
        self.assertFalse(result5)
        # Should refit at cycle 10
        result10 = detector.maybe_refit(df, 10)
        self.assertTrue(result10)

    def test_duration_estimate_positive(self):
        detector = AdvancedRegimeDetector({"min_samples": 100})
        df = _make_price_df(n=400)
        detector.fit(df)
        state = detector.predict(df)
        if state:
            self.assertGreaterEqual(state.regime_duration_estimate, 0)


if __name__ == "__main__":
    unittest.main()
