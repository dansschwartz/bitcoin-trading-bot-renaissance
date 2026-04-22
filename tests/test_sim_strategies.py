"""Tests for SimMeanReversionStrategy, SimContrarianScanner, SimBacktestEngine."""
import unittest
import numpy as np
from sim_strategies import (
    SimMeanReversionStrategy, SimContrarianScanner, SimBacktestEngine,
)


class TestMeanReversionStrategy(unittest.TestCase):

    def _make_strategy(self, **overrides):
        cfg = {"entry_z": 2.0, "exit_z": 0.0, "lookback": 30}
        cfg.update(overrides)
        return SimMeanReversionStrategy(cfg)

    def _make_mean_reverting_prices(self, n=200, seed=42):
        """Generate prices that oscillate around a mean."""
        rng = np.random.default_rng(seed)
        mean_price = 100.0
        prices = mean_price + 10 * np.sin(np.linspace(0, 6 * np.pi, n))
        prices += rng.normal(0, 0.5, n)
        return np.maximum(prices, 50)

    def test_generates_signals(self):
        strategy = self._make_strategy(lookback=20)
        prices = self._make_mean_reverting_prices()
        signals = strategy.generate_signals(prices)
        self.assertEqual(len(signals), len(prices))
        # Should have some buy and sell signals
        self.assertTrue(np.any(signals > 0))
        self.assertTrue(np.any(signals < 0))

    def test_no_signals_short_data(self):
        strategy = self._make_strategy(lookback=100)
        prices = np.array([100, 101, 102])
        signals = strategy.generate_signals(prices)
        self.assertTrue(np.all(signals == 0))

    def test_signals_range(self):
        strategy = self._make_strategy(lookback=20)
        prices = self._make_mean_reverting_prices()
        signals = strategy.generate_signals(prices)
        self.assertTrue(np.all(signals >= -1))
        self.assertTrue(np.all(signals <= 1))


class TestContrarianScanner(unittest.TestCase):

    def _make_scanner(self, **overrides):
        cfg = {"min_consecutive": 3}
        cfg.update(overrides)
        return SimContrarianScanner(cfg)

    def test_contrarian_buy_after_decline(self):
        scanner = self._make_scanner(min_consecutive=3)
        # 3 consecutive down days then flat
        prices = np.array([100, 99, 98, 97, 97, 97, 97, 97])
        signals = scanner.generate_signals(prices)
        # After 3 down days (indices 1,2,3), signal at index 4 should be buy
        self.assertEqual(signals[4], 1.0)

    def test_contrarian_sell_after_rally(self):
        scanner = self._make_scanner(min_consecutive=3)
        prices = np.array([100, 101, 102, 103, 103, 103, 103, 103])
        signals = scanner.generate_signals(prices)
        self.assertEqual(signals[4], -1.0)

    def test_no_signals_short_data(self):
        scanner = self._make_scanner(min_consecutive=5)
        prices = np.array([100, 101, 102])
        signals = scanner.generate_signals(prices)
        self.assertTrue(np.all(signals == 0))


class TestBacktestEngine(unittest.TestCase):

    def _make_engine(self, **overrides):
        cfg = {"initial_capital": 100000, "position_fraction": 0.25}
        cfg.update(overrides)
        return SimBacktestEngine(cfg)

    def test_backtest_returns_result(self):
        engine = self._make_engine()
        rng = np.random.default_rng(42)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, 200)))
        signals = np.zeros(200)
        signals[50] = 1.0  # buy
        signals[100] = -1.0  # sell
        result = engine.run_backtest(prices, signals)
        self.assertEqual(len(result.equity_curve), 200)
        self.assertIn("sharpe_ratio", result.metrics)
        self.assertIn("max_drawdown", result.metrics)

    def test_no_trades_equity_flat(self):
        engine = self._make_engine()
        prices = np.linspace(100, 110, 100)
        signals = np.zeros(100)
        result = engine.run_backtest(prices, signals)
        # With no trades, equity should stay at initial capital
        self.assertAlmostEqual(result.equity_curve[0], 100000)
        self.assertEqual(result.metrics["n_trades"], 0)

    def test_metrics_in_valid_range(self):
        engine = self._make_engine()
        rng = np.random.default_rng(42)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, 300)))
        signals = np.zeros(300)
        signals[50] = 1.0
        signals[100] = -1.0
        signals[150] = 1.0
        signals[250] = -1.0
        result = engine.run_backtest(prices, signals)
        m = result.metrics
        self.assertGreaterEqual(m["win_rate"], 0)
        self.assertLessEqual(m["win_rate"], 1)
        self.assertLessEqual(m["max_drawdown"], 0)

    def test_regime_breakdown(self):
        engine = self._make_engine()
        rng = np.random.default_rng(42)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, 200)))
        signals = np.zeros(200)
        signals[50] = 1.0
        signals[100] = -1.0
        regimes = np.array(["bull"] * 100 + ["bear"] * 100)
        result = engine.run_backtest(prices, signals, regimes=regimes)
        self.assertIsInstance(result.regime_performance, dict)


if __name__ == "__main__":
    unittest.main()
