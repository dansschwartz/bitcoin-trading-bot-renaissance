"""Tests for SimReporter."""
import json
import os
import tempfile
import unittest
import numpy as np
from sim_config import SimulationResult, ValidationScore, BacktestResult
from sim_reporting import SimReporter


class TestSimReporter(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.reporter = SimReporter({"output_dir": self._tmpdir,
                                      "save_plots": False,
                                      "save_parquet": False})

    def test_save_metrics_json(self):
        metrics = {"sharpe": 1.5, "array": np.array([1, 2, 3])}
        path = self.reporter.save_metrics_json(metrics, "test_metrics.json")
        with open(path) as f:
            loaded = json.load(f)
        self.assertAlmostEqual(loaded["sharpe"], 1.5)
        self.assertEqual(loaded["array"], [1, 2, 3])

    def test_save_summary_csv(self):
        summaries = [{"model": "GBM", "score": 7.5}, {"model": "MC", "score": 6.0}]
        path = self.reporter.save_summary_csv(summaries, "test_summary.csv")
        self.assertTrue(os.path.exists(path))
        import pandas as pd
        df = pd.read_csv(path)
        self.assertEqual(len(df), 2)
        self.assertIn("model", df.columns)

    def test_save_simulation_paths_csv(self):
        paths = np.random.default_rng(42).normal(100, 5, (5, 10))
        sr = SimulationResult(model_name="Test", asset="BTC", paths=paths)
        path = self.reporter.save_simulation_paths(sr, fmt="csv")
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith(".csv"))

    def test_generate_full_report(self):
        all_results = {
            "simulation_results": {},
            "validation_scores": [],
            "backtest_results": [],
            "metrics": {"test": 1.0},
        }
        report_path = self.reporter.generate_full_report(all_results)
        self.assertTrue(os.path.exists(report_path))
        with open(report_path) as f:
            content = f.read()
        self.assertIn("Renaissance Simulation Report", content)


if __name__ == "__main__":
    unittest.main()
