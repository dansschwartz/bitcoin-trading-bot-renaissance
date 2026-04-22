"""Main orchestrator: ties all simulation modules together.

Usage:
    python sim_runner.py --assets BTC-USD ETH-USD SOL-USD --output-dir sim_output
"""

import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from sim_config import (
    DEFAULT_CONFIG, SimulationResult, ValidationScore, BacktestResult, merge_config,
)
from sim_data_ingest import SimDataIngest
from sim_models_base import SimulationModel
from sim_model_monte_carlo import MonteCarloSimulator
from sim_model_gbm import GBMSimulator
from sim_model_heston import HestonSimulator
from sim_model_hmm_regime import HMMRegimeSimulator
from sim_model_ngram import NGramSimulator
from sim_statistics import SimStatistics
from sim_validation import SimValidationSuite
from sim_transaction_costs import SimTransactionCostModel
from sim_stress_test import SimStressTest
from sim_bayesian_uncertainty import SimBayesianUncertainty
from sim_strategies import (
    SimMeanReversionStrategy, SimContrarianScanner, SimBacktestEngine,
)
from sim_portfolio import SimPortfolioSimulator
from sim_reporting import SimReporter


class SimulationRunner:
    """Orchestrates the full simulation → validation → backtest → report pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = merge_config(config)
        self.logger = logger or self._setup_logger()

        sim_cfg = self.config.get("simulation", {})
        self.n_simulations = sim_cfg.get("n_simulations", 1000)
        self.n_steps = sim_cfg.get("n_steps", 252)
        self.dt = sim_cfg.get("dt", 1.0 / 252)
        self.seed = sim_cfg.get("random_seed", 42)

        # Initialise components
        self.data_ingest = SimDataIngest(self.config.get("data", {}), self.logger)
        self.stats = SimStatistics(self.config, self.logger)
        self.validator = SimValidationSuite(self.config, self.logger)
        self.cost_model = SimTransactionCostModel(
            self.config.get("transaction_costs", {}), self.logger
        )
        self.stress_tester = SimStressTest(
            self.config.get("stress_test", {}), self.logger
        )
        self.bayesian = SimBayesianUncertainty(
            self.config.get("bootstrap", {}), self.logger
        )
        self.portfolio_sim = SimPortfolioSimulator(self.config, self.logger)
        self.reporter = SimReporter(self.config.get("output", {}), self.logger)

        self.models = self._init_models()

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _init_models(self) -> Dict[str, SimulationModel]:
        model_cfg = self.config.get("models", {})
        models: Dict[str, SimulationModel] = {}

        if model_cfg.get("monte_carlo", {}).get("enabled", True):
            models["MonteCarloSimulator"] = MonteCarloSimulator(
                model_cfg.get("monte_carlo", {}), self.logger
            )
        if model_cfg.get("gbm", {}).get("enabled", True):
            models["GBMSimulator"] = GBMSimulator(
                model_cfg.get("gbm", {}), self.logger
            )
        if model_cfg.get("heston", {}).get("enabled", True):
            models["HestonSimulator"] = HestonSimulator(
                model_cfg.get("heston", {}), self.logger
            )
        if model_cfg.get("hmm_regime", {}).get("enabled", True):
            models["HMMRegimeSimulator"] = HMMRegimeSimulator(
                model_cfg.get("hmm_regime", {}), self.logger
            )
        if model_cfg.get("ngram", {}).get("enabled", True):
            models["NGramSimulator"] = NGramSimulator(
                model_cfg.get("ngram", {}), self.logger
            )
        return models

    # ------------------------------------------------------------------
    # Single-asset pipeline
    # ------------------------------------------------------------------

    def run_single_asset(self, symbol: str) -> Dict[str, Any]:
        """Full pipeline for one asset."""
        self.logger.info(f"=== Running simulation for {symbol} ===")
        result: Dict[str, Any] = {"asset": symbol}

        # 1. Fetch data
        df = self.data_ingest.fetch_ohlcv(symbol)
        if df.empty:
            self.logger.warning(f"No data for {symbol} — generating synthetic data")
            rng = np.random.default_rng(self.seed)
            returns = rng.normal(0.0003, 0.02, 500)
            prices = 100.0 * np.exp(np.cumsum(returns))
        else:
            returns = self.data_ingest.get_log_returns(df)
            prices = df["close"].values.astype(float)

        result["n_observations"] = len(returns)
        result["empirical_stats"] = self.stats.full_analysis(returns)

        # 2. Calibrate + simulate all models
        sim_results: Dict[str, SimulationResult] = {}
        validation_scores: List[ValidationScore] = []

        for model_name, model in self.models.items():
            self.logger.info(f"  Calibrating {model_name}...")
            try:
                model.calibrate(returns, prices)
            except Exception as e:
                self.logger.error(f"  Calibration failed for {model_name}: {e}")
                continue

            self.logger.info(f"  Simulating {model_name} ({self.n_simulations} paths)...")
            try:
                paths = model.simulate(
                    S0=prices[-1],
                    n_steps=self.n_steps,
                    n_simulations=self.n_simulations,
                    dt=self.dt,
                    seed=self.seed,
                )
            except Exception as e:
                self.logger.error(f"  Simulation failed for {model_name}: {e}")
                continue

            sr = SimulationResult(
                model_name=model_name,
                asset=symbol,
                paths=paths,
                parameters=model.parameters,
            )
            sim_results[model_name] = sr

            # Save paths
            self.reporter.save_simulation_paths(sr)

            # Plot fan
            self.reporter.plot_simulation_fan(sr, prices)

            # 3. Validate
            self.logger.info(f"  Validating {model_name}...")
            try:
                score = self.validator.compute_scorecard(
                    model_name, symbol, returns, paths
                )
                validation_scores.append(score)
                self.logger.info(
                    f"  {model_name} score: {score.composite_score:.1f}/10"
                )

                # Distribution comparison plot
                sim_rets = sr.log_returns()
                self.reporter.plot_distribution_comparison(
                    returns, sim_rets[:50000], model_name
                )
            except Exception as e:
                self.logger.error(f"  Validation failed for {model_name}: {e}")

        result["simulation_results"] = sim_results
        result["validation_scores"] = validation_scores

        # 4. Scorecard plot
        if validation_scores:
            self.reporter.plot_validation_scorecard(validation_scores)

        # 5. Backtest strategies on simulated data
        backtest_results = self._run_backtests(prices, symbol)
        result["backtest_results"] = backtest_results

        # 6. Stress tests (on best model's paths)
        if sim_results:
            best_model = max(
                validation_scores, key=lambda s: s.composite_score
            ).model_name if validation_scores else list(sim_results.keys())[0]
            best_paths = sim_results[best_model].paths
            stress = self._run_stress_tests(best_paths, symbol)
            result["stress_tests"] = stress

        return result

    # ------------------------------------------------------------------
    # Multi-asset pipeline
    # ------------------------------------------------------------------

    def run_multi_asset(self, symbols: List[str]) -> Dict[str, Any]:
        """Run per-asset pipelines + correlated portfolio simulation."""
        result: Dict[str, Any] = {"assets": symbols}
        per_asset: Dict[str, Any] = {}

        for sym in symbols:
            per_asset[sym] = self.run_single_asset(sym)
        result["per_asset"] = per_asset

        # Correlated portfolio simulation
        if len(symbols) >= 2:
            self.logger.info("=== Running multi-asset correlated simulation ===")
            aligned = self.data_ingest.get_aligned_returns(symbols)
            if not aligned.empty:
                corr = self.portfolio_sim.compute_correlation_matrix(aligned)
                self.reporter.plot_correlation_heatmap(corr, list(aligned.columns))

                # Use GBM for portfolio simulation (simplest for correlated paths)
                gbm_models: Dict[str, SimulationModel] = {}
                asset_prices: Dict[str, float] = {}
                for sym in aligned.columns:
                    m = GBMSimulator({})
                    rets = aligned[sym].values
                    data = per_asset.get(sym, {})
                    stats = data.get("empirical_stats", {})
                    p = 100.0  # default
                    m.calibrate(rets, np.exp(np.cumsum(np.concatenate([[np.log(p)], rets]))))
                    gbm_models[sym] = m
                    asset_prices[sym] = p

                portfolio_paths = self.portfolio_sim.generate_correlated_paths(
                    gbm_models, asset_prices, corr,
                    self.n_steps, min(self.n_simulations, 200),
                    seed=self.seed,
                )
                weights = {sym: 1.0 / len(symbols) for sym in symbols}
                portfolio_eq = self.portfolio_sim.portfolio_equity(
                    portfolio_paths, weights
                )
                result["portfolio_equity"] = portfolio_eq
                result["correlation_matrix"] = corr

        return result

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run_all(self) -> Dict[str, Any]:
        """Run for all configured assets + multi-asset portfolio."""
        asset_cfgs = self.config.get("assets", [])
        symbols = [a["symbol"] if isinstance(a, dict) else a for a in asset_cfgs]
        if not symbols:
            symbols = ["BTC-USD"]

        result = self.run_multi_asset(symbols)

        # Generate full report
        all_results = {
            "simulation_results": {},
            "validation_scores": [],
            "backtest_results": [],
            "metrics": {},
        }
        for sym, data in result.get("per_asset", {}).items():
            all_results["simulation_results"].update(
                data.get("simulation_results", {})
            )
            all_results["validation_scores"].extend(
                data.get("validation_scores", [])
            )
            all_results["backtest_results"].extend(
                data.get("backtest_results", [])
            )

        self.reporter.generate_full_report(all_results)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_backtests(self, prices: np.ndarray,
                       asset: str) -> List[BacktestResult]:
        """Run all configured strategies on the given price series."""
        results: List[BacktestResult] = []
        engine = SimBacktestEngine(
            self.config.get("backtest", {}),
            cost_model=self.cost_model,
            logger=self.logger,
        )

        # Mean reversion
        mr = SimMeanReversionStrategy(
            self.config.get("strategies", {}).get("mean_reversion", {})
        )
        signals = mr.generate_signals(prices)
        if np.any(signals != 0):
            bt = engine.run_backtest(
                prices, signals, asset=asset, strategy_name="MeanReversion"
            )
            results.append(bt)
            self.logger.info(
                f"  MeanReversion: Sharpe={bt.metrics['sharpe_ratio']:.2f}, "
                f"Return={bt.metrics['total_return']:.1%}"
            )

        # Contrarian
        cs = SimContrarianScanner(
            self.config.get("strategies", {}).get("contrarian_scanner", {})
        )
        signals = cs.generate_signals(prices)
        if np.any(signals != 0):
            bt = engine.run_backtest(
                prices, signals, asset=asset, strategy_name="Contrarian"
            )
            results.append(bt)
            self.logger.info(
                f"  Contrarian: Sharpe={bt.metrics['sharpe_ratio']:.2f}, "
                f"Return={bt.metrics['total_return']:.1%}"
            )

        if results:
            self.reporter.plot_equity_curves(results)

        return results

    def _run_stress_tests(self, paths: np.ndarray,
                          asset: str) -> Dict[str, Any]:
        """Apply stress scenarios to simulation paths."""
        n_steps = paths.shape[1]
        mid = n_steps // 2
        results: Dict[str, Any] = {}

        # Flash crash
        stressed = self.stress_tester.inject_flash_crash(paths, crash_day=mid)
        results["flash_crash"] = {
            "median_before": float(np.median(paths[:, mid])),
            "median_after": float(np.median(stressed[:, mid])),
            "decline_pct": float(
                np.median(stressed[:, mid]) / np.median(paths[:, mid]) - 1
            ),
        }

        # COVID decline
        stressed = self.stress_tester.inject_covid_decline(paths, start_day=mid)
        results["covid_decline"] = {
            "median_end_normal": float(np.median(paths[:, -1])),
            "median_end_stressed": float(np.median(stressed[:, -1])),
        }

        # Death spiral
        stressed = self.stress_tester.inject_death_spiral(paths, start_day=mid)
        results["death_spiral"] = {
            "median_end_normal": float(np.median(paths[:, -1])),
            "median_end_stressed": float(np.median(stressed[:, -1])),
        }

        return results

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger("SimRunner")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                                  datefmt="%H:%M:%S")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Renaissance-Inspired Crypto Price Simulation & Backtesting"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file (overrides defaults)"
    )
    parser.add_argument(
        "--assets", nargs="+",
        default=["BTC-USD", "ETH-USD", "SOL-USD"],
        help="Asset symbols to simulate"
    )
    parser.add_argument(
        "--output-dir", type=str, default="sim_output",
        help="Directory for output files"
    )
    parser.add_argument(
        "--n-simulations", type=int, default=None,
        help="Number of simulation paths per model"
    )
    parser.add_argument(
        "--n-steps", type=int, default=None,
        help="Number of time steps per simulation"
    )
    args = parser.parse_args()

    # Build config
    user_config: Dict[str, Any] = {}
    if args.config:
        with open(args.config) as f:
            user_config = json.load(f)

    user_config["assets"] = [{"symbol": s} for s in args.assets]
    user_config.setdefault("output", {})["output_dir"] = args.output_dir

    if args.n_simulations:
        user_config.setdefault("simulation", {})["n_simulations"] = args.n_simulations
    if args.n_steps:
        user_config.setdefault("simulation", {})["n_steps"] = args.n_steps

    runner = SimulationRunner(user_config)
    runner.run_all()


if __name__ == "__main__":
    main()
