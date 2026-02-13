"""Bayesian parameter uncertainty via block bootstrap.

Produces parameter distributions and simulation fans from resampled data.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional

from sim_config import ParameterDistribution
from sim_models_base import SimulationModel


class SimBayesianUncertainty:
    """Block bootstrap for parameter uncertainty estimation.

    Block bootstrap preserves autocorrelation structure by resampling
    contiguous blocks of returns rather than individual observations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        cfg = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.n_bootstrap = cfg.get("n_bootstrap", 200)
        self.block_size = cfg.get("block_size", 20)

    # ------------------------------------------------------------------
    # Block bootstrap
    # ------------------------------------------------------------------

    def block_bootstrap(self, data: np.ndarray,
                        n_bootstrap: Optional[int] = None,
                        block_size: Optional[int] = None,
                        seed: Optional[int] = None) -> List[np.ndarray]:
        """Generate *n_bootstrap* resampled series using block bootstrap.

        Each resample has the same length as the original *data*.
        """
        n = len(data)
        bs = block_size or self.block_size
        nb = n_bootstrap or self.n_bootstrap
        n_blocks = int(np.ceil(n / bs))
        rng = np.random.default_rng(seed)

        resampled: List[np.ndarray] = []
        for _ in range(nb):
            blocks = []
            for _ in range(n_blocks):
                start = rng.integers(0, max(n - bs + 1, 1))
                blocks.append(data[start: start + bs])
            sample = np.concatenate(blocks)[:n]
            resampled.append(sample)
        return resampled

    # ------------------------------------------------------------------
    # Parameter distributions
    # ------------------------------------------------------------------

    def estimate_parameter_distributions(
        self,
        model: SimulationModel,
        returns: np.ndarray,
        prices: np.ndarray,
        n_bootstrap: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, ParameterDistribution]:
        """For each bootstrap resample, recalibrate *model* and collect parameters.

        Returns a dict mapping parameter name → ParameterDistribution.
        """
        samples = self.block_bootstrap(returns, n_bootstrap=n_bootstrap, seed=seed)
        param_collections: Dict[str, List[float]] = {}

        for i, sample in enumerate(samples):
            # Reconstruct price series from resampled returns
            resampled_prices = prices[0] * np.exp(np.concatenate([[0.0], np.cumsum(sample)]))
            try:
                params = model.calibrate(sample, resampled_prices)
                for key, val in params.items():
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        param_collections.setdefault(key, []).append(float(val))
            except Exception as e:
                self.logger.debug(f"Bootstrap sample {i} failed: {e}")

        # Compute distributions
        result: Dict[str, ParameterDistribution] = {}
        for key, values in param_collections.items():
            arr = np.array(values)
            if len(arr) < 2:
                continue
            result[key] = ParameterDistribution(
                param_name=key,
                mean=float(np.mean(arr)),
                std=float(np.std(arr, ddof=1)),
                ci_lower=float(np.percentile(arr, 5)),
                ci_upper=float(np.percentile(arr, 95)),
                samples=arr,
            )

        # Re-calibrate model on original data to restore original params
        model.calibrate(returns, prices)

        return result

    # ------------------------------------------------------------------
    # Simulation fan
    # ------------------------------------------------------------------

    def simulation_fan(
        self,
        model: SimulationModel,
        S0: float,
        n_steps: int,
        returns: np.ndarray,
        prices: np.ndarray,
        n_sims_per_param: int = 10,
        n_bootstrap: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Run simulations from the full parameter distribution.

        For each bootstrap parameter set:
            1. Recalibrate model on resampled data.
            2. Run ``n_sims_per_param`` simulations.

        Returns all paths stacked: shape ``(total_sims, n_steps+1)``.
        """
        samples = self.block_bootstrap(returns, n_bootstrap=n_bootstrap, seed=seed)
        all_paths: List[np.ndarray] = []
        rng = np.random.default_rng(seed)

        for i, sample in enumerate(samples):
            resampled_prices = prices[0] * np.exp(np.concatenate([[0.0], np.cumsum(sample)]))
            try:
                model.calibrate(sample, resampled_prices)
                paths = model.simulate(
                    S0=S0,
                    n_steps=n_steps,
                    n_simulations=n_sims_per_param,
                    seed=int(rng.integers(0, 2**31)),
                )
                all_paths.append(paths)
            except Exception as e:
                self.logger.debug(f"Simulation fan sample {i} failed: {e}")

        # Restore original calibration
        model.calibrate(returns, prices)

        if not all_paths:
            return model.simulate(S0, n_steps, n_sims_per_param, seed=seed)

        return np.vstack(all_paths)
