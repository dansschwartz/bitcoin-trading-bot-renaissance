"""Simple Monte Carlo simulation with empirical drift and optional bootstrap."""

import logging
import numpy as np
from typing import Any, Dict, Optional

from sim_models_base import SimulationModel


class MonteCarloSimulator(SimulationModel):
    """Random walk with drift calibrated from empirical returns.

    Two modes:
    - ``parametric=True``  (default): draw from N(mu*dt, sigma*sqrt(dt))
    - ``parametric=False``: bootstrap-resample from empirical returns
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self._empirical_returns: Optional[np.ndarray] = None
        self._parametric = self.config.get("parametric", True)

    def calibrate(self, returns: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        self._empirical_returns = returns.copy()
        self._parameters = {
            "mu": mu,
            "sigma": sigma,
            "n_observations": len(returns),
        }
        self._calibrated = True
        self.logger.info(
            f"MonteCarlo calibrated: mu={mu:.6f}, sigma={sigma:.6f} "
            f"({len(returns)} obs)"
        )
        return self._parameters

    def simulate(self, S0: float, n_steps: int, n_simulations: int,
                 dt: float = 1.0 / 252, seed: Optional[int] = None) -> np.ndarray:
        if not self._calibrated:
            raise RuntimeError("Model not calibrated — call calibrate() first")

        rng = np.random.default_rng(seed)
        mu = self._parameters["mu"]
        sigma = self._parameters["sigma"]

        if self._parametric or self._empirical_returns is None:
            # Parametric normal draws
            daily_returns = rng.normal(mu, sigma, (n_simulations, n_steps))
        else:
            # Bootstrap from empirical distribution
            idx = rng.integers(0, len(self._empirical_returns),
                               (n_simulations, n_steps))
            daily_returns = self._empirical_returns[idx]

        log_prices = np.cumsum(daily_returns, axis=1)
        paths = np.column_stack([
            np.zeros(n_simulations),
            log_prices,
        ])
        paths = S0 * np.exp(paths)
        return paths
