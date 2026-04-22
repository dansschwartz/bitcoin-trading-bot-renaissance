"""Geometric Brownian Motion (GBM) simulator: dS = mu*S*dt + sigma*S*dW."""

import logging
import numpy as np
from typing import Any, Dict, Optional

from sim_models_base import SimulationModel


class GBMSimulator(SimulationModel):
    """Classic GBM with exact log-normal solution.

    Calibrates annualised mu and sigma from daily log returns, then
    generates paths via:
        S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    """

    def calibrate(self, returns: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        mu_daily = float(np.mean(returns))
        sigma_daily = float(np.std(returns, ddof=1))
        mu_annual = mu_daily * 252
        sigma_annual = sigma_daily * np.sqrt(252)

        self._parameters = {
            "mu": mu_annual,
            "sigma": sigma_annual,
            "mu_daily": mu_daily,
            "sigma_daily": sigma_daily,
        }
        self._calibrated = True
        self.logger.info(
            f"GBM calibrated: mu={mu_annual:.4f}/yr, sigma={sigma_annual:.4f}/yr"
        )
        return self._parameters

    def simulate(self, S0: float, n_steps: int, n_simulations: int,
                 dt: float = 1.0 / 252, seed: Optional[int] = None) -> np.ndarray:
        if not self._calibrated:
            raise RuntimeError("Model not calibrated — call calibrate() first")

        rng = np.random.default_rng(seed)
        mu = self._parameters["mu"]
        sigma = self._parameters["sigma"]

        Z = rng.standard_normal((n_simulations, n_steps))
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        log_increments = drift + diffusion

        log_paths = np.cumsum(log_increments, axis=1)
        paths = S0 * np.exp(
            np.column_stack([np.zeros(n_simulations), log_paths])
        )
        return paths
