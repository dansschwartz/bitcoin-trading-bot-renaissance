"""Heston stochastic-volatility model.

dS = mu*S*dt + sqrt(v)*S*dW_s
dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
Corr(dW_s, dW_v) = rho
"""

import logging
import numpy as np
from typing import Any, Dict, Optional

from sim_models_base import SimulationModel


class HestonSimulator(SimulationModel):
    """Euler-discretised Heston model with variance absorption at zero.

    Parameters can be supplied via *config* or calibrated from data.
    Calibration estimates ``v0`` and ``theta`` from empirical volatility
    while ``kappa``, ``xi``, ``rho`` are taken from config (they require
    option-price data or MLE for rigorous calibration).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        # User-supplied or default structural parameters
        self._kappa = self.config.get("kappa", 2.0)
        self._xi = self.config.get("xi", 0.5)
        self._rho = self.config.get("rho", -0.7)

    def calibrate(self, returns: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        mu_annual = float(np.mean(returns)) * 252
        sigma_daily = float(np.std(returns, ddof=1))
        v0_config = self.config.get("v0")
        v0 = v0_config if v0_config is not None else (sigma_daily ** 2) * 252
        theta = v0  # long-run variance ≈ current realised variance

        kappa = self._kappa
        xi = self._xi
        rho = self._rho

        feller_satisfied = (2 * kappa * theta) > (xi ** 2)

        self._parameters = {
            "mu": mu_annual,
            "v0": float(v0),
            "theta": float(theta),
            "kappa": float(kappa),
            "xi": float(xi),
            "rho": float(rho),
            "feller_satisfied": feller_satisfied,
        }
        self._calibrated = True

        feller_str = "YES" if feller_satisfied else "NO (variance may hit zero)"
        self.logger.info(
            f"Heston calibrated: mu={mu_annual:.4f}, v0={v0:.4f}, "
            f"theta={theta:.4f}, kappa={kappa}, xi={xi}, rho={rho}, "
            f"Feller={feller_str}"
        )
        return self._parameters

    def simulate(self, S0: float, n_steps: int, n_simulations: int,
                 dt: float = 1.0 / 252, seed: Optional[int] = None) -> np.ndarray:
        if not self._calibrated:
            raise RuntimeError("Model not calibrated — call calibrate() first")

        rng = np.random.default_rng(seed)
        p = self._parameters
        mu, v0, theta = p["mu"], p["v0"], p["theta"]
        kappa, xi, rho = p["kappa"], p["xi"], p["rho"]

        # Correlated Brownian motions
        Z1 = rng.standard_normal((n_simulations, n_steps))
        Z2 = rng.standard_normal((n_simulations, n_steps))
        W_s = Z1
        W_v = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2

        S = np.zeros((n_simulations, n_steps + 1))
        v = np.zeros((n_simulations, n_steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0

        sqrt_dt = np.sqrt(dt)

        for t in range(n_steps):
            v_pos = np.maximum(v[:, t], 0.0)       # absorption scheme
            sqrt_v = np.sqrt(v_pos)

            # Price step (log-Euler)
            S[:, t + 1] = S[:, t] * np.exp(
                (mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * W_s[:, t]
            )

            # Variance step
            v[:, t + 1] = (
                v[:, t]
                + kappa * (theta - v_pos) * dt
                + xi * sqrt_v * sqrt_dt * W_v[:, t]
            )
            v[:, t + 1] = np.maximum(v[:, t + 1], 0.0)

        return S
