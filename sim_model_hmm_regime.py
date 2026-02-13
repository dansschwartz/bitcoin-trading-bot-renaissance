"""HMM regime-switching simulation model.

Fits a multi-regime HMM to historical data (reusing AdvancedRegimeDetector),
extracts per-regime (mu, sigma) and the transition matrix, then generates
price paths by simulating regime transitions and drawing returns from the
active regime's distribution.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from sim_models_base import SimulationModel

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    GMM_AVAILABLE = True
except ImportError:
    GMM_AVAILABLE = False


class HMMRegimeSimulator(SimulationModel):
    """Simulate price paths using regime-switching dynamics.

    Calibration:
        1. Fit a GaussianHMM (or GMM fallback) on log-returns.
        2. Extract per-regime (mu, sigma) from hidden-state assignments.
        3. Store transition matrix.

    Simulation:
        1. Start from stationary distribution of the Markov chain.
        2. At each step, draw the next regime from the transition matrix.
        3. Sample a return from the current regime's N(mu, sigma).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self._n_regimes: int = self.config.get("n_regimes", 3)
        self._n_iter: int = self.config.get("n_iter", 150)
        self._covariance_type: str = self.config.get("covariance_type", "full")

        self._regime_params: Dict[int, Dict[str, float]] = {}
        self._transition_matrix: Optional[np.ndarray] = None
        self._stationary_dist: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, returns: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        if len(returns) < 50:
            self.logger.warning("Insufficient data for HMM calibration, need >=50 returns")
            self._set_fallback(returns)
            return self._parameters

        X = returns.reshape(-1, 1)

        if HMM_AVAILABLE:
            success = self._fit_hmm(X, returns)
        elif GMM_AVAILABLE:
            success = self._fit_gmm(X, returns)
        else:
            self.logger.warning("Neither hmmlearn nor sklearn available — using fallback")
            success = False

        if not success:
            self._set_fallback(returns)

        self._calibrated = True
        return self._parameters

    def _fit_hmm(self, X: np.ndarray, returns: np.ndarray) -> bool:
        try:
            model = GaussianHMM(
                n_components=self._n_regimes,
                covariance_type=self._covariance_type,
                n_iter=self._n_iter,
                random_state=42,
            )
            model.fit(X)
            hidden_states = model.predict(X)
            self._transition_matrix = model.transmat_.copy()
            self._extract_regime_params(returns, hidden_states)
            self._compute_stationary()
            self._parameters["method"] = "hmm"
            self.logger.info(f"HMM fit with {self._n_regimes} regimes, "
                             f"{len(returns)} observations")
            return True
        except Exception as e:
            self.logger.warning(f"HMM fit failed: {e}")
            return False

    def _fit_gmm(self, X: np.ndarray, returns: np.ndarray) -> bool:
        try:
            gmm = GaussianMixture(
                n_components=self._n_regimes,
                covariance_type=self._covariance_type,
                max_iter=self._n_iter,
                random_state=42,
            )
            gmm.fit(X)
            hidden_states = gmm.predict(X)
            # Build a synthetic transition matrix from consecutive state assignments
            n_reg = self._n_regimes
            T = np.zeros((n_reg, n_reg))
            for i in range(len(hidden_states) - 1):
                T[hidden_states[i], hidden_states[i + 1]] += 1
            row_sums = T.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            T = T / row_sums
            self._transition_matrix = T
            self._extract_regime_params(returns, hidden_states)
            self._compute_stationary()
            self._parameters["method"] = "gmm"
            self.logger.info(f"GMM fallback fit with {self._n_regimes} regimes")
            return True
        except Exception as e:
            self.logger.warning(f"GMM fit failed: {e}")
            return False

    def _extract_regime_params(self, returns: np.ndarray,
                                hidden_states: np.ndarray) -> None:
        self._regime_params = {}
        for state in range(self._n_regimes):
            mask = hidden_states == state
            if mask.sum() < 3:
                self._regime_params[state] = {
                    "mu": 0.0, "sigma": float(np.std(returns)), "count": 0
                }
            else:
                self._regime_params[state] = {
                    "mu": float(np.mean(returns[mask])),
                    "sigma": float(np.std(returns[mask], ddof=1)),
                    "count": int(mask.sum()),
                }

        self._parameters = {
            "n_regimes": self._n_regimes,
            "regime_params": {k: dict(v) for k, v in self._regime_params.items()},
            "transition_matrix": self._transition_matrix.tolist()
            if self._transition_matrix is not None else None,
        }

    def _compute_stationary(self) -> None:
        """Compute stationary distribution from transition matrix."""
        T = self._transition_matrix
        if T is None:
            self._stationary_dist = None
            return
        eigenvalues, eigenvectors = np.linalg.eig(T.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.abs(eigenvectors[:, idx].real)
        pi /= pi.sum()
        self._stationary_dist = pi

    def _set_fallback(self, returns: np.ndarray) -> None:
        """Single-regime fallback when HMM/GMM fitting fails."""
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.01
        self._regime_params = {0: {"mu": mu, "sigma": sigma, "count": len(returns)}}
        self._transition_matrix = np.array([[1.0]])
        self._stationary_dist = np.array([1.0])
        self._n_regimes = 1
        self._calibrated = True
        self._parameters = {
            "n_regimes": 1,
            "regime_params": {0: {"mu": mu, "sigma": sigma, "count": len(returns)}},
            "transition_matrix": [[1.0]],
            "method": "fallback",
        }

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, S0: float, n_steps: int, n_simulations: int,
                 dt: float = 1.0 / 252, seed: Optional[int] = None) -> np.ndarray:
        if not self._calibrated:
            raise RuntimeError("Model not calibrated — call calibrate() first")

        rng = np.random.default_rng(seed)
        T = self._transition_matrix
        n_reg = len(self._regime_params)
        pi = self._stationary_dist if self._stationary_dist is not None else np.ones(n_reg) / n_reg

        S = np.zeros((n_simulations, n_steps + 1))
        S[:, 0] = S0

        for i in range(n_simulations):
            regime = int(rng.choice(n_reg, p=pi))
            for t in range(n_steps):
                params = self._regime_params.get(
                    regime, {"mu": 0.0, "sigma": 0.01}
                )
                ret = rng.normal(params["mu"], max(params["sigma"], 1e-9))
                S[i, t + 1] = S[i, t] * np.exp(ret)
                # Transition
                regime = int(rng.choice(n_reg, p=T[regime]))

        return S

    def get_regime_labels(self) -> Dict[int, str]:
        """Heuristic labels based on per-regime mu and sigma."""
        labels = {}
        for state, params in self._regime_params.items():
            mu = params["mu"]
            sigma = params["sigma"]
            if mu > 0.001:
                label = "bull"
            elif mu < -0.001:
                label = "bear"
            else:
                label = "neutral"
            if sigma > np.median([p["sigma"] for p in self._regime_params.values()]):
                label += "_volatile"
            else:
                label += "_calm"
            labels[state] = label
        return labels
