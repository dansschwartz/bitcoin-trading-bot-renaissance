"""Multi-asset portfolio simulation with correlated paths via Cholesky."""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from sim_models_base import SimulationModel


class SimPortfolioSimulator:
    """Generate correlated multi-asset price paths.

    Uses Cholesky decomposition to inject cross-asset correlation
    into independent simulation models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    def compute_correlation_matrix(self,
                                   multi_asset_returns: pd.DataFrame) -> np.ndarray:
        """Compute empirical correlation matrix from aligned return DataFrame.

        Ensures the result is positive semi-definite via eigenvalue clipping.
        """
        corr = multi_asset_returns.corr().values
        return self._nearest_psd(corr)

    @staticmethod
    def _nearest_psd(A: np.ndarray) -> np.ndarray:
        """Project matrix to nearest positive semi-definite matrix."""
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Normalise to correlation (diag = 1)
        d = np.sqrt(np.diag(psd))
        d[d == 0] = 1.0
        psd = psd / np.outer(d, d)
        np.fill_diagonal(psd, 1.0)
        return psd

    # ------------------------------------------------------------------
    # Correlated path generation
    # ------------------------------------------------------------------

    def generate_correlated_paths(
        self,
        models: Dict[str, SimulationModel],
        asset_prices: Dict[str, float],
        correlation_matrix: np.ndarray,
        n_steps: int,
        n_simulations: int,
        dt: float = 1.0 / 252,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate correlated price paths for multiple assets.

        1. Cholesky-decompose the correlation matrix.
        2. Generate independent standard normals for each asset.
        3. Correlate them via L @ Z.
        4. For each asset, use its model's calibrated drift + vol
           with the correlated innovations.

        Returns dict mapping asset symbol → paths array (n_sims, n_steps+1).
        """
        assets = list(models.keys())
        n_assets = len(assets)
        rng = np.random.default_rng(seed)

        # Ensure correlation matrix matches asset count
        if correlation_matrix.shape[0] != n_assets:
            self.logger.warning("Correlation matrix size mismatch; falling back to independent")
            return self._independent_paths(models, asset_prices, n_steps, n_simulations, dt, seed)

        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            self.logger.warning("Cholesky failed — using nearest PSD")
            L = np.linalg.cholesky(self._nearest_psd(correlation_matrix))

        # Generate independent Z's: shape (n_assets, n_sims, n_steps)
        Z_indep = rng.standard_normal((n_assets, n_simulations, n_steps))

        # Correlate: for each (sim, step), Z_corr = L @ Z_indep
        Z_corr = np.zeros_like(Z_indep)
        for t in range(n_steps):
            for s in range(n_simulations):
                Z_corr[:, s, t] = L @ Z_indep[:, s, t]

        # Build paths per asset using correlated innovations
        result: Dict[str, np.ndarray] = {}
        for idx, asset in enumerate(assets):
            model = models[asset]
            S0 = asset_prices.get(asset, 100.0)
            params = model.parameters

            # Extract drift and vol — handle different model types
            mu = params.get("mu", 0.0)
            sigma = params.get("sigma", params.get("sigma_daily", 0.02) * np.sqrt(252))

            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt)

            log_increments = drift + diffusion * Z_corr[idx]
            log_paths = np.cumsum(log_increments, axis=1)
            paths = S0 * np.exp(
                np.column_stack([np.zeros(n_simulations), log_paths])
            )
            result[asset] = paths

        return result

    def _independent_paths(
        self,
        models: Dict[str, SimulationModel],
        asset_prices: Dict[str, float],
        n_steps: int,
        n_simulations: int,
        dt: float,
        seed: Optional[int],
    ) -> Dict[str, np.ndarray]:
        """Fallback: each model simulates independently."""
        result: Dict[str, np.ndarray] = {}
        rng = np.random.default_rng(seed)
        for asset, model in models.items():
            S0 = asset_prices.get(asset, 100.0)
            s = int(rng.integers(0, 2**31))
            result[asset] = model.simulate(S0, n_steps, n_simulations, dt, seed=s)
        return result

    # ------------------------------------------------------------------
    # Portfolio equity
    # ------------------------------------------------------------------

    def portfolio_equity(self, asset_paths: Dict[str, np.ndarray],
                         weights: Dict[str, float]) -> np.ndarray:
        """Compute portfolio equity from weighted normalised asset paths.

        Each asset path is normalised by its starting price, then weighted.
        """
        assets = list(asset_paths.keys())
        if not assets:
            return np.array([])

        # Use mean path per asset
        ref = asset_paths[assets[0]]
        n_steps = ref.shape[1]

        portfolio = np.zeros(n_steps)
        total_weight = sum(weights.get(a, 0.0) for a in assets)
        if total_weight == 0:
            total_weight = len(assets)
            weights = {a: 1.0 for a in assets}

        for asset in assets:
            paths = asset_paths[asset]
            mean_path = paths.mean(axis=0)
            normalised = mean_path / mean_path[0] if mean_path[0] > 0 else mean_path
            w = weights.get(asset, 0.0) / total_weight
            portfolio += w * normalised

        return portfolio
