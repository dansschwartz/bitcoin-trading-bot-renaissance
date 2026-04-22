"""Validation suite: compare simulated vs empirical distributions, produce scorecard."""

import logging
import numpy as np
from typing import Any, Dict, Optional

from sim_config import ValidationScore

try:
    from scipy import stats as sp_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import acf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


class SimValidationSuite:
    """Compare simulated paths against empirical data.

    Produces a 0--10 composite scorecard per model per asset.

    Score weights:
        KS test        25%
        ACF RMSE        20%
        GARCH params    20%
        Vol clustering  20%
        Anderson-Darling 15%
    """

    SCORE_WEIGHTS = {
        "ks": 0.25,
        "acf": 0.20,
        "garch": 0.20,
        "vol_clust": 0.20,
        "ad": 0.15,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def ks_test(self, empirical: np.ndarray,
                simulated: np.ndarray) -> Dict[str, float]:
        """Two-sample Kolmogorov–Smirnov test."""
        if not SCIPY_AVAILABLE:
            return {"ks_stat": 1.0, "ks_pvalue": 0.0}
        stat, pval = sp_stats.ks_2samp(empirical, simulated)
        return {"ks_stat": float(stat), "ks_pvalue": float(pval)}

    def anderson_darling_test(self, empirical: np.ndarray,
                              simulated: np.ndarray) -> Dict[str, float]:
        """k-sample Anderson–Darling test."""
        if not SCIPY_AVAILABLE:
            return {"ad_stat": 999.0, "ad_pvalue": 0.0}
        try:
            result = sp_stats.anderson_ksamp([empirical, simulated])
            return {"ad_stat": float(result.statistic),
                    "ad_pvalue": float(result.pvalue)}
        except Exception:
            return {"ad_stat": 999.0, "ad_pvalue": 0.0}

    def acf_comparison(self, empirical: np.ndarray,
                       simulated: np.ndarray,
                       max_lag: int = 20) -> float:
        """RMSE between return ACF vectors."""
        if not STATSMODELS_AVAILABLE:
            return 1.0
        if len(empirical) < max_lag + 5 or len(simulated) < max_lag + 5:
            return 1.0
        emp_acf = acf(empirical, nlags=max_lag, fft=True)
        sim_acf = acf(simulated, nlags=max_lag, fft=True)
        return float(np.sqrt(np.mean((emp_acf - sim_acf) ** 2)))

    def garch_comparison(self, empirical: np.ndarray,
                         simulated: np.ndarray) -> float:
        """Euclidean distance of GARCH(1,1) (alpha, beta) vectors."""
        if not ARCH_AVAILABLE:
            return 1.0

        def _fit(r: np.ndarray) -> np.ndarray:
            try:
                am = arch_model(r * 100, vol="Garch", p=1, q=1, mean="Zero")
                res = am.fit(disp="off", show_warning=False)
                return np.array([
                    res.params.get("alpha[1]", 0.0),
                    res.params.get("beta[1]", 0.0),
                ])
            except Exception:
                return np.array([0.0, 0.94])

        emp_p = _fit(empirical)
        sim_p = _fit(simulated)
        return float(np.linalg.norm(emp_p - sim_p))

    def vol_clustering_comparison(self, empirical: np.ndarray,
                                  simulated: np.ndarray,
                                  max_lag: int = 20) -> float:
        """RMSE between |returns| ACF vectors (volatility clustering proxy)."""
        if not STATSMODELS_AVAILABLE:
            return 1.0
        if len(empirical) < max_lag + 5 or len(simulated) < max_lag + 5:
            return 1.0
        emp_abs = acf(np.abs(empirical), nlags=max_lag, fft=True)
        sim_abs = acf(np.abs(simulated), nlags=max_lag, fft=True)
        return float(np.sqrt(np.mean((emp_abs - sim_abs) ** 2)))

    # ------------------------------------------------------------------
    # Scorecard
    # ------------------------------------------------------------------

    def compute_scorecard(self, model_name: str, asset: str,
                          empirical_returns: np.ndarray,
                          simulated_paths: np.ndarray) -> ValidationScore:
        """Compute a 0-10 composite scorecard for *simulated_paths*
        against *empirical_returns*."""

        # Flatten simulated paths to returns
        sim_returns = np.diff(np.log(np.maximum(simulated_paths, 1e-9)), axis=1).ravel()

        # Sub-sample to keep computation tractable
        max_samples = 50_000
        if len(sim_returns) > max_samples:
            rng = np.random.default_rng(42)
            sim_returns = rng.choice(sim_returns, max_samples, replace=False)

        ks = self.ks_test(empirical_returns, sim_returns)
        ad = self.anderson_darling_test(empirical_returns, sim_returns)
        acf_rmse = self.acf_comparison(empirical_returns, sim_returns)
        garch_dist = self.garch_comparison(empirical_returns, sim_returns)
        vol_clust = self.vol_clustering_comparison(empirical_returns, sim_returns)

        # Convert metrics to 0-10 scores
        ks_score = 10.0 * max(0.0, 1.0 - ks["ks_stat"])
        ad_score = 10.0 * min(max(ad["ad_pvalue"], 0.0), 1.0)
        acf_score = 10.0 * max(0.0, 1.0 - acf_rmse * 5.0)
        garch_score = 10.0 * max(0.0, 1.0 - garch_dist * 3.0)
        vol_clust_score = 10.0 * max(0.0, 1.0 - vol_clust * 5.0)

        w = self.SCORE_WEIGHTS
        composite = (
            w["ks"] * ks_score
            + w["ad"] * ad_score
            + w["acf"] * acf_score
            + w["garch"] * garch_score
            + w["vol_clust"] * vol_clust_score
        )
        composite = max(0.0, min(10.0, composite))

        return ValidationScore(
            model_name=model_name,
            asset=asset,
            ks_stat=ks["ks_stat"],
            ks_pvalue=ks["ks_pvalue"],
            ad_stat=ad["ad_stat"],
            ad_pvalue=ad["ad_pvalue"],
            acf_rmse=acf_rmse,
            garch_param_distance=garch_dist,
            vol_clustering_score=vol_clust,
            composite_score=composite,
            ks_score=ks_score,
            ad_score=ad_score,
            acf_score=acf_score,
            garch_score=garch_score,
            vol_clust_sub_score=vol_clust_score,
        )
