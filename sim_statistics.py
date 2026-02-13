"""Statistical analysis module: distribution, volatility, tail, and autocorrelation."""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

try:
    from scipy import stats as sp_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


class SimStatistics:
    """Compute statistical properties of any return series.

    Four analysis groups:
    - distribution_stats (mean, std, skew, kurtosis, Jarque-Bera)
    - volatility_properties (annualised vol, GARCH persistence, vol-of-vol)
    - tail_properties (VaR, CVaR, max drawdown, Hill tail index)
    - autocorrelation_structure (returns ACF, |returns| ACF, Ljung-Box)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    def distribution_stats(self, returns: np.ndarray) -> Dict[str, float]:
        result: Dict[str, float] = {
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns, ddof=1)),
            "skewness": 0.0,
            "kurtosis": 0.0,
            "jarque_bera_stat": 0.0,
            "jarque_bera_pvalue": 1.0,
        }
        if SCIPY_AVAILABLE:
            result["skewness"] = float(sp_stats.skew(returns))
            result["kurtosis"] = float(sp_stats.kurtosis(returns))  # excess
            jb_stat, jb_pval = sp_stats.jarque_bera(returns)
            result["jarque_bera_stat"] = float(jb_stat)
            result["jarque_bera_pvalue"] = float(jb_pval)
        else:
            n = len(returns)
            m = np.mean(returns)
            s = np.std(returns, ddof=0)
            if s > 0 and n > 3:
                result["skewness"] = float(np.mean(((returns - m) / s) ** 3))
                result["kurtosis"] = float(np.mean(((returns - m) / s) ** 4) - 3)
        return result

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def volatility_properties(self, returns: np.ndarray) -> Dict[str, float]:
        annualised_vol = float(np.std(returns, ddof=1) * np.sqrt(252))

        result: Dict[str, float] = {
            "annualized_vol": annualised_vol,
            "garch_omega": 0.0,
            "garch_alpha": 0.0,
            "garch_beta": 0.0,
            "garch_persistence": 0.0,
            "vol_of_vol": 0.0,
        }

        # GARCH(1,1)
        if ARCH_AVAILABLE and len(returns) >= 50:
            try:
                am = arch_model(returns * 100, vol="Garch", p=1, q=1, mean="Zero")
                res = am.fit(disp="off", show_warning=False)
                result["garch_omega"] = float(res.params.get("omega", 0.0))
                result["garch_alpha"] = float(res.params.get("alpha[1]", 0.0))
                result["garch_beta"] = float(res.params.get("beta[1]", 0.0))
                result["garch_persistence"] = result["garch_alpha"] + result["garch_beta"]
            except Exception as e:
                self.logger.debug(f"GARCH fit failed, using EWMA: {e}")
                self._ewma_vol(returns, result)
        elif len(returns) >= 50:
            self._ewma_vol(returns, result)

        # Vol-of-vol
        if len(returns) >= 30:
            rolling_vol = pd.Series(returns).rolling(20, min_periods=5).std().dropna()
            result["vol_of_vol"] = float(np.std(rolling_vol))

        return result

    @staticmethod
    def _ewma_vol(returns: np.ndarray, result: Dict[str, float],
                  lam: float = 0.94) -> None:
        """EWMA fallback for vol estimation."""
        var = np.var(returns)
        for r in returns:
            var = lam * var + (1 - lam) * r ** 2
        result["garch_persistence"] = lam

    # ------------------------------------------------------------------
    # Tail properties
    # ------------------------------------------------------------------

    def tail_properties(self, returns: np.ndarray,
                        confidence: float = 0.05) -> Dict[str, float]:
        var = float(np.quantile(returns, confidence))
        tail_mask = returns <= var
        cvar = float(np.mean(returns[tail_mask])) if tail_mask.any() else var

        # Max drawdown
        cum = np.cumsum(returns)
        running_max = np.maximum.accumulate(cum)
        drawdowns = cum - running_max
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Hill tail index estimator
        sorted_abs = np.sort(np.abs(returns))[::-1]
        k = max(int(len(sorted_abs) * 0.10), 5)
        top_k = sorted_abs[:k]
        threshold = sorted_abs[min(k, len(sorted_abs) - 1)]
        if threshold > 0 and k > 0:
            hill_est = float((1.0 / k) * np.sum(np.log(top_k / threshold + 1e-15)))
            tail_index = 1.0 / hill_est if hill_est > 0 else float("inf")
        else:
            tail_index = float("inf")

        return {
            "var_5pct": var,
            "cvar_5pct": cvar,
            "max_drawdown": max_drawdown,
            "hill_tail_index": tail_index,
        }

    # ------------------------------------------------------------------
    # Autocorrelation
    # ------------------------------------------------------------------

    def autocorrelation_structure(self, returns: np.ndarray,
                                  max_lag: int = 20) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "returns_acf": [],
            "abs_returns_acf": [],
            "ljung_box_10_stat": 0.0,
            "ljung_box_10_pvalue": 1.0,
            "ljung_box_20_stat": 0.0,
            "ljung_box_20_pvalue": 1.0,
        }

        if STATSMODELS_AVAILABLE and len(returns) > max_lag + 5:
            r_acf = acf(returns, nlags=max_lag, fft=True)
            abs_acf = acf(np.abs(returns), nlags=max_lag, fft=True)
            result["returns_acf"] = r_acf.tolist()
            result["abs_returns_acf"] = abs_acf.tolist()

            try:
                lb = acorr_ljungbox(returns, lags=[10, 20], return_df=True)
                result["ljung_box_10_stat"] = float(lb["lb_stat"].iloc[0])
                result["ljung_box_10_pvalue"] = float(lb["lb_pvalue"].iloc[0])
                if len(lb) > 1:
                    result["ljung_box_20_stat"] = float(lb["lb_stat"].iloc[1])
                    result["ljung_box_20_pvalue"] = float(lb["lb_pvalue"].iloc[1])
            except Exception:
                pass
        elif len(returns) > max_lag + 5:
            # Manual ACF fallback
            m = np.mean(returns)
            r = returns - m
            var = np.var(returns)
            if var > 0:
                acf_vals = [float(np.mean(r[lag:] * r[:-lag]) / var)
                            if lag > 0 else 1.0 for lag in range(max_lag + 1)]
                result["returns_acf"] = acf_vals
                abs_r = np.abs(returns) - np.mean(np.abs(returns))
                abs_var = np.var(np.abs(returns))
                if abs_var > 0:
                    abs_acf_vals = [float(np.mean(abs_r[lag:] * abs_r[:-lag]) / abs_var)
                                    if lag > 0 else 1.0 for lag in range(max_lag + 1)]
                    result["abs_returns_acf"] = abs_acf_vals

        return result

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------

    def full_analysis(self, returns: np.ndarray) -> Dict[str, Any]:
        """Run all four analyses and return a combined dict."""
        out: Dict[str, Any] = {}
        out["distribution"] = self.distribution_stats(returns)
        out["volatility"] = self.volatility_properties(returns)
        out["tail"] = self.tail_properties(returns)
        out["autocorrelation"] = self.autocorrelation_structure(returns)
        return out
