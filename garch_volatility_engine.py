"""
GARCH Volatility Forecasting Engine
Fits GARCH(1,1) and optionally EGARCH to return series.
Provides volatility ratio (forecast/historical) for position sizing
and dynamic threshold adjustment.
Falls back to EWMA (RiskMetrics) if 'arch' library is not installed.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from collections import deque

# Graceful import
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


class GARCHVolatilityEngine:
    """
    GARCH volatility forecasting with EWMA fallback.
    Provides vol_ratio for position sizing and dynamic thresholds.
    """

    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get("enabled", False)
        self.model_type = config.get("model_type", "GARCH")
        self.p = config.get("p", 1)
        self.q = config.get("q", 1)
        self.min_observations = config.get("min_observations", 20)
        self.refit_interval = config.get("refit_interval_cycles", 50)
        self.hist_vol_window = config.get("historical_vol_window", 10)
        self.forecast_horizon = config.get("forecast_horizon", 1)
        self.ewma_lambda = config.get("ewma_lambda", 0.94)
        self.position_impact = config.get("position_size_impact", 0.5)

        # Per-asset state
        self._returns: Dict[str, deque] = {}
        self._last_price: Dict[str, float] = {}
        self._models: Dict[str, Any] = {}
        self._forecasts: Dict[str, Dict[str, Any]] = {}
        self._fit_cycles: Dict[str, int] = {}
        self._cycle_count = 0

    @property
    def is_available(self) -> bool:
        return self.enabled

    def update_returns(self, product_id: str, price: float) -> None:
        """Compute and store log return from price update."""
        if not self.enabled or price <= 0:
            return

        if product_id in self._last_price:
            last = self._last_price[product_id]
            if last > 0:
                log_ret = np.log(price / last)
                if product_id not in self._returns:
                    self._returns[product_id] = deque(maxlen=2000)
                self._returns[product_id].append(log_ret)

        self._last_price[product_id] = price
        self._cycle_count += 1

    def should_refit(self, product_id: str) -> bool:
        """Check if model needs refitting for this asset."""
        if not self.enabled:
            return False
        if product_id not in self._returns:
            return False
        if len(self._returns[product_id]) < self.min_observations:
            return False
        last_fit = self._fit_cycles.get(product_id, 0)
        return (self._cycle_count - last_fit) >= self.refit_interval

    def fit_model(self, product_id: str) -> bool:
        """Fit GARCH model (or EWMA fallback) on stored returns."""
        if product_id not in self._returns:
            return False

        returns = np.array(self._returns[product_id])
        if len(returns) < self.min_observations:
            return False

        # Scale returns to percentage for arch library stability
        returns_pct = returns * 100.0

        fitted = False

        if ARCH_AVAILABLE:
            try:
                vol_model = self.model_type.upper()
                if vol_model == "EGARCH":
                    am = arch_model(returns_pct, vol=vol_model, p=self.p, q=self.q, mean="Zero")
                else:
                    am = arch_model(returns_pct, vol="Garch", p=self.p, q=self.q, mean="Zero")

                result = am.fit(disp="off", show_warning=False)
                self._models[product_id] = result
                fitted = True
                self._fit_cycles[product_id] = self._cycle_count
                self.logger.debug(f"GARCH({self.p},{self.q}) fitted for {product_id}")
            except Exception as e:
                self.logger.debug(f"GARCH fit failed for {product_id}: {e}")

        if not fitted:
            # EWMA fallback
            self._models[product_id] = "EWMA"
            self._fit_cycles[product_id] = self._cycle_count
            fitted = True

        return fitted

    def forecast_volatility(self, product_id: str) -> Dict[str, Any]:
        """
        Forecast next-period volatility and compute vol_ratio.
        Returns dict with forecast_vol, historical_vol, vol_ratio, vol_regime.
        """
        default = {
            "forecast_vol": 0.0,
            "historical_vol": 0.0,
            "vol_ratio": 1.0,
            "vol_regime": "stable",
            "confidence": 0.0,
            "model_type": "none",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not self.enabled or product_id not in self._returns:
            return default

        returns = np.array(self._returns[product_id])
        if len(returns) < self.hist_vol_window:
            return default

        # Historical volatility (annualized)
        hist_vol = float(np.std(returns[-self.hist_vol_window:]) * np.sqrt(252))

        model = self._models.get(product_id)

        forecast_vol = hist_vol  # Default
        model_type = "none"
        confidence = 0.3

        if model == "EWMA" or model is None:
            # EWMA variance: sigma2[t] = lambda * sigma2[t-1] + (1-lambda) * r[t-1]^2
            # Used both as explicit EWMA fallback and as pre-fit inline estimate
            ewma_var = returns[-1] ** 2
            for r in reversed(list(returns[:-1])[-50:]):
                ewma_var = self.ewma_lambda * ewma_var + (1 - self.ewma_lambda) * r ** 2
            forecast_vol = float(np.sqrt(ewma_var) * np.sqrt(252))
            model_type = "EWMA_inline" if model is None else "EWMA_fallback"
            confidence = 0.35 if model is None else 0.5
        elif ARCH_AVAILABLE and hasattr(model, "forecast"):
            try:
                fc = model.forecast(horizon=self.forecast_horizon, reindex=False)
                # Variance forecast is in percentage^2, convert back
                var_forecast = fc.variance.values[-1, 0]
                forecast_vol = float(np.sqrt(var_forecast / 10000.0) * np.sqrt(252))
                model_type = self.model_type
                confidence = 0.8
            except Exception:
                forecast_vol = hist_vol
                model_type = "GARCH_stale"
                confidence = 0.4

        # Vol ratio
        vol_ratio = forecast_vol / max(hist_vol, 1e-9) if hist_vol > 1e-9 else 1.0
        vol_ratio = float(np.clip(vol_ratio, 0.2, 5.0))

        # Regime classification
        if vol_ratio > 1.2:
            vol_regime = "expanding"
        elif vol_ratio < 0.8:
            vol_regime = "contracting"
        else:
            vol_regime = "stable"

        return {
            "forecast_vol": forecast_vol,
            "historical_vol": hist_vol,
            "vol_ratio": vol_ratio,
            "vol_regime": vol_regime,
            "confidence": confidence,
            "model_type": model_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_position_size_multiplier(self, product_id: str) -> float:
        """
        Returns multiplier in [0.5, 1.5] based on vol_ratio.
        Expanding vol -> reduce. Contracting -> increase.
        """
        if not self.enabled:
            return 1.0

        forecast = self.forecast_volatility(product_id)
        vol_ratio = forecast.get("vol_ratio", 1.0)

        # Linear interpolation with impact scaling
        if vol_ratio > 1.2:
            # Expanding: reduce position
            reduction = min((vol_ratio - 1.0) * self.position_impact, 0.5)
            return max(0.5, 1.0 - reduction)
        elif vol_ratio < 0.8:
            # Contracting: can increase
            increase = min((1.0 - vol_ratio) * self.position_impact, 0.5)
            return min(1.5, 1.0 + increase)
        else:
            return 1.0

    def get_dynamic_threshold_adjustment(self, product_id: str) -> Tuple[float, float]:
        """
        Returns (buy_threshold_delta, sell_threshold_delta).
        Expanding vol widens thresholds, contracting narrows.
        """
        if not self.enabled:
            return 0.0, 0.0

        forecast = self.forecast_volatility(product_id)
        vol_ratio = forecast.get("vol_ratio", 1.0)

        if vol_ratio > 1.2:
            delta = min((vol_ratio - 1.0) * 0.05, 0.05)
            return delta, -delta  # Widen: buy higher, sell lower
        elif vol_ratio < 0.8:
            delta = min((1.0 - vol_ratio) * 0.03, 0.02)
            return -delta, delta  # Narrow: buy easier, sell easier
        return 0.0, 0.0
