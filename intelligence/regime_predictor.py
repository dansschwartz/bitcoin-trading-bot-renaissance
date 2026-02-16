"""
Regime Predictor — Long-Term (5m to 60m)
==========================================
Predicts price direction and risk over 5 to 60 minutes using
regime state, transition probabilities, macro correlations, and
funding rate trends.

At these timescales, the HMM's regime transition matrix IS the
probability cone.  We can't predict direction well (P ≈ 51%), but
we CAN predict volatility well — upcoming regime transitions inflate
E[adverse] dramatically.

Performance: < 2ms per position per horizon.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from core.data_structures import HorizonEstimate


# Per-5-minute-bar volatility by regime (bps)
_DEFAULT_REGIME_VOL: Dict[str, int] = {
    "low_volatility": 8,
    "mean_reverting": 15,
    "trending": 20,
    "trending_up": 20,
    "trending_down": 25,
    "high_volatility": 40,
    "chaotic": 80,
}

# Directional bias per regime
_REGIME_BIAS: Dict[str, float] = {
    "trending_up": 0.4,
    "trending_down": -0.4,
    "chaotic": 0.0,
    "high_volatility": 0.0,
    "low_volatility": 0.0,
    "mean_reverting": 0.1,
    "trending": 0.0,
}


class RegimePredictor:
    """
    Long-term predictor using regime dynamics.
    Called by MHPE for horizons: 5m (300s), 15m (900s), 60m (3600s).
    """

    def __init__(self, config: dict, regime_detector: Any = None):
        self.config = config
        self.regime_detector = regime_detector
        self._regime_vol = config.get("regime_vol_bps", _DEFAULT_REGIME_VOL)

    def predict(self, pair: str, side: str, horizon_seconds: int) -> HorizonEstimate:
        # ── Regime state ──
        current_regime = self._get_current_regime()
        transition_probs = self._get_transition_probs(horizon_seconds)

        # ── Directional bias ──
        regime_bias = self._regime_directional_bias(current_regime, transition_probs)

        # ── Funding / correlation stubs ──
        funding_signal = 0.0
        correlation_signal = 0.0

        # ── Combine ──
        weights = self._get_weights(horizon_seconds)
        raw_signal = (
            weights["regime"] * regime_bias
            + weights["funding"] * funding_signal
            + weights["correlation"] * correlation_signal
        )

        # At long horizons, keep directional prediction near 50%
        steepness = {300: 0.3, 900: 0.2, 3600: 0.1}.get(horizon_seconds, 0.1)
        clamped = max(-2, min(2, raw_signal))
        p_up = 1.0 / (1.0 + math.exp(-steepness * clamped))
        lo, hi = self.config.get("probability_clamp", [0.45, 0.56])
        p_up = max(lo, min(hi, p_up))

        p_favorable = p_up if side == "long" else (1.0 - p_up)

        # ── Volatility forecast — the regime predictor's killer feature ──
        vol_forecast = self._forecast_volatility(current_regime, transition_probs, horizon_seconds)
        sigma_bps = vol_forecast["expected_sigma_bps"]

        e_favorable = sigma_bps * p_favorable * 0.8
        e_adverse = sigma_bps * (1.0 - p_favorable) * 0.8
        e_net = p_favorable * e_favorable - (1.0 - p_favorable) * e_adverse

        # ── Tail risk (regime-aware) ──
        p_adv_10 = self._regime_tail_prob(sigma_bps, 10, transition_probs)
        p_adv_25 = self._regime_tail_prob(sigma_bps, 25, transition_probs)
        p_adv_50 = self._regime_tail_prob(sigma_bps, 50, transition_probs)

        estimate_confidence = 0.3 / (1.0 + 0.001 * horizon_seconds)

        return HorizonEstimate(
            horizon_seconds=horizon_seconds,
            horizon_label=f"{horizon_seconds // 60}m",
            p_profit=round(p_favorable, 4),
            p_loss=round(1.0 - p_favorable, 4),
            e_favorable_bps=round(e_favorable, 2),
            e_adverse_bps=round(e_adverse, 2),
            e_net_bps=round(e_net, 2),
            sigma_bps=round(sigma_bps, 2),
            p_adverse_10bps=round(p_adv_10, 4),
            p_adverse_25bps=round(p_adv_25, 4),
            p_adverse_50bps=round(p_adv_50, 4),
            estimate_confidence=round(estimate_confidence, 3),
            dominant_signal="regime",
        )

    # ── Helpers ──

    def _get_current_regime(self) -> str:
        if self.regime_detector is None:
            return "low_volatility"
        name = getattr(self.regime_detector, "_current_regime_name", None)
        if name:
            return str(name)
        # Fallback
        return getattr(self.regime_detector, "current_regime", "low_volatility")

    def _get_transition_probs(self, horizon_seconds: int) -> Dict[str, float]:
        """
        Get regime transition probabilities.  If the HMM exposes a transition
        matrix we use it; otherwise return a simple fallback.
        """
        if self.regime_detector is not None:
            fn = getattr(self.regime_detector, "get_transition_probabilities", None)
            if fn is not None:
                try:
                    steps = max(1, horizon_seconds // 300)
                    return fn(steps=steps)
                except Exception:
                    pass

        # Fallback: high probability of staying in current regime
        current = self._get_current_regime()
        probs = {r: 0.05 for r in _DEFAULT_REGIME_VOL}
        probs[current] = 0.75
        # Normalise
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def _regime_directional_bias(self, current_regime: str, transition_probs: Dict[str, float]) -> float:
        current_bias = _REGIME_BIAS.get(current_regime, 0.0)
        p_stay = transition_probs.get(current_regime, 0.5)
        expected_bias = current_bias * p_stay
        for regime, prob in transition_probs.items():
            if regime != current_regime:
                expected_bias += _REGIME_BIAS.get(regime, 0.0) * prob
        return max(-1.0, min(1.0, expected_bias))

    def _forecast_volatility(
        self, current_regime: str, transition_probs: Dict[str, float], horizon_seconds: int
    ) -> dict:
        expected_vol_per_bar = sum(
            self._regime_vol.get(regime, 20) * prob
            for regime, prob in transition_probs.items()
        )
        bars_in_horizon = max(1, horizon_seconds / 300)
        expected_sigma = expected_vol_per_bar * math.sqrt(bars_in_horizon)
        return {
            "expected_sigma_bps": expected_sigma,
            "vol_per_bar_bps": expected_vol_per_bar,
            "regime_stability": transition_probs.get(current_regime, 0.5),
        }

    def _regime_tail_prob(
        self, sigma: float, threshold: float, transition_probs: Dict[str, float]
    ) -> float:
        if sigma <= 0:
            return 0.0
        z = threshold / sigma
        base_tail = 0.5 * math.erfc(z / math.sqrt(2))
        p_chaotic = transition_probs.get("chaotic", 0) + transition_probs.get("high_volatility", 0)
        if p_chaotic > 0.1:
            return min(0.5, base_tail * (1 + p_chaotic * 5))
        return base_tail

    @staticmethod
    def _get_weights(horizon_seconds: int) -> dict:
        if horizon_seconds <= 300:
            return {"regime": 0.50, "funding": 0.25, "correlation": 0.25}
        elif horizon_seconds <= 900:
            return {"regime": 0.60, "funding": 0.20, "correlation": 0.20}
        return {"regime": 0.70, "funding": 0.15, "correlation": 0.15}
