"""
Multi-Horizon Probability Estimator (MHPE)
============================================
Orchestrator that combines three sub-predictors into a unified
probability cone for each position.

Horizon routing:
  1s, 5s       → MicrostructurePredictor only
  30s          → Blend: 30% Micro + 70% Statistical
  2m (120s)    → StatisticalPredictor only
  5m (300s)    → Blend: 40% Statistical + 60% Regime
  15m, 60m     → RegimePredictor only

Called by PositionReEvaluator (Doc 10) every evaluation cycle.
Returns ProbabilityCone with ConeAnalysis.

Total compute budget: < 5ms per position (all 7 horizons combined).
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

from core.data_structures import (
    ConeAnalysis,
    HorizonEstimate,
    ProbabilityCone,
)

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = [1, 5, 30, 120, 300, 900, 3600]


class MultiHorizonEstimator:
    """Produces probability cones by combining three sub-predictors."""

    def __init__(
        self,
        config: dict,
        micro_predictor: Any = None,
        stat_predictor: Any = None,
        regime_predictor: Any = None,
    ):
        self.config = config
        self.micro = micro_predictor
        self.stat = stat_predictor
        self.regime = regime_predictor
        self.horizons: List[int] = config.get("horizons", DEFAULT_HORIZONS)

        # Metrics
        self._total_estimates = 0
        self._total_time_ms = 0.0

        logger.info(
            "MHPE init  horizons=%s  micro=%s  stat=%s  regime=%s",
            self.horizons,
            "ok" if micro_predictor else "none",
            "ok" if stat_predictor else "none",
            "ok" if regime_predictor else "none",
        )

    # ═════════════════════════════════════════════════════════════
    # MAIN ENTRY — Full probability cone for one position
    # ═════════════════════════════════════════════════════════════

    def estimate(self, position: Any) -> ProbabilityCone:
        start = time.time()
        pair = position.pair
        side = position.side

        horizon_estimates: List[HorizonEstimate] = []
        for h in self.horizons:
            est = self._estimate_horizon(pair, side, h)
            horizon_estimates.append(est)

        analysis = self._analyze_cone(position, horizon_estimates)
        computation_time_ms = (time.time() - start) * 1000

        self._total_estimates += 1
        self._total_time_ms += computation_time_ms

        return ProbabilityCone(
            position_id=position.position_id,
            pair=pair,
            side=side,
            timestamp=time.time(),
            computation_time_ms=computation_time_ms,
            horizons=horizon_estimates,
            peak_ev_horizon_seconds=analysis["peak_ev_horizon"],
            peak_ev_bps=analysis["peak_ev_bps"],
            ev_zero_crossing_seconds=analysis["ev_zero_crossing"],
            optimal_hold_remaining_seconds=analysis["optimal_hold_remaining"],
            recommended_action=analysis["recommended_action"],
            action_urgency=analysis["urgency"],
            max_horizon_with_positive_ev=analysis["max_positive_ev_horizon"],
            worst_case_5min_bps=analysis["worst_case_5min"],
            worst_case_15min_bps=analysis["worst_case_15min"],
        )

    # ═════════════════════════════════════════════════════════════
    # HORIZON ROUTING
    # ═════════════════════════════════════════════════════════════

    def _estimate_horizon(self, pair: str, side: str, horizon_seconds: int) -> HorizonEstimate:
        """Route to the correct predictor(s) and optionally blend."""

        if horizon_seconds <= 5:
            if self.micro:
                return self.micro.predict(pair, side, horizon_seconds)
            return self._neutral_estimate(horizon_seconds)

        elif horizon_seconds == 30:
            if self.micro and self.stat:
                micro_est = self.micro.predict(pair, side, horizon_seconds)
                stat_est = self.stat.predict(pair, side, horizon_seconds)
                return self._blend_estimates(micro_est, stat_est, 0.3, 0.7)
            if self.stat:
                return self.stat.predict(pair, side, horizon_seconds)
            if self.micro:
                return self.micro.predict(pair, side, horizon_seconds)
            return self._neutral_estimate(horizon_seconds)

        elif horizon_seconds <= 120:
            if self.stat:
                return self.stat.predict(pair, side, horizon_seconds)
            return self._neutral_estimate(horizon_seconds)

        elif horizon_seconds == 300:
            if self.stat and self.regime:
                stat_est = self.stat.predict(pair, side, horizon_seconds)
                regime_est = self.regime.predict(pair, side, horizon_seconds)
                return self._blend_estimates(stat_est, regime_est, 0.4, 0.6)
            if self.regime:
                return self.regime.predict(pair, side, horizon_seconds)
            if self.stat:
                return self.stat.predict(pair, side, horizon_seconds)
            return self._neutral_estimate(horizon_seconds)

        else:
            # 15m, 60m → pure regime
            if self.regime:
                return self.regime.predict(pair, side, horizon_seconds)
            return self._neutral_estimate(horizon_seconds)

    @staticmethod
    def _neutral_estimate(horizon_seconds: int) -> HorizonEstimate:
        """Fallback when no predictor is available."""
        label = f"{horizon_seconds}s" if horizon_seconds < 60 else f"{horizon_seconds // 60}m"
        return HorizonEstimate(
            horizon_seconds=horizon_seconds,
            horizon_label=label,
            p_profit=0.50,
            p_loss=0.50,
            e_favorable_bps=0.0,
            e_adverse_bps=0.0,
            e_net_bps=0.0,
            sigma_bps=0.0,
            p_adverse_10bps=0.0,
            p_adverse_25bps=0.0,
            p_adverse_50bps=0.0,
            estimate_confidence=0.0,
            dominant_signal="none",
        )

    @staticmethod
    def _blend_estimates(
        est_a: HorizonEstimate,
        est_b: HorizonEstimate,
        weight_a: float,
        weight_b: float,
    ) -> HorizonEstimate:
        return HorizonEstimate(
            horizon_seconds=est_a.horizon_seconds,
            horizon_label=est_a.horizon_label,
            p_profit=round(est_a.p_profit * weight_a + est_b.p_profit * weight_b, 4),
            p_loss=round(est_a.p_loss * weight_a + est_b.p_loss * weight_b, 4),
            e_favorable_bps=round(est_a.e_favorable_bps * weight_a + est_b.e_favorable_bps * weight_b, 2),
            e_adverse_bps=round(est_a.e_adverse_bps * weight_a + est_b.e_adverse_bps * weight_b, 2),
            e_net_bps=round(est_a.e_net_bps * weight_a + est_b.e_net_bps * weight_b, 2),
            sigma_bps=round(est_a.sigma_bps * weight_a + est_b.sigma_bps * weight_b, 2),
            p_adverse_10bps=round(est_a.p_adverse_10bps * weight_a + est_b.p_adverse_10bps * weight_b, 4),
            p_adverse_25bps=round(est_a.p_adverse_25bps * weight_a + est_b.p_adverse_25bps * weight_b, 4),
            p_adverse_50bps=round(est_a.p_adverse_50bps * weight_a + est_b.p_adverse_50bps * weight_b, 4),
            estimate_confidence=round(est_a.estimate_confidence * weight_a + est_b.estimate_confidence * weight_b, 3),
            dominant_signal=est_a.dominant_signal if weight_a > weight_b else est_b.dominant_signal,
        )

    # ═════════════════════════════════════════════════════════════
    # CONE ANALYSIS
    # ═════════════════════════════════════════════════════════════

    def _analyze_cone(self, position: Any, horizons: List[HorizonEstimate]) -> dict:
        # Peak EV horizon
        peak_idx = 0
        peak_ev = horizons[0].e_net_bps
        for i, h in enumerate(horizons):
            if h.e_net_bps > peak_ev:
                peak_ev = h.e_net_bps
                peak_idx = i
        peak_horizon = horizons[peak_idx].horizon_seconds

        # Where EV turns negative
        ev_zero_crossing = horizons[-1].horizon_seconds
        for i, h in enumerate(horizons):
            if h.e_net_bps < 0 and i > 0:
                prev = horizons[i - 1]
                if prev.e_net_bps > 0:
                    denom = prev.e_net_bps - h.e_net_bps
                    frac = prev.e_net_bps / denom if denom != 0 else 0
                    ev_zero_crossing = int(
                        prev.horizon_seconds + frac * (h.horizon_seconds - prev.horizon_seconds)
                    )
                    break

        # Optimal hold remaining
        time_held = position.age_seconds
        optimal_hold_remaining = max(0, peak_horizon - time_held)

        # Max positive EV horizon
        max_positive = 0
        for h in horizons:
            if h.e_net_bps > 0:
                max_positive = h.horizon_seconds

        # Worst-case scenarios (95th percentile ≈ 1.65 sigma)
        worst_5m = 0.0
        worst_15m = 0.0
        for h in horizons:
            if h.horizon_seconds == 300:
                worst_5m = h.sigma_bps * 1.65
            if h.horizon_seconds == 900:
                worst_15m = h.sigma_bps * 1.65

        action, urgency = self._determine_action(
            position, horizons, peak_horizon, peak_ev,
            ev_zero_crossing, optimal_hold_remaining,
        )

        return {
            "peak_ev_horizon": peak_horizon,
            "peak_ev_bps": round(peak_ev, 2),
            "ev_zero_crossing": ev_zero_crossing,
            "optimal_hold_remaining": int(optimal_hold_remaining),
            "max_positive_ev_horizon": max_positive,
            "worst_case_5min": round(worst_5m, 2),
            "worst_case_15min": round(worst_15m, 2),
            "recommended_action": action,
            "urgency": urgency,
        }

    def _determine_action(
        self, position: Any, horizons: List[HorizonEstimate],
        peak_horizon: int, peak_ev: float,
        ev_zero_crossing: int, optimal_hold_remaining: float,
    ) -> tuple:
        time_held = position.age_seconds

        # CASE 1: EV negative at ALL horizons
        if all(h.e_net_bps <= 0 for h in horizons):
            return "close_now", "high"

        # CASE 2: EV negative at shortest horizon
        if horizons[0].e_net_bps <= 0:
            if any(h.e_net_bps > 0 for h in horizons[2:]):
                if len(horizons) > 1 and horizons[1].p_adverse_25bps > 0.15:
                    return "close_now", "medium"
                return "hold_through_dip", "low"
            return "close_now", "high"

        # CASE 3: Past the peak EV horizon
        if time_held > peak_horizon:
            overhold = time_held - peak_horizon
            if overhold > peak_horizon * 0.5:
                return "close_now", "medium"
            elif overhold > 10:
                return "trim_or_close", "low"
            return "hold_to_peak", "low"

        # CASE 4: Approaching peak
        if optimal_hold_remaining <= 5:
            return "close_at_peak", "medium"
        if optimal_hold_remaining <= 30:
            return "hold_to_peak", "low"

        # CASE 5: Peak is far away, near-term EV positive
        if optimal_hold_remaining > 60 and horizons[0].e_net_bps > 0:
            if len(horizons) > 2 and horizons[2].p_adverse_25bps > 0.10:
                return "hold_with_caution", "low"
            return "hold_confident", "none"

        # CASE 6: EV turns negative before signal TTL
        signal_remaining = position.signal_ttl_seconds - time_held
        if ev_zero_crossing < signal_remaining:
            if ev_zero_crossing - time_held < 30:
                return "close_at_peak", "medium"
            return "hold_to_peak", "low"

        return "hold_confident", "none"

    # ═════════════════════════════════════════════════════════════
    # HIGH-LEVEL INTERFACE FOR RE-EVALUATOR (Doc 10)
    # ═════════════════════════════════════════════════════════════

    def analyze_for_reevaluator(self, position: Any) -> ConeAnalysis:
        """
        Returns a ConeAnalysis the re-evaluator can directly consume
        to inform hold/trim/close/add decisions.
        """
        cone = self.estimate(position)

        action_to_multiplier = {
            "close_now": 0.0,
            "trim_or_close": 0.3,
            "close_at_peak": 0.0,
            "hold_to_peak": 1.0,
            "hold_through_dip": 1.0,
            "hold_with_caution": 0.8,
            "hold_confident": 1.0,
        }
        size_multiplier = action_to_multiplier.get(cone.recommended_action, 1.0)

        # Risk flags
        tail_warning = any(
            h.horizon_seconds <= 300 and h.p_adverse_25bps > 0.10
            for h in cone.horizons
        )
        regime_risk = cone.horizons[-1].estimate_confidence < 0.15 if cone.horizons else False
        vol_expanding = False  # TODO: compare sigma growth vs sqrt(time)

        # Edge distribution
        short_ev = sum(h.e_net_bps for h in cone.horizons[:3])
        long_ev = sum(h.e_net_bps for h in cone.horizons[3:])
        total_ev = short_ev + long_ev

        edge_front_loaded = (short_ev > long_ev * 2) if long_ev != 0 else True
        edge_back_loaded = (long_ev > short_ev * 2 and long_ev > 0)
        edge_uniform = abs(short_ev - long_ev) < abs(total_ev) * 0.3 if total_ev != 0 else True

        return ConeAnalysis(
            position_id=position.position_id,
            optimal_hold_seconds=cone.optimal_hold_remaining_seconds,
            close_by_seconds=max(0, cone.ev_zero_crossing_seconds - int(position.age_seconds)),
            urgency=cone.action_urgency,
            size_multiplier=size_multiplier,
            reason=(
                f"{cone.recommended_action}: peak EV {cone.peak_ev_bps:.2f}bps "
                f"at {cone.peak_ev_horizon_seconds}s, "
                f"EV crosses zero at {cone.ev_zero_crossing_seconds}s"
            ),
            tail_risk_warning=tail_warning,
            volatility_expanding=vol_expanding,
            regime_transition_risk=regime_risk,
            edge_front_loaded=edge_front_loaded,
            edge_back_loaded=edge_back_loaded,
            edge_uniformly_distributed=edge_uniform,
        )

    # ═════════════════════════════════════════════════════════════
    # METRICS
    # ═════════════════════════════════════════════════════════════

    def get_metrics(self) -> dict:
        avg_ms = (self._total_time_ms / self._total_estimates) if self._total_estimates else 0
        return {
            "total_estimates": self._total_estimates,
            "avg_computation_ms": round(avg_ms, 2),
            "horizons": self.horizons,
            "predictors": {
                "micro": self.micro is not None,
                "stat": self.stat is not None,
                "regime": self.regime is not None,
            },
        }
