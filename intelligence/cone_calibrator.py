"""
Cone Calibrator — Measures and corrects MHPE prediction accuracy
=================================================================
When we say P(profit) = 60%, it should actually be profitable 60%
of the time.  The calibrator runs daily, buckets predictions by
probability decile, computes actual win rate, and feeds correction
factors back to sub-predictors.

Without calibration, the system's confidence drifts over time as
market conditions change, leading to bad sizing decisions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from intelligence.multi_horizon_estimator import DEFAULT_HORIZONS

logger = logging.getLogger(__name__)


class ConeCalibrator:
    """Runs daily. Analyzes prediction accuracy. Updates calibration weights."""

    def __init__(self, config: dict, devil_tracker: Any = None):
        self.config = config
        self.devil_tracker = devil_tracker
        self.calibration_data: Dict[int, Dict[float, float]] = {}
        self._min_samples = config.get("min_samples_per_bucket", 5)
        self._min_positions = config.get("min_positions_to_calibrate", 20)

    def run_daily_calibration(self) -> Dict[int, Dict[float, float]]:
        """
        Analyze recent exits and compute per-horizon calibration corrections.

        Returns: {horizon_seconds: {predicted_bucket: correction_factor}}
        """
        if self.devil_tracker is None:
            logger.debug("ConeCalibrator: no devil_tracker, skipping")
            return {}

        exits = self._get_recent_exits()
        if not exits:
            return {}

        for horizon in DEFAULT_HORIZONS:
            eligible = [
                e for e in exits
                if e.get("hold_time_seconds", 0) >= horizon
            ]
            if len(eligible) < self._min_positions:
                continue

            buckets: Dict[float, Dict[str, List[float]]] = {}
            for e in eligible:
                cone_preds = e.get("cone_predictions", {})
                cone_at_h = cone_preds.get(str(horizon))
                if not cone_at_h:
                    continue

                predicted_p = cone_at_h.get("p_profit", 0.5)
                actual_profitable = 1.0 if e.get("pnl_bps", 0) > 0 else 0.0

                bucket = round(predicted_p, 1)
                if bucket not in buckets:
                    buckets[bucket] = {"predicted": [], "actual": []}
                buckets[bucket]["predicted"].append(predicted_p)
                buckets[bucket]["actual"].append(actual_profitable)

            corrections: Dict[float, float] = {}
            for bucket, data in buckets.items():
                if len(data["actual"]) < self._min_samples:
                    corrections[bucket] = 1.0
                    continue
                avg_predicted = sum(data["predicted"]) / len(data["predicted"])
                avg_actual = sum(data["actual"]) / len(data["actual"])
                corrections[bucket] = avg_actual / avg_predicted if avg_predicted > 0 else 1.0

            self.calibration_data[horizon] = corrections

        logger.info(
            "ConeCalibrator: calibrated %d horizons from %d exits",
            len(self.calibration_data), len(exits),
        )
        return self.calibration_data

    def get_correction(self, horizon: int, predicted_p: float) -> float:
        """Get corrected probability for a given horizon and prediction."""
        corrections = self.calibration_data.get(horizon, {})
        if not corrections:
            return predicted_p
        bucket = round(predicted_p, 1)
        factor = corrections.get(bucket, 1.0)
        corrected = predicted_p * factor
        return max(0.40, min(0.70, corrected))

    def _get_recent_exits(self, hours: int = 24) -> List[Dict]:
        """Pull recent exits from DevilTracker."""
        fn = getattr(self.devil_tracker, "get_recent_exits", None)
        if fn is not None:
            try:
                return fn(hours=hours)
            except Exception:
                pass
        # Fallback: query the reeval_events table if available
        fn2 = getattr(self.devil_tracker, "get_reeval_effectiveness_report", None)
        if fn2 is not None:
            try:
                report = fn2()
                return report.get("exits", [])
            except Exception:
                pass
        return []

    def get_metrics(self) -> dict:
        return {
            "horizons_calibrated": len(self.calibration_data),
            "calibration_data": {
                str(h): {str(k): round(v, 3) for k, v in corr.items()}
                for h, corr in self.calibration_data.items()
            },
        }
