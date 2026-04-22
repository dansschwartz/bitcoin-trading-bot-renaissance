"""
Tests for intelligence/cone_calibrator.py — ConeCalibrator
============================================================
Covers daily accuracy tracking, per-horizon correction factors,
bucket computation, and edge cases (no tracker, no exits, etc.).
"""

import pytest
from unittest.mock import MagicMock, patch

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.cone_calibrator import ConeCalibrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    return {
        "min_samples_per_bucket": 2,
        "min_positions_to_calibrate": 3,
    }


def _make_exit(hold_time, cone_horizon, predicted_p, pnl_bps):
    """Helper to build a realistic exit dict."""
    return {
        "hold_time_seconds": hold_time,
        "cone_predictions": {
            str(cone_horizon): {"p_profit": predicted_p},
        },
        "pnl_bps": pnl_bps,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConeCalibrator:

    def test_init_stores_config(self, default_config):
        cal = ConeCalibrator(default_config)
        assert cal._min_samples == 2
        assert cal._min_positions == 3
        assert cal.calibration_data == {}

    def test_run_daily_calibration_no_tracker_returns_empty(self, default_config):
        cal = ConeCalibrator(default_config, devil_tracker=None)
        result = cal.run_daily_calibration()
        assert result == {}

    def test_run_daily_calibration_no_exits_returns_empty(self, default_config):
        tracker = MagicMock()
        tracker.get_recent_exits.return_value = []
        cal = ConeCalibrator(default_config, devil_tracker=tracker)
        result = cal.run_daily_calibration()
        assert result == {}

    def test_run_daily_calibration_insufficient_exits(self, default_config):
        """Fewer exits than min_positions_to_calibrate -> skip that horizon."""
        exits = [
            _make_exit(hold_time=10, cone_horizon=1, predicted_p=0.6, pnl_bps=5),
            _make_exit(hold_time=10, cone_horizon=1, predicted_p=0.7, pnl_bps=-3),
        ]
        tracker = MagicMock()
        tracker.get_recent_exits.return_value = exits
        cal = ConeCalibrator(default_config, devil_tracker=tracker)
        result = cal.run_daily_calibration()
        # 2 exits < min_positions_to_calibrate=3, so no horizons calibrated
        assert result == {}

    def test_run_daily_calibration_happy_path(self, default_config):
        """Provide enough exits with cone predictions to produce corrections."""
        # horizon=1, all exits profitable at predicted_p ~ 0.6
        exits = [
            _make_exit(hold_time=5, cone_horizon=1, predicted_p=0.60, pnl_bps=10),
            _make_exit(hold_time=5, cone_horizon=1, predicted_p=0.60, pnl_bps=5),
            _make_exit(hold_time=5, cone_horizon=1, predicted_p=0.60, pnl_bps=8),
            _make_exit(hold_time=5, cone_horizon=1, predicted_p=0.40, pnl_bps=-2),
            _make_exit(hold_time=5, cone_horizon=1, predicted_p=0.40, pnl_bps=-3),
        ]
        tracker = MagicMock()
        tracker.get_recent_exits.return_value = exits
        cal = ConeCalibrator(default_config, devil_tracker=tracker)
        result = cal.run_daily_calibration()

        # Should have corrections for horizon=1
        assert 1 in result
        corrections = result[1]
        # Bucket 0.6 -> all 3 profitable -> actual avg = 1.0, correction = 1.0/0.6 ~= 1.667
        assert 0.6 in corrections
        assert corrections[0.6] == pytest.approx(1.0 / 0.6, rel=0.01)

        # Bucket 0.4 -> both unprofitable -> actual avg = 0.0, correction = 0/0.4 = 0.0
        assert 0.4 in corrections
        assert corrections[0.4] == pytest.approx(0.0, abs=0.01)

    def test_get_correction_no_calibration_returns_predicted(self, default_config):
        """Without calibration data, get_correction returns the original p."""
        cal = ConeCalibrator(default_config)
        result = cal.get_correction(horizon=1, predicted_p=0.55)
        assert result == 0.55

    def test_get_correction_with_calibration_data(self, default_config):
        cal = ConeCalibrator(default_config)
        # Manually set calibration data
        cal.calibration_data = {1: {0.6: 1.2}}  # 20% upward correction

        corrected = cal.get_correction(horizon=1, predicted_p=0.60)
        # 0.60 * 1.2 = 0.72, but clamped to [0.40, 0.70]
        assert corrected == 0.70

    def test_get_correction_clamps_low(self, default_config):
        cal = ConeCalibrator(default_config)
        cal.calibration_data = {1: {0.5: 0.1}}  # severe downward correction

        corrected = cal.get_correction(horizon=1, predicted_p=0.50)
        # 0.50 * 0.1 = 0.05, clamped to 0.40
        assert corrected == 0.40

    def test_get_correction_missing_bucket_returns_identity(self, default_config):
        cal = ConeCalibrator(default_config)
        cal.calibration_data = {1: {0.6: 1.5}}

        # Ask for bucket 0.5 which doesn't exist -> factor=1.0 -> return p * 1.0
        corrected = cal.get_correction(horizon=1, predicted_p=0.50)
        assert corrected == 0.50

    def test_get_metrics_empty(self, default_config):
        cal = ConeCalibrator(default_config)
        metrics = cal.get_metrics()
        assert metrics["horizons_calibrated"] == 0
        assert metrics["calibration_data"] == {}

    def test_get_metrics_after_calibration(self, default_config):
        cal = ConeCalibrator(default_config)
        cal.calibration_data = {1: {0.6: 1.2}, 30: {0.5: 0.9}}
        metrics = cal.get_metrics()
        assert metrics["horizons_calibrated"] == 2
        assert "1" in metrics["calibration_data"]
        assert "30" in metrics["calibration_data"]

    def test_get_recent_exits_fallback_to_reeval_report(self, default_config):
        """When get_recent_exits doesn't exist, fall back to reeval report."""
        tracker = MagicMock(spec=[])  # no get_recent_exits attribute
        tracker.get_reeval_effectiveness_report = MagicMock(
            return_value={"exits": [{"hold_time_seconds": 5, "pnl_bps": 10}]}
        )
        cal = ConeCalibrator(default_config, devil_tracker=tracker)
        exits = cal._get_recent_exits()
        assert len(exits) == 1
        assert exits[0]["pnl_bps"] == 10

    def test_get_recent_exits_exception_handling(self, default_config):
        """If both methods raise, return empty list."""
        tracker = MagicMock()
        tracker.get_recent_exits.side_effect = RuntimeError("DB error")
        tracker.get_reeval_effectiveness_report.side_effect = RuntimeError("DB error")
        cal = ConeCalibrator(default_config, devil_tracker=tracker)
        exits = cal._get_recent_exits()
        assert exits == []

    def test_calibration_skips_exits_without_cone_predictions(self, default_config):
        """Exits lacking cone_predictions for the horizon should be skipped.

        When exits have hold_time >= horizon but no cone_predictions for that
        horizon, the horizon should produce an empty corrections dict (no buckets
        to compute). When hold_time < horizon, the exits are filtered out entirely.
        """
        # hold_time=400 means eligible for all horizons including 300
        # but cone_predictions only exists for horizon 300
        exits = [
            _make_exit(hold_time=400, cone_horizon=300, predicted_p=0.6, pnl_bps=10),
            _make_exit(hold_time=400, cone_horizon=300, predicted_p=0.6, pnl_bps=5),
            _make_exit(hold_time=400, cone_horizon=300, predicted_p=0.6, pnl_bps=8),
        ]
        tracker = MagicMock()
        tracker.get_recent_exits.return_value = exits
        cal = ConeCalibrator(default_config, devil_tracker=tracker)
        result = cal.run_daily_calibration()
        # Horizon 300 should have actual corrections because cone_predictions exist for it
        assert 300 in result
        assert 0.6 in result[300]
        # Horizon 1 may appear but with empty corrections (no cone_predictions match)
        if 1 in result:
            assert result[1] == {}

    def test_min_samples_enforced_per_bucket(self):
        """Buckets with fewer than min_samples get correction=1.0."""
        config = {"min_samples_per_bucket": 5, "min_positions_to_calibrate": 3}
        exits = [
            _make_exit(hold_time=5, cone_horizon=1, predicted_p=0.60, pnl_bps=10),
            _make_exit(hold_time=5, cone_horizon=1, predicted_p=0.60, pnl_bps=5),
            _make_exit(hold_time=5, cone_horizon=1, predicted_p=0.60, pnl_bps=8),
        ]
        tracker = MagicMock()
        tracker.get_recent_exits.return_value = exits
        cal = ConeCalibrator(config, devil_tracker=tracker)
        result = cal.run_daily_calibration()
        # 3 samples in bucket 0.6 < min_samples_per_bucket=5 -> correction=1.0
        assert result[1][0.6] == 1.0
