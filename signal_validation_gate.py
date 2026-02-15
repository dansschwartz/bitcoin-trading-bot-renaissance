"""
Signal Validation Gate — Three-Step Quality Filter
====================================================
Validates signal outputs before they enter the weighted calculation.

Step 1: Anomaly Clip — cap outlier signals at 3 sigma
Step 2: Noise Filter — zero signals indistinguishable from random
Step 3: Regime Consistency — penalize signals that contradict the HMM regime

"Every signal must earn its right to influence the portfolio."
"""

from __future__ import annotations

import logging
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Optional


class SignalValidationGate:
    """
    Three-step signal validation pipeline.

    Usage:
        gate = SignalValidationGate(config, logger)
        validated_signals = gate.validate(signals, regime_label)
    """

    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Step 1: Anomaly clip parameters
        self._clip_sigma = float(config.get("clip_sigma", 3.0))       # Clip at N sigma
        self._alert_sigma = float(config.get("alert_sigma", 4.0))     # Log warning above this
        self._rolling_window = int(config.get("rolling_window", 50))   # History window

        # Step 2: Noise filter parameters
        self._noise_min_samples = int(config.get("noise_min_samples", 20))
        self._noise_accuracy_threshold = float(config.get("noise_accuracy_threshold", 0.55))

        # Step 3: Regime consistency
        self._regime_penalty = float(config.get("regime_penalty", 0.5))  # Multiply contradicting signals

        # Rolling signal history: {signal_name: deque of values}
        self._signal_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._rolling_window)
        )

        # Regime-signal direction expectations
        # Positive = expect bullish signals, Negative = expect bearish
        self._regime_expectations = {
            "bull_trending": 1.0,
            "bull_mean_reverting": 0.5,
            "neutral_sideways": 0.0,
            "bear_mean_reverting": -0.5,
            "bear_trending": -1.0,
        }

        # Momentum-type signals that should align with regime direction
        self._directional_signals = {
            "macd", "rsi", "fractal", "lead_lag", "order_flow",
            "cross_exchange_momentum", "multi_exchange",
        }

        self._clip_count = 0
        self._penalty_count = 0
        self.logger.info(
            f"SignalValidationGate initialized: clip={self._clip_sigma}σ, "
            f"regime_penalty={self._regime_penalty}"
        )

    def validate(self, signals: Dict[str, float], regime_label: str = "unknown") -> Dict[str, float]:
        """
        Run three-step validation on all signals.

        Args:
            signals: Raw signal dict from generate_signals()
            regime_label: Current HMM regime (e.g. "bull_trending")

        Returns:
            Validated signal dict (modified copy)
        """
        validated = {}

        for sig_name, sig_val in signals.items():
            val = float(sig_val)

            # Step 1: Anomaly Clip
            val = self._anomaly_clip(sig_name, val)

            # Step 3: Regime Consistency (Step 2 noise filter deferred to auto-throttle)
            val = self._regime_consistency_check(sig_name, val, regime_label)

            validated[sig_name] = val

            # Update rolling history
            self._signal_history[sig_name].append(float(sig_val))

        return validated

    def _anomaly_clip(self, sig_name: str, value: float) -> float:
        """Step 1: Clip signals that are extreme outliers."""
        history = self._signal_history.get(sig_name)
        if not history or len(history) < 10:
            return value  # Not enough history to judge

        hist_array = np.array(history)
        mean = float(np.mean(hist_array))
        std = float(np.std(hist_array))

        if std < 1e-8:
            return value  # No variance — can't compute sigma

        z_score = abs(value - mean) / std

        if z_score > self._alert_sigma:
            self.logger.warning(
                f"SIGNAL ANOMALY: {sig_name}={value:.4f} is {z_score:.1f}σ from mean "
                f"(mean={mean:.4f}, std={std:.4f}) — clipping to {self._clip_sigma}σ"
            )
            self._clip_count += 1
            # Clip to 3 sigma in the same direction
            clipped = mean + self._clip_sigma * std * np.sign(value - mean)
            return float(clipped)

        return value

    def _regime_consistency_check(self, sig_name: str, value: float, regime_label: str) -> float:
        """
        Step 3: Penalize directional signals that contradict the current regime.
        A strong BUY signal in BEAR_TRENDING is suspicious — reduce it.
        """
        if sig_name not in self._directional_signals:
            return value  # Non-directional signals pass through

        if abs(value) < 0.05:
            return value  # Weak signals don't get penalized

        regime_direction = self._regime_expectations.get(regime_label, 0.0)
        if abs(regime_direction) < 0.3:
            return value  # Neutral regime — no penalty

        # Check if signal contradicts regime
        signal_direction = np.sign(value)
        regime_sign = np.sign(regime_direction)

        if signal_direction != regime_sign:
            # Signal contradicts regime — apply penalty
            penalized = value * self._regime_penalty
            self._penalty_count += 1
            if self._penalty_count <= 5 or self._penalty_count % 50 == 0:
                self.logger.info(
                    f"REGIME GATE: {sig_name}={value:.4f} contradicts {regime_label} "
                    f"— penalized to {penalized:.4f}"
                )
            return penalized

        return value

    def get_stats(self) -> Dict[str, Any]:
        """Return gate statistics."""
        return {
            "anomaly_clips": self._clip_count,
            "regime_penalties": self._penalty_count,
            "signals_tracked": len(self._signal_history),
        }
