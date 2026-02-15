"""
Signal Auto-Throttle — Medallion Intelligence Module
=====================================================
Tracks per-signal directional accuracy over rolling windows and automatically
kills signals that have been consistently losing. Dead signals get zeroed before
they enter the weighted signal calculation.

"If a signal has been losing for 3 days, it's not unlucky — it's broken."
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple


class SignalAutoThrottle:
    """
    Monitors per-signal performance and kills losers.

    Integration point: called BETWEEN generate_signals() and calculate_weighted_signal().

    Usage:
        throttle = SignalAutoThrottle(config, logger)
        # Each cycle:
        signals = throttle.filter(signals, product_id)  # zeros killed signals
        # After price is known:
        throttle.update(product_id, previous_signals, price_move_direction)
    """

    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Rolling window sizes (in observations, not time)
        self._short_window = int(config.get("short_window", 72))     # ~72 cycles = ~6 hours at 5min
        self._long_window = int(config.get("long_window", 288))      # ~288 cycles = ~24 hours

        # Kill thresholds
        self._kill_accuracy = float(config.get("kill_accuracy", 0.45))     # Below 45% = kill
        self._min_samples_to_kill = int(config.get("min_samples_to_kill", 30))  # Need 30+ obs

        # Re-entry threshold
        self._reentry_accuracy = float(config.get("reentry_accuracy", 0.52))  # Must prove >52%

        # Per-signal rolling performance: {signal_name: deque of (correct: bool, timestamp)}
        self._performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self._long_window))

        # Currently killed signals: {signal_name: kill_timestamp}
        self._killed_signals: Dict[str, float] = {}

        # Throttle action log for auditing
        self._action_log: List[Dict[str, Any]] = []

        self._cycle_count = 0
        self.logger.info(
            f"SignalAutoThrottle initialized: kill<{self._kill_accuracy:.0%} "
            f"(min {self._min_samples_to_kill} samples), reentry>{self._reentry_accuracy:.0%}"
        )

    def filter(self, signals: Dict[str, float], product_id: str = "") -> Dict[str, float]:
        """
        Zero out any signals that are currently killed.
        Returns a modified copy of the signals dict.
        """
        filtered = signals.copy()
        killed_this_cycle = []

        for sig_name in list(filtered.keys()):
            if sig_name in self._killed_signals:
                if abs(filtered[sig_name]) > 1e-8:
                    killed_this_cycle.append(sig_name)
                filtered[sig_name] = 0.0

        if killed_this_cycle:
            self.logger.info(
                f"SIGNAL THROTTLE [{product_id}]: Zeroed {len(killed_this_cycle)} killed signals: "
                f"{', '.join(killed_this_cycle)}"
            )

        return filtered

    def update(self, product_id: str, signals: Dict[str, float], price_move: float):
        """
        Update signal performance based on whether each signal's direction
        matched the actual price move.

        Args:
            product_id: The product being evaluated
            signals: The signal values from the previous cycle
            price_move: Actual price change (positive = up, negative = down)
        """
        self._cycle_count += 1
        now = time.time()

        for sig_name, sig_val in signals.items():
            if abs(sig_val) < 0.01:
                continue  # Skip near-zero signals (no opinion = no evaluation)

            # Was the signal directionally correct?
            correct = (sig_val > 0 and price_move > 0) or (sig_val < 0 and price_move < 0)
            self._performance[sig_name].append((correct, now))

        # Evaluate kill/revive decisions periodically (every 10 cycles to avoid churn)
        if self._cycle_count % 10 == 0:
            self._evaluate_signals(product_id)

    def _evaluate_signals(self, product_id: str):
        """Check each signal against kill/revive thresholds."""
        now = time.time()

        for sig_name, history in self._performance.items():
            if len(history) < self._min_samples_to_kill:
                continue

            # Short-window accuracy (most recent N observations)
            short_data = list(history)[-self._short_window:]
            if len(short_data) < self._min_samples_to_kill:
                continue

            short_correct = sum(1 for c, _ in short_data if c)
            short_accuracy = short_correct / len(short_data)

            if sig_name in self._killed_signals:
                # Signal is killed — check for revive
                if short_accuracy >= self._reentry_accuracy and len(short_data) >= self._min_samples_to_kill:
                    del self._killed_signals[sig_name]
                    self._log_action("REVIVE", sig_name, short_accuracy, len(short_data), product_id)
                    self.logger.info(
                        f"SIGNAL REVIVED: {sig_name} accuracy recovered to "
                        f"{short_accuracy:.1%} over {len(short_data)} observations"
                    )
            else:
                # Signal is alive — check for kill
                if short_accuracy < self._kill_accuracy:
                    self._killed_signals[sig_name] = now
                    self._log_action("KILL", sig_name, short_accuracy, len(short_data), product_id)
                    self.logger.warning(
                        f"SIGNAL KILLED: {sig_name} accuracy {short_accuracy:.1%} < "
                        f"{self._kill_accuracy:.0%} over {len(short_data)} observations"
                    )

    def _log_action(self, action: str, signal_name: str, accuracy: float,
                    sample_count: int, product_id: str):
        """Log throttle action for auditing."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "signal_name": signal_name,
            "accuracy": round(accuracy, 4),
            "sample_count": sample_count,
            "product_id": product_id,
        }
        self._action_log.append(entry)
        # Keep last 500 actions
        if len(self._action_log) > 500:
            self._action_log = self._action_log[-500:]

    def get_status(self) -> Dict[str, Any]:
        """Return current throttle status for logging/dashboard."""
        signal_stats = {}
        for sig_name, history in self._performance.items():
            if len(history) < 5:
                continue
            recent = list(history)[-self._short_window:]
            correct = sum(1 for c, _ in recent if c)
            signal_stats[sig_name] = {
                "accuracy": round(correct / len(recent), 3) if recent else 0.0,
                "samples": len(recent),
                "killed": sig_name in self._killed_signals,
            }

        return {
            "killed_signals": list(self._killed_signals.keys()),
            "total_signals_tracked": len(self._performance),
            "signal_stats": signal_stats,
            "recent_actions": self._action_log[-10:],
        }

    def get_killed_count(self) -> int:
        """Return number of currently killed signals."""
        return len(self._killed_signals)
