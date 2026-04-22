"""SignalAgent — wraps signal scorecard, signal throttle, ML bridge."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

from agents.base import BaseAgent


class SignalAgent(BaseAgent):
    """Observes signal accuracy and throttle state."""

    def __init__(
        self,
        event_bus: Any,
        db_path: str,
        config: Dict[str, Any],
        signal_scorecard: Any = None,
        signal_throttle: Any = None,
        ml_bridge: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("signal", event_bus, db_path, config, **kwargs)
        self.signal_scorecard = signal_scorecard
        self.signal_throttle = signal_throttle
        self.ml_bridge = ml_bridge
        self._state = "active"

    def get_status(self) -> Dict[str, Any]:
        status = self._base_status()
        # Extract scorecard summary if available
        if self.signal_scorecard and isinstance(self.signal_scorecard, dict):
            total_signals = 0
            total_correct = 0
            for pid, sigs in self.signal_scorecard.items():
                for sig_name, stats in sigs.items():
                    total_signals += stats.get("total", 0)
                    total_correct += stats.get("correct", 0)
            status["total_predictions"] = total_signals
            status["total_correct"] = total_correct
            status["accuracy"] = (
                total_correct / total_signals if total_signals > 0 else 0.0
            )
        # Throttle state
        if self.signal_throttle and hasattr(self.signal_throttle, "throttled_signals"):
            status["throttled_signals"] = list(
                getattr(self.signal_throttle, "throttled_signals", set())
            )
        return status

    def get_observations(self, window_hours: int = 168) -> Dict[str, Any]:
        self._record_run()
        obs: Dict[str, Any] = {"agent": self.name}
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            # Per-signal daily P&L
            rows = conn.execute(
                """SELECT signal_type, SUM(pnl), COUNT(*)
                   FROM signal_daily_pnl
                   WHERE date >= date('now', ? || ' days')
                   GROUP BY signal_type
                   ORDER BY SUM(pnl) DESC""",
                (f"-{window_hours // 24}",),
            ).fetchall()
            obs["signal_pnl"] = [
                {"signal": r[0], "total_pnl": r[1], "days": r[2]} for r in rows
            ]
            # Decision distribution
            rows = conn.execute(
                """SELECT action, COUNT(*) FROM decisions
                   WHERE timestamp >= datetime('now', ? || ' hours')
                   GROUP BY action""",
                (f"-{window_hours}",),
            ).fetchall()
            obs["decision_distribution"] = {r[0]: r[1] for r in rows}
            # ML prediction stats
            try:
                rows = conn.execute(
                    """SELECT model_name, COUNT(*), AVG(confidence)
                       FROM ml_predictions
                       WHERE timestamp >= datetime('now', ? || ' hours')
                       GROUP BY model_name""",
                    (f"-{window_hours}",),
                ).fetchall()
                obs["ml_predictions"] = [
                    {"model": r[0], "count": r[1], "avg_confidence": r[2]}
                    for r in rows
                ]
            except sqlite3.OperationalError:
                obs["ml_predictions"] = []
            conn.close()
        except Exception as exc:
            self._record_error(exc)
            obs["error"] = str(exc)
        return obs

    def on_cycle_complete(self, cycle_data: Dict[str, Any]) -> None:
        self._record_run()
        action = cycle_data.get("action", "HOLD")
        confidence = cycle_data.get("confidence", 0.0)
        if action != "HOLD" and confidence > 0.0:
            self.event_bus.emit(
                "signal.trade_signal",
                {"action": action, "confidence": confidence,
                 "product_id": cycle_data.get("product_id", "")},
            )
