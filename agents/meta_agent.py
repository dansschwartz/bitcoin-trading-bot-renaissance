"""MetaAgent — wraps adaptive weight engine, genetic optimizer."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

from agents.base import BaseAgent


class MetaAgent(BaseAgent):
    """Observes meta-level learning — weight adaptation, genetic optimisation."""

    def __init__(
        self,
        event_bus: Any,
        db_path: str,
        config: Dict[str, Any],
        adaptive_weights: Any = None,
        genetic_optimizer: Any = None,
        signal_weights: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("meta", event_bus, db_path, config, **kwargs)
        self.adaptive_weights = adaptive_weights
        self.genetic_optimizer = genetic_optimizer
        self.signal_weights = signal_weights  # dict reference from bot
        self._state = "active"

    def get_status(self) -> Dict[str, Any]:
        status = self._base_status()
        if self.signal_weights and isinstance(self.signal_weights, dict):
            status["signal_weight_count"] = len(self.signal_weights)
            # Top 5 weights
            sorted_weights = sorted(
                self.signal_weights.items(), key=lambda x: abs(x[1]), reverse=True,
            )
            status["top_weights"] = {k: round(v, 4) for k, v in sorted_weights[:5]}
        return status

    def get_observations(self, window_hours: int = 168) -> Dict[str, Any]:
        self._record_run()
        obs: Dict[str, Any] = {"agent": self.name}
        # Current signal weights snapshot
        if self.signal_weights and isinstance(self.signal_weights, dict):
            obs["current_weights"] = {
                k: round(v, 6) for k, v in self.signal_weights.items()
            }
        # Genetic optimizer state
        if self.genetic_optimizer:
            if hasattr(self.genetic_optimizer, "best_fitness"):
                obs["genetic_best_fitness"] = getattr(
                    self.genetic_optimizer, "best_fitness", None
                )
            if hasattr(self.genetic_optimizer, "generation"):
                obs["genetic_generation"] = getattr(
                    self.genetic_optimizer, "generation", 0
                )
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            # Signal throttle log
            try:
                rows = conn.execute(
                    """SELECT signal_type, action, reason
                       FROM signal_throttle_log
                       WHERE timestamp >= datetime('now', ? || ' hours')
                       ORDER BY timestamp DESC LIMIT 20""",
                    (f"-{window_hours}",),
                ).fetchall()
                obs["throttle_events"] = [
                    {"signal": r[0], "action": r[1], "reason": r[2]}
                    for r in rows
                ]
            except sqlite3.OperationalError:
                obs["throttle_events"] = []
            conn.close()
        except Exception as exc:
            self._record_error(exc)
            obs["error"] = str(exc)
        return obs
