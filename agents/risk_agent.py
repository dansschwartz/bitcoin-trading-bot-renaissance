"""RiskAgent — wraps risk_gateway, regime_overlay, circuit breakers."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

from agents.base import BaseAgent


class RiskAgent(BaseAgent):
    """Observes risk gateway decisions, regime state, and circuit breaker activity."""

    def __init__(
        self,
        event_bus: Any,
        db_path: str,
        config: Dict[str, Any],
        risk_gateway: Any = None,
        regime_overlay: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("risk", event_bus, db_path, config, **kwargs)
        self.risk_gateway = risk_gateway
        self.regime_overlay = regime_overlay
        self._circuit_breaker_count: int = 0
        self._state = "active"

    def get_status(self) -> Dict[str, Any]:
        status = self._base_status()
        # Current regime
        if self.regime_overlay and hasattr(self.regime_overlay, "current_regime"):
            status["current_regime"] = getattr(
                self.regime_overlay, "current_regime", "unknown"
            )
        # Risk gateway stats
        if self.risk_gateway:
            if hasattr(self.risk_gateway, "pass_count"):
                status["gateway_pass"] = getattr(self.risk_gateway, "pass_count", 0)
            if hasattr(self.risk_gateway, "reject_count"):
                status["gateway_reject"] = getattr(self.risk_gateway, "reject_count", 0)
        status["circuit_breaker_activations"] = self._circuit_breaker_count
        return status

    def get_observations(self, window_hours: int = 168) -> Dict[str, Any]:
        self._record_run()
        obs: Dict[str, Any] = {"agent": self.name}
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            # Regime distribution in window
            rows = conn.execute(
                """SELECT hmm_regime, COUNT(*) FROM decisions
                   WHERE timestamp >= datetime('now', ? || ' hours')
                   GROUP BY hmm_regime ORDER BY COUNT(*) DESC""",
                (f"-{window_hours}",),
            ).fetchall()
            obs["regime_distribution"] = {r[0]: r[1] for r in rows}
            # Risk events from agent_events table (if exists)
            try:
                rows = conn.execute(
                    """SELECT event_type, COUNT(*) FROM agent_events
                       WHERE agent_name = 'risk'
                       AND timestamp >= datetime('now', ? || ' hours')
                       GROUP BY event_type""",
                    (f"-{window_hours}",),
                ).fetchall()
                obs["risk_events"] = {r[0]: r[1] for r in rows}
            except sqlite3.OperationalError:
                obs["risk_events"] = {}
            conn.close()
        except Exception as exc:
            self._record_error(exc)
            obs["error"] = str(exc)
        return obs

    def on_cycle_complete(self, cycle_data: Dict[str, Any]) -> None:
        self._record_run()
        regime = cycle_data.get("regime", "unknown")
        if regime != "unknown":
            self.event_bus.emit(
                "regime.updated",
                {"regime": regime, "confidence": cycle_data.get("regime_confidence", 0.0)},
            )
        # Detect circuit breaker
        if cycle_data.get("circuit_breaker_active"):
            self._circuit_breaker_count += 1
            self.event_bus.emit(
                "risk.circuit_breaker",
                {"reason": cycle_data.get("circuit_breaker_reason", "unknown")},
            )
