"""ExecutionAgent — wraps devil_tracker, position_manager."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

from agents.base import BaseAgent


class ExecutionAgent(BaseAgent):
    """Observes execution quality — slippage, latency, fill rates."""

    def __init__(
        self,
        event_bus: Any,
        db_path: str,
        config: Dict[str, Any],
        devil_tracker: Any = None,
        position_manager: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("execution", event_bus, db_path, config, **kwargs)
        self.devil_tracker = devil_tracker
        self.position_manager = position_manager
        self._state = "active"

    def get_status(self) -> Dict[str, Any]:
        status = self._base_status()
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            row = conn.execute("SELECT COUNT(*) FROM devil_tracker").fetchone()
            status["devil_tracker_entries"] = row[0] if row else 0
            row = conn.execute(
                "SELECT AVG(slippage_bps) FROM devil_tracker"
            ).fetchone()
            status["avg_slippage_bps"] = round(row[0], 2) if row and row[0] else 0.0
            conn.close()
        except Exception as exc:
            status["db_error"] = str(exc)
        return status

    def get_observations(self, window_hours: int = 168) -> Dict[str, Any]:
        self._record_run()
        obs: Dict[str, Any] = {"agent": self.name}
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            rows = conn.execute(
                """SELECT signal_type,
                          COUNT(*) as fills,
                          AVG(slippage_bps) as avg_slip,
                          AVG(latency_signal_to_fill_ms) as avg_latency,
                          AVG(devil) as avg_devil
                   FROM devil_tracker
                   WHERE signal_timestamp >= datetime('now', ? || ' hours')
                   GROUP BY signal_type""",
                (f"-{window_hours}",),
            ).fetchall()
            obs["execution_quality"] = [
                {
                    "signal_type": r[0],
                    "fills": r[1],
                    "avg_slippage_bps": round(r[2], 2) if r[2] else 0.0,
                    "avg_latency_ms": round(r[3], 1) if r[3] else 0.0,
                    "avg_devil": round(r[4], 4) if r[4] else 0.0,
                }
                for r in rows
            ]
            conn.close()
        except Exception as exc:
            self._record_error(exc)
            obs["error"] = str(exc)
        return obs
