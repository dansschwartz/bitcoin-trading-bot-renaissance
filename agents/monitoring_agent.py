"""MonitoringAgent — wraps sharpe_monitor, beta_monitor, capacity_monitor."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

from agents.base import BaseAgent


class MonitoringAgent(BaseAgent):
    """Observes portfolio health metrics — Sharpe, beta, capacity."""

    def __init__(
        self,
        event_bus: Any,
        db_path: str,
        config: Dict[str, Any],
        sharpe_monitor: Any = None,
        beta_monitor: Any = None,
        capacity_monitor: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("monitoring", event_bus, db_path, config, **kwargs)
        self.sharpe_monitor = sharpe_monitor
        self.beta_monitor = beta_monitor
        self.capacity_monitor = capacity_monitor
        self._state = "active"

    def get_status(self) -> Dict[str, Any]:
        status = self._base_status()
        if self.sharpe_monitor and hasattr(self.sharpe_monitor, "current_sharpe"):
            status["current_sharpe"] = getattr(
                self.sharpe_monitor, "current_sharpe", None
            )
        if self.beta_monitor and hasattr(self.beta_monitor, "current_beta"):
            status["current_beta"] = getattr(
                self.beta_monitor, "current_beta", None
            )
        if self.capacity_monitor and hasattr(self.capacity_monitor, "at_capacity"):
            status["at_capacity"] = getattr(
                self.capacity_monitor, "at_capacity", False
            )
        return status

    def get_observations(self, window_hours: int = 168) -> Dict[str, Any]:
        self._record_run()
        obs: Dict[str, Any] = {"agent": self.name}
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            # Daily performance in window
            rows = conn.execute(
                """SELECT date, total_pnl, sharpe_ratio, total_trades, win_rate
                   FROM daily_performance
                   WHERE date >= date('now', ? || ' days')
                   ORDER BY date""",
                (f"-{window_hours // 24}",),
            ).fetchall()
            obs["daily_performance"] = [
                {
                    "date": r[0],
                    "pnl": r[1],
                    "sharpe": r[2],
                    "trades": r[3],
                    "win_rate": r[4],
                }
                for r in rows
            ]
            # Balance snapshots
            rows = conn.execute(
                """SELECT timestamp, total_balance_usd
                   FROM balance_snapshots
                   WHERE timestamp >= datetime('now', ? || ' hours')
                   ORDER BY timestamp""",
                (f"-{window_hours}",),
            ).fetchall()
            obs["balance_history"] = [
                {"ts": r[0], "balance": r[1]} for r in rows
            ]
            conn.close()
        except Exception as exc:
            self._record_error(exc)
            obs["error"] = str(exc)
        return obs

    def on_cycle_complete(self, cycle_data: Dict[str, Any]) -> None:
        self._record_run()
        sharpe = cycle_data.get("rolling_sharpe")
        if sharpe is not None:
            self.event_bus.emit("monitoring.sharpe_update", {"sharpe": sharpe})
