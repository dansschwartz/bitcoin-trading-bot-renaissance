"""PortfolioAgent — wraps portfolio_engine, position_reevaluator, kelly_sizer."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

from agents.base import BaseAgent


class PortfolioAgent(BaseAgent):
    """Observes portfolio state — positions, P&L, re-evaluation activity."""

    def __init__(
        self,
        event_bus: Any,
        db_path: str,
        config: Dict[str, Any],
        portfolio_engine: Any = None,
        position_reevaluator: Any = None,
        kelly_sizer: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("portfolio", event_bus, db_path, config, **kwargs)
        self.portfolio_engine = portfolio_engine
        self.position_reevaluator = position_reevaluator
        self.kelly_sizer = kelly_sizer
        self._state = "active"

    def get_status(self) -> Dict[str, Any]:
        status = self._base_status()
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            row = conn.execute(
                "SELECT COUNT(*) FROM open_positions WHERE status='open'"
            ).fetchone()
            status["open_positions"] = row[0] if row else 0
            # Recent P&L
            row = conn.execute(
                """SELECT SUM(pnl) FROM trades
                   WHERE close_time >= datetime('now', '-24 hours')"""
            ).fetchone()
            status["pnl_24h"] = round(row[0], 2) if row and row[0] else 0.0
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
            # Closed trades in window
            rows = conn.execute(
                """SELECT COUNT(*) as trades,
                          SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                          SUM(pnl) as total_pnl,
                          AVG(pnl) as avg_pnl
                   FROM trades
                   WHERE close_time >= datetime('now', ? || ' hours')""",
                (f"-{window_hours}",),
            ).fetchall()
            if rows and rows[0][0]:
                r = rows[0]
                trades = r[0] or 0
                wins = r[1] or 0
                obs["trades"] = trades
                obs["wins"] = wins
                obs["win_rate"] = round(wins / trades, 4) if trades > 0 else 0.0
                obs["total_pnl"] = round(r[2], 2) if r[2] else 0.0
                obs["avg_pnl"] = round(r[3], 4) if r[3] else 0.0
            else:
                obs["trades"] = 0
                obs["win_rate"] = 0.0
                obs["total_pnl"] = 0.0
            # Re-evaluation events
            try:
                row = conn.execute(
                    """SELECT COUNT(*) FROM reeval_events
                       WHERE timestamp >= datetime('now', ? || ' hours')""",
                    (f"-{window_hours}",),
                ).fetchone()
                obs["reeval_events"] = row[0] if row else 0
            except sqlite3.OperationalError:
                obs["reeval_events"] = 0
            conn.close()
        except Exception as exc:
            self._record_error(exc)
            obs["error"] = str(exc)
        return obs

    def on_cycle_complete(self, cycle_data: Dict[str, Any]) -> None:
        self._record_run()
        pnl = cycle_data.get("cycle_pnl")
        if pnl is not None and pnl != 0:
            self.event_bus.emit(
                "portfolio.pnl_update",
                {"pnl": pnl, "product_id": cycle_data.get("product_id", "")},
            )
