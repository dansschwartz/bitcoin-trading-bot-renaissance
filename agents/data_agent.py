"""DataCollectionAgent — wraps market_data_provider, bar_aggregator, data_validator."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Optional

from agents.base import BaseAgent


class DataCollectionAgent(BaseAgent):
    """Observes data ingestion quality and bar completeness."""

    def __init__(
        self,
        event_bus: Any,
        db_path: str,
        config: Dict[str, Any],
        market_data_provider: Any = None,
        bar_aggregator: Any = None,
        data_validator: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("data", event_bus, db_path, config, **kwargs)
        self.market_data_provider = market_data_provider
        self.bar_aggregator = bar_aggregator
        self.data_validator = data_validator
        self._state = "active"

    def get_status(self) -> Dict[str, Any]:
        status = self._base_status()
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            row = conn.execute(
                "SELECT COUNT(*) FROM five_minute_bars"
            ).fetchone()
            status["total_bars"] = row[0] if row else 0
            pairs = conn.execute(
                "SELECT pair, COUNT(*) FROM five_minute_bars GROUP BY pair"
            ).fetchall()
            status["bars_per_pair"] = {p: c for p, c in pairs}
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
            # Bar completeness per pair in window
            rows = conn.execute(
                """SELECT pair, COUNT(*), MIN(bar_start), MAX(bar_end)
                   FROM five_minute_bars
                   WHERE bar_start >= datetime('now', ? || ' hours')
                   GROUP BY pair""",
                (f"-{window_hours}",),
            ).fetchall()
            obs["bar_completeness"] = [
                {"pair": r[0], "bars": r[1], "first": r[2], "last": r[3]}
                for r in rows
            ]
            # Total bars
            total = conn.execute("SELECT COUNT(*) FROM five_minute_bars").fetchone()
            obs["total_bars"] = total[0] if total else 0
            conn.close()
        except Exception as exc:
            self._record_error(exc)
            obs["error"] = str(exc)
        return obs

    def on_cycle_complete(self, cycle_data: Dict[str, Any]) -> None:
        self._record_run()
        bar_count = cycle_data.get("bar_count")
        if bar_count is not None:
            self.event_bus.emit("data.bars_updated", {"bar_count": bar_count})
