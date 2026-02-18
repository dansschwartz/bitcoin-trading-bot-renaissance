"""BaseAgent ABC — common interface for all agent wrappers."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.event_bus import EventBus


class BaseAgent(ABC):
    """Abstract base for every agent in the coordination system.

    Agents are *observers* — they wrap existing bot modules via constructor
    injection, extract metrics from DB queries, and emit events.
    They do NOT modify the trading loop behaviour.
    """

    def __init__(
        self,
        name: str,
        event_bus: EventBus,
        db_path: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.name = name
        self.event_bus = event_bus
        self.db_path = db_path
        self.config = config
        self.logger = logger or logging.getLogger(f"agent.{name}")
        self._run_count: int = 0
        self._error_count: int = 0
        self._last_run_ts: float = 0.0
        self._state: str = "idle"

    # ── Public interface ──

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return a dashboard-friendly status dict."""

    @abstractmethod
    def get_observations(self, window_hours: int = 168) -> Dict[str, Any]:
        """Compile observations for the weekly research report.

        *window_hours* defaults to 168 (7 days).
        """

    def on_cycle_complete(self, cycle_data: Dict[str, Any]) -> None:
        """Optional hook called at the end of every trading cycle."""

    # ── Helpers ──

    def _base_status(self) -> Dict[str, Any]:
        """Common status fields shared by all agents."""
        return {
            "name": self.name,
            "state": self._state,
            "run_count": self._run_count,
            "error_count": self._error_count,
            "last_run_ts": self._last_run_ts,
        }

    def _record_run(self) -> None:
        self._run_count += 1
        self._last_run_ts = time.time()

    def _record_error(self, err: Exception) -> None:
        self._error_count += 1
        self.logger.warning(f"[{self.name}] error: {err}")
