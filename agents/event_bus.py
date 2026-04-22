"""In-process, thread-safe EventBus with dot-notation channels and wildcard subscribe.

Modelled after ``dashboard/event_emitter.py`` but purpose-built for the agent
coordination layer.  Channels use dot-notation (``regime.changed``,
``risk.circuit_breaker``).  Subscribers can use wildcards (``risk.*``).
A ring buffer keeps the last *max_history* events for diagnostics / dashboard.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Type alias for handler signature
Handler = Callable[[str, Dict[str, Any]], None]


class EventBus:
    """Thread-safe, synchronous-dispatch event bus.

    Parameters
    ----------
    max_history : int
        Number of events to keep in the ring buffer (default 500).
    """

    def __init__(self, max_history: int = 500) -> None:
        self._lock = threading.Lock()
        # channel_pattern → list of (handler, subscriber_name)
        self._handlers: Dict[str, List[Tuple[Handler, str]]] = {}
        self._history: deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._channel_cache: Dict[str, Any] = {}

    # ── Subscribe / Unsubscribe ──

    def subscribe(
        self,
        channel_pattern: str,
        handler: Handler,
        subscriber: str = "",
    ) -> None:
        """Register *handler* for events matching *channel_pattern*.

        *channel_pattern* may contain ``*`` and ``?`` wildcards
        (``fnmatch`` semantics).  E.g. ``"risk.*"`` matches
        ``"risk.circuit_breaker"`` and ``"risk.alert"``.
        """
        with self._lock:
            self._handlers.setdefault(channel_pattern, []).append(
                (handler, subscriber)
            )

    def unsubscribe(self, channel_pattern: str, handler: Handler) -> None:
        with self._lock:
            entries = self._handlers.get(channel_pattern, [])
            self._handlers[channel_pattern] = [
                (h, s) for h, s in entries if h is not handler
            ]

    # ── Emit ──

    def emit(self, channel: str, payload: Dict[str, Any]) -> None:
        """Emit an event on *channel* with *payload*.

        Dispatches synchronously to all matching handlers.  Exceptions in
        handlers are caught and logged — one bad handler cannot break others.
        """
        event = {
            "channel": channel,
            "data": payload,
            "ts": datetime.now(timezone.utc).isoformat(),
            "ts_epoch": time.time(),
        }

        with self._lock:
            self._history.append(event)
            self._channel_cache[channel] = payload
            # Snapshot handlers to avoid holding lock during dispatch
            matching: List[Tuple[Handler, str]] = []
            for pattern, entries in self._handlers.items():
                if fnmatch.fnmatch(channel, pattern):
                    matching.extend(entries)

        for handler, subscriber in matching:
            try:
                handler(channel, payload)
            except Exception as exc:
                logger.error(
                    "EventBus handler error [%s] on channel %s: %s",
                    subscriber or handler.__name__,
                    channel,
                    exc,
                )

    # ── Query ──

    def get_cached(self, channel: str) -> Optional[Any]:
        """Return the last payload emitted on *channel*, or ``None``."""
        return self._channel_cache.get(channel)

    def get_history(
        self,
        channel_pattern: str = "*",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return recent events matching *channel_pattern* (newest first)."""
        with self._lock:
            events = list(self._history)
        events.reverse()
        if channel_pattern != "*":
            events = [
                e for e in events
                if fnmatch.fnmatch(e["channel"], channel_pattern)
            ]
        return events[:limit]

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._channel_cache.clear()
