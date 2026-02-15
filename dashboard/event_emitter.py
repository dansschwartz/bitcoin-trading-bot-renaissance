"""Lightweight event queue: bot pushes events, dashboard consumes them.

Uses threading primitives so the bot (running in its own event loop / thread)
can safely push events that the dashboard's asyncio WebSocket relay consumes.
"""

import asyncio
import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DashboardEventEmitter:
    """Thread-safe fan-out queue for real-time dashboard events.

    Subscribers get a thread-safe ``queue.Queue``.  The dashboard's WS relay
    wraps it with ``await asyncio.to_thread(q.get)`` so it doesn't block
    the event loop.
    """

    def __init__(self) -> None:
        self._subscribers: List[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    async def emit(self, channel: str, payload: Dict[str, Any]) -> None:
        self._do_emit(channel, payload)

    def emit_sync(self, channel: str, payload: Dict[str, Any]) -> None:
        self._do_emit(channel, payload)

    def _do_emit(self, channel: str, payload: Dict[str, Any]) -> None:
        msg = {
            "channel": channel,
            "data": payload,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass  # drop if consumer is slow
