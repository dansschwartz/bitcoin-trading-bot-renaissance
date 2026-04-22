"""WebSocket connection manager — broadcasts events to all connected clients."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)
        logger.info(f"WS client connected ({len(self._active)} total)")

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)
        logger.info(f"WS client disconnected ({len(self._active)} total)")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        payload = json.dumps(message, default=str)
        dead: List[WebSocket] = []
        for ws in self._active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._active.discard(ws)

    @property
    def active_count(self) -> int:
        return len(self._active)
