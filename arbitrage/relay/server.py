"""
Order Book Relay Server — broadcasts MEXC order book updates to remote clients.

Runs on the Bangalore droplet where MEXC WebSocket is accessible.
Relays real-time depth data to the US droplet over a lightweight WebSocket.
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, Set

import websockets
from websockets.server import serve

from ..exchanges.base import OrderBook

logger = logging.getLogger("arb.relay.server")


class OrderBookRelayServer:
    """Lightweight WebSocket server that broadcasts MEXC order book updates."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765,
                 auth_token: Optional[str] = None):
        self._host = host
        self._port = port
        self._auth_token = auth_token or os.getenv("RELAY_AUTH_TOKEN", "")
        self._clients: Set[websockets.WebSocketServerProtocol] = set()
        self._seq = 0
        self._running = False
        self._server = None
        # Stats
        self._messages_sent = 0
        self._clients_total = 0
        self._start_time: Optional[float] = None

    async def start(self) -> None:
        """Bind and start accepting connections."""
        self._running = True
        self._start_time = time.monotonic()
        self._server = await serve(
            self._handle_client,
            self._host,
            self._port,
            ping_interval=20,
            ping_timeout=10,
            max_size=2**20,  # 1 MB max message
        )
        logger.info(
            f"Relay server listening on {self._host}:{self._port} "
            f"(auth={'required' if self._auth_token else 'disabled'})"
        )

    async def stop(self) -> None:
        """Shut down the server."""
        self._running = False
        # Close all client connections
        if self._clients:
            await asyncio.gather(
                *(ws.close(1001, "server shutting down") for ws in self._clients),
                return_exceptions=True,
            )
            self._clients.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Relay server stopped")

    async def _handle_client(self, ws: websockets.WebSocketServerProtocol) -> None:
        """Handle a single client connection lifecycle."""
        remote = ws.remote_address
        logger.info(f"Relay client connecting from {remote}")

        # Wait for subscribe message with auth
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            msg = json.loads(raw)
        except (asyncio.TimeoutError, json.JSONDecodeError, websockets.ConnectionClosed) as e:
            logger.warning(f"Relay client {remote} failed handshake: {e}")
            return

        if msg.get("type") != "subscribe":
            await ws.close(4001, "expected subscribe message")
            return

        # Authenticate
        if self._auth_token and msg.get("auth_token") != self._auth_token:
            logger.warning(f"Relay client {remote} auth failed")
            await ws.close(4003, "authentication failed")
            return

        pairs = msg.get("pairs", [])
        self._clients.add(ws)
        self._clients_total += 1
        logger.info(
            f"Relay client {remote} authenticated — subscribed to {len(pairs)} pairs "
            f"(total clients: {len(self._clients)})"
        )

        try:
            # Hold connection open — client only receives, never sends
            # (except pong frames handled by websockets library)
            async for _ in ws:
                pass  # Ignore any messages from client after subscribe
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            logger.info(f"Relay client {remote} disconnected (remaining: {len(self._clients)})")

    async def broadcast_book(self, pair: str, book: OrderBook) -> None:
        """Serialize an OrderBook and broadcast to all connected clients.

        Called as the _mexc_relay_hook from UnifiedBookManager._on_mexc_update().
        """
        if not self._clients:
            return

        self._seq += 1
        msg = json.dumps({
            "type": "book",
            "symbol": pair,
            "ts": book.timestamp.isoformat(),
            "bids": [[str(lvl.price), str(lvl.quantity)] for lvl in book.bids],
            "asks": [[str(lvl.price), str(lvl.quantity)] for lvl in book.asks],
            "seq": self._seq,
        })

        dead: list = []
        for ws in list(self._clients):
            try:
                await ws.send(msg)
                self._messages_sent += 1
            except websockets.ConnectionClosed:
                dead.append(ws)
            except Exception as e:
                logger.debug(f"Relay send error: {e}")
                dead.append(ws)

        for ws in dead:
            self._clients.discard(ws)

    def get_status(self) -> dict:
        """Return server status for monitoring."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "host": self._host,
            "port": self._port,
            "connected_clients": len(self._clients),
            "total_clients_seen": self._clients_total,
            "messages_sent": self._messages_sent,
            "sequence": self._seq,
            "uptime_seconds": round(uptime, 1),
        }
