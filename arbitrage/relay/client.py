"""
Order Book Relay Client — receives MEXC order book updates from the Bangalore relay.

Runs on the US droplet where MEXC WebSocket is geo-blocked.
Replaces 15s REST polling with ~100ms real-time relay data.
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from decimal import Decimal
from typing import Optional, List

import websockets

from ..exchanges.base import OrderBook, OrderBookLevel

logger = logging.getLogger("arb.relay.client")


class OrderBookRelayClient:
    """WebSocket client that receives relayed MEXC order book updates."""

    def __init__(self, relay_url: str, auth_token: Optional[str] = None,
                 pairs: Optional[List[str]] = None, book_manager=None,
                 reconnect_max_backoff: float = 30.0):
        self._url = relay_url
        self._auth_token = auth_token or os.getenv("RELAY_AUTH_TOKEN", "")
        self._pairs = pairs or []
        self._book_manager = book_manager
        self._reconnect_max_backoff = reconnect_max_backoff
        self._running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        # Stats
        self._connected = False
        self._messages_received = 0
        self._last_message_time: Optional[float] = None
        self._last_seq = 0
        self._seq_gaps = 0
        self._reconnect_count = 0
        self._start_time: Optional[float] = None

    async def start(self) -> None:
        """Connect to the relay server with automatic reconnection."""
        self._running = True
        self._start_time = time.monotonic()
        backoff = 2.0

        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                if self._running:
                    logger.warning(
                        f"Relay connection lost: {e} — reconnecting in {backoff:.0f}s"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self._reconnect_max_backoff)
                    self._reconnect_count += 1

        self._connected = False
        logger.info("Relay client stopped")

    async def stop(self) -> None:
        """Disconnect from the relay server."""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _connect_and_listen(self) -> None:
        """Single connection lifecycle: connect, subscribe, receive loop."""
        logger.info(f"Connecting to relay at {self._url}")

        async with websockets.connect(
            self._url,
            ping_interval=20,
            ping_timeout=10,
            max_size=2**20,
            open_timeout=10,
        ) as ws:
            self._ws = ws

            # Send subscribe with auth
            subscribe_msg = json.dumps({
                "type": "subscribe",
                "pairs": self._pairs,
                "auth_token": self._auth_token,
            })
            await ws.send(subscribe_msg)

            self._connected = True
            self._last_seq = 0  # Reset sequence tracking on new connection
            logger.info(
                f"Relay connected — subscribed to {len(self._pairs)} pairs"
            )

            # Receive loop
            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                    if msg.get("type") == "book":
                        await self._handle_book(msg)
                except json.JSONDecodeError:
                    logger.debug("Relay received malformed JSON")
                except Exception as e:
                    logger.warning(f"Relay message handling error: {e}")

        self._ws = None
        self._connected = False

    async def _handle_book(self, msg: dict) -> None:
        """Deserialize a book message and forward to the book manager."""
        symbol = msg["symbol"]
        seq = msg.get("seq", 0)

        # Sequence gap detection
        if self._last_seq > 0 and seq > self._last_seq + 1:
            gap = seq - self._last_seq - 1
            self._seq_gaps += gap
            logger.warning(f"Relay sequence gap: expected {self._last_seq + 1}, got {seq} ({gap} missed)")
        self._last_seq = seq

        # Deserialize bids/asks with Decimal precision
        bids = [
            OrderBookLevel(Decimal(p), Decimal(q))
            for p, q in msg.get("bids", [])
        ]
        asks = [
            OrderBookLevel(Decimal(p), Decimal(q))
            for p, q in msg.get("asks", [])
        ]

        # Parse timestamp
        try:
            ts = datetime.fromisoformat(msg["ts"])
        except (KeyError, ValueError):
            ts = datetime.utcnow()

        book = OrderBook(
            exchange="mexc",
            symbol=symbol,
            timestamp=ts,
            bids=bids,
            asks=asks,
        )

        self._messages_received += 1
        self._last_message_time = time.monotonic()

        # Forward to book manager (same callback as MEXC WS)
        if self._book_manager:
            await self._book_manager._on_mexc_update(symbol, book)

    def get_status(self) -> dict:
        """Return client status for monitoring."""
        now = time.monotonic()
        uptime = now - self._start_time if self._start_time else 0
        last_age = now - self._last_message_time if self._last_message_time else None
        return {
            "connected": self._connected,
            "url": self._url,
            "messages_received": self._messages_received,
            "last_message_age_seconds": round(last_age, 2) if last_age is not None else None,
            "last_sequence": self._last_seq,
            "sequence_gaps": self._seq_gaps,
            "reconnect_count": self._reconnect_count,
            "uptime_seconds": round(uptime, 1),
        }
