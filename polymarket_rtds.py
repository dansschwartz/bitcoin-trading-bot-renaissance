"""
polymarket_rtds.py — Real-Time Data Socket for Polymarket

Provides live Binance + Chainlink prices for all traded assets.
The Chainlink feed is the RESOLUTION ORACLE — this is ground truth.
The Binance feed is what the CROWD trades on — this drives market prices.
The SPREAD between them is where the edge lives.

WebSocket endpoint: wss://ws-subscriptions-clob.polymarket.com/ws/market
No authentication required.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Dict, Optional, Callable

try:
    import websockets
except ImportError:
    websockets = None

logger = logging.getLogger("polymarket_rtds")

RTDS_URL = "wss://ws-live-data.polymarket.com"

# Symbols for subscription — all 7 assets traded on Polymarket direction markets
SYMBOLS = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
    "XRP": "xrpusdt",
    "DOGE": "dogeusdt",
    "BNB": "bnbusdt",
    "HYPE": "hypeusdt",
}

# Chainlink uses different format
CHAINLINK_SYMBOLS = {
    "BTC": "btc/usd",
    "ETH": "eth/usd",
    "SOL": "sol/usd",
    "XRP": "xrp/usd",
    "DOGE": "doge/usd",
    "BNB": "bnb/usd",
    "HYPE": "hype/usd",
}


class PolymarketRTDS:
    """
    Real-time price feed from Polymarket.

    Provides:
    - get_binance_price(asset) -> latest Binance price
    - get_chainlink_price(asset) -> latest Chainlink price (resolution oracle)
    - get_spread(asset) -> Binance vs Chainlink spread
    - get_resolution_direction(asset, window_start) -> UP/DOWN based on Chainlink
    """

    def __init__(self):
        self._binance_prices: Dict[str, dict] = {}   # asset -> {price, timestamp}
        self._chainlink_prices: Dict[str, dict] = {}  # asset -> {price, timestamp}
        self._window_starts: Dict[str, float] = {}    # asset:window -> chainlink start price
        self._running = False
        self._ws = None
        self._callbacks = []
        self._connected = False
        self._last_binance_update = 0.0
        self._last_chainlink_update = 0.0
        self._thread: Optional[threading.Thread] = None
        self._thread_loop: Optional[asyncio.AbstractEventLoop] = None

    def on_price_update(self, callback: Callable):
        """Register callback for price updates. Called with (source, asset, price, ts)."""
        self._callbacks.append(callback)

    def start_in_thread(self):
        """Run the RTDS WebSocket in a dedicated daemon thread.

        This isolates the WebSocket from the main bot's busy event loop,
        preventing ping/pong starvation that causes disconnections.
        Dict updates in Python are atomic for simple assignments, so
        price data is safely shared between threads.
        """
        def _thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._thread_loop = loop
            try:
                loop.run_until_complete(self.connect())
            except Exception as e:
                logger.error(f"RTDS thread crashed: {e}")
            finally:
                loop.close()

        self._thread = threading.Thread(
            target=_thread_target,
            name="rtds-websocket",
            daemon=True,
        )
        self._thread.start()
        logger.info("RTDS WebSocket thread started (isolated from main event loop)")

    async def connect(self):
        """Connect to RTDS WebSocket and subscribe to both feeds."""
        if websockets is None:
            logger.error("websockets package not installed — pip install websockets")
            return

        self._running = True

        while self._running:
            try:
                async with websockets.connect(
                    RTDS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                    open_timeout=15,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    logger.info(f"RTDS connected to {RTDS_URL}")

                    # Subscribe to Binance prices (one sub per symbol)
                    for asset, b_sym in SYMBOLS.items():
                        binance_sub = {
                            "action": "subscribe",
                            "subscriptions": [{
                                "topic": "crypto_prices",
                                "type": "update",
                                "filters": json.dumps({"symbol": b_sym})
                            }]
                        }
                        await ws.send(json.dumps(binance_sub))
                        logger.info(f"Subscribed to Binance: {b_sym}")

                    # Subscribe to Chainlink prices (one sub per symbol, type="update")
                    for asset, cl_sym in CHAINLINK_SYMBOLS.items():
                        chainlink_sub = {
                            "action": "subscribe",
                            "subscriptions": [{
                                "topic": "crypto_prices_chainlink",
                                "type": "update",
                                "filters": json.dumps({"symbol": cl_sym})
                            }]
                        }
                        await ws.send(json.dumps(chainlink_sub))
                        logger.info(f"Subscribed to Chainlink: {cl_sym}")

                    # Process messages
                    async for msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(msg)
                            self._handle_message(data)
                        except json.JSONDecodeError:
                            continue

            except asyncio.CancelledError:
                self._connected = False
                return
            except Exception as e:
                self._connected = False
                logger.warning(f"RTDS connection error: {e}, reconnecting in 3s...")
                await asyncio.sleep(3)

    def _handle_message(self, data: dict):
        """Route incoming messages to the right handler.

        Message format from RTDS:
        {
            "topic": "crypto_prices",
            "type": "update",
            "timestamp": 1775661348000,
            "payload": {
                "symbol": "btcusdt" | "btc/usd",
                "data": [{"timestamp": 1775..., "value": 71124.47}, ...]
            }
        }

        Note: Both Binance and Chainlink messages arrive with topic="crypto_prices".
        We distinguish them by symbol format: "btcusdt" = Binance, "btc/usd" = Chainlink.
        """
        payload = data.get("payload", {})
        symbol = payload.get("symbol", "")
        data_points = payload.get("data", [])

        if not symbol or not data_points:
            return

        # Get the latest data point (last in the array)
        latest = data_points[-1] if data_points else {}
        price = latest.get("value")
        ts = latest.get("timestamp", int(time.time() * 1000))

        if price is None:
            return

        # Distinguish Binance vs Chainlink by symbol format
        # Chainlink uses slash format: "btc/usd", "eth/usd"
        # Binance uses concatenated: "btcusdt", "ethusdt"
        if "/" in symbol:
            # Chainlink price (RESOLUTION ORACLE)
            asset = self._symbol_to_asset(symbol, CHAINLINK_SYMBOLS)
            if asset:
                self._chainlink_prices[asset] = {
                    "price": float(price),
                    "timestamp": ts,
                }
                self._last_chainlink_update = time.time()
                for cb in self._callbacks:
                    try:
                        cb("chainlink", asset, float(price), ts)
                    except Exception:
                        pass
        else:
            # Binance price
            asset = self._symbol_to_asset(symbol, SYMBOLS)
            if asset:
                self._binance_prices[asset] = {
                    "price": float(price),
                    "timestamp": ts,
                }
                self._last_binance_update = time.time()
                for cb in self._callbacks:
                    try:
                        cb("binance", asset, float(price), ts)
                    except Exception:
                        pass

    def _symbol_to_asset(self, symbol: str, mapping: dict) -> Optional[str]:
        """Reverse lookup: symbol -> asset name."""
        symbol_lower = symbol.lower().replace("/", "")
        for asset, sym in mapping.items():
            if sym.replace("/", "").lower() == symbol_lower:
                return asset
        return None

    # ═══════════════════════════════════════════════════════════
    # PUBLIC GETTERS
    # ═══════════════════════════════════════════════════════════

    def get_binance_price(self, asset: str) -> Optional[float]:
        entry = self._binance_prices.get(asset)
        return entry["price"] if entry else None

    def get_chainlink_price(self, asset: str) -> Optional[float]:
        entry = self._chainlink_prices.get(asset)
        return entry["price"] if entry else None

    def get_spread(self, asset: str) -> Optional[float]:
        """Binance - Chainlink spread. Positive = Binance higher."""
        b = self.get_binance_price(asset)
        c = self.get_chainlink_price(asset)
        if b and c:
            return b - c
        return None

    def record_window_start(self, asset: str, window_start: int):
        """Record Chainlink price at window open for resolution tracking."""
        cl = self.get_chainlink_price(asset)
        if cl:
            key = f"{asset}:{window_start}"
            self._window_starts[key] = cl
            logger.info(f"[RTDS] Window start recorded: {asset} @ ${cl:.2f}")

    def get_resolution_direction(self, asset: str, window_start: int) -> Optional[str]:
        """
        Based on CHAINLINK prices (the actual resolution oracle):
        Returns 'UP' if current Chainlink >= window start Chainlink
        Returns 'DOWN' if current Chainlink < window start Chainlink
        Returns None if data unavailable
        """
        key = f"{asset}:{window_start}"
        start_price = self._window_starts.get(key)
        current_price = self.get_chainlink_price(asset)

        if start_price is None or current_price is None:
            return None

        return "UP" if current_price >= start_price else "DOWN"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_status(self) -> dict:
        """Return status dict for dashboard."""
        status = {
            "connected": self._connected,
            "last_binance_update": self._last_binance_update,
            "last_chainlink_update": self._last_chainlink_update,
            "window_starts_tracked": len(self._window_starts),
            "assets": {},
        }
        for asset in SYMBOLS:
            status["assets"][asset] = {
                "binance": self.get_binance_price(asset),
                "chainlink": self.get_chainlink_price(asset),
                "spread": self.get_spread(asset),
            }
        # Backward compat flat keys for BTC/ETH
        status["btc_binance"] = self.get_binance_price("BTC")
        status["btc_chainlink"] = self.get_chainlink_price("BTC")
        status["btc_spread"] = self.get_spread("BTC")
        status["eth_binance"] = self.get_binance_price("ETH")
        status["eth_chainlink"] = self.get_chainlink_price("ETH")
        status["eth_spread"] = self.get_spread("ETH")
        return status

    async def stop(self):
        self._running = False
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    def stop_sync(self):
        """Stop from non-async context (e.g. during shutdown)."""
        self._running = False
        self._connected = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
