"""
Unified WebSocket Price Feed — Binance discovery, MEXC execution.

Replaces MEXC WS as the primary price discovery source with Binance WS.
MEXC REST stays as fallback. MEXC is still used for order execution (0% maker fees).

Also connects to Binance Futures !forceOrder@arr for real-time liquidation tracking.

Two classes:
  - BinanceUnifiedPriceFeed: connects to Binance combined streams + futures liquidation feed
  - HybridTickerProvider: wraps Binance WS (primary) + MEXC REST (fallback)
"""
import asyncio
import json
import logging
import time
from collections import deque
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple

import websockets

logger = logging.getLogger("arb.feeds.unified")

# Binance combined stream endpoints (up to 1024 streams per connection)
# Port 443 is more firewall-friendly; 9443 is the traditional endpoint.
BINANCE_WS_ENDPOINTS = [
    "wss://stream.binance.com:443/stream",
    "wss://stream.binance.com:9443/stream",
    "wss://data-stream.binance.vision/stream",  # AWS backup
]

# Key assets that get real bid/ask from bookTicker streams
BOOK_TICKER_ASSETS = ["btcusdt", "ethusdt", "solusdt", "dogeusdt", "xrpusdt"]

# Max session age before reconnect (Binance recommends <24h)
WS_MAX_AGE_HOURS = 23

# Known quote currencies for symbol normalization
KNOWN_QUOTES = ("USDT", "USDC", "BTC", "ETH", "BNB")

# Binance Futures liquidation stream
BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws/!forceOrder@arr"


def _normalize_symbol(raw: str) -> str:
    """BTCUSDT -> BTC/USDT. Best-effort using known quote currencies."""
    raw = raw.upper()
    for quote in KNOWN_QUOTES:
        if raw.endswith(quote):
            base = raw[:-len(quote)]
            if base:
                return f"{base}/{quote}"
    return raw


class BinanceUnifiedPriceFeed:
    """Connects to Binance combined streams for real-time price discovery.

    Streams:
      - !miniTicker@arr: all 2000+ tickers, ~1s updates (close price, no bid/ask)
      - {sym}@bookTicker: real bid/ask for 5 key assets (BTC, ETH, SOL, DOGE, XRP)

    Stores tickers in the same format as MEXCClient._ws_tickers for compatibility:
      {"BTC/USDT": {"symbol": "BTC/USDT", "bid": Decimal, "ask": Decimal,
                     "last_price": Decimal, "volume_24h": Decimal}}
    """

    def __init__(self):
        self._tickers: Dict[str, dict] = {}
        self._last_update: float = 0.0  # monotonic time of last miniTicker batch
        self._last_book_update: Dict[str, float] = {}  # per-asset bookTicker times
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._futures_task: Optional[asyncio.Task] = None
        self._connect_time: float = 0.0
        self._reconnect_count: int = 0
        self._endpoint_idx: int = 0  # cycles through BINANCE_WS_ENDPOINTS

        # Liquidation tracking (from Binance Futures !forceOrder@arr)
        self._liquidations: Dict[str, dict] = {}  # symbol → rolling stats
        self._liq_callbacks: List[Callable] = []
        self._liq_history: deque = deque(maxlen=10000)  # last 10K liquidation orders
        self._liq_count: int = 0  # total orders received
        self._liq_connected: bool = False

    async def start(self) -> None:
        """Spawn background WebSocket tasks (spot + futures liquidation)."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._ws_loop())
        self._futures_task = asyncio.create_task(self._futures_ws_loop())
        logger.info("BinanceUnifiedPriceFeed: start requested (spot + liquidation feeds)")

    async def stop(self) -> None:
        """Cancel tasks and close WebSockets."""
        self._running = False
        for task in (self._task, self._futures_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._futures_task = None
        self._liq_connected = False
        logger.info("BinanceUnifiedPriceFeed: stopped")

    def get_tickers(self) -> Optional[Dict[str, dict]]:
        """All tickers if <10s fresh, else None."""
        if self._tickers and self._last_update > 0:
            age = time.monotonic() - self._last_update
            if age < 10.0:
                return self._tickers
        return None

    def get_ticker(self, symbol: str) -> Optional[dict]:
        """Single ticker by normalized name (e.g. 'BTC/USDT')."""
        tickers = self.get_tickers()
        if tickers:
            return tickers.get(symbol)
        return None

    def get_age_ms(self) -> float:
        """Age of last miniTicker batch in milliseconds."""
        if self._last_update <= 0:
            return float('inf')
        return (time.monotonic() - self._last_update) * 1000

    def is_healthy(self) -> bool:
        """Running + data fresh (<10s)."""
        return self._running and self.get_tickers() is not None

    def get_health(self) -> dict:
        """Full status for dashboard consumption."""
        tickers = self.get_tickers()
        ticker_count = len(tickers) if tickers else 0
        age_ms = self.get_age_ms()

        # bookTicker detail for key assets
        book_detail = {}
        for sym in BOOK_TICKER_ASSETS:
            normalized = _normalize_symbol(sym)
            t = self._tickers.get(normalized) if self._tickers else None
            if t:
                book_age = self._last_book_update.get(normalized, 0)
                book_detail[normalized] = {
                    "bid": str(t.get("bid", 0)),
                    "ask": str(t.get("ask", 0)),
                    "last_price": str(t.get("last_price", 0)),
                    "book_age_ms": round((time.monotonic() - book_age) * 1000) if book_age > 0 else None,
                }

        return {
            "source": "binance_ws",
            "running": self._running,
            "healthy": self.is_healthy(),
            "ticker_count": ticker_count,
            "age_ms": round(age_ms, 1) if age_ms != float('inf') else None,
            "reconnects": self._reconnect_count,
            "book_tickers": book_detail,
            "liquidation_feed": {
                "connected": self._liq_connected,
                "total_orders": self._liq_count,
                "symbols_tracked": len(self._liquidations),
            },
        }

    # ─── Liquidation Public API ───

    def get_liquidation_stats(self, symbol: str) -> Optional[dict]:
        """Get current liquidation stats for a symbol (e.g. 'BTC/USDT')."""
        normalized = _normalize_symbol(symbol) if "/" not in symbol else symbol
        stats = self._liquidations.get(normalized)
        if not stats:
            return None
        return {
            "long_usd_1m": stats["long_usd_1m"],
            "short_usd_1m": stats["short_usd_1m"],
            "long_usd_5m": stats["long_usd_5m"],
            "short_usd_5m": stats["short_usd_5m"],
            "long_count_5m": stats["long_count_5m"],
            "short_count_5m": stats["short_count_5m"],
            "cascade_active": stats.get("cascade_active", False),
            "squeeze_active": stats.get("squeeze_active", False),
            "imbalance_ratio": (
                stats["long_usd_5m"] / max(stats["short_usd_5m"], 1)
            ),
            "last_cascade_time": stats.get("last_cascade_time", 0),
        }

    def get_all_liquidation_stats(self) -> Dict[str, dict]:
        """Get liquidation stats for all symbols with recent activity."""
        result = {}
        for sym, s in self._liquidations.items():
            if s["long_usd_5m"] + s["short_usd_5m"] > 0:
                result[sym] = {
                    "long_usd_1m": s["long_usd_1m"],
                    "short_usd_1m": s["short_usd_1m"],
                    "long_usd_5m": s["long_usd_5m"],
                    "short_usd_5m": s["short_usd_5m"],
                    "long_count_5m": s["long_count_5m"],
                    "short_count_5m": s["short_count_5m"],
                    "cascade_active": s.get("cascade_active", False),
                    "squeeze_active": s.get("squeeze_active", False),
                    "imbalance_ratio": (
                        s["long_usd_5m"] / max(s["short_usd_5m"], 1)
                    ),
                }
        return result

    def is_cascade_active(self, symbol: str) -> bool:
        """Quick check: is a liquidation cascade happening right now?"""
        normalized = _normalize_symbol(symbol) if "/" not in symbol else symbol
        stats = self._liquidations.get(normalized)
        if not stats:
            return False
        return stats.get("cascade_active", False)

    def register_liquidation_callback(self, callback: Callable) -> None:
        """Register callback for liquidation events. Signature: (symbol, entry, stats)."""
        self._liq_callbacks.append(callback)

    # ─── Internal WebSocket Logic ───

    async def _ws_loop(self) -> None:
        """Outer loop: reconnection with exponential backoff + endpoint cycling."""
        backoff = 1
        consecutive_failures = 0
        while self._running:
            try:
                await self._ws_session()
                # Clean disconnect (max age reached) — reconnect immediately
                backoff = 1
                consecutive_failures = 0
            except asyncio.CancelledError:
                return
            except Exception as e:
                self._reconnect_count += 1
                consecutive_failures += 1
                # Cycle to next endpoint on failure
                self._endpoint_idx = (self._endpoint_idx + 1) % len(BINANCE_WS_ENDPOINTS)
                next_ep = BINANCE_WS_ENDPOINTS[self._endpoint_idx].split("//")[1].split("/")[0]
                # After 6 consecutive failures, enter long cooldown (5 min)
                if consecutive_failures >= 6:
                    cooldown = 300
                    logger.warning(
                        f"Binance WS {consecutive_failures} consecutive failures, "
                        f"cooling down {cooldown}s before retry on {next_ep}"
                    )
                    await asyncio.sleep(cooldown)
                    consecutive_failures = 0
                    backoff = 1
                else:
                    logger.warning(
                        f"Binance WS disconnected: {type(e).__name__}: {e} "
                        f"— reconnecting in {backoff}s to {next_ep}"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)

    async def _ws_session(self) -> None:
        """Single WebSocket session on the combined stream."""
        # Build combined stream URL
        streams = ["!miniTicker@arr"]
        for sym in BOOK_TICKER_ASSETS:
            streams.append(f"{sym}@bookTicker")
        stream_param = "/".join(streams)
        base = BINANCE_WS_ENDPOINTS[self._endpoint_idx % len(BINANCE_WS_ENDPOINTS)]
        url = f"{base}?streams={stream_param}"

        self._connect_time = time.monotonic()
        max_age_sec = WS_MAX_AGE_HOURS * 3600

        # Generous timeouts for CPU-starved VPS:
        # - ping_interval=30: less frequent pings = less event loop pressure
        # - ping_timeout=30: tolerate up to 30s of event loop lag
        # - open_timeout=60: TLS handshake can be slow under load
        async with websockets.connect(
            url,
            ping_interval=30,
            ping_timeout=30,
            close_timeout=10,
            open_timeout=60,
        ) as ws:
            logger.info(
                f"Binance combined WS connected ({len(streams)} streams: "
                f"miniTicker + {len(BOOK_TICKER_ASSETS)} bookTickers)"
            )

            async for raw in ws:
                if not self._running:
                    break
                # Max session age check
                if time.monotonic() - self._connect_time > max_age_sec:
                    logger.info("Binance WS max age reached, reconnecting")
                    break

                try:
                    msg = json.loads(raw)
                    stream = msg.get("stream", "")
                    data = msg.get("data")
                    if not data:
                        continue

                    if stream == "!miniTicker@arr":
                        self._handle_mini_ticker_arr(data)
                    elif stream.endswith("@bookTicker"):
                        self._handle_book_ticker(data)
                except Exception as e:
                    logger.debug(f"Binance WS parse error: {e}")

    # ─── Futures Liquidation WebSocket ───

    async def _futures_ws_loop(self) -> None:
        """Outer loop for Binance Futures !forceOrder@arr WS with reconnection."""
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(
                    BINANCE_FUTURES_WS,
                    ping_interval=20,
                    ping_timeout=60,
                    close_timeout=5,
                    open_timeout=30,
                ) as ws:
                    self._liq_connected = True
                    logger.info("Liquidation feed CONNECTED (Binance Futures !forceOrder@arr)")
                    backoff = 1

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            self._handle_liquidation(msg)
                        except Exception:
                            pass

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug(f"Futures WS error: {type(e).__name__}: {e}")

            self._liq_connected = False
            if not self._running:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    def _handle_liquidation(self, msg: dict) -> None:
        """Process a forceOrder liquidation message.

        Tracks per-symbol liquidation volume and detects cascades.
        Side="SELL" → long liquidated (forced sell). Side="BUY" → short squeezed.
        """
        data = msg.get("o", msg)

        symbol_raw = data.get("s", "")
        side = data.get("S", "")
        qty = float(data.get("q", 0) or 0)
        price = float(data.get("ap", data.get("p", 0)) or 0)

        if not symbol_raw or price <= 0 or qty <= 0:
            return

        symbol = _normalize_symbol(symbol_raw)
        usd_value = qty * price
        liq_type = "long" if side == "SELL" else "short"
        now = time.time()
        self._liq_count += 1

        entry = {
            "symbol": symbol,
            "raw_symbol": symbol_raw,
            "side": side,
            "type": liq_type,
            "qty": qty,
            "price": price,
            "usd_value": usd_value,
            "timestamp": now,
        }
        self._liq_history.append(entry)

        # Update per-symbol rolling stats
        if symbol not in self._liquidations:
            self._liquidations[symbol] = {
                "long_usd_1m": 0.0,
                "short_usd_1m": 0.0,
                "long_usd_5m": 0.0,
                "short_usd_5m": 0.0,
                "long_count_5m": 0,
                "short_count_5m": 0,
                "last_cascade_time": 0,
                "cascade_active": False,
                "squeeze_active": False,
                "last_update": now,
                "history_1m": deque(maxlen=200),
                "history_5m": deque(maxlen=1000),
            }

        stats = self._liquidations[symbol]
        stats["history_1m"].append(entry)
        stats["history_5m"].append(entry)

        # Prune old entries and recompute rolling sums
        cutoff_1m = now - 60
        cutoff_5m = now - 300

        stats["history_1m"] = deque(
            (e for e in stats["history_1m"] if e["timestamp"] > cutoff_1m),
            maxlen=200,
        )
        stats["history_5m"] = deque(
            (e for e in stats["history_5m"] if e["timestamp"] > cutoff_5m),
            maxlen=1000,
        )

        stats["long_usd_1m"] = sum(e["usd_value"] for e in stats["history_1m"] if e["type"] == "long")
        stats["short_usd_1m"] = sum(e["usd_value"] for e in stats["history_1m"] if e["type"] == "short")
        stats["long_usd_5m"] = sum(e["usd_value"] for e in stats["history_5m"] if e["type"] == "long")
        stats["short_usd_5m"] = sum(e["usd_value"] for e in stats["history_5m"] if e["type"] == "short")
        stats["long_count_5m"] = sum(1 for e in stats["history_5m"] if e["type"] == "long")
        stats["short_count_5m"] = sum(1 for e in stats["history_5m"] if e["type"] == "short")
        stats["last_update"] = now

        # Cascade detection: long liquidations > threshold in 5 minutes
        cascade_threshold = 5_000_000 if "BTC" in symbol_raw else 500_000

        if stats["long_usd_5m"] > cascade_threshold:
            if not stats["cascade_active"]:
                stats["cascade_active"] = True
                stats["last_cascade_time"] = now
                logger.info(
                    f"*** LIQUIDATION CASCADE: {symbol} ***  "
                    f"Long liqs (5m): ${stats['long_usd_5m']:,.0f} "
                    f"({stats['long_count_5m']} orders)  "
                    f"Short liqs (5m): ${stats['short_usd_5m']:,.0f}  "
                    f"Imbalance: {stats['long_usd_5m'] / max(stats['short_usd_5m'], 1):.1f}x longs"
                )
        elif stats["long_usd_5m"] < cascade_threshold * 0.3:
            if stats["cascade_active"]:
                stats["cascade_active"] = False
                logger.info(f"Liquidation cascade ENDED: {symbol}")

        # Short squeeze detection (symmetric)
        if stats["short_usd_5m"] > cascade_threshold:
            if not stats.get("squeeze_active"):
                stats["squeeze_active"] = True
                logger.info(
                    f"*** SHORT SQUEEZE: {symbol} ***  "
                    f"Short liqs (5m): ${stats['short_usd_5m']:,.0f}"
                )
        elif stats["short_usd_5m"] < cascade_threshold * 0.3:
            if stats.get("squeeze_active"):
                stats["squeeze_active"] = False
                logger.info(f"Short squeeze ENDED: {symbol}")

        # Fire callbacks
        for cb in self._liq_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(symbol, entry, stats))
                else:
                    cb(symbol, entry, stats)
            except Exception:
                pass

    def _handle_mini_ticker_arr(self, tickers: list) -> None:
        """Process !miniTicker@arr batch.

        Each item: {"e":"24hrMiniTicker","s":"BTCUSDT","c":"67800.00",
                     "o":"67500.00","h":"68200.00","l":"67100.00",
                     "v":"12345.678","q":"836000000.00"}

        miniTicker has NO bid/ask — use close as bid=ask=last_price.
        For key assets, bookTicker will overlay real bid/ask.
        """
        update_count = 0
        for t in tickers:
            raw_sym = t.get("s", "")
            if not raw_sym:
                continue
            symbol = _normalize_symbol(raw_sym)
            if "/" not in symbol:
                continue

            close = t.get("c")
            volume = t.get("v", "0")
            if not close:
                continue

            close_d = Decimal(close)
            if close_d <= 0:
                continue

            existing = self._tickers.get(symbol)
            if existing and symbol in self._last_book_update:
                # bookTicker has real bid/ask — only update last_price and volume
                existing["last_price"] = close_d
                existing["volume_24h"] = Decimal(volume) if volume else Decimal("0")
            else:
                # No bookTicker overlay — bid=ask=close
                self._tickers[symbol] = {
                    "symbol": symbol,
                    "bid": close_d,
                    "ask": close_d,
                    "last_price": close_d,
                    "volume_24h": Decimal(volume) if volume else Decimal("0"),
                }
            update_count += 1

        if update_count > 0:
            self._last_update = time.monotonic()

    def _handle_book_ticker(self, data: dict) -> None:
        """Process individual bookTicker update.

        {"u":12345,"s":"BTCUSDT","b":"67758.00","B":"1.234",
         "a":"67759.00","A":"0.567"}

        Overlays real bid/ask onto the ticker for key assets.
        """
        raw_sym = data.get("s", "")
        if not raw_sym:
            return
        symbol = _normalize_symbol(raw_sym)

        bid = data.get("b")
        ask = data.get("a")
        if not bid or not ask:
            return

        bid_d = Decimal(bid)
        ask_d = Decimal(ask)
        if bid_d <= 0 or ask_d <= 0:
            return

        existing = self._tickers.get(symbol)
        if existing:
            existing["bid"] = bid_d
            existing["ask"] = ask_d
            # Update last_price to midpoint if we have real bid/ask
            existing["last_price"] = (bid_d + ask_d) / 2
        else:
            self._tickers[symbol] = {
                "symbol": symbol,
                "bid": bid_d,
                "ask": ask_d,
                "last_price": (bid_d + ask_d) / 2,
                "volume_24h": Decimal("0"),
            }

        self._last_book_update[symbol] = time.monotonic()
        # Also count as a general update
        self._last_update = time.monotonic()


class HybridTickerProvider:
    """Wraps BinanceUnifiedPriceFeed (primary) + MEXCClient (REST fallback).

    Returns the same signature as TriangularArbitrage._get_tickers():
        (Dict[str, dict], str, float)  ->  (tickers, source, age_ms)
    """

    def __init__(self, binance_feed: BinanceUnifiedPriceFeed, mexc_client):
        self._feed = binance_feed
        self._mexc = mexc_client
        self._fallback_count: int = 0
        self._primary_count: int = 0

    async def get_tickers(self) -> Tuple[Dict[str, dict], str, float]:
        """Get tickers: prefer Binance WS, fallback to MEXC REST."""
        # Primary: Binance WS
        if self._feed.is_healthy():
            tickers = self._feed.get_tickers()
            if tickers and len(tickers) > 100:
                self._primary_count += 1
                age_ms = self._feed.get_age_ms()
                return tickers, "binance_ws", age_ms

        # Fallback: MEXC REST
        self._fallback_count += 1
        if self._fallback_count % 50 == 1:
            logger.info(
                f"HybridTickerProvider: MEXC REST fallback "
                f"(primary={self._primary_count}, fallback={self._fallback_count})"
            )
        tickers = await self._mexc.get_all_tickers()
        return tickers, "mexc_rest", 0.0

    def get_stats(self) -> dict:
        """Stats for monitoring."""
        return {
            "primary_count": self._primary_count,
            "fallback_count": self._fallback_count,
            "feed_healthy": self._feed.is_healthy(),
            "feed_ticker_count": len(self._feed._tickers),
        }
