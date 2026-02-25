"""
MEXC exchange client — PRIMARY exchange for arbitrage.
Uses ccxt for REST API, real WebSocket for streaming market data.

CRITICAL: MEXC 0% maker fee is our structural edge.
All orders default to LIMIT_MAKER (post-only).
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Callable, Awaitable, Dict, List

import aiohttp
import ccxt.async_support as ccxt_async
import websockets

from .base import (
    ExchangeClient, OrderBook, OrderBookLevel, OrderRequest, OrderResult,
    OrderSide, OrderType, OrderStatus, TimeInForce, Balance, FundingRate,
    Trade,
)

logger = logging.getLogger("arb.mexc")

# Akamai WAF blocks default aiohttp user-agent — use a browser-like UA
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}

# MEXC documented rate limit: 5 orders per second
ORDER_RATE_LIMIT_PER_SECOND = 5

# WebSocket constants
WS_ENDPOINT = "wss://wbs-api.mexc.com/ws"
WS_PING_INTERVAL = 15       # seconds
WS_MAX_AGE_HOURS = 23       # reconnect before 24h limit
WS_MAX_RECONNECT = 3        # attempts before REST fallback
WS_FALLBACK_DURATION = 180  # seconds of REST fallback before retrying WS
WS_DEPTH_LEVELS = 20


class _WSBlockedError(Exception):
    """Raised when MEXC WS subscriptions are geo-blocked."""
    pass


class MEXCClient(ExchangeClient):

    def __init__(self, api_key: str = "", api_secret: str = "", paper_trading: bool = True):
        self.paper_trading = paper_trading
        self._api_key = api_key
        self._api_secret = api_secret
        self._exchange: Optional[ccxt_async.mexc] = None
        self._symbol_info_cache: Dict[str, dict] = {}
        self._ws_callbacks: Dict[str, Callable] = {}      # symbol → depth callback
        self._trade_callbacks: Dict[str, Callable] = {}    # symbol → trade callback
        self._ws_running = False
        self._last_books: Dict[str, OrderBook] = {}
        self._ws_task: Optional[asyncio.Task] = None
        # All-ticker WebSocket feed for triangular scanner
        self._ws_tickers: Dict[str, dict] = {}
        self._ws_ticker_last_update: float = 0.0
        self._ws_ticker_running = False
        self._ws_ticker_task: Optional[asyncio.Task] = None
        self._ws_ticker_callback: Optional[Callable] = None
        # Volume limiter (set by orchestrator for fill rate degradation)
        self.volume_limiter = None
        # Shared aiohttp session for REST calls (reuses TCP connections)
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> None:
        config = {
            'apiKey': self._api_key,
            'secret': self._api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'fetchMarkets': ['spot'],  # Skip contract API (geo-blocked on VPS)
            },
        }
        if not self._api_key:
            config.pop('apiKey')
            config.pop('secret')

        self._exchange = ccxt_async.mexc(config)
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5),
            headers=_HTTP_HEADERS,
        )

        try:
            await self._exchange.load_markets()
            logger.info(f"MEXC connected — {len(self._exchange.markets)} markets loaded")
        except Exception as e:
            logger.warning(f"MEXC market load failed (read-only mode): {e}")

    async def disconnect(self) -> None:
        self._ws_running = False
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    # --- Market Data ---

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        """Fetch order book via direct REST, with sync thread fallback.

        Primary: aiohttp (async, efficient). Falls back to urllib in a thread
        when the event loop is CPU-starved (common during VPS startup).
        """
        try:
            return await asyncio.wait_for(
                self._fetch_order_book_direct(symbol, depth), timeout=3
            )
        except (asyncio.TimeoutError, Exception):
            # Event loop too busy for aiohttp — use sync HTTP in thread
            return await asyncio.to_thread(self._fetch_order_book_sync, symbol, depth)

    async def subscribe_order_book(
        self, symbol: str,
        callback: Callable[[OrderBook], Awaitable[None]],
        depth: int = 20,
    ) -> None:
        self._ws_callbacks[symbol] = callback
        if not self._ws_running:
            self._ws_running = True
            self._ws_task = asyncio.create_task(self._ws_main_loop())

    async def subscribe_trades(
        self, symbol: str,
        callback: Callable[[Trade], Awaitable[None]],
    ) -> None:
        self._trade_callbacks[symbol] = callback
        # WS loop already started by subscribe_order_book

    # ========== WebSocket Engine ==========

    async def _ws_main_loop(self):
        """Outer loop: reconnection state machine."""
        reconnect_attempts = 0
        while self._ws_running:
            try:
                await self._ws_connect_and_run()
                # Clean exit (e.g. max age) — reset counter
                reconnect_attempts = 0
            except asyncio.CancelledError:
                return
            except _WSBlockedError:
                # Geo-blocked — skip WS entirely, run REST permanently
                logger.warning("MEXC WS geo-blocked — using REST polling permanently")
                await self._rest_fallback_loop_permanent()
                return
            except Exception as e:
                reconnect_attempts += 1
                if reconnect_attempts >= WS_MAX_RECONNECT:
                    logger.warning(f"MEXC WS failed {reconnect_attempts}x, falling back to REST for {WS_FALLBACK_DURATION}s")
                    await self._rest_fallback_loop()
                    reconnect_attempts = 0
                else:
                    backoff = min(2 ** reconnect_attempts, 30)
                    logger.warning(f"MEXC WS disconnected: {e} — reconnecting in {backoff}s (attempt {reconnect_attempts})")
                    await asyncio.sleep(backoff)

    async def _ws_connect_and_run(self):
        """Single WebSocket session lifecycle."""
        connect_time = time.monotonic()
        max_age_sec = WS_MAX_AGE_HOURS * 3600
        self._ws_got_data = False

        async with websockets.connect(
            WS_ENDPOINT,
            ping_interval=None,  # We send our own PING
            close_timeout=5,
            open_timeout=30,  # VPS event loop is CPU-starved during startup
        ) as ws:
            logger.info("MEXC WebSocket connected")

            # Subscribe all current symbols
            all_syms = set(list(self._ws_callbacks.keys()) + list(self._trade_callbacks.keys()))
            for symbol in all_syms:
                await self._ws_subscribe_symbol(ws, symbol)
                await asyncio.sleep(0.1)

            # Run ping + message loops concurrently
            ping_task = asyncio.create_task(self._ws_ping_loop(ws))
            data_watchdog = asyncio.create_task(self._ws_data_watchdog())
            try:
                async for raw in ws:
                    if not self._ws_running:
                        break
                    if time.monotonic() - connect_time > max_age_sec:
                        logger.info("MEXC WS max age reached (23h), reconnecting")
                        break
                    try:
                        self._ws_handle_message(raw)
                    except _WSBlockedError:
                        raise  # Propagate to trigger fallback
                    except Exception as e:
                        logger.debug(f"MEXC WS message parse error: {e}")
            finally:
                ping_task.cancel()
                data_watchdog.cancel()

    async def _ws_data_watchdog(self):
        """If no market data arrives within 10s of subscribing, raise to trigger fallback."""
        await asyncio.sleep(10)
        if not self._ws_got_data:
            logger.warning("MEXC WS: no market data received in 10s — likely geo-blocked")
            raise _WSBlockedError("No data received")

    async def _ws_subscribe_symbol(self, ws, symbol: str):
        """Send depth + trade subscription messages for one symbol."""
        raw_sym = self._normalized_to_mexc_sym(symbol)

        # Depth subscription
        depth_msg = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.limit.depth.v3.api@{raw_sym}@{WS_DEPTH_LEVELS}"]
        }
        await ws.send(json.dumps(depth_msg))

        # Trade subscription
        trade_msg = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.deals.v3.api@{raw_sym}"]
        }
        await ws.send(json.dumps(trade_msg))
        logger.debug(f"MEXC WS subscribed: {symbol} ({raw_sym})")

    async def _ws_ping_loop(self, ws):
        """Send PING every 15 seconds to keep connection alive."""
        try:
            while self._ws_running:
                await asyncio.sleep(WS_PING_INTERVAL)
                await ws.send(json.dumps({"method": "PING"}))
        except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
            pass

    def _ws_handle_message(self, raw: str):
        """Parse JSON message and dispatch to depth/trade handlers."""
        msg = json.loads(raw)

        if isinstance(msg, dict):
            # Check for subscription failure (geo-block)
            msg_text = msg.get("msg", "")
            if "Blocked" in msg_text:
                logger.warning(f"MEXC WS subscription blocked: {msg_text}")
                raise _WSBlockedError(msg_text)
            # Ignore PONG and successful subscription confirmations
            if msg_text == "PONG" or msg.get("id") is not None:
                return

        channel = msg.get("c", "")

        if "limit.depth" in channel:
            self._ws_got_data = True
            self._ws_handle_depth(msg)
        elif "public.deals" in channel:
            self._ws_got_data = True
            self._ws_handle_trade(msg)

    def _ws_handle_depth(self, msg: dict):
        """Parse depth message → OrderBook → invoke callback."""
        data = msg.get("d", {})
        channel = msg.get("c", "")

        # Extract symbol from channel: spot@public.limit.depth.v3.api@BTCUSDT@20
        parts = channel.split("@")
        if len(parts) < 3:
            return
        raw_sym = parts[2]
        symbol = self._mexc_sym_to_normalized(raw_sym)

        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])

        bids = [OrderBookLevel(Decimal(str(b["p"])), Decimal(str(b["v"]))) for b in bids_raw]
        asks = [OrderBookLevel(Decimal(str(a["p"])), Decimal(str(a["v"]))) for a in asks_raw]

        ts = data.get("t") or data.get("r")
        dt = datetime.utcfromtimestamp(ts / 1000) if ts else datetime.utcnow()

        book = OrderBook(exchange="mexc", symbol=symbol, timestamp=dt, bids=bids, asks=asks)
        self._last_books[symbol] = book

        cb = self._ws_callbacks.get(symbol)
        if cb:
            asyncio.get_event_loop().create_task(cb(book))

    def _ws_handle_trade(self, msg: dict):
        """Parse deals message → Trade → invoke callback."""
        data = msg.get("d", {})
        channel = msg.get("c", "")

        parts = channel.split("@")
        if len(parts) < 3:
            return
        raw_sym = parts[2]
        symbol = self._mexc_sym_to_normalized(raw_sym)

        deals = data.get("deals", [])
        cb = self._trade_callbacks.get(symbol)
        if not cb:
            return

        for deal in deals:
            side = OrderSide.BUY if deal.get("S") == 1 else OrderSide.SELL
            ts = deal.get("t", 0)
            trade = Trade(
                exchange="mexc",
                symbol=symbol,
                trade_id=str(deal.get("t", "")),
                price=Decimal(str(deal.get("p", "0"))),
                quantity=Decimal(str(deal.get("v", "0"))),
                side=side,
                timestamp=datetime.utcfromtimestamp(ts / 1000) if ts else datetime.utcnow(),
            )
            asyncio.get_event_loop().create_task(cb(trade))

    async def _rest_fallback_loop(self):
        """REST polling fallback when WS fails repeatedly.
        Sequential to avoid MEXC rate limiting (parallel requests → 403).
        """
        end_time = time.monotonic() + WS_FALLBACK_DURATION
        symbols = list(self._ws_callbacks.items())
        logger.info(f"MEXC entering REST fallback mode ({len(symbols)} symbols, sequential)")
        ok_count = 0
        fail_count = 0
        while self._ws_running and time.monotonic() < end_time:
            for sym, cb in symbols:
                if not self._ws_running or time.monotonic() >= end_time:
                    break
                try:
                    book = await self._fetch_order_book_direct(sym, depth=20)
                    self._last_books[sym] = book
                    await cb(book)
                    ok_count += 1
                except Exception as e:
                    fail_count += 1
                    logger.debug(f"MEXC REST fallback error {sym}: {e}")
                await asyncio.sleep(0.15)  # 150ms between calls
            await asyncio.sleep(1.0)
        logger.info(f"MEXC REST fallback complete — {ok_count} ok, {fail_count} failed")

    async def _rest_fallback_loop_permanent(self):
        """Permanent REST polling when WS is geo-blocked.
        Sequential to avoid MEXC rate limiting.
        """
        symbols = list(self._ws_callbacks.items())
        logger.info(f"MEXC running permanent REST polling (WS unavailable) — {len(symbols)} symbols")
        poll_count = 0
        ok_count = 0
        fail_count = 0
        while self._ws_running:
            for sym, cb in list(self._ws_callbacks.items()):
                if not self._ws_running:
                    break
                try:
                    book = await self._fetch_order_book_direct(sym, depth=20)
                    self._last_books[sym] = book
                    await cb(book)
                    ok_count += 1
                except Exception as e:
                    fail_count += 1
                    logger.warning(f"MEXC REST poll error {sym}: {e}")
                await asyncio.sleep(0.15)  # 150ms between calls
            poll_count += 1
            if poll_count % 10 == 1:  # Log every ~10 cycles
                logger.info(f"MEXC REST poll #{poll_count}: {len(self._ws_callbacks)} symbols, {ok_count} ok/{fail_count} fail")
            await asyncio.sleep(1.0)

    # ========== All-Ticker WebSocket Feed ==========

    async def subscribe_all_tickers(self, callback: Optional[Callable] = None):
        """Subscribe to MEXC mini-ticker WebSocket stream for ALL symbols.

        Updates self._ws_tickers in real-time. The callback (if set) is
        called on each batch with the full ticker dict.

        Message format from MEXC:
          {"s": "BTCUSDT", "p": "67758.64", "bid": "67758.00", "ask": "67759.00", ...}
        """
        self._ws_ticker_callback = callback
        if self._ws_ticker_running:
            return
        self._ws_ticker_running = True
        self._ws_ticker_task = asyncio.create_task(self._ws_ticker_loop())
        logger.info("All-ticker WebSocket subscription started")

    async def _ws_ticker_loop(self):
        """Outer loop: reconnection for all-ticker stream."""
        reconnect_attempts = 0
        while self._ws_ticker_running:
            try:
                await self._ws_ticker_connect()
                reconnect_attempts = 0
            except asyncio.CancelledError:
                return
            except _WSBlockedError:
                logger.warning("MEXC all-ticker WS geo-blocked — ticker feed unavailable")
                self._ws_ticker_running = False
                return
            except Exception as e:
                reconnect_attempts += 1
                backoff = min(2 ** reconnect_attempts, 30)
                logger.warning(f"Ticker WS disconnected: {e} — reconnecting in {backoff}s")
                await asyncio.sleep(backoff)

    async def _ws_ticker_connect(self):
        """Single WebSocket session for all-ticker stream."""
        connect_time = time.monotonic()
        max_age_sec = WS_MAX_AGE_HOURS * 3600
        got_data = False

        async with websockets.connect(
            WS_ENDPOINT,
            ping_interval=None,
            close_timeout=5,
            open_timeout=30,
        ) as ws:
            logger.info("MEXC all-ticker WebSocket connected")

            # Subscribe to mini-tickers for all symbols
            sub_msg = {
                "method": "SUBSCRIPTION",
                "params": ["spot@public.miniTickers.v3.api@UTC+0"]
            }
            await ws.send(json.dumps(sub_msg))

            # Ping loop
            ping_task = asyncio.create_task(self._ws_ping_loop(ws))

            # Watchdog: if no data in 10s, probably geo-blocked
            async def watchdog():
                await asyncio.sleep(10)
                if not got_data:
                    raise _WSBlockedError("No ticker data received")

            watchdog_task = asyncio.create_task(watchdog())

            try:
                async for raw in ws:
                    if not self._ws_ticker_running:
                        break
                    if time.monotonic() - connect_time > max_age_sec:
                        logger.info("Ticker WS max age reached, reconnecting")
                        break
                    try:
                        msg = json.loads(raw)
                        if isinstance(msg, dict):
                            msg_text = msg.get("msg", "")
                            if "Blocked" in msg_text:
                                raise _WSBlockedError(msg_text)
                            if msg_text == "PONG" or msg.get("id") is not None:
                                continue

                        channel = msg.get("c", "")
                        if "miniTickers" in channel:
                            got_data = True
                            watchdog_task.cancel()
                            self._handle_mini_tickers(msg)
                    except _WSBlockedError:
                        raise
                    except Exception as e:
                        logger.debug(f"Ticker WS parse error: {e}")
            finally:
                ping_task.cancel()
                watchdog_task.cancel()

    def _handle_mini_tickers(self, msg: dict):
        """Parse mini-ticker batch and update internal cache."""
        data = msg.get("d", {})
        tickers = data if isinstance(data, list) else data.get("data", [])
        if not isinstance(tickers, list):
            # Single ticker update
            tickers = [data] if isinstance(data, dict) and "s" in data else []

        update_count = 0
        for t in tickers:
            raw_sym = t.get("s", "")
            if not raw_sym:
                continue
            symbol = self._mexc_sym_to_normalized(raw_sym)
            if '/' not in symbol:
                continue

            bid = t.get("bid") or t.get("b")
            ask = t.get("ask") or t.get("a")
            last = t.get("p") or t.get("c")  # last price

            bid_d = Decimal(str(bid)) if bid else Decimal('0')
            ask_d = Decimal(str(ask)) if ask else Decimal('0')
            last_d = Decimal(str(last)) if last else Decimal('0')

            # Use last price as fallback for bid/ask
            if bid_d <= 0 and last_d > 0:
                bid_d = last_d
            if ask_d <= 0 and last_d > 0:
                ask_d = last_d

            if bid_d > 0 or ask_d > 0:
                self._ws_tickers[symbol] = {
                    'symbol': symbol,
                    'last_price': last_d if last_d > 0 else (bid_d + ask_d) / 2,
                    'bid': bid_d,
                    'ask': ask_d,
                    'volume_24h': Decimal('0'),
                }
                update_count += 1

        if update_count > 0:
            self._ws_ticker_last_update = time.time()

        # Invoke callback if set
        if self._ws_ticker_callback and update_count > 0:
            try:
                self._ws_ticker_callback(self._ws_tickers)
            except Exception as e:
                logger.debug(f"Ticker callback error: {e}")

    def get_ws_tickers(self) -> Optional[Dict[str, dict]]:
        """Return WebSocket tickers if fresh (< 10s old), else None."""
        if self._ws_tickers and self._ws_ticker_last_update > 0:
            age = time.time() - self._ws_ticker_last_update
            if age < 10:
                return self._ws_tickers
        return None

    def get_ws_ticker_age_ms(self) -> float:
        """Age of the most recent WS ticker update in milliseconds."""
        if self._ws_ticker_last_update <= 0:
            return float('inf')
        return (time.time() - self._ws_ticker_last_update) * 1000

    # ========== Symbol Conversion ==========

    def _normalized_to_mexc_sym(self, symbol: str) -> str:
        """BTC/USDT → BTCUSDT (uppercase, no slash)."""
        return symbol.replace("/", "").upper()

    def _mexc_sym_to_normalized(self, raw: str) -> str:
        """BTCUSDT → BTC/USDT. Best-effort using known quote currencies."""
        raw = raw.upper()
        for quote in ("USDT", "USDC", "BTC", "ETH"):
            if raw.endswith(quote):
                base = raw[: -len(quote)]
                if base:
                    return f"{base}/{quote}"
        return raw

    # ========== REST endpoints (unchanged) ==========

    async def get_ticker(self, symbol: str) -> dict:
        ticker = await self._exchange.fetch_ticker(symbol)
        return {
            'symbol': symbol,
            'last_price': Decimal(str(ticker.get('last', 0))),
            'bid': Decimal(str(ticker.get('bid', 0))),
            'ask': Decimal(str(ticker.get('ask', 0))),
            'volume_24h': Decimal(str(ticker.get('baseVolume', 0))),
            'quote_volume_24h': Decimal(str(ticker.get('quoteVolume', 0))),
            'timestamp': datetime.utcnow(),
        }

    async def get_all_tickers(self) -> Dict[str, dict]:
        """Fetch all tickers via direct REST (avoids ccxt load_markets overhead)."""
        return await self._fetch_all_tickers_direct()

    async def _fetch_all_tickers_direct(self) -> Dict[str, dict]:
        """Fetch all tickers directly from MEXC spot REST API, bypassing ccxt.

        Retries on 403/429 with backoff.
        """
        url = "https://api.mexc.com/api/v3/ticker/bookTicker"
        session = self._http_session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5), headers=_HTTP_HEADERS)
        close_after = self._http_session is None
        data = None
        try:
            for attempt in range(3):
                async with session.get(url) as resp:
                    if resp.status in (403, 429):
                        wait = (attempt + 1) * 2
                        if attempt < 2:
                            logger.debug(f"MEXC rate limited ({resp.status}) on bookTicker, retry in {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        raise Exception(f"MEXC REST {resp.status} after 3 retries")
                    if resp.status != 200:
                        raise Exception(f"MEXC REST {resp.status}")
                    data = await resp.json()
                    break
        finally:
            if close_after:
                await session.close()
        if data is None:
            raise Exception("MEXC bookTicker fetch returned no data")
        result = {}
        for t in data:
            raw_sym = t.get('symbol', '')
            symbol = self._mexc_sym_to_normalized(raw_sym)
            if '/' not in symbol:
                continue
            bid = Decimal(str(t.get('bidPrice', '0') or '0'))
            ask = Decimal(str(t.get('askPrice', '0') or '0'))
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else Decimal('0')
            result[symbol] = {
                'symbol': symbol,
                'last_price': mid,
                'bid': bid,
                'ask': ask,
                'volume_24h': Decimal('0'),  # bookTicker doesn't include volume
            }
        return result

    async def get_funding_rate(self, symbol: str) -> FundingRate:
        try:
            fr = await self._exchange.fetch_funding_rate(symbol)
            return FundingRate(
                exchange="mexc",
                symbol=symbol,
                current_rate=Decimal(str(fr.get('fundingRate', 0) or 0)),
                predicted_rate=Decimal(str(fr.get('fundingRate', 0) or 0)),
                next_funding_time=datetime.utcnow(),
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            logger.debug(f"MEXC funding rate fetch failed for {symbol}: {e}")
            return FundingRate(
                exchange="mexc", symbol=symbol,
                current_rate=Decimal('0'), predicted_rate=None,
                next_funding_time=datetime.utcnow(), timestamp=datetime.utcnow(),
            )

    # --- Trading ---

    async def place_order(self, order: OrderRequest) -> OrderResult:
        if self.paper_trading:
            return self._paper_fill(order)

        ccxt_symbol = order.symbol
        params = {}

        # CRITICAL: Enforce maker-only on MEXC
        if order.order_type == OrderType.LIMIT_MAKER:
            order_type_str = 'limit'
            params['timeInForce'] = 'GTX'  # Post-only
        elif order.order_type == OrderType.MARKET:
            order_type_str = 'market'
        else:
            order_type_str = 'limit'

        side_str = order.side.value
        price = float(order.price) if order.price else None
        amount = float(order.quantity)

        try:
            result = await self._exchange.create_order(
                symbol=ccxt_symbol,
                type=order_type_str,
                side=side_str,
                amount=amount,
                price=price,
                params=params,
            )
            return self._parse_order_result(result, order)
        except Exception as e:
            logger.error(f"MEXC order failed: {e}")
            return OrderResult(
                exchange="mexc", symbol=order.symbol,
                order_id="", client_order_id=order.client_order_id,
                status=OrderStatus.REJECTED, side=order.side,
                order_type=order.order_type,
                requested_quantity=order.quantity,
                filled_quantity=Decimal('0'),
                average_fill_price=None, fee_amount=Decimal('0'),
                fee_currency="USDT", timestamp=datetime.utcnow(),
                raw_response={"error": str(e)},
            )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if self.paper_trading:
            return True
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"MEXC cancel failed: {e}")
            return False

    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult:
        result = await self._exchange.fetch_order(order_id, symbol)
        return self._parse_order_result(result, None)

    # --- Account ---

    async def get_balances(self) -> Dict[str, Balance]:
        if self.paper_trading:
            return self._paper_balances()
        raw = await self._exchange.fetch_balance()
        balances = {}
        for currency, info in raw.get('total', {}).items():
            total = Decimal(str(info or 0))
            if total > 0:
                free = Decimal(str(raw.get('free', {}).get(currency, 0) or 0))
                locked = Decimal(str(raw.get('used', {}).get(currency, 0) or 0))
                balances[currency] = Balance(
                    exchange="mexc", currency=currency,
                    free=free, locked=locked, total=total,
                )
        return balances

    async def get_balance(self, currency: str) -> Balance:
        balances = await self.get_balances()
        bal = balances.get(currency)
        if bal:
            return bal
        # Paper trading: return generous default for any unlisted token
        if self.paper_trading:
            return Balance(
                exchange="mexc", currency=currency,
                free=Decimal('100000'), locked=Decimal('0'), total=Decimal('100000'),
            )
        return Balance(
            exchange="mexc", currency=currency,
            free=Decimal('0'), locked=Decimal('0'), total=Decimal('0'),
        )

    # --- Exchange Info ---

    async def get_trading_fees(self, symbol: str) -> dict:
        return {"maker": Decimal('0.0000'), "taker": Decimal('0.0005')}

    async def get_symbol_info(self, symbol: str) -> dict:
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        if self._exchange and self._exchange.markets and symbol in self._exchange.markets:
            market = self._exchange.markets[symbol]
            info = {
                'symbol': symbol,
                'price_precision': market.get('precision', {}).get('price', 8),
                'quantity_precision': market.get('precision', {}).get('amount', 8),
                'min_quantity': Decimal(str(market.get('limits', {}).get('amount', {}).get('min', 0) or 0)),
                'min_notional': Decimal(str(market.get('limits', {}).get('cost', {}).get('min', 0) or 0)),
            }
            self._symbol_info_cache[symbol] = info
            return info

        return {
            'symbol': symbol, 'price_precision': 8,
            'quantity_precision': 8, 'min_quantity': Decimal('0'),
            'min_notional': Decimal('0'),
        }

    # --- Direct REST (bypasses ccxt, avoids contract API geo-block) ---

    async def _fetch_order_book_direct(self, symbol: str, depth: int = 20) -> OrderBook:
        """Fetch order book directly from MEXC spot REST API, bypassing ccxt.

        Uses shared HTTP session with browser-like User-Agent to avoid Akamai WAF blocks.
        """
        raw_sym = self._normalized_to_mexc_sym(symbol)
        url = f"https://api.mexc.com/api/v3/depth?symbol={raw_sym}&limit={depth}"
        session = self._http_session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5), headers=_HTTP_HEADERS)
        close_after = self._http_session is None

        try:
            for attempt in range(2):
                async with session.get(url) as resp:
                    if resp.status in (403, 429):
                        if attempt < 1:
                            await asyncio.sleep(0.5)
                            continue
                        raise Exception(f"MEXC REST {resp.status}")
                    if resp.status != 200:
                        raise Exception(f"MEXC REST {resp.status}: {await resp.text()}")
                    data = await resp.json()

                bids = [OrderBookLevel(Decimal(str(b[0])), Decimal(str(b[1]))) for b in data.get('bids', [])[:depth]]
                asks = [OrderBookLevel(Decimal(str(a[0])), Decimal(str(a[1]))) for a in data.get('asks', [])[:depth]]
                ts = data.get('timestamp')
                dt = datetime.utcfromtimestamp(ts / 1000) if ts else datetime.utcnow()
                return OrderBook(exchange="mexc", symbol=symbol, timestamp=dt, bids=bids, asks=asks)
        finally:
            if close_after:
                await session.close()

        raise Exception(f"MEXC REST exhausted retries for {symbol}")

    def _fetch_order_book_sync(self, symbol: str, depth: int = 20) -> OrderBook:
        """Synchronous HTTP fallback for order book fetch.

        Uses urllib (stdlib) in a thread pool — bypasses asyncio event loop
        entirely. Used when aiohttp times out due to CPU-starved event loop.
        """
        import urllib.request
        import json as _json

        raw_sym = self._normalized_to_mexc_sym(symbol)
        url = f"https://api.mexc.com/api/v3/depth?symbol={raw_sym}&limit={depth}"
        req = urllib.request.Request(url, headers=_HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read())

        bids = [OrderBookLevel(Decimal(str(b[0])), Decimal(str(b[1]))) for b in data.get('bids', [])[:depth]]
        asks = [OrderBookLevel(Decimal(str(a[0])), Decimal(str(a[1]))) for a in data.get('asks', [])[:depth]]
        ts = data.get('timestamp')
        dt = datetime.utcfromtimestamp(ts / 1000) if ts else datetime.utcnow()
        return OrderBook(exchange="mexc", symbol=symbol, timestamp=dt, bids=bids, asks=asks)

    # --- Helpers ---

    def _parse_order_book(self, raw: dict, symbol: str) -> OrderBook:
        bids = [OrderBookLevel(Decimal(str(b[0])), Decimal(str(b[1]))) for b in raw.get('bids', [])[:20]]
        asks = [OrderBookLevel(Decimal(str(a[0])), Decimal(str(a[1]))) for a in raw.get('asks', [])[:20]]
        ts = raw.get('timestamp')
        dt = datetime.utcfromtimestamp(ts / 1000) if ts else datetime.utcnow()
        return OrderBook(exchange="mexc", symbol=symbol, timestamp=dt, bids=bids, asks=asks)

    def _parse_order_result(self, raw: dict, req: Optional[OrderRequest]) -> OrderResult:
        status_map = {
            'open': OrderStatus.OPEN, 'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED, 'cancelled': OrderStatus.CANCELLED,
            'expired': OrderStatus.CANCELLED, 'rejected': OrderStatus.REJECTED,
        }
        raw_status = raw.get('status', 'open')
        filled = Decimal(str(raw.get('filled', 0) or 0))
        requested = Decimal(str(raw.get('amount', 0) or 0))
        if filled > 0 and filled < requested:
            status = OrderStatus.PARTIALLY_FILLED
        else:
            status = status_map.get(raw_status, OrderStatus.OPEN)

        fee_info = raw.get('fee', {}) or {}
        return OrderResult(
            exchange="mexc",
            symbol=raw.get('symbol', req.symbol if req else ''),
            order_id=str(raw.get('id', '')),
            client_order_id=raw.get('clientOrderId', req.client_order_id if req else None),
            status=status,
            side=OrderSide(raw.get('side', 'buy')),
            order_type=OrderType.LIMIT,
            requested_quantity=requested,
            filled_quantity=filled,
            average_fill_price=Decimal(str(raw.get('average', 0) or 0)) if raw.get('average') else None,
            fee_amount=Decimal(str(fee_info.get('cost', 0) or 0)),
            fee_currency=str(fee_info.get('currency', 'USDT')),
            timestamp=datetime.utcnow(),
            raw_response=raw,
        )

    def _paper_fill(self, order: OrderRequest) -> OrderResult:
        """Simulate a fill for paper trading with realistic fee model.

        Fee rules (MEXC spot):
          MARKET      → taker (0.05%)
          LIMIT_MAKER → maker (0.00%, post-only guaranteed)
          LIMIT       → taker if price crosses spread, maker otherwise
                        (no book data → assume taker, conservative)

        Fee denomination matches the received currency so the
        triangular executor's ``quantity_out -= fee_amount`` is correct:
          BUY  → fee in base  (qty * rate)
          SELL → fee in quote (qty * price * rate)
        """
        book = self._last_books.get(order.symbol)

        # --- Fill price ---
        if order.order_type == OrderType.MARKET:
            if book:
                fill_price = book.best_ask if order.side == OrderSide.BUY else book.best_bid
            else:
                fill_price = order.price or Decimal('0')
        else:
            fill_price = order.price

        # --- Fee rate ---
        MAKER_FEE = Decimal('0')       # 0.00% — MEXC spot maker
        TAKER_FEE = Decimal('0.0005')  # 0.05% — MEXC spot taker

        if order.order_type == OrderType.MARKET:
            fee_rate = TAKER_FEE
        elif order.order_type == OrderType.LIMIT_MAKER:
            fee_rate = MAKER_FEE  # Post-only = guaranteed maker
        else:
            # LIMIT: crosses spread → taker, rests in book → maker
            crosses_spread = False
            if book and fill_price:
                if order.side == OrderSide.BUY and book.best_ask and fill_price >= book.best_ask:
                    crosses_spread = True
                elif order.side == OrderSide.SELL and book.best_bid and fill_price <= book.best_bid:
                    crosses_spread = True
            elif fill_price:
                # No book data — conservatively assume taker
                crosses_spread = True
            fee_rate = TAKER_FEE if crosses_spread else MAKER_FEE

        # --- Fee in received-currency denomination ---
        if order.side == OrderSide.BUY:
            fee = order.quantity * fee_rate                                    # base units
        else:
            fee = order.quantity * (fill_price or Decimal('0')) * fee_rate     # quote units

        # Simulate fill rate for LIMIT_MAKER (degraded by volume participation)
        import random
        base_fill_rate = 0.85
        if self.volume_limiter:
            vol_rate = self.volume_limiter.get_fill_rate_modifier(order.symbol)
            base_fill_rate = min(base_fill_rate, vol_rate)
        if order.order_type == OrderType.LIMIT_MAKER and random.random() > base_fill_rate:
            return OrderResult(
                exchange="mexc", symbol=order.symbol,
                order_id=f"paper_{int(time.time()*1000)}",
                client_order_id=order.client_order_id,
                status=OrderStatus.CANCELLED,
                side=order.side, order_type=order.order_type,
                requested_quantity=order.quantity,
                filled_quantity=Decimal('0'),
                average_fill_price=None,
                fee_amount=Decimal('0'), fee_currency="USDT",
                timestamp=datetime.utcnow(),
            )

        return OrderResult(
            exchange="mexc", symbol=order.symbol,
            order_id=f"paper_{int(time.time()*1000)}",
            client_order_id=order.client_order_id,
            status=OrderStatus.FILLED,
            side=order.side, order_type=order.order_type,
            requested_quantity=order.quantity,
            filled_quantity=order.quantity,
            average_fill_price=fill_price,
            fee_amount=fee, fee_currency="USDT",
            timestamp=datetime.utcnow(),
        )

    def _paper_balances(self) -> Dict[str, Balance]:
        """Paper trading balances — pre-funded at $50K USDT equivalent per asset."""
        return {
            "USDT": Balance("mexc", "USDT", Decimal('50000'), Decimal('0'), Decimal('50000')),
            "BTC": Balance("mexc", "BTC", Decimal('0.75'), Decimal('0'), Decimal('0.75')),
            "ETH": Balance("mexc", "ETH", Decimal('12.5'), Decimal('0'), Decimal('12.5')),
            "SOL": Balance("mexc", "SOL", Decimal('250'), Decimal('0'), Decimal('250')),
            "BNB": Balance("mexc", "BNB", Decimal('25'), Decimal('0'), Decimal('25')),
            "XRP": Balance("mexc", "XRP", Decimal('10000'), Decimal('0'), Decimal('10000')),
            "DOGE": Balance("mexc", "DOGE", Decimal('50000'), Decimal('0'), Decimal('50000')),
            "ADA": Balance("mexc", "ADA", Decimal('25000'), Decimal('0'), Decimal('25000')),
            "AVAX": Balance("mexc", "AVAX", Decimal('250'), Decimal('0'), Decimal('250')),
            "LINK": Balance("mexc", "LINK", Decimal('1000'), Decimal('0'), Decimal('1000')),
            "DOT": Balance("mexc", "DOT", Decimal('1500'), Decimal('0'), Decimal('1500')),
        }
