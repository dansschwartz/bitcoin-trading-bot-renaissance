"""
KuCoin exchange client — SPOKE exchange for cross-exchange arbitrage.
Uses ccxt for REST API, real WebSocket for streaming market data.

KuCoin fee: 0.10% maker/taker (Class A), higher than Binance (7.5bps).
The edge comes from wider spreads on mid-cap tokens and decorrelated
opportunity surface from a different token universe (~800 pairs).

KuCoin API peculiarities:
- Symbol format uses hyphens: "BTC-USDT" (not slashes or concatenated)
- 3-part auth: API key + secret + passphrase
- WebSocket: token-based connection (POST /api/v1/bullet-public for token)
- WebSocket pings: JSON pings (not WS protocol pings)
- Depth snapshots: /market/level2:five:{symbol} for top-5 snapshot
"""
import asyncio
import json
import logging
import random
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Callable, Awaitable, Dict

import aiohttp
import ccxt.async_support as ccxt_async

from .base import (
    ExchangeClient, OrderBook, OrderBookLevel, OrderRequest, OrderResult,
    OrderSide, OrderType, OrderStatus, TimeInForce, Balance, FundingRate,
    Trade,
)

logger = logging.getLogger("arb.kucoin")

# Akamai WAF blocks default aiohttp user-agent
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}

# KuCoin fee schedule (Class A — default tier)
MAKER_FEE = Decimal('0.001')   # 0.10%
TAKER_FEE = Decimal('0.001')   # 0.10%

# WebSocket constants
WS_PING_INTERVAL = 25          # KuCoin requires ping every 30s, we do 25s for safety
WS_MAX_AGE_HOURS = 23
WS_DEPTH_SNAPSHOT_TOPIC = "/market/level2:five"  # Top-5 depth snapshot


class KuCoinClient(ExchangeClient):
    """KuCoin exchange client implementing ExchangeClient ABC.

    Used as a spoke exchange in hub-and-spoke cross-exchange arb
    with MEXC as the hub (0% maker fee).
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        paper_trading: bool = True,
    ):
        self.paper_trading = paper_trading
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self._exchange: Optional[ccxt_async.kucoin] = None
        self._symbol_info_cache: Dict[str, dict] = {}
        self._ws_callbacks: Dict[str, Callable] = {}      # symbol -> depth callback
        self._trade_callbacks: Dict[str, Callable] = {}    # symbol -> trade callback
        self._ws_running = False
        self._last_books: Dict[str, OrderBook] = {}
        self._ws_task: Optional[asyncio.Task] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        # Volume limiter (set by orchestrator for fill rate degradation)
        self.volume_limiter = None

    # ── Lifecycle ─────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize KuCoin exchange — lightweight, skips load_markets().

        Uses direct REST (aiohttp) for order books and tickers.
        Keeps a minimal ccxt instance for fetch_currencies (contract verifier)
        without loading all 1141 market definitions into memory.
        """
        config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            },
        }
        if self._api_key:
            config['apiKey'] = self._api_key
            config['secret'] = self._api_secret
            config['password'] = self._passphrase  # ccxt uses 'password' for passphrase

        self._exchange = ccxt_async.kucoin(config)
        # Skip load_markets() — saves ~200MB RAM and 30s CPU on VPS
        # We use direct REST for order books, tickers, and symbol info

        # Shared aiohttp session for direct REST calls
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers=_HTTP_HEADERS,
        )

        # Quick connectivity check
        try:
            url = "https://api.kucoin.com/api/v1/market/orderbook/level2_20?symbol=BTC-USDT"
            async with self._http_session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("code") == "200000":
                        logger.info("KuCoin connected — REST API OK (lightweight mode, no market load)")
                    else:
                        logger.warning(f"KuCoin API returned code {data.get('code')}")
                else:
                    logger.warning(f"KuCoin connectivity check HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"KuCoin connectivity check failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect WebSocket and close ccxt/HTTP sessions."""
        self._ws_running = False
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    # ── Symbol Conversion ─────────────────────────────────────────

    def _to_kucoin_symbol(self, symbol: str) -> str:
        """Convert normalized symbol to KuCoin format.

        "BTC/USDT" -> "BTC-USDT"
        """
        return symbol.replace("/", "-")

    def _from_kucoin_symbol(self, kc_symbol: str) -> str:
        """Convert KuCoin symbol to normalized format.

        "BTC-USDT" -> "BTC/USDT"
        """
        return kc_symbol.replace("-", "/")

    # ── Market Data ───────────────────────────────────────────────

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        """Fetch order book via direct REST, with ccxt fallback."""
        kc_symbol = self._to_kucoin_symbol(symbol)

        # Primary: direct REST via aiohttp
        try:
            if self._http_session:
                url = f"https://api.kucoin.com/api/v1/market/orderbook/level2_20?symbol={kc_symbol}"
                async with self._http_session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("code") == "200000":
                            book_data = data["data"]
                            return self._parse_rest_book(symbol, book_data)
        except Exception as e:
            logger.debug(f"KuCoin REST book failed for {symbol}: {e}")

        # Fallback: ccxt
        try:
            if self._exchange:
                raw = await self._exchange.fetch_order_book(symbol, limit=depth)
                bids = [OrderBookLevel(Decimal(str(p)), Decimal(str(q))) for p, q in raw.get("bids", [])]
                asks = [OrderBookLevel(Decimal(str(p)), Decimal(str(q))) for p, q in raw.get("asks", [])]
                return OrderBook(
                    exchange="kucoin", symbol=symbol,
                    timestamp=datetime.utcnow(),
                    bids=sorted(bids, key=lambda x: x.price, reverse=True),
                    asks=sorted(asks, key=lambda x: x.price),
                )
        except Exception as e:
            logger.debug(f"KuCoin ccxt book fallback failed for {symbol}: {e}")

        # Return empty book as last resort
        return OrderBook(
            exchange="kucoin", symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=[], asks=[],
        )

    def _parse_rest_book(self, symbol: str, data: dict) -> OrderBook:
        """Parse KuCoin REST orderbook response into OrderBook."""
        bids = [
            OrderBookLevel(Decimal(str(p)), Decimal(str(q)))
            for p, q in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(Decimal(str(p)), Decimal(str(q)))
            for p, q in data.get("asks", [])
        ]
        return OrderBook(
            exchange="kucoin", symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
        )

    async def subscribe_order_book(
        self, symbol: str,
        callback: Callable[[OrderBook], Awaitable[None]],
        depth: int = 20,
    ) -> None:
        """Subscribe to order book updates via WebSocket.

        KuCoin WS uses token-based auth — need to POST /api/v1/bullet-public first.
        Falls back to REST polling if WS fails.
        """
        self._ws_callbacks[symbol] = callback

        # Start WS loop if not already running
        if not self._ws_running and not self._ws_task:
            self._ws_running = True
            self._ws_task = asyncio.create_task(self._ws_loop())

    async def subscribe_trades(
        self, symbol: str,
        callback: Callable[['Trade'], Awaitable[None]],
    ) -> None:
        """Subscribe to trade stream. Uses REST polling for simplicity."""
        self._trade_callbacks[symbol] = callback

    async def _ws_loop(self):
        """WebSocket event loop with reconnection logic.

        KuCoin WS flow:
        1. POST /api/v1/bullet-public to get token + endpoint
        2. Connect to endpoint?token=xxx&connectId=yyy
        3. Subscribe to depth channels
        4. Send JSON pings every 25s
        """
        reconnect_attempts = 0
        max_reconnects = 3

        while self._ws_running:
            try:
                # Step 1: Get WS token
                token_data = await self._get_ws_token()
                if not token_data:
                    logger.warning("KuCoin WS: Failed to get token, falling back to REST")
                    await self._rest_fallback_loop()
                    continue

                endpoint = token_data["endpoint"]
                token = token_data["token"]
                ping_interval = token_data.get("ping_interval", WS_PING_INTERVAL)

                ws_url = f"{endpoint}?token={token}&connectId={int(time.time()*1000)}"

                # Step 2: Connect
                try:
                    import websockets
                    async with websockets.connect(
                        ws_url,
                        ping_interval=None,  # We handle pings manually (JSON, not WS protocol)
                        close_timeout=5,
                    ) as ws:
                        logger.info(f"KuCoin WS connected — {len(self._ws_callbacks)} symbols to subscribe")
                        reconnect_attempts = 0

                        # Step 3: Subscribe to all depth channels
                        symbols_to_sub = list(self._ws_callbacks.keys())
                        if symbols_to_sub:
                            kc_symbols = [self._to_kucoin_symbol(s) for s in symbols_to_sub]
                            topic = f"{WS_DEPTH_SNAPSHOT_TOPIC}:{','.join(kc_symbols)}"
                            sub_msg = {
                                "id": str(int(time.time() * 1000)),
                                "type": "subscribe",
                                "topic": topic,
                                "privateChannel": False,
                                "response": True,
                            }
                            await ws.send(json.dumps(sub_msg))

                        # Step 4: Message + ping loop
                        connection_start = time.monotonic()
                        last_ping = time.monotonic()

                        while self._ws_running:
                            # Check connection age
                            if time.monotonic() - connection_start > WS_MAX_AGE_HOURS * 3600:
                                logger.info("KuCoin WS: Max age reached, reconnecting")
                                break

                            # Send ping if needed (JSON ping, NOT WS protocol ping)
                            if time.monotonic() - last_ping > ping_interval:
                                ping_msg = {
                                    "id": str(int(time.time() * 1000)),
                                    "type": "ping",
                                }
                                try:
                                    await ws.send(json.dumps(ping_msg))
                                    last_ping = time.monotonic()
                                except Exception:
                                    break

                            # Receive with timeout
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=ping_interval + 5)
                                msg = json.loads(raw)
                                await self._handle_ws_message(msg)
                            except asyncio.TimeoutError:
                                # No data — ping and continue
                                continue
                            except Exception as e:
                                logger.debug(f"KuCoin WS recv error: {e}")
                                break

                except Exception as e:
                    logger.warning(f"KuCoin WS connection error: {e}")
                    reconnect_attempts += 1

                if reconnect_attempts >= max_reconnects:
                    logger.warning(f"KuCoin WS: {max_reconnects} failures, falling back to REST")
                    await self._rest_fallback_loop()
                    reconnect_attempts = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"KuCoin WS loop error: {e}")
                await asyncio.sleep(5)

    async def _get_ws_token(self) -> Optional[dict]:
        """Get WebSocket connection token from KuCoin public endpoint."""
        try:
            if self._http_session:
                url = "https://api.kucoin.com/api/v1/bullet-public"
                async with self._http_session.post(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("code") == "200000":
                            servers = data["data"]["instanceServers"]
                            if servers:
                                return {
                                    "endpoint": servers[0]["endpoint"],
                                    "token": data["data"]["token"],
                                    "ping_interval": servers[0].get("pingInterval", 25000) // 1000,
                                }
        except Exception as e:
            logger.warning(f"KuCoin WS token fetch failed: {e}")
        return None

    async def _handle_ws_message(self, msg: dict) -> None:
        """Handle incoming KuCoin WebSocket message."""
        msg_type = msg.get("type")

        if msg_type == "pong":
            return  # Ping response, ignore

        if msg_type == "ack":
            return  # Subscription acknowledgement

        if msg_type != "message":
            return

        topic = msg.get("topic", "")
        data = msg.get("data", {})

        # Depth snapshot: /market/level2:five:BTC-USDT
        if WS_DEPTH_SNAPSHOT_TOPIC in topic:
            # Extract symbol from topic
            parts = topic.split(":")
            if len(parts) >= 3:
                kc_symbol = parts[2]  # "BTC-USDT"
                norm_symbol = self._from_kucoin_symbol(kc_symbol)

                bids = [
                    OrderBookLevel(Decimal(str(p)), Decimal(str(q)))
                    for p, q in data.get("bids", [])
                ]
                asks = [
                    OrderBookLevel(Decimal(str(p)), Decimal(str(q)))
                    for p, q in data.get("asks", [])
                ]

                book = OrderBook(
                    exchange="kucoin", symbol=norm_symbol,
                    timestamp=datetime.utcnow(),
                    bids=sorted(bids, key=lambda x: x.price, reverse=True),
                    asks=sorted(asks, key=lambda x: x.price),
                )

                self._last_books[norm_symbol] = book

                callback = self._ws_callbacks.get(norm_symbol)
                if callback:
                    try:
                        await callback(book)
                    except Exception as e:
                        logger.debug(f"KuCoin WS callback error for {norm_symbol}: {e}")

    async def _rest_fallback_loop(self):
        """REST polling fallback when WebSocket is unavailable."""
        logger.info("KuCoin: Entering REST fallback mode (60s)")
        end_time = time.monotonic() + 60  # 60s of REST before retrying WS

        while self._ws_running and time.monotonic() < end_time:
            for symbol, callback in list(self._ws_callbacks.items()):
                if not self._ws_running:
                    break
                try:
                    book = await self.get_order_book(symbol, depth=20)
                    self._last_books[symbol] = book
                    await callback(book)
                except Exception:
                    pass
                await asyncio.sleep(0.2)  # 200ms between pairs
            await asyncio.sleep(2)  # 2s between full cycles

    async def get_ticker(self, symbol: str) -> dict:
        """Get current ticker for a symbol."""
        try:
            if self._exchange:
                return await self._exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.debug(f"KuCoin ticker fetch failed for {symbol}: {e}")
        return {}

    async def get_all_tickers(self) -> Dict[str, dict]:
        """Get tickers for all trading pairs via direct REST."""
        try:
            if self._http_session:
                url = "https://api.kucoin.com/api/v1/market/allTickers"
                async with self._http_session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("code") == "200000":
                            tickers = {}
                            for t in data.get("data", {}).get("ticker", []):
                                kc_sym = t.get("symbol", "")
                                norm_sym = self._from_kucoin_symbol(kc_sym)
                                tickers[norm_sym] = {
                                    "symbol": norm_sym,
                                    "last": float(t.get("last", 0)),
                                    "bid": float(t.get("buy", 0)),
                                    "ask": float(t.get("sell", 0)),
                                    "volume": float(t.get("vol", 0)),
                                    "quoteVolume": float(t.get("volValue", 0)),
                                }
                            return tickers
        except Exception as e:
            logger.debug(f"KuCoin all tickers fetch failed: {e}")
        return {}

    async def get_funding_rate(self, symbol: str) -> FundingRate:
        """Get funding rate. KuCoin spot has no funding rate — returns zero."""
        return FundingRate(
            exchange="kucoin", symbol=symbol,
            current_rate=Decimal('0'),
            predicted_rate=None,
            next_funding_time=datetime.utcnow(),
            timestamp=datetime.utcnow(),
        )

    # ── Trading ───────────────────────────────────────────────────

    async def place_order(self, order: OrderRequest) -> OrderResult:
        """Place an order on KuCoin (paper or live)."""
        if self.paper_trading:
            return self._paper_fill(order)

        # Live order via ccxt
        try:
            params = {}
            if order.order_type == OrderType.LIMIT_MAKER:
                params['postOnly'] = True

            side = order.side.value
            order_type = 'limit' if order.order_type in (OrderType.LIMIT, OrderType.LIMIT_MAKER) else 'market'

            result = await self._exchange.create_order(
                symbol=order.symbol,
                type=order_type,
                side=side,
                amount=float(order.quantity),
                price=float(order.price) if order.price else None,
                params=params,
            )

            status_map = {
                'open': OrderStatus.OPEN,
                'closed': OrderStatus.FILLED,
                'canceled': OrderStatus.CANCELLED,
            }

            return OrderResult(
                exchange="kucoin",
                symbol=order.symbol,
                order_id=result.get('id', ''),
                client_order_id=order.client_order_id,
                status=status_map.get(result.get('status'), OrderStatus.PENDING),
                side=order.side,
                order_type=order.order_type,
                requested_quantity=order.quantity,
                filled_quantity=Decimal(str(result.get('filled', 0))),
                average_fill_price=Decimal(str(result.get('average', 0))) if result.get('average') else None,
                fee_amount=Decimal(str(result.get('fee', {}).get('cost', 0) or 0)),
                fee_currency=str(result.get('fee', {}).get('currency', 'USDT')),
                timestamp=datetime.utcnow(),
                raw_response=result,
            )
        except Exception as e:
            logger.error(f"KuCoin order failed: {e}")
            return OrderResult(
                exchange="kucoin", symbol=order.symbol,
                order_id="", client_order_id=order.client_order_id,
                status=OrderStatus.REJECTED,
                side=order.side, order_type=order.order_type,
                requested_quantity=order.quantity,
                filled_quantity=Decimal('0'),
                average_fill_price=None,
                fee_amount=Decimal('0'), fee_currency="USDT",
                timestamp=datetime.utcnow(),
            )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        if self.paper_trading:
            return True
        try:
            if self._exchange:
                await self._exchange.cancel_order(order_id, symbol)
                return True
        except Exception as e:
            logger.error(f"KuCoin cancel failed: {e}")
        return False

    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult:
        """Get order status."""
        if self.paper_trading:
            return OrderResult(
                exchange="kucoin", symbol=symbol,
                order_id=order_id, client_order_id=None,
                status=OrderStatus.FILLED,
                side=OrderSide.BUY, order_type=OrderType.LIMIT,
                requested_quantity=Decimal('0'),
                filled_quantity=Decimal('0'),
                average_fill_price=None,
                fee_amount=Decimal('0'), fee_currency="USDT",
                timestamp=datetime.utcnow(),
            )
        try:
            if self._exchange:
                raw = await self._exchange.fetch_order(order_id, symbol)
                status_map = {
                    'open': OrderStatus.OPEN,
                    'closed': OrderStatus.FILLED,
                    'canceled': OrderStatus.CANCELLED,
                }
                return OrderResult(
                    exchange="kucoin", symbol=symbol,
                    order_id=order_id, client_order_id=None,
                    status=status_map.get(raw.get('status'), OrderStatus.PENDING),
                    side=OrderSide.BUY if raw.get('side') == 'buy' else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    requested_quantity=Decimal(str(raw.get('amount', 0))),
                    filled_quantity=Decimal(str(raw.get('filled', 0))),
                    average_fill_price=Decimal(str(raw.get('average', 0))) if raw.get('average') else None,
                    fee_amount=Decimal(str(raw.get('fee', {}).get('cost', 0) or 0)),
                    fee_currency=str(raw.get('fee', {}).get('currency', 'USDT')),
                    timestamp=datetime.utcnow(),
                    raw_response=raw,
                )
        except Exception as e:
            logger.error(f"KuCoin order status failed: {e}")

        return OrderResult(
            exchange="kucoin", symbol=symbol,
            order_id=order_id, client_order_id=None,
            status=OrderStatus.PENDING,
            side=OrderSide.BUY, order_type=OrderType.LIMIT,
            requested_quantity=Decimal('0'),
            filled_quantity=Decimal('0'),
            average_fill_price=None,
            fee_amount=Decimal('0'), fee_currency="USDT",
            timestamp=datetime.utcnow(),
        )

    # ── Account ───────────────────────────────────────────────────

    async def get_balances(self) -> Dict[str, Balance]:
        """Get all balances."""
        if self.paper_trading:
            return self._paper_balances()
        try:
            if self._exchange:
                raw = await self._exchange.fetch_balance()
                result = {}
                for currency, info in raw.get('total', {}).items():
                    total = Decimal(str(info or 0))
                    free = Decimal(str(raw.get('free', {}).get(currency, 0) or 0))
                    locked = Decimal(str(raw.get('used', {}).get(currency, 0) or 0))
                    if total > 0:
                        result[currency] = Balance("kucoin", currency, free, locked, total)
                return result
        except Exception as e:
            logger.error(f"KuCoin balances failed: {e}")
        return self._paper_balances() if self.paper_trading else {}

    async def get_balance(self, currency: str) -> Balance:
        """Get balance for a specific currency."""
        balances = await self.get_balances()
        if currency in balances:
            return balances[currency]
        return Balance("kucoin", currency, Decimal('0'), Decimal('0'), Decimal('0'))

    # ── Exchange Info ─────────────────────────────────────────────

    async def get_trading_fees(self, symbol: str) -> dict:
        """Get trading fees for a symbol."""
        return {
            "maker": MAKER_FEE,
            "taker": TAKER_FEE,
        }

    async def get_symbol_info(self, symbol: str) -> dict:
        """Get symbol precision info via REST API (cached)."""
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        info = {"price_precision": 8, "quantity_precision": 8}

        try:
            kc_symbol = self._to_kucoin_symbol(symbol)
            if self._http_session:
                url = f"https://api.kucoin.com/api/v2/symbols/{kc_symbol}"
                async with self._http_session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("code") == "200000":
                            sym_data = data.get("data", {})
                            # KuCoin returns priceIncrement/baseIncrement as strings
                            price_inc = sym_data.get("priceIncrement", "0.00000001")
                            base_inc = sym_data.get("baseIncrement", "0.00000001")
                            # Count decimal places
                            price_prec = len(price_inc.rstrip('0').split('.')[-1]) if '.' in price_inc else 0
                            qty_prec = len(base_inc.rstrip('0').split('.')[-1]) if '.' in base_inc else 0
                            info = {
                                "price_precision": price_prec,
                                "quantity_precision": qty_prec,
                            }
        except Exception as e:
            logger.debug(f"KuCoin symbol info failed for {symbol}: {e}")

        self._symbol_info_cache[symbol] = info
        return info

    # ── Paper Trading ─────────────────────────────────────────────

    def _paper_fill(self, order: OrderRequest) -> OrderResult:
        """Simulate a fill for paper trading with KuCoin fee model.

        Fee rules (KuCoin spot Class A):
          MARKET      -> taker (0.10%)
          LIMIT_MAKER -> maker (0.10%, post-only)
          LIMIT       -> taker if crosses spread, maker otherwise

        Fee denomination: BUY -> base units, SELL -> quote units.
        """
        book = self._last_books.get(order.symbol)

        # Fill price
        if order.order_type == OrderType.MARKET:
            if book:
                fill_price = book.best_ask if order.side == OrderSide.BUY else book.best_bid
            else:
                fill_price = order.price or Decimal('0')
        else:
            fill_price = order.price

        # Fee rate (KuCoin Class A: 0.10% maker and taker)
        if order.order_type == OrderType.MARKET:
            fee_rate = TAKER_FEE
        elif order.order_type == OrderType.LIMIT_MAKER:
            fee_rate = MAKER_FEE
        else:
            # LIMIT: check if it crosses the spread
            crosses_spread = False
            if book and fill_price:
                if order.side == OrderSide.BUY and book.best_ask and fill_price >= book.best_ask:
                    crosses_spread = True
                elif order.side == OrderSide.SELL and book.best_bid and fill_price <= book.best_bid:
                    crosses_spread = True
            elif fill_price:
                crosses_spread = True  # No book data — assume taker
            fee_rate = TAKER_FEE if crosses_spread else MAKER_FEE

        # Fee denomination: BUY -> base, SELL -> quote
        if order.side == OrderSide.BUY:
            fee = order.quantity * fee_rate
        else:
            fee = order.quantity * (fill_price or Decimal('0')) * fee_rate

        # Simulate fill rate for LIMIT_MAKER (degraded by volume participation)
        base_fill_rate = 0.85
        if self.volume_limiter:
            vol_rate = self.volume_limiter.get_fill_rate_modifier(order.symbol)
            base_fill_rate = min(base_fill_rate, vol_rate)
        if order.order_type == OrderType.LIMIT_MAKER and random.random() > base_fill_rate:
            return OrderResult(
                exchange="kucoin", symbol=order.symbol,
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
            exchange="kucoin", symbol=order.symbol,
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
            "USDT": Balance("kucoin", "USDT", Decimal('50000'), Decimal('0'), Decimal('50000')),
            "BTC": Balance("kucoin", "BTC", Decimal('0.75'), Decimal('0'), Decimal('0.75')),
            "ETH": Balance("kucoin", "ETH", Decimal('12.5'), Decimal('0'), Decimal('12.5')),
            "SOL": Balance("kucoin", "SOL", Decimal('250'), Decimal('0'), Decimal('250')),
            "XRP": Balance("kucoin", "XRP", Decimal('10000'), Decimal('0'), Decimal('10000')),
            "DOGE": Balance("kucoin", "DOGE", Decimal('50000'), Decimal('0'), Decimal('50000')),
            "ADA": Balance("kucoin", "ADA", Decimal('25000'), Decimal('0'), Decimal('25000')),
            "AVAX": Balance("kucoin", "AVAX", Decimal('250'), Decimal('0'), Decimal('250')),
            "LINK": Balance("kucoin", "LINK", Decimal('1000'), Decimal('0'), Decimal('1000')),
            "DOT": Balance("kucoin", "DOT", Decimal('1500'), Decimal('0'), Decimal('1500')),
        }
