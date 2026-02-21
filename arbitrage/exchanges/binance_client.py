"""
Binance exchange client — SECONDARY exchange for arbitrage.
Uses ccxt for REST API, real WebSocket for streaming market data.
Includes BNB fee discount logic.
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable, Awaitable, Dict

import ccxt.async_support as ccxt_async
import websockets

from .base import (
    ExchangeClient, OrderBook, OrderBookLevel, OrderRequest, OrderResult,
    OrderSide, OrderType, OrderStatus, TimeInForce, Balance, FundingRate,
    Trade,
)

logger = logging.getLogger("arb.binance")

# WebSocket constants
WS_ENDPOINT = "wss://stream.binance.com:9443/ws"
WS_MAX_AGE_HOURS = 23
WS_MAX_RECONNECT = 3
WS_FALLBACK_DURATION = 60
WS_DEPTH_LEVELS = 20
WS_DEPTH_UPDATE_SPEED_MS = 100


class BinanceClient(ExchangeClient):

    def __init__(self, api_key: str = "", api_secret: str = "",
                 paper_trading: bool = True, bnb_fee_discount: bool = True):
        self.paper_trading = paper_trading
        self.bnb_fee_discount = bnb_fee_discount
        self._api_key = api_key
        self._api_secret = api_secret
        self._exchange: Optional[ccxt_async.binance] = None
        self._symbol_info_cache: Dict[str, dict] = {}
        self._ws_callbacks: Dict[str, Callable] = {}      # symbol → depth callback
        self._trade_callbacks: Dict[str, Callable] = {}    # symbol → trade callback
        self._ws_running = False
        self._last_books: Dict[str, OrderBook] = {}
        self._ws_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        config = {
            'apiKey': self._api_key,
            'secret': self._api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        }
        if not self._api_key:
            config.pop('apiKey')
            config.pop('secret')

        self._exchange = ccxt_async.binance(config)

        try:
            await self._exchange.load_markets()
            logger.info(f"Binance connected — {len(self._exchange.markets)} markets loaded")
        except Exception as e:
            logger.warning(f"Binance market load failed (read-only mode): {e}")

    async def disconnect(self) -> None:
        self._ws_running = False
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    # --- Market Data ---

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        raw = await self._exchange.fetch_order_book(symbol, limit=depth)
        return self._parse_order_book(raw, symbol)

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

    # ========== WebSocket Engine ==========

    async def _ws_main_loop(self):
        """Outer loop: reconnection state machine."""
        reconnect_attempts = 0
        while self._ws_running:
            try:
                await self._ws_connect_and_run()
                reconnect_attempts = 0
            except asyncio.CancelledError:
                return
            except Exception as e:
                reconnect_attempts += 1
                if reconnect_attempts >= WS_MAX_RECONNECT:
                    logger.warning(f"Binance WS failed {reconnect_attempts}x, falling back to REST for {WS_FALLBACK_DURATION}s")
                    await self._rest_fallback_loop()
                    reconnect_attempts = 0
                else:
                    backoff = min(2 ** reconnect_attempts, 30)
                    logger.warning(f"Binance WS disconnected: {e} — reconnecting in {backoff}s (attempt {reconnect_attempts})")
                    await asyncio.sleep(backoff)

    async def _ws_connect_and_run(self):
        """Single WebSocket session lifecycle using combined stream URL."""
        connect_time = time.monotonic()
        max_age_sec = WS_MAX_AGE_HOURS * 3600

        # Build combined stream URL — includes symbol in every message wrapper
        all_syms = set(list(self._ws_callbacks.keys()) + list(self._trade_callbacks.keys()))
        streams = []
        for symbol in all_syms:
            raw_sym = self._normalized_to_binance_sym(symbol)
            if symbol in self._ws_callbacks:
                streams.append(f"{raw_sym}@depth{WS_DEPTH_LEVELS}@{WS_DEPTH_UPDATE_SPEED_MS}ms")
            if symbol in self._trade_callbacks:
                streams.append(f"{raw_sym}@trade")

        stream_path = "/".join(streams)
        url = f"{WS_ENDPOINT.replace('/ws', '')}/stream?streams={stream_path}"

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=60,
            close_timeout=5,
        ) as ws:
            logger.info(f"Binance WebSocket connected ({len(streams)} streams)")

            async for raw in ws:
                if not self._ws_running:
                    break
                if time.monotonic() - connect_time > max_age_sec:
                    logger.info("Binance WS max age reached (23h), reconnecting")
                    break
                try:
                    self._ws_handle_message(raw)
                except Exception as e:
                    logger.debug(f"Binance WS message parse error: {e}")

    def _ws_handle_message(self, raw: str):
        """Parse combined-stream JSON message and dispatch to depth/trade handlers."""
        wrapper = json.loads(raw)

        # Combined stream format: {"stream": "btcusdt@depth20@100ms", "data": {...}}
        stream_name = wrapper.get("stream", "")
        msg = wrapper.get("data", wrapper)  # Fall back to raw if no wrapper

        if not stream_name:
            return

        # Extract symbol from stream name: "btcusdt@depth20@100ms" → "btcusdt"
        raw_sym = stream_name.split("@")[0].upper()
        symbol = self._binance_sym_to_normalized(raw_sym)

        if "@depth" in stream_name:
            self._ws_handle_depth(msg, symbol)
        elif "@trade" in stream_name:
            self._ws_handle_trade(msg, symbol)

    def _ws_handle_depth(self, msg: dict, symbol: str):
        """Parse depth snapshot → OrderBook → invoke callback."""
        bids = msg.get("bids", msg.get("b", []))
        asks = msg.get("asks", msg.get("a", []))

        book = OrderBook(
            exchange="binance",
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=[OrderBookLevel(Decimal(str(b[0])), Decimal(str(b[1]))) for b in bids[:WS_DEPTH_LEVELS]],
            asks=[OrderBookLevel(Decimal(str(a[0])), Decimal(str(a[1]))) for a in asks[:WS_DEPTH_LEVELS]],
        )
        self._last_books[symbol] = book

        cb = self._ws_callbacks.get(symbol)
        if cb:
            asyncio.get_event_loop().create_task(cb(book))

    def _ws_handle_trade(self, msg: dict, symbol: str):
        """Parse trade event → Trade → invoke callback."""
        cb = self._trade_callbacks.get(symbol)
        if not cb:
            return

        # "m" == true means buyer is market maker, so the taker side is SELL
        side = OrderSide.SELL if msg.get("m", False) else OrderSide.BUY

        trade = Trade(
            exchange="binance",
            symbol=symbol,
            trade_id=str(msg.get("t", "")),
            price=Decimal(str(msg.get("p", "0"))),
            quantity=Decimal(str(msg.get("q", "0"))),
            side=side,
            timestamp=datetime.utcfromtimestamp(msg.get("T", time.time() * 1000) / 1000),
        )
        asyncio.get_event_loop().create_task(cb(trade))

    async def _rest_fallback_loop(self):
        """REST polling fallback when WS fails repeatedly."""
        end_time = time.monotonic() + WS_FALLBACK_DURATION
        logger.info("Binance entering REST fallback mode")
        while self._ws_running and time.monotonic() < end_time:
            async def _fetch_one(sym, cb):
                try:
                    raw = await self._exchange.fetch_order_book(sym, limit=20)
                    book = self._parse_order_book(raw, sym)
                    self._last_books[sym] = book
                    await cb(book)
                except Exception as e:
                    logger.debug(f"Binance REST fallback error {sym}: {e}")

            tasks = [_fetch_one(s, c) for s, c in list(self._ws_callbacks.items())]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.5)
        logger.info("Binance REST fallback complete, retrying WebSocket")

    # ========== Symbol Conversion ==========

    def _normalized_to_binance_sym(self, symbol: str) -> str:
        """BTC/USDT → btcusdt (lowercase, no slash)."""
        return symbol.replace("/", "").lower()

    def _binance_sym_to_normalized(self, raw: str) -> str:
        """BTCUSDT → BTC/USDT. Best-effort using known quote currencies."""
        raw = raw.upper()
        for quote in ("USDT", "USDC", "BTC", "ETH", "BNB"):
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
        raw = await self._exchange.fetch_tickers()
        result = {}
        for symbol, ticker in raw.items():
            result[symbol] = {
                'symbol': symbol,
                'last_price': Decimal(str(ticker.get('last', 0) or 0)),
                'bid': Decimal(str(ticker.get('bid', 0) or 0)),
                'ask': Decimal(str(ticker.get('ask', 0) or 0)),
                'volume_24h': Decimal(str(ticker.get('baseVolume', 0) or 0)),
            }
        return result

    async def get_funding_rate(self, symbol: str) -> FundingRate:
        try:
            fr = await self._exchange.fetch_funding_rate(symbol)
            return FundingRate(
                exchange="binance",
                symbol=symbol,
                current_rate=Decimal(str(fr.get('fundingRate', 0) or 0)),
                predicted_rate=Decimal(str(fr.get('fundingRate', 0) or 0)),
                next_funding_time=datetime.utcnow(),
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            logger.debug(f"Binance funding rate fetch failed for {symbol}: {e}")
            return FundingRate(
                exchange="binance", symbol=symbol,
                current_rate=Decimal('0'), predicted_rate=None,
                next_funding_time=datetime.utcnow(), timestamp=datetime.utcnow(),
            )

    # --- Trading ---

    async def place_order(self, order: OrderRequest) -> OrderResult:
        if self.paper_trading:
            return self._paper_fill(order)

        params = {}
        if order.order_type == OrderType.LIMIT_MAKER:
            order_type_str = 'limit'
            params['timeInForce'] = 'GTX'
        elif order.order_type == OrderType.MARKET:
            order_type_str = 'market'
        else:
            order_type_str = 'limit'

        try:
            result = await self._exchange.create_order(
                symbol=order.symbol,
                type=order_type_str,
                side=order.side.value,
                amount=float(order.quantity),
                price=float(order.price) if order.price else None,
                params=params,
            )
            return self._parse_order_result(result, order)
        except Exception as e:
            logger.error(f"Binance order failed: {e}")
            return OrderResult(
                exchange="binance", symbol=order.symbol,
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
            logger.error(f"Binance cancel failed: {e}")
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
                    exchange="binance", currency=currency,
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
                exchange="binance", currency=currency,
                free=Decimal('100000'), locked=Decimal('0'), total=Decimal('100000'),
            )
        return Balance(
            exchange="binance", currency=currency,
            free=Decimal('0'), locked=Decimal('0'), total=Decimal('0'),
        )

    # --- Exchange Info ---

    async def get_trading_fees(self, symbol: str) -> dict:
        base_fee = Decimal('0.001')  # 0.1%
        if self.bnb_fee_discount:
            base_fee = Decimal('0.00075')  # 0.075% with BNB
        return {"maker": base_fee, "taker": base_fee}

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

    # --- Helpers ---

    def _parse_order_book(self, raw: dict, symbol: str) -> OrderBook:
        bids = [OrderBookLevel(Decimal(str(b[0])), Decimal(str(b[1]))) for b in raw.get('bids', [])[:20]]
        asks = [OrderBookLevel(Decimal(str(a[0])), Decimal(str(a[1]))) for a in raw.get('asks', [])[:20]]
        ts = raw.get('timestamp')
        dt = datetime.utcfromtimestamp(ts / 1000) if ts else datetime.utcnow()
        return OrderBook(exchange="binance", symbol=symbol, timestamp=dt, bids=bids, asks=asks)

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
            exchange="binance",
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
        """Simulate a fill with realistic Binance fee model.

        Fee denomination matches received currency:
          BUY  → fee in base  (qty * rate)
          SELL → fee in quote (qty * price * rate)
        """
        book = self._last_books.get(order.symbol)
        if order.order_type == OrderType.MARKET:
            fill_price = (book.best_ask if order.side == OrderSide.BUY else book.best_bid) if book else order.price
            fee_rate = Decimal('0.00075') if self.bnb_fee_discount else Decimal('0.001')
        else:
            fill_price = order.price
            fee_rate = Decimal('0.00075') if self.bnb_fee_discount else Decimal('0.001')

        if order.side == OrderSide.BUY:
            fee = order.quantity * fee_rate                                    # base units
        else:
            fee = order.quantity * (fill_price or Decimal('0')) * fee_rate     # quote units

        import random
        if order.order_type == OrderType.LIMIT_MAKER and random.random() > 0.90:
            return OrderResult(
                exchange="binance", symbol=order.symbol,
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
            exchange="binance", symbol=order.symbol,
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
            "USDT": Balance("binance", "USDT", Decimal('50000'), Decimal('0'), Decimal('50000')),
            "BTC": Balance("binance", "BTC", Decimal('0.75'), Decimal('0'), Decimal('0.75')),
            "ETH": Balance("binance", "ETH", Decimal('12.5'), Decimal('0'), Decimal('12.5')),
            "SOL": Balance("binance", "SOL", Decimal('250'), Decimal('0'), Decimal('250')),
            "BNB": Balance("binance", "BNB", Decimal('25'), Decimal('0'), Decimal('25')),
            "XRP": Balance("binance", "XRP", Decimal('10000'), Decimal('0'), Decimal('10000')),
            "DOGE": Balance("binance", "DOGE", Decimal('50000'), Decimal('0'), Decimal('50000')),
            "ADA": Balance("binance", "ADA", Decimal('25000'), Decimal('0'), Decimal('25000')),
            "AVAX": Balance("binance", "AVAX", Decimal('250'), Decimal('0'), Decimal('250')),
            "LINK": Balance("binance", "LINK", Decimal('1000'), Decimal('0'), Decimal('1000')),
            "DOT": Balance("binance", "DOT", Decimal('1500'), Decimal('0'), Decimal('1500')),
        }
