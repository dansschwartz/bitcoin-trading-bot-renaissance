"""
Binance exchange client — SECONDARY exchange for arbitrage.
Uses ccxt for unified interface. Includes BNB fee discount logic.
"""
import asyncio
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable, Awaitable, Dict

import ccxt.async_support as ccxt_async

from .base import (
    ExchangeClient, OrderBook, OrderBookLevel, OrderRequest, OrderResult,
    OrderSide, OrderType, OrderStatus, TimeInForce, Balance, FundingRate,
)

logger = logging.getLogger("arb.binance")


class BinanceClient(ExchangeClient):

    def __init__(self, api_key: str = "", api_secret: str = "",
                 paper_trading: bool = True, bnb_fee_discount: bool = True):
        self.paper_trading = paper_trading
        self.bnb_fee_discount = bnb_fee_discount
        self._api_key = api_key
        self._api_secret = api_secret
        self._exchange: Optional[ccxt_async.binance] = None
        self._symbol_info_cache: Dict[str, dict] = {}
        self._ws_callbacks: Dict[str, Callable] = {}
        self._ws_running = False
        self._last_books: Dict[str, OrderBook] = {}

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
            asyncio.create_task(self._ws_order_book_loop())

    async def _ws_order_book_loop(self):
        """Fetch order books for all subscribed symbols in parallel."""
        while self._ws_running:
            async def _fetch_one(sym, cb):
                try:
                    raw = await self._exchange.fetch_order_book(sym, limit=20)
                    book = self._parse_order_book(raw, sym)
                    self._last_books[sym] = book
                    await cb(book)
                except Exception as e:
                    logger.debug(f"Binance book update error {sym}: {e}")

            tasks = [_fetch_one(s, c) for s, c in list(self._ws_callbacks.items())]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.5)  # 500ms cycle for REST rate limit safety

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
        return balances.get(currency, Balance(
            exchange="binance", currency=currency,
            free=Decimal('0'), locked=Decimal('0'), total=Decimal('0'),
        ))

    # --- Exchange Info ---

    async def get_trading_fees(self, symbol: str) -> dict:
        base_fee = Decimal('0.001')  # 0.1%
        if self.bnb_fee_discount:
            base_fee = Decimal('0.00075')  # 0.075% with BNB
        return {"maker": base_fee, "taker": base_fee}

    async def get_symbol_info(self, symbol: str) -> dict:
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        if self._exchange and symbol in self._exchange.markets:
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
        book = self._last_books.get(order.symbol)
        if order.order_type == OrderType.MARKET:
            fill_price = (book.best_ask if order.side == OrderSide.BUY else book.best_bid) if book else order.price
            fee_rate = Decimal('0.00075') if self.bnb_fee_discount else Decimal('0.001')
        else:
            fill_price = order.price
            fee_rate = Decimal('0.00075') if self.bnb_fee_discount else Decimal('0.001')

        fee = order.quantity * (fill_price or Decimal('0')) * fee_rate

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
        return {
            "USDT": Balance("binance", "USDT", Decimal('5000'), Decimal('0'), Decimal('5000')),
            "BTC": Balance("binance", "BTC", Decimal('0.15'), Decimal('0'), Decimal('0.15')),
            "ETH": Balance("binance", "ETH", Decimal('2.5'), Decimal('0'), Decimal('2.5')),
            "SOL": Balance("binance", "SOL", Decimal('50'), Decimal('0'), Decimal('50')),
            "BNB": Balance("binance", "BNB", Decimal('5'), Decimal('0'), Decimal('5')),
            "XRP": Balance("binance", "XRP", Decimal('2000'), Decimal('0'), Decimal('2000')),
            "DOGE": Balance("binance", "DOGE", Decimal('10000'), Decimal('0'), Decimal('10000')),
            "ADA": Balance("binance", "ADA", Decimal('5000'), Decimal('0'), Decimal('5000')),
            "AVAX": Balance("binance", "AVAX", Decimal('50'), Decimal('0'), Decimal('50')),
            "LINK": Balance("binance", "LINK", Decimal('200'), Decimal('0'), Decimal('200')),
            "DOT": Balance("binance", "DOT", Decimal('300'), Decimal('0'), Decimal('300')),
        }
