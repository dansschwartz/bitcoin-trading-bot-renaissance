"""
Bybit exchange client — secondary exchange for arbitrage.
Uses Bybit v5 unified API for spot trading.

IMPORTANT: Bybit VIP 0 spot maker fee = 0.10% (10 bps).
This is significantly higher than MEXC (0%), so triangular arb
requires edges > 35 bps to be profitable (10 bps × 3 legs + 5 bps buffer).
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Callable, Awaitable, Dict, List

import aiohttp

from .base import (
    ExchangeClient, OrderBook, OrderBookLevel, OrderRequest, OrderResult,
    OrderSide, OrderType, OrderStatus, TimeInForce, Balance, FundingRate,
    Trade,
)

logger = logging.getLogger("arb.bybit")

BYBIT_REST = "https://api.bybit.com"
BYBIT_WS = "wss://stream.bybit.com/v5/public/spot"

# Bybit VIP 0 fee schedule (as of Feb 2026)
MAKER_FEE = Decimal('0.0010')   # 0.10%
TAKER_FEE = Decimal('0.0010')   # 0.10%


class BybitClient(ExchangeClient):
    """Bybit spot exchange client implementing the standard interface.

    Key differences from MEXC:
    - Symbol format: "BTCUSDT" (no slash) vs MEXC "BTC/USDT"
    - Unified v5 API with category=spot parameter
    - Rate limits: 120 requests per 5 seconds
    - Maker fee: 0.10% (vs MEXC 0%)
    """

    def __init__(self, api_key: str = "", api_secret: str = "",
                 paper_trading: bool = True):
        self.paper_trading = paper_trading
        self._api_key = api_key
        self._api_secret = api_secret
        self._session: Optional[aiohttp.ClientSession] = None
        self._symbol_info_cache: Dict[str, dict] = {}
        self._last_books: Dict[str, OrderBook] = {}
        self._markets: Dict[str, dict] = {}  # symbol info from instruments-info

        # All-ticker WebSocket
        self._ws_tickers: Dict[str, dict] = {}
        self._ws_ticker_last_update: float = 0.0
        self._ws_ticker_running = False
        self._ws_ticker_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        # Load market info
        try:
            await self._load_instruments()
            logger.info(f"Bybit connected — {len(self._markets)} spot instruments loaded")
        except Exception as e:
            logger.warning(f"Bybit instrument load failed: {e}")

    async def disconnect(self) -> None:
        self._ws_ticker_running = False
        if self._ws_ticker_task and not self._ws_ticker_task.done():
            self._ws_ticker_task.cancel()
        if self._session:
            await self._session.close()
            self._session = None

    # --- Market Data ---

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        raw_sym = self._to_bybit_sym(symbol)
        url = f"{BYBIT_REST}/v5/market/orderbook"
        params = {"category": "spot", "symbol": raw_sym, "limit": depth}
        data = await self._get(url, params)
        result = data.get("result", {})

        bids = [
            OrderBookLevel(Decimal(str(b[0])), Decimal(str(b[1])))
            for b in result.get("b", [])[:depth]
        ]
        asks = [
            OrderBookLevel(Decimal(str(a[0])), Decimal(str(a[1])))
            for a in result.get("a", [])[:depth]
        ]
        ts = result.get("ts", 0)
        dt = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc) if ts else datetime.utcnow()

        book = OrderBook(exchange="bybit", symbol=symbol, timestamp=dt,
                         bids=bids, asks=asks)
        self._last_books[symbol] = book
        return book

    async def subscribe_order_book(self, symbol: str,
                                   callback: Callable[[OrderBook], Awaitable[None]],
                                   depth: int = 20) -> None:
        # Not implemented — use REST polling
        pass

    async def subscribe_trades(self, symbol: str,
                               callback: Callable[[Trade], Awaitable[None]]) -> None:
        # Not implemented — use REST polling
        pass

    async def get_ticker(self, symbol: str) -> dict:
        raw_sym = self._to_bybit_sym(symbol)
        url = f"{BYBIT_REST}/v5/market/tickers"
        params = {"category": "spot", "symbol": raw_sym}
        data = await self._get(url, params)
        items = data.get("result", {}).get("list", [])
        if not items:
            return {'symbol': symbol, 'last_price': Decimal('0'),
                    'bid': Decimal('0'), 'ask': Decimal('0')}
        t = items[0]
        return {
            'symbol': symbol,
            'last_price': Decimal(str(t.get('lastPrice', '0'))),
            'bid': Decimal(str(t.get('bid1Price', '0'))),
            'ask': Decimal(str(t.get('ask1Price', '0'))),
            'volume_24h': Decimal(str(t.get('volume24h', '0'))),
        }

    async def get_all_tickers(self) -> Dict[str, dict]:
        url = f"{BYBIT_REST}/v5/market/tickers"
        params = {"category": "spot"}
        data = await self._get(url, params)
        items = data.get("result", {}).get("list", [])

        result = {}
        for t in items:
            raw_sym = t.get('symbol', '')
            symbol = self._from_bybit_sym(raw_sym)
            if '/' not in symbol:
                continue

            bid = Decimal(str(t.get('bid1Price', '0') or '0'))
            ask = Decimal(str(t.get('ask1Price', '0') or '0'))
            last = Decimal(str(t.get('lastPrice', '0') or '0'))

            result[symbol] = {
                'symbol': symbol,
                'last_price': last,
                'bid': bid,
                'ask': ask,
                'volume_24h': Decimal(str(t.get('volume24h', '0') or '0')),
            }

        return result

    async def get_funding_rate(self, symbol: str) -> FundingRate:
        # Bybit funding rate (futures)
        raw_sym = self._to_bybit_sym(symbol)
        url = f"{BYBIT_REST}/v5/market/tickers"
        params = {"category": "linear", "symbol": raw_sym}
        try:
            data = await self._get(url, params)
            items = data.get("result", {}).get("list", [])
            if items:
                rate = Decimal(str(items[0].get('fundingRate', '0')))
                next_time_str = items[0].get('nextFundingTime', '0')
                next_time = datetime.fromtimestamp(
                    int(next_time_str) / 1000, tz=timezone.utc
                ) if next_time_str and next_time_str != '0' else datetime.utcnow()
                return FundingRate(
                    exchange="bybit", symbol=symbol,
                    current_rate=rate, predicted_rate=None,
                    next_funding_time=next_time,
                    timestamp=datetime.utcnow(),
                )
        except Exception as e:
            logger.debug(f"Bybit funding rate error: {e}")

        return FundingRate(
            exchange="bybit", symbol=symbol,
            current_rate=Decimal('0'), predicted_rate=None,
            next_funding_time=datetime.utcnow(),
            timestamp=datetime.utcnow(),
        )

    # --- Trading ---

    async def place_order(self, order: OrderRequest) -> OrderResult:
        if self.paper_trading:
            return self._paper_fill(order)
        # Live trading not implemented
        raise NotImplementedError("Bybit live trading not implemented")

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if self.paper_trading:
            return True
        raise NotImplementedError

    async def get_order_status(self, symbol: str, order_id: str) -> OrderResult:
        raise NotImplementedError

    # --- Account ---

    async def get_balances(self) -> Dict[str, Balance]:
        if self.paper_trading:
            return self._paper_balances()
        raise NotImplementedError

    async def get_balance(self, currency: str) -> Balance:
        balances = await self.get_balances()
        return balances.get(currency, Balance(
            exchange="bybit", currency=currency,
            free=Decimal('0'), locked=Decimal('0'), total=Decimal('0'),
        ))

    # --- Exchange Info ---

    async def get_trading_fees(self, symbol: str) -> dict:
        return {"maker": MAKER_FEE, "taker": TAKER_FEE}

    async def get_symbol_info(self, symbol: str) -> dict:
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        raw_sym = self._to_bybit_sym(symbol)
        if raw_sym in self._markets:
            m = self._markets[raw_sym]
            info = {
                'symbol': symbol,
                'price_precision': int(m.get('price_precision', 8)),
                'quantity_precision': int(m.get('quantity_precision', 8)),
                'min_quantity': Decimal(str(m.get('min_quantity', '0'))),
                'min_notional': Decimal(str(m.get('min_notional', '0'))),
            }
            self._symbol_info_cache[symbol] = info
            return info

        return {
            'symbol': symbol, 'price_precision': 8,
            'quantity_precision': 8, 'min_quantity': Decimal('0'),
            'min_notional': Decimal('0'),
        }

    # --- WebSocket All-Ticker Feed ---

    async def subscribe_all_tickers(self, callback: Optional[Callable] = None):
        """Subscribe to Bybit spot ticker stream."""
        if self._ws_ticker_running:
            return
        self._ws_ticker_running = True
        self._ws_ticker_task = asyncio.create_task(self._ws_ticker_loop())
        logger.info("Bybit all-ticker WebSocket subscription started")

    async def _ws_ticker_loop(self):
        """Reconnection loop for all-ticker stream."""
        import websockets
        reconnect_attempts = 0
        while self._ws_ticker_running:
            try:
                async with websockets.connect(
                    BYBIT_WS, ping_interval=20, close_timeout=5
                ) as ws:
                    logger.info("Bybit ticker WebSocket connected")
                    # Subscribe to all tickers
                    sub = {"op": "subscribe", "args": ["tickers.BTCUSDT"]}
                    # Bybit doesn't have a single "all tickers" stream,
                    # so we subscribe to individual ones from our market list
                    symbols_to_sub = list(self._markets.keys())[:500]
                    # Subscribe in batches of 10
                    for i in range(0, len(symbols_to_sub), 10):
                        batch = symbols_to_sub[i:i+10]
                        sub_msg = {
                            "op": "subscribe",
                            "args": [f"tickers.{s}" for s in batch]
                        }
                        await ws.send(json.dumps(sub_msg))
                        await asyncio.sleep(0.1)

                    reconnect_attempts = 0
                    connect_time = time.monotonic()

                    async for raw in ws:
                        if not self._ws_ticker_running:
                            break
                        if time.monotonic() - connect_time > 23 * 3600:
                            break
                        try:
                            msg = json.loads(raw)
                            if msg.get("topic", "").startswith("tickers."):
                                self._handle_bybit_ticker(msg)
                        except Exception as e:
                            logger.debug(f"Bybit WS parse error: {e}")

            except asyncio.CancelledError:
                return
            except Exception as e:
                reconnect_attempts += 1
                backoff = min(2 ** reconnect_attempts, 30)
                logger.warning(f"Bybit ticker WS error: {e} — retry in {backoff}s")
                await asyncio.sleep(backoff)

    def _handle_bybit_ticker(self, msg: dict):
        """Parse Bybit ticker update."""
        data = msg.get("data", {})
        raw_sym = data.get("symbol", "")
        symbol = self._from_bybit_sym(raw_sym)
        if '/' not in symbol:
            return

        bid = data.get("bid1Price", "0")
        ask = data.get("ask1Price", "0")
        last = data.get("lastPrice", "0")

        self._ws_tickers[symbol] = {
            'symbol': symbol,
            'last_price': Decimal(str(last)),
            'bid': Decimal(str(bid)),
            'ask': Decimal(str(ask)),
            'volume_24h': Decimal(str(data.get('volume24h', '0'))),
        }
        self._ws_ticker_last_update = time.time()

    def get_ws_tickers(self) -> Optional[Dict[str, dict]]:
        if self._ws_tickers and self._ws_ticker_last_update > 0:
            age = time.time() - self._ws_ticker_last_update
            if age < 10:
                return self._ws_tickers
        return None

    def get_ws_ticker_age_ms(self) -> float:
        if self._ws_ticker_last_update <= 0:
            return float('inf')
        return (time.time() - self._ws_ticker_last_update) * 1000

    # --- Internal Helpers ---

    async def _load_instruments(self):
        """Load all spot instrument info from Bybit."""
        url = f"{BYBIT_REST}/v5/market/instruments-info"
        params = {"category": "spot", "limit": 1000}
        data = await self._get(url, params)
        items = data.get("result", {}).get("list", [])

        for item in items:
            sym = item.get("symbol", "")
            lot_filter = item.get("lotSizeFilter", {})
            price_filter = item.get("priceFilter", {})

            # Calculate precision from tick/step size
            tick_size = price_filter.get("tickSize", "0.01")
            step_size = lot_filter.get("basePrecision", "0.01")

            price_prec = self._precision_from_step(tick_size)
            qty_prec = self._precision_from_step(step_size)

            self._markets[sym] = {
                'symbol': sym,
                'price_precision': price_prec,
                'quantity_precision': qty_prec,
                'min_quantity': lot_filter.get("minOrderQty", "0"),
                'min_notional': lot_filter.get("minOrderAmt", "0"),
            }

    @staticmethod
    def _precision_from_step(step: str) -> int:
        """Convert step size like '0.001' to precision 3."""
        if '.' not in step:
            return 0
        return len(step.rstrip('0').split('.')[1])

    async def _get(self, url: str, params: dict = None) -> dict:
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        async with self._session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Bybit API {resp.status}: {text[:200]}")
            data = await resp.json()
            if data.get("retCode", 0) != 0:
                raise Exception(f"Bybit API error: {data.get('retMsg', 'unknown')}")
            return data

    def _to_bybit_sym(self, symbol: str) -> str:
        """BTC/USDT → BTCUSDT"""
        return symbol.replace("/", "").upper()

    def _from_bybit_sym(self, raw: str) -> str:
        """BTCUSDT → BTC/USDT"""
        raw = raw.upper()
        for quote in ("USDT", "USDC", "BTC", "ETH", "EUR", "DAI"):
            if raw.endswith(quote):
                base = raw[:-len(quote)]
                if base:
                    return f"{base}/{quote}"
        return raw

    def _paper_fill(self, order: OrderRequest) -> OrderResult:
        """Simulate a fill for paper trading with Bybit fee model."""
        book = self._last_books.get(order.symbol)

        if order.order_type == OrderType.MARKET:
            if book:
                fill_price = book.best_ask if order.side == OrderSide.BUY else book.best_bid
            else:
                fill_price = order.price or Decimal('0')
        else:
            fill_price = order.price

        # Bybit fees
        if order.order_type == OrderType.MARKET:
            fee_rate = TAKER_FEE
        elif order.order_type == OrderType.LIMIT_MAKER:
            fee_rate = MAKER_FEE
        else:
            fee_rate = TAKER_FEE  # Conservative

        if order.side == OrderSide.BUY:
            fee = order.quantity * fee_rate
        else:
            fee = order.quantity * (fill_price or Decimal('0')) * fee_rate

        import random
        if order.order_type == OrderType.LIMIT_MAKER and random.random() > 0.80:
            return OrderResult(
                exchange="bybit", symbol=order.symbol,
                order_id=f"bybit_paper_{int(time.time()*1000)}",
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
            exchange="bybit", symbol=order.symbol,
            order_id=f"bybit_paper_{int(time.time()*1000)}",
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
            "USDT": Balance("bybit", "USDT", Decimal('10000'), Decimal('0'), Decimal('10000')),
            "BTC": Balance("bybit", "BTC", Decimal('0.15'), Decimal('0'), Decimal('0.15')),
            "ETH": Balance("bybit", "ETH", Decimal('2.5'), Decimal('0'), Decimal('2.5')),
        }
