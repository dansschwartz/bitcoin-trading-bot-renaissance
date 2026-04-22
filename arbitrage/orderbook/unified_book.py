"""
Unified Order Book Manager — maintains real-time order books for all
monitored pairs across both MEXC and Binance simultaneously.

Source of truth for detecting arbitrage opportunities.
Uses Decimal for all prices/quantities. Validates freshness before trading.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List

from ..exchanges.base import OrderBook, OrderBookLevel, OrderSide

logger = logging.getLogger("arb.orderbook")

PHASE_1_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
]


@dataclass
class UnifiedPairView:
    """Combined view of one trading pair across exchanges (MEXC, Binance, optional KuCoin, optional Binance US)."""
    symbol: str
    mexc_book: Optional[OrderBook] = None
    binance_book: Optional[OrderBook] = None
    kucoin_book: Optional[OrderBook] = None
    binance_us_book: Optional[OrderBook] = None
    mexc_last_update: datetime = field(default_factory=datetime.utcnow)
    binance_last_update: datetime = field(default_factory=datetime.utcnow)
    kucoin_last_update: datetime = field(default_factory=datetime.utcnow)
    binance_us_last_update: datetime = field(default_factory=datetime.utcnow)
    # Tracking
    mexc_update_count: int = 0
    binance_update_count: int = 0
    kucoin_update_count: int = 0
    binance_us_update_count: int = 0

    @property
    def is_fresh(self) -> bool:
        now = datetime.utcnow()
        max_age = timedelta(seconds=90)  # 90s — tolerant of sequential REST polling (30 pairs × 2 exchanges)
        fresh = (
            (now - self.mexc_last_update) < max_age
            and (now - self.binance_last_update) < max_age
        )
        # KuCoin freshness only required if book exists
        if self.kucoin_book is not None:
            fresh = fresh and (now - self.kucoin_last_update) < max_age
        # Binance US freshness only required if book exists
        if self.binance_us_book is not None:
            fresh = fresh and (now - self.binance_us_last_update) < max_age
        return fresh

    @property
    def is_tradeable(self) -> bool:
        return (
            self.mexc_book is not None
            and self.binance_book is not None
            and self.is_fresh
            and self.mexc_book.best_bid is not None
            and self.mexc_book.best_ask is not None
            and self.binance_book.best_bid is not None
            and self.binance_book.best_ask is not None
        )

    @property
    def has_any_cross_exchange_books(self) -> bool:
        """True if MEXC has a book AND at least one other exchange has a book.
        Used by the detector to allow MEXC↔BinanceUS arb even when Binance Intl is down."""
        if self.mexc_book is None or self.mexc_book.best_bid is None:
            return False
        now = datetime.utcnow()
        max_age = timedelta(seconds=90)
        if (now - self.mexc_last_update) >= max_age:
            return False
        for book, ts in [
            (self.binance_book, self.binance_last_update),
            (self.kucoin_book, self.kucoin_last_update),
            (self.binance_us_book, self.binance_us_last_update),
        ]:
            if book is not None and book.best_bid is not None and (now - ts) < max_age:
                return True
        return False

    def get_cross_exchange_spread(self) -> Optional[dict]:
        """
        Calculate spreads in both directions:
        1. Buy MEXC, Sell Binance
        2. Buy Binance, Sell MEXC
        Returns the profitable direction (if any).
        """
        if not self.is_tradeable:
            return None

        mexc_ask = self.mexc_book.best_ask
        mexc_bid = self.mexc_book.best_bid
        binance_ask = self.binance_book.best_ask
        binance_bid = self.binance_book.best_bid

        # Direction 1: Buy on MEXC (at ask), Sell on Binance (at bid)
        if mexc_ask > 0:
            spread_1_bps = ((binance_bid - mexc_ask) / mexc_ask) * 10000
        else:
            spread_1_bps = Decimal('0')

        # Direction 2: Buy on Binance (at ask), Sell on MEXC (at bid)
        if binance_ask > 0:
            spread_2_bps = ((mexc_bid - binance_ask) / binance_ask) * 10000
        else:
            spread_2_bps = Decimal('0')

        if spread_1_bps > spread_2_bps and spread_1_bps > 0:
            return {
                "direction": "buy_mexc_sell_binance",
                "buy_exchange": "mexc",
                "sell_exchange": "binance",
                "buy_price": mexc_ask,
                "sell_price": binance_bid,
                "gross_spread_bps": spread_1_bps,
                "buy_depth": self.mexc_book.available_liquidity_at_impact(
                    OrderSide.BUY, Decimal('5')
                ),
                "sell_depth": self.binance_book.available_liquidity_at_impact(
                    OrderSide.SELL, Decimal('5')
                ),
            }
        elif spread_2_bps > 0:
            return {
                "direction": "buy_binance_sell_mexc",
                "buy_exchange": "binance",
                "sell_exchange": "mexc",
                "buy_price": binance_ask,
                "sell_price": mexc_bid,
                "gross_spread_bps": spread_2_bps,
                "buy_depth": self.binance_book.available_liquidity_at_impact(
                    OrderSide.BUY, Decimal('5')
                ),
                "sell_depth": self.mexc_book.available_liquidity_at_impact(
                    OrderSide.SELL, Decimal('5')
                ),
            }

        return None

    def get_all_cross_exchange_spreads(self) -> List[dict]:
        """Returns all profitable cross-exchange spreads (may include multiple).

        Checks MEXC↔Binance (existing) plus MEXC↔KuCoin if kucoin book exists.
        Each spread dict has the same format as get_cross_exchange_spread().
        """
        results = []

        # MEXC vs Binance (existing logic)
        spread = self.get_cross_exchange_spread()
        if spread:
            results.append(spread)

        # MEXC vs KuCoin
        if self.kucoin_book and self.mexc_book:
            kc_spread = self._cross_spread_between(
                "mexc", self.mexc_book, "kucoin", self.kucoin_book
            )
            if kc_spread:
                results.append(kc_spread)

        # MEXC vs Binance US (most profitable route: 0.1 bps total cost)
        if self.binance_us_book and self.mexc_book:
            bus_spread = self._cross_spread_between(
                "mexc", self.mexc_book, "binance_us", self.binance_us_book
            )
            if bus_spread:
                results.append(bus_spread)

        # Binance International vs Binance US
        if self.binance_us_book and self.binance_book:
            b_bus_spread = self._cross_spread_between(
                "binance", self.binance_book, "binance_us", self.binance_us_book
            )
            if b_bus_spread:
                results.append(b_bus_spread)

        return results

    def _cross_spread_between(
        self, exch_a: str, book_a: OrderBook, exch_b: str, book_b: OrderBook
    ) -> Optional[dict]:
        """Generic spread calculation between two exchange books.

        Same logic as get_cross_exchange_spread() but parameterized.
        """
        if not (book_a.best_bid and book_a.best_ask and book_b.best_bid and book_b.best_ask):
            return None

        a_ask = book_a.best_ask
        a_bid = book_a.best_bid
        b_ask = book_b.best_ask
        b_bid = book_b.best_bid

        # Direction 1: Buy on A, Sell on B
        spread_1_bps = ((b_bid - a_ask) / a_ask) * 10000 if a_ask > 0 else Decimal('0')

        # Direction 2: Buy on B, Sell on A
        spread_2_bps = ((a_bid - b_ask) / b_ask) * 10000 if b_ask > 0 else Decimal('0')

        if spread_1_bps > spread_2_bps and spread_1_bps > 0:
            return {
                "direction": f"buy_{exch_a}_sell_{exch_b}",
                "buy_exchange": exch_a,
                "sell_exchange": exch_b,
                "buy_price": a_ask,
                "sell_price": b_bid,
                "gross_spread_bps": spread_1_bps,
                "buy_depth": book_a.available_liquidity_at_impact(OrderSide.BUY, Decimal('5')),
                "sell_depth": book_b.available_liquidity_at_impact(OrderSide.SELL, Decimal('5')),
            }
        elif spread_2_bps > 0:
            return {
                "direction": f"buy_{exch_b}_sell_{exch_a}",
                "buy_exchange": exch_b,
                "sell_exchange": exch_a,
                "buy_price": b_ask,
                "sell_price": a_bid,
                "gross_spread_bps": spread_2_bps,
                "buy_depth": book_b.available_liquidity_at_impact(OrderSide.BUY, Decimal('5')),
                "sell_depth": book_a.available_liquidity_at_impact(OrderSide.SELL, Decimal('5')),
            }

        return None


class UnifiedBookManager:
    """Manages order books for all monitored pairs across both exchanges.

    Supports dynamic pair addition/removal via add_pair()/remove_pair()
    for use by PairDiscoveryEngine.
    """

    def __init__(self, mexc_client, binance_client, pairs: Optional[List[str]] = None,
                 bar_aggregator=None, kucoin=None, binance_us=None,
                 skip_mexc_ws: bool = False):
        self.mexc = mexc_client
        self.binance = binance_client
        self.kucoin = kucoin  # Optional KuCoin spoke exchange
        self.binance_us = binance_us  # Optional Binance US spoke exchange
        self.monitored_pairs = pairs or PHASE_1_PAIRS
        self._initial_pairs: List[str] = list(self.monitored_pairs)  # Static pairs (never demoted)
        self.pairs: Dict[str, UnifiedPairView] = {}
        self._bar_aggregator = bar_aggregator
        self._running = False
        self._validation_task = None
        self._skip_mexc_ws = skip_mexc_ws  # When True, MEXC WS subscriptions are skipped (relay provides data)
        self._mexc_relay_hook = None  # Optional callback: async (pair, book) -> None

    async def start(self):
        self._running = True

        for pair in self.monitored_pairs:
            self.pairs[pair] = UnifiedPairView(symbol=pair)

        tasks = []
        for pair in self.monitored_pairs:
            if not self._skip_mexc_ws:
                tasks.append(self.mexc.subscribe_order_book(
                    pair,
                    callback=lambda book, p=pair: self._on_mexc_update(p, book),
                    depth=20,
                ))
            tasks.append(self.binance.subscribe_order_book(
                pair,
                callback=lambda book, p=pair: self._on_binance_update(p, book),
                depth=20,
            ))
            if self.kucoin:
                tasks.append(self.kucoin.subscribe_order_book(
                    pair,
                    callback=lambda book, p=pair: self._on_kucoin_update(p, book),
                    depth=20,
                ))
            if self.binance_us:
                tasks.append(self.binance_us.subscribe_order_book(
                    pair,
                    callback=lambda book, p=pair: self._on_binance_us_update(p, book),
                    depth=20,
                ))

        self._validation_task = asyncio.create_task(self._validation_loop())

        kc_tag = " + KuCoin" if self.kucoin else ""
        bus_tag = " + BinanceUS" if self.binance_us else ""
        relay_tag = " [MEXC via relay]" if self._skip_mexc_ws else ""
        logger.info(f"UnifiedBookManager started — monitoring {len(self.monitored_pairs)} pairs (MEXC + Binance{kc_tag}{bus_tag}){relay_tag}")
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        self._running = False
        if self._validation_task:
            self._validation_task.cancel()

    async def add_pair(self, pair: str) -> bool:
        """Dynamically add a pair to active monitoring. Returns True if newly added."""
        if pair in self.pairs:
            return False
        self.pairs[pair] = UnifiedPairView(symbol=pair)
        # Subscribe MEXC order book (per-symbol WebSocket, works immediately)
        try:
            await self.mexc.subscribe_order_book(
                pair,
                callback=lambda book, p=pair: self._on_mexc_update(p, book),
                depth=20,
            )
        except Exception as e:
            logger.warning(f"MEXC subscribe failed for dynamic pair {pair}: {e}")
        # Binance: initial REST fetch (no WS reconnect needed — REST refresh handles it)
        try:
            rest_book = await self.binance.get_order_book(pair, depth=20)
            if pair in self.pairs:
                self.pairs[pair].binance_book = rest_book
                self.pairs[pair].binance_last_update = datetime.utcnow()
        except Exception as e:
            logger.debug(f"Binance initial fetch failed for dynamic pair {pair}: {e}")
        # Binance US: initial REST fetch if available
        if self.binance_us:
            try:
                rest_book = await self.binance_us.get_order_book(pair, depth=20)
                if pair in self.pairs:
                    self.pairs[pair].binance_us_book = rest_book
                    self.pairs[pair].binance_us_last_update = datetime.utcnow()
            except Exception as e:
                logger.debug(f"Binance US initial fetch failed for dynamic pair {pair}: {e}")
        return True

    async def remove_pair(self, pair: str) -> bool:
        """Dynamically remove a pair from active monitoring.

        Only removes dynamically-added pairs; initial (static) pairs are protected.
        Returns True if the pair was removed.
        """
        if pair not in self.pairs:
            return False
        if pair in self._initial_pairs:
            logger.debug(f"Cannot remove static pair {pair}")
            return False
        del self.pairs[pair]
        # MEXC WS callback cleanup happens naturally (no messages for removed symbol)
        return True

    async def _on_mexc_update(self, pair: str, book: OrderBook):
        if pair in self.pairs:
            self.pairs[pair].mexc_book = book
            self.pairs[pair].mexc_last_update = datetime.utcnow()
            self.pairs[pair].mexc_update_count += 1
            if self.pairs[pair].mexc_update_count == 1:
                logger.info(f"MEXC first book for {pair}: bid={book.best_bid} ask={book.best_ask}")
            if self._bar_aggregator and book.best_bid and book.best_ask:
                try:
                    self._bar_aggregator.on_orderbook_snapshot(
                        pair=pair, exchange="mexc",
                        best_bid=float(book.best_bid), best_ask=float(book.best_ask),
                        timestamp=book.timestamp.timestamp(),
                    )
                except Exception:
                    pass
            # Relay hook — broadcast to remote clients (Bangalore → US)
            if self._mexc_relay_hook:
                try:
                    await self._mexc_relay_hook(pair, book)
                except Exception:
                    pass
        else:
            # Silently ignore — dynamic pairs may be demoted but WS subscription lingers
            pass

    async def _on_binance_update(self, pair: str, book: OrderBook):
        if pair in self.pairs:
            self.pairs[pair].binance_book = book
            self.pairs[pair].binance_last_update = datetime.utcnow()
            self.pairs[pair].binance_update_count += 1
            if self._bar_aggregator and book.best_bid and book.best_ask:
                try:
                    self._bar_aggregator.on_orderbook_snapshot(
                        pair=pair, exchange="binance",
                        best_bid=float(book.best_bid), best_ask=float(book.best_ask),
                        timestamp=book.timestamp.timestamp(),
                    )
                except Exception:
                    pass

    async def _on_kucoin_update(self, pair: str, book: OrderBook):
        if pair in self.pairs:
            self.pairs[pair].kucoin_book = book
            self.pairs[pair].kucoin_last_update = datetime.utcnow()
            self.pairs[pair].kucoin_update_count += 1
            if self.pairs[pair].kucoin_update_count == 1:
                logger.info(f"KuCoin first book for {pair}: bid={book.best_bid} ask={book.best_ask}")

    async def _on_binance_us_update(self, pair: str, book: OrderBook):
        if pair in self.pairs:
            self.pairs[pair].binance_us_book = book
            self.pairs[pair].binance_us_last_update = datetime.utcnow()
            self.pairs[pair].binance_us_update_count += 1
            if self.pairs[pair].binance_us_update_count == 1:
                logger.info(f"Binance US first book for {pair}: bid={book.best_bid} ask={book.best_ask}")

    async def _refresh_pair(self, pair: str) -> None:
        """REST-refresh a single pair on both exchanges (used by validation loop)."""
        # MEXC
        try:
            rest_book = await self.mexc.get_order_book(pair, depth=20)
            if pair in self.pairs:
                local_book = self.pairs[pair].mexc_book
                if local_book and rest_book and rest_book.best_bid and local_book.best_bid:
                    diff = abs(local_book.best_bid - rest_book.best_bid)
                    if rest_book.best_bid > 0 and diff / rest_book.best_bid > Decimal('0.001'):
                        logger.warning(
                            f"Book drift: {pair} MEXC local={local_book.best_bid} "
                            f"rest={rest_book.best_bid}"
                        )
                self.pairs[pair].mexc_book = rest_book
                self.pairs[pair].mexc_last_update = datetime.utcnow()
        except Exception as e:
            logger.debug(f"MEXC validation error {pair}: {e}")

        # Binance
        try:
            rest_book = await self.binance.get_order_book(pair, depth=20)
            if pair in self.pairs:
                local_book = self.pairs[pair].binance_book
                if local_book and rest_book and rest_book.best_bid and local_book.best_bid:
                    diff = abs(local_book.best_bid - rest_book.best_bid)
                    if rest_book.best_bid > 0 and diff / rest_book.best_bid > Decimal('0.001'):
                        logger.warning(
                            f"Book drift: {pair} Binance local={local_book.best_bid} "
                            f"rest={rest_book.best_bid}"
                        )
                self.pairs[pair].binance_book = rest_book
                self.pairs[pair].binance_last_update = datetime.utcnow()
        except Exception as e:
            logger.debug(f"Binance validation error {pair}: {e}")

    def _validation_thread_func(self):
        """Validation loop running in a standalone daemon thread.

        Completely bypasses the asyncio event loop — uses synchronous urllib
        with OS-level socket timeouts (enforced by kernel setsockopt).
        Updates self.pairs directly — attribute assignments are atomic under GIL.
        """
        import json as _json
        import time as _time
        import urllib.request

        SOCK_TIMEOUT = 5.0   # OS-level socket timeout per request
        CYCLE_INTERVAL = 15  # seconds between refresh cycles
        PAIR_DELAY = 0.10    # 100ms between pairs (rate limiting)
        WS_FRESH_THRESHOLD = 5.0  # skip REST if WS/relay data is younger than this (seconds)
        _HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}

        _URLS = {
            "mexc": ("https://api.mexc.com/api/v3/depth?symbol={sym}&limit=20",
                     lambda p: p.replace("/", "")),
            "binance": ("https://api.binance.com/api/v3/depth?symbol={sym}&limit=20",
                        lambda p: p.replace("/", "")),
            "kucoin": ("https://api.kucoin.com/api/v1/market/orderbook/level2_20?symbol={sym}",
                       lambda p: p.replace("/", "-")),
            "binance_us": ("https://api.binance.us/api/v3/depth?symbol={sym}&limit=20",
                           lambda p: p.replace("/", "")),
        }

        def _fetch(exchange: str, pair: str) -> Optional[dict]:
            try:
                url_tpl, sym_fn = _URLS[exchange]
                url = url_tpl.format(sym=sym_fn(pair))
                req = urllib.request.Request(url, headers=_HEADERS)
                with urllib.request.urlopen(req, timeout=SOCK_TIMEOUT) as resp:
                    return _json.loads(resp.read().decode())
            except Exception:
                return None

        def _parse_book(exchange: str, pair: str, data: dict) -> Optional[OrderBook]:
            try:
                bids_raw = data.get("bids", [])[:20]
                asks_raw = data.get("asks", [])[:20]
                bids = [OrderBookLevel(Decimal(str(b[0])), Decimal(str(b[1]))) for b in bids_raw]
                asks = [OrderBookLevel(Decimal(str(a[0])), Decimal(str(a[1]))) for a in asks_raw]
                return OrderBook(exchange=exchange, symbol=pair,
                                 timestamp=datetime.utcnow(), bids=bids, asks=asks)
            except Exception:
                return None

        logger.info("Validation thread started — urllib sync, refresh every %ds", CYCLE_INTERVAL)

        while self._running:
            _time.sleep(CYCLE_INTERVAL)
            if not self._running:
                break

            pairs = list(self.pairs.keys())
            if not pairs:
                continue

            t_start = _time.monotonic()
            mexc_ws = mexc_rest = mexc_fail = 0
            binance_ws = binance_rest = binance_fail = 0
            kucoin_ws = kucoin_rest = kucoin_fail = 0
            binance_us_ws = binance_us_rest = binance_us_fail = 0
            has_kucoin = self.kucoin is not None
            has_binance_us = self.binance_us is not None

            for pair in pairs:
                if not self._running:
                    break

                # --- MEXC ---
                if pair in self.pairs:
                    _age = (datetime.utcnow() - self.pairs[pair].mexc_last_update).total_seconds()
                    if _age < WS_FRESH_THRESHOLD:
                        mexc_ws += 1
                    else:
                        data = _fetch("mexc", pair)
                        if data and "bids" in data:
                            book = _parse_book("mexc", pair, data)
                            if book:
                                self.pairs[pair].mexc_book = book
                                self.pairs[pair].mexc_last_update = datetime.utcnow()
                                self.pairs[pair].mexc_update_count += 1
                                mexc_rest += 1
                            else:
                                mexc_fail += 1
                        else:
                            mexc_fail += 1

                # --- Binance ---
                if pair in self.pairs:
                    _age = (datetime.utcnow() - self.pairs[pair].binance_last_update).total_seconds()
                    if _age < WS_FRESH_THRESHOLD:
                        binance_ws += 1
                    else:
                        data = _fetch("binance", pair)
                        if data and "bids" in data:
                            book = _parse_book("binance", pair, data)
                            if book:
                                self.pairs[pair].binance_book = book
                                self.pairs[pair].binance_last_update = datetime.utcnow()
                                self.pairs[pair].binance_update_count += 1
                                binance_rest += 1
                            else:
                                binance_fail += 1
                        else:
                            binance_fail += 1

                # --- KuCoin ---
                if has_kucoin and pair in self.pairs:
                    _age = (datetime.utcnow() - self.pairs[pair].kucoin_last_update).total_seconds()
                    if _age < WS_FRESH_THRESHOLD:
                        kucoin_ws += 1
                    else:
                        data = _fetch("kucoin", pair)
                        if data and data.get("code") == "200000" and "data" in data:
                            try:
                                rest_book = self.kucoin._parse_rest_book(pair, data["data"])
                                self.pairs[pair].kucoin_book = rest_book
                                self.pairs[pair].kucoin_last_update = datetime.utcnow()
                                self.pairs[pair].kucoin_update_count += 1
                                kucoin_rest += 1
                            except Exception:
                                kucoin_fail += 1
                        else:
                            kucoin_fail += 1

                # --- Binance US ---
                if has_binance_us and pair in self.pairs:
                    _age = (datetime.utcnow() - self.pairs[pair].binance_us_last_update).total_seconds()
                    if _age < WS_FRESH_THRESHOLD:
                        binance_us_ws += 1
                    else:
                        data = _fetch("binance_us", pair)
                        if data and "bids" in data:
                            book = _parse_book("binance_us", pair, data)
                            if book:
                                self.pairs[pair].binance_us_book = book
                                self.pairs[pair].binance_us_last_update = datetime.utcnow()
                                self.pairs[pair].binance_us_update_count += 1
                                binance_us_rest += 1
                            else:
                                binance_us_fail += 1
                        else:
                            binance_us_fail += 1

                _time.sleep(PAIR_DELAY)

            elapsed = _time.monotonic() - t_start
            tradeable = sum(1 for v in self.pairs.values() if v.is_tradeable)
            kc_str = f" | KuCoin {kucoin_ws}ws/{kucoin_rest}rest/{kucoin_fail}fail" if has_kucoin else ""
            bus_str = f" | BinanceUS {binance_us_ws}ws/{binance_us_rest}rest/{binance_us_fail}fail" if has_binance_us else ""
            logger.info(
                f"Book refresh: {len(pairs)} pairs | "
                f"MEXC {mexc_ws}ws/{mexc_rest}rest/{mexc_fail}fail | "
                f"Binance {binance_ws}ws/{binance_rest}rest/{binance_fail}fail"
                f"{kc_str}{bus_str} | "
                f"{tradeable} tradeable | {elapsed:.1f}s"
            )

        logger.info("Validation thread stopped")

    async def _validation_loop(self):
        """Start the validation thread (daemon) — no event loop dependency."""
        import threading
        t = threading.Thread(
            target=self._validation_thread_func,
            name="val-book-thread",
            daemon=True,
        )
        t.start()
        logger.info("Validation daemon thread launched (tid=%s)", t.ident)
        # Keep this coroutine alive so the task isn't garbage-collected
        while self._running:
            await asyncio.sleep(5)
        t.join(timeout=3)

    def get_status(self) -> dict:
        fresh = sum(1 for v in self.pairs.values() if v.is_tradeable)
        total_updates = sum(
            v.mexc_update_count + v.binance_update_count + v.kucoin_update_count + v.binance_us_update_count
            for v in self.pairs.values()
        )
        return {
            "total_pairs": len(self.pairs),
            "tradeable_pairs": fresh,
            "total_updates": total_updates,
            "kucoin_enabled": self.kucoin is not None,
            "binance_us_enabled": self.binance_us is not None,
            "pairs": {
                p: {
                    "tradeable": v.is_tradeable,
                    "mexc_updates": v.mexc_update_count,
                    "binance_updates": v.binance_update_count,
                    "kucoin_updates": v.kucoin_update_count,
                    "binance_us_updates": v.binance_us_update_count,
                    "mexc_spread": float(v.mexc_book.spread_bps) if v.mexc_book and v.mexc_book.spread_bps else None,
                    "binance_spread": float(v.binance_book.spread_bps) if v.binance_book and v.binance_book.spread_bps else None,
                    "kucoin_spread": float(v.kucoin_book.spread_bps) if v.kucoin_book and v.kucoin_book.spread_bps else None,
                    "binance_us_spread": float(v.binance_us_book.spread_bps) if v.binance_us_book and v.binance_us_book.spread_bps else None,
                }
                for p, v in self.pairs.items()
            }
        }
