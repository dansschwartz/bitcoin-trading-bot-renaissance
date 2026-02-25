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

from ..exchanges.base import OrderBook, OrderSide

logger = logging.getLogger("arb.orderbook")

PHASE_1_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
]


@dataclass
class UnifiedPairView:
    """Combined view of one trading pair across both exchanges."""
    symbol: str
    mexc_book: Optional[OrderBook] = None
    binance_book: Optional[OrderBook] = None
    mexc_last_update: datetime = field(default_factory=datetime.utcnow)
    binance_last_update: datetime = field(default_factory=datetime.utcnow)
    # Tracking
    mexc_update_count: int = 0
    binance_update_count: int = 0

    @property
    def is_fresh(self) -> bool:
        now = datetime.utcnow()
        max_age = timedelta(seconds=10)  # 10s — tolerant of REST fallback polling
        return (
            (now - self.mexc_last_update) < max_age
            and (now - self.binance_last_update) < max_age
        )

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


class UnifiedBookManager:
    """Manages order books for all monitored pairs across both exchanges.

    Supports dynamic pair addition/removal via add_pair()/remove_pair()
    for use by PairDiscoveryEngine.
    """

    def __init__(self, mexc_client, binance_client, pairs: Optional[List[str]] = None,
                 bar_aggregator=None):
        self.mexc = mexc_client
        self.binance = binance_client
        self.monitored_pairs = pairs or PHASE_1_PAIRS
        self._initial_pairs: List[str] = list(self.monitored_pairs)  # Static pairs (never demoted)
        self.pairs: Dict[str, UnifiedPairView] = {}
        self._bar_aggregator = bar_aggregator
        self._running = False
        self._validation_task = None

    async def start(self):
        self._running = True

        for pair in self.monitored_pairs:
            self.pairs[pair] = UnifiedPairView(symbol=pair)

        tasks = []
        for pair in self.monitored_pairs:
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

        self._validation_task = asyncio.create_task(self._validation_loop())

        logger.info(f"UnifiedBookManager started — monitoring {len(self.monitored_pairs)} pairs")
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

    async def _validation_loop(self):
        """Periodically refresh all pairs via REST (parallel with concurrency limit).

        Uses self.pairs.keys() instead of self.monitored_pairs so dynamically
        added pairs are also refreshed. Semaphore limits concurrent REST calls
        to avoid hitting Binance rate limits (1200 weight/min, depth=20 costs 5 each).
        """
        sem = asyncio.Semaphore(5)  # Max 5 concurrent pair refreshes (= 10 REST calls)

        async def _limited_refresh(pair: str) -> None:
            async with sem:
                await self._refresh_pair(pair)

        while self._running:
            await asyncio.sleep(30)  # 30s refresh — 40 pairs × 2 × 5 weight / 30s = 667 wt/min (under 1200)
            pairs = list(self.pairs.keys())
            if not pairs:
                continue
            await asyncio.gather(
                *[_limited_refresh(p) for p in pairs],
                return_exceptions=True,
            )

    def get_status(self) -> dict:
        fresh = sum(1 for v in self.pairs.values() if v.is_tradeable)
        total_updates = sum(v.mexc_update_count + v.binance_update_count for v in self.pairs.values())
        return {
            "total_pairs": len(self.pairs),
            "tradeable_pairs": fresh,
            "total_updates": total_updates,
            "pairs": {
                p: {
                    "tradeable": v.is_tradeable,
                    "mexc_updates": v.mexc_update_count,
                    "binance_updates": v.binance_update_count,
                    "mexc_spread": float(v.mexc_book.spread_bps) if v.mexc_book and v.mexc_book.spread_bps else None,
                    "binance_spread": float(v.binance_book.spread_bps) if v.binance_book and v.binance_book.spread_bps else None,
                }
                for p, v in self.pairs.items()
            }
        }
