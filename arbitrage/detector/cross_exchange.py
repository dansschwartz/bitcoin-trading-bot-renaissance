"""
Cross-Exchange Arbitrage Detector — scans unified order books for
profitable price discrepancies between MEXC and Binance.

Produces ArbitrageSignal objects for the execution engine.
Does NOT execute trades — only detects and signals.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
import uuid

logger = logging.getLogger("arb.detector")


@dataclass
class ArbitrageSignal:
    signal_id: str
    signal_type: str              # "cross_exchange" | "funding_rate" | "triangular"
    timestamp: datetime
    symbol: str

    # Direction
    buy_exchange: str
    sell_exchange: str

    # Prices
    buy_price: Decimal
    sell_price: Decimal

    # Spread analysis
    gross_spread_bps: Decimal
    total_cost_bps: Decimal
    net_spread_bps: Decimal

    # Sizing
    max_quantity: Decimal
    recommended_quantity: Decimal
    expected_profit_usd: Decimal

    # Cost breakdown
    buy_fee_bps: Decimal
    sell_fee_bps: Decimal
    buy_slippage_bps: Decimal
    sell_slippage_bps: Decimal

    # Validity
    expires_at: datetime
    confidence: Decimal


class CrossExchangeDetector:
    """Scans for cross-exchange spot arbitrage opportunities."""

    MIN_NET_SPREAD_BPS = Decimal('1.0')   # 1 bps minimum profit
    SIGNAL_TTL_SECONDS = 5
    MIN_TRADE_USD = Decimal('100')
    MAX_TRADE_USD = Decimal('1000')

    def __init__(self, book_manager, cost_model, risk_engine, signal_queue: asyncio.Queue,
                 config: Optional[dict] = None):
        self.books = book_manager
        self.costs = cost_model
        self.risk = risk_engine
        self.signal_queue = signal_queue

        # Override from config
        if config:
            cx_cfg = config.get('cross_exchange', {})
            if 'min_trade_usd' in cx_cfg:
                self.MIN_TRADE_USD = Decimal(str(cx_cfg['min_trade_usd']))
            if 'max_trade_usd' in cx_cfg:
                self.MAX_TRADE_USD = Decimal(str(cx_cfg['max_trade_usd']))
            if 'min_net_spread_bps' in cx_cfg:
                self.MIN_NET_SPREAD_BPS = Decimal(str(cx_cfg['min_net_spread_bps']))
            if 'signal_ttl_seconds' in cx_cfg:
                self.SIGNAL_TTL_SECONDS = cx_cfg['signal_ttl_seconds']
        self._running = False
        self._scan_count = 0
        self._signals_generated = 0
        self._signals_approved = 0
        self._last_spreads: dict = {}  # pair -> last gross spread for stability check

    async def run(self):
        self._running = True
        logger.info("CrossExchangeDetector started")

        while self._running:
            scan_start = datetime.utcnow()

            for pair in self.books.pairs:
                view = self.books.pairs[pair]

                if not view.is_tradeable:
                    continue

                spread_info = view.get_cross_exchange_spread()
                if spread_info is None or spread_info['gross_spread_bps'] <= 0:
                    continue

                # Calculate costs
                cost_est = self.costs.estimate_arbitrage_cost(
                    symbol=pair,
                    buy_exchange=spread_info['buy_exchange'],
                    sell_exchange=spread_info['sell_exchange'],
                    buy_price=spread_info['buy_price'],
                    sell_price=spread_info['sell_price'],
                )

                net_spread = spread_info['gross_spread_bps'] - cost_est.total_cost_bps

                if net_spread < self.MIN_NET_SPREAD_BPS:
                    continue

                # Size the trade
                max_qty = min(spread_info['buy_depth'], spread_info['sell_depth'])
                mid_price = (spread_info['buy_price'] + spread_info['sell_price']) / 2

                if mid_price <= 0:
                    continue

                max_qty_by_usd = self.MAX_TRADE_USD / mid_price
                recommended_qty = min(max_qty, max_qty_by_usd)

                notional = recommended_qty * mid_price
                if notional < self.MIN_TRADE_USD:
                    continue

                expected_profit = notional * (net_spread / 10000)

                signal = ArbitrageSignal(
                    signal_id=f"arb_{pair.replace('/', '')}_{self._scan_count}_{uuid.uuid4().hex[:8]}",
                    signal_type="cross_exchange",
                    timestamp=datetime.utcnow(),
                    symbol=pair,
                    buy_exchange=spread_info['buy_exchange'],
                    sell_exchange=spread_info['sell_exchange'],
                    buy_price=spread_info['buy_price'],
                    sell_price=spread_info['sell_price'],
                    gross_spread_bps=spread_info['gross_spread_bps'],
                    total_cost_bps=cost_est.total_cost_bps,
                    net_spread_bps=net_spread,
                    max_quantity=max_qty,
                    recommended_quantity=recommended_qty,
                    expected_profit_usd=expected_profit,
                    buy_fee_bps=cost_est.buy_fee_bps,
                    sell_fee_bps=cost_est.sell_fee_bps,
                    buy_slippage_bps=cost_est.buy_slippage_bps,
                    sell_slippage_bps=cost_est.sell_slippage_bps,
                    expires_at=datetime.utcnow() + timedelta(seconds=self.SIGNAL_TTL_SECONDS),
                    confidence=self._calculate_confidence(view, spread_info),
                )

                self._signals_generated += 1

                # Risk check
                if self.risk.approve_arbitrage(signal):
                    try:
                        self.signal_queue.put_nowait(signal)
                        self._signals_approved += 1
                        logger.info(
                            f"ARB SIGNAL: {pair} {spread_info['direction']} | "
                            f"gross={float(spread_info['gross_spread_bps']):.1f}bps "
                            f"net={float(net_spread):.1f}bps "
                            f"profit=${float(expected_profit):.2f} "
                            f"qty={float(recommended_qty):.6f}"
                        )
                    except asyncio.QueueFull:
                        pass  # Drop if consumer is slow

                self._last_spreads[pair] = spread_info['gross_spread_bps']

            self._scan_count += 1

            elapsed = (datetime.utcnow() - scan_start).total_seconds()
            sleep_time = max(0.05, 0.1 - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self):
        self._running = False

    def _calculate_confidence(self, view, spread_info) -> Decimal:
        confidence = Decimal('0.5')

        # Freshness bonus
        age_mexc = (datetime.utcnow() - view.mexc_last_update).total_seconds()
        age_binance = (datetime.utcnow() - view.binance_last_update).total_seconds()
        max_age = max(age_mexc, age_binance)
        if max_age < 0.5:
            confidence += Decimal('0.2')
        elif max_age < 1.0:
            confidence += Decimal('0.1')

        # Depth bonus
        min_depth = min(spread_info['buy_depth'], spread_info['sell_depth'])
        if min_depth > Decimal('1.0'):
            confidence += Decimal('0.2')
        elif min_depth > Decimal('0.1'):
            confidence += Decimal('0.1')

        # Spread stability bonus (same direction as last scan)
        prev_spread = self._last_spreads.get(view.symbol)
        if prev_spread and prev_spread > 0:
            confidence += Decimal('0.05')

        return min(confidence, Decimal('0.95'))

    def get_stats(self) -> dict:
        return {
            "scan_count": self._scan_count,
            "signals_generated": self._signals_generated,
            "signals_approved": self._signals_approved,
            "approval_rate": (
                self._signals_approved / self._signals_generated
                if self._signals_generated > 0 else 0
            ),
        }
