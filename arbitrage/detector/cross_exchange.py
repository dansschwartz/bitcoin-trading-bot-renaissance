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
                 config: Optional[dict] = None, contract_verifier=None):
        self.books = book_manager
        self.costs = cost_model
        self.risk = risk_engine
        self.signal_queue = signal_queue
        self.contract_verifier = contract_verifier

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
        self._contract_skips = 0
        self._price_sanity_skips = 0
        self._last_spreads: dict = {}  # pair -> last gross spread for stability check

    async def run(self):
        self._running = True
        logger.info("CrossExchangeDetector started")
        # Diagnostic counters
        _diag_best_gross = 0.0
        _diag_best_net = -999.0
        _diag_best_pair = ""
        _diag_below_threshold = 0
        _diag_below_notional = 0

        while self._running:
            scan_start = datetime.utcnow()

            for pair in self.books.pairs:
                view = self.books.pairs[pair]

                if not view.is_tradeable:
                    continue

                # Layer 1: Contract/blocklist verification (cached, runs once per token)
                if self.contract_verifier and not self.contract_verifier.is_verified(pair):
                    self._contract_skips += 1
                    continue

                spread_info = view.get_cross_exchange_spread()
                if spread_info is None or spread_info['gross_spread_bps'] <= 0:
                    continue

                # Layer 2: Price sanity check (instant, no API needed)
                if self.contract_verifier:
                    buy_p = float(spread_info['buy_price'])
                    sell_p = float(spread_info['sell_price'])
                    if not self.contract_verifier.price_sanity_check(pair, buy_p, sell_p):
                        self._price_sanity_skips += 1
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

                # Track best spread seen for diagnostics
                gross = float(spread_info['gross_spread_bps'])
                net = float(net_spread)
                if gross > _diag_best_gross:
                    _diag_best_gross = gross
                    _diag_best_net = net
                    _diag_best_pair = pair

                if net_spread < self.MIN_NET_SPREAD_BPS:
                    _diag_below_threshold += 1
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
                    _diag_below_notional += 1
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

            # Periodic diagnostics every 1000 scans (~100 seconds)
            if self._scan_count % 1000 == 0:
                logger.info(
                    f"CROSS-X DIAG [{self._scan_count} scans]: "
                    f"signals={self._signals_generated} approved={self._signals_approved} "
                    f"below_threshold={_diag_below_threshold} below_notional={_diag_below_notional} "
                    f"contract_skip={self._contract_skips} price_sanity_skip={self._price_sanity_skips} | "
                    f"best_spread: {_diag_best_pair} gross={_diag_best_gross:.1f}bps "
                    f"net={_diag_best_net:.1f}bps (min={self.MIN_NET_SPREAD_BPS}bps) "
                    f"cost={_diag_best_gross - _diag_best_net:.1f}bps"
                )
                # Reset diagnostics for next window
                _diag_best_gross = 0.0
                _diag_best_net = -999.0
                _diag_best_pair = ""
                _diag_below_threshold = 0
                _diag_below_notional = 0

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
        stats = {
            "scan_count": self._scan_count,
            "signals_generated": self._signals_generated,
            "signals_approved": self._signals_approved,
            "approval_rate": (
                self._signals_approved / self._signals_generated
                if self._signals_generated > 0 else 0
            ),
            "contract_skips": self._contract_skips,
            "price_sanity_skips": self._price_sanity_skips,
        }
        if self.contract_verifier:
            stats["contract_verification"] = self.contract_verifier.get_stats()
        return stats
