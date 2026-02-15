"""
Funding Rate Arbitrage — delta-neutral strategy capturing persistent
funding rate differentials between MEXC and Binance perpetual futures.

HOW IT WORKS:
- Short on exchange with higher funding rate (collect funding)
- Long on exchange with lower funding rate (pay less)
- Net result: collect the differential. Price-neutral.

MEXC 0% maker futures fee makes opening positions nearly free.
Typical returns: 5-15% APR with low risk.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from ..exchanges.base import OrderSide

logger = logging.getLogger("arb.funding")

# Futures pairs to monitor for funding rate arb
FUTURES_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
]


@dataclass
class FundingOpportunity:
    symbol: str
    mexc_rate: Decimal
    binance_rate: Decimal
    differential: Decimal
    annual_apr: Decimal
    short_exchange: str
    long_exchange: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FundingPosition:
    position_id: str
    symbol: str
    short_exchange: str
    long_exchange: str
    quantity: Decimal
    entry_differential: Decimal
    entry_time: datetime
    cumulative_funding_collected: Decimal = Decimal('0')
    is_open: bool = True


class FundingRateArbitrage:

    MIN_DIFFERENTIAL_APR = Decimal('5.0')
    CLOSE_DIFFERENTIAL_APR = Decimal('2.0')
    MAX_LEVERAGE = Decimal('3.0')
    CHECK_INTERVAL_SECONDS = 300
    MAX_POSITION_USD = Decimal('2000')

    def __init__(self, mexc_client, binance_client, risk_engine):
        self.clients = {"mexc": mexc_client, "binance": binance_client}
        self.mexc = mexc_client
        self.binance = binance_client
        self.risk = risk_engine
        self._positions: Dict[str, FundingPosition] = {}
        self._opportunities_found = 0
        self._running = False

    async def run(self):
        """Main monitoring loop."""
        self._running = True
        logger.info("FundingRateArbitrage started")

        while self._running:
            try:
                # Scan for new opportunities
                opportunities = await self.scan_opportunities()
                for opp in opportunities:
                    if opp.symbol not in self._positions:
                        await self._evaluate_and_open(opp)

                # Monitor existing positions
                await self._monitor_positions()

            except Exception as e:
                logger.error(f"Funding rate scan error: {e}")

            await asyncio.sleep(self.CHECK_INTERVAL_SECONDS)

    def stop(self):
        self._running = False

    async def scan_opportunities(self) -> List[FundingOpportunity]:
        opportunities = []

        for symbol in FUTURES_PAIRS:
            try:
                mexc_fr = await self.mexc.get_funding_rate(symbol)
                binance_fr = await self.binance.get_funding_rate(symbol)

                mexc_rate = mexc_fr.current_rate
                binance_rate = binance_fr.current_rate

                differential = abs(mexc_rate - binance_rate)
                # Annualize: funding settles 3x daily
                annual_apr = differential * 3 * 365 * 100

                if annual_apr >= self.MIN_DIFFERENTIAL_APR:
                    if mexc_rate > binance_rate:
                        short_exchange = "mexc"
                        long_exchange = "binance"
                    else:
                        short_exchange = "binance"
                        long_exchange = "mexc"

                    opp = FundingOpportunity(
                        symbol=symbol,
                        mexc_rate=mexc_rate,
                        binance_rate=binance_rate,
                        differential=differential,
                        annual_apr=annual_apr,
                        short_exchange=short_exchange,
                        long_exchange=long_exchange,
                    )
                    opportunities.append(opp)
                    self._opportunities_found += 1

                    logger.info(
                        f"FUNDING OPP: {symbol} | "
                        f"MEXC={float(mexc_rate)*100:.4f}% Binance={float(binance_rate)*100:.4f}% | "
                        f"Diff={float(annual_apr):.1f}% APR | "
                        f"Short {short_exchange}, Long {long_exchange}"
                    )
            except Exception as e:
                logger.debug(f"Funding rate fetch error for {symbol}: {e}")

        opportunities.sort(key=lambda x: x.annual_apr, reverse=True)
        return opportunities

    async def _evaluate_and_open(self, opp: FundingOpportunity):
        """Evaluate and potentially open a funding arb position."""
        # Check risk limits
        if len(self._positions) >= 5:  # Max 5 simultaneous funding positions
            return

        if not self.risk.approve_funding_arb(opp.symbol, self.MAX_POSITION_USD):
            return

        logger.info(
            f"FUNDING ARB OPEN (paper): {opp.symbol} | "
            f"Short {opp.short_exchange}, Long {opp.long_exchange} | "
            f"APR: {float(opp.annual_apr):.1f}%"
        )

        position = FundingPosition(
            position_id=f"fund_{opp.symbol.replace('/', '')}_{int(datetime.utcnow().timestamp())}",
            symbol=opp.symbol,
            short_exchange=opp.short_exchange,
            long_exchange=opp.long_exchange,
            quantity=self.MAX_POSITION_USD,  # USD notional
            entry_differential=opp.differential,
            entry_time=datetime.utcnow(),
        )
        self._positions[opp.symbol] = position

    async def _monitor_positions(self):
        """Check existing positions — close if differential collapses."""
        to_close = []

        for symbol, position in self._positions.items():
            if not position.is_open:
                continue

            try:
                mexc_fr = await self.mexc.get_funding_rate(symbol)
                binance_fr = await self.binance.get_funding_rate(symbol)

                current_diff = abs(mexc_fr.current_rate - binance_fr.current_rate)
                current_apr = current_diff * 3 * 365 * 100

                # Accumulate funding income (simplified)
                funding_per_period = position.quantity * current_diff
                position.cumulative_funding_collected += funding_per_period

                if current_apr < self.CLOSE_DIFFERENTIAL_APR:
                    logger.info(
                        f"FUNDING ARB CLOSE: {symbol} | "
                        f"APR dropped to {float(current_apr):.1f}% | "
                        f"Collected: ${float(position.cumulative_funding_collected):.2f}"
                    )
                    to_close.append(symbol)

            except Exception as e:
                logger.debug(f"Position monitor error for {symbol}: {e}")

        for symbol in to_close:
            self._positions[symbol].is_open = False

    def get_stats(self) -> dict:
        open_positions = {s: p for s, p in self._positions.items() if p.is_open}
        total_collected = sum(p.cumulative_funding_collected for p in self._positions.values())
        return {
            "opportunities_found": self._opportunities_found,
            "open_positions": len(open_positions),
            "closed_positions": len(self._positions) - len(open_positions),
            "total_funding_collected_usd": float(total_collected),
            "positions": {
                s: {
                    "short_on": p.short_exchange,
                    "long_on": p.long_exchange,
                    "notional_usd": float(p.quantity),
                    "collected_usd": float(p.cumulative_funding_collected),
                    "age_hours": (datetime.utcnow() - p.entry_time).total_seconds() / 3600,
                }
                for s, p in open_positions.items()
            }
        }
