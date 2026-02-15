"""
Triangular Arbitrage — exploits pricing inconsistencies across three
trading pairs on a SINGLE exchange (MEXC preferred — zero maker fees).

EXAMPLE:
  Start: 1000 USDT
  Step 1: Buy BTC with USDT  (BTC/USDT)
  Step 2: Buy ETH with BTC   (ETH/BTC)
  Step 3: Sell ETH for USDT  (ETH/USDT)
  Result: If cycle rate > 1.0, we profit.

On MEXC: 0% maker fee on ALL three legs = pure edge capture.
Challenges: tiny edges, must execute all 3 legs near-simultaneously.
"""
import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("arb.triangular")


@dataclass
class TrianglePath:
    """A three-step currency cycle."""
    start_currency: str
    path: List[Tuple[str, str, str]]  # [(pair, side, intermediate_currency), ...]
    cycle_rate: Decimal               # Product of exchange rates; >1 = profit
    profit_bps: Decimal
    exchange: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TriangularArbitrage:

    MIN_NET_PROFIT_BPS = Decimal('0.5')
    SCAN_INTERVAL_SECONDS = 0.5
    MAX_TRADE_USD = Decimal('200')     # Smaller for triangular (thinner books)
    START_CURRENCIES = ["USDT", "BTC", "ETH"]

    def __init__(self, mexc_client, cost_model, risk_engine, signal_queue: asyncio.Queue):
        self.client = mexc_client
        self.costs = cost_model
        self.risk = risk_engine
        self.signal_queue = signal_queue
        self._running = False
        self._scan_count = 0
        self._opportunities_found = 0
        self._pair_graph: Dict[str, Dict[str, dict]] = defaultdict(dict)

    async def run(self):
        self._running = True
        logger.info("TriangularArbitrage scanner started")

        # Build initial pair graph
        await self._build_pair_graph()

        while self._running:
            try:
                # Refresh tickers
                tickers = await self.client.get_all_tickers()
                self._update_graph(tickers)

                # Scan for profitable triangles
                opportunities = self._find_profitable_cycles(tickers)

                for opp in opportunities[:3]:  # Top 3 per scan
                    self._opportunities_found += 1
                    # Only log every 20th scan to avoid spam
                    if self._scan_count % 20 == 0:
                        logger.info(
                            f"TRIANGLE OPP: {' -> '.join(s[0] for s in opp.path)} -> {opp.start_currency} | "
                            f"Profit: {float(opp.profit_bps):.2f}bps | "
                            f"Rate: {float(opp.cycle_rate):.8f}"
                        )

                    # Create signal for execution
                    from ..detector.cross_exchange import ArbitrageSignal
                    signal = ArbitrageSignal(
                        signal_id=f"tri_{opp.start_currency}_{self._scan_count}",
                        signal_type="triangular",
                        timestamp=datetime.utcnow(),
                        symbol=f"{opp.path[0][0]}",
                        buy_exchange="mexc",
                        sell_exchange="mexc",
                        buy_price=Decimal('0'),
                        sell_price=Decimal('0'),
                        gross_spread_bps=opp.profit_bps,
                        total_cost_bps=Decimal('0'),  # 0% maker on MEXC
                        net_spread_bps=opp.profit_bps,
                        max_quantity=Decimal('0'),
                        recommended_quantity=Decimal('0'),
                        expected_profit_usd=self.MAX_TRADE_USD * opp.profit_bps / 10000,
                        buy_fee_bps=Decimal('0'),
                        sell_fee_bps=Decimal('0'),
                        buy_slippage_bps=Decimal('0.5'),
                        sell_slippage_bps=Decimal('0.5'),
                        expires_at=datetime.utcnow(),
                        confidence=min(Decimal('0.8'), opp.profit_bps / 10),
                    )

                    if self.risk.approve_arbitrage(signal):
                        try:
                            self.signal_queue.put_nowait(signal)
                        except asyncio.QueueFull:
                            pass

                self._scan_count += 1

            except Exception as e:
                logger.error(f"Triangle scan error: {e}")

            await asyncio.sleep(self.SCAN_INTERVAL_SECONDS)

    def stop(self):
        self._running = False

    async def _build_pair_graph(self):
        """Build adjacency graph of all trading pairs."""
        try:
            tickers = await self.client.get_all_tickers()
            self._update_graph(tickers)
            logger.info(f"Pair graph built with {len(self._pair_graph)} currencies, "
                       f"{sum(len(v) for v in self._pair_graph.values())} edges")
        except Exception as e:
            logger.error(f"Failed to build pair graph: {e}")

    def _update_graph(self, tickers: Dict[str, dict]):
        """Update graph edges with latest prices."""
        self._pair_graph.clear()

        for symbol, ticker in tickers.items():
            if '/' not in symbol:
                continue

            bid = ticker.get('bid', Decimal('0'))
            ask = ticker.get('ask', Decimal('0'))
            last = ticker.get('last_price', Decimal('0'))

            if not bid or not ask or bid <= 0 or ask <= 0:
                # Use last price as fallback
                if last and last > 0:
                    bid = last
                    ask = last
                else:
                    continue

            parts = symbol.split('/')
            if len(parts) != 2:
                continue
            base, quote = parts

            # Edge: base -> quote (selling base, getting quote)
            self._pair_graph[base][quote] = {
                'symbol': symbol, 'rate': bid, 'side': 'sell',
            }
            # Edge: quote -> base (buying base with quote)
            if ask > 0:
                self._pair_graph[quote][base] = {
                    'symbol': symbol, 'rate': Decimal('1') / ask, 'side': 'buy',
                }

    def _find_profitable_cycles(self, tickers: Dict) -> List[TrianglePath]:
        """Find all profitable 3-step cycles starting from each start currency."""
        opportunities = []

        for start in self.START_CURRENCIES:
            if start not in self._pair_graph:
                continue

            # Step 1: start -> A
            for a_currency, edge_1 in self._pair_graph[start].items():
                if a_currency == start:
                    continue

                # Step 2: A -> B
                if a_currency not in self._pair_graph:
                    continue

                for b_currency, edge_2 in self._pair_graph[a_currency].items():
                    if b_currency == start or b_currency == a_currency:
                        continue

                    # Step 3: B -> start (must close the cycle)
                    if b_currency not in self._pair_graph:
                        continue
                    if start not in self._pair_graph[b_currency]:
                        continue

                    edge_3 = self._pair_graph[b_currency][start]

                    # Calculate cycle rate
                    cycle_rate = edge_1['rate'] * edge_2['rate'] * edge_3['rate']

                    if cycle_rate <= 1:
                        continue

                    profit_bps = (cycle_rate - 1) * 10000

                    # Subtract fees (3 legs × maker fee = 0 on MEXC)
                    fee = self.costs.FEES.get("mexc", {}).get("spot", {}).get("maker", Decimal('0'))
                    total_fee_bps = fee * 3 * 10000
                    net_profit_bps = profit_bps - total_fee_bps

                    if net_profit_bps > self.MIN_NET_PROFIT_BPS:
                        opportunities.append(TrianglePath(
                            start_currency=start,
                            path=[
                                (edge_1['symbol'], edge_1['side'], a_currency),
                                (edge_2['symbol'], edge_2['side'], b_currency),
                                (edge_3['symbol'], edge_3['side'], start),
                            ],
                            cycle_rate=cycle_rate,
                            profit_bps=net_profit_bps,
                            exchange="mexc",
                        ))

        opportunities.sort(key=lambda x: x.profit_bps, reverse=True)
        return opportunities

    def get_stats(self) -> dict:
        return {
            "scan_count": self._scan_count,
            "opportunities_found": self._opportunities_found,
            "graph_currencies": len(self._pair_graph),
            "graph_edges": sum(len(v) for v in self._pair_graph.values()),
        }
