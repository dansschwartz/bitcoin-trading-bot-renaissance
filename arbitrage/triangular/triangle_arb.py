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
from datetime import datetime, timedelta
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

    MIN_NET_PROFIT_BPS = Decimal('3.0')   # Raised from 0.5 — must clear round-trip costs
    SCAN_INTERVAL_SECONDS = 5.0           # REST polling interval
    SCAN_INTERVAL_WS = 0.5               # WebSocket mode: scan every 500ms
    MAX_TRADE_USD = Decimal('500')        # Configurable via YAML
    START_CURRENCIES = ["USDT", "BTC", "ETH"]
    MAX_SIGNALS_PER_CYCLE = 3             # Max signals pushed per 60s cycle
    OBSERVATION_MODE = False              # Execute trades (was True)

    # Pairs with low precision that cause structural rounding losses.
    # BTC/ETH-quote pairs have ~8 decimal places of price but only 2-4
    # decimals of quantity precision, so rounding eats the entire edge.
    BLOCKED_QUOTE_CURRENCIES = {"BTC", "ETH"}

    def __init__(self, mexc_client, cost_model, risk_engine, signal_queue: asyncio.Queue,
                 config: Optional[dict] = None, tracker=None):
        self.client = mexc_client
        self.costs = cost_model
        self.risk = risk_engine
        self.signal_queue = signal_queue

        # Dedicated N-leg executor (handles both 3-leg and 4-leg cycles)
        from ..execution.triangular_executor import TriangularExecutor
        self.tri_executor = TriangularExecutor(mexc_client)
        self.tracker = tracker
        self._exchange_name = "mexc"  # Default; overridable for multi-exchange

        # Override class defaults from config if provided
        if config:
            tri_cfg = config.get('triangular', {})
            if 'observation_mode' in tri_cfg:
                self.OBSERVATION_MODE = tri_cfg['observation_mode']
            if 'max_trade_usd' in tri_cfg:
                self.MAX_TRADE_USD = Decimal(str(tri_cfg['max_trade_usd']))
            if 'min_net_profit_bps' in tri_cfg:
                self.MIN_NET_PROFIT_BPS = Decimal(str(tri_cfg['min_net_profit_bps']))
            if 'max_signals_per_cycle' in tri_cfg:
                self.MAX_SIGNALS_PER_CYCLE = tri_cfg['max_signals_per_cycle']
            if 'start_currencies' in tri_cfg:
                self.START_CURRENCIES = tri_cfg['start_currencies']
            if 'scan_interval_ms_ws' in tri_cfg:
                self.SCAN_INTERVAL_WS = tri_cfg['scan_interval_ms_ws'] / 1000.0
            if 'scan_interval_ms_rest' in tri_cfg:
                self.SCAN_INTERVAL_SECONDS = tri_cfg['scan_interval_ms_rest'] / 1000.0
        self._running = False
        self._scan_count = 0
        self._opportunities_found = 0
        self._signals_submitted = 0
        self._signals_skipped_balance = 0
        self._signals_skipped_size = 0
        self._signals_skipped_observation = 0
        self._last_signal_time: Optional[datetime] = None
        self._signals_this_cycle = 0
        self._cycle_start: Optional[datetime] = None
        self._pair_graph: Dict[str, Dict[str, dict]] = defaultdict(dict)

        # Competition detector — rolling window of edge sizes
        self._edge_history: List[Tuple[datetime, float]] = []  # (timestamp, profit_bps)
        self._edge_window_minutes = 60  # Track last 60 min
        self._edge_alert_threshold_bps = 4.0  # Warn if median drops below this
        self._edge_alert_logged = False

    async def run(self):
        self._running = True
        logger.info("TriangularArbitrage scanner started")

        # Build initial pair graph — retry up to 3 times
        for attempt in range(3):
            if await self._build_pair_graph():
                break
            logger.warning(f"TriangularArbitrage: pair graph build failed (attempt {attempt + 1}/3), retrying in 30s")
            await asyncio.sleep(30)
        else:
            logger.warning("TriangularArbitrage: pair graph unavailable after 3 attempts, scanner disabled")
            while self._running:
                await asyncio.sleep(300)
            return

        # Start WebSocket all-ticker feed (non-blocking)
        try:
            if hasattr(self.client, 'subscribe_all_tickers'):
                await self.client.subscribe_all_tickers()
                logger.info("TriangularArbitrage: WebSocket ticker feed requested")
        except Exception as e:
            logger.warning(f"WebSocket ticker feed unavailable: {e}")

        while self._running:
            try:
                # Hybrid data source: prefer WebSocket, fallback to REST
                tickers, data_source, ticker_age_ms = await self._get_tickers()
                self._update_graph(tickers)

                # Scan for profitable cycles (3-leg + 4-leg)
                opportunities = self._find_profitable_cycles(tickers)

                # Competition detector: record edge sizes
                self._record_edges(opportunities)

                # Rate-limit: reset cycle counter every 60s
                now = datetime.utcnow()
                if self._cycle_start is None or (now - self._cycle_start).total_seconds() > 60:
                    self._cycle_start = now
                    self._signals_this_cycle = 0

                for opp in opportunities[:3]:  # Top 3 per scan
                    self._opportunities_found += 1
                    n_legs = len(opp.path)
                    prefix = "QUAD OPP" if n_legs == 4 else "TRIANGLE OPP"

                    # Log opportunity (always, but throttled)
                    if self._scan_count % 20 == 0 or n_legs == 4:
                        logger.info(
                            f"{prefix}: {' -> '.join(s[0] for s in opp.path)} -> {opp.start_currency} | "
                            f"Profit: {float(opp.profit_bps):.2f}bps | "
                            f"Rate: {float(opp.cycle_rate):.8f} | {n_legs} legs"
                            f"{' [OBSERVATION]' if self.OBSERVATION_MODE else ''}"
                        )

                    # Observation mode: log but don't execute
                    if self.OBSERVATION_MODE:
                        self._signals_skipped_observation += 1
                        continue

                    # Rate limit: max signals per cycle
                    if self._signals_this_cycle >= self.MAX_SIGNALS_PER_CYCLE:
                        logger.debug("Triangular arb rate limited, skipping")
                        continue

                    # Execute directly via 3-leg executor
                    try:
                        result = await self.tri_executor.execute(
                            path=opp.path,
                            start_currency=opp.start_currency,
                            trade_usd=self.MAX_TRADE_USD,
                        )
                        self._signals_submitted += 1
                        self._signals_this_cycle += 1

                        if result.status == "filled":
                            self.risk.record_trade_result(result.profit_usd)

                        # Persist to DB for dashboard visibility
                        if self.tracker:
                            try:
                                self.tracker.record_triangular_trade(result, opportunity=opp)
                            except Exception as track_err:
                                logger.debug(f"Triangular trade tracking error: {track_err}")
                    except Exception as e:
                        logger.error(f"Triangular execution error: {e}")

                self._scan_count += 1

                # Log data source periodically
                if self._scan_count % 100 == 1:
                    logger.info(
                        f"TRI SCAN: source={data_source}, tickers={len(tickers)}, "
                        f"age={ticker_age_ms:.0f}ms, opportunities={len(opportunities)}"
                    )

            except Exception as e:
                logger.error(f"Triangle scan error: {e}")

            # Dynamic interval: faster with WebSocket, slower with REST
            interval = self.SCAN_INTERVAL_WS if data_source == "websocket" else self.SCAN_INTERVAL_SECONDS
            await asyncio.sleep(interval)

    def stop(self):
        self._running = False

    async def _get_tickers(self) -> Tuple[Dict[str, dict], str, float]:
        """Hybrid data source: prefer WebSocket tickers, fallback to REST.

        Returns:
            (tickers_dict, data_source, age_ms)
        """
        # Try WebSocket tickers first
        if hasattr(self.client, 'get_ws_tickers'):
            ws_tickers = self.client.get_ws_tickers()
            if ws_tickers and len(ws_tickers) > 100:
                age_ms = self.client.get_ws_ticker_age_ms()
                return ws_tickers, "websocket", age_ms

        # Fallback to REST
        tickers = await self.client.get_all_tickers()
        return tickers, "rest", 0.0

    async def _build_pair_graph(self) -> bool:
        """Build adjacency graph of all trading pairs. Returns True on success."""
        try:
            tickers = await self.client.get_all_tickers()
            self._update_graph(tickers)
            logger.info(f"Pair graph built with {len(self._pair_graph)} currencies, "
                       f"{sum(len(v) for v in self._pair_graph.values())} edges")
            return True
        except Exception as e:
            logger.error(f"Failed to build pair graph: {e}")
            return False

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
        """Find all profitable 3-step and 4-step cycles starting from each start currency."""
        opportunities = []
        fee = self.costs.FEES.get("mexc", {}).get("spot", {}).get("maker", Decimal('0'))

        for start in self.START_CURRENCIES:
            if start not in self._pair_graph:
                continue

            # --- 3-leg triangles ---
            for a_currency, edge_1 in self._pair_graph[start].items():
                if a_currency == start:
                    continue

                if a_currency not in self._pair_graph:
                    continue

                for b_currency, edge_2 in self._pair_graph[a_currency].items():
                    if b_currency == start or b_currency == a_currency:
                        continue

                    if b_currency not in self._pair_graph:
                        continue
                    if start not in self._pair_graph[b_currency]:
                        continue

                    edge_3 = self._pair_graph[b_currency][start]

                    cycle_rate = edge_1['rate'] * edge_2['rate'] * edge_3['rate']

                    if cycle_rate <= 1:
                        continue

                    profit_bps = (cycle_rate - 1) * 10000
                    total_fee_bps = fee * 3 * 10000
                    net_profit_bps = profit_bps - total_fee_bps

                    if net_profit_bps > self.MIN_NET_PROFIT_BPS:
                        leg_symbols = [edge_1['symbol'], edge_2['symbol'], edge_3['symbol']]
                        has_blocked = any(
                            sym.split('/')[1] in self.BLOCKED_QUOTE_CURRENCIES
                            for sym in leg_symbols if '/' in sym
                        )
                        if has_blocked:
                            continue

                        opportunities.append(TrianglePath(
                            start_currency=start,
                            path=[
                                (edge_1['symbol'], edge_1['side'], a_currency),
                                (edge_2['symbol'], edge_2['side'], b_currency),
                                (edge_3['symbol'], edge_3['side'], start),
                            ],
                            cycle_rate=cycle_rate,
                            profit_bps=net_profit_bps,
                            exchange=self._exchange_name,
                        ))

            # --- 4-leg quadrangles ---
            quad_opps = self._find_quadrangles(start, fee)
            opportunities.extend(quad_opps)

        opportunities.sort(key=lambda x: x.profit_bps, reverse=True)
        return opportunities

    def _find_quadrangles(self, start: str, fee: Decimal) -> List[TrianglePath]:
        """Find 4-leg cycles: start -> A -> B -> C -> start.

        Performance: prune paths where cumulative rate < 0.998 after 2 legs,
        which eliminates ~95% of search space.
        """
        results = []
        total_fee_bps = fee * 4 * 10000  # 4 legs

        if start not in self._pair_graph:
            return results

        for a_currency, edge_1 in self._pair_graph[start].items():
            if a_currency == start:
                continue
            if a_currency not in self._pair_graph:
                continue

            rate_2 = edge_1['rate']

            for b_currency, edge_2 in self._pair_graph[a_currency].items():
                if b_currency in (start, a_currency):
                    continue
                if b_currency not in self._pair_graph:
                    continue

                cum_rate_2 = rate_2 * edge_2['rate']
                # Prune: if cumulative rate after 2 legs is too low, skip
                if cum_rate_2 < Decimal('0.998'):
                    continue

                for c_currency, edge_3 in self._pair_graph[b_currency].items():
                    if c_currency in (start, a_currency, b_currency):
                        continue
                    if c_currency not in self._pair_graph:
                        continue
                    if start not in self._pair_graph[c_currency]:
                        continue

                    cum_rate_3 = cum_rate_2 * edge_3['rate']
                    # Prune after 3 legs
                    if cum_rate_3 < Decimal('0.998'):
                        continue

                    edge_4 = self._pair_graph[c_currency][start]
                    cycle_rate = cum_rate_3 * edge_4['rate']

                    if cycle_rate <= 1:
                        continue

                    profit_bps = (cycle_rate - 1) * 10000
                    net_profit_bps = profit_bps - total_fee_bps

                    if net_profit_bps > self.MIN_NET_PROFIT_BPS:
                        leg_symbols = [
                            edge_1['symbol'], edge_2['symbol'],
                            edge_3['symbol'], edge_4['symbol'],
                        ]
                        has_blocked = any(
                            sym.split('/')[1] in self.BLOCKED_QUOTE_CURRENCIES
                            for sym in leg_symbols if '/' in sym
                        )
                        if has_blocked:
                            continue

                        results.append(TrianglePath(
                            start_currency=start,
                            path=[
                                (edge_1['symbol'], edge_1['side'], a_currency),
                                (edge_2['symbol'], edge_2['side'], b_currency),
                                (edge_3['symbol'], edge_3['side'], c_currency),
                                (edge_4['symbol'], edge_4['side'], start),
                            ],
                            cycle_rate=cycle_rate,
                            profit_bps=net_profit_bps,
                            exchange=self._exchange_name,
                        ))

        return results

    # --- Competition Detector ---

    def _record_edges(self, opportunities: List[TrianglePath]) -> None:
        """Record edge sizes from detected opportunities for trend analysis."""
        now = datetime.utcnow()
        for opp in opportunities:
            self._edge_history.append((now, float(opp.profit_bps)))

        # Prune entries older than the window
        cutoff = now - timedelta(minutes=self._edge_window_minutes)
        while self._edge_history and self._edge_history[0][0] < cutoff:
            self._edge_history.pop(0)

        # Check for competition warning
        competition = self._get_competition_stats()
        if competition["sample_count"] >= 20:
            median = competition["median_edge_bps"]
            if median < self._edge_alert_threshold_bps and not self._edge_alert_logged:
                logger.warning(
                    f"COMPETITION ALERT: median edge dropped to {median:.1f} bps "
                    f"(threshold: {self._edge_alert_threshold_bps} bps) over last "
                    f"{self._edge_window_minutes} min — {competition['sample_count']} samples. "
                    f"Consider reducing position size."
                )
                self._edge_alert_logged = True
            elif median >= self._edge_alert_threshold_bps:
                self._edge_alert_logged = False  # Reset alert

    def _get_competition_stats(self) -> dict:
        """Compute edge size statistics for competition detection."""
        if not self._edge_history:
            return {
                "sample_count": 0, "median_edge_bps": 0.0,
                "mean_edge_bps": 0.0, "min_edge_bps": 0.0,
                "max_edge_bps": 0.0, "window_minutes": self._edge_window_minutes,
                "alert_active": self._edge_alert_logged,
            }

        edges = [e[1] for e in self._edge_history]
        edges_sorted = sorted(edges)
        n = len(edges_sorted)
        median = edges_sorted[n // 2] if n % 2 else (edges_sorted[n // 2 - 1] + edges_sorted[n // 2]) / 2

        return {
            "sample_count": n,
            "median_edge_bps": round(median, 2),
            "mean_edge_bps": round(sum(edges) / n, 2),
            "min_edge_bps": round(min(edges), 2),
            "max_edge_bps": round(max(edges), 2),
            "window_minutes": self._edge_window_minutes,
            "alert_active": self._edge_alert_logged,
        }

    def get_stats(self) -> dict:
        exec_stats = self.tri_executor.get_stats() if self.tri_executor else {}
        # Data source info
        ws_available = False
        ws_tickers_count = 0
        ws_age_ms = float('inf')
        if hasattr(self.client, 'get_ws_tickers'):
            ws_data = self.client.get_ws_tickers()
            if ws_data:
                ws_available = True
                ws_tickers_count = len(ws_data)
                ws_age_ms = self.client.get_ws_ticker_age_ms()
        return {
            "scan_count": self._scan_count,
            "opportunities_found": self._opportunities_found,
            "signals_submitted": self._signals_submitted,
            "signals_skipped_observation": self._signals_skipped_observation,
            "signals_skipped_balance": self._signals_skipped_balance,
            "signals_skipped_size": self._signals_skipped_size,
            "observation_mode": self.OBSERVATION_MODE,
            "min_profit_bps": float(self.MIN_NET_PROFIT_BPS),
            "graph_currencies": len(self._pair_graph),
            "graph_edges": sum(len(v) for v in self._pair_graph.values()),
            "executor": exec_stats,
            "competition": self._get_competition_stats(),
            "data_source": "websocket" if ws_available else "rest",
            "ws_tickers": ws_tickers_count,
            "ws_age_ms": round(ws_age_ms, 0) if ws_age_ms < 1e9 else None,
            "exchange": self._exchange_name,
        }
