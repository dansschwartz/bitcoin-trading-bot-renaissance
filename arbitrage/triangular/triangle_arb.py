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
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("arb.triangular")


class StalenessFilter:
    """If a path failed to fill, impose a short cooldown before retrying.

    Prevents the bot from hammering the same dead opportunity every cycle.
    """

    def __init__(self, cooldown_seconds: int = 15):
        self.cooldown = cooldown_seconds
        self._recent_failures: Dict[str, float] = {}  # path -> last_failure_timestamp

    def record_failure(self, path: str) -> None:
        self._recent_failures[path] = time.time()

    def is_on_cooldown(self, path: str) -> bool:
        last_fail = self._recent_failures.get(path)
        if last_fail is None:
            return False
        if time.time() - last_fail < self.cooldown:
            return True
        # Cooldown expired, clean up
        del self._recent_failures[path]
        return False

    def cleanup(self) -> None:
        """Remove expired cooldowns. Call periodically."""
        now = time.time()
        self._recent_failures = {
            p: t for p, t in self._recent_failures.items()
            if now - t < self.cooldown
        }

    @property
    def active_cooldowns(self) -> int:
        return sum(1 for t in self._recent_failures.values()
                   if time.time() - t < self.cooldown)


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

            # Dynamic sizing params
            self._min_trade_usd = tri_cfg.get('min_trade_usd', 50)
            self._depth_fraction = tri_cfg.get('max_depth_fraction',
                                               tri_cfg.get('depth_fraction', 0.15))
            self._dynamic_sizing = tri_cfg.get('dynamic_sizing_enabled', True)
            # New: raised ceiling (overrides MAX_TRADE_USD for executor calls)
            self._max_single_trade_usd = tri_cfg.get('max_single_trade_usd',
                                                      float(self.MAX_TRADE_USD))
            # Per-cycle and hourly capital budgets
            self._max_capital_per_cycle = tri_cfg.get('max_capital_per_cycle', 25000)
            self._hourly_capital_limit = tri_cfg.get('hourly_capital_limit', 500000)
            # Edge scaling thresholds
            edge_cfg = tri_cfg.get('edge_scaling', {})
            self._edge_thresholds = {
                'thin_bps': edge_cfg.get('thin_bps', 5.0),
                'moderate_bps': edge_cfg.get('moderate_bps', 10.0),
                'full_bps': edge_cfg.get('full_bps', 20.0),
            }

            # 4-leg (quadrangular) config
            quad_cfg = config.get('quadrangular', {})
            self._quad_enabled = quad_cfg.get('enabled', True)
            self._quad_max_signals = quad_cfg.get('max_signals_per_cycle', 1)
            self._quad_min_profit_bps = Decimal(str(quad_cfg.get('min_net_profit_bps', 6.0)))
        else:
            self._min_trade_usd = 50
            self._depth_fraction = 0.15
            self._dynamic_sizing = True
            self._max_single_trade_usd = 15000
            self._max_capital_per_cycle = 25000
            self._hourly_capital_limit = 500000
            self._edge_thresholds = {'thin_bps': 5.0, 'moderate_bps': 10.0, 'full_bps': 20.0}
            self._quad_enabled = True
            self._quad_max_signals = 1
            self._quad_min_profit_bps = Decimal('6.0')

        self._running = False
        self._scan_count = 0
        self._opportunities_found = 0
        self._signals_submitted = 0
        self._signals_skipped_balance = 0
        self._signals_skipped_size = 0
        self._signals_skipped_observation = 0
        self._signals_skipped_capital = 0
        self._last_signal_time: Optional[datetime] = None
        self._signals_this_cycle = 0
        self._cycle_start: Optional[datetime] = None
        self._pair_graph: Dict[str, Dict[str, dict]] = defaultdict(dict)

        # Per-cycle and hourly capital tracking
        self._cycle_capital_deployed: float = 0.0
        self._hourly_capital: float = 0.0
        self._hourly_capital_reset_time: float = time.time()

        # Competition detector — rolling window of edge sizes
        self._edge_history: List[Tuple[datetime, float]] = []  # (timestamp, profit_bps)
        self._edge_window_minutes = 60  # Track last 60 min
        self._edge_alert_threshold_bps = 4.0  # Warn if median drops below this
        self._edge_alert_logged = False

        # Pre-flight freshness check stats
        self._preflight_stats = {
            'passed': 0, 'price_moved': 0, 'profit_gone': 0, 'fetch_failed': 0,
        }
        self._preflight_log_counter = 0

        # Adaptive threshold configuration
        self._adaptive_enabled = False
        self._adaptive_config = {
            'check_interval_attempts': 500,
            'raise_if_fill_rate_below': 0.25,
            'lower_if_fill_rate_above': 0.60,
            'step_size_bps': 0.5,
            'floor_bps': 3.0,
            'ceiling_bps': 15.0,
        }
        self._attempts_since_threshold_check = 0
        if config:
            tri_cfg = config.get('triangular', {})
            self._adaptive_enabled = tri_cfg.get('adaptive_threshold_enabled', False)
            if 'adaptive_threshold' in tri_cfg:
                self._adaptive_config.update(tri_cfg['adaptive_threshold'])

        # Staleness filter — prevents hammering failed paths
        self._staleness_filter = StalenessFilter(cooldown_seconds=15)
        self._staleness_skips = 0

        # 4-leg diagnostic counters
        self._quad_signals_this_cycle = 0
        self._quad_rejection_reasons = {
            'executed': 0,
            'rate_limited': 0,
            'lost_to_3leg_priority': 0,
            'edge_below_threshold': 0,
            'skipped_observation': 0,
            'execution_error': 0,
        }
        self._quad_diag_cycle_count = 0

    async def run(self):
        self._running = True
        logger.info("TriangularArbitrage scanner started")

        # Build initial pair graph — retry with backoff until success
        attempt = 0
        while self._running:
            attempt += 1
            if await self._build_pair_graph():
                break
            backoff = min(30 * attempt, 300)  # 30s, 60s, 90s, ... capped at 5min
            logger.warning(f"TriangularArbitrage: pair graph build failed (attempt {attempt}), retrying in {backoff}s")
            await asyncio.sleep(backoff)
        if not self._running:
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
                    self._quad_signals_this_cycle = 0
                    self._cycle_capital_deployed = 0.0

                # Hourly capital breaker reset
                if time.time() - self._hourly_capital_reset_time >= 3600:
                    self._hourly_capital = 0.0
                    self._hourly_capital_reset_time = time.time()

                # Separate 3-leg and 4-leg opportunities
                three_leg_opps = [o for o in opportunities if len(o.path) == 3]
                four_leg_opps = [o for o in opportunities if len(o.path) == 4]

                # Rank by expected value (profit × fill_rate)
                three_leg_opps = self._rank_opportunities(three_leg_opps)
                four_leg_opps = self._rank_opportunities(four_leg_opps)

                # --- Execute 3-leg (up to MAX_SIGNALS_PER_CYCLE) ---
                for opp in three_leg_opps[:3]:
                    self._opportunities_found += 1

                    if self._scan_count % 20 == 0:
                        logger.info(
                            f"TRIANGLE OPP: {' -> '.join(s[0] for s in opp.path)} -> {opp.start_currency} | "
                            f"Profit: {float(opp.profit_bps):.2f}bps | "
                            f"Rate: {float(opp.cycle_rate):.8f} | 3 legs"
                            f"{' [OBSERVATION]' if self.OBSERVATION_MODE else ''}"
                        )

                    if self.OBSERVATION_MODE:
                        self._signals_skipped_observation += 1
                        continue

                    if self._signals_this_cycle >= self.MAX_SIGNALS_PER_CYCLE:
                        continue

                    # Staleness filter: skip if this path recently failed
                    path_key = self._opp_path_key(opp)
                    if self._staleness_filter.is_on_cooldown(path_key):
                        self._staleness_skips += 1
                        continue

                    # Pre-flight freshness check: verify edge still exists
                    if not await self._preflight_freshness_check(opp):
                        self._staleness_filter.record_failure(path_key)
                        continue

                    # Capital budget checks
                    if not self._check_capital_budget():
                        continue

                    try:
                        result = await self.tri_executor.execute(
                            path=opp.path,
                            start_currency=opp.start_currency,
                            trade_usd=Decimal(str(self._max_single_trade_usd)),
                            min_trade_usd=self._min_trade_usd,
                            depth_fraction=self._depth_fraction,
                            edge_bps=float(opp.profit_bps),
                            edge_thresholds=self._edge_thresholds,
                        )
                        self._signals_submitted += 1
                        self._signals_this_cycle += 1

                        if result.status == "filled":
                            self.risk.record_trade_result(result.profit_usd)
                            deployed = float(result.start_amount)
                            self._cycle_capital_deployed += deployed
                            self._hourly_capital += deployed
                        elif result.status not in ("skipped",):
                            self._staleness_filter.record_failure(path_key)

                        if self.tracker:
                            try:
                                self.tracker.record_triangular_trade(result, opportunity=opp)
                            except Exception as track_err:
                                logger.debug(f"Triangular trade tracking error: {track_err}")
                    except Exception as e:
                        self._staleness_filter.record_failure(path_key)
                        logger.error(f"Triangular execution error: {e}")

                # --- Execute 4-leg (dedicated slot, up to _quad_max_signals) ---
                for opp in four_leg_opps[:3]:
                    self._opportunities_found += 1

                    # Always log 4-leg opportunities
                    logger.info(
                        f"QUAD OPP: {' -> '.join(s[0] for s in opp.path)} -> {opp.start_currency} | "
                        f"Profit: {float(opp.profit_bps):.2f}bps | "
                        f"Rate: {float(opp.cycle_rate):.8f} | 4 legs"
                        f"{' [OBSERVATION]' if self.OBSERVATION_MODE else ''}"
                    )

                    if self.OBSERVATION_MODE:
                        self._quad_rejection_reasons['skipped_observation'] += 1
                        continue

                    if not self._quad_enabled:
                        continue

                    # 4-leg has its own min_profit_bps threshold
                    if opp.profit_bps < self._quad_min_profit_bps:
                        self._quad_rejection_reasons['edge_below_threshold'] += 1
                        continue

                    if self._quad_signals_this_cycle >= self._quad_max_signals:
                        self._quad_rejection_reasons['rate_limited'] += 1
                        continue

                    # Staleness filter for 4-leg
                    path_key = self._opp_path_key(opp)
                    if self._staleness_filter.is_on_cooldown(path_key):
                        self._staleness_skips += 1
                        continue

                    # Pre-flight freshness check for 4-leg
                    if not await self._preflight_freshness_check(opp):
                        self._staleness_filter.record_failure(path_key)
                        continue

                    # Capital budget checks
                    if not self._check_capital_budget():
                        self._quad_rejection_reasons['rate_limited'] += 1
                        continue

                    try:
                        result = await self.tri_executor.execute(
                            path=opp.path,
                            start_currency=opp.start_currency,
                            trade_usd=Decimal(str(self._max_single_trade_usd)),
                            min_trade_usd=self._min_trade_usd,
                            depth_fraction=self._depth_fraction,
                            edge_bps=float(opp.profit_bps),
                            edge_thresholds=self._edge_thresholds,
                        )
                        self._signals_submitted += 1
                        self._quad_signals_this_cycle += 1
                        self._quad_rejection_reasons['executed'] += 1

                        if result.status == "filled":
                            self.risk.record_trade_result(result.profit_usd)
                            deployed = float(result.start_amount)
                            self._cycle_capital_deployed += deployed
                            self._hourly_capital += deployed
                        elif result.status not in ("skipped",):
                            self._staleness_filter.record_failure(path_key)

                        if self.tracker:
                            try:
                                self.tracker.record_triangular_trade(result, opportunity=opp)
                            except Exception as track_err:
                                logger.debug(f"Triangular trade tracking error: {track_err}")
                    except Exception as e:
                        self._staleness_filter.record_failure(path_key)
                        self._quad_rejection_reasons['execution_error'] += 1
                        logger.error(f"Quad execution error: {e}")

                # Periodic staleness filter cleanup + preflight summary
                if self._scan_count % 50 == 0:
                    self._staleness_filter.cleanup()

                # Preflight summary every 100 scans
                self._preflight_log_counter += 1
                if self._preflight_log_counter >= 100:
                    pf = self._preflight_stats
                    total_checked = pf['passed'] + pf['price_moved'] + pf['profit_gone'] + pf['fetch_failed']
                    if total_checked > 0:
                        logger.info(
                            f"PREFLIGHT SUMMARY: {total_checked} checked | "
                            f"{pf['passed']} passed ({pf['passed']/total_checked:.0%}) | "
                            f"{pf['price_moved']} price_moved | {pf['profit_gone']} profit_gone | "
                            f"{pf['fetch_failed']} fetch_failed | "
                            f"staleness_skips={self._staleness_skips} "
                            f"cooldowns_active={self._staleness_filter.active_cooldowns}"
                        )
                    self._preflight_log_counter = 0

                # Periodic quad diagnostics (every 100 scans)
                self._quad_diag_cycle_count += 1
                if self._quad_diag_cycle_count >= 100:
                    qr = self._quad_rejection_reasons
                    logger.info(
                        f"QUAD DIAGNOSTICS (last 100 cycles): "
                        f"executed={qr['executed']}, rate_limited={qr['rate_limited']}, "
                        f"edge_below={qr['edge_below_threshold']}, "
                        f"observation={qr['skipped_observation']}, "
                        f"errors={qr['execution_error']}"
                    )
                    self._quad_diag_cycle_count = 0
                    self._quad_rejection_reasons = {k: 0 for k in qr}

                self._scan_count += 1

                # Adaptive threshold: adjust min_profit_bps based on fill data
                self._adapt_min_profit_threshold()

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

    def _adapt_min_profit_threshold(self) -> None:
        """Adaptive threshold: if marginal trades aren't filling, raise the bar.

        If they're filling easily, lower it to capture more volume.
        Only adjusts by step_size_bps at a time. Checked every N attempts.
        """
        if not self._adaptive_enabled or not self.tracker:
            return

        self._attempts_since_threshold_check += 1
        cfg = self._adaptive_config
        if self._attempts_since_threshold_check < cfg['check_interval_attempts']:
            return

        self._attempts_since_threshold_check = 0
        current = float(self.MIN_NET_PROFIT_BPS)

        # Query fill rate for trades near the current threshold
        # "Marginal" = trades with expected profit in [current, current + 2.0 bps]
        import sqlite3
        try:
            conn = sqlite3.connect(self.tracker.db_path)
            conn.row_factory = sqlite3.Row
            marginal_range_bps = 2.0
            # expected_profit_usd is in USD; we need bps. Use net_spread_bps instead.
            rows = conn.execute(
                "SELECT status FROM arb_trades "
                "WHERE strategy = 'triangular' "
                "AND net_spread_bps >= ? AND net_spread_bps < ? "
                "ORDER BY id DESC LIMIT 500",
                (current, current + marginal_range_bps),
            ).fetchall()
            conn.close()
        except Exception:
            return

        if len(rows) < 50:
            return  # Not enough data

        fill_rate = sum(1 for r in rows if r['status'] == 'filled') / len(rows)
        step = cfg['step_size_bps']

        if fill_rate < cfg['raise_if_fill_rate_below']:
            new_threshold = min(current + step, cfg['ceiling_bps'])
            if new_threshold != current:
                logger.info(
                    f"ADAPTIVE THRESHOLD: Raising min_profit from {current:.1f}bps to "
                    f"{new_threshold:.1f}bps (marginal fill rate {fill_rate:.1%} < "
                    f"{cfg['raise_if_fill_rate_below']:.0%})"
                )
                self.MIN_NET_PROFIT_BPS = Decimal(str(new_threshold))

        elif fill_rate > cfg['lower_if_fill_rate_above']:
            new_threshold = max(current - step, cfg['floor_bps'])
            if new_threshold != current:
                logger.info(
                    f"ADAPTIVE THRESHOLD: Lowering min_profit from {current:.1f}bps to "
                    f"{new_threshold:.1f}bps (marginal fill rate {fill_rate:.1%} > "
                    f"{cfg['lower_if_fill_rate_above']:.0%})"
                )
                self.MIN_NET_PROFIT_BPS = Decimal(str(new_threshold))
        else:
            logger.debug(
                f"ADAPTIVE THRESHOLD: Holding at {current:.1f}bps "
                f"(marginal fill rate {fill_rate:.1%})"
            )

    def _rank_opportunities(self, opportunities: List[TrianglePath]) -> List[TrianglePath]:
        """Sort opportunities by expected value, using path performance data.

        Ranking formula:
            score = profit_bps × path_fill_rate

        - profit_bps: detected profit for this specific opportunity
        - path_fill_rate: historical fill rate (default 0.5 for unknown paths)

        Deprioritized paths (fill_rate < 30% after 50+ attempts) are moved to the back.
        """
        if not self.tracker:
            return opportunities  # No tracker = can't rank

        scored = []
        for opp in opportunities:
            path_key = self._opp_path_key(opp)
            perf = self.tracker.get_path_performance(path_key)

            fill_rate = 0.5  # Default for unknown paths
            is_deprioritized = False
            if perf and perf.get('total_attempts', 0) >= 20:
                fill_rate = perf.get('fill_rate', 0.5)
                is_deprioritized = bool(perf.get('is_deprioritized', 0))

            score = float(opp.profit_bps) * fill_rate
            scored.append((opp, score, is_deprioritized))

        # Sort: non-deprioritized first (by score desc), then deprioritized (by score desc)
        scored.sort(key=lambda x: (not x[2], x[1]), reverse=True)
        return [s[0] for s in scored]

    async def _preflight_freshness_check(self, opp: TrianglePath) -> bool:
        """Re-fetch the bottleneck leg's price and verify the edge still exists.

        The bottleneck leg is the one with the smallest book depth (if depth
        data is available from the current tickers), otherwise defaults to
        leg index 1 (the middle altcoin cross — typically least liquid).

        Returns True if opportunity is still valid, False if stale.
        """
        legs = opp.path  # [(symbol, side, next_currency), ...]
        if not legs:
            return False

        # Pick the bottleneck leg — default to middle leg (index 1)
        bottleneck_idx = min(1, len(legs) - 1)
        bottleneck = legs[bottleneck_idx]
        bn_symbol, bn_side, _ = bottleneck

        # Re-fetch current price for bottleneck leg
        try:
            book = await self.client.get_order_book(bn_symbol, depth=5)
            if not book:
                self._preflight_stats['fetch_failed'] += 1
                return False
            fresh_price = book.best_ask if bn_side == 'buy' else book.best_bid
            if not fresh_price or fresh_price <= 0:
                self._preflight_stats['fetch_failed'] += 1
                return False
        except Exception:
            self._preflight_stats['fetch_failed'] += 1
            return False

        # Get the original price used in graph calculation
        # For 'buy': rate = 1/ask, so original ask = 1/rate
        # For 'sell': rate = bid
        edge_data = self._pair_graph.get(
            opp.path[bottleneck_idx - 1][2] if bottleneck_idx > 0 else opp.start_currency,
            {},
        )
        # Alternatively, use the graph edge rates to get original price
        original_rate = Decimal('0')
        for curr, edges in self._pair_graph.items():
            for dest, edge in edges.items():
                if edge['symbol'] == bn_symbol and edge['side'] == bn_side:
                    original_rate = edge['rate']
                    break
            if original_rate:
                break

        if not original_rate:
            # Can't find original — allow (permissive)
            self._preflight_stats['passed'] += 1
            return True

        # Compute the original price from rate
        if bn_side == 'buy':
            original_price = Decimal('1') / original_rate if original_rate > 0 else Decimal('0')
        else:
            original_price = original_rate

        if original_price <= 0:
            self._preflight_stats['passed'] += 1
            return True

        # Quick sanity: if price moved more than 0.5%, opportunity is stale
        price_change_pct = abs(float(fresh_price) - float(original_price)) / float(original_price)
        if price_change_pct > 0.005:
            self._preflight_stats['price_moved'] += 1
            logger.debug(
                f"PREFLIGHT STALE: {self._opp_path_key(opp)} — bottleneck {bn_symbol} "
                f"price moved {price_change_pct:.3%} ({float(original_price):.6f} → {float(fresh_price):.6f})"
            )
            return False

        # Recalculate cycle rate with fresh price substituted in
        fresh_rate = (Decimal('1') / fresh_price) if bn_side == 'buy' else fresh_price
        # Rebuild cycle rate: multiply all leg rates, replacing bottleneck
        new_cycle_rate = Decimal('1')
        for i, (sym, side, _) in enumerate(opp.path):
            if i == bottleneck_idx:
                new_cycle_rate *= fresh_rate
            else:
                # Use current graph rate for other legs
                for curr, edges in self._pair_graph.items():
                    for dest, edge in edges.items():
                        if edge['symbol'] == sym and edge['side'] == side:
                            new_cycle_rate *= edge['rate']
                            break
                    else:
                        continue
                    break

        fee = self.costs.FEES.get("mexc", {}).get("spot", {}).get("maker", Decimal('0'))
        total_fee_bps = fee * len(opp.path) * 10000
        new_profit_bps = (new_cycle_rate - 1) * 10000 - total_fee_bps

        if new_profit_bps < self.MIN_NET_PROFIT_BPS:
            self._preflight_stats['profit_gone'] += 1
            logger.debug(
                f"PREFLIGHT KILLED: {self._opp_path_key(opp)} — profit dropped from "
                f"{float(opp.profit_bps):.1f}bps to {float(new_profit_bps):.1f}bps"
            )
            return False

        self._preflight_stats['passed'] += 1
        return True

    @staticmethod
    def _opp_path_key(opp: TrianglePath) -> str:
        """Build a currency path key for tracking (e.g., 'USDT→BTC→ETH→USDT')."""
        currencies = [opp.start_currency]
        for _, _, next_curr in opp.path:
            currencies.append(next_curr)
        return "→".join(currencies)

    # --- Capital Budget ---

    def _check_capital_budget(self) -> bool:
        """Check per-cycle and hourly capital budgets. Returns False if over limit."""
        if self._cycle_capital_deployed + self._min_trade_usd > self._max_capital_per_cycle:
            self._signals_skipped_capital += 1
            if self._signals_skipped_capital % 50 == 1:
                logger.warning(
                    f"CAPITAL BUDGET: cycle limit reached "
                    f"(${self._cycle_capital_deployed:.0f}/${self._max_capital_per_cycle:.0f})"
                )
            return False
        if self._hourly_capital + self._min_trade_usd > self._hourly_capital_limit:
            self._signals_skipped_capital += 1
            logger.warning(
                f"CAPITAL BREAKER: hourly limit reached "
                f"(${self._hourly_capital:.0f}/${self._hourly_capital_limit:.0f})"
            )
            return False
        return True

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
            "signals_skipped_capital": self._signals_skipped_capital,
            "preflight": self._preflight_stats.copy(),
            "staleness_skips": self._staleness_skips,
            "staleness_active_cooldowns": self._staleness_filter.active_cooldowns,
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
            "adaptive_threshold": {
                "enabled": self._adaptive_enabled,
                "current_min_profit_bps": float(self.MIN_NET_PROFIT_BPS),
                "attempts_until_check": max(0, self._adaptive_config['check_interval_attempts'] - self._attempts_since_threshold_check),
            },
            "sizing": {
                "max_single_trade_usd": self._max_single_trade_usd,
                "depth_fraction": self._depth_fraction,
                "max_capital_per_cycle": self._max_capital_per_cycle,
                "hourly_capital_limit": self._hourly_capital_limit,
                "edge_thresholds": self._edge_thresholds,
                "cycle_capital_deployed": round(self._cycle_capital_deployed, 2),
                "hourly_capital_deployed": round(self._hourly_capital, 2),
                "trades_skipped_capital": self._signals_skipped_capital,
            },
        }
