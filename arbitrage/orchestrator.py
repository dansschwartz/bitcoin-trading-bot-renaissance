"""
Arbitrage Orchestrator — coordinates all three strategies and modules.

This is the main entry point for the arbitrage system.
It initializes all components, starts the data feeds, runs the
detectors, routes signals to the execution engine, and logs everything.

Usage:
    python -m arbitrage.orchestrator [--paper] [--config path/to/config.yaml]
"""
import asyncio
import logging
import os
import signal as sig
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import yaml

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbitrage.exchanges.mexc_client import MEXCClient
from arbitrage.exchanges.binance_client import BinanceClient
from arbitrage.exchanges.bybit_client import BybitClient
from arbitrage.orderbook.unified_book import UnifiedBookManager
from arbitrage.costs.model import ArbitrageCostModel
from arbitrage.detector.cross_exchange import CrossExchangeDetector
from arbitrage.execution.engine import ArbitrageExecutor
from arbitrage.funding.funding_rate_arb import FundingRateArbitrage
from arbitrage.triangular.triangle_arb import TriangularArbitrage
from arbitrage.risk.arb_risk import ArbitrageRiskEngine
from arbitrage.inventory.manager import InventoryManager
from arbitrage.tracking.performance import PerformanceTracker
from arbitrage.exchanges.base import Trade
from arbitrage.safety.contract_verifier import ContractVerifier
from arbitrage.detector.pair_discovery import PairDiscoveryEngine

logger = logging.getLogger("arb.orchestrator")


class ArbitrageOrchestrator:
    """
    Main coordinator for all arbitrage strategies.
    Manages lifecycle, data flow, and monitoring.
    """

    def __init__(self, config_path: str = "arbitrage/config/arbitrage.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()

        paper = self.config.get('paper_trading', {}).get('enabled', True)
        logger.info(f"{'PAPER' if paper else 'LIVE'} TRADING MODE")

        # Exchange clients
        self.mexc = MEXCClient(
            api_key=os.getenv('MEXC_API_KEY', ''),
            api_secret=os.getenv('MEXC_API_SECRET', ''),
            paper_trading=paper,
        )
        self.binance = BinanceClient(
            api_key=os.getenv('BINANCE_API_KEY', ''),
            api_secret=os.getenv('BINANCE_API_SECRET', ''),
            paper_trading=paper,
        )

        # Bybit exchange client (secondary, for triangular arb)
        bybit_cfg = self.config.get('triangular_bybit', {})
        self.bybit = BybitClient(
            api_key=os.getenv('BYBIT_API_KEY', ''),
            api_secret=os.getenv('BYBIT_API_SECRET', ''),
            paper_trading=paper,
        ) if bybit_cfg.get('enabled', False) else None

        # BarAggregator for trade/book data
        self.bar_aggregator = self._init_bar_aggregator()

        # Core modules — combine phase_1 (large-cap) + phase_2 (mid-cap) pairs
        pairs_cfg = self.config.get('pairs', {})
        pairs = pairs_cfg.get('phase_1', []) + pairs_cfg.get('phase_2', [])
        self.book_manager = UnifiedBookManager(
            self.mexc, self.binance, pairs=pairs,
            bar_aggregator=self.bar_aggregator,
        )
        self.cost_model = ArbitrageCostModel()
        self.risk_engine = ArbitrageRiskEngine(self.config.get('risk', {}))

        # Signal queue: detector → executor
        self.signal_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Contract verification safety layer
        self.contract_verifier = ContractVerifier(
            self.mexc, self.binance, config=self.config,
        )

        # Dynamic pair discovery for cross-exchange arb
        self.pair_discovery = PairDiscoveryEngine(
            mexc=self.mexc,
            binance=self.binance,
            book_manager=self.book_manager,
            contract_verifier=self.contract_verifier,
            config=self.config,
        )

        # Strategies
        self.cross_exchange_detector = CrossExchangeDetector(
            self.book_manager, self.cost_model, self.risk_engine, self.signal_queue,
            config=self.config, contract_verifier=self.contract_verifier,
        )
        self.executor = ArbitrageExecutor(
            self.mexc, self.binance, self.cost_model, self.risk_engine,
        )
        # Support modules (tracker must be created before funding_arb and triangular_arb)
        self.inventory = InventoryManager(self.mexc, self.binance)
        self.tracker = PerformanceTracker()

        self.funding_arb = FundingRateArbitrage(
            self.mexc, self.binance, self.risk_engine,
            config=self.config, tracker=self.tracker,
        )

        self.triangular_arb = TriangularArbitrage(
            self.mexc, self.cost_model, self.risk_engine, self.signal_queue,
            config=self.config, tracker=self.tracker,
        )

        # Bybit triangular arb (separate instance, separate config)
        self.triangular_arb_bybit = None
        if self.bybit and bybit_cfg.get('enabled', False):
            bybit_config = dict(self.config)
            bybit_config['triangular'] = bybit_cfg
            self.triangular_arb_bybit = TriangularArbitrage(
                self.bybit, self.cost_model, self.risk_engine, self.signal_queue,
                config=bybit_config, tracker=self.tracker,
            )
            self.triangular_arb_bybit._exchange_name = "bybit"

        self._running = False
        self._start_time: Optional[datetime] = None

    def _load_config(self, path: str) -> dict:
        config_file = Path(path)
        if not config_file.exists():
            # Look relative to this file
            config_file = Path(__file__).parent / "config" / "arbitrage.yaml"
        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f)
        logger.warning(f"Config not found at {path}, using defaults")
        return {}

    def _setup_logging(self):
        log_cfg = self.config.get('logging', {})
        log_level = log_cfg.get('level', 'INFO')
        log_file = log_cfg.get('file', 'logs/arbitrage.log')

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure arbitrage loggers
        arb_logger = logging.getLogger("arb")
        arb_logger.setLevel(getattr(logging, log_level, logging.INFO))

        if not arb_logger.handlers:
            # File handler
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            ))
            arb_logger.addHandler(fh)

            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(getattr(logging, log_level, logging.INFO))
            ch.setFormatter(logging.Formatter(
                '%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s',
                datefmt='%H:%M:%S',
            ))
            arb_logger.addHandler(ch)

    async def start(self):
        """Initialize all components and start the arbitrage system."""
        self._running = True
        self._start_time = datetime.utcnow()

        logger.info("=" * 60)
        logger.info("  ARBITRAGE ENGINE — Renaissance of One")
        logger.info("  Three Uncorrelated Revenue Streams")
        logger.info("=" * 60)

        # Connect exchanges
        logger.info("Connecting to exchanges...")
        connect_tasks = [self.mexc.connect(), self.binance.connect()]
        if self.bybit:
            connect_tasks.append(self.bybit.connect())
        await asyncio.gather(*connect_tasks)

        # Contract verification cache (try to populate, degrade gracefully)
        try:
            await self.contract_verifier.refresh_cache()
        except Exception as e:
            logger.warning(f"Contract verifier cache refresh failed (degrading to permissive): {e}")

        # Initial inventory check
        try:
            snapshot = await self.inventory.check_inventory()
            logger.info(f"Inventory check: {len(snapshot.imbalances)} currencies tracked")
        except Exception as e:
            logger.warning(f"Initial inventory check failed: {e}")

        # Start all async tasks
        cross_exchange_enabled = self.config.get('cross_exchange', {}).get('enabled', True)
        if not cross_exchange_enabled:
            logger.info("Cross-exchange arbitrage DISABLED by config")
        pair_discovery_enabled = (
            cross_exchange_enabled
            and self.config.get('pair_discovery', {}).get('enabled', False)
        )
        if pair_discovery_enabled:
            logger.info("Dynamic pair discovery ENABLED for cross-exchange arb")

        tasks = [
            asyncio.create_task(self._run_book_manager(), name="book_manager"),
            *(
                [asyncio.create_task(self._run_cross_exchange(), name="cross_exchange")]
                if cross_exchange_enabled else []
            ),
            *(
                [asyncio.create_task(self._run_pair_discovery(), name="pair_discovery")]
                if pair_discovery_enabled else []
            ),
            asyncio.create_task(self._run_execution_loop(), name="executor"),
            asyncio.create_task(self._run_funding_arb(), name="funding_arb"),
            asyncio.create_task(self._run_triangular_arb(), name="triangular_arb"),
            *(
                [asyncio.create_task(self._run_triangular_arb_bybit(), name="triangular_arb_bybit")]
                if self.triangular_arb_bybit else []
            ),
            asyncio.create_task(self._run_monitoring(), name="monitoring"),
            asyncio.create_task(self._run_inventory_checks(), name="inventory"),
            asyncio.create_task(self._subscribe_trade_feeds(), name="trade_feeds"),
            asyncio.create_task(self._run_fee_monitor(), name="fee_monitor"),
        ]

        logger.info(f"All {len(tasks)} subsystems launched")

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("Orchestrator shutting down...")
        finally:
            await self.stop()

    async def stop(self):
        self._running = False
        self.cross_exchange_detector.stop()
        self.pair_discovery.stop()
        self.funding_arb.stop()
        self.triangular_arb.stop()
        if self.triangular_arb_bybit:
            self.triangular_arb_bybit.stop()
        await self.book_manager.stop()
        disconnect_tasks = [self.mexc.disconnect(), self.binance.disconnect()]
        if self.bybit:
            disconnect_tasks.append(self.bybit.disconnect())
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        logger.info("Arbitrage engine stopped")
        self._log_final_summary()

    # --- Subsystem runners ---

    async def _run_book_manager(self):
        """Start order book feeds."""
        try:
            await self.book_manager.start()
        except Exception as e:
            logger.error(f"Book manager error: {e}")

    async def _run_cross_exchange(self):
        """Run cross-exchange arbitrage detector."""
        # Wait for books to populate
        await asyncio.sleep(5)
        try:
            await self.cross_exchange_detector.run()
        except Exception as e:
            logger.error(f"Cross-exchange detector error: {e}")

    async def _run_pair_discovery(self):
        """Run dynamic pair discovery for cross-exchange arb."""
        await asyncio.sleep(15)  # Wait for exchange connections + initial books
        try:
            await self.pair_discovery.run()
        except Exception as e:
            logger.error(f"Pair discovery error: {e}")

    async def _run_execution_loop(self):
        """Consume signals from queue and execute trades."""
        await asyncio.sleep(6)  # Wait for detector to start
        logger.info("Execution loop started")

        while self._running:
            try:
                signal = await asyncio.wait_for(
                    self.signal_queue.get(), timeout=1.0
                )

                # Check signal expiry
                if datetime.utcnow() > signal.expires_at:
                    continue

                # Execute
                result = await self.executor.execute_arbitrage(signal)

                # Track
                self.tracker.record_trade(result)

                # Update risk engine
                if result.status == "filled":
                    self.risk_engine.record_trade_result(result.actual_profit_usd)
                elif result.status and "one_sided" in result.status:
                    self.risk_engine.record_trade_result(Decimal('0'), one_sided=True)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                import traceback
                logger.error(f"Execution loop error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(1)

    async def _run_funding_arb(self):
        """Run funding rate arbitrage scanner."""
        await asyncio.sleep(10)  # Let order books stabilize
        try:
            await self.funding_arb.run()
        except Exception as e:
            logger.error(f"Funding arb error: {e}")

    async def _run_triangular_arb(self):
        """Run triangular arbitrage scanner (MEXC)."""
        await asyncio.sleep(8)
        try:
            await self.triangular_arb.run()
        except Exception as e:
            logger.error(f"Triangular arb error: {e}")

    async def _run_triangular_arb_bybit(self):
        """Run triangular arbitrage scanner (Bybit)."""
        await asyncio.sleep(12)  # Start after MEXC scanner
        if not self.triangular_arb_bybit:
            return
        try:
            await self.triangular_arb_bybit.run()
        except Exception as e:
            logger.error(f"Bybit triangular arb error: {e}")

    async def _run_monitoring(self):
        """Periodic status logging."""
        await asyncio.sleep(15)

        while self._running:
            try:
                self._log_status()
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
            await asyncio.sleep(60)  # Every 60 seconds

    async def _run_inventory_checks(self):
        """Periodic inventory checks."""
        await asyncio.sleep(30)
        interval = self.config.get('inventory', {}).get('check_interval_minutes', 15) * 60

        while self._running:
            try:
                snapshot = await self.inventory.check_inventory()
                rebalance_plan = self.inventory.generate_rebalance_plan(snapshot)
                if rebalance_plan:
                    logger.warning(f"Rebalance needed: {len(rebalance_plan)} currencies")
            except Exception as e:
                logger.debug(f"Inventory check error: {e}")
            await asyncio.sleep(interval)

    async def _run_fee_monitor(self):
        """Periodically verify MEXC maker fee is still 0%.

        The entire triangular arb strategy depends on 0% maker fees.
        If MEXC changes their fee policy, we must pause immediately.
        Checks every hour, logs every check, and pauses tri-arb on change.
        """
        await asyncio.sleep(60)  # Let other systems stabilize first
        check_interval = 3600  # 1 hour
        consecutive_failures = 0

        while self._running:
            try:
                fees = await self.mexc.get_trading_fees("BTC/USDT")
                maker_fee = fees.get("maker", Decimal('0'))

                if maker_fee == Decimal('0'):
                    logger.info(
                        f"FEE CHECK OK: MEXC maker fee = {float(maker_fee)*100:.3f}% "
                        f"(triangular arb edge intact)"
                    )
                    consecutive_failures = 0

                    # Re-enable tri-arb if it was paused by a previous alert
                    if self.triangular_arb.OBSERVATION_MODE and hasattr(self, '_fee_paused'):
                        self.triangular_arb.OBSERVATION_MODE = False
                        del self._fee_paused
                        logger.info("FEE CHECK: maker fee restored to 0% — triangular arb re-enabled")
                else:
                    fee_bps = float(maker_fee) * 10000
                    logger.critical(
                        f"FEE ALERT: MEXC maker fee changed to {float(maker_fee)*100:.3f}% "
                        f"({fee_bps:.1f} bps)! Triangular arb edge may be destroyed. "
                        f"Pausing triangular execution."
                    )
                    # Pause triangular arb by switching to observation mode
                    self.triangular_arb.OBSERVATION_MODE = True
                    self._fee_paused = True

            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures <= 3:
                    logger.warning(f"FEE CHECK FAILED (attempt {consecutive_failures}): {e}")
                else:
                    logger.debug(f"FEE CHECK FAILED (attempt {consecutive_failures}): {e}")

            await asyncio.sleep(check_interval)

    # --- BarAggregator + Trade Feeds ---

    def _init_bar_aggregator(self):
        """Create BarAggregator if the module is available."""
        try:
            from data_module.bar_aggregator import BarAggregator
            bar_agg = BarAggregator(config=self.config)
            logger.info("BarAggregator initialized for trade/book data")
            return bar_agg
        except Exception as e:
            logger.warning(f"BarAggregator not available (trade data won't aggregate): {e}")
            return None

    async def _subscribe_trade_feeds(self):
        """Subscribe to trade streams on both exchanges for all pairs."""
        await asyncio.sleep(3)  # Wait for WS connections to establish
        pairs_cfg = self.config.get('pairs', {})
        pairs = pairs_cfg.get('phase_1', []) + pairs_cfg.get('phase_2', [])

        for pair in pairs:
            try:
                await self.mexc.subscribe_trades(pair, self._on_trade)
                await self.binance.subscribe_trades(pair, self._on_trade)
            except Exception as e:
                logger.debug(f"Trade subscribe error for {pair}: {e}")

        logger.info(f"Trade feeds subscribed for {len(pairs)} pairs on both exchanges")

        # Keep task alive
        while self._running:
            await asyncio.sleep(60)

    async def _on_trade(self, trade: Trade):
        """Forward trade data to BarAggregator."""
        if self.bar_aggregator:
            try:
                self.bar_aggregator.on_trade(
                    pair=trade.symbol,
                    exchange=trade.exchange,
                    price=float(trade.price),
                    quantity=float(trade.quantity),
                    side=trade.side.value,
                    timestamp=trade.timestamp.timestamp(),
                )
            except Exception:
                pass

    # --- Reporting ---

    def _log_status(self):
        uptime = (datetime.utcnow() - self._start_time).total_seconds() / 60 if self._start_time else 0
        book_status = self.book_manager.get_status()
        detector_stats = self.cross_exchange_detector.get_stats()
        executor_stats = self.executor.get_stats()
        risk_status = self.risk_engine.get_status()
        funding_stats = self.funding_arb.get_stats()
        tri_stats = self.triangular_arb.get_stats()

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  ARBITRAGE STATUS — Uptime: {uptime:.0f}min")
        logger.info("=" * 60)
        logger.info(f"  Books: {book_status['tradeable_pairs']}/{book_status['total_pairs']} tradeable | "
                    f"Updates: {book_status['total_updates']}")
        # Diagnostic: log why pairs aren't tradeable (first 3 only)
        if book_status['tradeable_pairs'] == 0:
            for pair, pdata in list(book_status.get('pairs', {}).items())[:3]:
                view = self.book_manager.pairs.get(pair)
                if view:
                    m_ok = view.mexc_book is not None
                    b_ok = view.binance_book is not None
                    fresh = view.is_fresh if (m_ok and b_ok) else False
                    m_bid = view.mexc_book.best_bid is not None if m_ok else False
                    b_bid = view.binance_book.best_bid is not None if b_ok else False
                    logger.info(
                        f"    {pair}: mexc_book={m_ok} binance_book={b_ok} "
                        f"fresh={fresh} mexc_bid={m_bid} binance_bid={b_bid} "
                        f"updates=M:{pdata.get('mexc_updates',0)}/B:{pdata.get('binance_updates',0)}"
                    )
        logger.info(f"  Detector: {detector_stats['scan_count']} scans | "
                    f"{detector_stats['signals_generated']} signals | "
                    f"{detector_stats['signals_approved']} approved")
        logger.info(f"  Executor: {executor_stats['total_trades']} trades | "
                    f"{executor_stats['total_fills']} fills | "
                    f"Profit: ${executor_stats['total_profit_usd']:.2f} | "
                    f"Win rate: {executor_stats['win_rate']*100:.0f}%")
        logger.info(f"  Funding: {funding_stats['open_positions']} open positions | "
                    f"Collected: ${funding_stats['total_funding_collected_usd']:.2f}")
        logger.info(f"  Triangular: {tri_stats['scan_count']} scans | "
                    f"{tri_stats['opportunities_found']} opportunities")
        comp = tri_stats.get('competition', {})
        if comp.get('sample_count', 0) > 0:
            logger.info(f"  Competition: median={comp['median_edge_bps']:.1f}bps "
                        f"mean={comp['mean_edge_bps']:.1f}bps "
                        f"range=[{comp['min_edge_bps']:.1f}-{comp['max_edge_bps']:.1f}] "
                        f"samples={comp['sample_count']} "
                        f"{'ALERT' if comp.get('alert_active') else 'OK'}")
        logger.info(f"  Risk: {'HALTED' if risk_status['halted'] else 'OK'} | "
                    f"Daily PnL: ${risk_status['daily_pnl_usd']:.2f} | "
                    f"Exposure: ${risk_status['total_exposure_usd']:.2f}")
        try:
            disc_stats = self.pair_discovery.get_stats()
            if disc_stats.get('scan_count', 0) > 0:
                logger.info(
                    f"  Discovery: {disc_stats['total_overlapping']} overlapping | "
                    f"{disc_stats['above_threshold']} above threshold | "
                    f"{disc_stats['active_discovered_pairs']} active"
                )
        except Exception:
            pass
        logger.info("=" * 60)

    def get_full_status(self) -> dict:
        """Aggregate status from all subsystems for dashboard consumption."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0
        try:
            book_status = self.book_manager.get_status()
        except Exception:
            book_status = {}
        try:
            cross_stats = self.cross_exchange_detector.get_stats()
        except Exception:
            cross_stats = {}
        try:
            executor_stats = self.executor.get_stats()
        except Exception:
            executor_stats = {}
        try:
            funding_stats = self.funding_arb.get_stats()
        except Exception:
            funding_stats = {}
        try:
            tri_stats = self.triangular_arb.get_stats()
        except Exception:
            tri_stats = {}
        try:
            tri_bybit_stats = self.triangular_arb_bybit.get_stats() if self.triangular_arb_bybit else {}
        except Exception:
            tri_bybit_stats = {}
        try:
            risk_status = self.risk_engine.get_status()
        except Exception:
            risk_status = {}
        try:
            tracker_summary = self.tracker.get_summary()
        except Exception:
            tracker_summary = {}
        try:
            contract_stats = self.contract_verifier.get_stats()
        except Exception:
            contract_stats = {}
        try:
            discovery_stats = self.pair_discovery.get_stats()
        except Exception:
            discovery_stats = {}
        return {
            "running": self._running,
            "uptime_seconds": round(uptime, 1),
            "book_status": book_status,
            "cross_exchange": cross_stats,
            "executor": executor_stats,
            "funding": funding_stats,
            "triangular": tri_stats,
            "triangular_bybit": tri_bybit_stats,
            "risk": risk_status,
            "tracker_summary": tracker_summary,
            "contract_verification": contract_stats,
            "pair_discovery": discovery_stats,
        }

    def _log_final_summary(self):
        summary = self.tracker.get_summary()
        logger.info("")
        logger.info("=" * 60)
        logger.info("  FINAL SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Runtime: {summary['uptime_hours']:.1f} hours")
        logger.info(f"  Total Trades: {summary['total_trades']}")
        logger.info(f"  Total Fills: {summary['total_fills']}")
        logger.info(f"  Total Profit: ${summary['total_profit_usd']:.2f}")
        logger.info(f"  Win Rate: {summary['win_rate']*100:.0f}%")
        logger.info(f"  Avg Profit/Fill: ${summary['avg_profit_per_fill']:.4f}")
        for strategy, stats in summary['by_strategy'].items():
            if stats['trades'] > 0:
                logger.info(f"  {strategy}: {stats['trades']} trades, ${stats['profit_usd']:.2f} profit")
        logger.info("=" * 60)


async def main():
    """Entry point for standalone arbitrage engine."""
    import argparse
    parser = argparse.ArgumentParser(description="Renaissance Arbitrage Engine")
    parser.add_argument('--config', default='arbitrage/config/arbitrage.yaml', help='Config file path')
    parser.add_argument('--paper', action='store_true', default=True, help='Paper trading mode')
    parser.add_argument('--live', action='store_true', help='Live trading mode (overrides --paper)')
    args = parser.parse_args()

    orchestrator = ArbitrageOrchestrator(config_path=args.config)

    if args.live:
        orchestrator.config.setdefault('paper_trading', {})['enabled'] = False

    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    for s in (sig.SIGINT, sig.SIGTERM):
        loop.add_signal_handler(s, lambda: asyncio.create_task(orchestrator.stop()))

    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(main())
