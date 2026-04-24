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
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv
load_dotenv()
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
from arbitrage.exchanges.kucoin_client import KuCoinClient
from arbitrage.exchanges.binance_us_client import BinanceUSClient
from arbitrage.orderbook.unified_book import UnifiedBookManager
from arbitrage.costs.model import ArbitrageCostModel
from arbitrage.detector.cross_exchange import CrossExchangeDetector
from arbitrage.execution.engine import ArbitrageExecutor
from arbitrage.funding.funding_rate_arb import FundingRateArbitrage
from arbitrage.basis.basis_arb import BasisArbitrage
from arbitrage.listing.listing_monitor import ListingMonitor
from arbitrage.listing.listing_arb import ListingArbitrage
from arbitrage.pairs.pairs_arb import PairsArbitrage
from arbitrage.triangular.triangle_arb import TriangularArbitrage
from arbitrage.risk.arb_risk import ArbitrageRiskEngine
from arbitrage.inventory.manager import InventoryManager
from arbitrage.inventory.rebalancer import SpotRebalancer
from arbitrage.tracking.performance import PerformanceTracker
from arbitrage.exchanges.base import Trade
from arbitrage.market_maker.hedged_mm import HedgedMarketMaker
from arbitrage.safety.contract_verifier import ContractVerifier
from arbitrage.safety.volume_limiter import VolumeParticipationLimiter
from arbitrage.detector.pair_discovery import PairDiscoveryEngine
from arbitrage.analytics.temporal_analyzer import TemporalAnalyzer
from arbitrage.analytics.temporal_bias import TemporalBias
from arbitrage.analytics.capital_velocity import CapitalVelocityTracker
from arbitrage.analytics.edge_decay import EdgeDecayMonitor
from arbitrage.analytics.strategy_allocator import StrategyAllocator
from arbitrage.analytics.exhaust_capture import ExhaustCapture
from arbitrage.pairs.discovery import MexcPairDiscovery
from arbitrage.pairs.pair_manager import ExpandedPairManager
from arbitrage.capital_allocator import CapitalAllocator
from arbitrage.capital_guard import CapitalGuard
from arbitrage.feeds.unified_price_feed import BinanceUnifiedPriceFeed, HybridTickerProvider
from arbitrage.relay.server import OrderBookRelayServer
from arbitrage.relay.client import OrderBookRelayClient

logger = logging.getLogger("arb.orchestrator")


class ArbitrageOrchestrator:
    """
    Main coordinator for all arbitrage strategies.
    Manages lifecycle, data flow, and monitoring.
    """

    def __init__(self, config_path: str = None, config: dict = None):
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            self.config = self._load_config("arbitrage/config/arbitrage.yaml")
        self._setup_logging()

        # paper_trading flag is now correctly set BEFORE clients are constructed
        self._paper_trading = self.config.get('paper_trading', {}).get('enabled', True)
        paper = self._paper_trading
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

        # KuCoin exchange client (spoke, for cross-exchange arb)
        kucoin_enabled = self.config.get('exchanges', {}).get('kucoin_enabled', False)
        self.kucoin = KuCoinClient(
            api_key=os.getenv('KUCOIN_API_KEY', ''),
            api_secret=os.getenv('KUCOIN_API_SECRET', ''),
            passphrase=os.getenv('KUCOIN_PASSPHRASE', ''),
            paper_trading=paper,
        ) if kucoin_enabled else None
        if self.kucoin:
            logger.info("KuCoin spoke exchange ENABLED")

        # Binance US exchange client (spoke, for cross-exchange arb — 0% maker, 0.01% taker)
        binance_us_enabled = self.config.get('exchanges', {}).get('binance_us_enabled', False)
        self.binance_us = BinanceUSClient(
            api_key=os.getenv('BINANCE_US_API_KEY', ''),
            api_secret=os.getenv('BINANCE_US_API_SECRET', ''),
            paper_trading=paper,
        ) if binance_us_enabled else None
        if self.binance_us:
            logger.info("Binance US spoke exchange ENABLED (Tier 0: 0% maker, 0.01% taker)")

        # BarAggregator for trade/book data
        self.bar_aggregator = self._init_bar_aggregator()

        # Core modules — combine phase_1 (large-cap) + phase_2 (mid-cap) pairs
        pairs_cfg = self.config.get('pairs', {})
        pairs = pairs_cfg.get('phase_1', []) + pairs_cfg.get('phase_2', [])

        # Relay client replaces MEXC WS on the US side
        relay_client_cfg = self.config.get('relay_client', {})
        skip_mexc_ws = relay_client_cfg.get('enabled', False)

        self.book_manager = UnifiedBookManager(
            self.mexc, self.binance, pairs=pairs,
            bar_aggregator=self.bar_aggregator,
            kucoin=self.kucoin,
            binance_us=self.binance_us,
            skip_mexc_ws=skip_mexc_ws,
        )

        # Order book relay: Bangalore server / US client
        relay_server_cfg = self.config.get('relay_server', {})
        self.relay_server = None
        if relay_server_cfg.get('enabled', False):
            self.relay_server = OrderBookRelayServer(
                host=relay_server_cfg.get('host', '0.0.0.0'),
                port=relay_server_cfg.get('port', 8765),
                auth_token=os.getenv('RELAY_AUTH_TOKEN', ''),
            )

        self.relay_client = None
        if relay_client_cfg.get('enabled', False):
            self.relay_client = OrderBookRelayClient(
                relay_url=relay_client_cfg['url'],
                auth_token=os.getenv('RELAY_AUTH_TOKEN', ''),
                pairs=pairs,
                book_manager=self.book_manager,
                reconnect_max_backoff=relay_client_cfg.get('reconnect_max_backoff_seconds', 30.0),
            )
        self.cost_model = ArbitrageCostModel()
        self.risk_engine = ArbitrageRiskEngine(self.config.get('risk', {}))

        # Signal queue: detector → executor
        self.signal_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Contract verification safety layer
        self.contract_verifier = ContractVerifier(
            self.mexc, self.binance,
            kucoin_client=self.kucoin,
            binance_us_client=self.binance_us,
            config=self.config,
        )

        # Volume participation limiter for cross-exchange arb
        self.volume_limiter = VolumeParticipationLimiter(config=self.config)

        # Wire volume limiter into exchange clients for fill rate degradation
        self.mexc.volume_limiter = self.volume_limiter
        self.binance.volume_limiter = self.volume_limiter
        if self.kucoin:
            self.kucoin.volume_limiter = self.volume_limiter
        if self.binance_us:
            self.binance_us.volume_limiter = self.volume_limiter

        # Dynamic pair discovery for cross-exchange arb
        self.pair_discovery = PairDiscoveryEngine(
            mexc=self.mexc,
            binance=self.binance,
            book_manager=self.book_manager,
            contract_verifier=self.contract_verifier,
            config=self.config,
            volume_limiter=self.volume_limiter,
        )

        # Strategies
        self.cross_exchange_detector = CrossExchangeDetector(
            self.book_manager, self.cost_model, self.risk_engine, self.signal_queue,
            config=self.config, contract_verifier=self.contract_verifier,
        )
        self.executor = ArbitrageExecutor(
            self.mexc, self.binance, self.cost_model, self.risk_engine,
            config=self.config, book_manager=self.book_manager,
            kucoin_client=self.kucoin, binance_us_client=self.binance_us,
        )
        # Support modules (tracker must be created before funding_arb and triangular_arb)
        self.inventory = InventoryManager(self.mexc, self.binance)
        self.tracker = PerformanceTracker()

        # Spot rebalancer — keeps wallets funded for cross-exchange arb
        rebal_cfg = self.config.get('auto_rebalance', {})
        rebal_clients = {"mexc": self.mexc}
        if self.binance_us:
            rebal_clients["binance_us"] = self.binance_us
        # Build approved token list from configured pairs
        approved_tokens = set()
        for pair in pairs:
            base = pair.split('/')[0] if '/' in pair else pair
            approved_tokens.add(base)
        self.rebalancer = SpotRebalancer(
            clients=rebal_clients, config=rebal_cfg,
            approved_tokens=list(approved_tokens),
        )

        # Hedged Market Maker
        mm_cfg = self.config.get("hedged_market_maker", {})
        mm_enabled = mm_cfg.get("enabled", False)
        self.hedged_mm = None
        if mm_enabled:
            self.hedged_mm = HedgedMarketMaker(
                mexc_client=self.mexc,
                config=mm_cfg,
                db_path=self.tracker.db_path if self.tracker else "data/arbitrage.db",
            )
            logger.info("Hedged Market Maker initialized")

        # Temporal pattern analyzer — mines trade history for hour/dow bias
        db_path = self.config.get('db_path', 'data/arbitrage.db')
        self.temporal_analyzer = TemporalAnalyzer(db_path=db_path)
        self.temporal_bias = TemporalBias(self.temporal_analyzer)

        # Analytics modules (capital velocity, edge decay, strategy allocator, data exhaust)
        self.velocity_tracker = CapitalVelocityTracker(db_path=db_path)
        self.edge_decay_monitor = EdgeDecayMonitor(db_path=db_path)
        self.strategy_allocator = StrategyAllocator(db_path=db_path)
        self.exhaust_capture = ExhaustCapture(db_path=db_path)

        # Wire exhaust capture into detector
        self.cross_exchange_detector.exhaust_capture = self.exhaust_capture
        self.executor.exhaust_capture = self.exhaust_capture

        # MEXC pair expansion — discover and tier 50-100 pairs
        self.mexc_pair_discovery = MexcPairDiscovery(
            mexc_client=self.mexc,
            binance_client=self.binance,
            temporal_analyzer=self.temporal_analyzer,
            config=self.config,
        )
        self.expanded_pair_manager = ExpandedPairManager(
            discovery=self.mexc_pair_discovery,
            orchestrator=self,
        )

        self.funding_arb = FundingRateArbitrage(
            self.mexc, self.binance, self.risk_engine,
            config=self.config, tracker=self.tracker,
        )

        self.basis_arb = BasisArbitrage(
            self.mexc, config=self.config, tracker=self.tracker,
        )

        # Listing arbitrage (new token listing monitor + arb evaluator)
        self.listing_arb = ListingArbitrage(
            mexc_spot_client=self.mexc,
            observation_mode=True,
            config=self.config,
        )
        self.listing_monitor = ListingMonitor(
            mexc_client=self.mexc,
            on_new_listing=self.listing_arb.on_new_listing,
            config=self.config,
        )

        # Unified price feed: Binance WS primary, MEXC REST fallback
        self.price_feed = BinanceUnifiedPriceFeed()
        self.ticker_provider = HybridTickerProvider(self.price_feed, self.mexc)

        # Triangular arb uses MEXC REST tickers directly (not Binance WS).
        # Tri arb finds intra-exchange price inconsistencies — needs MEXC prices,
        # not Binance prices, since trades execute on MEXC.
        self.triangular_arb = TriangularArbitrage(
            self.mexc, self.cost_model, self.risk_engine, self.signal_queue,
            config=self.config, tracker=self.tracker,
            ticker_provider=None,
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

        # Statistical pairs arbitrage (cointegration-based)
        self.pairs_arb = PairsArbitrage(
            mexc_spot_client=self.mexc,
            observation_mode=True,
            config=self.config,
        )

        # Capital allocator — prevents modules from starving each other
        async def _balance_getter():
            """Fetch MEXC balances for capital allocator."""
            import hmac, hashlib
            api_key = getattr(self.mexc, '_api_key', '')
            api_secret = getattr(self.mexc, '_api_secret', '')
            if not api_key or not api_secret:
                return {"USDT": 0.0, "USDC": 0.0}
            try:
                import requests as _req
                ts = str(int(time.time() * 1000))
                params = {'timestamp': ts, 'recvWindow': '60000'}
                query = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
                _sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
                url = f"https://api.mexc.com/api/v3/account?{query}&signature={_sig}"
                resp = _req.get(url, headers={'X-MEXC-APIKEY': api_key, 'Content-Type': 'application/json'}, timeout=8)
                data = resp.json()
                result = {"USDT": 0.0, "USDC": 0.0}
                for bal in data.get('balances', []):
                    asset = bal.get('asset', '')
                    if asset in result:
                        result[asset] = float(bal.get('free', 0))
                return result
            except Exception as e:
                logger.warning(f"Capital allocator balance fetch failed: {e}")
                return {"USDT": 0.0, "USDC": 0.0}

        self.capital_allocator = CapitalAllocator(_balance_getter)

        # Capital guard — hard USDT reserve enforcement
        self.capital_guard = CapitalGuard(config=self.config)

        # Wire capital guard into all trade-executing subsystems
        self.executor.capital_guard = self.capital_guard
        self.rebalancer.capital_guard = self.capital_guard
        self.triangular_arb.capital_guard = self.capital_guard
        self.triangular_arb.tri_executor.capital_guard = self.capital_guard
        if self.triangular_arb_bybit:
            self.triangular_arb_bybit.capital_guard = self.capital_guard
            self.triangular_arb_bybit.tri_executor.capital_guard = self.capital_guard
        if self.hedged_mm:
            self.hedged_mm.capital_guard = self.capital_guard

        # Wire allocator into modules
        self.triangular_arb.capital_allocator = self.capital_allocator
        if self.hedged_mm:
            self.hedged_mm.capital_allocator = self.capital_allocator

        # Wire temporal bias into detectors
        self.cross_exchange_detector.temporal_bias = self.temporal_bias
        self.triangular_arb.temporal_bias = self.temporal_bias
        self.triangular_arb.velocity_tracker = self.velocity_tracker
        if self.triangular_arb_bybit:
            self.triangular_arb_bybit.temporal_bias = self.temporal_bias
            self.triangular_arb_bybit.velocity_tracker = self.velocity_tracker

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
            fh = RotatingFileHandler(
                log_path,
                maxBytes=50 * 1024 * 1024,  # 50MB per file
                backupCount=5,               # Keep 5 rotated files = 250MB max
            )
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
        if self.kucoin:
            connect_tasks.append(self.kucoin.connect())
        if self.binance_us:
            connect_tasks.append(self.binance_us.connect())
        results = await asyncio.gather(*connect_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                names = ["mexc", "binance"] + (["bybit"] if self.bybit else []) + (["kucoin"] if self.kucoin else []) + (["binance_us"] if self.binance_us else [])
                logger.error(f"Exchange {names[i]} connect failed: {result}")
        logger.info(f"Exchange connections complete ({len(connect_tasks)} exchanges)")

        # Start Binance unified price feed (non-blocking WS task)
        try:
            await self.price_feed.start()
            logger.info("Binance unified price feed started")
        except Exception as e:
            logger.warning(f"Binance price feed start failed (degrading to MEXC REST): {e}")

        # Start MEXC all-ticker WS feed (powers triangular arb scanner)
        try:
            await self.mexc.subscribe_all_tickers()
            logger.info("MEXC all-ticker WS feed started (tri arb scanner)")
        except Exception as e:
            logger.warning(f"MEXC all-ticker WS failed (tri arb will use REST fallback): {e}")

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

        # Initialize temporal analyzer (mine historical trade patterns)
        try:
            await self.temporal_analyzer.initialize()
            logger.info("Temporal analyzer initialized — bias weights ready")
        except Exception as e:
            logger.warning(f"Temporal analyzer init failed (continuing without bias): {e}")

        # Initialize expanded pair manager
        pair_expansion_enabled = self.config.get('pair_expansion', {}).get('enabled', False)
        if pair_expansion_enabled:
            try:
                locked_pairs = self.config.get('pair_expansion', {}).get('locked_pairs', [])
                await self.expanded_pair_manager.initialize(locked_pairs=locked_pairs)
                logger.info("Expanded pair manager initialized")
            except Exception as e:
                logger.warning(f"Expanded pair manager init failed: {e}")

        # Start relay server (Bangalore side) — must happen before book_manager.start()
        if self.relay_server:
            await self.relay_server.start()
            self.book_manager._mexc_relay_hook = self.relay_server.broadcast_book
            logger.info("Relay server started — broadcasting MEXC books to remote clients")

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
            asyncio.create_task(self._run_basis_arb(), name="basis_arb"),
            asyncio.create_task(self._run_listing_monitor(), name="listing_monitor"),
            asyncio.create_task(self._run_listing_arb(), name="listing_arb"),
            asyncio.create_task(self._run_pairs_arb(), name="pairs_arb"),
            asyncio.create_task(self._run_triangular_arb(), name="triangular_arb"),
            *(
                [asyncio.create_task(self._run_triangular_arb_bybit(), name="triangular_arb_bybit")]
                if self.triangular_arb_bybit else []
            ),
            asyncio.create_task(self._run_monitoring(), name="monitoring"),
            asyncio.create_task(self._run_inventory_checks(), name="inventory"),
            asyncio.create_task(self._run_rebalancer(), name="rebalancer"),
            asyncio.create_task(self._subscribe_trade_feeds(), name="trade_feeds"),
            asyncio.create_task(self._run_fee_monitor(), name="fee_monitor"),
            asyncio.create_task(self._run_temporal_refresh(), name="temporal_refresh"),
            *(
                [asyncio.create_task(self._run_pair_expansion(), name="pair_expansion")]
                if pair_expansion_enabled else []
            ),
            asyncio.create_task(self._run_velocity_cache_writer(), name="velocity_cache"),
            asyncio.create_task(self._run_edge_decay(), name="edge_decay"),
            asyncio.create_task(self._run_strategy_allocator(), name="strategy_allocator"),
            *(
                [asyncio.create_task(self._run_hedged_mm(), name="hedged_mm")]
                if self.hedged_mm else []
            ),
            *(
                [asyncio.create_task(self._run_relay_client(), name="relay_client")]
                if self.relay_client else []
            ),
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
        if self.relay_client:
            await self.relay_client.stop()
        if self.relay_server:
            await self.relay_server.stop()
        await self.price_feed.stop()
        self.cross_exchange_detector.stop()
        self.pair_discovery.stop()
        self.funding_arb.stop()
        self.basis_arb.stop()
        self.listing_monitor.stop()
        self.listing_arb.stop()
        self.pairs_arb.stop()
        self.triangular_arb.stop()
        if self.triangular_arb_bybit:
            self.triangular_arb_bybit.stop()
        await self.book_manager.stop()
        disconnect_tasks = [self.mexc.disconnect(), self.binance.disconnect()]
        if self.bybit:
            disconnect_tasks.append(self.bybit.disconnect())
        if self.kucoin:
            disconnect_tasks.append(self.kucoin.disconnect())
        if self.binance_us:
            disconnect_tasks.append(self.binance_us.disconnect())
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

                # Volume participation check (replaces flat concentration limits)
                if signal.signal_type == "cross_exchange":
                    trade_size = float(signal.recommended_quantity * signal.buy_price)
                    allowed, reason = self.volume_limiter.check(
                        signal.symbol, trade_size_usd=trade_size
                    )
                    if not allowed:
                        logger.debug(f"Volume limiter blocked {signal.symbol}: {reason}")
                        continue

                # Execute
                result = await self.executor.execute_arbitrage(signal)

                # Track inventory misses and successful cross-exchange trades
                # for dynamic seed promotion/demotion
                if result.status == "insufficient_balance":
                    self.rebalancer.record_inventory_miss(signal.symbol)
                if result.status == "filled" and signal.signal_type == "cross_exchange":
                    self.rebalancer.record_trade(signal.symbol)

                # Track
                self.tracker.record_trade(result)

                # Record capital velocity for cross-exchange trades
                if result.status == "filled" and hasattr(result, 'signal'):
                    trade_size = float(result.signal.recommended_quantity * result.signal.buy_price)
                    # Cross-exchange hold is ~instant (simultaneous legs)
                    hold_sec = getattr(result, 'hold_duration_seconds', 2.0)
                    self.velocity_tracker.record_trade(
                        strategy=result.signal.signal_type,
                        trade_size_usd=trade_size,
                        hold_seconds=hold_sec,
                        profit_usd=float(result.actual_profit_usd),
                    )

                # Record outcome in volume limiter
                if signal.signal_type == "cross_exchange":
                    trade_vol = float(signal.recommended_quantity * signal.buy_price)
                    self.volume_limiter.record_trade(
                        signal.symbol, trade_vol, result.status
                    )

                # Update risk engine with realistic profit (not phantom paper profit)
                if result.status == "filled":
                    realistic_pnl = (
                        result.realistic_costs.realistic_profit_usd
                        if result.realistic_costs
                        else result.actual_profit_usd
                    )
                    self.risk_engine.record_trade_result(realistic_pnl)
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

    async def _run_basis_arb(self):
        """Run basis (spot-futures convergence) arbitrage scanner."""
        await asyncio.sleep(15)  # Let exchange connections stabilize
        try:
            await self.basis_arb.run()
        except Exception as e:
            logger.error(f"Basis arb error: {e}")

    async def _run_listing_monitor(self):
        """Run listing monitor (new MEXC token detection)."""
        await asyncio.sleep(20)  # Let exchange connections fully stabilize
        try:
            await self.listing_monitor.run()
        except Exception as e:
            logger.error(f"Listing monitor error: {e}")

    async def _run_listing_arb(self):
        """Run listing arbitrage position monitor."""
        await asyncio.sleep(25)  # Start after listing monitor
        try:
            await self.listing_arb.run()
        except Exception as e:
            logger.error(f"Listing arb error: {e}")

    async def _run_pairs_arb(self):
        """Run statistical pairs arbitrage (cointegration-based)."""
        await asyncio.sleep(30)  # Let exchange connections and price feeds stabilize
        try:
            await self.pairs_arb.run()
        except Exception as e:
            logger.error(f"Pairs arb error: {e}")

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

    async def _run_temporal_refresh(self):
        """Periodically refresh temporal bias weights from trade history."""
        await asyncio.sleep(120)  # Let trades accumulate first

        while self._running:
            try:
                refreshed = await self.temporal_analyzer.maybe_refresh()
                if refreshed:
                    logger.info("Temporal analyzer refreshed bias weights")
            except Exception as e:
                logger.debug(f"Temporal refresh error: {e}")
            await asyncio.sleep(300)  # Check every 5 minutes (actual refresh is hourly)

    async def _run_pair_expansion(self):
        """Periodically update expanded pair sets (rescan + rotation)."""
        await asyncio.sleep(60)  # Let exchanges stabilize

        while self._running:
            try:
                await self.expanded_pair_manager.maybe_update()
                # Write cache file for dashboard (runs as separate process)
                self._write_pair_expansion_cache()
            except Exception as e:
                logger.debug(f"Pair expansion update error: {e}")
            await asyncio.sleep(600)  # Check every 10 minutes

    def _write_pair_expansion_cache(self):
        """Write pair expansion state to JSON cache for dashboard."""
        import json
        from pathlib import Path
        try:
            cache = {
                "manager": self.expanded_pair_manager.get_report(),
                "discovery": self.mexc_pair_discovery.get_report(),
            }
            cache_path = Path("data/pair_expansion_cache.json")
            with open(cache_path, "w") as f:
                json.dump(cache, f, default=str)
        except Exception as e:
            logger.debug(f"Failed to write pair expansion cache: {e}")

    async def _run_velocity_cache_writer(self):
        """Periodically write capital velocity cache for dashboard."""
        await asyncio.sleep(90)
        while self._running:
            try:
                import json
                report = self.velocity_tracker.get_velocity_report()
                cache_path = Path("data/capital_velocity_cache.json")
                with open(cache_path, "w") as f:
                    json.dump(report, f, default=str)
            except Exception as e:
                logger.debug(f"Velocity cache write error: {e}")
            await asyncio.sleep(60)

    async def _run_edge_decay(self):
        """Periodically compute edge decay analysis."""
        await asyncio.sleep(300)  # Let trades accumulate
        while self._running:
            try:
                import json
                report = self.edge_decay_monitor.compute_daily()
                cache_path = Path("data/edge_decay_cache.json")
                with open(cache_path, "w") as f:
                    json.dump(report, f, default=str)
                logger.info("Edge decay analysis updated")
            except Exception as e:
                logger.debug(f"Edge decay error: {e}")
            await asyncio.sleep(21600)  # Every 6 hours

    async def _run_strategy_allocator(self):
        """Periodically compute optimal strategy allocation."""
        await asyncio.sleep(600)  # Wait for velocity + decay data
        while self._running:
            try:
                import json
                velocity = self.velocity_tracker.get_velocity_report()
                decay = self.edge_decay_monitor.get_decay_report()
                report = self.strategy_allocator.rebalance(velocity, decay)
                cache_path = Path("data/strategy_allocation_cache.json")
                with open(cache_path, "w") as f:
                    json.dump(report, f, default=str)
                mode = "OBSERVATION" if report.get("observation_mode") else "LIVE"
                logger.info(f"Strategy allocator updated [{mode}]: {report.get('target_allocation', {})}")
            except Exception as e:
                logger.debug(f"Strategy allocator error: {e}")
            await asyncio.sleep(604800)  # Weekly

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

    async def _run_rebalancer(self):
        """Periodic spot rebalancing — keeps wallets funded for arb."""
        rebal_cfg = self.config.get('auto_rebalance', {})
        if not rebal_cfg.get('enabled', False):
            logger.info("Auto-rebalancer disabled")
            return

        # Wait for books + balances to stabilize before first rebalance
        await asyncio.sleep(60)
        interval = rebal_cfg.get('check_interval_minutes', 30) * 60

        while self._running:
            try:
                trades = await self.rebalancer.check_and_rebalance()
                if trades:
                    logger.info(
                        f"REBALANCER: executed {len(trades)} trades | "
                        f"total: ${sum(float(t.usdt_value) for t in trades):.2f}"
                    )
            except Exception as e:
                logger.error(f"Rebalancer error: {e}")
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
                if self.binance_us:
                    await self.binance_us.subscribe_trades(pair, self._on_trade)
            except Exception as e:
                logger.debug(f"Trade subscribe error for {pair}: {e}")

        n_exchanges = 2 + (1 if self.binance_us else 0)
        logger.info(f"Trade feeds subscribed for {len(pairs)} pairs on {n_exchanges} exchanges")

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
            except Exception as e:
                logger.warning(f"self.bar_aggregator.on_trade failed: {e}")

    # --- Reporting ---

    def _log_status(self):
        uptime = (datetime.utcnow() - self._start_time).total_seconds() / 60 if self._start_time else 0
        book_status = self.book_manager.get_status()
        detector_stats = self.cross_exchange_detector.get_stats()
        executor_stats = self.executor.get_stats()
        risk_status = self.risk_engine.get_status()
        funding_stats = self.funding_arb.get_stats()
        basis_stats = self.basis_arb.get_stats()
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
        paper_pnl = executor_stats.get('paper_profit_usd', executor_stats.get('total_profit_usd', 0))
        realistic_pnl = executor_stats.get('realistic_profit_usd', paper_pnl)
        edge_rate = executor_stats.get('edge_survival_rate', 0)
        logger.info(f"  Executor: {executor_stats['total_trades']} trades | "
                    f"{executor_stats['total_fills']} fills | "
                    f"Paper: ${paper_pnl:.2f} | "
                    f"Realistic: ${realistic_pnl:.2f} | "
                    f"Edge survival: {edge_rate*100:.0f}%")
        logger.info(f"  Basis: {basis_stats['scans_completed']} scans | "
                    f"{basis_stats['opportunities_found']} opportunities | "
                    f"mode={'OBS' if basis_stats['observation_mode'] else 'LIVE'}")
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
        except Exception as e:
            logger.warning(f"self.pair_discovery.get_stats failed: {e}")
        try:
            vol_stats = self.volume_limiter.get_stats()
            rej = vol_stats.get('rejections', {})
            total_rej = rej.get('total', 0)
            if total_rej > 0 or vol_stats.get('blocked_count', 0) > 0:
                logger.info(
                    f"  Volume limiter: {vol_stats['pairs_liquid']} liquid / "
                    f"{vol_stats['pairs_tracked']} tracked | "
                    f"{vol_stats['blocked_count']} blocked | "
                    f"Rejected: {total_rej} "
                    f"(vol={rej.get('volume',0)} part={rej.get('participation',0)} "
                    f"block={rej.get('blocked',0)})"
                )
        except Exception as e:
            logger.warning(f"self.volume_limiter.get_stats failed: {e}")
        # Dynamic seed status
        try:
            ds_status = self.rebalancer.get_dynamic_seeds_status()
            if ds_status['dynamic_seed_count'] > 0 or ds_status['pending_promotions']:
                seed_names = ', '.join(ds_status['seeds'].keys()) or 'none'
                pending = ', '.join(f"{b}({n})" for b, n in ds_status['pending_promotions'].items()) or 'none'
                logger.info(
                    f"  Dynamic seeds: {ds_status['dynamic_seed_count']} active [{seed_names}] | "
                    f"pending: {pending}"
                )
        except Exception as e:
            logger.warning(f"self.rebalancer.get_dynamic_seeds_status failed: {e}")
        # Relay status
        if self.relay_server:
            try:
                rs = self.relay_server.get_status()
                logger.info(
                    f"  Relay server: {rs['connected_clients']} clients | "
                    f"{rs['messages_sent']} msgs sent | seq={rs['sequence']}"
                )
            except Exception as e:
                logger.warning(f"self.relay_server.get_status failed: {e}")
        if self.relay_client:
            try:
                rc = self.relay_client.get_status()
                age_str = f"{rc['last_message_age_seconds']:.1f}s" if rc['last_message_age_seconds'] is not None else "never"
                logger.info(
                    f"  Relay client: {'CONNECTED' if rc['connected'] else 'DISCONNECTED'} | "
                    f"{rc['messages_received']} msgs | last={age_str} | "
                    f"gaps={rc['sequence_gaps']} | reconnects={rc['reconnect_count']}"
                )
            except Exception as e:
                logger.warning(f"self.relay_client.get_status failed: {e}")
        # Capital allocation status
        try:
            cap_summary = self.capital_allocator.get_summary()
            deployed = cap_summary.get('deployed', {})
            tri_dep = deployed.get('triangular', 0)
            mm_dep = deployed.get('market_maker', 0)
            logger.info(
                f"  Capital: tri=40% mm=50% rsv=10% | "
                f"deployed: tri=${tri_dep:.0f} mm=${mm_dep:.0f}"
            )
        except Exception as e:
            logger.warning(f"self.capital_allocator.get_summary failed: {e}")
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
            basis_stats = self.basis_arb.get_stats()
        except Exception:
            basis_stats = {}
        try:
            listing_monitor_stats = self.listing_monitor.get_stats()
        except Exception:
            listing_monitor_stats = {}
        try:
            listing_arb_stats = self.listing_arb.get_status()
        except Exception:
            listing_arb_stats = {}
        try:
            pairs_arb_stats = self.pairs_arb.get_status()
        except Exception:
            pairs_arb_stats = {}
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
        try:
            volume_limiter_stats = self.volume_limiter.get_stats()
        except Exception:
            volume_limiter_stats = {}
        try:
            temporal_stats = self.temporal_analyzer.get_report()
        except Exception:
            temporal_stats = {}
        try:
            pair_expansion_stats = self.expanded_pair_manager.get_report()
        except Exception:
            pair_expansion_stats = {}
        try:
            velocity_stats = self.velocity_tracker.get_velocity_report()
        except Exception:
            velocity_stats = {}
        try:
            edge_decay_stats = self.edge_decay_monitor.get_decay_report()
        except Exception:
            edge_decay_stats = {}
        try:
            allocator_stats = self.strategy_allocator.get_allocation_report()
        except Exception:
            allocator_stats = {}
        try:
            capital_alloc_stats = self.capital_allocator.get_summary()
        except Exception:
            capital_alloc_stats = {}
        try:
            capital_guard_stats = self.capital_guard.get_stats()
        except Exception:
            capital_guard_stats = {}
        try:
            price_feed_stats = self.price_feed.get_health()
        except Exception:
            price_feed_stats = {}
        try:
            ticker_provider_stats = self.ticker_provider.get_stats()
        except Exception:
            ticker_provider_stats = {}
        try:
            dynamic_seeds_stats = self.rebalancer.get_dynamic_seeds_status()
        except Exception:
            dynamic_seeds_stats = {}
        try:
            relay_server_stats = self.relay_server.get_status() if self.relay_server else {}
        except Exception:
            relay_server_stats = {}
        try:
            relay_client_stats = self.relay_client.get_status() if self.relay_client else {}
        except Exception:
            relay_client_stats = {}
        return {
            "running": self._running,
            "uptime_seconds": round(uptime, 1),
            "book_status": book_status,
            "cross_exchange": cross_stats,
            "executor": executor_stats,
            "funding": funding_stats,
            "triangular": tri_stats,
            "triangular_bybit": tri_bybit_stats,
            "basis": basis_stats,
            "listing_monitor": listing_monitor_stats,
            "listing": listing_arb_stats,
            "pairs": pairs_arb_stats,
            "risk": risk_status,
            "tracker_summary": tracker_summary,
            "contract_verification": contract_stats,
            "pair_discovery": discovery_stats,
            "volume_limiter": volume_limiter_stats,
            "temporal_analyzer": temporal_stats,
            "pair_expansion": pair_expansion_stats,
            "capital_velocity": velocity_stats,
            "edge_decay": edge_decay_stats,
            "strategy_allocation": allocator_stats,
            "capital_allocation": capital_alloc_stats,
            "capital_guard": capital_guard_stats,
            "price_feed": price_feed_stats,
            "ticker_provider": ticker_provider_stats,
            "dynamic_seeds": dynamic_seeds_stats,
            "relay_server": relay_server_stats,
            "relay_client": relay_client_stats,
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


    async def _run_relay_client(self):
        """Run the order book relay client (receives MEXC books from Bangalore)."""
        await asyncio.sleep(3)  # Wait for book manager to initialize
        logger.info(f"Starting relay client → {self.relay_client._url}")
        try:
            await self.relay_client.start()
        except asyncio.CancelledError:
            logger.info("Relay client task cancelled")
        except Exception as e:
            logger.error(f"Relay client fatal error: {e}")

    async def _run_hedged_mm(self):
        """Run hedged market maker strategy."""
        await asyncio.sleep(20)  # Let other systems stabilize
        logger.info("Starting Hedged Market Maker...")
        try:
            await self.hedged_mm.start()
        except asyncio.CancelledError:
            logger.info("Hedged MM task cancelled")
            if self.hedged_mm:
                await self.hedged_mm.stop()
        except Exception as e:
            logger.error(f"Hedged MM fatal error: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())


async def main():
    """Entry point for standalone arbitrage engine."""
    import argparse
    parser = argparse.ArgumentParser(description="Renaissance Arbitrage Engine")
    parser.add_argument('--config', default='arbitrage/config/arbitrage.yaml', help='Config file path')
    parser.add_argument('--paper', action='store_true', default=True, help='Paper trading mode')
    parser.add_argument('--live', action='store_true', help='Live trading mode (overrides --paper)')
    args = parser.parse_args()

    import yaml as _yaml
    with open(args.config) as _f:
        cfg = _yaml.safe_load(_f)

    if args.live:
        cfg.setdefault('paper_trading', {})['enabled'] = False
        logger.info("*** LIVE TRADING MODE — Real orders will be placed ***")
    else:
        cfg.setdefault('paper_trading', {})['enabled'] = True

    orchestrator = ArbitrageOrchestrator(config=cfg)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    for s in (sig.SIGINT, sig.SIGTERM):
        loop.add_signal_handler(s, lambda: asyncio.create_task(orchestrator.stop()))

    await orchestrator.start()




if __name__ == "__main__":
    asyncio.run(main())

