"""
Hedged Market Maker — posts maker orders on wide-spread USDC pairs
and instantly hedges on tight USDT pairs.

NOT arbitrage. This is market making with directional hedge.

Revenue model:
- Capture portion of USDC spread (maker, 0% fee)
- Pay hedge cost on USDT pair (taker, 5 bps)
- Pay USDC/USDT rebalancing cost (taker, 5 bps)
- Net: USDC capture minus 10 bps

Risk model:
- Directional exposure between fill and hedge (~milliseconds)
- Adverse selection (informed traders hit our quotes)
- Inventory buildup if one side fills more than other

Capital model:
- Each active pair ties up: order_size on maker side
- Order size scaled to pair volume for fast fills
- Target: no order waiting longer than max_wait_minutes
"""
import asyncio
import json
import logging
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..exchanges.base import (
    OrderRequest, OrderResult, OrderSide, OrderType, OrderStatus, TimeInForce,
)
from ..utils.tick_check import get_tick_size_cached, check_tick_viability

logger = logging.getLogger("arb.market_maker")

# Hard safety limits — never exceed these regardless of config
ABSOLUTE_MAX_ORDER_USD = 200
ABSOLUTE_MAX_PAIRS = 15
ABSOLUTE_MAX_CAPITAL_PCT = 0.80
ABSOLUTE_MAX_DAILY_LOSS = 50
ABSOLUTE_MAX_DIRECTIONAL_USD = 200


@dataclass
class PairState:
    """Tracks state for one active market-making pair."""
    token: str
    usdc_symbol: str        # e.g. "FET/USDC"
    usdt_symbol: str        # e.g. "FET/USDT"
    order_size_usd: float
    usdc_spread_bps: float
    usdt_spread_bps: float
    usdc_vol_24h: float
    sides: int              # 1 = buy-only, 2 = both sides

    # Active maker orders on USDC pair
    buy_order_id: Optional[str] = None
    buy_price: Optional[Decimal] = None
    buy_qty: Optional[Decimal] = None
    buy_posted_at: Optional[float] = None
    buy_ref_usdt_mid: Optional[float] = None

    sell_order_id: Optional[str] = None
    sell_price: Optional[Decimal] = None
    sell_qty: Optional[Decimal] = None
    sell_posted_at: Optional[float] = None
    sell_ref_usdt_mid: Optional[float] = None

    # Stats
    fills_count: int = 0
    profit_usd: float = 0.0
    adverse_fills: int = 0


class HedgedMarketMaker:

    def __init__(self, mexc_client, config: dict, db_path: str = "data/arbitrage.db"):
        self.client = mexc_client
        self.config = config
        self.db_path = db_path

        # Config with defaults
        self.observation_mode = config.get('observation_mode', True)
        self.min_usdc_spread_bps = config.get('min_usdc_spread_bps', 15.0)
        self.max_usdt_spread_bps = config.get('max_usdt_spread_bps', 10.0)
        self.min_spread_ratio = config.get('min_spread_ratio', 3.0)
        self.min_usdc_volume = config.get('min_usdc_volume_24h', 10000)
        self.min_usdt_volume = config.get('min_usdt_volume_24h', 50000)
        self.min_trade_usd = config.get('min_trade_usd', 20)
        self.max_trade_usd = min(config.get('max_trade_usd', 200), ABSOLUTE_MAX_ORDER_USD)
        self.target_fill_min = config.get('target_fill_minutes', 5)
        self.max_volume_pct = config.get('max_volume_pct', 0.02)
        self.spread_capture_pct = config.get('spread_capture_pct', 0.3)
        self.refresh_threshold_bps = config.get('refresh_threshold_bps', 10)
        self.max_order_age_sec = config.get('max_order_age_seconds', 120)
        self.max_pairs = min(config.get('max_pairs', 10), ABSOLUTE_MAX_PAIRS)
        self.max_capital_pct = min(config.get('max_capital_deployed_pct', 0.80), ABSOLUTE_MAX_CAPITAL_PCT)
        self.max_directional_usd = min(config.get('max_directional_exposure_usd', 200), ABSOLUTE_MAX_DIRECTIONAL_USD)
        self.max_daily_loss = min(config.get('max_daily_loss_usd', 25), ABSOLUTE_MAX_DAILY_LOSS)
        self.hedge_timeout_ms = config.get('hedge_timeout_ms', 3000)
        self.hedge_slippage_bps = config.get('hedge_slippage_bps', 2)
        self.rebalance_threshold = config.get('rebalance_threshold_usd', 50)
        self.rescan_interval_min = config.get('rescan_interval_minutes', 60)
        self.adverse_threshold_bps = config.get('adverse_threshold_bps', 20)
        self.adverse_removal_pct = config.get('adverse_removal_pct', 0.30)
        self.check_interval = config.get('check_interval_seconds', 0.5)

        # Heartbeat
        self._heartbeat_interval = 300  # 5 minutes
        self._last_heartbeat = 0
        self._heartbeat_loop_count = 0
        self._heartbeat_state_changes = 0

        # State
        self.active_pairs: Dict[str, PairState] = {}
        self._running = False
        self._precision_cache: Dict[str, Tuple[int, int]] = {}  # symbol -> (price_prec, qty_prec)

        # Performance
        self.total_fills = 0
        self.total_profit_usd = 0.0
        self.total_hedge_cost_usd = 0.0
        self.daily_pnl = 0.0
        self._day_start = datetime.utcnow().date()
        self._start_time = time.time()

        # Candidate cache
        self._candidates: List[dict] = []
        self._last_scan_time = 0.0

        # Dedicated thread pool for REST calls (avoids default pool exhaustion)
        from concurrent.futures import ThreadPoolExecutor
        self._rest_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="mm_rest")

        # Persistent HTTP session for REST calls
        import requests as _requests
        self._rest_session = _requests.Session()
        self._rest_session.headers.update({"Content-Type": "application/json"})

        # API validation cache (pairs that return error 10007)
        self._api_blocked: Dict[str, float] = {}  # token -> timestamp when blocked
        self._api_blocked_recheck_hours: float = 6.0

        # Monitoring (all candidates, independent of capital)
        self.missed_opportunities: int = 0
        self.missed_profit_usd: float = 0.0
        self._monitor_interval: float = 30.0
        self._last_monitor_scan: float = 0.0
        self._monitor_scan_count: int = 0
        self.deployable_capital: float = 0.0

        # Init DB
        self._init_db()

    def _init_db(self):
        """Create mm_trades table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mm_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                token TEXT NOT NULL,
                side TEXT NOT NULL,
                maker_symbol TEXT,
                maker_price REAL,
                maker_quantity REAL,
                maker_fill_ms INTEGER,
                maker_fee_bps REAL DEFAULT 0,
                hedge_symbol TEXT,
                hedge_price REAL,
                hedge_quantity REAL,
                hedge_fill_ms INTEGER,
                hedge_fee_bps REAL,
                hedge_status TEXT,
                gross_profit_usd REAL,
                hedge_cost_usd REAL,
                rebalance_cost_usd REAL,
                net_profit_usd REAL,
                capture_bps REAL,
                usdc_spread_bps REAL,
                usdt_spread_bps REAL,
                usdc_volume_24h REAL,
                order_size_usd REAL,
                time_to_fill_seconds REAL,
                status TEXT DEFAULT 'pending'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mm_token ON mm_trades(token)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mm_status ON mm_trades(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mm_timestamp ON mm_trades(timestamp)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mm_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                token TEXT NOT NULL,
                side TEXT NOT NULL,
                usdc_spread_bps REAL,
                usdt_spread_bps REAL,
                net_bps REAL,
                est_profit_usd REAL,
                order_size_usd REAL,
                est_fill_minutes REAL,
                status TEXT NOT NULL,
                capital_available REAL,
                capital_needed REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mm_opp_token ON mm_opportunities(token)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mm_opp_status ON mm_opportunities(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mm_opp_ts ON mm_opportunities(timestamp)")
        conn.commit()
        conn.close()
        logger.info("MM DB: mm_trades + mm_opportunities tables ready")

    async def _get_available_capital(self) -> float:
        """Query ACTUAL MEXC balance via direct REST API with retry."""
        import hmac, hashlib, aiohttp

        api_key = getattr(self.client, '_api_key', '')
        api_secret = getattr(self.client, '_api_secret', '')

        if not api_key or not api_secret:
            logger.warning("MM CAPITAL: no API credentials — using fallback")
            return self.max_trade_usd * self.max_pairs * 2

        last_err = None
        for attempt in range(3):
            try:
                def _sync_balance():
                    import requests as _requests
                    ts = str(int(time.time() * 1000))
                    params = {'timestamp': ts, 'recvWindow': '60000'}
                    query = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
                    sig = hmac.new(
                        api_secret.encode(), query.encode(), hashlib.sha256
                    ).hexdigest()
                    url = f"https://api.mexc.com/api/v3/account?{query}&signature={sig}"
                    resp = _requests.get(url, headers={
                        'X-MEXC-APIKEY': api_key,
                        'Content-Type': 'application/json',
                    }, timeout=8)
                    data = resp.json()
                    if resp.status_code != 200:
                        raise Exception(f"HTTP {resp.status_code}: {data.get('msg', data)}")
                    return data
                loop = asyncio.get_running_loop()
                data = await asyncio.wait_for(loop.run_in_executor(self._rest_executor, _sync_balance), timeout=10.0)

                usdt_free = 0.0
                usdc_free = 0.0
                for bal in data.get('balances', []):
                    asset = bal.get('asset', '')
                    free = float(bal.get('free', 0))
                    if asset == 'USDT':
                        usdt_free = free
                    elif asset == 'USDC':
                        usdc_free = free

                # Auto-rebalance if USDC is low
                usdc_free = await self._auto_rebalance_usdc(usdc_free, usdt_free)

                total_available = usdc_free  # USDC primary — MM posts on TOKEN/USDC pairs
                deployable = total_available * self.max_capital_pct

                # Cap by capital allocator budget if available
                allocator = getattr(self, 'capital_allocator', None)
                alloc_cap = None
                if allocator:
                    try:
                        budget = await allocator.get_available_budget("market_maker")
                        alloc_cap = budget.get("USDC", 0) + budget.get("USDT", 0)
                        if alloc_cap > 0:
                            deployable = min(deployable, alloc_cap)
                    except Exception as e:
                        logger.warning(f"MM CAPITAL: allocator check failed: {e}")

                logger.info(
                    f"MM CAPITAL: USDC_free=${usdc_free:.2f} USDT_free=${usdt_free:.2f} "
                    f"total={total_available:.2f} deployable={deployable:.2f} "
                    f"({self.max_capital_pct*100:.0f}% cap)"
                    f"{f' alloc_cap=${alloc_cap:.0f}' if alloc_cap is not None else ''}"
                )
                return deployable

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < 2:
                    logger.warning(f"MM CAPITAL attempt {attempt+1} failed: {type(e).__name__}: {e} — retrying in 2s")
                    await asyncio.sleep(2)

        logger.error(f"MM CAPITAL ERROR after 3 attempts: {last_err} — using fallback")
        return self.max_trade_usd * self.max_pairs * 2


    async def _auto_rebalance_usdc(self, usdc_free: float, usdt_free: float) -> float:
        """If USDC is low but USDT has funds, convert USDT to USDC via market order."""
        min_usdc = self.rebalance_threshold  # default $50
        # Only rebalance if USDC < threshold and USDT has enough to be useful
        if usdc_free >= min_usdc or usdt_free < 100:
            return usdc_free  # no rebalance needed

        # Convert 80% of USDT to USDC (keep some USDT for hedge costs)
        convert_amount = round(usdt_free * 0.80, 2)
        if convert_amount < 50:
            return usdc_free

        logger.info(
            f"MM REBALANCE: USDC=${usdc_free:.2f} below ${min_usdc:.0f} threshold, "
            f"converting ${convert_amount:.2f} USDT to USDC"
        )

        import hmac as _hmac, hashlib as _hashlib
        api_key = getattr(self.client, '_api_key', '')
        api_secret = getattr(self.client, '_api_secret', '')
        if not api_key:
            logger.warning("MM REBALANCE: no API credentials — skipping")
            return usdc_free

        try:
            def _sync_rebalance():
                ts = str(int(time.time() * 1000))
                params = {
                    "symbol": "USDCUSDT",
                    "side": "BUY",
                    "type": "MARKET",
                    "quoteOrderQty": str(convert_amount),
                    "timestamp": ts,
                    "recvWindow": "60000",
                }
                query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
                sig = _hmac.new(
                    api_secret.encode(), query.encode(), _hashlib.sha256
                ).hexdigest()
                url = f"https://api.mexc.com/api/v3/order?{query}&signature={sig}"
                import requests as _rq
                resp = _rq.post(url, headers={
                    "X-MEXC-APIKEY": api_key,
                    "Content-Type": "application/json",
                }, timeout=10)
                return resp.status_code, resp.json()

            loop = asyncio.get_running_loop()
            status_code, result = await asyncio.wait_for(
                loop.run_in_executor(self._rest_executor, _sync_rebalance), timeout=15.0
            )
            if status_code == 200:
                logger.info(f"MM REBALANCE OK: converted ${convert_amount:.2f} USDT to USDC")
                await asyncio.sleep(1)
                return usdc_free + convert_amount * 0.999
            else:
                msg = result.get("msg", result)
                logger.warning(f"MM REBALANCE FAILED: HTTP {status_code} — {msg}")
                return usdc_free
        except Exception as e:
            logger.warning(f"MM REBALANCE ERROR: {type(e).__name__}: {e}")
            return usdc_free

    async def _fetch_with_timeout(self, coro, timeout_seconds: float = 10.0, label: str = ""):
        """Wrap any async call with a hard timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(f"API TIMEOUT ({timeout_seconds}s): {label}")
            return None
        except Exception as e:
            logger.warning(f"API ERROR: {label}: {e}")
            return None

    async def _fetch_tickers_raw(self) -> dict:
        """Fetch all MEXC bid/ask via raw REST (thread pool)."""
        url = "https://api.mexc.com/api/v3/ticker/bookTicker"

        def _sync_fetch():
            import requests as _requests
            resp = _requests.get(url, timeout=15)
            if resp.status_code != 200:
                return {}
            data = resp.json()
            tickers = {}
            for item in data:
                symbol = item.get("symbol", "")
                for quote in ("USDC", "USDT"):
                    if symbol.endswith(quote):
                        base = symbol[:-len(quote)]
                        key = f"{base}/{quote}"
                        bid = float(item.get("bidPrice", 0) or 0)
                        ask = float(item.get("askPrice", 0) or 0)
                        if bid > 0 and ask > 0:
                            tickers[key] = {"bid": bid, "ask": ask}
                        break
            return tickers

        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(self._rest_executor, _sync_fetch),
                timeout=20.0
            )
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("MM MONITOR: bookTicker timed out (20s)")
            return {}
        except Exception as e:
            logger.warning(f"MM MONITOR: bookTicker error: {type(e).__name__}: {e}")
            return {}

    async def _monitor_all_candidates(self) -> list:
        """Fetch all tickers once, evaluate every candidate for opportunities."""
        scan_start = time.time()

        all_tickers = await self._fetch_tickers_raw()
        if not all_tickers:
            logger.warning("MM MONITOR: no ticker data — skipping cycle")
            return []

        opportunities = []
        capital_in_use = sum(p.order_size_usd * p.sides for p in self.active_pairs.values())
        capital_remaining = max(0, self.deployable_capital - capital_in_use)

        for candidate in self._candidates:
            token = candidate["token"]
            usdc_t = all_tickers.get(f"{token}/USDC")
            usdt_t = all_tickers.get(f"{token}/USDT")
            if not usdc_t or not usdt_t:
                continue

            opp = self._evaluate_from_tickers(candidate, usdc_t, usdt_t)
            if not opp:
                continue

            opp["capital_available"] = capital_remaining
            opp["capital_needed"] = opp["order_size"] * 2

            if token in self.active_pairs:
                opp["status"] = "traded"
            elif capital_remaining >= opp["order_size"]:
                opp["status"] = "available"
            else:
                opp["status"] = "no_capital"

            opportunities.append(opp)

        scan_ms = (time.time() - scan_start) * 1000
        self._monitor_scan_count += 1

        traded = sum(1 for o in opportunities if o["status"] == "traded")
        available = sum(1 for o in opportunities if o["status"] == "available")
        no_cap = sum(1 for o in opportunities if o["status"] == "no_capital")

        logger.info(
            f"MM SCAN: {len(self._candidates)} pairs in {scan_ms:.0f}ms | "
            f"{len(opportunities)} opps (traded={traded} avail={available} no_cap={no_cap})"
        )

        for opp in opportunities:
            if opp["status"] == "no_capital":
                self.missed_opportunities += 1
                self.missed_profit_usd += opp.get("est_profit_usd", 0)
                logger.info(
                    f"MM NO CAPITAL: {opp['token']} | "
                    f"net={opp['net_bps']:.1f}bps | "
                    f"would_profit=${opp.get('est_profit_usd', 0):.4f} | "
                    f"need=${opp['capital_needed']:.0f} USDC | "
                    f"total_missed=${self.missed_profit_usd:.2f} ({self.missed_opportunities} opps)"
                )
            self._persist_opportunity(opp)

        return opportunities

    def _evaluate_from_tickers(self, candidate: dict, usdc_t: dict, usdt_t: dict):
        """Evaluate one candidate pair from live ticker data."""
        bid_usdc = usdc_t.get("bid")
        ask_usdc = usdc_t.get("ask")
        bid_usdt = usdt_t.get("bid")
        ask_usdt = usdt_t.get("ask")

        if not all([bid_usdc, ask_usdc, bid_usdt, ask_usdt]):
            return None
        if bid_usdc <= 0 or ask_usdc <= 0 or bid_usdt <= 0 or ask_usdt <= 0:
            return None

        usdc_spread_bps = (ask_usdc - bid_usdc) / ask_usdc * 10000
        usdt_spread_bps = (ask_usdt - bid_usdt) / ask_usdt * 10000

        if usdc_spread_bps < self.min_usdc_spread_bps:
            return None
        if usdt_spread_bps > self.max_usdt_spread_bps:
            return None

        capture_bps = usdc_spread_bps * self.spread_capture_pct
        net_bps = capture_bps - 10  # 5bps hedge + 5bps rebalance

        if net_bps <= 0:
            return None

        order_size = self._calculate_order_size(candidate)
        usdc_vol = usdc_t.get("quoteVolume") or candidate.get("usdc_vol_24h", 0) or 0
        vol_per_min = max(usdc_vol, 1) / 1440
        est_fill_min = order_size / max(vol_per_min, 0.01)
        fills_per_hour = 60 / max(est_fill_min, 0.1)
        est_profit_per_fill = order_size * net_bps / 10000
        est_profit_hourly = est_profit_per_fill * fills_per_hour

        return {
            "token": candidate["token"],
            "side": "both",
            "usdc_spread_bps": round(usdc_spread_bps, 1),
            "usdt_spread_bps": round(usdt_spread_bps, 1),
            "net_bps": round(net_bps, 1),
            "order_size": order_size,
            "est_fill_min": round(est_fill_min, 1),
            "est_profit_usd": round(est_profit_per_fill, 4),
            "est_profit_hourly": round(est_profit_hourly, 4),
            "usdc_vol_24h": round(usdc_vol, 0),
        }

    def _persist_opportunity(self, opp: dict):
        """Save opportunity to mm_opportunities table."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """INSERT INTO mm_opportunities
                   (token, side, usdc_spread_bps, usdt_spread_bps, net_bps,
                    est_profit_usd, order_size_usd, est_fill_minutes,
                    status, capital_available, capital_needed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    opp["token"], opp.get("side", "both"),
                    opp.get("usdc_spread_bps"), opp.get("usdt_spread_bps"),
                    opp.get("net_bps"), opp.get("est_profit_usd"),
                    opp.get("order_size"), opp.get("est_fill_min"),
                    opp.get("status", "unknown"),
                    opp.get("capital_available"), opp.get("capital_needed"),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"MM opp persist error: {e}")

    async def _validate_pair_api(self, token: str) -> bool:
        """Test if TOKEN/USDC can be traded via MEXC API."""
        # Check cache first
        if token in self._api_blocked:
            blocked_at = self._api_blocked[token]
            hours_ago = (time.time() - blocked_at) / 3600
            if hours_ago < self._api_blocked_recheck_hours:
                return False  # still blocked
            else:
                logger.info(f"MM RECHECK: {token}/USDC — {hours_ago:.1f}h since blocked, retesting")
                del self._api_blocked[token]

        usdc_sym = f"{token}/USDC"
        try:
            book = await asyncio.wait_for(
                self.client.get_order_book(usdc_sym, depth=5),
                timeout=5.0,
            )
            if book and book.best_bid and book.best_ask:
                return True
            else:
                logger.info(f"MM PAIR BLOCKED: {usdc_sym} — empty order book, monitoring only")
                self._api_blocked[token] = time.time()
                return False
        except asyncio.TimeoutError:
            # Timeout is transient — don't permanently block, just allow
            logger.debug(f"MM VALIDATE: {usdc_sym} timed out — assuming valid")
            return True
        except Exception as e:
            err_str = str(e)
            if "10007" in err_str or "not support" in err_str.lower():
                logger.info(f"MM PAIR BLOCKED: {usdc_sym} — API error 10007, monitoring only")
                self._api_blocked[token] = time.time()
                return False
            else:
                # Other errors are transient — don't block
                logger.debug(f"MM VALIDATE: {usdc_sym} error: {e} — assuming valid")
                return True

    async def start(self):
        """Load candidates and start market making."""
        self._running = True

        # Load candidates from file or scan
        await self._load_or_scan_candidates()

        if not self._candidates:
            logger.warning("MM: No candidates found — exiting")
            return

        # Select pairs based on capital velocity
        # Query real wallet balance for capital allocation
        try:
            deployable = await asyncio.wait_for(self._get_available_capital(), timeout=45.0)
        except asyncio.TimeoutError:
            logger.warning('MM CAPITAL: hard timeout after 45s — using fallback')
            deployable = self.max_trade_usd * self.max_pairs * 2
        self.deployable_capital = deployable
        selected = self._select_pairs(self._candidates, available_capital=deployable)
        if not selected:
            logger.warning("MM: No pairs selected after capital allocation — exiting")
            return

        # Validate pairs exist on MEXC API before starting (parallel, 15s total max)
        try:
            async def _validate_one(pi):
                try:
                    ok = await asyncio.wait_for(self._validate_pair_api(pi['token']), timeout=8.0)
                    return (pi, ok)
                except (asyncio.TimeoutError, TimeoutError):
                    logger.debug(f"MM VALIDATE: {pi['token']} timed out — assuming valid")
                    return (pi, True)
            results = await asyncio.wait_for(
                asyncio.gather(*[_validate_one(pi) for pi in selected], return_exceptions=True),
                timeout=15.0
            )
            validated = []
            for r in results:
                if isinstance(r, Exception):
                    continue
                pi, ok = r
                if ok:
                    validated.append(pi)
                else:
                    logger.info(f"MM SKIP PAIR: {pi['token']} — API blocked, skipping")
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("MM VALIDATE: total timeout — skipping validation, using all selected pairs")
            validated = list(selected)

        if not validated:
            logger.warning("MM: No valid pairs after API check — exiting")
            return

        # Start each validated pair
        for pair_info in validated:
            await self._start_pair(pair_info)

        mode_str = "OBSERVATION" if self.observation_mode else "LIVE"
        blocked_count = len(self._api_blocked)
        logger.info(
            f"MM STARTED ({mode_str}): {len(self.active_pairs)} valid pairs "
            f"(skipped {blocked_count} API-blocked), "
            f"${sum(p.order_size_usd * p.sides for p in self.active_pairs.values()):.0f} deployed"
        )

        # Main loop
        await self._run_loop()

    async def _load_or_scan_candidates(self):
        """Load candidates from cached file or run fresh scan."""
        cache_path = Path("data/market_maker_candidates.json")
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    self._candidates = json.load(f)
                age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
                logger.info(f"MM: Loaded {len(self._candidates)} candidates from cache ({age_hours:.1f}h old)")
                if age_hours < 1:
                    return
                logger.info("MM: Cache > 1h old — will rescan in background")
            except Exception as e:
                logger.warning(f"MM: Cache load failed: {e}")

        # Scan in background if no candidates or stale
        if not self._candidates:
            logger.info("MM: No cached candidates — running scan (this takes ~2 min)")
            self._candidates = await self._scan_candidates()
            self._last_scan_time = time.time()

    async def _scan_candidates(self) -> List[dict]:
        """Scan MEXC for candidate pairs matching our criteria."""
        try:
            # Use ccxt via the client's exchange object
            exchange = self.client._exchange
            if not exchange.markets:
                await exchange.load_markets()

            # Find tokens with both USDC and USDT pairs
            usdc_pairs = {}
            usdt_pairs = {}
            for sym in exchange.symbols:
                parts = sym.split('/')
                if len(parts) != 2:
                    continue
                token, quote = parts
                if quote == 'USDC':
                    usdc_pairs[token] = sym
                elif quote == 'USDT':
                    usdt_pairs[token] = sym

            both = set(usdc_pairs.keys()) & set(usdt_pairs.keys())
            logger.info(f"MM SCAN: {len(both)} tokens with both USDC+USDT pairs")

            candidates = []
            for token in sorted(both):
                try:
                    ticker_usdc = await self._fetch_with_timeout(
                        exchange.fetch_ticker(f'{token}/USDC'),
                        timeout_seconds=10, label=f"ticker {token}/USDC",
                    )
                    ticker_usdt = await self._fetch_with_timeout(
                        exchange.fetch_ticker(f'{token}/USDT'),
                        timeout_seconds=10, label=f"ticker {token}/USDT",
                    )
                    if not ticker_usdc or not ticker_usdt:
                        continue
                    await asyncio.sleep(0.15)

                    if not ticker_usdc.get('bid') or not ticker_usdc.get('ask'):
                        continue
                    if not ticker_usdt.get('bid') or not ticker_usdt.get('ask'):
                        continue
                    if ticker_usdc['bid'] <= 0 or ticker_usdt['bid'] <= 0:
                        continue

                    usdc_spread_bps = ((ticker_usdc['ask'] - ticker_usdc['bid'])
                                       / ticker_usdc['ask'] * 10000)
                    usdt_spread_bps = ((ticker_usdt['ask'] - ticker_usdt['bid'])
                                       / ticker_usdt['ask'] * 10000)

                    usdc_vol = ticker_usdc.get('quoteVolume', 0) or 0
                    usdt_vol = ticker_usdt.get('quoteVolume', 0) or 0

                    if usdc_spread_bps < self.min_usdc_spread_bps:
                        continue
                    if usdt_spread_bps > self.max_usdt_spread_bps:
                        continue
                    if usdc_vol < self.min_usdc_volume:
                        continue
                    if usdt_vol < self.min_usdt_volume:
                        continue
                    if usdt_spread_bps > 0:
                        spread_ratio = usdc_spread_bps / usdt_spread_bps
                    else:
                        spread_ratio = 999
                    if spread_ratio < self.min_spread_ratio:
                        continue

                    capture_bps = usdc_spread_bps * 0.4
                    net_bps = capture_bps - 10

                    # Tick size check: reject if rounding eats > 50% of profit
                    usdc_price = (ticker_usdc['bid'] + ticker_usdc['ask']) / 2
                    usdt_price = (ticker_usdt['bid'] + ticker_usdt['ask']) / 2
                    tick_usdc = get_tick_size_cached(exchange.markets, f'{token}/USDC')
                    tick_usdt = get_tick_size_cached(exchange.markets, f'{token}/USDT')

                    tick_usdc_bps = 0.0
                    tick_usdt_bps = 0.0
                    total_rounding = 0.0
                    if tick_usdc and usdc_price > 0:
                        tick_usdc_bps = (tick_usdc / usdc_price) * 10000
                        total_rounding += tick_usdc_bps
                    if tick_usdt and usdt_price > 0:
                        tick_usdt_bps = (tick_usdt / usdt_price) * 10000
                        total_rounding += tick_usdt_bps

                    if net_bps > 0 and total_rounding > net_bps * 0.5:
                        continue  # Tick rounding kills profitability

                    usdc_vol_per_min = usdc_vol / 1440
                    est_fill_min = 100 / max(usdc_vol_per_min, 0.01)
                    fills_per_hour = 60 / max(est_fill_min, 0.1)
                    bps_per_hour = net_bps * fills_per_hour

                    candidates.append({
                        'token': token,
                        'usdc_spread_bps': round(usdc_spread_bps, 1),
                        'usdt_spread_bps': round(usdt_spread_bps, 1),
                        'spread_ratio': round(spread_ratio, 1),
                        'usdc_vol_24h': round(usdc_vol, 0),
                        'usdt_vol_24h': round(usdt_vol, 0),
                        'capture_bps': round(capture_bps, 1),
                        'net_bps': round(net_bps, 1),
                        'est_fill_min': round(est_fill_min, 1),
                        'bps_per_hour': round(bps_per_hour, 1),
                        'tick_usdc_bps': round(tick_usdc_bps, 1),
                        'tick_usdt_bps': round(tick_usdt_bps, 1),
                        'total_rounding_bps': round(total_rounding, 1),
                        'net_after_rounding_bps': round(net_bps - total_rounding, 1),
                    })

                except Exception:
                    continue

            candidates.sort(key=lambda x: x['bps_per_hour'], reverse=True)

            # Save to cache
            try:
                with open('data/market_maker_candidates.json', 'w') as f:
                    json.dump(candidates, f, indent=2)
            except Exception:
                pass

            logger.info(f"MM SCAN: {len(candidates)} candidates found")
            return candidates

        except Exception as e:
            logger.error(f"MM SCAN failed: {e}")
            return []

    def _select_pairs(self, candidates: List[dict], available_capital: float = None) -> List[dict]:
        """Allocate capital across pairs to maximize bps/hour."""
        if available_capital is not None and available_capital > 0:
            total_capital = available_capital
        else:
            total_capital = self.max_trade_usd * self.max_pairs * 2  # fallback
        allocated = 0
        selected = []

        for c in candidates:
            if len(selected) >= self.max_pairs:
                break

            order_size = self._calculate_order_size(c)
            capital_needed = order_size * 2  # both sides

            if allocated + capital_needed > total_capital:
                if allocated + order_size <= total_capital:
                    c['order_size'] = order_size
                    c['sides'] = 1
                    selected.append(c)
                    allocated += order_size
                continue

            c['order_size'] = order_size
            c['sides'] = 2
            selected.append(c)
            allocated += capital_needed

        logger.info(
            f"MM ALLOCATION: {len(selected)} pairs, "
            f"${allocated:.0f} deployed"
        )
        return selected

    def _calculate_order_size(self, pair_info: dict) -> float:
        """Size order relative to pair volume for fast fills."""
        vol_per_min = pair_info['usdc_vol_24h'] / 1440
        raw_size = vol_per_min * self.target_fill_min

        max_pct_vol = pair_info['usdc_vol_24h'] * self.max_volume_pct
        size = max(self.min_trade_usd, min(raw_size, self.max_trade_usd, max_pct_vol))
        return round(size, 2)

    async def _start_pair(self, pair_info: dict):
        """Initialize a pair and post initial maker orders."""
        token = pair_info['token']
        usdc_sym = f"{token}/USDC"
        usdt_sym = f"{token}/USDT"

        state = PairState(
            token=token,
            usdc_symbol=usdc_sym,
            usdt_symbol=usdt_sym,
            order_size_usd=pair_info['order_size'],
            usdc_spread_bps=pair_info['usdc_spread_bps'],
            usdt_spread_bps=pair_info['usdt_spread_bps'],
            usdc_vol_24h=pair_info['usdc_vol_24h'],
            sides=pair_info['sides'],
        )

        # Cache precisions
        for sym in [usdc_sym, usdt_sym]:
            if sym not in self._precision_cache:
                try:
                    info = await self.client.get_symbol_info(sym)
                    if info:
                        pp = self._safe_precision(info.get('price_precision', 8))
                        qp = self._safe_precision(info.get('quantity_precision', 8))
                        self._precision_cache[sym] = (pp, qp)
                except Exception as e:
                    logger.debug(f"MM: precision fetch for {sym} failed: {e}")
                    self._precision_cache[sym] = (8, 8)

        self.active_pairs[token] = state

        # Post initial orders (unless observation mode)
        if not self.observation_mode:
            await self._post_maker_orders(token)
        else:
            # Log what we WOULD post
            try:
                book = await asyncio.wait_for(
                    self.client.get_order_book(usdc_sym, depth=5), timeout=10.0
                )
                if book and book.best_bid and book.best_ask:
                    buy_price = self._calc_maker_price('buy', book)
                    sell_price = self._calc_maker_price('sell', book)
                    qty = self._calc_quantity(state, buy_price)
                    logger.info(
                        f"MM OBSERVE: {token} would post "
                        f"BUY@{float(buy_price):.6f} SELL@{float(sell_price):.6f} "
                        f"qty={float(qty):.4f} (${state.order_size_usd:.0f})"
                    )
            except Exception as e:
                logger.debug(f"MM OBSERVE: {token} book fetch failed: {e}")

    async def _post_maker_orders(self, token: str):
        """Post buy and/or sell LIMIT_MAKER orders on USDC pair."""
        state = self.active_pairs.get(token)
        if not state:
            return

        # Skip API-blocked pairs
        if token in self._api_blocked:
            return

        try:
            # Use direct REST API (faster + avoids ccxt timeout issues)
            book = await self._get_order_book_rest(state.usdc_symbol, depth=5)
            if book == "BLOCKED":
                logger.warning(f"MM API BLOCKED: {token} — MEXC error from REST")
                self._api_blocked[token] = time.time()
                if token in self.active_pairs:
                    del self.active_pairs[token]
                    logger.info(f"MM REMOVED: {token} — API blocked, {len(self.active_pairs)} pairs remaining")
                return
            if not book or not book.best_bid or not book.best_ask:
                logger.warning(f"MM: {token} no book data — skipping post")
                return

            # Get USDT mid for reference
            try:
                usdt_mid = await asyncio.wait_for(self._get_usdt_mid(token), timeout=3.0)
            except (asyncio.TimeoutError, TimeoutError, Exception):
                usdt_mid = Decimal('0')

            spread_dec = book.best_ask - book.best_bid
            if spread_dec <= 0:
                logger.debug(f"MM SKIP: {token} spread={float(spread_dec):.8f} — zero or negative")
                return

            # Post buy side
            if state.buy_order_id is None:
                buy_price = self._calc_maker_price('buy', book, state.usdc_spread_bps)
                prec_p = self._precision_cache.get(state.usdc_symbol, (8, 8))[0]
                buy_price = self._round_decimal(buy_price, prec_p)
                buy_qty = self._calc_quantity(state, buy_price)
                if buy_price <= 0 or buy_price >= book.best_ask:
                    logger.info(f"MM SKIP BUY: {token} price={float(buy_price):.6f} ask={float(book.best_ask):.6f} — would cross")
                elif buy_qty and buy_qty > 0:
                    result = await self._place_maker_order(
                        state.usdc_symbol, 'buy', buy_price, buy_qty, token
                    )
                    if result and result.order_id:
                        state.buy_order_id = result.order_id
                        state.buy_price = buy_price
                        state.buy_qty = buy_qty
                        state.buy_posted_at = time.time()
                        state.buy_ref_usdt_mid = usdt_mid
                        logger.info(
                            f"MM POST: {token} buy@{float(buy_price):.6f} USDC "
                            f"qty={float(buy_qty):.4f} (${state.order_size_usd:.0f}) "
                            f"— inside {state.usdc_spread_bps:.0f}bps spread"
                        )

            # Post sell side (if allocated both sides)
            if state.sides >= 2 and state.sell_order_id is None:
                # Re-fetch book to avoid stale data (buy order took time)
                sell_book = await self._get_order_book_rest(state.usdc_symbol, depth=5)
                if sell_book and sell_book != "BLOCKED" and sell_book.best_bid and sell_book.best_ask:
                    book = sell_book  # use fresh book for sell pricing
                sell_price_raw = self._calc_maker_price('sell', book, state.usdc_spread_bps)
                prec_p = self._precision_cache.get(state.usdc_symbol, (8, 8))[0]
                sell_price = self._round_decimal_up(sell_price_raw, prec_p)
                sell_qty = self._calc_quantity(state, sell_price)
                logger.info(
                    f"MM SELL CALC: {token} bid={float(book.best_bid):.6f} ask={float(book.best_ask):.6f} "
                    f"spread={float(book.best_ask - book.best_bid):.6f} raw={float(sell_price_raw):.6f} "
                    f"rounded={float(sell_price):.6f} prec={prec_p}"
                )
                if sell_price <= 0 or sell_price <= book.best_bid:
                    logger.info(f"MM SKIP SELL: {token} price={float(sell_price):.6f} bid={float(book.best_bid):.6f} — would cross")
                elif sell_qty and sell_qty > 0:
                    result = await self._place_maker_order(
                        state.usdc_symbol, 'sell', sell_price, sell_qty, token
                    )
                    if result and result.order_id:
                        state.sell_order_id = result.order_id
                        state.sell_price = sell_price
                        state.sell_qty = sell_qty
                        state.sell_posted_at = time.time()
                        state.sell_ref_usdt_mid = usdt_mid
                        logger.info(
                            f"MM POST: {token} sell@{float(sell_price):.6f} USDC "
                            f"qty={float(sell_qty):.4f} (${state.order_size_usd:.0f}) "
                            f"— inside {state.usdc_spread_bps:.0f}bps spread"
                        )

        except Exception as e:
            err_str = str(e)
            if "10007" in err_str or "not support" in err_str.lower():
                logger.warning(f"MM API BLOCKED: {token} — {e}")
                self._api_blocked[token] = time.time()
                if token in self.active_pairs:
                    del self.active_pairs[token]
                    logger.info(f"MM REMOVED: {token} — API blocked, {len(self.active_pairs)} pairs remaining")
            else:
                logger.error(f"MM POST ERROR {token}: {type(e).__name__}: {e}")

    async def _get_order_book_rest(self, symbol: str, depth: int = 5):
        """Fetch order book via direct REST (thread pool, doesn't block event loop)."""
        mexc_sym = symbol.replace("/", "")

        def _sync_fetch():
            import requests as _requests
            url = f"https://api.mexc.com/api/v3/depth?symbol={mexc_sym}&limit={depth}"
            resp = _requests.get(url, timeout=5)
            return resp.json()

        try:
            loop = asyncio.get_running_loop()
            data = await asyncio.wait_for(
                loop.run_in_executor(self._rest_executor, _sync_fetch),
                timeout=8.0
            )
            # Check for MEXC error response
            if "code" in data:
                error_code = data.get("code", 0)
                error_msg = data.get("msg", "")
                if error_code == 10007 or "not support" in str(error_msg).lower():
                    logger.info(f"MM REST book {mexc_sym}: API error {error_code} — {error_msg}")
                    return "BLOCKED"
                logger.debug(f"MM REST book {mexc_sym}: error {error_code}: {error_msg}")
                return None
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            if not bids or not asks:
                logger.debug(f"MM REST book {mexc_sym}: empty — bids={len(bids)} asks={len(asks)}")
                return None
            class _Book:
                pass
            book = _Book()
            book.best_bid = Decimal(str(bids[0][0]))
            book.best_ask = Decimal(str(asks[0][0]))
            book.bids = [(Decimal(str(b[0])), Decimal(str(b[1]))) for b in bids]
            book.asks = [(Decimal(str(a[0])), Decimal(str(a[1]))) for a in asks]
            return book
        except (asyncio.TimeoutError, TimeoutError):
            logger.debug(f"MM REST book {mexc_sym}: timeout")
            return None
        except Exception as e:
            logger.debug(f"MM REST book error for {symbol}: {e}")
            return None

    async def _place_maker_order(self, symbol: str, side: str,
                                  price: Decimal, quantity: Decimal,
                                  token: str) -> Optional[OrderResult]:
        """Place a LIMIT_MAKER order via direct REST API (bypasses ccxt)."""
        import hmac as _hmac, hashlib as _hashlib

        api_key = getattr(self.client, '_api_key', '')
        api_secret = getattr(self.client, '_api_secret', '')
        if not api_key:
            logger.error(f"MM ORDER: no API key — cannot place {token} {side}")
            return None

        mexc_sym = symbol.replace("/", "")
        oid = f"mm_{token}_{side}_{uuid.uuid4().hex[:8]}"

        # Get precision for this pair
        prec_p, prec_q = self._precision_cache.get(symbol, (8, 8))

        def _sync_place():
            ts = str(int(time.time() * 1000))
            params = {
                "symbol": mexc_sym,
                "side": side.upper(),
                "type": "LIMIT_MAKER",
                "quantity": str(float(quantity)),
                "price": str(float(price)),
                "newClientOrderId": oid,
                "timestamp": ts,
                "recvWindow": "60000",
            }
            query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            sig = _hmac.new(
                api_secret.encode(), query.encode(), _hashlib.sha256
            ).hexdigest()
            url = f"https://api.mexc.com/api/v3/order?{query}&signature={sig}"
            import requests as _requests
            resp = _requests.post(url, headers={
                "X-MEXC-APIKEY": api_key,
                "Content-Type": "application/json",
            }, timeout=10)
            return resp.status_code, resp.json()

        try:
            loop = asyncio.get_running_loop()
            status_code, data = await asyncio.wait_for(
                loop.run_in_executor(self._rest_executor, _sync_place),
                timeout=15.0
            )

            if status_code == 200:
                order_id = data.get("orderId", oid)
                from datetime import datetime as _dt
                result = OrderResult(
                    exchange="mexc",
                    symbol=symbol,
                    order_id=str(order_id),
                    client_order_id=oid,
                    status=OrderStatus.OPEN,
                    side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                    order_type=OrderType.LIMIT_MAKER,
                    requested_quantity=quantity,
                    filled_quantity=Decimal('0'),
                    average_fill_price=price,
                    fee_amount=Decimal('0'),
                    fee_currency="USDC",
                    timestamp=_dt.utcnow(),
                    raw_response=data,
                )
                return result
            else:
                error_code = data.get("code", 0)
                error_msg = data.get("msg", "")
                if error_code == 10007 or "not support" in str(error_msg).lower():
                    logger.warning(
                        f"MM API BLOCKED: {token} {side} — MEXC error {error_code}: {error_msg}"
                    )
                    self._api_blocked[token] = time.time()
                    if token in self.active_pairs:
                        del self.active_pairs[token]
                        logger.info(f"MM REMOVED: {token} — API blocked, {len(self.active_pairs)} pairs remaining")
                elif error_code == 30005:
                    logger.info(
                        f"MM REJECTED: {token} {side} — Oversold (30005) price={float(price):.6f}"
                    )
                else:
                    logger.warning(
                        f"MM REJECTED: {token} {side} — "
                        f"code={error_code} msg={error_msg} price={float(price):.6f}"
                    )
                return None
        except (asyncio.TimeoutError, TimeoutError):
            logger.error(f"MM TIMEOUT: {token} {side} order placement")
            return None
        except Exception as e:
            logger.error(f"MM ORDER ERROR: {token} {side}: {e}")
            return None

    def _calc_maker_price(self, side: str, book, usdc_spread_bps: float = 0) -> Decimal:
        """Calculate price inside the USDC spread with safety checks."""
        best_bid = book.best_bid
        best_ask = book.best_ask
        spread = best_ask - best_bid

        if spread <= 0:
            return Decimal('0')  # zero/negative spread — caller will skip

        # Dynamic capture: more aggressive on wide spreads for faster fills
        capture = self.spread_capture_pct
        if usdc_spread_bps > 30:
            capture = 0.5  # 50% inside on wide spreads — closer to counterparty

        if side == 'buy':
            price = best_bid + spread * Decimal(str(capture))
            # Safety: must be strictly below best_ask
            if price >= best_ask:
                price = best_ask - Decimal('1E-8')
        else:
            price = best_ask - spread * Decimal(str(capture))
            # Safety: must be strictly above best_bid
            if price <= best_bid:
                price = best_bid + Decimal('1E-8')

        return price

    def _calc_quantity(self, state: PairState, price: Decimal) -> Optional[Decimal]:
        """Calculate order quantity based on target USD size."""
        if price <= 0:
            return None
        raw_qty = Decimal(str(state.order_size_usd)) / price

        # Round down to precision
        prec = self._precision_cache.get(state.usdc_symbol, (8, 8))
        qty_prec = prec[1]
        rounded = self._round_decimal(raw_qty, qty_prec)

        if rounded <= 0:
            return None
        return rounded

    async def _run_loop(self):
        """Main loop: check fills, manage orders, refresh quotes."""
        rescan_counter = 0
        hourly_log_counter = 0

        while self._running:
            try:
                # Reset daily P&L
                today = datetime.utcnow().date()
                if today != self._day_start:
                    logger.info(
                        f"MM DAILY RESET: yesterday P&L=${self.daily_pnl:.2f}, "
                        f"fills={self.total_fills}"
                    )
                    self.daily_pnl = 0.0
                    self._day_start = today

                # Check daily loss limit
                if self.daily_pnl < -self.max_daily_loss:
                    logger.warning(
                        f"MM DAILY LOSS LIMIT: ${self.daily_pnl:.2f} < "
                        f"-${self.max_daily_loss} — pausing 10 min"
                    )
                    await asyncio.sleep(600)
                    continue

                # Periodic monitoring scan (all candidates)
                now_mon = time.time()
                if now_mon - self._last_monitor_scan >= self._monitor_interval:
                    self._last_monitor_scan = now_mon
                    try:
                        await self._monitor_all_candidates()
                    except Exception as e:
                        logger.error(f"MM MONITOR ERROR: {type(e).__name__}: {e}")

                # Check each active pair
                for token in list(self.active_pairs.keys()):
                    state = self.active_pairs[token]

                    # Check for fills
                    await self._check_fills(token, state)

                    # Refresh stale quotes
                    if not self.observation_mode:
                        await self._refresh_quotes(token, state)

                # Hourly summary
                hourly_log_counter += 1
                if hourly_log_counter >= int(3600 / max(self.check_interval, 0.5)):
                    hourly_log_counter = 0
                    elapsed_hours = (time.time() - self._start_time) / 3600
                    fills_per_hour = self.total_fills / max(elapsed_hours, 0.01)
                    profit_per_hour = self.total_profit_usd / max(elapsed_hours, 0.01)
                    logger.info(
                        f"MM HOURLY: {len(self.active_pairs)} pairs | "
                        f"{self.total_fills} fills ({fills_per_hour:.1f}/hr) | "
                        f"${self.total_profit_usd:.2f} total (${profit_per_hour:.2f}/hr) | "
                        f"${sum(p.order_size_usd * p.sides for p in self.active_pairs.values()):.0f} deployed"
                    )

                # Periodic rescan
                rescan_counter += 1
                if rescan_counter >= int(self.rescan_interval_min * 60 / max(self.check_interval, 0.5)):
                    rescan_counter = 0
                    asyncio.create_task(self._background_rescan())

            except Exception as e:
                logger.error(f"MM LOOP ERROR: {type(e).__name__}: {e}")

            # Heartbeat logging (runs every iteration, not just on error)
            self._heartbeat_loop_count += 1
            now_hb = time.time()
            if now_hb - self._last_heartbeat >= self._heartbeat_interval:
                mode = "observation" if self.observation_mode else "live"
                logger.info(
                    f"MM HEARTBEAT: "
                    f"monitoring={len(self._candidates)} | "
                    f"trading={len(self.active_pairs)} | "
                    f"capital=${self.deployable_capital:.0f} USDC | "
                    f"fills={self.total_fills} | "
                    f"profit=${self.total_profit_usd:.4f} | "
                    f"mode={mode} | "
                    f"missed={self.missed_opportunities} opps, "
                    f"${self.missed_profit_usd:.2f} left on table"
                )
                self._heartbeat_loop_count = 0
                self._heartbeat_state_changes = 0
                self._last_heartbeat = now_hb

            await asyncio.sleep(self.check_interval)

    async def _check_fills(self, token: str, state: PairState):
        """Check if any maker orders have filled via direct REST."""
        import hmac as _hmac, hashlib as _hashlib

        api_key = getattr(self.client, '_api_key', '')
        api_secret = getattr(self.client, '_api_secret', '')
        if not api_key:
            return

        mexc_sym = state.usdc_symbol.replace("/", "")

        # Check buy order
        if state.buy_order_id:
            try:
                status_data = await self._get_order_status_rest(
                    mexc_sym, state.buy_order_id, api_key, api_secret
                )
                if status_data:
                    order_status = status_data.get("status", "")
                    if order_status == "FILLED":
                        fill_time = time.time() - (state.buy_posted_at or time.time())
                        logger.info(
                            f"MM FILL: {token} buy@{float(state.buy_price):.6f} USDC "
                            f"(waited {fill_time:.0f}s)"
                        )
                        # Build a minimal OrderResult for the fill handler
                        from datetime import datetime as _dt
                        fill_result = OrderResult(
                            exchange="mexc", symbol=state.usdc_symbol,
                            order_id=state.buy_order_id, client_order_id=state.buy_order_id,
                            status=OrderStatus.FILLED,
                            side=OrderSide.BUY, order_type=OrderType.LIMIT_MAKER,
                            requested_quantity=state.buy_qty or Decimal('0'),
                            filled_quantity=Decimal(str(status_data.get("executedQty", "0"))),
                            average_fill_price=state.buy_price,
                            fee_amount=Decimal('0'), fee_currency="USDC",
                            timestamp=_dt.utcnow(), raw_response=status_data,
                        )
                        state.buy_order_id = None
                        await self._on_maker_fill(token, 'buy', fill_result, state, fill_time)
                    elif order_status == "PARTIALLY_FILLED":
                        filled = status_data.get("executedQty", "0")
                        logger.info(
                            f"MM PARTIAL: {token} buy — "
                            f"{filled}/{float(state.buy_qty):.4f} filled"
                        )
                    elif order_status in ("CANCELED", "CANCELLED"):
                        logger.info(f"MM CANCELLED: {token} buy order externally cancelled")
                        state.buy_order_id = None
                        if not self.observation_mode:
                            await self._post_maker_orders(token)
            except Exception as e:
                logger.debug(f"MM fill check {token} buy: {e}")

        # Check sell order
        if state.sell_order_id:
            try:
                status_data = await self._get_order_status_rest(
                    mexc_sym, state.sell_order_id, api_key, api_secret
                )
                if status_data:
                    order_status = status_data.get("status", "")
                    if order_status == "FILLED":
                        fill_time = time.time() - (state.sell_posted_at or time.time())
                        logger.info(
                            f"MM FILL: {token} sell@{float(state.sell_price):.6f} USDC "
                            f"(waited {fill_time:.0f}s)"
                        )
                        from datetime import datetime as _dt
                        fill_result = OrderResult(
                            exchange="mexc", symbol=state.usdc_symbol,
                            order_id=state.sell_order_id, client_order_id=state.sell_order_id,
                            status=OrderStatus.FILLED,
                            side=OrderSide.SELL, order_type=OrderType.LIMIT_MAKER,
                            requested_quantity=state.sell_qty or Decimal('0'),
                            filled_quantity=Decimal(str(status_data.get("executedQty", "0"))),
                            average_fill_price=state.sell_price,
                            fee_amount=Decimal('0'), fee_currency="USDC",
                            timestamp=_dt.utcnow(), raw_response=status_data,
                        )
                        state.sell_order_id = None
                        await self._on_maker_fill(token, 'sell', fill_result, state, fill_time)
                    elif order_status == "PARTIALLY_FILLED":
                        filled = status_data.get("executedQty", "0")
                        logger.info(
                            f"MM PARTIAL: {token} sell — "
                            f"{filled}/{float(state.sell_qty):.4f} filled"
                        )
                    elif order_status in ("CANCELED", "CANCELLED"):
                        logger.info(f"MM CANCELLED: {token} sell order externally cancelled")
                        state.sell_order_id = None
                        if not self.observation_mode:
                            await self._post_maker_orders(token)
            except Exception as e:
                logger.debug(f"MM fill check {token} sell: {e}")

    async def _get_order_status_rest(self, mexc_sym: str, order_id: str,
                                      api_key: str, api_secret: str) -> Optional[dict]:
        """Query order status via direct REST."""
        import hmac as _hmac, hashlib as _hashlib

        def _sync_check():
            import requests as _requests
            ts = str(int(time.time() * 1000))
            params = {
                "symbol": mexc_sym,
                "orderId": order_id,
                "timestamp": ts,
                "recvWindow": "60000",
            }
            query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            sig = _hmac.new(
                api_secret.encode(), query.encode(), _hashlib.sha256
            ).hexdigest()
            url = f"https://api.mexc.com/api/v3/order?{query}&signature={sig}"
            resp = _requests.get(url, headers={
                "X-MEXC-APIKEY": api_key,
                "Content-Type": "application/json",
            }, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return None

        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(self._rest_executor, _sync_check),
                timeout=8.0
            )
        except Exception:
            return None

    async def _cancel_order_rest(self, symbol: str, order_id: str):
        """Cancel an order via direct REST."""
        import hmac as _hmac, hashlib as _hashlib

        api_key = getattr(self.client, '_api_key', '')
        api_secret = getattr(self.client, '_api_secret', '')
        if not api_key:
            return

        mexc_sym = symbol.replace("/", "")

        def _sync_cancel():
            import requests as _requests
            ts = str(int(time.time() * 1000))
            params = {
                "symbol": mexc_sym,
                "orderId": order_id,
                "timestamp": ts,
                "recvWindow": "60000",
            }
            query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            sig = _hmac.new(
                api_secret.encode(), query.encode(), _hashlib.sha256
            ).hexdigest()
            url = f"https://api.mexc.com/api/v3/order?{query}&signature={sig}"
            resp = _requests.delete(url, headers={
                "X-MEXC-APIKEY": api_key,
                "Content-Type": "application/json",
            }, timeout=5)
            return resp.status_code

        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(self._rest_executor, _sync_cancel),
                timeout=8.0
            )
        except Exception:
            pass

    async def _on_maker_fill(self, token: str, side: str, fill_result: OrderResult,
                              state: PairState, fill_time: float):
        """Maker order filled — instantly hedge on USDT pair."""
        hedge_side = 'sell' if side == 'buy' else 'buy'
        hedge_symbol = state.usdt_symbol
        fill_price = fill_result.average_fill_price or state.buy_price if side == 'buy' else state.sell_price
        fill_qty = fill_result.filled_quantity

        if not fill_qty or fill_qty <= 0:
            logger.warning(f"MM FILL: {token} {side} — zero quantity, skipping hedge")
            return

        # Fire hedge via direct REST
        hedge_start = time.time()
        try:
            # Get USDT book via REST
            book = await self._get_order_book_rest(hedge_symbol, depth=5)
            if not book or book == "BLOCKED":
                logger.error(f"MM HEDGE: {token} no USDT book — EXPOSED")
                await self._emergency_hedge(token, hedge_side, fill_qty, state)
                return

            if hedge_side == 'sell':
                hedge_price = book.best_bid
                hedge_price = hedge_price * (Decimal('1') - Decimal(str(self.hedge_slippage_bps)) / Decimal('10000'))
            else:
                hedge_price = book.best_ask
                hedge_price = hedge_price * (Decimal('1') + Decimal(str(self.hedge_slippage_bps)) / Decimal('10000'))

            prec = self._precision_cache.get(hedge_symbol, (8, 8))
            hedge_price = self._round_decimal(hedge_price, prec[0])
            hedge_qty = self._round_decimal(fill_qty, prec[1])

            if hedge_qty <= 0:
                logger.warning(f"MM HEDGE: {token} rounded qty=0 — skipping")
                return

            # Place hedge via direct REST (IOC order)
            hedge_result = await self._place_hedge_order_rest(
                hedge_symbol, hedge_side, hedge_price, hedge_qty, token
            )
            hedge_ms = (time.time() - hedge_start) * 1000

            if not hedge_result:
                logger.error(
                    f"MM HEDGE FAIL: {token} {side} filled on USDC but "
                    f"IOC {hedge_side} on USDT failed — EXPOSED"
                )
                await self._emergency_hedge(token, hedge_side, fill_qty, state)
                return

        except Exception as e:
            hedge_ms = (time.time() - hedge_start) * 1000
            logger.error(
                f"MM HEDGE FAIL: {token} {side} filled on USDC but "
                f"IOC {hedge_side} on USDT failed: {e} — EXPOSED"
            )
            await self._emergency_hedge(token, hedge_side, fill_qty, state)
            return

        # Calculate profit
        if hedge_result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            actual_hedge_price = hedge_result.average_fill_price or hedge_price
            hedge_qty_filled = hedge_result.filled_quantity or Decimal('0')

            if side == 'buy':
                # Bought TOKEN at fill_price USDC, sold at actual_hedge_price USDT
                profit_per_token = actual_hedge_price - (fill_price or Decimal('0'))
            else:
                # Sold TOKEN at fill_price USDC, bought back at actual_hedge_price USDT
                profit_per_token = (fill_price or Decimal('0')) - actual_hedge_price

            gross_profit = float(profit_per_token * hedge_qty_filled)
            hedge_fee = float(hedge_qty_filled * actual_hedge_price) * 0.0005  # 5 bps
            net_profit = gross_profit - hedge_fee

            # Check for adverse selection
            if fill_price and actual_hedge_price:
                if side == 'buy':
                    adverse_bps = float((fill_price - actual_hedge_price) / fill_price * 10000)
                else:
                    adverse_bps = float((actual_hedge_price - fill_price) / fill_price * 10000)
                if adverse_bps > self.adverse_threshold_bps:
                    state.adverse_fills += 1
                    logger.warning(
                        f"MM ADVERSE: {token} {side} — hedge {adverse_bps:.1f}bps worse than entry"
                    )

            state.fills_count += 1
            state.profit_usd += net_profit
            self.total_fills += 1
            self.total_profit_usd += net_profit
            self.total_hedge_cost_usd += hedge_fee
            self.daily_pnl += net_profit

            capture_bps = 0.0
            if fill_price and fill_price > 0:
                capture_bps = abs(float(profit_per_token / fill_price * 10000))

            logger.info(
                f"MM HEDGED: {token} {side}@{float(fill_price or 0):.6f} USDC "
                f"-> {hedge_side}@{float(actual_hedge_price):.6f} USDT "
                f"profit=${net_profit:.4f} ({capture_bps:.1f}bps) hedge={hedge_ms:.0f}ms"
            )

            # Persist to DB
            self._persist_trade({
                'token': token,
                'side': side,
                'maker_symbol': state.usdc_symbol,
                'maker_price': float(fill_price or 0),
                'maker_quantity': float(fill_qty),
                'maker_fill_ms': int(fill_time * 1000),
                'maker_fee_bps': 0.0,
                'hedge_symbol': hedge_symbol,
                'hedge_price': float(actual_hedge_price),
                'hedge_quantity': float(hedge_qty_filled),
                'hedge_fill_ms': int(hedge_ms),
                'hedge_fee_bps': 5.0,
                'hedge_status': hedge_result.status.value,
                'gross_profit_usd': gross_profit,
                'hedge_cost_usd': hedge_fee,
                'rebalance_cost_usd': 0.0,
                'net_profit_usd': net_profit,
                'capture_bps': capture_bps,
                'usdc_spread_bps': state.usdc_spread_bps,
                'usdt_spread_bps': state.usdt_spread_bps,
                'usdc_volume_24h': state.usdc_vol_24h,
                'order_size_usd': state.order_size_usd,
                'time_to_fill_seconds': fill_time,
                'status': 'hedged',
            })

            # Check adverse selection rate — remove pair if too high
            if state.fills_count >= 10:
                adverse_rate = state.adverse_fills / state.fills_count
                if adverse_rate > self.adverse_removal_pct:
                    logger.warning(
                        f"MM REMOVE: {token} — adverse rate {adverse_rate:.0%} "
                        f"({state.adverse_fills}/{state.fills_count}) > {self.adverse_removal_pct:.0%}"
                    )
                    await self._stop_pair(token)
                    return

        else:
            logger.warning(
                f"MM HEDGE FAIL: {token} {side} filled on USDC but "
                f"IOC {hedge_side} on USDT returned {hedge_result.status.value} — EXPOSED"
            )
            await self._emergency_hedge(token, hedge_side, fill_qty, state)

        # Repost maker order
        if not self.observation_mode:
            await self._post_maker_orders(token)

    async def _emergency_hedge(self, token: str, hedge_side: str,
                                quantity: Decimal, state: PairState):
        """Emergency hedge with more aggressive pricing via direct REST."""
        logger.warning(f"MM EMERGENCY HEDGE: {token} {hedge_side} qty={float(quantity):.6f}")
        hedge_symbol = state.usdt_symbol

        for attempt in range(3):
            try:
                book = await self._get_order_book_rest(hedge_symbol, depth=5)
                if not book or book == "BLOCKED":
                    await asyncio.sleep(0.5)
                    continue

                # Very aggressive price — 10 bps past the book
                if hedge_side == 'sell':
                    price = book.best_bid * (Decimal('1') - Decimal('0.001'))
                else:
                    price = book.best_ask * (Decimal('1') + Decimal('0.001'))

                prec = self._precision_cache.get(hedge_symbol, (8, 8))
                price = self._round_decimal(price, prec[0])
                qty = self._round_decimal(quantity, prec[1])

                result = await self._place_hedge_order_rest(
                    hedge_symbol, hedge_side, price, qty, token
                )

                if result and result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    logger.info(
                        f"MM EMERGENCY FILLED: {token} {hedge_side} attempt {attempt+1}"
                    )
                    return
                elif result:
                    logger.warning(
                        f"MM EMERGENCY attempt {attempt+1}: {result.status.value}"
                    )

            except Exception as e:
                logger.error(f"MM EMERGENCY attempt {attempt+1}: {e}")

            await asyncio.sleep(0.5)

        logger.error(
            f"MM EMERGENCY FAILED: {token} {hedge_side} — 3 attempts exhausted. "
            f"MANUAL INTERVENTION NEEDED"
        )

    async def _refresh_quotes(self, token: str, state: PairState):
        """Refresh maker orders if USDT price moved significantly."""
        try:
            current_usdt_mid = await self._get_usdt_mid(token)
            if current_usdt_mid is None or current_usdt_mid <= 0:
                return

            now = time.time()

            # Check buy order
            if state.buy_order_id and state.buy_ref_usdt_mid:
                move_bps = abs(current_usdt_mid - state.buy_ref_usdt_mid) / state.buy_ref_usdt_mid * 10000
                age = now - (state.buy_posted_at or now)

                if move_bps > self.refresh_threshold_bps or age > self.max_order_age_sec:
                    reason = f"moved {move_bps:.1f}bps" if move_bps > self.refresh_threshold_bps else f"aged {age:.0f}s"
                    await self._cancel_order_rest(state.usdc_symbol, state.buy_order_id)
                    logger.info(f"MM REFRESH: {token} buy cancelled — {reason}")
                    state.buy_order_id = None
                    await self._post_maker_orders(token)

            # Check sell order
            if state.sell_order_id and state.sell_ref_usdt_mid:
                move_bps = abs(current_usdt_mid - state.sell_ref_usdt_mid) / state.sell_ref_usdt_mid * 10000
                age = now - (state.sell_posted_at or now)

                if move_bps > self.refresh_threshold_bps or age > self.max_order_age_sec:
                    reason = f"moved {move_bps:.1f}bps" if move_bps > self.refresh_threshold_bps else f"aged {age:.0f}s"
                    await self._cancel_order_rest(state.usdc_symbol, state.sell_order_id)
                    logger.info(f"MM REFRESH: {token} sell cancelled — {reason}")
                    state.sell_order_id = None
                    await self._post_maker_orders(token)

        except Exception as e:
            logger.debug(f"MM REFRESH {token}: {e}")

    async def _get_usdt_mid(self, token: str) -> Optional[float]:
        """Get USDT pair mid price via direct REST."""
        mexc_sym = f"{token}USDT"

        def _sync_fetch():
            import requests as _requests
            url = f"https://api.mexc.com/api/v3/depth?symbol={mexc_sym}&limit=5"
            resp = _requests.get(url, timeout=5)
            return resp.json()

        try:
            loop = asyncio.get_running_loop()
            data = await asyncio.wait_for(
                loop.run_in_executor(self._rest_executor, _sync_fetch),
                timeout=8.0
            )
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            if bids and asks:
                return (float(bids[0][0]) + float(asks[0][0])) / 2
        except Exception:
            pass
        return None

    async def _stop_pair(self, token: str):
        """Cancel all orders and remove pair."""
        state = self.active_pairs.get(token)
        if not state:
            return

        for order_id in [state.buy_order_id, state.sell_order_id]:
            if order_id:
                await self._cancel_order_rest(state.usdc_symbol, order_id)

        del self.active_pairs[token]
        logger.info(f"MM STOPPED: {token} removed — {len(self.active_pairs)} pairs remaining")

    async def _background_rescan(self):
        """Periodically rescan for new candidates."""
        try:
            new_candidates = await self._scan_candidates()
            if new_candidates:
                self._candidates = new_candidates
                self._last_scan_time = time.time()
                logger.info(f"MM RESCAN: {len(new_candidates)} candidates updated")
        except Exception as e:
            logger.error(f"MM RESCAN failed: {e}")

    def _persist_trade(self, trade: dict):
        """Save trade to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cols = ", ".join(trade.keys())
            placeholders = ", ".join("?" * len(trade))
            conn.execute(
                f"INSERT INTO mm_trades ({cols}) VALUES ({placeholders})",
                tuple(trade.values()),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"MM persist error: {e}")

    async def stop(self):
        """Stop all pairs and cancel all orders."""
        self._running = False
        for token in list(self.active_pairs.keys()):
            await self._stop_pair(token)
        logger.info(
            f"MM STOPPED: total_fills={self.total_fills}, "
            f"total_profit=${self.total_profit_usd:.2f}"
        )

    def get_stats(self) -> dict:
        """Return stats for dashboard/API."""
        elapsed_hours = (time.time() - self._start_time) / 3600
        return {
            'active_pairs': len(self.active_pairs),
            'capital_deployed_usd': sum(
                p.order_size_usd * p.sides for p in self.active_pairs.values()
            ),
            'total_fills': self.total_fills,
            'total_profit_usd': round(self.total_profit_usd, 4),
            'total_hedge_cost_usd': round(self.total_hedge_cost_usd, 4),
            'fills_per_hour': round(self.total_fills / max(elapsed_hours, 0.01), 1),
            'profit_per_hour': round(self.total_profit_usd / max(elapsed_hours, 0.01), 4),
            'daily_pnl': round(self.daily_pnl, 4),
            'observation_mode': self.observation_mode,
            'monitoring_pairs': len(self._candidates),
            'missed_opportunities': self.missed_opportunities,
            'missed_profit_usd': round(self.missed_profit_usd, 4),
            'monitor_scans': self._monitor_scan_count,
            'pairs': [
                {
                    'token': p.token,
                    'order_size': p.order_size_usd,
                    'usdc_spread_bps': p.usdc_spread_bps,
                    'fills': p.fills_count,
                    'profit': round(p.profit_usd, 4),
                    'active_buy': p.buy_order_id is not None,
                    'active_sell': p.sell_order_id is not None,
                }
                for p in self.active_pairs.values()
            ],
        }

    @staticmethod
    def _safe_precision(raw) -> int:
        """Convert precision value to int decimal places."""
        if isinstance(raw, int) and raw >= 1:
            return raw
        t = float(raw)
        if t >= 1:
            return int(t)
        if t <= 0:
            return 8
        return max(0, -int(math.floor(math.log10(t))))

    @staticmethod
    async def _place_hedge_order_rest(self, symbol: str, side: str,
                                      price: Decimal, quantity: Decimal,
                                      token: str) -> Optional[OrderResult]:
        """Place an IOC hedge order via direct REST."""
        import hmac as _hmac, hashlib as _hashlib

        api_key = getattr(self.client, '_api_key', '')
        api_secret = getattr(self.client, '_api_secret', '')
        if not api_key:
            return None

        mexc_sym = symbol.replace("/", "")
        oid = f"mm_hedge_{token}_{side}_{uuid.uuid4().hex[:8]}"

        def _sync_place():
            import requests as _requests
            ts = str(int(time.time() * 1000))
            params = {
                "symbol": mexc_sym,
                "side": side.upper(),
                "type": "LIMIT",
                "quantity": str(float(quantity)),
                "price": str(float(price)),
                "newClientOrderId": oid,
                "timestamp": ts,
                "recvWindow": "60000",
            }
            query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            sig = _hmac.new(
                api_secret.encode(), query.encode(), _hashlib.sha256
            ).hexdigest()
            url = f"https://api.mexc.com/api/v3/order?{query}&signature={sig}"
            resp = _requests.post(url, headers={
                "X-MEXC-APIKEY": api_key,
                "Content-Type": "application/json",
            }, timeout=5)
            return resp.status_code, resp.json()

        try:
            loop = asyncio.get_running_loop()
            status_code, data = await asyncio.wait_for(
                loop.run_in_executor(self._rest_executor, _sync_place),
                timeout=8.0
            )

            if status_code == 200:
                from datetime import datetime as _dt
                order_id = data.get("orderId", oid)
                filled_qty = Decimal(str(data.get("executedQty", "0")))
                avg_price = Decimal(str(data.get("price", str(price))))
                status = OrderStatus.FILLED if filled_qty > 0 else OrderStatus.OPEN
                result = OrderResult(
                    exchange="mexc", symbol=symbol,
                    order_id=str(order_id), client_order_id=oid,
                    status=status,
                    side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    requested_quantity=quantity,
                    filled_quantity=filled_qty,
                    average_fill_price=avg_price,
                    fee_amount=Decimal('0'), fee_currency="USDT",
                    timestamp=_dt.utcnow(), raw_response=data,
                )
                logger.info(
                    f"MM HEDGE OK: {token} {side}@{float(price):.6f} USDT "
                    f"qty={float(filled_qty):.4f}/{float(quantity):.4f}"
                )
                return result
            else:
                error_code = data.get("code", 0)
                error_msg = data.get("msg", "")
                logger.error(
                    f"MM HEDGE REJECTED: {token} {side} — "
                    f"code={error_code} msg={error_msg}"
                )
                return None
        except Exception as e:
            logger.error(f"MM HEDGE ERROR: {token} {side}: {e}")
            return None

    @staticmethod
    def _round_decimal(value: Decimal, precision) -> Decimal:
        """Round a Decimal to given precision (rounds DOWN)."""
        p = float(precision)
        if 0 < p < 1:
            tick = Decimal(str(precision))
            return (value / tick).to_integral_value(rounding=ROUND_DOWN) * tick
        places = int(p)
        if places <= 0:
            return value.quantize(Decimal('1'), rounding=ROUND_DOWN)
        quant = Decimal(10) ** -places
        return value.quantize(quant, rounding=ROUND_DOWN)

    @staticmethod
    def _round_decimal_up(value: Decimal, precision) -> Decimal:
        """Round a Decimal UP to given precision (for sell orders)."""
        from decimal import ROUND_CEILING
        p = float(precision)
        if 0 < p < 1:
            tick = Decimal(str(precision))
            return (value / tick).to_integral_value(rounding=ROUND_CEILING) * tick
        places = int(p)
        if places <= 0:
            return value.quantize(Decimal('1'), rounding=ROUND_CEILING)
        quant = Decimal(10) ** -places
        return value.quantize(quant, rounding=ROUND_CEILING)
