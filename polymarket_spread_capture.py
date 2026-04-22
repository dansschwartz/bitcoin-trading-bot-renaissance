"""
polymarket_spread_capture.py — 0x8dxd-Style Dual Accumulation + Early Exit

Replicates the proven strategy from wallet 0x8dxd (122 windows, $8,860 profit):

Phase 1 — DUAL ACCUMULATION (T+5 to T-30):
  Buy BOTH YES and NO whenever either side drops below $0.48.
  No favorite/underdog distinction. Both sides checked independently.
  93% of windows end up hedged (both sides filled).

Phase 2 — EARLY EXIT (T-180 to T-10):
  Sell ALL shares of the losing side (the cheaper one).
  Recovers capital instead of letting it go to $0 at resolution.
  61% of windows see an early exit.

Phase 3 — SCALP (any time):
  If any individual fill is up 1000% AND $10+ profit, sell it.
  Rare but extremely profitable when it hits.

Phase 4 — RESOLUTION:
  Winning side pays $1.00 per share. Losing side already sold or $0.

Key metrics (from 0x8dxd analysis):
  - Hedge rate: 93% (both sides filled)
  - Average buy price: $0.47
  - Early exit rate: 61% of windows
  - Pair cost: ~$0.94
  - Win rate: 55.7% (68W / 54L)
  - Net P&L: $8,860 across 122 windows ($72.63/window avg)
"""

import asyncio
import json
import logging
import math
import os
import sqlite3
import time
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("spread_capture")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Calibrated from 0x8dxd's 122-window dataset
# ═══════════════════════════════════════════════════════════════

ASSETS = {
    "BTC": {"slug_5m": "btc-updown-5m", "slug_15m": "btc-updown-15m"},
    "ETH": {"slug_5m": "eth-updown-5m", "slug_15m": "eth-updown-15m"},
    "SOL": {"slug_5m": "sol-updown-5m", "slug_15m": "sol-updown-15m"},
    "XRP": {"slug_15m": "xrp-updown-15m"},
}

TIMEFRAMES = {
    "5m": {"seconds": 300, "alignment": 300},
    "15m": {"seconds": 900, "alignment": 900},
}

# ── Phase 1: Dual Accumulation ──
BUY_THRESHOLD = 0.48              # Buy either side at or below this price
FILL_SIZE_USD = 4.00              # $ per fill (conservative start; 0x8dxd uses ~$37)
COOLDOWN_PER_SIDE = 3.0           # Seconds between fills on the same side
PRICE_IMPROVEMENT = 0.02          # Must improve by $0.02 before re-buying same side
MAX_FILLS_PER_SIDE = 15           # Max fills per side per window
ACCUMULATION_START = 5            # Seconds after window open to begin
ACCUMULATION_END_BEFORE = 30      # Stop accumulating this many seconds before window end

# ── Phase 2: Early Exit ──
EARLY_EXIT_SECONDS_BEFORE = 180   # Start checking 3 minutes before close
EARLY_EXIT_END_BEFORE = 10        # Stop 10 seconds before close
EARLY_EXIT_MIN_PRICE = 0.05       # Don't sell if losing side < $0.05 (not worth it)

# ── Phase 3: Scalp ──
SCALP_MULTIPLIER = 11.0           # Sell if fill is up 1000%+
SCALP_MIN_PROFIT = 10.00          # AND profit >= $10

# ── Safety Limits ──
MAX_DAILY_LOSS = 50.00
MAX_WINDOWS_PER_DAY = 300
MAX_EXPOSURE_PER_WINDOW = 100.00  # Max $ deployed per window (both sides)
MAX_GLOBAL_EXPOSURE = 500.00      # Total $ across ALL active windows

DB_PATH = "data/renaissance_bot.db"
SECRETS_PATH = "config/polymarket_secrets.json"
GAMMA_BASE = "https://gamma-api.polymarket.com"


@dataclass
class WindowPosition:
    """Tracks position state for a single market window."""
    asset: str
    timeframe: str
    window_start: int
    slug: str
    market_id: str = ""

    # Token IDs (from Gamma API)
    yes_token_id: str = ""
    no_token_id: str = ""

    # YES side accumulation
    yes_shares: float = 0.0
    yes_cost: float = 0.0
    yes_orders: int = 0
    last_yes_buy_time: float = 0.0
    last_yes_buy_price: float = 0.0

    # NO side accumulation
    no_shares: float = 0.0
    no_cost: float = 0.0
    no_orders: int = 0
    last_no_buy_time: float = 0.0
    last_no_buy_price: float = 0.0

    # Sells (early exit + scalp)
    yes_sold_shares: float = 0.0
    yes_sold_revenue: float = 0.0
    no_sold_shares: float = 0.0
    no_sold_revenue: float = 0.0

    # Early exit tracking
    early_exit_done: bool = False
    early_exit_side: str = ""      # Which side was sold early
    early_exit_price: float = 0.0
    early_exit_recovered: float = 0.0

    # Individual fill tracking for scalping
    # Each: {"side": str, "shares": float, "cost_per_share": float, "total_cost": float}
    fills: list = field(default_factory=list)

    # Chainlink resolution tracking
    chainlink_start_price: float = 0.0

    # Status
    status: str = "active"   # active, resolved, error
    resolution: str = ""     # UP, DOWN
    pnl: float = 0.0

    @property
    def avg_yes_price(self) -> float:
        return self.yes_cost / self.yes_shares if self.yes_shares > 0 else 0

    @property
    def avg_no_price(self) -> float:
        return self.no_cost / self.no_shares if self.no_shares > 0 else 0

    @property
    def pair_cost(self) -> float:
        """Average cost for one YES + one NO share."""
        if self.yes_shares == 0 or self.no_shares == 0:
            return 1.0
        return self.avg_yes_price + self.avg_no_price

    @property
    def total_cost(self) -> float:
        return self.yes_cost + self.no_cost - self.yes_sold_revenue - self.no_sold_revenue

    @property
    def net_yes_shares(self) -> float:
        return self.yes_shares - self.yes_sold_shares

    @property
    def net_no_shares(self) -> float:
        return self.no_shares - self.no_sold_shares

    @property
    def is_hedged(self) -> bool:
        """True if we hold shares on both sides."""
        return self.net_yes_shares > 0 and self.net_no_shares > 0

    @property
    def guaranteed_profit(self) -> float:
        """Profit if the WORSE side wins. Negative = not yet hedged."""
        min_shares = min(self.net_yes_shares, self.net_no_shares)
        return min_shares - self.total_cost

    @property
    def is_arb(self) -> bool:
        return self.guaranteed_profit > 0


class SpreadCaptureEngine:
    """
    0x8dxd-style spread capture on Polymarket direction markets.

    For each 5m/15m window:
    Phase 1 (T+5 to T-30):  Dual accumulation — buy both sides below $0.48
    Phase 2 (T-180 to T-10): Early exit — sell losing side to recover capital
    Phase 3 (any time):      Scalp — sell individual fills at 1000%+ gain
    Phase 4 (T+30):          Resolution — winning side pays $1.00
    """

    def __init__(self, rtds):
        self._rtds = rtds
        self._clob_client = None
        self._running = False
        self._positions: Dict[str, WindowPosition] = {}
        self._pending_orders: List[dict] = []
        self._entry_sem = asyncio.Semaphore(5)
        self._daily_pnl = 0.0
        self._daily_date = ""
        self._daily_windows = 0
        self._total_windows_resolved = 0

        self._init_db()
        self._init_clob()
        self._load_active_windows()

    def _get_global_exposure(self) -> float:
        return sum(p.total_cost for p in self._positions.values() if p.status == "active")

    # ═══════════════════════════════════════════════════════════
    # DB SETUP
    # ═══════════════════════════════════════════════════════════

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS spread_capture_windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                slug TEXT NOT NULL,
                window_start INTEGER NOT NULL,

                yes_shares REAL DEFAULT 0,
                yes_cost REAL DEFAULT 0,
                yes_avg_price REAL DEFAULT 0,
                yes_orders INTEGER DEFAULT 0,

                no_shares REAL DEFAULT 0,
                no_cost REAL DEFAULT 0,
                no_avg_price REAL DEFAULT 0,
                no_orders INTEGER DEFAULT 0,

                yes_sold_shares REAL DEFAULT 0,
                yes_sold_revenue REAL DEFAULT 0,
                no_sold_shares REAL DEFAULT 0,
                no_sold_revenue REAL DEFAULT 0,

                pair_cost REAL,
                total_deployed REAL DEFAULT 0,
                guaranteed_profit REAL,
                is_arb INTEGER DEFAULT 0,

                chainlink_start REAL,
                chainlink_end REAL,
                resolution TEXT,

                pnl REAL,
                pnl_pct REAL,

                max_phase INTEGER DEFAULT 0,
                phase2_triggered INTEGER DEFAULT 0,

                status TEXT DEFAULT 'active',

                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                resolved_at TEXT,

                UNIQUE(asset, timeframe, window_start)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS spread_capture_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_slug TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                shares REAL NOT NULL,
                amount_usd REAL NOT NULL,
                order_id TEXT,
                phase INTEGER,
                timestamp REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add columns for v2 fields (safe if they already exist)
        for col, typedef in [
            ("yes_token_id", "TEXT DEFAULT ''"),
            ("no_token_id", "TEXT DEFAULT ''"),
            ("market_id", "TEXT DEFAULT ''"),
            ("favorite_side", "TEXT DEFAULT ''"),
            ("phase2_fills", "INTEGER DEFAULT 0"),
            ("early_exit_side", "TEXT DEFAULT ''"),
            ("early_exit_price", "REAL DEFAULT 0"),
            ("early_exit_recovered", "REAL DEFAULT 0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE spread_capture_windows ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError as e:
                logger.warning(f"conn.execute failed: {e}")

        conn.commit()
        conn.close()
        logger.info("Spread capture DB initialized (v2 dual-accumulation)")

    # ═══════════════════════════════════════════════════════════
    # CLOB CLIENT
    # ═══════════════════════════════════════════════════════════

    def _init_clob(self):
        try:
            from py_clob_client.client import ClobClient

            if not os.path.exists(SECRETS_PATH):
                logger.error(f"No secrets at {SECRETS_PATH}")
                return

            with open(SECRETS_PATH) as f:
                secrets = json.load(f)

            proxy_addr = secrets.get("proxy_wallet_address", "")
            init_kwargs = {
                "host": "https://clob.polymarket.com",
                "chain_id": 137,
                "key": secrets["private_key"],
            }
            if proxy_addr:
                init_kwargs["signature_type"] = 1
                init_kwargs["funder"] = proxy_addr
                logger.info(f"[SC] Using proxy wallet: {proxy_addr[:10]}...{proxy_addr[-6:]}")

            self._clob_client = ClobClient(**init_kwargs)
            api_creds = self._clob_client.create_or_derive_api_creds()
            self._clob_client.set_api_creds(api_creds)
            server_time = self._clob_client.get_server_time()
            logger.info(f"[SC] CLOB client initialized (server_time={server_time})")
        except Exception as e:
            logger.error(f"CLOB init failed: {e}")

    # ═══════════════════════════════════════════════════════════
    # POSITION PERSISTENCE
    # ═══════════════════════════════════════════════════════════

    def _load_active_windows(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM spread_capture_windows WHERE status = 'active'"
            ).fetchall()
            conn.close()

            loaded = 0
            for row in rows:
                slug = row["slug"]
                if slug in self._positions:
                    continue

                pos = WindowPosition(
                    asset=row["asset"],
                    timeframe=row["timeframe"],
                    window_start=row["window_start"],
                    slug=slug,
                    market_id=row["market_id"] or "" if "market_id" in row.keys() else "",
                    yes_token_id=row["yes_token_id"] or "" if "yes_token_id" in row.keys() else "",
                    no_token_id=row["no_token_id"] or "" if "no_token_id" in row.keys() else "",
                    yes_shares=row["yes_shares"] or 0,
                    yes_cost=row["yes_cost"] or 0,
                    yes_orders=row["yes_orders"] or 0,
                    no_shares=row["no_shares"] or 0,
                    no_cost=row["no_cost"] or 0,
                    no_orders=row["no_orders"] or 0,
                    yes_sold_shares=row["yes_sold_shares"] or 0,
                    yes_sold_revenue=row["yes_sold_revenue"] or 0,
                    no_sold_shares=row["no_sold_shares"] or 0,
                    no_sold_revenue=row["no_sold_revenue"] or 0,
                    chainlink_start_price=row["chainlink_start"] or 0,
                    status="active",
                )
                # Restore early exit state from DB columns
                if "early_exit_side" in row.keys() and row["early_exit_side"]:
                    pos.early_exit_done = True
                    pos.early_exit_side = row["early_exit_side"]
                    pos.early_exit_price = row["early_exit_price"] or 0
                    pos.early_exit_recovered = row["early_exit_recovered"] or 0

                self._positions[slug] = pos
                loaded += 1

            if loaded:
                logger.info(
                    f"[SC] RESTORED {loaded} active positions from DB | "
                    f"Slugs: {list(self._positions.keys())[:5]}..."
                )
        except Exception as e:
            logger.warning(f"[SC] Failed to load active windows: {e}")

    def _save_active_window(self, pos: WindowPosition):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                INSERT OR REPLACE INTO spread_capture_windows
                (asset, timeframe, slug, window_start,
                 yes_shares, yes_cost, yes_avg_price, yes_orders,
                 no_shares, no_cost, no_avg_price, no_orders,
                 yes_sold_shares, yes_sold_revenue,
                 no_sold_shares, no_sold_revenue,
                 pair_cost, total_deployed, guaranteed_profit, is_arb,
                 chainlink_start, max_phase, phase2_triggered, status,
                 yes_token_id, no_token_id, market_id,
                 early_exit_side, early_exit_price, early_exit_recovered)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos.asset, pos.timeframe, pos.slug, pos.window_start,
                pos.yes_shares, pos.yes_cost, pos.avg_yes_price, pos.yes_orders,
                pos.no_shares, pos.no_cost, pos.avg_no_price, pos.no_orders,
                pos.yes_sold_shares, pos.yes_sold_revenue,
                pos.no_sold_shares, pos.no_sold_revenue,
                pos.pair_cost, pos.total_cost, pos.guaranteed_profit,
                1 if pos.is_arb else 0,
                pos.chainlink_start_price,
                2 if pos.is_hedged else 1,
                1 if pos.is_hedged else 0,
                "active",
                pos.yes_token_id, pos.no_token_id, pos.market_id,
                pos.early_exit_side, pos.early_exit_price, pos.early_exit_recovered,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"[SC] Active window save error: {e}")

    # ═══════════════════════════════════════════════════════════
    # PENDING ORDER VERIFICATION
    # ═══════════════════════════════════════════════════════════

    async def _process_pending_orders(self):
        """Check all pending orders for fills. Credit verified fills, cancel stale ones."""
        if not self._pending_orders:
            return

        still_pending = []
        for order in self._pending_orders:
            order_id = order["order_id"]
            slug = order["slug"]
            order["checks"] += 1

            pos = self._positions.get(slug)
            if not pos or pos.status != "active":
                await asyncio.to_thread(self._cancel_order_sync, order_id)
                continue

            resp = await asyncio.to_thread(self._get_order_sync, order_id)
            if not resp:
                if order["checks"] >= 4:
                    continue
                still_pending.append(order)
                continue

            status = str(resp.get("status", "")).upper()
            size_matched = float(
                resp.get("sizeMatched", 0) or resp.get("size_matched", 0) or 0
            )
            filled_price = float(resp.get("price", 0) or 0)

            if status in ("MATCHED", "FILLED") or size_matched > 0:
                actual_shares = size_matched if size_matched > 0 else float(
                    resp.get("size", 0) or 0
                )
                actual_price = filled_price if filled_price > 0 else order["price"]
                amount = actual_price * actual_shares

                if order["is_buy"]:
                    if order["side"] == "YES":
                        pos.yes_shares += actual_shares
                        pos.yes_cost += amount
                        pos.yes_orders += 1
                    else:
                        pos.no_shares += actual_shares
                        pos.no_cost += amount
                        pos.no_orders += 1
                    # Track for scalping
                    pos.fills.append({
                        "side": order["side"],
                        "shares": actual_shares,
                        "cost_per_share": actual_price,
                        "total_cost": amount,
                    })
                    self._record_fill(
                        slug, order["side"], actual_price, actual_shares,
                        amount, order_id, order["phase"]
                    )
                else:
                    revenue = actual_price * actual_shares
                    if order["side"] == "YES":
                        pos.yes_sold_shares += actual_shares
                        pos.yes_sold_revenue += revenue
                    else:
                        pos.no_sold_shares += actual_shares
                        pos.no_sold_revenue += revenue
                    self._record_fill(
                        slug, f"SELL_{order['side']}", actual_price,
                        actual_shares, revenue, order_id, order["phase"]
                    )

                self._save_active_window(pos)
                action = "FILL" if order["is_buy"] else "SELL"
                logger.info(
                    f"[SC] VERIFIED {action}: {order_id[:12]}... | "
                    f"{actual_shares:.1f} {order['side']} @ ${actual_price:.2f} "
                    f"(${amount:.2f})"
                )

            elif order["checks"] >= 3:
                logger.info(
                    f"[SC] Order {order_id[:12]}... unfilled after "
                    f"{order['checks']} checks — cancelling"
                )
                await asyncio.to_thread(self._cancel_order_sync, order_id)
            else:
                still_pending.append(order)

        self._pending_orders = still_pending

    # ═══════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════

    async def run(self):
        self._running = True
        logger.info(
            f"[SC] Spread capture engine STARTED (v2 dual-accumulation)\n"
            f"  Assets: {list(ASSETS.keys())}\n"
            f"  Timeframes: {list(TIMEFRAMES.keys())}\n"
            f"  Buy threshold: ${BUY_THRESHOLD}\n"
            f"  Fill size: ${FILL_SIZE_USD}/fill\n"
            f"  Early exit: T-{EARLY_EXIT_SECONDS_BEFORE}s (min ${EARLY_EXIT_MIN_PRICE})"
        )

        while self._running:
            try:
                now = time.time()

                # Daily reset
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if today != self._daily_date:
                    if self._daily_date:
                        logger.info(
                            f"[SC] Daily reset. Yesterday: "
                            f"{self._daily_windows} windows, ${self._daily_pnl:+.2f}"
                        )
                    self._daily_date = today
                    self._daily_pnl = 0.0
                    self._daily_windows = 0

                if self._daily_pnl <= -MAX_DAILY_LOSS:
                    logger.warning(f"[SC] Daily loss limit (${self._daily_pnl:.2f}). Paused.")
                    await asyncio.sleep(60)
                    continue

                # ── STEP 0: Verify pending orders ──
                await self._process_pending_orders()

                # ── STEP 1: Process EXISTING positions ──
                expired_positions = []
                tradeable_positions = []
                for slug, pos in list(self._positions.items()):
                    if pos.status != "active":
                        continue
                    window_seconds = TIMEFRAMES[pos.timeframe]["seconds"]
                    elapsed = now - pos.window_start
                    if elapsed > window_seconds + 30:
                        expired_positions.append(pos)
                    else:
                        tradeable_positions.append(pos)

                for pos in expired_positions:
                    await self._resolve_window(pos)

                # Batch fetch CLOB prices for all tradeable positions
                if tradeable_positions:
                    price_results = await asyncio.gather(
                        *(self._get_clob_prices(p) for p in tradeable_positions),
                        return_exceptions=True,
                    )
                else:
                    price_results = []

                for pos, prices in zip(tradeable_positions, price_results):
                    if isinstance(prices, Exception):
                        continue
                    yes_price, no_price = prices
                    if yes_price is None or no_price is None:
                        continue

                    actual_elapsed = time.time() - pos.window_start
                    window_seconds = TIMEFRAMES[pos.timeframe]["seconds"]

                    # ── PHASE 2: EARLY EXIT (T-180 to T-10) ──
                    exit_start = window_seconds - EARLY_EXIT_SECONDS_BEFORE
                    exit_end = window_seconds - EARLY_EXIT_END_BEFORE
                    if exit_start <= actual_elapsed < exit_end:
                        await self._early_exit(pos, yes_price, no_price)

                    # ── PHASE 1: DUAL ACCUMULATION (T+5 to T-30) ──
                    accum_end = window_seconds - ACCUMULATION_END_BEFORE
                    if ACCUMULATION_START <= actual_elapsed < accum_end:
                        if yes_price <= BUY_THRESHOLD:
                            await self._accumulate(pos, "YES", yes_price)
                        if no_price <= BUY_THRESHOLD:
                            await self._accumulate(pos, "NO", no_price)

                    # ── PHASE 3: SCALP (any time during accumulation window) ──
                    if ACCUMULATION_START <= actual_elapsed < accum_end:
                        await self._check_scalps(pos, yes_price, no_price)

                    # ── Periodic status log ──
                    if int(now) % 30 < 2:
                        hedge_tag = "HEDGED" if pos.is_hedged else "NAKED"
                        logger.info(
                            f"[SC] {pos.asset} {pos.timeframe} | "
                            f"YES: {pos.net_yes_shares:.1f}sh@${pos.avg_yes_price:.3f} | "
                            f"NO: {pos.net_no_shares:.1f}sh@${pos.avg_no_price:.3f} | "
                            f"CLOB: Y${yes_price:.2f}/N${no_price:.2f} | "
                            f"Pair: ${pos.pair_cost:.3f} | {hedge_tag} | "
                            f"Fills: Y{pos.yes_orders}/N{pos.no_orders}"
                        )

                # Check resolutions every 30s
                if int(now) % 30 == 0:
                    await self._check_resolutions()

                # ── STEP 2: Enter NEW windows ──
                global_exposure = self._get_global_exposure()
                entry_tasks = []

                for asset, config in ASSETS.items():
                    for tf_name in config:
                        if not tf_name.startswith("slug_"):
                            continue
                        tf = tf_name.replace("slug_", "")
                        if tf not in TIMEFRAMES:
                            continue
                        tf_config = TIMEFRAMES[tf]
                        alignment = tf_config["alignment"]
                        window_start = int(math.floor(now / alignment) * alignment)
                        slug = f"{config[tf_name]}-{window_start}"
                        elapsed = now - window_start

                        window_seconds = tf_config["seconds"]
                        if slug not in self._positions and elapsed < window_seconds - ACCUMULATION_END_BEFORE:
                            if elapsed >= ACCUMULATION_START:
                                if global_exposure >= MAX_GLOBAL_EXPOSURE:
                                    continue
                                entry_tasks.append(
                                    self._enter_window(asset, tf, window_start, slug, config)
                                )
                                global_exposure += FILL_SIZE_USD * 2

                if entry_tasks:
                    await asyncio.gather(*entry_tasks, return_exceptions=True)

                # Heartbeat
                if int(now) % 30 < 2:
                    active = len([p for p in self._positions.values() if p.status == "active"])
                    hedged = len([p for p in self._positions.values()
                                  if p.status == "active" and p.is_hedged])
                    logger.info(
                        f"[SC] Heartbeat: {active} active ({hedged} hedged) | "
                        f"${global_exposure:.2f} deployed | "
                        f"Daily: ${self._daily_pnl:+.2f} ({self._daily_windows}w)"
                    )

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                logger.info("[SC] Engine cancelled — shutting down")
                return
            except Exception as e:
                logger.error(f"[SC] Loop error: {e}")
                await asyncio.sleep(5)

    # ═══════════════════════════════════════════════════════════
    # WINDOW ENTRY — Just discover market, no directional bet
    # ═══════════════════════════════════════════════════════════

    async def _enter_window(self, asset: str, timeframe: str,
                             window_start: int, slug: str, config: dict):
        async with self._entry_sem:
            await self._enter_window_inner(asset, timeframe, window_start, slug, config)

    async def _enter_window_inner(self, asset: str, timeframe: str,
                                    window_start: int, slug: str, config: dict):
        self._rtds.record_window_start(asset, window_start)

        market_data = await self._fetch_market(slug)
        if not market_data:
            logger.info(f"[SC] {asset} {timeframe}: market {slug} not found")
            return

        pos = WindowPosition(
            asset=asset,
            timeframe=timeframe,
            window_start=window_start,
            slug=slug,
            market_id=market_data.get("market_id", ""),
            yes_token_id=market_data.get("token_id_yes", ""),
            no_token_id=market_data.get("token_id_no", ""),
            chainlink_start_price=self._rtds.get_chainlink_price(asset) or 0,
        )

        self._positions[slug] = pos
        self._daily_windows += 1
        self._save_active_window(pos)

        crowd_yes = market_data.get("crowd_yes", 0.50)
        logger.info(
            f"[SC] *** WINDOW OPENED: {asset} {timeframe} ***\n"
            f"  Slug: {slug}\n"
            f"  Crowd: YES=${crowd_yes:.3f} NO=${1-crowd_yes:.3f}\n"
            f"  Buy threshold: ${BUY_THRESHOLD} (both sides)\n"
            f"  Chainlink: ${pos.chainlink_start_price:.2f}"
        )

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: DUAL ACCUMULATION
    # ═══════════════════════════════════════════════════════════

    async def _accumulate(self, pos: WindowPosition, side: str, current_price: float):
        """
        Buy either side when it drops to or below BUY_THRESHOLD.
        Both sides checked independently every poll.
        Guards: cooldown, price improvement, max fills, exposure caps.
        """
        now = time.time()

        # ── Per-side state ──
        if side == "YES":
            orders = pos.yes_orders
            last_buy_time = pos.last_yes_buy_time
            last_buy_price = pos.last_yes_buy_price
        else:
            orders = pos.no_orders
            last_buy_time = pos.last_no_buy_time
            last_buy_price = pos.last_no_buy_price

        # ── Guards ──

        # Max fills per side
        if orders >= MAX_FILLS_PER_SIDE:
            return

        # Per-window exposure
        if pos.total_cost >= MAX_EXPOSURE_PER_WINDOW:
            return

        # Global exposure
        if self._get_global_exposure() >= MAX_GLOBAL_EXPOSURE:
            return

        # Cooldown per side
        if now - last_buy_time < COOLDOWN_PER_SIDE:
            return

        # Price improvement: only rebuy if price dropped by at least $0.02
        if last_buy_price > 0:
            if current_price >= last_buy_price - PRICE_IMPROVEMENT:
                return

        # ── Sizing ──
        buy_amount = FILL_SIZE_USD
        remaining = MAX_EXPOSURE_PER_WINDOW - pos.total_cost
        global_remaining = MAX_GLOBAL_EXPOSURE - self._get_global_exposure()
        buy_amount = min(buy_amount, remaining, global_remaining)

        if buy_amount < 1.00:
            return

        shares = buy_amount / current_price if current_price > 0 else 0

        # CLOB 5-share minimum
        if shares < 5:
            min_cost = 5 * current_price
            if min_cost <= min(remaining, global_remaining):
                shares = 5
                buy_amount = min_cost
            else:
                return

        order_id = await self._place_order(pos, side, current_price, shares, phase=1)
        if not order_id:
            return

        # Update per-side cooldown tracking
        if side == "YES":
            pos.last_yes_buy_time = now
            pos.last_yes_buy_price = current_price
        else:
            pos.last_no_buy_time = now
            pos.last_no_buy_price = current_price

        logger.info(
            f"[SC] ACCUMULATE: {pos.asset} {pos.timeframe} | "
            f"{side} @ ${current_price:.3f} ({shares:.1f}sh, ${buy_amount:.2f}) | "
            f"Fills: Y{pos.yes_orders}/N{pos.no_orders} | "
            f"{'HEDGED' if pos.is_hedged else 'one-side'}"
        )

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: EARLY EXIT — Sell losing side before resolution
    # ═══════════════════════════════════════════════════════════

    async def _early_exit(self, pos: WindowPosition,
                           yes_price: float, no_price: float):
        """
        At T-180, identify the losing side and sell ALL shares.
        Recovers capital vs letting it go to $0 at resolution.
        0x8dxd does this in 61% of windows.
        """
        if pos.early_exit_done:
            return

        # Must have shares on both sides to identify loser
        if pos.net_yes_shares <= 0 or pos.net_no_shares <= 0:
            return

        # Identify losing side (cheaper one is likely to lose)
        if yes_price < no_price:
            losing_side = "YES"
            losing_price = yes_price
            losing_shares = pos.net_yes_shares
        else:
            losing_side = "NO"
            losing_price = no_price
            losing_shares = pos.net_no_shares

        # Not worth selling if price too low (gas + spread > recovery)
        if losing_price < EARLY_EXIT_MIN_PRICE:
            logger.debug(
                f"[SC] Early exit skip: {pos.asset} {losing_side} "
                f"@ ${losing_price:.3f} < ${EARLY_EXIT_MIN_PRICE} threshold"
            )
            return

        pos.early_exit_done = True

        # Sell 95% of shares (CLOB safety margin for rounding)
        sell_shares = round(losing_shares * 0.95, 2)
        if sell_shares < 5:
            logger.debug(
                f"[SC] Early exit skip: {pos.asset} only "
                f"{sell_shares:.1f} shares after margin"
            )
            return

        # Sell at current price (willing to cross spread to get out)
        sell_price = round(losing_price, 2)
        if sell_price < 0.01:
            sell_price = 0.01

        order_id = await self._sell_order(pos, losing_side, sell_price, sell_shares, phase=2)

        if order_id:
            expected_recovery = sell_shares * sell_price
            pos.early_exit_side = losing_side
            pos.early_exit_price = sell_price
            pos.early_exit_recovered = expected_recovery
            self._save_active_window(pos)

            logger.info(
                f"[SC] *** EARLY EXIT: {pos.asset} {pos.timeframe} | "
                f"Sold {sell_shares:.1f} {losing_side} @ ${sell_price:.2f} | "
                f"Recovered ${expected_recovery:.2f} vs $0 at resolution"
            )
        else:
            # Reset so we can retry next poll
            pos.early_exit_done = False
            logger.warning(
                f"[SC] Early exit FAILED: {pos.asset} {pos.timeframe} | "
                f"Tried {sell_shares:.1f} {losing_side} @ ${sell_price:.2f}"
            )

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: SCALP — Sell individual fills at 1000%+ gain
    # ═══════════════════════════════════════════════════════════

    async def _check_scalps(self, pos: WindowPosition,
                             yes_price: float, no_price: float):
        if not pos.fills:
            return

        remaining = []
        for fill in pos.fills:
            current_price = yes_price if fill["side"] == "YES" else no_price
            if current_price <= 0 or fill["cost_per_share"] <= 0:
                remaining.append(fill)
                continue

            multiplier = current_price / fill["cost_per_share"]
            fill_profit = (current_price - fill["cost_per_share"]) * fill["shares"]

            if multiplier >= SCALP_MULTIPLIER and fill_profit >= SCALP_MIN_PROFIT:
                order_id = await self._sell_order(
                    pos, fill["side"], current_price, fill["shares"], phase=3
                )
                if order_id:
                    logger.info(
                        f"[SC] SCALP: {pos.asset} {pos.timeframe} | "
                        f"Sold {fill['shares']:.1f} {fill['side']} @ ${current_price:.2f} "
                        f"(bought @ ${fill['cost_per_share']:.3f}) | "
                        f"{multiplier:.1f}x | Profit: ${fill_profit:.2f}"
                    )
                else:
                    remaining.append(fill)
            else:
                remaining.append(fill)

        pos.fills = remaining

    # ═══════════════════════════════════════════════════════════
    # ORDER EXECUTION
    # ═══════════════════════════════════════════════════════════

    def _get_order_sync(self, order_id: str) -> Optional[dict]:
        try:
            return self._clob_client.get_order(order_id)
        except Exception as e:
            logger.debug(f"[SC] Order status check error: {e}")
            return None

    def _cancel_order_sync(self, order_id: str) -> bool:
        try:
            self._clob_client.cancel(order_id)
            return True
        except Exception as e:
            logger.debug(f"[SC] Cancel error: {e}")
            return False

    def _post_order_sync(self, token_id: str, price: float,
                          shares: float, is_buy: bool) -> Optional[dict]:
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY, SELL

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=shares,
            side=BUY if is_buy else SELL,
        )
        return self._clob_client.create_and_post_order(order_args)

    async def _place_order(self, pos: WindowPosition, side: str,
                            price: float, shares: float, phase: int) -> Optional[str]:
        """Post a BUY order on the CLOB and queue for fill verification."""
        if not self._clob_client:
            logger.warning("[SC] No CLOB client")
            return None

        price = round(price, 2)
        shares = round(shares, 2)
        if price <= 0 or shares <= 0:
            return None

        token_id = pos.yes_token_id if side == "YES" else pos.no_token_id
        if not token_id:
            logger.warning(f"[SC] No token ID for {side}")
            return None

        try:
            result = await asyncio.to_thread(
                self._post_order_sync, token_id, price, shares, True
            )
            order_id = result.get("orderID", "") if result else ""
            if not order_id:
                logger.warning(f"[SC] No orderID returned for {side} buy")
                return None

            self._pending_orders.append({
                "order_id": order_id,
                "slug": pos.slug,
                "side": side,
                "price": price,
                "shares": shares,
                "phase": phase,
                "is_buy": True,
                "posted_at": time.time(),
                "checks": 0,
            })
            return order_id
        except Exception as e:
            logger.warning(f"[SC] Order error ({side}): {e}")
            return None

    async def _sell_order(self, pos: WindowPosition, side: str,
                           price: float, shares: float,
                           phase: int = 2) -> Optional[str]:
        """Post a SELL order on the CLOB and queue for fill verification."""
        if not self._clob_client:
            return None

        price = round(price, 2)
        shares = round(shares, 2)
        if price <= 0 or shares <= 0:
            return None

        token_id = pos.yes_token_id if side == "YES" else pos.no_token_id
        if not token_id:
            return None

        try:
            result = await asyncio.to_thread(
                self._post_order_sync, token_id, price, shares, False
            )
            order_id = result.get("orderID", "") if result else ""
            if not order_id:
                logger.warning(f"[SC] No orderID returned for {side} sell")
                return None

            self._pending_orders.append({
                "order_id": order_id,
                "slug": pos.slug,
                "side": side,
                "price": price,
                "shares": shares,
                "phase": phase,
                "is_buy": False,
                "posted_at": time.time(),
                "checks": 0,
            })
            return order_id
        except Exception as e:
            logger.warning(f"[SC] Sell error ({side}): {e}")
            return None

    def _record_fill(self, slug, side, price, shares, amount, order_id, phase):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                INSERT INTO spread_capture_fills
                (window_slug, side, price, shares, amount_usd, order_id, phase, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (slug, side, price, shares, amount, order_id, phase, time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"[SC] DB write error: {e}")

    # ═══════════════════════════════════════════════════════════
    # MARKET DATA
    # ═══════════════════════════════════════════════════════════

    def _fetch_market_sync(self, slug: str) -> Optional[dict]:
        try:
            resp = requests.get(
                f"{GAMMA_BASE}/events",
                params={"slug": slug},
                timeout=5,
            )
            if resp.status_code != 200:
                return None

            events = resp.json()
            if not events:
                return None

            event = events[0] if isinstance(events, list) else events
            markets = event.get("markets", [])

            for m in markets:
                if m.get("closed") or m.get("resolved"):
                    continue

                prices = m.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    prices = json.loads(prices)

                token_ids = m.get("clobTokenIds", "[]")
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)

                return {
                    "market_id": m.get("id", ""),
                    "crowd_yes": float(prices[0]) if prices else 0.5,
                    "token_id_yes": token_ids[0] if len(token_ids) >= 1 else None,
                    "token_id_no": token_ids[1] if len(token_ids) >= 2 else None,
                }

            return None
        except Exception as e:
            logger.warning(f"[SC] Market fetch error for {slug}: {e}")
            return None

    async def _fetch_market(self, slug: str) -> Optional[dict]:
        return await asyncio.to_thread(self._fetch_market_sync, slug)

    def _get_clob_prices_sync(self, pos: WindowPosition) -> Tuple[Optional[float], Optional[float]]:
        try:
            if not self._clob_client:
                return None, None

            yes_price = None
            no_price = None

            if pos.yes_token_id:
                resp = self._clob_client.get_price(
                    pos.yes_token_id, side="BUY"
                )
                yes_price = float(resp.get("price", 0.5)) if resp else None

            if pos.no_token_id:
                resp = self._clob_client.get_price(
                    pos.no_token_id, side="BUY"
                )
                no_price = float(resp.get("price", 0.5)) if resp else None

            return yes_price, no_price
        except Exception:
            return None, None

    async def _get_clob_prices(self, pos: WindowPosition) -> Tuple[Optional[float], Optional[float]]:
        return await asyncio.to_thread(self._get_clob_prices_sync, pos)

    # ═══════════════════════════════════════════════════════════
    # RESOLUTION
    # ═══════════════════════════════════════════════════════════

    async def _resolve_window(self, pos: WindowPosition):
        cl_price = self._rtds.get_chainlink_price(pos.asset)

        if cl_price and pos.chainlink_start_price:
            went_up = cl_price >= pos.chainlink_start_price
        else:
            went_up = await asyncio.to_thread(self._check_gamma_resolution_sync, pos.slug)

        if went_up is None:
            return

        pos.resolution = "UP" if went_up else "DOWN"

        # P&L: winning side pays $1/share, losing side already sold or $0
        if went_up:
            payout = pos.net_yes_shares * 1.0 * 0.98  # 2% fee estimate
        else:
            payout = pos.net_no_shares * 1.0 * 0.98

        pos.pnl = payout - pos.total_cost
        pos.status = "resolved"

        self._daily_pnl += pos.pnl
        self._total_windows_resolved += 1

        self._save_window(pos)

        marker = "+++" if pos.pnl > 0 else "---"
        exit_note = ""
        if pos.early_exit_done:
            exit_note = f" | Early exit: sold {pos.early_exit_side} @ ${pos.early_exit_price:.2f} (recovered ${pos.early_exit_recovered:.2f})"

        logger.info(
            f"[SC] {marker} RESOLVED: {pos.asset} {pos.timeframe} -> {pos.resolution}\n"
            f"  YES: {pos.net_yes_shares:.1f}sh @ ${pos.avg_yes_price:.3f}\n"
            f"  NO: {pos.net_no_shares:.1f}sh @ ${pos.avg_no_price:.3f}\n"
            f"  Pair cost: ${pos.pair_cost:.3f} | {'HEDGED' if pos.is_hedged else 'NAKED'}\n"
            f"  P&L: ${pos.pnl:+.2f} | Daily: ${self._daily_pnl:+.2f}"
            f"{exit_note}"
        )

        del self._positions[pos.slug]

    async def _check_resolutions(self):
        now = time.time()
        for slug, pos in list(self._positions.items()):
            if pos.status != "active":
                continue
            window_end = pos.window_start + TIMEFRAMES[pos.timeframe]["seconds"]
            if now > window_end + 60:
                await self._resolve_window(pos)

    def _check_gamma_resolution_sync(self, slug: str) -> Optional[bool]:
        try:
            resp = requests.get(f"{GAMMA_BASE}/events", params={"slug": slug}, timeout=5)
            if resp.status_code != 200:
                return None
            events = resp.json()
            if not events:
                return None
            event = events[0] if isinstance(events, list) else events
            for m in event.get("markets", []):
                if m.get("closed"):
                    prices = m.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        prices = json.loads(prices)
                    if prices:
                        return float(prices[0]) >= 0.95
            return None
        except Exception:
            return None

    def _save_window(self, pos: WindowPosition):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                INSERT OR REPLACE INTO spread_capture_windows
                (asset, timeframe, slug, window_start,
                 yes_shares, yes_cost, yes_avg_price, yes_orders,
                 no_shares, no_cost, no_avg_price, no_orders,
                 yes_sold_shares, yes_sold_revenue,
                 no_sold_shares, no_sold_revenue,
                 pair_cost, total_deployed, guaranteed_profit, is_arb,
                 chainlink_start, resolution, pnl,
                 pnl_pct, max_phase, phase2_triggered, status, resolved_at,
                 yes_token_id, no_token_id, market_id,
                 early_exit_side, early_exit_price, early_exit_recovered)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?)
            """, (
                pos.asset, pos.timeframe, pos.slug, pos.window_start,
                pos.yes_shares, pos.yes_cost, pos.avg_yes_price, pos.yes_orders,
                pos.no_shares, pos.no_cost, pos.avg_no_price, pos.no_orders,
                pos.yes_sold_shares, pos.yes_sold_revenue,
                pos.no_sold_shares, pos.no_sold_revenue,
                pos.pair_cost, pos.total_cost, pos.guaranteed_profit,
                1 if pos.is_arb else 0,
                pos.chainlink_start_price, pos.resolution, pos.pnl,
                pos.pnl / pos.total_cost * 100 if pos.total_cost > 0 else 0,
                2 if pos.is_hedged else 1,
                1 if pos.is_hedged else 0,
                pos.status,
                datetime.now(timezone.utc).isoformat(),
                pos.yes_token_id, pos.no_token_id, pos.market_id,
                pos.early_exit_side, pos.early_exit_price, pos.early_exit_recovered,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"[SC] DB save error: {e}")

    # ═══════════════════════════════════════════════════════════
    # STATS & DASHBOARD
    # ═══════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row

            stats = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN is_arb = 1 THEN 1 ELSE 0 END) as arb_windows,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pair_cost), 0) as avg_pair_cost,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    SUM(CASE WHEN early_exit_side != '' AND early_exit_side IS NOT NULL
                        THEN 1 ELSE 0 END) as early_exits
                FROM spread_capture_windows
                WHERE status = 'resolved'
            """).fetchone()

            total = stats["total"] or 0
            wins = stats["wins"] or 0
            early_exits = stats["early_exits"] or 0

            # Hedge rate from active + resolved
            hedged = conn.execute("""
                SELECT COUNT(*) FROM spread_capture_windows
                WHERE yes_shares > 0 AND no_shares > 0
            """).fetchone()[0]
            all_windows = conn.execute(
                "SELECT COUNT(*) FROM spread_capture_windows"
            ).fetchone()[0]

            result = {
                "strategy": "0x8dxd_dual_accumulation_v2",
                "total_windows": total,
                "wins": wins,
                "losses": stats["losses"] or 0,
                "win_rate": round(wins / max(total, 1) * 100, 1),
                "hedge_rate": round(hedged / max(all_windows, 1) * 100, 1),
                "early_exit_rate": round(early_exits / max(total, 1) * 100, 1),
                "arb_windows": stats["arb_windows"] or 0,
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_pair_cost": round(stats["avg_pair_cost"], 3),
                "avg_pnl_per_window": round(stats["avg_pnl"], 3),
                "active_positions": len([p for p in self._positions.values()
                                         if p.status == "active"]),
                "daily_pnl": round(self._daily_pnl, 2),
                "daily_windows": self._daily_windows,
            }

            conn.close()
            return result
        except Exception as e:
            logger.warning(f"[SC] Stats error: {e}")
            return {"strategy": "0x8dxd_dual_accumulation_v2", "error": str(e)}

    def get_active_positions(self) -> list:
        result = []
        for slug, pos in self._positions.items():
            if pos.status != "active":
                continue
            result.append({
                "slug": slug,
                "asset": pos.asset,
                "timeframe": pos.timeframe,
                "yes_shares": round(pos.net_yes_shares, 2),
                "yes_avg_price": round(pos.avg_yes_price, 3),
                "no_shares": round(pos.net_no_shares, 2),
                "no_avg_price": round(pos.avg_no_price, 3),
                "pair_cost": round(pos.pair_cost, 3),
                "total_cost": round(pos.total_cost, 2),
                "is_hedged": pos.is_hedged,
                "is_arb": pos.is_arb,
                "guaranteed_profit": round(pos.guaranteed_profit, 2),
                "early_exit_done": pos.early_exit_done,
                "fills_yes": pos.yes_orders,
                "fills_no": pos.no_orders,
            })
        return result

    async def stop(self):
        self._running = False
        if self._pending_orders:
            await self._process_pending_orders()
        for slug, pos in self._positions.items():
            if pos.status == "active":
                self._save_active_window(pos)
        logger.info(f"[SC] Saved {len(self._positions)} positions on shutdown")
