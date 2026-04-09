"""
polymarket_spread_capture.py — v2: Passive Limit Order Market Maker

Places resting limit orders on BOTH sides at multiple price levels.
The market fills them as prices swing. No direction prediction.
No waiting for crashes. Just set traps and collect the spread.

Based on 0x8dxd: $636M volume, $5.75M profit, 170+ fills/day, 24/7.

v1 picked a "favorite" side → Phase 2 only fired 12% of the time.
v2 places orders on BOTH sides from the start → fills whenever
the crowd pushes either side down.
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
# CONFIGURATION
# Based on 0x8dxd: $636M volume, $5.75M profit, 0.9% margin
# ═══════════════════════════════════════════════════════════════

# Assets with confirmed Polymarket CLOB orderbooks
# Removed: BNB (no markets), HYPE (no markets), DOGE (no 5m/15m markets)
ASSETS = {
    "BTC": {"slug_5m": "btc-updown-5m", "slug_15m": "btc-updown-15m"},
    "ETH": {"slug_5m": "eth-updown-5m", "slug_15m": "eth-updown-15m"},
    "SOL": {"slug_5m": "sol-updown-5m", "slug_15m": "sol-updown-15m"},
    "XRP": {"slug_15m": "xrp-updown-15m"},  # 5m orderbook doesn't exist
}

TIMEFRAMES = {
    "5m":  {"seconds": 300, "alignment": 300},
    "15m": {"seconds": 900, "alignment": 900},
}

# ── ORDER LADDER ──
# Resting limit orders placed at these prices on BOTH sides.
# Each tuple: (price, shares)
# CLOB minimums: 5 shares per order, $1.00 for marketable orders
# Total budget per side if ALL levels fill: $4.50
# Total budget BOTH sides if ALL fill: $9.00
ORDER_LADDER = [
    (0.40, 5),   # $2.00 — fills when underdog dips to 40¢
    (0.30, 5),   # $1.50 — fills on moderate moves
    (0.20, 5),   # $1.00 — fills on strong moves (meets $1 min)
]

# ── TIMING ──
ORDER_PLACEMENT_DELAY = 3        # Seconds after window open to place orders
ORDER_CANCEL_BEFORE_END = 30     # Cancel unfilled orders N seconds before close
FILL_CHECK_INTERVAL = 5          # Check for fills every N seconds

# ── EXPOSURE LIMITS ──
MAX_EXPOSURE_PER_WINDOW = 10.00  # Max $ deployed in one window (both sides)
MAX_TOTAL_EXPOSURE = 200.00      # Across ALL active windows simultaneously
MAX_DAILY_LOSS = 100.00          # Stop trading for the day
MAX_OPEN_ORDERS = 200            # Cap total open orders

# ── POST-RESOLUTION ──
SELL_WINNERS_AT = 0.99           # Price to sell winning shares post-resolution
ENABLE_AUTO_SELL = True          # Sell winning shares automatically

# ── DATABASE ──
DB_PATH = "data/renaissance_bot.db"
SECRETS_PATH = "config/polymarket_secrets.json"
GAMMA_BASE = "https://gamma-api.polymarket.com"


@dataclass
class PendingOrder:
    """An order placed but not yet confirmed filled."""
    order_id: str
    side: str           # "YES" or "NO"
    price: float
    shares: float
    amount_usd: float   # price × shares
    token_id: str
    placed_at: float
    checks: int = 0
    filled: bool = False
    cancelled: bool = False


@dataclass
class WindowState:
    """Tracks all state for a single market window."""
    asset: str
    timeframe: str
    window_start: int
    slug: str
    market_id: str = ""
    yes_token_id: str = ""
    no_token_id: str = ""

    # Filled shares (confirmed only)
    yes_shares: float = 0.0
    yes_cost: float = 0.0
    no_shares: float = 0.0
    no_cost: float = 0.0

    # Pending orders
    pending_orders: list = field(default_factory=list)

    # All order IDs for cancellation
    all_order_ids: list = field(default_factory=list)

    # Flags
    orders_placed: bool = False
    orders_cancelled: bool = False

    # Status
    status: str = "active"   # active, resolved, error
    resolution: str = ""     # UP, DOWN
    pnl: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.yes_cost + self.no_cost

    @property
    def hedged_pairs(self) -> float:
        return min(self.yes_shares, self.no_shares)

    @property
    def pair_cost(self) -> float:
        if self.hedged_pairs == 0:
            return 1.0
        return self.total_cost / self.hedged_pairs

    @property
    def is_hedged(self) -> bool:
        return self.yes_shares > 0 and self.no_shares > 0

    @property
    def total_fills(self) -> int:
        return sum(1 for o in self.pending_orders if o.filled)


class SpreadCaptureV2:
    """
    Passive limit order market maker for Polymarket direction markets.

    For each window:
    1. Place resting BUY orders on BOTH YES and NO at 5 price levels
    2. Let the market fill them as prices swing
    3. Cancel unfilled orders 30s before window close
    4. Collect payout at resolution
    5. Sell winning shares at 99¢ to recycle USDC

    No direction prediction. No waiting for crashes.
    Just resting orders that get filled when the crowd overreacts.
    """

    def __init__(self):
        self._clob_client = None
        self._running = False
        self._windows: Dict[str, WindowState] = {}
        self._daily_pnl = 0.0
        self._daily_date = ""
        self._daily_windows = 0
        self._total_open_orders = 0
        self._entering: set = set()          # slugs currently being entered (prevent dupes)
        self._entry_sem = asyncio.Semaphore(1)  # serialize entry to avoid CLOB rate-limits
        self._last_fill_check = 0.0          # track last fill-check time

        self._init_db()
        self._init_clob()
        self._restore_positions()

    def _init_db(self):
        """Create tracking tables for v2."""
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS spread_v2_windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                slug TEXT NOT NULL,
                window_start INTEGER NOT NULL,

                -- Confirmed fills only
                yes_shares REAL DEFAULT 0,
                yes_cost REAL DEFAULT 0,
                no_shares REAL DEFAULT 0,
                no_cost REAL DEFAULT 0,

                -- Orders
                orders_placed INTEGER DEFAULT 0,
                orders_filled INTEGER DEFAULT 0,
                orders_cancelled INTEGER DEFAULT 0,

                -- Metrics
                pair_cost REAL,
                hedged_pairs REAL,
                total_deployed REAL DEFAULT 0,
                is_hedged INTEGER DEFAULT 0,

                -- Resolution
                resolution TEXT,
                pnl REAL,

                -- Status
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                resolved_at TEXT,

                UNIQUE(asset, timeframe, window_start)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS spread_v2_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_slug TEXT NOT NULL,
                order_id TEXT,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                shares REAL NOT NULL,
                amount_usd REAL NOT NULL,
                token_id TEXT,
                status TEXT DEFAULT 'pending',
                placed_at REAL,
                filled_at REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_sv2w_status ON spread_v2_windows(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sv2w_slug ON spread_v2_windows(slug)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sv2o_slug ON spread_v2_orders(window_slug)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sv2o_oid ON spread_v2_orders(order_id)")
        conn.commit()
        conn.close()
        logger.info("[SC] Spread Capture v2 DB initialized")

    def _init_clob(self):
        """Initialize CLOB client with api creds derivation."""
        try:
            from py_clob_client.client import ClobClient

            if not os.path.exists(SECRETS_PATH):
                logger.error(f"[SC] No secrets at {SECRETS_PATH}")
                return

            with open(SECRETS_PATH) as f:
                secrets = json.load(f)

            # Build init kwargs — proxy wallet support
            proxy_addr = secrets.get("proxy_wallet_address", "")
            init_kwargs = {
                "host": "https://clob.polymarket.com",
                "chain_id": 137,
                "key": secrets["private_key"],
            }
            if proxy_addr:
                init_kwargs["signature_type"] = 1  # POLY_PROXY
                init_kwargs["funder"] = proxy_addr
                logger.info(f"[SC] Using proxy wallet: {proxy_addr[:10]}...{proxy_addr[-6:]}")

            self._clob_client = ClobClient(**init_kwargs)

            # Derive API credentials (returns ApiCreds object with .api_key etc.)
            api_creds = self._clob_client.create_or_derive_api_creds()
            self._clob_client.set_api_creds(api_creds)

            server_time = self._clob_client.get_server_time()
            logger.info(f"[SC] CLOB client initialized (server_time={server_time})")
        except Exception as e:
            logger.error(f"[SC] CLOB init failed: {e}")

    def _restore_positions(self):
        """Load active windows from DB on restart."""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row

            active = conn.execute("""
                SELECT * FROM spread_v2_windows WHERE status = 'active'
            """).fetchall()

            for row in active:
                ws = WindowState(
                    asset=row["asset"],
                    timeframe=row["timeframe"],
                    window_start=row["window_start"],
                    slug=row["slug"],
                    yes_shares=row["yes_shares"] or 0,
                    yes_cost=row["yes_cost"] or 0,
                    no_shares=row["no_shares"] or 0,
                    no_cost=row["no_cost"] or 0,
                    orders_placed=True,
                    orders_cancelled=True,  # Don't cancel old orders
                    status="active",
                )
                self._windows[row["slug"]] = ws

            conn.close()
            if active:
                logger.info(f"[SC] Restored {len(active)} active windows from DB")
        except Exception as e:
            logger.warning(f"[SC] Restore error: {e}")

    # ═══════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════

    async def run(self):
        """Main loop. Runs every second."""
        self._running = True
        per_side = sum(p * s for p, s in ORDER_LADDER)
        logger.info(
            f"[SC] Spread Capture v2 STARTED\n"
            f"  Strategy: Passive limit orders on both sides\n"
            f"  Assets: {list(ASSETS.keys())} ({len(ASSETS)} assets)\n"
            f"  Timeframes: {list(TIMEFRAMES.keys())}\n"
            f"  Order ladder: {ORDER_LADDER}\n"
            f"  Max per side: ${per_side:.2f} | Max per window: ${MAX_EXPOSURE_PER_WINDOW}\n"
            f"  Max global: ${MAX_TOTAL_EXPOSURE}"
        )

        while self._running:
            try:
                now = time.time()

                # Daily reset
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if today != self._daily_date:
                    if self._daily_date:
                        logger.info(
                            f"[SC] Daily reset. Yesterday: {self._daily_windows} windows, "
                            f"${self._daily_pnl:+.2f}"
                        )
                    self._daily_date = today
                    self._daily_pnl = 0.0
                    self._daily_windows = 0

                # Safety: daily loss limit
                if self._daily_pnl <= -MAX_DAILY_LOSS:
                    if int(now) % 60 < 2:
                        logger.warning(f"[SC] Daily loss limit hit (${self._daily_pnl:.2f}). Paused.")
                    await asyncio.sleep(60)
                    continue

                # ── CHECK FILLS on active windows ──
                if now - self._last_fill_check >= FILL_CHECK_INTERVAL:
                    self._last_fill_check = now
                    for slug, ws in list(self._windows.items()):
                        if ws.status == "active" and ws.orders_placed and not ws.orders_cancelled:
                            await self._check_fills(ws)

                # ── MANAGE ACTIVE WINDOWS ──
                for slug, ws in list(self._windows.items()):
                    if ws.status != "active":
                        continue

                    elapsed = now - ws.window_start
                    window_seconds = TIMEFRAMES[ws.timeframe]["seconds"]

                    # Window expired? Resolve.
                    if elapsed > window_seconds + 60:
                        await self._resolve_window(ws)
                        continue

                    # Cancel unfilled orders before window closes
                    if elapsed >= window_seconds - ORDER_CANCEL_BEFORE_END and not ws.orders_cancelled:
                        await self._cancel_unfilled(ws)

                # ── ENTER NEW WINDOWS ──
                total_exposure = sum(
                    w.total_cost for w in self._windows.values() if w.status == "active"
                )

                for asset, config in ASSETS.items():
                    for tf_name, tf_config in TIMEFRAMES.items():
                        slug_prefix = config.get(f"slug_{tf_name}")
                        if not slug_prefix:
                            continue  # asset doesn't have this timeframe

                        alignment = tf_config["alignment"]
                        window_start = int(math.floor(now / alignment) * alignment)
                        slug = f"{slug_prefix}-{window_start}"
                        elapsed = now - window_start

                        # Entry window: 3s to (close - 30s)
                        window_seconds = tf_config["seconds"]
                        if (slug not in self._windows
                                and slug not in self._entering
                                and elapsed >= ORDER_PLACEMENT_DELAY
                                and elapsed < window_seconds - ORDER_CANCEL_BEFORE_END):

                            if total_exposure >= MAX_TOTAL_EXPOSURE:
                                continue
                            if self._total_open_orders >= MAX_OPEN_ORDERS:
                                continue

                            # Non-blocking: launch entry in background so fill
                            # checks and lifecycle management keep running
                            self._entering.add(slug)
                            asyncio.create_task(
                                self._safe_enter_window(
                                    asset, tf_name, window_start, slug, config
                                )
                            )
                            total_exposure += MAX_EXPOSURE_PER_WINDOW  # pre-reserve

                # ── HEARTBEAT ──
                if int(now) % 60 < 2:
                    active = len([w for w in self._windows.values() if w.status == "active"])
                    deployed = sum(
                        w.total_cost for w in self._windows.values() if w.status == "active"
                    )
                    hedged = sum(
                        1 for w in self._windows.values()
                        if w.status == "active" and w.is_hedged
                    )
                    logger.info(
                        f"[SC] Heartbeat | {active} active | {hedged} hedged | "
                        f"${deployed:.2f} deployed | {self._total_open_orders} orders | "
                        f"daily ${self._daily_pnl:+.2f}"
                    )

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                logger.info("[SC] Spread capture v2 cancelled — shutting down")
                return
            except Exception as e:
                logger.error(f"[SC] Loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    # ═══════════════════════════════════════════════════════════
    # ENTER WINDOW — Place limit orders on BOTH sides
    # ═══════════════════════════════════════════════════════════

    async def _safe_enter_window(self, asset: str, timeframe: str,
                                  window_start: int, slug: str, config: dict):
        """Wrapper for non-blocking entry. Serialized via semaphore."""
        async with self._entry_sem:
            try:
                await self._enter_window(asset, timeframe, window_start, slug, config)
            except Exception as e:
                logger.error(f"[SC] Entry error for {asset} {timeframe}: {e}", exc_info=True)
            finally:
                self._entering.discard(slug)

    async def _enter_window(self, asset: str, timeframe: str,
                             window_start: int, slug: str, config: dict):
        """
        Place resting limit BUY orders on BOTH YES and NO at multiple prices.
        No direction picking. Orders on BOTH sides.
        """
        # Fetch market data
        logger.info(f"[SC] Entering {asset} {timeframe}: fetching market {slug}...")
        market_data = await asyncio.to_thread(self._fetch_market_sync, slug)
        if not market_data:
            logger.warning(f"[SC] {asset} {timeframe}: market {slug} not found on Gamma API")
            return

        yes_token = market_data.get("token_id_yes")
        no_token = market_data.get("token_id_no")

        if not yes_token or not no_token:
            logger.warning(f"[SC] {asset} {timeframe}: missing token IDs")
            return

        ws = WindowState(
            asset=asset,
            timeframe=timeframe,
            window_start=window_start,
            slug=slug,
            market_id=market_data.get("market_id", ""),
            yes_token_id=yes_token,
            no_token_id=no_token,
        )

        # Place orders on BOTH sides
        yes_orders = 0
        no_orders = 0

        for price, shares in ORDER_LADDER:
            # YES side
            order_id = await self._place_limit_order(yes_token, price, shares)
            if order_id:
                ws.pending_orders.append(PendingOrder(
                    order_id=order_id,
                    side="YES",
                    price=price,
                    shares=shares,
                    amount_usd=price * shares,
                    token_id=yes_token,
                    placed_at=time.time(),
                ))
                ws.all_order_ids.append(order_id)
                yes_orders += 1

            # NO side
            order_id = await self._place_limit_order(no_token, price, shares)
            if order_id:
                ws.pending_orders.append(PendingOrder(
                    order_id=order_id,
                    side="NO",
                    price=price,
                    shares=shares,
                    amount_usd=price * shares,
                    token_id=no_token,
                    placed_at=time.time(),
                ))
                ws.all_order_ids.append(order_id)
                no_orders += 1

        total_placed = yes_orders + no_orders
        if total_placed == 0:
            logger.warning(f"[SC] {asset} {timeframe}: all orders failed, skipping window")
            return

        ws.orders_placed = True
        self._windows[slug] = ws
        self._daily_windows += 1
        self._total_open_orders += total_placed

        # Save to DB
        self._save_window(ws)

        # Record all orders
        conn = sqlite3.connect(DB_PATH)
        for order in ws.pending_orders:
            conn.execute("""
                INSERT INTO spread_v2_orders
                (window_slug, order_id, side, price, shares, amount_usd,
                 token_id, status, placed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """, (slug, order.order_id, order.side, order.price,
                  order.shares, order.amount_usd, order.token_id, order.placed_at))
        conn.commit()
        conn.close()

        logger.info(
            f"[SC] *** ENTERED: {asset} {timeframe} ***\n"
            f"  Placed {yes_orders} YES + {no_orders} NO limit orders\n"
            f"  Ladder: {', '.join(f'${p:.2f}x{s}' for p, s in ORDER_LADDER)}\n"
            f"  Max exposure: ${sum(p * s for p, s in ORDER_LADDER) * 2:.2f}"
        )

    # ═══════════════════════════════════════════════════════════
    # CHECK FILLS — Verify which orders actually executed
    # ═══════════════════════════════════════════════════════════

    async def _check_fills(self, ws: WindowState):
        """
        Check pending orders for fills. NON-BLOCKING.
        Only credit shares when we confirm the order matched.
        """
        if not self._clob_client:
            return

        pending_count = sum(1 for o in ws.pending_orders if not o.filled and not o.cancelled)
        if pending_count == 0:
            return

        for order in ws.pending_orders:
            if order.filled or order.cancelled:
                continue

            order.checks += 1

            try:
                result = await asyncio.to_thread(
                    self._get_order_sync, order.order_id
                )

                if not result:
                    if order.checks <= 3:
                        logger.info(
                            f"[SC] Fill check #{order.checks}: {ws.asset} {ws.timeframe} "
                            f"{order.side} @ ${order.price:.2f} — no result from CLOB"
                        )
                    continue

                status = str(result.get("status", "")).upper()
                size_matched = float(
                    result.get("sizeMatched", 0) or result.get("size_matched", 0) or 0
                )

                # Log first few checks and any status changes
                if order.checks <= 3 or order.checks % 20 == 0:
                    logger.info(
                        f"[SC] Fill check #{order.checks}: {ws.asset} {ws.timeframe} "
                        f"{order.side} @ ${order.price:.2f} — status={status} "
                        f"sizeMatched={size_matched}"
                    )

                if status in ("MATCHED", "FILLED", "CLOSED") or size_matched > 0:
                    order.filled = True

                    fill_size = size_matched if size_matched > 0 else float(
                        result.get("size", order.shares) or order.shares
                    )
                    fill_price = order.price  # Limit order fills at our price or better

                    if order.side == "YES":
                        ws.yes_shares += fill_size
                        ws.yes_cost += fill_price * fill_size
                    else:
                        ws.no_shares += fill_size
                        ws.no_cost += fill_price * fill_size

                    self._total_open_orders = max(0, self._total_open_orders - 1)

                    # Update order in DB
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("""
                        UPDATE spread_v2_orders SET status='filled', filled_at=?
                        WHERE order_id=?
                    """, (time.time(), order.order_id))
                    conn.commit()
                    conn.close()

                    self._save_window(ws)

                    logger.info(
                        f"[SC] FILLED: {ws.asset} {ws.timeframe} {order.side} "
                        f"{fill_size:.1f}sh @ ${fill_price:.2f} "
                        f"(${fill_price * fill_size:.2f}) | "
                        f"YES:{ws.yes_shares:.0f} NO:{ws.no_shares:.0f} | "
                        f"{'HEDGED' if ws.is_hedged else 'ONE-SIDE'}"
                    )

                elif "CANCEL" in status:
                    # Handles CANCELLED, CANCELED, CANCELED_MARKET_RESOLVED
                    order.cancelled = True
                    self._total_open_orders = max(0, self._total_open_orders - 1)

                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("""
                        UPDATE spread_v2_orders SET status='cancelled'
                        WHERE order_id=?
                    """, (order.order_id,))
                    conn.commit()
                    conn.close()

            except Exception as e:
                if order.checks % 10 == 0:
                    logger.debug(f"[SC] Fill check error: {e}")

    # ═══════════════════════════════════════════════════════════
    # CANCEL UNFILLED — Clean up before window closes
    # ═══════════════════════════════════════════════════════════

    async def _cancel_unfilled(self, ws: WindowState):
        """Cancel all unfilled orders before window closes."""
        if not self._clob_client:
            ws.orders_cancelled = True
            return

        cancelled = 0
        for order in ws.pending_orders:
            if order.filled or order.cancelled:
                continue

            try:
                await asyncio.to_thread(self._cancel_order_sync, order.order_id)
                order.cancelled = True
                self._total_open_orders = max(0, self._total_open_orders - 1)
                cancelled += 1

                conn = sqlite3.connect(DB_PATH)
                conn.execute("""
                    UPDATE spread_v2_orders SET status='cancelled'
                    WHERE order_id=?
                """, (order.order_id,))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.debug(f"[SC] Cancel error: {e}")

        ws.orders_cancelled = True

        if cancelled > 0:
            logger.info(
                f"[SC] Cancelled {cancelled} unfilled orders | "
                f"{ws.asset} {ws.timeframe} | "
                f"Final: YES:{ws.yes_shares:.0f} NO:{ws.no_shares:.0f}"
            )

    # ═══════════════════════════════════════════════════════════
    # PLACE LIMIT ORDER — Single order placement
    # ═══════════════════════════════════════════════════════════

    async def _place_limit_order(self, token_id: str, price: float,
                                   shares: float) -> Optional[str]:
        """Place a single limit BUY order. Returns order_id or None."""
        if not self._clob_client:
            return None

        try:
            result = await asyncio.to_thread(
                self._post_order_sync, token_id, round(price, 2), round(shares, 2), True
            )

            if result:
                order_id = result.get("orderID", result.get("id", ""))
                if order_id:
                    logger.info(f"[SC] Order placed: {token_id[:10]}... ${price:.2f} x {shares:.0f} → {order_id[:20]}")
                    return order_id
                else:
                    logger.warning(f"[SC] Order returned no ID: {result}")

            return None
        except Exception as e:
            logger.warning(f"[SC] Order FAILED @ ${price:.2f} x {shares:.0f}: {e}")
            return None

    # ═══════════════════════════════════════════════════════════
    # RESOLVE — Compute P&L and sell winners
    # ═══════════════════════════════════════════════════════════

    async def _resolve_window(self, ws: WindowState):
        """Resolve a completed window. Compute P&L. Sell winners."""
        # Cancel any remaining orders first
        if not ws.orders_cancelled:
            await self._cancel_unfilled(ws)

        # Fast-close empty windows (no fills, nothing to resolve)
        if ws.yes_shares == 0 and ws.no_shares == 0 and len(ws.pending_orders) == 0:
            ws.status = "resolved"
            ws.pnl = 0.0
            self._save_window(ws)
            if ws.slug in self._windows:
                del self._windows[ws.slug]
            logger.info(f"[SC] Closed empty window: {ws.asset} {ws.timeframe} (no fills)")
            return

        # Check resolution via Gamma API
        resolved = await asyncio.to_thread(
            self._check_gamma_resolution_sync, ws.slug
        )

        if resolved is None:
            # Check if window is way past expiry — force resolve as error
            elapsed = time.time() - ws.window_start
            window_seconds = TIMEFRAMES[ws.timeframe]["seconds"]
            # Give Gamma 30 minutes to resolve (markets can lag)
            if elapsed > window_seconds + 1800:
                ws.status = "error"
                ws.pnl = -ws.total_cost  # assume total loss
                self._daily_pnl += ws.pnl
                self._save_window(ws)
                del self._windows[ws.slug]
                logger.warning(
                    f"[SC] EXPIRED: {ws.asset} {ws.timeframe} — "
                    f"never resolved after 30min. Lost ${ws.total_cost:.2f}"
                )
            return

        went_up = resolved
        ws.resolution = "UP" if went_up else "DOWN"

        # Compute P&L
        if ws.yes_shares == 0 and ws.no_shares == 0:
            ws.pnl = 0.0
        elif went_up:
            payout = ws.yes_shares * 1.0
            ws.pnl = payout - ws.total_cost
        else:
            payout = ws.no_shares * 1.0
            ws.pnl = payout - ws.total_cost

        ws.status = "resolved"
        self._daily_pnl += ws.pnl

        self._save_window(ws)

        # Log
        fills = ws.total_fills
        marker = "+++" if ws.pnl > 0 else "---" if ws.pnl < 0 else "==="
        hedge_tag = "HEDGED" if ws.is_hedged else "NAKED"

        logger.info(
            f"[SC] {marker} RESOLVED: {ws.asset} {ws.timeframe} -> {ws.resolution}\n"
            f"  YES: {ws.yes_shares:.0f}sh (${ws.yes_cost:.2f}) | "
            f"NO: {ws.no_shares:.0f}sh (${ws.no_cost:.2f})\n"
            f"  Deployed: ${ws.total_cost:.2f} | "
            f"Pair: ${ws.pair_cost:.3f} | {hedge_tag} | {fills} fills\n"
            f"  P&L: ${ws.pnl:+.2f} | Daily: ${self._daily_pnl:+.2f}"
        )

        # Sell winning shares to recycle USDC
        if ENABLE_AUTO_SELL and ws.pnl >= 0:
            await self._sell_winners(ws)

        # Remove from active
        del self._windows[ws.slug]

    async def _sell_winners(self, ws: WindowState):
        """Sell winning shares at $0.99 to get USDC back in wallet."""
        if not self._clob_client:
            return

        try:
            if ws.resolution == "UP" and ws.yes_shares > 0:
                token_id = ws.yes_token_id
                shares = ws.yes_shares
            elif ws.resolution == "DOWN" and ws.no_shares > 0:
                token_id = ws.no_token_id
                shares = ws.no_shares
            else:
                return

            if not token_id or shares <= 0:
                return

            result = await asyncio.to_thread(
                self._post_order_sync, token_id,
                SELL_WINNERS_AT, round(shares, 2), False
            )

            if result:
                logger.info(
                    f"[SC] SELL WINNER: {ws.asset} {shares:.0f}sh @ "
                    f"${SELL_WINNERS_AT} -> ${shares * SELL_WINNERS_AT:.2f} USDC"
                )
        except Exception as e:
            logger.warning(f"[SC] Sell winner error: {e}")

    # ═══════════════════════════════════════════════════════════
    # CLOB HELPERS (blocking — called via to_thread)
    # ═══════════════════════════════════════════════════════════

    def _post_order_sync(self, token_id: str, price: float,
                          shares: float, is_buy: bool) -> Optional[dict]:
        """Post order to CLOB. (Blocking)"""
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY, SELL

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=shares,
            side=BUY if is_buy else SELL,
        )
        return self._clob_client.create_and_post_order(order_args)

    def _get_order_sync(self, order_id: str) -> Optional[dict]:
        """Check order status on the CLOB. (Blocking)"""
        try:
            result = self._clob_client.get_order(order_id)
            return result
        except Exception as e:
            logger.debug(f"[SC] get_order error for {order_id[:16]}: {e}")
            return None

    def _cancel_order_sync(self, order_id: str) -> bool:
        """Cancel an unfilled order. (Blocking)"""
        try:
            self._clob_client.cancel(order_id)
            return True
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════
    # MARKET DATA HELPERS
    # ═══════════════════════════════════════════════════════════

    def _fetch_market_sync(self, slug: str) -> Optional[dict]:
        """Fetch market token IDs from Gamma API. (Blocking)"""
        try:
            resp = requests.get(
                f"{GAMMA_BASE}/markets",
                params={"slug": slug},
                timeout=5,
            )
            if resp.status_code != 200:
                return None

            markets = resp.json()
            if not markets:
                return None

            for m in markets:
                if m.get("closed") or m.get("resolved"):
                    continue

                token_ids = m.get("clobTokenIds", "[]")
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)

                prices = m.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    prices = json.loads(prices)

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

    def _check_gamma_resolution_sync(self, slug: str) -> Optional[bool]:
        """Check if market resolved. Returns True=UP, False=DOWN, None=not yet."""
        try:
            resp = requests.get(
                f"{GAMMA_BASE}/markets", params={"slug": slug}, timeout=5
            )
            if resp.status_code != 200:
                return None

            markets = resp.json()
            for m in markets:
                if m.get("resolved"):
                    prices = m.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        prices = json.loads(prices)
                    if prices:
                        return float(prices[0]) >= 0.95
            return None
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════
    # DB PERSISTENCE
    # ═══════════════════════════════════════════════════════════

    def _save_window(self, ws: WindowState):
        """Save/update window state in DB."""
        conn = sqlite3.connect(DB_PATH)

        filled = sum(1 for o in ws.pending_orders if o.filled)
        cancelled = sum(1 for o in ws.pending_orders if o.cancelled)

        conn.execute("""
            INSERT OR REPLACE INTO spread_v2_windows
            (asset, timeframe, slug, window_start,
             yes_shares, yes_cost, no_shares, no_cost,
             orders_placed, orders_filled, orders_cancelled,
             pair_cost, hedged_pairs, total_deployed, is_hedged,
             resolution, pnl, status, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ws.asset, ws.timeframe, ws.slug, ws.window_start,
            ws.yes_shares, ws.yes_cost, ws.no_shares, ws.no_cost,
            len(ws.pending_orders), filled, cancelled,
            ws.pair_cost if ws.hedged_pairs > 0 else None,
            ws.hedged_pairs,
            ws.total_cost,
            1 if ws.is_hedged else 0,
            ws.resolution, ws.pnl, ws.status,
            datetime.now(timezone.utc).isoformat() if ws.status == "resolved" else None,
        ))
        conn.commit()
        conn.close()

    # ═══════════════════════════════════════════════════════════
    # STATS & DASHBOARD
    # ═══════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """Stats for dashboard endpoint."""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row

            stats = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN is_hedged = 1 THEN 1 ELSE 0 END) as hedged,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pair_cost), 0) as avg_pair,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COALESCE(SUM(total_deployed), 0) as total_deployed
                FROM spread_v2_windows
                WHERE status = 'resolved'
            """).fetchone()

            total = stats["total"] or 0
            wins = stats["wins"] or 0

            conn.close()

            return {
                "strategy": "spread_capture_v2",
                "version": "passive_limit_orders",
                "total_windows": total,
                "wins": wins,
                "losses": stats["losses"] or 0,
                "win_rate": round(wins / max(total, 1) * 100, 1),
                "hedged_windows": stats["hedged"] or 0,
                "hedge_rate": round((stats["hedged"] or 0) / max(total, 1) * 100, 1),
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_pnl": round(stats["avg_pnl"], 3),
                "avg_pair_cost": round(stats["avg_pair"], 3),
                "total_deployed": round(stats["total_deployed"], 2),
                "active_windows": len(
                    [w for w in self._windows.values() if w.status == "active"]
                ),
                "daily_pnl": round(self._daily_pnl, 2),
                "daily_windows": self._daily_windows,
                "open_orders": self._total_open_orders,
            }
        except Exception as e:
            logger.warning(f"[SC] Stats error: {e}")
            return {"strategy": "spread_capture_v2", "error": str(e)}

    def get_active_positions(self) -> list:
        """Return active position details for dashboard."""
        result = []
        for slug, ws in self._windows.items():
            if ws.status != "active":
                continue
            result.append({
                "slug": slug,
                "asset": ws.asset,
                "timeframe": ws.timeframe,
                "yes_shares": round(ws.yes_shares, 2),
                "no_shares": round(ws.no_shares, 2),
                "yes_cost": round(ws.yes_cost, 2),
                "no_cost": round(ws.no_cost, 2),
                "total_cost": round(ws.total_cost, 2),
                "pair_cost": round(ws.pair_cost, 3),
                "is_hedged": ws.is_hedged,
                "hedged_pairs": round(ws.hedged_pairs, 1),
                "fills": ws.total_fills,
                "orders_cancelled": ws.orders_cancelled,
            })
        return result

    async def stop(self):
        """Graceful shutdown — cancel all open orders, save state."""
        self._running = False
        logger.info("[SC] Stopping spread capture v2...")
        for ws in list(self._windows.values()):
            if not ws.orders_cancelled:
                await self._cancel_unfilled(ws)
            if ws.status == "active":
                self._save_window(ws)
        logger.info(f"[SC] Saved {len(self._windows)} windows on shutdown")
