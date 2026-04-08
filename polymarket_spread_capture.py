"""
polymarket_spread_capture.py — 0x8dxd-Style Spread Capture

Replicates the #1 Polymarket crypto trader's strategy:
- Buy BOTH sides of each 5m/15m direction market
- Accumulate the underdog when it crashes to pennies
- Achieve pair cost < $1.00 for guaranteed profit
- Optional sell/rotation if momentum reverses

No direction prediction needed. +8.19% EV at 50/50 accuracy.

Key parameters (from 0x8dxd analysis of 3,500 trades):
- Assets: BTC + ETH only (15m and 5m)
- Entry: within 30 seconds of window open
- 87% limit orders at round cent prices
- 28 fills per window average, up to 121
- Pair cost target: < $1.05 ideal, always < $1.20
- Sell rate: 15% (rotation when momentum reverses)
- Never skips a window
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
# CONFIGURATION — Based on 0x8dxd's observed parameters
# ═══════════════════════════════════════════════════════════════

# Assets: 0x8dxd trades BTC and ETH only
ASSETS = {
    "BTC": {
        "slug_5m": "btc-updown-5m",
        "slug_15m": "btc-updown-15m",
    },
    "ETH": {
        "slug_5m": "eth-updown-5m",
        "slug_15m": "eth-updown-15m",
    },
}

# Timeframes
TIMEFRAMES = {
    "5m": {"seconds": 300, "alignment": 300},
    "15m": {"seconds": 900, "alignment": 900},
}

# Position sizing
# 0x8dxd deploys ~$58/window on average. Start much smaller for testing.
INITIAL_BET_SIZE = 2.00          # Total $ per side per window (start small!)
MAX_BET_SIZE = 20.00             # Maximum after proving it works
UNDERDOG_MULTIPLIER = 3.0        # Buy 3x more of underdog vs initial position

# Entry thresholds
PHASE1_ENTRY_DELAY = 5           # Seconds after window open to start Phase 1
PHASE1_YES_PRICE = 0.50          # Initial YES limit order price
PHASE1_NO_PRICE = 0.50           # Initial NO limit order price

# Phase 2: Underdog accumulation
UNDERDOG_THRESHOLD = 0.20        # Buy aggressively when a side drops below this
UNDERDOG_CHEAP_THRESHOLD = 0.10  # Buy VERY aggressively below this
UNDERDOG_PENNY_THRESHOLD = 0.05  # Maximum accumulation below this

# Phase 3: Sell/rotation — DISABLED for first 50 windows per spec
SELL_ENABLED = False             # Start disabled, enable after first 50 windows
SELL_LOSS_THRESHOLD = 0.70       # Sell losing side if it drops below this % of entry
ROTATION_ENABLED = False         # Rotate capital from losing side to underdog

# Pair cost targets
PAIR_COST_TARGET = 0.95          # Ideal: guaranteed 5% profit
PAIR_COST_MAX = 1.10             # Stop buying if pair cost exceeds this
PAIR_COST_GUARANTEED = 1.00      # Below this = guaranteed profit regardless

# Safety limits
MAX_DAILY_LOSS = 50.00           # Stop all trading if daily loss exceeds this
MAX_WINDOWS_PER_DAY = 300        # Sanity limit
MAX_EXPOSURE_PER_WINDOW = 50.00  # Never deploy more than this per window

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

    # YES side
    yes_shares: float = 0.0
    yes_cost: float = 0.0
    yes_orders: int = 0

    # NO side
    no_shares: float = 0.0
    no_cost: float = 0.0
    no_orders: int = 0

    # Sells
    yes_sold_shares: float = 0.0
    yes_sold_revenue: float = 0.0
    no_sold_shares: float = 0.0
    no_sold_revenue: float = 0.0

    # Chainlink resolution tracking
    chainlink_start_price: float = 0.0

    # Phase tracking
    phase: int = 0  # 0=waiting, 1=initial, 2=underdog, 3=rotation
    phase2_triggered: bool = False

    # Status
    status: str = "active"  # active, resolved, error
    resolution: str = ""    # UP, DOWN
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
            return 1.0  # Can't compute yet
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
    def guaranteed_profit(self) -> float:
        """Profit if the WORSE side wins. Negative = not yet hedged."""
        min_shares = min(self.net_yes_shares, self.net_no_shares)
        return min_shares - self.total_cost

    @property
    def is_arb(self) -> bool:
        """True if profitable regardless of outcome."""
        return self.guaranteed_profit > 0


class SpreadCaptureEngine:
    """
    0x8dxd-style spread capture on Polymarket direction markets.

    For each 5m/15m window:
    Phase 1 (0-10s):  Place limit buys on BOTH YES and NO at ~$0.50
    Phase 2 (10-180s): When underdog crashes, accumulate at $0.02-$0.15
    Phase 3 (60-240s): If momentum reverses, sell losing side, rotate

    Result: pair_cost < $1.00 -> guaranteed profit regardless of outcome
    """

    def __init__(self, rtds):
        """
        Args:
            rtds: PolymarketRTDS instance for live price feeds
        """
        self._rtds = rtds
        self._clob_client = None
        self._running = False
        self._positions: Dict[str, WindowPosition] = {}  # slug -> position
        self._daily_pnl = 0.0
        self._daily_date = ""
        self._daily_windows = 0
        self._total_windows_resolved = 0

        self._init_db()
        self._init_clob()

    def _init_db(self):
        """Create tracking tables."""
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS spread_capture_windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                slug TEXT NOT NULL,
                window_start INTEGER NOT NULL,

                -- YES side
                yes_shares REAL DEFAULT 0,
                yes_cost REAL DEFAULT 0,
                yes_avg_price REAL DEFAULT 0,
                yes_orders INTEGER DEFAULT 0,

                -- NO side
                no_shares REAL DEFAULT 0,
                no_cost REAL DEFAULT 0,
                no_avg_price REAL DEFAULT 0,
                no_orders INTEGER DEFAULT 0,

                -- Sells
                yes_sold_shares REAL DEFAULT 0,
                yes_sold_revenue REAL DEFAULT 0,
                no_sold_shares REAL DEFAULT 0,
                no_sold_revenue REAL DEFAULT 0,

                -- Key metrics
                pair_cost REAL,
                total_deployed REAL DEFAULT 0,
                guaranteed_profit REAL,
                is_arb INTEGER DEFAULT 0,

                -- Resolution (Chainlink-based)
                chainlink_start REAL,
                chainlink_end REAL,
                resolution TEXT,  -- UP or DOWN

                -- P&L
                pnl REAL,
                pnl_pct REAL,

                -- Phases reached
                max_phase INTEGER DEFAULT 0,
                phase2_triggered INTEGER DEFAULT 0,

                -- Status
                status TEXT DEFAULT 'active',

                -- Timestamps
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                resolved_at TEXT,

                UNIQUE(asset, timeframe, window_start)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS spread_capture_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_slug TEXT NOT NULL,
                side TEXT NOT NULL,  -- YES, NO, SELL_YES, SELL_NO
                price REAL NOT NULL,
                shares REAL NOT NULL,
                amount_usd REAL NOT NULL,
                order_id TEXT,
                phase INTEGER,
                timestamp REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Spread capture DB initialized")

    def _init_clob(self):
        """Initialize CLOB client."""
        try:
            from py_clob_client.client import ClobClient

            if not os.path.exists(SECRETS_PATH):
                logger.error(f"No secrets at {SECRETS_PATH}")
                return

            with open(SECRETS_PATH) as f:
                secrets = json.load(f)

            self._clob_client = ClobClient(
                host="https://clob.polymarket.com",
                chain_id=137,
                key=secrets["private_key"],
                creds={
                    "api_key": secrets["api_key"],
                    "api_secret": secrets["api_secret"],
                    "api_passphrase": secrets["api_passphrase"],
                },
                signature_type=secrets.get("signature_type", 1),
                funder=secrets.get("funder", secrets["wallet_address"]),
            )
            logger.info("CLOB client initialized for spread capture")
        except Exception as e:
            logger.error(f"CLOB init failed: {e}")

    # ═══════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════

    async def run(self):
        """
        Main loop. Runs every second.

        For each active window, evaluates which phase to execute.
        Also checks for new windows to enter and old windows to resolve.
        """
        self._running = True
        logger.info(
            f"Spread capture engine STARTED\n"
            f"  Assets: {list(ASSETS.keys())}\n"
            f"  Timeframes: {list(TIMEFRAMES.keys())}\n"
            f"  Initial size: ${INITIAL_BET_SIZE}/side\n"
            f"  Pair cost target: ${PAIR_COST_TARGET}\n"
            f"  Sells enabled: {SELL_ENABLED}"
        )

        while self._running:
            try:
                now = time.time()

                # Daily reset
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if today != self._daily_date:
                    if self._daily_date:
                        logger.info(f"Daily reset. Yesterday: {self._daily_windows} windows, ${self._daily_pnl:+.2f}")
                    self._daily_date = today
                    self._daily_pnl = 0.0
                    self._daily_windows = 0

                # Safety check
                if self._daily_pnl <= -MAX_DAILY_LOSS:
                    logger.warning(f"Daily loss limit (${self._daily_pnl:.2f}). Paused.")
                    await asyncio.sleep(60)
                    continue

                # Check for new windows to enter
                for asset, config in ASSETS.items():
                    for tf_name, tf_config in TIMEFRAMES.items():
                        alignment = tf_config["alignment"]
                        window_start = int(math.floor(now / alignment) * alignment)
                        slug = f"{config[f'slug_{tf_name}']}-{window_start}"
                        elapsed = now - window_start

                        # New window? Start Phase 1
                        if slug not in self._positions and elapsed < PHASE1_ENTRY_DELAY + 10:
                            if elapsed >= PHASE1_ENTRY_DELAY:
                                await self._enter_window(asset, tf_name, window_start, slug, config)

                # Process active positions
                for slug, pos in list(self._positions.items()):
                    if pos.status != "active":
                        continue

                    elapsed = now - pos.window_start
                    window_seconds = TIMEFRAMES[pos.timeframe]["seconds"]

                    # Window expired? Resolve.
                    if elapsed > window_seconds + 30:
                        await self._resolve_window(pos)
                        continue

                    # Get current market prices from CLOB
                    yes_price, no_price = self._get_clob_prices(pos)
                    if yes_price is None:
                        continue

                    # Get Chainlink direction
                    cl_direction = self._rtds.get_resolution_direction(
                        pos.asset, pos.window_start
                    )

                    # -- PHASE 2: Underdog accumulation --
                    if elapsed > 10 and elapsed < window_seconds - 30:
                        # Is either side an underdog?
                        if yes_price <= UNDERDOG_THRESHOLD:
                            await self._accumulate_underdog(
                                pos, "YES", yes_price, elapsed, cl_direction
                            )

                        if no_price <= UNDERDOG_THRESHOLD:
                            await self._accumulate_underdog(
                                pos, "NO", no_price, elapsed, cl_direction
                            )

                    # -- PHASE 3: Sell/rotation --
                    if SELL_ENABLED and elapsed > 60 and elapsed < window_seconds - 60:
                        await self._check_rotation(pos, yes_price, no_price, cl_direction)

                    # Log status periodically
                    if int(now) % 30 < 2:
                        logger.info(
                            f"[SC] {pos.asset} {pos.timeframe} | "
                            f"YES: {pos.net_yes_shares:.1f}sh@${pos.avg_yes_price:.3f} | "
                            f"NO: {pos.net_no_shares:.1f}sh@${pos.avg_no_price:.3f} | "
                            f"Pair: ${pos.pair_cost:.3f} | "
                            f"{'ARB' if pos.is_arb else 'DIR'} | "
                            f"CL:{cl_direction or '?'}"
                        )

                # Resolve check every 30 seconds
                if int(now) % 30 == 0:
                    await self._check_resolutions()

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                logger.info("Spread capture engine cancelled — shutting down")
                return
            except Exception as e:
                logger.error(f"Spread capture loop error: {e}")
                await asyncio.sleep(5)

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: INITIAL ENTRY
    # ═══════════════════════════════════════════════════════════

    async def _enter_window(self, asset: str, timeframe: str,
                             window_start: int, slug: str, config: dict):
        """
        Phase 1: Place initial limit orders on BOTH sides.

        0x8dxd enters within 7-11 seconds of window open.
        Initial orders at ~$0.45-$0.50 per side.
        """
        # Record Chainlink start price for resolution tracking
        self._rtds.record_window_start(asset, window_start)

        # Fetch market data (token IDs)
        market_data = self._fetch_market(slug)
        if not market_data:
            logger.debug(f"[SC] {asset} {timeframe}: market {slug} not found")
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
            phase=1,
        )

        self._positions[slug] = pos
        self._daily_windows += 1

        # Place initial YES buy
        yes_price = min(PHASE1_YES_PRICE, market_data.get("crowd_yes", 0.50))
        yes_shares = INITIAL_BET_SIZE / yes_price if yes_price > 0 else 0

        await self._place_order(pos, "YES", yes_price, yes_shares, phase=1)

        # Place initial NO buy
        no_price = min(PHASE1_NO_PRICE, 1.0 - market_data.get("crowd_yes", 0.50))
        no_shares = INITIAL_BET_SIZE / no_price if no_price > 0 else 0

        await self._place_order(pos, "NO", no_price, no_shares, phase=1)

        logger.info(
            f"[SC] *** ENTERED: {asset} {timeframe} ***\n"
            f"  Slug: {slug}\n"
            f"  YES: {yes_shares:.1f}sh @ ${yes_price:.2f}\n"
            f"  NO: {no_shares:.1f}sh @ ${no_price:.2f}\n"
            f"  Initial pair cost: ${yes_price + no_price:.3f}\n"
            f"  Chainlink start: ${pos.chainlink_start_price:.2f}"
        )

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: UNDERDOG ACCUMULATION
    # ═══════════════════════════════════════════════════════════

    async def _accumulate_underdog(self, pos: WindowPosition, side: str,
                                     current_price: float, elapsed: float,
                                     cl_direction: Optional[str]):
        """
        Phase 2: Buy the cheap side aggressively.

        When the crowd piles on one direction, the underdog crashes.
        We accumulate it at pennies. This drives pair cost below $1.00.

        0x8dxd buys at $0.02-$0.15 on the underdog side.
        76% of his trades happen AFTER the first 30 seconds.
        """
        # Don't over-accumulate
        if pos.total_cost >= MAX_EXPOSURE_PER_WINDOW:
            return

        # Don't buy if pair cost already too high
        if pos.pair_cost > PAIR_COST_MAX and pos.yes_shares > 0 and pos.no_shares > 0:
            return

        # Size based on how cheap the underdog is
        if current_price <= UNDERDOG_PENNY_THRESHOLD:
            # Pennies! Buy aggressively
            buy_amount = INITIAL_BET_SIZE * UNDERDOG_MULTIPLIER
        elif current_price <= UNDERDOG_CHEAP_THRESHOLD:
            # Cheap — buy moderately
            buy_amount = INITIAL_BET_SIZE * 2.0
        elif current_price <= UNDERDOG_THRESHOLD:
            # Mild discount — buy small
            buy_amount = INITIAL_BET_SIZE
        else:
            return

        # Cap total exposure
        buy_amount = min(buy_amount, MAX_EXPOSURE_PER_WINDOW - pos.total_cost)
        if buy_amount < 0.50:
            return

        shares = buy_amount / current_price if current_price > 0 else 0

        await self._place_order(pos, side, current_price, shares, phase=2)
        pos.phase = 2
        pos.phase2_triggered = True

        logger.info(
            f"[SC] UNDERDOG: {pos.asset} {side} @ ${current_price:.3f} "
            f"({shares:.1f}sh for ${buy_amount:.2f}) | "
            f"Pair cost now: ${pos.pair_cost:.3f} | "
            f"CL says: {cl_direction or '?'}"
        )

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: SELL / ROTATION
    # ═══════════════════════════════════════════════════════════

    async def _check_rotation(self, pos: WindowPosition, yes_price: float,
                                no_price: float, cl_direction: Optional[str]):
        """
        Phase 3: Sell the losing side and rotate into the new underdog.

        0x8dxd sells in 15% of trades, recovering $384/window average.
        Triggers when momentum clearly reverses.

        This is optional and can be disabled (SELL_ENABLED=False).
        """
        if not cl_direction:
            return

        # If Chainlink says UP, but we have heavy NO position and YES is cheap
        if cl_direction == "UP" and pos.net_no_shares > pos.net_yes_shares * 1.5:
            if no_price < pos.avg_no_price * SELL_LOSS_THRESHOLD:
                # NO is losing badly — sell some to recover capital
                sell_shares = pos.net_no_shares * 0.3  # Sell 30%
                if sell_shares > 0 and no_price > 0.01:
                    await self._sell_order(pos, "NO", no_price, sell_shares)
                    pos.phase = 3
                    logger.info(
                        f"[SC] ROTATION: Selling {sell_shares:.1f} NO @ ${no_price:.3f} "
                        f"(CL says UP, cutting losses)"
                    )

        # Mirror for DOWN
        elif cl_direction == "DOWN" and pos.net_yes_shares > pos.net_no_shares * 1.5:
            if yes_price < pos.avg_yes_price * SELL_LOSS_THRESHOLD:
                sell_shares = pos.net_yes_shares * 0.3
                if sell_shares > 0 and yes_price > 0.01:
                    await self._sell_order(pos, "YES", yes_price, sell_shares)
                    pos.phase = 3
                    logger.info(
                        f"[SC] ROTATION: Selling {sell_shares:.1f} YES @ ${yes_price:.3f} "
                        f"(CL says DOWN, cutting losses)"
                    )

    # ═══════════════════════════════════════════════════════════
    # ORDER EXECUTION
    # ═══════════════════════════════════════════════════════════

    async def _place_order(self, pos: WindowPosition, side: str,
                            price: float, shares: float, phase: int) -> Optional[str]:
        """
        Place a BUY order on the CLOB.

        Uses limit orders at round cent prices (87% of 0x8dxd's orders).
        """
        if not self._clob_client:
            logger.warning("[SC] No CLOB client")
            return None

        # Round price to nearest cent (0x8dxd uses round cents)
        price = round(price, 2)
        shares = round(shares, 2)

        if price <= 0 or shares <= 0:
            return None

        token_id = pos.yes_token_id if side == "YES" else pos.no_token_id
        if not token_id:
            logger.warning(f"[SC] No token ID for {side}")
            return None

        try:
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=shares,
                side=BUY,
            )

            result = self._clob_client.create_and_post_order(order_args)
            order_id = result.get("orderID", "unknown") if result else "error"

            # Update position
            amount = price * shares
            if side == "YES":
                pos.yes_shares += shares
                pos.yes_cost += amount
                pos.yes_orders += 1
            else:
                pos.no_shares += shares
                pos.no_cost += amount
                pos.no_orders += 1

            # Record fill
            self._record_fill(pos.slug, side, price, shares, amount, order_id, phase)

            return order_id

        except Exception as e:
            logger.warning(f"[SC] Order error ({side}): {e}")
            return None

    async def _sell_order(self, pos: WindowPosition, side: str,
                           price: float, shares: float) -> Optional[str]:
        """Place a SELL order."""
        if not self._clob_client:
            return None

        price = round(price, 2)
        shares = round(shares, 2)

        token_id = pos.yes_token_id if side == "YES" else pos.no_token_id
        if not token_id:
            return None

        try:
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import SELL

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=shares,
                side=SELL,
            )

            result = self._clob_client.create_and_post_order(order_args)
            order_id = result.get("orderID", "unknown") if result else "error"

            revenue = price * shares
            if side == "YES":
                pos.yes_sold_shares += shares
                pos.yes_sold_revenue += revenue
            else:
                pos.no_sold_shares += shares
                pos.no_sold_revenue += revenue

            self._record_fill(
                pos.slug, f"SELL_{side}", price, shares, revenue, order_id, 3
            )

            return order_id

        except Exception as e:
            logger.warning(f"[SC] Sell error ({side}): {e}")
            return None

    def _record_fill(self, slug, side, price, shares, amount, order_id, phase):
        """Record a fill in the database."""
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

    def _fetch_market(self, slug: str) -> Optional[dict]:
        """Fetch market data (token IDs, crowd price) from Gamma API."""
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
            logger.debug(f"Market fetch error: {e}")
            return None

    def _get_clob_prices(self, pos: WindowPosition) -> Tuple[Optional[float], Optional[float]]:
        """Get current YES and NO prices from the CLOB orderbook."""
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

    # ═══════════════════════════════════════════════════════════
    # RESOLUTION
    # ═══════════════════════════════════════════════════════════

    async def _resolve_window(self, pos: WindowPosition):
        """Resolve a completed window and compute P&L."""
        # Get final Chainlink price
        cl_price = self._rtds.get_chainlink_price(pos.asset)

        if cl_price and pos.chainlink_start_price:
            went_up = cl_price >= pos.chainlink_start_price
        else:
            # Fallback: check Gamma API for resolution
            went_up = self._check_gamma_resolution(pos.slug)

        if went_up is None:
            return  # Can't resolve yet

        pos.resolution = "UP" if went_up else "DOWN"

        # Calculate P&L
        if went_up:
            # YES wins: collect net_yes_shares * $1
            payout = pos.net_yes_shares * 1.0 * 0.98  # 2% fee estimate
        else:
            # NO wins: collect net_no_shares * $1
            payout = pos.net_no_shares * 1.0 * 0.98

        pos.pnl = payout - pos.total_cost
        pos.status = "resolved"

        self._daily_pnl += pos.pnl
        self._total_windows_resolved += 1

        # Save to DB
        self._save_window(pos)

        # Log
        marker = "+++" if pos.pnl > 0 else "---"
        arb_tag = "ARB" if pos.is_arb else "DIR"
        logger.info(
            f"[SC] {marker} RESOLVED: {pos.asset} {pos.timeframe} -> {pos.resolution}\n"
            f"  YES: {pos.net_yes_shares:.1f}sh @ ${pos.avg_yes_price:.3f}\n"
            f"  NO: {pos.net_no_shares:.1f}sh @ ${pos.avg_no_price:.3f}\n"
            f"  Pair cost: ${pos.pair_cost:.3f} ({arb_tag})\n"
            f"  P&L: ${pos.pnl:+.2f} | Daily: ${self._daily_pnl:+.2f}"
        )

        # Remove from active
        del self._positions[pos.slug]

    async def _check_resolutions(self):
        """Check Gamma API for any resolved markets we missed."""
        now = time.time()
        for slug, pos in list(self._positions.items()):
            if pos.status != "active":
                continue
            window_end = pos.window_start + TIMEFRAMES[pos.timeframe]["seconds"]
            if now > window_end + 60:
                await self._resolve_window(pos)

    def _check_gamma_resolution(self, slug: str) -> Optional[bool]:
        """Fallback: check Gamma API for market resolution."""
        try:
            resp = requests.get(f"{GAMMA_BASE}/markets", params={"slug": slug}, timeout=5)
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

    def _save_window(self, pos: WindowPosition):
        """Save completed window to database."""
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
                 pnl_pct, max_phase, phase2_triggered, status, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                pos.phase, 1 if pos.phase2_triggered else 0,
                pos.status,
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"[SC] DB save error: {e}")

    # ═══════════════════════════════════════════════════════════
    # STATS & DASHBOARD
    # ═══════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """Stats for dashboard."""
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
                    COALESCE(AVG(pnl), 0) as avg_pnl
                FROM spread_capture_windows
                WHERE status = 'resolved'
            """).fetchone()

            total = stats["total"] or 0
            wins = stats["wins"] or 0

            result = {
                "strategy": "0x8dxd_spread_capture",
                "total_windows": total,
                "wins": wins,
                "losses": stats["losses"] or 0,
                "win_rate": round(wins / max(total, 1) * 100, 1),
                "arb_windows": stats["arb_windows"] or 0,
                "arb_rate": round((stats["arb_windows"] or 0) / max(total, 1) * 100, 1),
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_pair_cost": round(stats["avg_pair_cost"], 3),
                "avg_pnl_per_window": round(stats["avg_pnl"], 3),
                "active_positions": len([p for p in self._positions.values() if p.status == "active"]),
                "daily_pnl": round(self._daily_pnl, 2),
                "daily_windows": self._daily_windows,
            }

            conn.close()
            return result
        except Exception as e:
            logger.warning(f"[SC] Stats error: {e}")
            return {"strategy": "0x8dxd_spread_capture", "error": str(e)}

    def get_active_positions(self) -> list:
        """Return active position details for dashboard."""
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
                "is_arb": pos.is_arb,
                "guaranteed_profit": round(pos.guaranteed_profit, 2),
                "phase": pos.phase,
                "chainlink_direction": self._rtds.get_resolution_direction(
                    pos.asset, pos.window_start
                ),
            })
        return result

    async def stop(self):
        self._running = False
