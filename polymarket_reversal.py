"""
polymarket_reversal.py -- "BTC Already Told You" Contrarian Strategy

Monitors BTC price WITHIN each 5-minute Polymarket window. When BTC
reverses direction but altcoin markets haven't repriced yet, places
contrarian bets on the altcoin at extreme odds (typically $0.10-$0.20).

This strategy exploits three asymmetries:
1. BTC leads altcoins by 30-120 seconds
2. Polymarket crowd watches the altcoin, not BTC
3. Binary resolution only cares about close vs open

Expected accuracy: 20-30% (only needs >15% to profit at $0.15 entry)
Expected EV per bet: +$0.50-$1.00 on $5 bets
Expected frequency: 5-15 eligible setups per day across 3 altcoins

Runs alongside Strategy A (early entry), not as a replacement.
"""

import logging
import time
import math
import json
import sqlite3
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


# ===================================================================
# CONFIGURATION
# ===================================================================

# Assets we trade (altcoins only -- BTC and ETH have no leader)
REVERSAL_ASSETS = {
    "SOL": {"pair": "SOL-USD", "slug_prefix": "sol-updown-5m", "leader": "BTC-USD"},
    "DOGE": {"pair": "DOGE-USD", "slug_prefix": "doge-updown-5m", "leader": "BTC-USD"},
    "XRP": {"pair": "XRP-USD", "slug_prefix": "xrp-updown-5m", "leader": "BTC-USD"},
}

# Timing parameters
WINDOW_SECONDS = 300                # 5-minute window
ENTRY_EARLIEST_SEC = 195            # Don't enter before t=3:15 (need trend + reversal)
ENTRY_LATEST_SEC = 255              # Don't enter after t=4:15 (need time for propagation)
MIN_TREND_DURATION_SEC = 120        # BTC must have been trending for 2+ minutes
MIN_REVERSAL_DURATION_SEC = 20      # BTC reversal must persist for 20+ seconds
BTC_PRICE_SAMPLE_INTERVAL = 10      # Sample BTC price every 10 seconds

# Signal thresholds
MIN_BTC_TREND_PCT = 0.08            # BTC must have moved 0.08%+ in original direction
MIN_BTC_REVERSAL_PCT = 0.04         # BTC must have reversed 0.04%+ from its extreme
MIN_DIVERGENCE_PCT = 0.05           # Altcoin must still be "behind" BTC's reversal
MAX_ENTRY_PRICE = 0.25              # Never buy more expensive than $0.25 (need 4:1 payout)
MIN_ENTRY_PRICE = 0.03              # Below $0.03 means market has no liquidity
IDEAL_ENTRY_PRICE = 0.15            # Sweet spot for risk/reward

# Sizing
BASE_BET_SIZE = 3.0                 # Small bets -- we're betting on low-probability high-payout
MAX_BET_SIZE = 8.0                  # Cap per bet
MAX_OPEN_REVERSAL_BETS = 3          # Max simultaneous reversal bets
MAX_DAILY_REVERSAL_LOSS = 30.0      # Stop if we lose $30 in a day on reversals
BANKROLL_FRACTION = 0.01            # Max 1% of bankroll per reversal bet

# Polymarket API
GAMMA_BASE = "https://gamma-api.polymarket.com"


class BTCWindowTracker:
    """
    Tracks BTC price trajectory within each 5-minute window.

    Maintains a ring buffer of (timestamp, price) samples taken every
    ~10 seconds. Detects trends and reversals within the current window.
    """

    def __init__(self, max_samples: int = 50):
        self._samples: deque = deque(maxlen=max_samples)
        self._current_window: int = 0
        self._window_open_price: Optional[float] = None

    def update(self, btc_price: float, timestamp: Optional[float] = None):
        """Record a BTC price sample."""
        ts = timestamp or time.time()
        window = int(math.floor(ts / WINDOW_SECONDS) * WINDOW_SECONDS)

        # New window?
        if window != self._current_window:
            self._current_window = window
            self._window_open_price = btc_price
            self._samples.clear()

        self._samples.append((ts, btc_price))

    @property
    def window_start(self) -> int:
        return self._current_window

    @property
    def elapsed_seconds(self) -> float:
        if not self._samples:
            return 0
        return self._samples[-1][0] - self._current_window

    @property
    def window_open_price(self) -> Optional[float]:
        return self._window_open_price

    def detect_reversal(self) -> Optional[Dict]:
        """
        Detect if BTC has reversed direction within this window.

        A reversal means:
        1. BTC trended in one direction for 2+ minutes
        2. BTC has now moved in the opposite direction for 20+ seconds
        3. The reversal magnitude is at least MIN_BTC_REVERSAL_PCT

        Returns None if no reversal detected, or dict with:
            trend_direction: "up" or "down" (the ORIGINAL trend)
            reversal_direction: "up" or "down" (the NEW direction, opposite of trend)
            trend_magnitude_pct: how far BTC moved in the original trend
            reversal_magnitude_pct: how far BTC has reversed
            trend_peak_time: when BTC hit its extreme (before reversing)
            reversal_duration_sec: how long the reversal has been going
            btc_net_move_pct: overall BTC move from window open to now
        """
        if len(self._samples) < 5:
            return None

        if not self._window_open_price or self._window_open_price <= 0:
            return None

        open_price = self._window_open_price
        samples = list(self._samples)

        # Find the price extreme (high or low) within the window
        prices = [p for _, p in samples]
        timestamps = [t for t, _ in samples]

        max_price = max(prices)
        min_price = min(prices)
        max_idx = prices.index(max_price)
        min_idx = prices.index(min_price)
        current_price = prices[-1]
        current_time = timestamps[-1]

        # Was BTC trending UP then reversed DOWN?
        up_then_down = None
        if max_price > open_price:
            up_move_pct = (max_price - open_price) / open_price * 100
            reversal_pct = (max_price - current_price) / open_price * 100
            reversal_duration = current_time - timestamps[max_idx]

            if (up_move_pct >= MIN_BTC_TREND_PCT and
                reversal_pct >= MIN_BTC_REVERSAL_PCT and
                reversal_duration >= MIN_REVERSAL_DURATION_SEC and
                timestamps[max_idx] - self._current_window >= MIN_TREND_DURATION_SEC):

                up_then_down = {
                    "trend_direction": "up",
                    "reversal_direction": "down",
                    "trend_magnitude_pct": round(up_move_pct, 4),
                    "reversal_magnitude_pct": round(reversal_pct, 4),
                    "trend_peak_time": timestamps[max_idx],
                    "reversal_duration_sec": round(reversal_duration, 1),
                    "btc_net_move_pct": round((current_price - open_price) / open_price * 100, 4),
                }

        # Was BTC trending DOWN then reversed UP?
        down_then_up = None
        if min_price < open_price:
            down_move_pct = (open_price - min_price) / open_price * 100
            reversal_pct = (current_price - min_price) / open_price * 100
            reversal_duration = current_time - timestamps[min_idx]

            if (down_move_pct >= MIN_BTC_TREND_PCT and
                reversal_pct >= MIN_BTC_REVERSAL_PCT and
                reversal_duration >= MIN_REVERSAL_DURATION_SEC and
                timestamps[min_idx] - self._current_window >= MIN_TREND_DURATION_SEC):

                down_then_up = {
                    "trend_direction": "down",
                    "reversal_direction": "up",
                    "trend_magnitude_pct": round(down_move_pct, 4),
                    "reversal_magnitude_pct": round(reversal_pct, 4),
                    "trend_peak_time": timestamps[min_idx],
                    "reversal_duration_sec": round(reversal_duration, 1),
                    "btc_net_move_pct": round((current_price - open_price) / open_price * 100, 4),
                }

        # Return the stronger reversal (if both exist)
        if up_then_down and down_then_up:
            if up_then_down["reversal_magnitude_pct"] > down_then_up["reversal_magnitude_pct"]:
                return up_then_down
            return down_then_up

        return up_then_down or down_then_up


class ReversalStrategy:
    """
    The "BTC Already Told You" contrarian strategy.

    Every ~10 seconds during each 5-minute window:
    1. Update BTC price tracker
    2. Check if we're in the entry window (t=3:15 to t=4:15)
    3. Detect if BTC has reversed direction
    4. Check if altcoin market is still priced for the OLD trend
    5. If all conditions met: place contrarian bet on the altcoin

    When a liquidation cascade is active (from the Binance Futures feed),
    the confidence multiplier increases: forced selling has a mechanical end,
    so the recovery is predictable.

    This runs as a SEPARATE strategy alongside Strategy A.
    Strategy A bets early (t=0) with the ML prediction.
    This strategy bets late (t=4) against the crowd, using BTC lead-lag.
    """

    def __init__(self, db_path: str = "data/renaissance_bot.db"):
        self.db_path = db_path
        self.btc_tracker = BTCWindowTracker()
        self.paper_mode = True  # Always paper for now

        # Track daily P&L for circuit breaker
        self._daily_pnl = 0.0
        self._daily_reset_date = ""
        self._open_bets = 0

        # Track which windows we've already bet on (prevent duplicates)
        self._bet_windows: set = set()  # (asset, window_start)

        # Reference to unified price feed (set externally by bot)
        self._price_feed = None

        self._init_db()

    def _init_db(self):
        """Create the reversal positions table."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS polymarket_reversal_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Market identification
                asset TEXT NOT NULL,
                slug TEXT,
                window_start INTEGER NOT NULL,

                -- Entry details
                direction TEXT NOT NULL,          -- "UP" or "DOWN" (what we bet)
                entry_price REAL NOT NULL,        -- what we paid per share ($0.10-$0.25)
                bet_amount REAL NOT NULL,          -- total $ risked
                shares REAL NOT NULL,             -- bet_amount / entry_price

                -- BTC signal that triggered this
                btc_trend_direction TEXT,          -- "up" or "down" (the OLD trend)
                btc_reversal_direction TEXT,       -- "up" or "down" (the NEW direction)
                btc_trend_magnitude_pct REAL,
                btc_reversal_magnitude_pct REAL,
                btc_reversal_duration_sec REAL,

                -- Altcoin state at entry
                altcoin_move_pct REAL,            -- how much altcoin moved in old direction
                crowd_price_at_entry REAL,        -- what crowd thought (e.g., 0.85 for DOWN)
                divergence_pct REAL,              -- BTC reversed but altcoin hasn't
                seconds_remaining REAL,           -- time left in window when we entered

                -- Resolution
                status TEXT DEFAULT 'open',       -- open, won, lost
                pnl REAL,
                resolved_at TEXT,
                altcoin_close_vs_open TEXT,        -- "up" or "down"

                -- Metadata
                opened_at TEXT DEFAULT CURRENT_TIMESTAMP,
                strategy TEXT DEFAULT 'reversal',

                UNIQUE(asset, window_start)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rev_status
            ON polymarket_reversal_bets(status)
        """)
        conn.commit()
        conn.close()

    def update_btc_price(self, btc_price: float):
        """Call this every cycle (or more frequently) with current BTC price."""
        self.btc_tracker.update(btc_price)

    def check_and_execute(
        self,
        current_prices: Dict[str, float],
        bankroll: float = 500.0,
    ) -> List[Dict]:
        """
        Main entry point. Called every cycle (~5-10 seconds ideally,
        or every 5-minute cycle at minimum).

        Checks all altcoin markets for reversal opportunities.
        Returns list of bets placed (or empty list).
        """
        bets_placed = []
        now = time.time()

        # Reset daily P&L tracker
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_reset_date = today
            self._daily_pnl = 0.0

        # Daily loss circuit breaker
        if self._daily_pnl <= -MAX_DAILY_REVERSAL_LOSS:
            logger.info(
                f"[REVERSAL] Daily loss limit hit (${self._daily_pnl:.2f}). "
                f"No more reversal bets today."
            )
            return []

        # Check if we're in the entry window
        elapsed = self.btc_tracker.elapsed_seconds
        if elapsed < ENTRY_EARLIEST_SEC or elapsed > ENTRY_LATEST_SEC:
            return []  # Not in the right part of the window

        # Detect BTC reversal
        reversal = self.btc_tracker.detect_reversal()
        if not reversal:
            return []  # No reversal detected

        logger.info(
            f"[REVERSAL] BTC reversal detected: "
            f"{reversal['trend_direction']}->{reversal['reversal_direction']} "
            f"trend={reversal['trend_magnitude_pct']:.3f}% "
            f"reversal={reversal['reversal_magnitude_pct']:.3f}% "
            f"duration={reversal['reversal_duration_sec']:.0f}s "
            f"net={reversal['btc_net_move_pct']:+.3f}%"
        )

        # Check max open bets
        conn = sqlite3.connect(self.db_path)
        open_count = conn.execute(
            "SELECT COUNT(*) FROM polymarket_reversal_bets WHERE status='open'"
        ).fetchone()[0]
        conn.close()

        if open_count >= MAX_OPEN_REVERSAL_BETS:
            return []

        # Check each altcoin
        btc_price = current_prices.get("BTC-USD", 0)
        btc_open = self.btc_tracker.window_open_price

        for asset, config in REVERSAL_ASSETS.items():
            pair = config["pair"]
            alt_price = current_prices.get(pair, 0)

            if not alt_price or not btc_price or not btc_open:
                continue

            # Skip if we already bet this asset in this window
            window_key = (asset, self.btc_tracker.window_start)
            if window_key in self._bet_windows:
                continue

            # Check altcoin divergence: has it caught up to BTC's reversal?
            # We need the altcoin's open price for this window
            alt_open = self._get_altcoin_window_open(pair)
            if not alt_open:
                continue

            alt_move_pct = (alt_price - alt_open) / alt_open * 100

            # The key check: BTC reversed but altcoin still shows old trend
            # BTC was down, now reversing up -> altcoin should still be down
            # BTC was up, now reversing down -> altcoin should still be up

            if reversal["reversal_direction"] == "up":
                # BTC was DOWN, now going UP. Altcoin should still be DOWN.
                if alt_move_pct >= 0:
                    continue  # Altcoin already positive, no divergence

                # Our contrarian bet: buy UP (altcoin will follow BTC up)
                bet_direction = "UP"
                divergence = abs(alt_move_pct)  # Altcoin is down X% while BTC reversed up

            elif reversal["reversal_direction"] == "down":
                # BTC was UP, now going DOWN. Altcoin should still be UP.
                if alt_move_pct <= 0:
                    continue  # Altcoin already negative, no divergence

                # Our contrarian bet: buy DOWN (altcoin will follow BTC down)
                bet_direction = "DOWN"
                divergence = abs(alt_move_pct)  # Altcoin is up X% while BTC reversed down
            else:
                continue

            if divergence < MIN_DIVERGENCE_PCT:
                logger.debug(f"[REVERSAL] {asset}: divergence {divergence:.3f}% too small")
                continue

            # Check Polymarket crowd price
            crowd_price = self._get_crowd_price(asset, config["slug_prefix"])
            if crowd_price is None:
                continue

            # Determine our entry price
            if bet_direction == "UP":
                entry_price = crowd_price  # YES price (should be cheap, e.g., $0.15)
            else:
                entry_price = 1.0 - crowd_price  # NO price (should be cheap)

            if entry_price > MAX_ENTRY_PRICE:
                logger.info(
                    f"[REVERSAL] {asset}: entry ${entry_price:.2f} too expensive "
                    f"(max ${MAX_ENTRY_PRICE}). Crowd has already repriced."
                )
                continue

            if entry_price < MIN_ENTRY_PRICE:
                logger.debug(f"[REVERSAL] {asset}: entry ${entry_price:.3f} too cheap (no liquidity)")
                continue

            # Compute bet size
            bet_size = min(
                BASE_BET_SIZE,
                MAX_BET_SIZE,
                bankroll * BANKROLL_FRACTION,
            )

            # Scale up slightly for better entries (cheaper = more confident)
            if entry_price <= 0.12:
                bet_size = min(bet_size * 1.5, MAX_BET_SIZE)

            if bet_size < 1.0:
                continue

            # ── Liquidation cascade confidence boost ──
            # When a cascade is active, forced selling is mechanical and has
            # a predictable end. The recovery after cascade exhaustion is the
            # highest-conviction reversal signal.
            liq_boost = 1.0
            liq_info = {}

            if self._price_feed:
                alt_binance_sym = f"{asset}/USDT"
                alt_liq = self._price_feed.get_liquidation_stats(alt_binance_sym)
                btc_liq = self._price_feed.get_liquidation_stats("BTC/USDT")

                if alt_liq:
                    liq_info["altcoin_long_liq_5m"] = alt_liq.get("long_usd_5m", 0)
                    liq_info["altcoin_cascade"] = alt_liq.get("cascade_active", False)
                    liq_info["altcoin_imbalance"] = alt_liq.get("imbalance_ratio", 1.0)

                if btc_liq:
                    liq_info["btc_long_liq_5m"] = btc_liq.get("long_usd_5m", 0)
                    liq_info["btc_cascade"] = btc_liq.get("cascade_active", False)

                if reversal["reversal_direction"] == "up":
                    # BTC reversed UP → we bet altcoin UP
                    # Long cascade = forced selling ending → recovery
                    if alt_liq and alt_liq.get("cascade_active"):
                        liq_boost = 1.5
                        logger.info(
                            f"[REVERSAL] CASCADE BOOST: {asset} long cascade active "
                            f"(${alt_liq.get('long_usd_5m', 0):,.0f} in 5m). "
                            f"Forced selling exhausting → recovery likely."
                        )
                    elif btc_liq and btc_liq.get("cascade_active"):
                        liq_boost = 1.3
                        logger.info(
                            f"[REVERSAL] BTC CASCADE BOOST: BTC long cascade active "
                            f"(${btc_liq.get('long_usd_5m', 0):,.0f}). "
                            f"BTC reversing + cascade ending → altcoin recovery likely."
                        )
                    # Extreme imbalance extra boost
                    if alt_liq and alt_liq.get("imbalance_ratio", 1) > 5.0:
                        liq_boost *= 1.2

                elif reversal["reversal_direction"] == "down":
                    # BTC reversed DOWN → we bet altcoin DOWN
                    # Short squeeze = forced buying ending → sell-off
                    if alt_liq and alt_liq.get("squeeze_active"):
                        liq_boost = 1.5
                    elif btc_liq and btc_liq.get("squeeze_active"):
                        liq_boost = 1.3

            # Apply cascade boost (capped at 2x)
            bet_size = min(bet_size * min(liq_boost, 2.0), MAX_BET_SIZE)

            if bet_size < 1.0:
                continue

            shares = bet_size / entry_price
            potential_profit = shares * (1.0 - entry_price) * 0.99  # 1% fee

            seconds_left = WINDOW_SECONDS - elapsed

            logger.info(
                f"[REVERSAL] *** CONTRARIAN BET: {asset} {bet_direction} ***\n"
                f"  Entry: ${entry_price:.2f} | Bet: ${bet_size:.2f} | "
                f"Potential profit: ${potential_profit:.2f}\n"
                f"  BTC: was {reversal['trend_direction']}, reversed "
                f"{reversal['reversal_direction']} ({reversal['reversal_magnitude_pct']:.3f}%)\n"
                f"  {asset}: still {-alt_move_pct:+.3f}% (divergence: {divergence:.3f}%)\n"
                f"  Crowd: {crowd_price:.0%} UP | Seconds left: {seconds_left:.0f}s\n"
                f"  Breakeven accuracy: {entry_price:.0%} | Payout ratio: "
                f"{(1.0-entry_price)/entry_price:.1f}:1\n"
                f"  Liquidation: boost={liq_boost:.1f}x | "
                f"altcoin_cascade={liq_info.get('altcoin_cascade', False)} | "
                f"btc_cascade={liq_info.get('btc_cascade', False)} | "
                f"alt_long_liqs_5m=${liq_info.get('altcoin_long_liq_5m', 0):,.0f}"
            )

            # Record the bet
            self._record_bet(
                asset=asset,
                slug=f"{config['slug_prefix']}-{self.btc_tracker.window_start}",
                window_start=self.btc_tracker.window_start,
                direction=bet_direction,
                entry_price=entry_price,
                bet_amount=bet_size,
                shares=shares,
                reversal=reversal,
                alt_move_pct=alt_move_pct,
                crowd_price=crowd_price,
                divergence=divergence,
                seconds_remaining=seconds_left,
            )

            self._bet_windows.add(window_key)
            bets_placed.append({
                "asset": asset,
                "direction": bet_direction,
                "entry_price": entry_price,
                "bet_amount": bet_size,
                "potential_profit": potential_profit,
            })

        return bets_placed

    def _get_altcoin_window_open(self, pair: str) -> Optional[float]:
        """
        Get the altcoin's price at the start of the current 5-minute window.

        Try multiple sources:
        1. The bot's price cache (recent bars)
        2. Our five_minute_bars table
        3. The Polymarket oracle start price (if available via API)
        """
        window_start = self.btc_tracker.window_start

        conn = sqlite3.connect(self.db_path)

        # Try five_minute_bars table (supports both pair formats)
        pair_slash = pair.replace("-", "/").replace("USD", "USDT") if "-" in pair else pair
        row = conn.execute("""
            SELECT open FROM five_minute_bars
            WHERE pair IN (?, ?) AND bar_start >= ? AND bar_start <= ?
            ORDER BY ABS(bar_start - ?) ASC
            LIMIT 1
        """, (pair, pair_slash, window_start - 30, window_start + 30, window_start)).fetchone()

        conn.close()

        if row:
            return row[0]

        return None

    def _get_crowd_price(self, asset: str, slug_prefix: str) -> Optional[float]:
        """
        Get the current Polymarket crowd YES price for this asset's 5m market.

        Returns the YES (UP) price, or None if market not found.
        """
        slug = f"{slug_prefix}-{self.btc_tracker.window_start}"

        try:
            resp = requests.get(
                f"{GAMMA_BASE}/markets",
                params={"slug": slug},
                timeout=5,
            )
            if resp.status_code != 200:
                return None

            markets = resp.json()
            if not isinstance(markets, list) or not markets:
                return None

            market = markets[0]

            # Don't bet on closed/resolved markets
            if market.get("closed") or market.get("resolved"):
                return None

            prices = market.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)

            if prices and len(prices) >= 1:
                return float(prices[0])  # YES price

            return None

        except Exception as e:
            logger.debug(f"[REVERSAL] Crowd price fetch failed for {slug}: {e}")
            return None

    def _record_bet(self, **kwargs):
        """Record a reversal bet in the database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO polymarket_reversal_bets
                (asset, slug, window_start, direction, entry_price, bet_amount, shares,
                 btc_trend_direction, btc_reversal_direction,
                 btc_trend_magnitude_pct, btc_reversal_magnitude_pct,
                 btc_reversal_duration_sec,
                 altcoin_move_pct, crowd_price_at_entry, divergence_pct,
                 seconds_remaining)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kwargs["asset"], kwargs["slug"], kwargs["window_start"],
                kwargs["direction"], kwargs["entry_price"],
                kwargs["bet_amount"], kwargs["shares"],
                kwargs["reversal"]["trend_direction"],
                kwargs["reversal"]["reversal_direction"],
                kwargs["reversal"]["trend_magnitude_pct"],
                kwargs["reversal"]["reversal_magnitude_pct"],
                kwargs["reversal"]["reversal_duration_sec"],
                kwargs["alt_move_pct"], kwargs["crowd_price"],
                kwargs["divergence"], kwargs["seconds_remaining"],
            ))
            conn.commit()
        except Exception as e:
            logger.warning(f"[REVERSAL] Failed to record bet: {e}")
        finally:
            conn.close()

    def check_resolutions(self):
        """
        Check if any open reversal bets have resolved.

        For paper trading: compare the altcoin's close price vs open price
        for the window. Uses our five_minute_bars table or Gamma API.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        open_bets = conn.execute(
            "SELECT * FROM polymarket_reversal_bets WHERE status='open'"
        ).fetchall()

        for bet in open_bets:
            window_end = bet["window_start"] + WINDOW_SECONDS
            now = time.time()

            # Don't try to resolve until the window is complete + buffer
            if now < window_end + 60:
                continue

            # Try to determine outcome
            outcome = self._check_outcome(
                bet["asset"], bet["window_start"], conn
            )

            if outcome is None:
                # Check if it's been too long (>10 minutes past window end)
                if now > window_end + 600:
                    # Mark as lost by default (couldn't verify)
                    conn.execute("""
                        UPDATE polymarket_reversal_bets
                        SET status='lost', pnl=?, resolved_at=?,
                            altcoin_close_vs_open='unknown'
                        WHERE id=?
                    """, (-bet["bet_amount"], datetime.now(timezone.utc).isoformat(), bet["id"]))
                    conn.commit()
                    self._daily_pnl -= bet["bet_amount"]
                    logger.warning(
                        f"[REVERSAL] {bet['asset']}: timed out, marked as LOST "
                        f"(-${bet['bet_amount']:.2f})"
                    )
                continue

            # Determine win/loss
            altcoin_went = outcome  # "up" or "down"

            if ((bet["direction"] == "UP" and altcoin_went == "up") or
                (bet["direction"] == "DOWN" and altcoin_went == "down")):
                # WON
                pnl = bet["shares"] * (1.0 - bet["entry_price"]) * 0.99 - bet["bet_amount"]
                status = "won"
                pnl_net = pnl
            else:
                # LOST
                pnl_net = -bet["bet_amount"]
                status = "lost"

            conn.execute("""
                UPDATE polymarket_reversal_bets
                SET status=?, pnl=?, resolved_at=?, altcoin_close_vs_open=?
                WHERE id=?
            """, (status, pnl_net, datetime.now(timezone.utc).isoformat(),
                  altcoin_went, bet["id"]))
            conn.commit()

            self._daily_pnl += pnl_net

            emoji = "+++" if status == "won" else "---"
            logger.info(
                f"[REVERSAL] {emoji} {bet['asset']} {bet['direction']}: {status.upper()} "
                f"PnL=${pnl_net:+.2f} | Entry=${bet['entry_price']:.2f} | "
                f"Altcoin closed {altcoin_went} | Daily: ${self._daily_pnl:+.2f}"
            )

        conn.close()

    def _check_outcome(self, asset: str, window_start: int, conn) -> Optional[str]:
        """
        Determine if the altcoin closed UP or DOWN for this window.

        Returns "up", "down", or None if can't determine.
        """
        config = REVERSAL_ASSETS.get(asset)
        if not config:
            return None

        pair = config["pair"]
        pair_slash = pair.replace("-", "/").replace("USD", "USDT") if "-" in pair else pair
        window_end = window_start + WINDOW_SECONDS

        # Method 1: Check five_minute_bars (supports both pair formats)
        open_row = conn.execute("""
            SELECT open, close FROM five_minute_bars
            WHERE pair IN (?, ?) AND bar_start >= ? AND bar_start <= ?
            ORDER BY ABS(bar_start - ?) ASC LIMIT 1
        """, (pair, pair_slash, window_start - 30, window_start + 30, window_start)).fetchone()

        close_row = conn.execute("""
            SELECT close FROM five_minute_bars
            WHERE pair IN (?, ?) AND bar_start >= ? AND bar_start <= ?
            ORDER BY ABS(bar_start - ?) ASC LIMIT 1
        """, (pair, pair_slash, window_end - 30, window_end + 30, window_end)).fetchone()

        if open_row and close_row:
            open_price = open_row[0]
            close_price = close_row[0]
            if close_price >= open_price:
                return "up"
            else:
                return "down"

        # Method 2: Check Gamma API for resolved market
        slug = f"{config['slug_prefix']}-{window_start}"
        try:
            resp = requests.get(
                f"{GAMMA_BASE}/markets",
                params={"slug": slug},
                timeout=5,
            )
            if resp.status_code == 200:
                markets = resp.json()
                if markets and isinstance(markets, list):
                    market = None
                    for m in markets:
                        if m.get("resolved"):
                            market = m
                            break

                    if market:
                        prices = market.get("outcomePrices", "[]")
                        if isinstance(prices, str):
                            prices = json.loads(prices)
                        if prices and len(prices) >= 2:
                            yes_price = float(prices[0])
                            if yes_price >= 0.95:
                                return "up"
                            elif yes_price <= 0.05:
                                return "down"
        except Exception as e:
            logger.warning(f"[REVERSAL] Gamma API resolution check failed for {slug}: {e}")

        return None

    def get_stats(self) -> Dict:
        """Get performance statistics for dashboard."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        stats = conn.execute("""
            SELECT
                COUNT(*) as total_bets,
                SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_bets,
                COALESCE(SUM(CASE WHEN status IN ('won','lost') THEN pnl ELSE 0 END), 0) as total_pnl,
                COALESCE(SUM(bet_amount), 0) as total_wagered,
                AVG(entry_price) as avg_entry_price,
                AVG(btc_reversal_magnitude_pct) as avg_btc_reversal,
                AVG(divergence_pct) as avg_divergence
            FROM polymarket_reversal_bets
        """).fetchone()

        recent = conn.execute("""
            SELECT asset, direction, entry_price, bet_amount, pnl, status,
                   btc_trend_direction, btc_reversal_direction,
                   altcoin_close_vs_open, opened_at
            FROM polymarket_reversal_bets
            ORDER BY opened_at DESC LIMIT 20
        """).fetchall()

        conn.close()

        total = stats["total_bets"] or 0
        resolved = (stats["wins"] or 0) + (stats["losses"] or 0)

        return {
            "total_bets": total,
            "wins": stats["wins"] or 0,
            "losses": stats["losses"] or 0,
            "open_bets": stats["open_bets"] or 0,
            "win_rate": round((stats["wins"] or 0) / resolved * 100, 1) if resolved > 0 else 0,
            "total_pnl": round(stats["total_pnl"], 2),
            "total_wagered": round(stats["total_wagered"] or 0, 2),
            "roi": round(stats["total_pnl"] / (stats["total_wagered"] or 1) * 100, 1),
            "avg_entry_price": round(stats["avg_entry_price"] or 0, 3),
            "avg_btc_reversal_pct": round(stats["avg_btc_reversal"] or 0, 4),
            "avg_divergence_pct": round(stats["avg_divergence"] or 0, 4),
            "daily_pnl": round(self._daily_pnl, 2),
            "recent_bets": [dict(r) for r in recent],
            "breakeven_accuracy": "15-25% depending on entry price",
        }
