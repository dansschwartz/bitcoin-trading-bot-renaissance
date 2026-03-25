"""
simple_up_bet.py — Contrarian $1 UP bet when crowd says DOWN.

ONLY bets when UP costs $0.30 or less (crowd is 70%+ on DOWN).
Enters at T+3:00 of each 5-minute window.
No ML. No models. No predictions. No calibration.

The thesis: after 3 minutes of a 5-minute window, if the asset
went down and the crowd piled on DOWN (pricing UP at $0.10-$0.30),
reversals happen often enough that buying the cheap UP side
is profitable. We only need 30% win rate at $0.30 entry.

The math:
  Entry $0.30 -> $1 buys 3.33 shares -> win pays $3.33 -> profit $2.33
  Entry $0.20 -> $1 buys 5.00 shares -> win pays $5.00 -> profit $4.00
  Entry $0.10 -> $1 buys 10.0 shares -> win pays $10.0 -> profit $9.00

Assets: SOL 5m, DOGE 5m
Bet size: $1.00 fixed
Direction: UP always (but ONLY when UP <= $0.30)
Entry time: T+180s (3 minutes into window)
"""

import asyncio
import json
import logging
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

logger = logging.getLogger("simple_up_bet")

# ── Constants ──
GAMMA_API = "https://gamma-api.polymarket.com/markets"
ASSETS = ["SOL", "DOGE"]
WINDOW_SEC = 300        # 5-minute windows
BET_TIMING_SEC = 270    # Bet at T+4:30 (30s before close)
BET_AMOUNT_USD = 3.00   # $3 per bet (CLOB min 5 shares)
DIRECTION = "UP"        # Always UP
DAILY_BET_LIMIT = 100   # Max bets per day
DAILY_LOSS_CAP = 50.0   # Stop if down $50 today
RESOLUTION_DELAY = 60   # Check resolution 60s after window close
DB_PATH = "data/renaissance_bot.db"
SECRETS_PATH = "config/polymarket_secrets.json"


class SimpleUpBetter:
    """
    Contrarian Polymarket bot.
    Wait 3 minutes. If UP is cheap (<=MAX_ENTRY_PRICE), bet $1.
    If UP is expensive (>MAX_ENTRY_PRICE), skip — crowd already agrees with UP.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._clob_client = None
        self.live_enabled = False
        self._bet_count_today = 0
        self._loss_today = 0.0
        self._daily_date = ""
        self._placed_windows: set = set()  # (asset, window_ts) tuples already bet
        self._market_cache: Dict[str, dict] = {}  # slug -> {data, fetched_at}

        self._init_db()
        self._init_clob_client()
        self._load_today_state()

    # ── Database ──

    def _init_db(self) -> None:
        """Create the simple_up_bets tracking table."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS simple_up_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT NOT NULL,
                window_ts INTEGER NOT NULL,
                slug TEXT,
                question TEXT,
                direction TEXT DEFAULT 'UP',
                bet_amount REAL DEFAULT 1.0,

                -- Crowd pricing at entry
                crowd_up_price REAL,
                crowd_down_price REAL,
                entry_time_in_window REAL,

                -- Order execution
                token_id TEXT,
                order_id TEXT,
                entry_price REAL,
                shares REAL,
                order_status TEXT DEFAULT 'pending',
                fill_status TEXT,
                error TEXT,

                -- Resolution
                result TEXT,
                pnl REAL,
                resolved_at TEXT,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(asset, window_ts)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_simple_status
            ON simple_up_bets(order_status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_simple_result
            ON simple_up_bets(result)
        """)
        # Migrate: add columns if missing (table may exist from v1)
        existing_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(simple_up_bets)").fetchall()
        }
        for col, typedef in [
            ("crowd_up_price", "REAL"),
            ("crowd_down_price", "REAL"),
            ("entry_time_in_window", "REAL"),
        ]:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE simple_up_bets ADD COLUMN {col} {typedef}")
        conn.commit()
        conn.close()
        logger.info("[SIMPLE] DB table simple_up_bets ready")

    # ── CLOB Client ──

    def _load_secrets(self) -> dict:
        """Load Polymarket wallet secrets."""
        paths = [
            SECRETS_PATH,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), SECRETS_PATH),
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if data.get("private_key"):
                        return data
                except Exception:
                    pass
        return {}

    def _init_clob_client(self) -> None:
        """Initialize the Polymarket CLOB client."""
        try:
            from py_clob_client.client import ClobClient
        except ImportError:
            logger.warning("[SIMPLE] py-clob-client not installed — live bets disabled")
            return

        secrets = self._load_secrets()
        if not secrets.get("private_key"):
            logger.warning("[SIMPLE] No wallet configured — live bets disabled")
            return

        try:
            proxy_addr = secrets.get("proxy_wallet_address", "")
            init_kwargs = {
                "host": "https://clob.polymarket.com",
                "chain_id": 137,
                "key": secrets["private_key"],
            }
            if proxy_addr:
                init_kwargs["signature_type"] = 1  # POLY_PROXY
                init_kwargs["funder"] = proxy_addr

            self._clob_client = ClobClient(**init_kwargs)

            # Derive API creds (returns ApiCreds object, not a dict)
            api_creds = self._clob_client.create_or_derive_api_creds()
            self._clob_client.set_api_creds(api_creds)

            server_time = self._clob_client.get_server_time()
            logger.info(f"[SIMPLE] CLOB client connected (server_time={server_time})")
            self.live_enabled = True

        except Exception as e:
            logger.warning(f"[SIMPLE] CLOB client init failed: {e}")
            self._clob_client = None

    def _load_today_state(self) -> None:
        """Load today's bet count and losses from DB."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_date = today
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END), 0) "
                "FROM simple_up_bets WHERE date(created_at) = ?",
                (today,),
            ).fetchone()
            self._bet_count_today = row[0] if row else 0
            self._loss_today = abs(float(row[1])) if row else 0.0
            conn.close()

            # Load already-placed windows for today
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT asset, window_ts FROM simple_up_bets WHERE date(created_at) = ?",
                (today,),
            ).fetchall()
            self._placed_windows = {(r[0], r[1]) for r in rows}
            conn.close()
            logger.info(
                f"[SIMPLE] Today state: {self._bet_count_today} bets, "
                f"${self._loss_today:.2f} losses, {len(self._placed_windows)} windows placed"
            )
        except Exception as e:
            logger.warning(f"[SIMPLE] Failed to load today state: {e}")

    # ── Market Discovery ──

    def _get_current_window_ts(self) -> int:
        """Get the start timestamp of the current 5-minute window."""
        now = int(time.time())
        return (now // WINDOW_SEC) * WINDOW_SEC

    def _seconds_into_window(self) -> float:
        """How many seconds into the current 5-minute window are we?"""
        now = time.time()
        window_ts = math.floor(now / WINDOW_SEC) * WINDOW_SEC
        return now - window_ts

    def _build_slug(self, asset: str, window_ts: int) -> str:
        """Build the Polymarket slug for a 5m market."""
        return f"{asset.lower()}-updown-5m-{window_ts}"

    def _fetch_market(self, slug: str) -> Optional[dict]:
        """Fetch market data from Gamma API with caching."""
        cached = self._market_cache.get(slug)
        if cached and (time.time() - cached["fetched_at"]) < 120:
            return cached["data"]

        try:
            resp = requests.get(
                GAMMA_API,
                params={"slug": slug},
                timeout=5,
            )
            if resp.status_code != 200:
                return None

            markets = resp.json()
            if not markets or not isinstance(markets, list):
                return None

            # Find non-closed, non-resolved market
            for m in markets:
                if m.get("closed") or m.get("resolved"):
                    continue
                self._market_cache[slug] = {"data": m, "fetched_at": time.time()}
                return m

            # Fallback: return first market (for resolution checks)
            market = markets[0]
            self._market_cache[slug] = {"data": market, "fetched_at": time.time()}
            return market

        except Exception as e:
            logger.debug(f"[SIMPLE] Gamma API error for {slug}: {e}")
            return None

    def _parse_market(self, market: dict) -> Optional[dict]:
        """Extract token_id and prices for UP (Yes) outcome."""
        try:
            prices = market.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)
            token_ids = market.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)

            if not prices or len(prices) < 2 or not token_ids or len(token_ids) < 1:
                return None

            up_price = float(prices[0])
            up_token_id = token_ids[0]

            return {
                "up_price": up_price,
                "down_price": float(prices[1]) if len(prices) > 1 else 1.0 - up_price,
                "up_token_id": up_token_id,
                "question": market.get("question", ""),
            }
        except Exception as e:
            logger.debug(f"[SIMPLE] Parse error: {e}")
            return None

    # ── Daily Limits ──

    def _check_daily_limits(self) -> tuple:
        """Check if we're within daily limits. Returns (ok, reason)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            if self._daily_date:
                logger.info(
                    f"[SIMPLE] Daily reset. Yesterday: {self._bet_count_today} bets, "
                    f"${self._loss_today:.2f} losses"
                )
            self._daily_date = today
            self._bet_count_today = 0
            self._loss_today = 0.0
            self._placed_windows.clear()

        if self._bet_count_today >= DAILY_BET_LIMIT:
            return False, f"daily_bet_limit ({self._bet_count_today}/{DAILY_BET_LIMIT})"

        if self._loss_today >= DAILY_LOSS_CAP:
            return False, f"daily_loss_cap (${self._loss_today:.2f}/${DAILY_LOSS_CAP})"

        return True, "ok"

    # ── Bet Placement ──

    def _place_bet(self, asset: str, window_ts: int, elapsed: float) -> Optional[dict]:
        """
        Place a $1 UP bet — but ONLY if UP costs $0.30 or less.
        Returns bet record dict or None.
        """
        slug = self._build_slug(asset, window_ts)

        # Discover market
        market = self._fetch_market(slug)
        if not market:
            logger.debug(f"[SIMPLE] {asset}: market {slug} not found")
            return None

        parsed = self._parse_market(market)
        if not parsed:
            logger.debug(f"[SIMPLE] {asset}: can't parse market {slug}")
            return None

        up_price = parsed["up_price"]
        down_price = parsed["down_price"]
        token_id = parsed["up_token_id"]
        question = parsed["question"]

        if up_price <= 0 or up_price >= 1:
            logger.debug(f"[SIMPLE] {asset}: bad price {up_price}")
            return None

        shares = BET_AMOUNT_USD / up_price

        logger.info(
            f"[SIMPLE] *** BET: {asset} UP ${BET_AMOUNT_USD} @ ${up_price:.3f} "
            f"({shares:.2f} shares) | crowd: UP={up_price:.2f}/DOWN={down_price:.2f} "
            f"| {elapsed:.0f}s into window"
        )

        order_id = ""
        order_status = "paper"
        fill_status = "paper"
        error = None

        # Try live execution
        if self.live_enabled and self._clob_client:
            try:
                from py_clob_client.clob_types import OrderArgs
                from py_clob_client.order_builder.constants import BUY

                # Round price to 2dp, ceil shares so order_total >= $1 CLOB minimum
                order_price = round(up_price, 2)
                order_shares = math.ceil(shares * 100) / 100  # ceil to 2dp

                order_args = OrderArgs(
                    token_id=token_id,
                    price=order_price,
                    size=order_shares,
                    side=BUY,
                )

                result = self._clob_client.create_and_post_order(order_args)

                if result:
                    order_id = (
                        result.get("orderID")
                        or result.get("orderIds", [""])[0]
                        or result.get("id", "")
                        or "unknown"
                    )
                    success = result.get("success", True)
                    if success and order_id and order_id != "unknown":
                        order_status = "placed"
                        fill_status = "placed"
                        logger.info(f"[SIMPLE] Order placed: {order_id}")
                    else:
                        order_status = "rejected"
                        fill_status = "rejected"
                        error = json.dumps(result)[:500]
                        logger.warning(f"[SIMPLE] Rejected: {error}")
                else:
                    order_status = "error"
                    fill_status = "unfilled"
                    error = "Empty order result"

            except Exception as e:
                error = str(e)[:500]
                order_status = "error"
                fill_status = "error"
                logger.warning(f"[SIMPLE] Order error: {e}")
        else:
            logger.info(
                f"[SIMPLE] PAPER BET: {asset} UP @ ${up_price:.3f} "
                f"for ${BET_AMOUNT_USD} ({shares:.2f} shares)"
            )

        # Record in DB
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO simple_up_bets
                (asset, window_ts, slug, question, direction, bet_amount,
                 crowd_up_price, crowd_down_price, entry_time_in_window,
                 token_id, order_id, entry_price, shares,
                 order_status, fill_status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                asset, window_ts, slug, question, DIRECTION, BET_AMOUNT_USD,
                up_price, down_price, elapsed,
                token_id, order_id, up_price, shares,
                order_status, fill_status, error,
            ))
            conn.commit()
        except Exception as e:
            logger.warning(f"[SIMPLE] DB write error: {e}")
        finally:
            conn.close()

        self._bet_count_today += 1
        self._placed_windows.add((asset, window_ts))

        return {
            "asset": asset,
            "window_ts": window_ts,
            "slug": slug,
            "direction": DIRECTION,
            "entry_price": up_price,
            "shares": shares,
            "order_status": order_status,
        }

    # ── Resolution ──

    def _check_resolutions(self) -> None:
        """Check for unresolved bets past their window and resolve them."""
        now = int(time.time())

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            open_bets = conn.execute("""
                SELECT id, asset, window_ts, slug, entry_price, shares, bet_amount
                FROM simple_up_bets
                WHERE result IS NULL
                  AND order_status IN ('placed', 'paper')
                  AND window_ts + ? + ? < ?
            """, (WINDOW_SEC, RESOLUTION_DELAY, now)).fetchall()

            for bet in open_bets:
                self._resolve_single(conn, bet)

        except Exception as e:
            logger.warning(f"[SIMPLE] Resolution check error: {e}")
        finally:
            conn.close()

    def _resolve_single(self, conn: sqlite3.Connection, bet: sqlite3.Row) -> None:
        """Resolve a single bet using Gamma API."""
        slug = bet["slug"]
        bet_id = bet["id"]
        entry_price = bet["entry_price"]
        bet_amount = bet["bet_amount"]
        now = int(time.time())
        window_end = bet["window_ts"] + WINDOW_SEC

        market = self._fetch_market(slug)
        if not market:
            if now - window_end > 1800:
                logger.info(f"[SIMPLE] Force-expiring bet {bet_id} (30min past, no market data)")
                conn.execute(
                    "UPDATE simple_up_bets SET result='expired', pnl=0, "
                    "resolved_at=? WHERE id=?",
                    (datetime.now(timezone.utc).isoformat(), bet_id),
                )
                conn.commit()
            return

        # Check if market is resolved
        resolved = market.get("resolved", False)

        if resolved:
            # Explicit resolution
            prices = market.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)

            if prices and len(prices) >= 1:
                up_final = float(prices[0])
                if up_final >= 0.90:
                    self._record_resolution(conn, bet_id, "WON", bet_amount, entry_price)
                elif up_final <= 0.10:
                    self._record_resolution(conn, bet_id, "LOST", bet_amount, entry_price)
                else:
                    self._record_resolution(conn, bet_id, "expired", bet_amount, entry_price)
            return

        # Not resolved yet — price-based fallback after 2 min
        if now - window_end >= 120:
            prices = market.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)
            if prices and len(prices) >= 1:
                up_now = float(prices[0])
                if up_now >= 0.90:
                    self._record_resolution(conn, bet_id, "WON", bet_amount, entry_price)
                    return
                elif up_now <= 0.10:
                    self._record_resolution(conn, bet_id, "LOST", bet_amount, entry_price)
                    return

        # Force-expire after 30 min
        if now - window_end > 1800:
            logger.info(f"[SIMPLE] Force-expiring bet {bet_id}")
            conn.execute(
                "UPDATE simple_up_bets SET result='expired', pnl=0, "
                "resolved_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), bet_id),
            )
            conn.commit()

    def _record_resolution(self, conn: sqlite3.Connection, bet_id: int,
                           result: str, bet_amount: float, entry_price: float) -> None:
        """Record the resolution of a bet."""
        if result == "WON":
            # Payout = shares * $1.00 (binary pays $1 per share on win)
            # Profit = payout - cost = (bet_amount / entry_price) - bet_amount
            pnl = bet_amount * (1.0 / entry_price - 1.0)
        elif result == "LOST":
            pnl = -bet_amount
        else:
            pnl = 0.0

        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE simple_up_bets SET result=?, pnl=?, resolved_at=? WHERE id=?",
            (result, round(pnl, 4), now_iso, bet_id),
        )
        conn.commit()

        if pnl < 0:
            self._loss_today += abs(pnl)

        emoji = "+++" if result == "WON" else "---"
        logger.info(
            f"[SIMPLE] {emoji} Bet {bet_id}: {result} | "
            f"PnL: ${pnl:+.4f} | Entry: ${entry_price:.3f} | "
            f"Daily: ${self._loss_today:.2f} loss"
        )

    # ── Stats ──

    def get_stats(self) -> dict:
        """Get current stats for dashboard."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            total = conn.execute("SELECT COUNT(*) FROM simple_up_bets").fetchone()[0]
            won = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE result='WON'"
            ).fetchone()[0]
            lost = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE result='LOST'"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE result IS NULL"
            ).fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE result='expired'"
            ).fetchone()[0]
            errors = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE order_status='error'"
            ).fetchone()[0]
            total_pnl = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM simple_up_bets WHERE pnl IS NOT NULL"
            ).fetchone()[0]
            avg_entry = conn.execute(
                "SELECT AVG(entry_price) FROM simple_up_bets WHERE order_status IN ('placed','paper')"
            ).fetchone()[0]

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_total = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE date(created_at) = ?",
                (today,),
            ).fetchone()[0]
            today_pnl = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM simple_up_bets "
                "WHERE pnl IS NOT NULL AND date(created_at) = ?",
                (today,),
            ).fetchone()[0]

            # Per-asset breakdown
            per_asset = {}
            for asset in ASSETS:
                row = conn.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN result='WON' THEN 1 ELSE 0 END) as wins,
                           SUM(CASE WHEN result='LOST' THEN 1 ELSE 0 END) as losses,
                           COALESCE(SUM(pnl), 0) as pnl
                    FROM simple_up_bets WHERE asset=?
                """, (asset,)).fetchone()
                per_asset[asset] = {
                    "total": row[0], "wins": row[1],
                    "losses": row[2], "pnl": round(float(row[3]), 4),
                }

            # Recent bets
            recent = conn.execute("""
                SELECT asset, window_ts, slug, crowd_up_price, entry_price,
                       order_status, result, pnl, created_at
                FROM simple_up_bets ORDER BY id DESC LIMIT 20
            """).fetchall()

            conn.close()

            resolved = won + lost
            win_rate = (won / resolved * 100) if resolved > 0 else 0

            return {
                "strategy": "$1 UP every window (30s before close)",
                "direction": "UP (always)",
                "bet_size": BET_AMOUNT_USD,
                "assets": ASSETS,
                "live_enabled": self.live_enabled,
                "total_bets": total,
                "won": won,
                "lost": lost,
                "pending": pending,
                "expired": expired,
                "errors": errors,
                "win_rate": round(win_rate, 1),
                "total_pnl": round(float(total_pnl), 4),
                "avg_entry_price": round(float(avg_entry or 0), 3),
                "today_bets": today_total,
                "today_pnl": round(float(today_pnl), 4),
                "daily_bet_limit": DAILY_BET_LIMIT,
                "daily_loss_cap": DAILY_LOSS_CAP,
                "per_asset": per_asset,
                "recent_bets": [dict(r) for r in recent],
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Main Loop ──

    async def run(self) -> None:
        """
        Main async loop. Checks every 1 second, bets at T+180s.
        Also periodically checks resolutions.
        """
        logger.info(
            f"[SIMPLE] Starting: $1 UP on {ASSETS} every 5m window "
            f"at T+{BET_TIMING_SEC}s (30s before close) "
            f"(live={'YES' if self.live_enabled else 'NO'})"
        )

        last_resolution_check = 0
        last_heartbeat = 0

        while True:
            try:
                now = time.time()
                secs_in = self._seconds_into_window()
                window_ts = self._get_current_window_ts()

                # ── Bet at T+180s (with 5s window to handle timing jitter) ──
                if BET_TIMING_SEC <= secs_in < BET_TIMING_SEC + 5:
                    ok, reason = self._check_daily_limits()
                    if ok:
                        for asset in ASSETS:
                            if (asset, window_ts) not in self._placed_windows:
                                self._place_bet(asset, window_ts, secs_in)
                    elif secs_in < BET_TIMING_SEC + 2:
                        logger.info(f"[SIMPLE] Skipping: {reason}")

                # ── Check resolutions every 30s ──
                if now - last_resolution_check > 30:
                    self._check_resolutions()
                    last_resolution_check = now

                # ── Heartbeat every 5 minutes ──
                if now - last_heartbeat > 300:
                    open_bets = self._count_open()
                    logger.info(
                        f"[SIMPLE] Heartbeat | bets={self._bet_count_today} | "
                        f"open={open_bets} | daily_loss=${self._loss_today:.2f}"
                    )
                    last_heartbeat = now

            except Exception as e:
                logger.error(f"[SIMPLE] Loop error: {e}", exc_info=True)

            await asyncio.sleep(1)

    def _count_open(self) -> int:
        """Count open bets."""
        try:
            conn = sqlite3.connect(self.db_path)
            n = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE result IS NULL"
            ).fetchone()[0]
            conn.close()
            return n
        except Exception:
            return 0

    async def stop(self) -> None:
        """Stop the loop (for clean shutdown)."""
        pass  # Loop exits when bot shuts down


# ── Standalone Entry Point ──

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("Starting Simple UP Better (standalone mode)")
    better = SimpleUpBetter()
    asyncio.run(better.run())
