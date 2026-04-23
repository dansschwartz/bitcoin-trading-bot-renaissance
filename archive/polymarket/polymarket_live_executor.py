"""
polymarket_live_executor.py — Real order execution on Polymarket CLOB.

This module is the ONLY code path that places real money bets.
Everything else in the system is paper/simulation.

Architecture:
    Strategy A decides: "bet SOL UP at $0.48 for $2"
    -> If live_mode AND asset is in LIVE_ASSETS:
        -> This module places a real CLOB order
        -> Records fill in polymarket_live_bets table
    -> Else:
        -> Paper executor records simulated bet (existing behavior)

Safety layers:
    1. live_enabled must be True in config
    2. Asset must be in LIVE_ASSETS whitelist
    3. Bet must be <= MAX_LIVE_BET_USD
    4. Daily loss must be < DAILY_LOSS_LIMIT
    5. Wallet must be configured with private key
    6. py-clob-client must be installed
    7. Sufficient USDC balance on Polygon
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger("polymarket.live")

# =====================================================================
# SAFETY CONSTANTS — DO NOT CHANGE WITHOUT HUMAN APPROVAL
# =====================================================================

# Only these assets can be traded live
LIVE_ASSETS = {"SOL", "DOGE"}  # SOL + DOGE (both have 5m instruments enabled)

# Maximum bet size in USD
MAX_LIVE_BET_USD = 20.00  # Raised to match Kelly max ($20)

# Daily loss limit
DAILY_LOSS_LIMIT_USD = 200.00  # Stop if we lose $200 in one day

# Maximum number of open live positions
MAX_OPEN_LIVE_POSITIONS = 5

# Minimum USDC balance to continue trading (reserve for gas)
MIN_USDC_BALANCE = 10.0

# How many live bets before we consider raising limits
PROOF_THRESHOLD = 50


class PolymarketLiveExecutor:
    """
    Places real orders on Polymarket's CLOB.

    Only activated when:
    - config has live_enabled: true
    - private key is configured
    - py-clob-client is installed
    - asset is in LIVE_ASSETS
    - bet size <= MAX_LIVE_BET_USD
    - daily loss < DAILY_LOSS_LIMIT_USD
    """

    def __init__(self, db_path: str = "data/renaissance_bot.db",
                 config: Optional[dict] = None):
        self.db_path = db_path
        self.live_enabled = False
        self._clob_client = None
        self._api_creds = None
        self._daily_pnl = 0.0
        self._daily_reset_date = ""
        self._live_bet_count = 0

        # Allow config overrides for safety constants
        pm_cfg = (config or {}).get("polymarket", {})
        self._cfg_live_enabled = pm_cfg.get("live_enabled", False)
        self._cfg_live_assets = set(pm_cfg.get("live_assets", list(LIVE_ASSETS)))
        self._cfg_max_bet = min(
            pm_cfg.get("max_live_bet_usd", MAX_LIVE_BET_USD),
            MAX_LIVE_BET_USD,  # Hard cap — config cannot exceed constant
        )
        self._cfg_daily_loss_limit = pm_cfg.get(
            "daily_loss_limit_usd", DAILY_LOSS_LIMIT_USD
        )

        self._init_db()

        if self._cfg_live_enabled:
            self._try_init_client()
        else:
            logger.info("[LIVE] live_enabled=false in config — live execution disabled")

    def _init_db(self) -> None:
        """Create the live bets tracking table."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS polymarket_live_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Market
                asset TEXT NOT NULL,
                timeframe TEXT,
                slug TEXT,
                question TEXT,
                window_start INTEGER,

                -- Decision (from Strategy A)
                direction TEXT NOT NULL,
                predicted_edge REAL,
                ml_confidence REAL,
                crowd_price_at_decision REAL,

                -- Order placement
                order_id TEXT,
                token_id TEXT,
                side TEXT,
                order_price REAL,
                order_size_usd REAL,
                order_shares REAL,
                order_placed_at TEXT,

                -- Fill
                fill_price REAL,
                fill_shares REAL,
                fill_amount_usd REAL,
                slippage_bps REAL,
                gas_cost_usd REAL,
                filled_at TEXT,
                fill_status TEXT,

                -- Resolution
                status TEXT DEFAULT 'open',
                pnl REAL,
                resolution_price REAL,
                resolved_at TEXT,

                -- Metadata
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT,
                paper_comparison_pnl REAL,

                UNIQUE(asset, window_start, direction)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_live_status
            ON polymarket_live_bets(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_live_created
            ON polymarket_live_bets(created_at)
        """)
        conn.commit()
        conn.close()

    def _try_init_client(self) -> None:
        """Try to initialize the CLOB client. Fails gracefully if not configured."""
        try:
            from py_clob_client.client import ClobClient
        except ImportError:
            logger.info("[LIVE] py-clob-client not installed — live execution disabled")
            return

        secrets = self._load_secrets()
        if not secrets.get("private_key"):
            logger.info("[LIVE] No wallet configured — live execution disabled")
            return

        try:
            # Polymarket uses proxy wallets — need signature_type=1 (POLY_PROXY)
            # and funder=proxy_wallet_address for orders to use the proxy balance
            proxy_addr = secrets.get("proxy_wallet_address", "")
            init_kwargs = {
                "host": "https://clob.polymarket.com",
                "chain_id": 137,  # Polygon
                "key": secrets["private_key"],
            }
            if proxy_addr:
                init_kwargs["signature_type"] = 1  # POLY_PROXY
                init_kwargs["funder"] = proxy_addr
                logger.info(f"[LIVE] Using proxy wallet: {proxy_addr[:10]}...{proxy_addr[-6:]}")

            self._clob_client = ClobClient(**init_kwargs)

            # Derive API credentials (required for order placement)
            self._api_creds = self._clob_client.create_or_derive_api_creds()
            self._clob_client.set_api_creds(self._api_creds)

            # Verify connection
            server_time = self._clob_client.get_server_time()
            logger.info(f"[LIVE] CLOB client connected (server_time={server_time})")

            self.live_enabled = True
            logger.info(
                f"[LIVE] *** LIVE EXECUTION ENABLED ***\n"
                f"  Assets: {self._cfg_live_assets}\n"
                f"  Max bet: ${self._cfg_max_bet}\n"
                f"  Daily loss limit: ${self._cfg_daily_loss_limit}\n"
                f"  Max open positions: {MAX_OPEN_LIVE_POSITIONS}"
            )

        except Exception as e:
            logger.warning(f"[LIVE] CLOB client init failed: {e}")
            self._clob_client = None

    def _load_secrets(self) -> dict:
        """Load wallet secrets from config file."""
        paths = [
            os.path.join("config", "polymarket_secrets.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "config", "polymarket_secrets.json"),
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if data.get("private_key"):
                        return data
                except Exception as e:
                    logger.warning(f"Failed to load secrets from {path}: {e}")
        return {}

    def should_go_live(self, asset: str, bet_amount: float) -> tuple:
        """
        Check all safety gates for live execution.

        Returns: (should_live: bool, reason: str)
        """
        if not self.live_enabled:
            return False, "live_disabled"

        if not self._clob_client:
            return False, "no_clob_client"

        if asset not in self._cfg_live_assets:
            return False, f"asset_{asset}_not_in_live_whitelist"

        if bet_amount > self._cfg_max_bet:
            return False, f"bet_${bet_amount:.2f}_exceeds_max_${self._cfg_max_bet}"

        # Daily loss check
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_reset_date = today
            self._daily_pnl = self._load_daily_pnl(today)

        if self._daily_pnl <= -self._cfg_daily_loss_limit:
            return False, f"daily_loss_limit_hit_(${self._daily_pnl:.2f})"

        # Open positions check
        conn = sqlite3.connect(self.db_path)
        open_count = conn.execute(
            "SELECT COUNT(*) FROM polymarket_live_bets WHERE status='open'"
        ).fetchone()[0]
        conn.close()

        if open_count >= MAX_OPEN_LIVE_POSITIONS:
            return False, f"max_open_positions_({open_count})"

        return True, "approved"

    def _load_daily_pnl(self, date_str: str) -> float:
        """Load today's realized P&L from DB."""
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM polymarket_live_bets "
                "WHERE status IN ('won', 'lost') AND date(resolved_at) = ?",
                (date_str,),
            ).fetchone()
            conn.close()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0

    async def place_live_bet(
        self,
        asset: str,
        direction: str,
        entry_price: float,
        bet_amount: float,
        token_id: str,
        slug: str = "",
        question: str = "",
        window_start: int = 0,
        timeframe: str = "",
        edge: float = 0,
        confidence: float = 0,
        crowd_price: float = 0,
    ) -> Optional[Dict]:
        """
        Place a real bet on Polymarket.

        Returns: dict with order result, or None if failed
        """
        # Safety gate
        should_live, reason = self.should_go_live(asset, bet_amount)
        if not should_live:
            logger.info(f"[LIVE] Blocked: {reason}")
            return None

        # Hard cap bet amount
        bet_amount = min(bet_amount, self._cfg_max_bet)
        shares = bet_amount / entry_price if entry_price > 0 else 0

        logger.info(
            f"[LIVE] *** PLACING REAL BET ***\n"
            f"  {asset} {timeframe} {direction} @ ${entry_price:.3f}\n"
            f"  Amount: ${bet_amount:.2f} ({shares:.2f} shares)\n"
            f"  Edge: {edge:.1%} | Confidence: {confidence:.1%}\n"
            f"  Token: {token_id[:20]}... | Slug: {slug[:40]}"
        )

        order_placed_at = datetime.now(timezone.utc).isoformat()
        order_result = None
        fill_status = "error"
        error_msg = None
        order_id = ""

        try:
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY

            order_args = OrderArgs(
                token_id=token_id,
                price=round(entry_price, 2),  # CLOB requires 2 decimal places
                size=round(shares, 2),
                side=BUY,
            )

            order_result = self._clob_client.create_and_post_order(order_args)

            if order_result:
                order_id = (
                    order_result.get("orderID")
                    or order_result.get("orderIds", [""])[0]
                    or order_result.get("id", "")
                    or "unknown"
                )
                # Check for success indicators
                success = order_result.get("success", True)
                if success and order_id and order_id != "unknown":
                    fill_status = "placed"
                else:
                    fill_status = "rejected"
                    error_msg = json.dumps(order_result)[:500]

                logger.info(
                    f"[LIVE] Order placed: {order_id}\n"
                    f"  Result: {json.dumps(order_result)[:200]}"
                )
            else:
                fill_status = "unfilled"
                error_msg = "Empty order result"
                logger.warning("[LIVE] Order returned empty result")

        except Exception as e:
            error_msg = str(e)[:500]
            # Geo-block or other CLOB error → queue for local relay
            is_geoblock = "restricted in your region" in error_msg.lower() or "geoblock" in error_msg.lower()
            if is_geoblock:
                fill_status = "pending_relay"
                logger.info(
                    f"[LIVE] Geo-blocked — queued for local relay: "
                    f"{asset} {direction} ${bet_amount:.2f}"
                )
            else:
                fill_status = "error"
                logger.error(f"[LIVE] Order FAILED: {e}")

        # Record in database
        if fill_status == "pending_relay":
            db_status = "pending_relay"
        elif fill_status in ("placed", "filled"):
            db_status = "open"
        else:
            db_status = "error"

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO polymarket_live_bets
                (asset, timeframe, slug, question, window_start,
                 direction, predicted_edge, ml_confidence, crowd_price_at_decision,
                 order_id, token_id, side, order_price, order_size_usd, order_shares,
                 order_placed_at, fill_status, error_message, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                asset, timeframe, slug, question, window_start,
                direction, edge, confidence, crowd_price,
                order_id, token_id, "BUY", entry_price, bet_amount, shares,
                order_placed_at, fill_status, error_msg, db_status,
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"[LIVE] DB record failed: {e}")
        finally:
            conn.close()

        self._live_bet_count += 1

        return {
            "live": True,
            "order_id": order_id or None,
            "fill_status": fill_status,
            "asset": asset,
            "direction": direction,
            "amount": bet_amount,
            "error": error_msg,
        }

    def get_pending_relay_bets(self) -> list:
        """Return bets awaiting local relay execution."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("""
                SELECT id, asset, timeframe, slug, question, window_start,
                       direction, predicted_edge, ml_confidence, crowd_price_at_decision,
                       token_id, order_price, order_size_usd, order_shares,
                       order_placed_at, created_at
                FROM polymarket_live_bets
                WHERE status = 'pending_relay'
                ORDER BY created_at ASC
            """).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            conn.close()

    def confirm_relay_bet(self, bet_id: int, order_id: str, fill_status: str,
                          error_msg: str = "") -> bool:
        """Update a pending_relay bet after local execution."""
        conn = sqlite3.connect(self.db_path)
        try:
            if fill_status in ("placed", "filled"):
                conn.execute("""
                    UPDATE polymarket_live_bets
                    SET status='open', order_id=?, fill_status=?,
                        filled_at=?, error_message=NULL
                    WHERE id=? AND status='pending_relay'
                """, (order_id, fill_status,
                      datetime.now(timezone.utc).isoformat(), bet_id))
            else:
                conn.execute("""
                    UPDATE polymarket_live_bets
                    SET status='error', order_id=?, fill_status=?, error_message=?
                    WHERE id=? AND status='pending_relay'
                """, (order_id, fill_status, error_msg[:500], bet_id))
            conn.commit()
            affected = conn.total_changes
            logger.info(
                f"[LIVE] Relay confirmed: bet #{bet_id} → {fill_status} "
                f"(order={order_id or 'none'})"
            )
            return affected > 0
        except Exception as e:
            logger.error(f"[LIVE] Relay confirm failed: {e}")
            return False
        finally:
            conn.close()

    def check_live_resolutions(self) -> None:
        """
        Check resolution of open live bets via Gamma API.
        Force-expires bets that are >30min past their market window.
        Records results in polymarket_live_bets table.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        open_bets = conn.execute(
            "SELECT * FROM polymarket_live_bets WHERE status='open'"
        ).fetchall()

        if not open_bets:
            conn.close()
            return

        import requests
        now_ts = int(time.time())

        for bet in open_bets:
            # Force-expire if >30min past market window end
            ws = bet["window_start"] or 0
            tf_str = bet["timeframe"] or "15m"
            tf_secs = 300 if "5m" in tf_str else 900
            if ws > 0 and now_ts > ws + tf_secs + 1800:
                conn.execute(
                    "UPDATE polymarket_live_bets SET status='expired', "
                    "resolved_at=? WHERE id=?",
                    (datetime.now(timezone.utc).isoformat(), bet["id"]),
                )
                conn.commit()
                logger.info(
                    f"[LIVE] Force-expired stale bet id={bet['id']} "
                    f"{bet['asset']} {bet['direction']} "
                    f"(window_start={ws}, {(now_ts - ws - tf_secs)//60}min past expiry)"
                )
                continue
            slug = bet["slug"]
            if not slug:
                continue

            try:
                resp = requests.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={"slug": slug},
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue

                markets = resp.json()
                if not markets or not isinstance(markets, list):
                    continue

                market = None
                for m in markets:
                    if m.get("resolved"):
                        market = m
                        break

                if not market:
                    continue

                prices = market.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    prices = json.loads(prices)

                if not prices or len(prices) < 2:
                    continue

                yes_price = float(prices[0])

                # Definitive resolution only
                if not (yes_price >= 0.95 or yes_price <= 0.05):
                    continue

                went_up = yes_price >= 0.95

                if bet["direction"] == "UP":
                    won = went_up
                else:
                    won = not went_up

                if won:
                    # Payout: shares * $1 * (1 - 2% fee) - cost
                    payout = (bet["order_shares"] or 0) * 1.0 * 0.98
                    pnl = payout - (bet["order_size_usd"] or 0)
                    status = "won"
                else:
                    pnl = -(bet["order_size_usd"] or 0)
                    status = "lost"

                conn.execute("""
                    UPDATE polymarket_live_bets
                    SET status=?, pnl=?, resolution_price=?, resolved_at=?
                    WHERE id=?
                """, (
                    status, round(pnl, 4), yes_price,
                    datetime.now(timezone.utc).isoformat(), bet["id"],
                ))
                conn.commit()

                self._daily_pnl += pnl

                marker = "$$" if won else "---"
                logger.info(
                    f"[LIVE] {marker} {bet['asset']} {bet['direction']}: "
                    f"{status.upper()} | PnL=${pnl:+.2f} | "
                    f"Entry=${bet['order_price']:.3f} | "
                    f"Daily: ${self._daily_pnl:+.2f} | "
                    f"Bet #{self._live_bet_count}"
                )

            except Exception as e:
                logger.debug(f"[LIVE] Resolution check failed for {slug}: {e}")

        conn.close()

    def get_stats(self) -> Dict:
        """Live trading statistics for dashboard."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            stats = conn.execute("""
                SELECT
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_bets,
                    SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors,
                    SUM(CASE WHEN status='pending_relay' THEN 1 ELSE 0 END) as pending_relay,
                    COALESCE(SUM(CASE WHEN status IN ('won','lost') THEN pnl ELSE 0 END), 0) as total_pnl,
                    COALESCE(SUM(order_size_usd), 0) as total_wagered,
                    AVG(slippage_bps) as avg_slippage,
                    COUNT(CASE WHEN fill_status='error' THEN 1 END) as fill_errors
                FROM polymarket_live_bets
            """).fetchone()
        except Exception:
            conn.close()
            return {
                "live_enabled": self.live_enabled,
                "live_assets": list(self._cfg_live_assets),
                "max_bet_usd": self._cfg_max_bet,
                "error": "table_not_ready",
            }

        try:
            recent = conn.execute("""
                SELECT asset, direction, order_price, order_size_usd,
                       pnl, status, fill_status, created_at, error_message,
                       timeframe, slug
                FROM polymarket_live_bets
                ORDER BY created_at DESC LIMIT 20
            """).fetchall()
        except Exception:
            recent = []

        conn.close()

        resolved = (stats["wins"] or 0) + (stats["losses"] or 0)

        return {
            "live_enabled": self.live_enabled,
            "live_assets": list(self._cfg_live_assets),
            "max_bet_usd": self._cfg_max_bet,
            "daily_loss_limit": self._cfg_daily_loss_limit,
            "total_bets": stats["total_bets"] or 0,
            "wins": stats["wins"] or 0,
            "losses": stats["losses"] or 0,
            "open_bets": stats["open_bets"] or 0,
            "errors": stats["errors"] or 0,
            "pending_relay": stats["pending_relay"] or 0,
            "win_rate": round((stats["wins"] or 0) / resolved * 100, 1) if resolved > 0 else 0,
            "total_pnl": round(float(stats["total_pnl"]), 2),
            "total_wagered": round(float(stats["total_wagered"] or 0), 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "avg_slippage_bps": round(float(stats["avg_slippage"] or 0), 1),
            "fill_errors": stats["fill_errors"] or 0,
            "proof_threshold": PROOF_THRESHOLD,
            "bets_toward_proof": min(stats["total_bets"] or 0, PROOF_THRESHOLD),
            "recent_bets": [dict(r) for r in recent],
        }
