"""Spread Capture dashboard endpoints — 0x8dxd strategy monitoring.

All endpoints are READ-ONLY. No trading logic is modified.
Includes on-chain wallet balance tracking as ground truth P&L source.
"""

import logging
import sqlite3
import time
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import requests as http_requests
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/spread-capture", tags=["spread-capture"])

DB_PATH = "data/renaissance_bot.db"

# ─── On-chain wallet tracking ────────────────────────────────
WALLET_ADDRESS = "0x183b2c70dA92Ef34c7C6eE68D515AA6E54e897d1"
USDC_CONTRACT = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDCe on Polygon

# balanceOf(address) function selector
BALANCE_OF_SELECTOR = "0x70a08231"

# Multiple RPCs for reliability (Tenderly works from DO, polygon-rpc.com sometimes fails)
POLYGON_RPCS = [
    "https://polygon.gateway.tenderly.co",
    "https://polygon-rpc.com",
    "https://rpc.ankr.com/polygon",
]

# In-memory cache for wallet balance (refreshed every 30s by background thread)
_wallet_cache = {
    "usdc_balance": 0.0,
    "last_updated": 0.0,
    "rpc_used": "",
    "error": "",
}
_wallet_lock = threading.Lock()
_wallet_thread: Optional[threading.Thread] = None


def _fetch_usdc_balance() -> tuple[float, str]:
    """Fetch USDCe balance from Polygon via JSON-RPC. Returns (balance, rpc_used)."""
    addr_padded = WALLET_ADDRESS[2:].lower().zfill(64)
    call_data = BALANCE_OF_SELECTOR + addr_padded

    for rpc_url in POLYGON_RPCS:
        try:
            resp = http_requests.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "eth_call",
                    "params": [{"to": USDC_CONTRACT, "data": call_data}, "latest"],
                    "id": 1,
                },
                timeout=8,
            )
            result = resp.json().get("result", "0x0")
            if result and result != "0x" and len(result) > 2:
                wei = int(result, 16)
                return wei / 1_000_000, rpc_url.split("/")[2]
        except Exception:
            continue
    return 0.0, "failed"


def _wallet_poll_loop():
    """Background thread: poll wallet balance every 30 seconds."""
    while True:
        try:
            balance, rpc = _fetch_usdc_balance()
            now = time.time()
            with _wallet_lock:
                _wallet_cache["usdc_balance"] = balance
                _wallet_cache["last_updated"] = now
                _wallet_cache["rpc_used"] = rpc
                _wallet_cache["error"] = "" if rpc != "failed" else "all RPCs failed"

            # Log balance to DB for historical tracking
            if balance > 0:
                _log_balance(balance)

        except Exception as e:
            with _wallet_lock:
                _wallet_cache["error"] = str(e)

        time.sleep(30)


def _start_wallet_tracker():
    """Start the background wallet balance poller (once)."""
    global _wallet_thread
    if _wallet_thread is not None and _wallet_thread.is_alive():
        return
    _wallet_thread = threading.Thread(target=_wallet_poll_loop, daemon=True, name="wallet-tracker")
    _wallet_thread.start()
    logger.info("Wallet balance tracker started (30s interval)")


def _log_balance(balance: float):
    """Append balance snapshot to DB table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wallet_balance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                usdc_balance REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # Only insert if balance changed or >30s since last entry
        last = conn.execute(
            "SELECT usdc_balance, timestamp FROM wallet_balance_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        now = time.time()
        if not last or abs(last[0] - balance) > 0.01 or (now - last[1]) > 300:
            conn.execute(
                "INSERT INTO wallet_balance_log (timestamp, usdc_balance) VALUES (?, ?)",
                (now, round(balance, 6)),
            )
            conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f"Wallet log error: {e}")


# Start tracker on module import
_start_wallet_tracker()


# ─── Helpers ──────────────────────────────────────────────────

def _safe_query(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    """Execute read-only query, returning empty list if table doesn't exist."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return []
        raise
    except Exception:
        return []


def _get_db(request: Request) -> str:
    cfg = getattr(request.app.state, "dashboard_config", None)
    return cfg.db_path if cfg else DB_PATH


def _get_engine(bot):
    if bot and hasattr(bot, "spread_capture"):
        return bot.spread_capture
    return None


def _format_v2_summary(rows: list[dict]) -> dict:
    """Format v2 resolved windows into summary dict."""
    r = rows[0] if rows else {}
    total = r.get("total", 0) or 0
    wins = r.get("wins", 0) or 0
    return {
        "strategy": "spread_capture_v2",
        "version": "passive_limit_orders",
        "total_windows": total,
        "wins": wins,
        "losses": r.get("losses", 0) or 0,
        "win_rate": round(wins / max(total, 1) * 100, 1),
        "hedged_windows": r.get("hedged", 0) or 0,
        "hedge_rate": round((r.get("hedged", 0) or 0) / max(total, 1) * 100, 1),
        "total_pnl": round(r.get("total_pnl", 0) or 0, 2),
        "avg_pair_cost": round(r.get("avg_pair_cost", 0) or 0, 3),
        "avg_pnl_per_window": round(r.get("avg_pnl", 0) or 0, 3),
        "total_deployed": round(r.get("total_deployed", 0) or 0, 2),
        "active_positions": 0,
        "daily_pnl": 0,
        "daily_windows": 0,
    }


# ─── Endpoints ────────────────────────────────────────────────

@router.get("/wallet")
async def spread_wallet(request: Request) -> dict[str, Any]:
    """On-chain wallet balance — the GROUND TRUTH for P&L.

    Returns current USDCe balance, starting balance (first snapshot today),
    and actual P&L = current - starting.
    """
    with _wallet_lock:
        balance = _wallet_cache["usdc_balance"]
        last_updated = _wallet_cache["last_updated"]
        rpc = _wallet_cache["rpc_used"]
        error = _wallet_cache["error"]

    # Get starting balance (first snapshot of the day)
    db = _get_db(request)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    starting_rows = _safe_query(db, """
        SELECT usdc_balance FROM wallet_balance_log
        WHERE created_at >= ?
        ORDER BY id ASC LIMIT 1
    """, (today,))
    starting = starting_rows[0]["usdc_balance"] if starting_rows else balance

    # Get all-time first snapshot for total P&L
    first_ever = _safe_query(db, """
        SELECT usdc_balance FROM wallet_balance_log
        ORDER BY id ASC LIMIT 1
    """)
    first_balance = first_ever[0]["usdc_balance"] if first_ever else balance

    # Bot estimated P&L for comparison (v2 table + v1 legacy)
    bot_pnl_rows = _safe_query(db, """
        SELECT COALESCE(SUM(pnl), 0) as total_pnl
        FROM spread_v2_windows WHERE status = 'resolved'
    """)
    bot_pnl = bot_pnl_rows[0]["total_pnl"] if bot_pnl_rows else 0

    actual_daily_pnl = round(balance - starting, 2)
    actual_total_pnl = round(balance - first_balance, 2)
    tracking_error = round(bot_pnl - actual_total_pnl, 2)

    return {
        "usdc_balance": round(balance, 2),
        "last_updated": last_updated,
        "rpc": rpc,
        "error": error,
        "starting_balance_today": round(starting, 2),
        "first_balance_ever": round(first_balance, 2),
        "actual_daily_pnl": actual_daily_pnl,
        "actual_total_pnl": actual_total_pnl,
        "bot_estimated_pnl": round(bot_pnl, 2),
        "tracking_error": tracking_error,
        "wallet_address": WALLET_ADDRESS,
    }


@router.get("/wallet-history")
async def spread_wallet_history(request: Request, hours: int = 48) -> list[dict]:
    """Wallet balance snapshots for charting the REAL P&L curve."""
    db = _get_db(request)
    cutoff = time.time() - hours * 3600
    return _safe_query(db, """
        SELECT timestamp, usdc_balance, created_at
        FROM wallet_balance_log
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
    """, (cutoff,))


@router.get("/summary")
async def spread_summary(request: Request) -> dict[str, Any]:
    """Top-level summary: total P&L, win rate, active windows, pair cost stats."""
    bot = getattr(request.app.state, "bot", None)
    sc = _get_engine(bot)

    if sc:
        try:
            stats = sc.get_stats()
            return stats
        except Exception as e:
            logger.warning(f"Spread summary live error: {e}")

    db = _get_db(request)
    resolved = _safe_query(db, """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN is_hedged = 1 THEN 1 ELSE 0 END) as hedged,
            COALESCE(SUM(pnl), 0) as total_pnl,
            COALESCE(AVG(pair_cost), 0) as avg_pair_cost,
            COALESCE(AVG(pnl), 0) as avg_pnl,
            COALESCE(SUM(total_deployed), 0) as total_deployed
        FROM spread_v2_windows
        WHERE status = 'resolved'
    """)

    result = _format_v2_summary(resolved)

    active_rows = _safe_query(db, """
        SELECT COUNT(*) as cnt FROM spread_v2_windows WHERE status = 'active'
    """)
    result["active_positions"] = active_rows[0]["cnt"] if active_rows else 0

    return result


@router.get("/active")
async def spread_active(request: Request) -> list[dict]:
    """Currently active window positions with live data."""
    db = _get_db(request)
    return _safe_query(db, """
        SELECT asset, timeframe, slug, window_start,
               yes_shares, yes_cost, no_shares, no_cost,
               orders_placed, orders_filled, orders_cancelled,
               pair_cost, hedged_pairs, total_deployed, is_hedged,
               status
        FROM spread_v2_windows
        WHERE status = 'active'
        ORDER BY window_start DESC
    """)


@router.get("/history")
async def spread_history(request: Request, limit: int = 200, asset: str = "") -> list[dict]:
    db = _get_db(request)
    asset_filter = "AND asset = ?" if asset else ""
    params = (asset.upper(), limit) if asset else (limit,)
    return _safe_query(db, f"""
        SELECT id, asset, timeframe, slug, window_start,
               yes_shares, yes_cost, no_shares, no_cost,
               orders_placed, orders_filled, orders_cancelled,
               pair_cost, hedged_pairs, total_deployed, is_hedged,
               resolution, pnl, status,
               resolved_at
        FROM spread_v2_windows
        WHERE status = 'resolved' {asset_filter}
        ORDER BY resolved_at DESC
        LIMIT ?
    """, params)


@router.get("/fills")
async def spread_fills(request: Request, limit: int = 200, slug: str = "") -> list[dict]:
    db = _get_db(request)
    slug_filter = "WHERE window_slug = ?" if slug else ""
    params = (slug, limit) if slug else (limit,)
    return _safe_query(db, f"""
        SELECT id, window_slug, side, price, shares, amount_usd,
               order_id, status, placed_at, filled_at, created_at
        FROM spread_v2_orders
        {slug_filter}
        ORDER BY placed_at DESC
        LIMIT ?
    """, params)


@router.get("/pnl-chart")
async def spread_pnl_chart(request: Request) -> dict[str, Any]:
    """Cumulative P&L chart with BOTH bot estimated and wallet actual lines."""
    db = _get_db(request)

    # Bot estimated cumulative P&L
    rows = _safe_query(db, """
        SELECT resolved_at as ts, pnl, pair_cost, asset, timeframe
        FROM spread_v2_windows
        WHERE status = 'resolved' AND resolved_at IS NOT NULL
        ORDER BY resolved_at ASC
    """)
    cumulative = 0.0
    bot_series = []
    for row in rows:
        cumulative += row.get("pnl", 0) or 0
        bot_series.append({
            "ts": row["ts"],
            "pnl": round(row.get("pnl", 0) or 0, 3),
            "cumulative": round(cumulative, 3),
            "pair_cost": row.get("pair_cost", 0),
            "asset": row.get("asset", ""),
            "timeframe": row.get("timeframe", ""),
        })

    # Wallet balance history for actual P&L line
    wallet_rows = _safe_query(db, """
        SELECT created_at as ts, usdc_balance
        FROM wallet_balance_log
        ORDER BY timestamp ASC
    """)
    first_bal = wallet_rows[0]["usdc_balance"] if wallet_rows else 0
    wallet_series = []
    for wr in wallet_rows:
        wallet_series.append({
            "ts": wr["ts"],
            "wallet_pnl": round(wr["usdc_balance"] - first_bal, 2),
            "wallet_balance": round(wr["usdc_balance"], 2),
        })

    return {
        "bot_series": bot_series,
        "wallet_series": wallet_series,
    }


@router.get("/by-asset")
async def spread_by_asset(request: Request) -> list[dict]:
    db = _get_db(request)
    return _safe_query(db, """
        SELECT asset,
               COUNT(*) as total,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(pnl), 3) as total_pnl,
               ROUND(AVG(pnl), 4) as avg_pnl,
               ROUND(AVG(pair_cost), 3) as avg_pair_cost,
               SUM(CASE WHEN is_hedged = 1 THEN 1 ELSE 0 END) as hedged_count,
               SUM(orders_filled) as total_fills
        FROM spread_v2_windows
        WHERE status = 'resolved'
        GROUP BY asset
        ORDER BY total_pnl DESC
    """)


@router.get("/by-timeframe")
async def spread_by_timeframe(request: Request) -> list[dict]:
    db = _get_db(request)
    return _safe_query(db, """
        SELECT timeframe,
               COUNT(*) as total,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(pnl), 3) as total_pnl,
               ROUND(AVG(pnl), 4) as avg_pnl,
               ROUND(AVG(pair_cost), 3) as avg_pair_cost,
               SUM(CASE WHEN is_hedged = 1 THEN 1 ELSE 0 END) as hedged_count
        FROM spread_v2_windows
        WHERE status = 'resolved'
        GROUP BY timeframe
    """)


@router.get("/orders")
async def spread_orders(request: Request, limit: int = 200, status: str = "") -> list[dict]:
    """All orders with optional status filter (pending/filled/cancelled)."""
    db = _get_db(request)
    status_filter = "WHERE status = ?" if status else ""
    params = (status, limit) if status else (limit,)
    return _safe_query(db, f"""
        SELECT id, window_slug, order_id, side, price, shares, amount_usd,
               status, placed_at, filled_at, created_at
        FROM spread_v2_orders
        {status_filter}
        ORDER BY placed_at DESC
        LIMIT ?
    """, params)


@router.get("/hourly")
async def spread_hourly(request: Request, hours: int = 48) -> list[dict]:
    db = _get_db(request)
    return _safe_query(db, """
        SELECT strftime('%Y-%m-%d %H:00', resolved_at) as hour,
               COUNT(*) as windows,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(pnl), 3) as pnl,
               ROUND(AVG(pair_cost), 3) as avg_pair_cost
        FROM spread_v2_windows
        WHERE status = 'resolved' AND resolved_at IS NOT NULL
        GROUP BY hour
        ORDER BY hour DESC
        LIMIT ?
    """, (hours,))


@router.get("/pair-cost-distribution")
async def spread_pair_cost_dist(request: Request) -> list[dict]:
    db = _get_db(request)
    return _safe_query(db, """
        SELECT
            CASE
                WHEN pair_cost < 0.70 THEN '<0.70'
                WHEN pair_cost < 0.80 THEN '0.70-0.80'
                WHEN pair_cost < 0.90 THEN '0.80-0.90'
                WHEN pair_cost < 0.95 THEN '0.90-0.95'
                WHEN pair_cost < 1.00 THEN '0.95-1.00'
                WHEN pair_cost < 1.05 THEN '1.00-1.05'
                WHEN pair_cost < 1.10 THEN '1.05-1.10'
                ELSE '>1.10'
            END as bucket,
            COUNT(*) as count,
            ROUND(AVG(pnl), 3) as avg_pnl
        FROM spread_v2_windows
        WHERE status = 'resolved' AND pair_cost > 0
        GROUP BY bucket
        ORDER BY MIN(pair_cost) ASC
    """)


@router.get("/fill-rate")
async def spread_fill_rate(request: Request) -> dict[str, Any]:
    db = _get_db(request)

    status_dist = _safe_query(db, """
        SELECT status, COUNT(*) as count,
               ROUND(SUM(amount_usd), 2) as total_usd
        FROM spread_v2_orders GROUP BY status ORDER BY status
    """)
    side_dist = _safe_query(db, """
        SELECT side, COUNT(*) as count,
               ROUND(AVG(price), 3) as avg_price,
               ROUND(SUM(amount_usd), 2) as total_usd
        FROM spread_v2_orders WHERE status = 'filled' GROUP BY side
    """)
    total_placed = _safe_query(db, "SELECT COUNT(*) as cnt FROM spread_v2_orders")
    total_filled = _safe_query(db, "SELECT COUNT(*) as cnt FROM spread_v2_orders WHERE status='filled'")
    return {
        "total_placed": total_placed[0]["cnt"] if total_placed else 0,
        "total_filled": total_filled[0]["cnt"] if total_filled else 0,
        "fill_rate": round(
            (total_filled[0]["cnt"] if total_filled else 0) /
            max(total_placed[0]["cnt"] if total_placed else 1, 1) * 100, 1
        ),
        "by_status": status_dist,
        "by_side": side_dist,
    }


@router.get("/errors")
async def spread_errors(request: Request, limit: int = 50) -> list[dict]:
    db = _get_db(request)
    return _safe_query(db, """
        SELECT id, asset, timeframe, slug, window_start,
               pair_cost, total_deployed, status
        FROM spread_v2_windows
        WHERE status = 'error'
        ORDER BY id DESC LIMIT ?
    """, (limit,))
