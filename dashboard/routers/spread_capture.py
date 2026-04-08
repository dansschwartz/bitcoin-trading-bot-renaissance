"""Spread Capture dashboard endpoints — 0x8dxd strategy monitoring.

All endpoints are READ-ONLY. No trading logic is modified.
"""

import logging
import sqlite3
import time
from typing import Any

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/spread-capture", tags=["spread-capture"])

DB_PATH = "data/renaissance_bot.db"


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


@router.get("/summary")
async def spread_summary(request: Request) -> dict[str, Any]:
    """Top-level summary: total P&L, win rate, active windows, pair cost stats."""
    bot = getattr(request.app.state, "bot", None)
    sc = _get_engine(bot)

    # Live stats from engine if available
    if sc:
        try:
            stats = sc.get_stats()
            active = sc.get_active_positions()
            stats["active_positions_detail"] = active
            return stats
        except Exception as e:
            logger.warning(f"Spread summary live error: {e}")

    # Fallback to DB
    db = _get_db(request)
    resolved = _safe_query(db, """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN is_arb = 1 THEN 1 ELSE 0 END) as arb_windows,
            COALESCE(SUM(pnl), 0) as total_pnl,
            COALESCE(AVG(pair_cost), 0) as avg_pair_cost,
            COALESCE(AVG(pnl), 0) as avg_pnl,
            COALESCE(SUM(total_deployed), 0) as total_deployed
        FROM spread_capture_windows
        WHERE status = 'resolved'
    """)
    r = resolved[0] if resolved else {}
    total = r.get("total", 0) or 0
    wins = r.get("wins", 0) or 0

    active_rows = _safe_query(db, """
        SELECT COUNT(*) as cnt FROM spread_capture_windows WHERE status = 'active'
    """)
    active_cnt = active_rows[0]["cnt"] if active_rows else 0

    return {
        "strategy": "0x8dxd_spread_capture",
        "total_windows": total,
        "wins": wins,
        "losses": r.get("losses", 0) or 0,
        "win_rate": round(wins / max(total, 1) * 100, 1),
        "arb_windows": r.get("arb_windows", 0) or 0,
        "arb_rate": round((r.get("arb_windows", 0) or 0) / max(total, 1) * 100, 1),
        "total_pnl": round(r.get("total_pnl", 0) or 0, 2),
        "avg_pair_cost": round(r.get("avg_pair_cost", 0) or 0, 3),
        "avg_pnl_per_window": round(r.get("avg_pnl", 0) or 0, 3),
        "total_deployed": round(r.get("total_deployed", 0) or 0, 2),
        "active_positions": active_cnt,
        "daily_pnl": 0,
        "daily_windows": 0,
    }


@router.get("/active")
async def spread_active(request: Request) -> list[dict]:
    """Currently active window positions with live data."""
    bot = getattr(request.app.state, "bot", None)
    sc = _get_engine(bot)

    if sc:
        try:
            return sc.get_active_positions()
        except Exception:
            pass

    # Fallback: show active windows from DB
    db = _get_db(request)
    return _safe_query(db, """
        SELECT asset, timeframe, slug, window_start,
               yes_shares, yes_cost, yes_avg_price, yes_orders,
               no_shares, no_cost, no_avg_price, no_orders,
               pair_cost, total_deployed, guaranteed_profit, is_arb,
               max_phase as phase, status
        FROM spread_capture_windows
        WHERE status = 'active'
        ORDER BY window_start DESC
    """)


@router.get("/history")
async def spread_history(request: Request, limit: int = 200, asset: str = "") -> list[dict]:
    """Resolved windows with P&L."""
    db = _get_db(request)
    asset_filter = "AND asset = ?" if asset else ""
    params = (asset.upper(), limit) if asset else (limit,)
    return _safe_query(db, f"""
        SELECT id, asset, timeframe, slug, window_start,
               yes_shares, yes_cost, yes_avg_price, yes_orders,
               no_shares, no_cost, no_avg_price, no_orders,
               pair_cost, total_deployed, guaranteed_profit, is_arb,
               resolution, pnl, pnl_pct,
               max_phase, phase2_triggered, status,
               created_at, resolved_at
        FROM spread_capture_windows
        WHERE status = 'resolved' {asset_filter}
        ORDER BY resolved_at DESC
        LIMIT ?
    """, params)


@router.get("/fills")
async def spread_fills(request: Request, limit: int = 200, slug: str = "") -> list[dict]:
    """Recent order fills across all windows."""
    db = _get_db(request)
    slug_filter = "WHERE window_slug = ?" if slug else ""
    params = (slug, limit) if slug else (limit,)
    return _safe_query(db, f"""
        SELECT id, window_slug, side, price, shares, amount_usd,
               order_id, phase, timestamp, created_at
        FROM spread_capture_fills
        {slug_filter}
        ORDER BY timestamp DESC
        LIMIT ?
    """, params)


@router.get("/pnl-chart")
async def spread_pnl_chart(request: Request) -> list[dict]:
    """Cumulative P&L over time for chart rendering."""
    db = _get_db(request)
    rows = _safe_query(db, """
        SELECT resolved_at as ts, pnl, pair_cost, asset, timeframe
        FROM spread_capture_windows
        WHERE status = 'resolved' AND resolved_at IS NOT NULL
        ORDER BY resolved_at ASC
    """)
    cumulative = 0.0
    result = []
    for row in rows:
        cumulative += row.get("pnl", 0) or 0
        result.append({
            "ts": row["ts"],
            "pnl": round(row.get("pnl", 0) or 0, 3),
            "cumulative": round(cumulative, 3),
            "pair_cost": row.get("pair_cost", 0),
            "asset": row.get("asset", ""),
            "timeframe": row.get("timeframe", ""),
        })
    return result


@router.get("/by-asset")
async def spread_by_asset(request: Request) -> list[dict]:
    """Performance breakdown by asset."""
    db = _get_db(request)
    return _safe_query(db, """
        SELECT asset,
               COUNT(*) as total,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(pnl), 3) as total_pnl,
               ROUND(AVG(pnl), 4) as avg_pnl,
               ROUND(AVG(pair_cost), 3) as avg_pair_cost,
               SUM(CASE WHEN is_arb = 1 THEN 1 ELSE 0 END) as arb_count,
               SUM(CASE WHEN phase2_triggered = 1 THEN 1 ELSE 0 END) as phase2_count
        FROM spread_capture_windows
        WHERE status = 'resolved'
        GROUP BY asset
        ORDER BY total_pnl DESC
    """)


@router.get("/by-timeframe")
async def spread_by_timeframe(request: Request) -> list[dict]:
    """Performance breakdown by timeframe (5m vs 15m)."""
    db = _get_db(request)
    return _safe_query(db, """
        SELECT timeframe,
               COUNT(*) as total,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(pnl), 3) as total_pnl,
               ROUND(AVG(pnl), 4) as avg_pnl,
               ROUND(AVG(pair_cost), 3) as avg_pair_cost,
               SUM(CASE WHEN is_arb = 1 THEN 1 ELSE 0 END) as arb_count
        FROM spread_capture_windows
        WHERE status = 'resolved'
        GROUP BY timeframe
    """)


@router.get("/rtds")
async def spread_rtds(request: Request) -> dict[str, Any]:
    """Live RTDS oracle feed status."""
    bot = getattr(request.app.state, "bot", None)
    rtds = _get_rtds(bot)
    if rtds:
        try:
            return rtds.get_status()
        except Exception:
            pass
    return {"connected": False, "assets": {}}


@router.get("/hourly")
async def spread_hourly(request: Request, hours: int = 48) -> list[dict]:
    """Hourly P&L aggregation."""
    db = _get_db(request)
    return _safe_query(db, """
        SELECT strftime('%Y-%m-%d %H:00', resolved_at) as hour,
               COUNT(*) as windows,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(pnl), 3) as pnl,
               ROUND(AVG(pair_cost), 3) as avg_pair_cost
        FROM spread_capture_windows
        WHERE status = 'resolved' AND resolved_at IS NOT NULL
        GROUP BY hour
        ORDER BY hour DESC
        LIMIT ?
    """, (hours,))


@router.get("/pair-cost-distribution")
async def spread_pair_cost_dist(request: Request) -> list[dict]:
    """Distribution of pair costs for histogram."""
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
        FROM spread_capture_windows
        WHERE status = 'resolved' AND pair_cost > 0
        GROUP BY bucket
        ORDER BY MIN(pair_cost) ASC
    """)


@router.get("/fill-rate")
async def spread_fill_rate(request: Request) -> dict[str, Any]:
    """Fill rate and phase distribution stats."""
    db = _get_db(request)

    phase_dist = _safe_query(db, """
        SELECT phase,
               COUNT(*) as count,
               ROUND(SUM(amount_usd), 2) as total_usd
        FROM spread_capture_fills
        GROUP BY phase
        ORDER BY phase
    """)

    side_dist = _safe_query(db, """
        SELECT side,
               COUNT(*) as count,
               ROUND(AVG(price), 3) as avg_price,
               ROUND(SUM(amount_usd), 2) as total_usd
        FROM spread_capture_fills
        GROUP BY side
    """)

    total_fills = _safe_query(db, "SELECT COUNT(*) as cnt FROM spread_capture_fills")
    total = total_fills[0]["cnt"] if total_fills else 0

    return {
        "total_fills": total,
        "by_phase": phase_dist,
        "by_side": side_dist,
    }


@router.get("/errors")
async def spread_errors(request: Request, limit: int = 50) -> list[dict]:
    """Windows that ended in error status."""
    db = _get_db(request)
    return _safe_query(db, """
        SELECT id, asset, timeframe, slug, window_start,
               pair_cost, total_deployed, status, created_at
        FROM spread_capture_windows
        WHERE status = 'error'
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))


# ─── Helpers ──────────────────────────────────────────────────

def _get_db(request: Request) -> str:
    return getattr(request.app.state, "dashboard_config", None) and request.app.state.dashboard_config.db_path or DB_PATH


def _get_engine(bot):
    """Get SpreadCaptureEngine from bot if available."""
    if bot and hasattr(bot, "spread_capture"):
        return bot.spread_capture
    return None


def _get_rtds(bot):
    """Get RTDS feed from bot if available."""
    if bot and hasattr(bot, "rtds"):
        return bot.rtds
    # Also try via spread capture engine
    sc = _get_engine(bot)
    if sc and hasattr(sc, "_rtds"):
        return sc._rtds
    return None
