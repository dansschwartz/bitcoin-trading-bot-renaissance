"""Devil Tracker dashboard endpoints."""

import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/devil", tags=["devil"])


def _conn(db_path: str):
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/summary")
async def devil_summary(request: Request, hours: int = 24):
    """Devil Tracker cost summary over the last N hours."""
    db = request.app.state.dashboard_config.db_path
    try:
        conn = _conn(db)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        # Total entries
        total = conn.execute(
            "SELECT COUNT(*) FROM devil_tracker WHERE signal_timestamp >= ?",
            (cutoff,),
        ).fetchone()[0]

        # Entries with fills
        filled = conn.execute(
            "SELECT COUNT(*) FROM devil_tracker WHERE fill_price IS NOT NULL AND signal_timestamp >= ?",
            (cutoff,),
        ).fetchone()[0]

        # Average slippage
        avg_slip = conn.execute(
            "SELECT AVG(slippage_bps) FROM devil_tracker WHERE slippage_bps IS NOT NULL AND signal_timestamp >= ?",
            (cutoff,),
        ).fetchone()[0]

        # Average latency
        avg_lat = conn.execute(
            "SELECT AVG(latency_signal_to_fill_ms) FROM devil_tracker "
            "WHERE latency_signal_to_fill_ms IS NOT NULL AND signal_timestamp >= ?",
            (cutoff,),
        ).fetchone()[0]

        # Total devil cost
        total_devil = conn.execute(
            "SELECT SUM(devil) FROM devil_tracker WHERE devil IS NOT NULL AND signal_timestamp >= ?",
            (cutoff,),
        ).fetchone()[0]

        # Total fees
        total_fees = conn.execute(
            "SELECT SUM(fill_fee) FROM devil_tracker WHERE fill_fee IS NOT NULL AND signal_timestamp >= ?",
            (cutoff,),
        ).fetchone()[0]

        # Recent entries
        recent = conn.execute(
            "SELECT trade_id, pair, side, signal_price, fill_price, slippage_bps, "
            "fill_fee, devil, latency_signal_to_fill_ms, signal_timestamp "
            "FROM devil_tracker WHERE signal_timestamp >= ? ORDER BY rowid DESC LIMIT 20",
            (cutoff,),
        ).fetchall()

        conn.close()

        return {
            "window_hours": hours,
            "total_signals": total,
            "total_fills": filled,
            "avg_slippage_bps": round(avg_slip or 0, 2),
            "avg_latency_ms": round(avg_lat or 0, 1),
            "total_devil_cost_usd": round(total_devil or 0, 4),
            "total_fees_usd": round(total_fees or 0, 4),
            "recent": [dict(r) for r in recent],
        }
    except Exception as e:
        return {"error": str(e), "total_signals": 0, "total_fills": 0}


@router.get("/recent")
async def devil_recent(request: Request, limit: int = 50):
    """Recent devil tracker entries."""
    db = request.app.state.dashboard_config.db_path
    try:
        conn = _conn(db)
        rows = conn.execute(
            "SELECT * FROM devil_tracker ORDER BY rowid DESC LIMIT ?",
            (min(limit, 200),),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}
