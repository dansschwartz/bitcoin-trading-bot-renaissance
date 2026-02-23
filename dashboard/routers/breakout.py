"""Breakout Scanner dashboard endpoints — live scan results + history."""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/breakout", tags=["breakout"])

BOT_DB = Path(__file__).resolve().parent.parent.parent / "data" / "renaissance_bot.db"


@contextmanager
def _conn(db_path: str | None = None):
    path = db_path or str(BOT_DB)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@router.get("/summary")
async def breakout_summary(request: Request):
    """Breakout scanner overview stats."""
    cfg = request.app.state.dashboard_config
    emitter = getattr(request.app.state, "emitter", None)

    # Try live data from emitter cache
    cached = emitter.get_cached("breakout.summary") if emitter else None
    if cached:
        return cached

    # Fallback: return defaults
    return {
        "total_scans": 0,
        "total_flagged": 0,
        "avg_flagged_per_scan": 0,
        "last_scan_seconds": 0,
        "pairs_tracked": 0,
        "last_scan_time": None,
    }


@router.get("/signals")
async def breakout_signals(request: Request, limit: int = 30):
    """Current flagged breakout signals from emitter cache."""
    emitter = getattr(request.app.state, "emitter", None)
    cached = emitter.get_cached("breakout.signals") if emitter else None
    if cached and isinstance(cached, dict):
        signals = cached.get("signals", [])[:limit]
        return {
            "scan_time": cached.get("scan_time"),
            "total_scanned": cached.get("total_scanned", 0),
            "total_flagged": len(signals),
            "signals": signals,
        }

    return {
        "scan_time": None,
        "total_scanned": 0,
        "total_flagged": 0,
        "signals": [],
    }


@router.get("/history")
async def breakout_history(request: Request, hours: int = 24, limit: int = 100):
    """Breakout flag history from decisions table."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            rows = c.execute("""
                SELECT product_id, confidence as score, signal_strength,
                       reasoning, timestamp
                FROM trading_decisions
                WHERE action = 'BREAKOUT_FLAG' AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (cutoff, limit)).fetchall()
            return {
                "period_hours": hours,
                "count": len(rows),
                "entries": [dict(r) for r in rows],
            }
    except Exception:
        return {"period_hours": hours, "count": 0, "entries": []}


@router.get("/heatmap")
async def breakout_heatmap(request: Request):
    """Score heatmap from emitter cache — top 100 pairs by breakout score."""
    emitter = getattr(request.app.state, "emitter", None)
    cached = emitter.get_cached("breakout.heatmap") if emitter else None
    if cached and isinstance(cached, dict):
        return cached

    return {"pairs": []}
