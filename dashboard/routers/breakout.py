"""Breakout Scanner dashboard endpoints — reads from breakout_scans DB table."""

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


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


@router.get("/summary")
async def breakout_summary(request: Request):
    """Breakout scanner overview stats from DB."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "breakout_scans"):
                return _empty_summary()

            # Latest scan time
            row = c.execute("SELECT MAX(scan_time) FROM breakout_scans").fetchone()
            if not row or not row[0]:
                return _empty_summary()
            latest = row[0]

            # Count flagged in latest scan
            flagged = c.execute(
                "SELECT COUNT(*) FROM breakout_scans WHERE scan_time = ?", (latest,)
            ).fetchone()[0]

            # Total scanned (stored per row)
            total_scanned = c.execute(
                "SELECT total_scanned FROM breakout_scans WHERE scan_time = ? LIMIT 1",
                (latest,)
            ).fetchone()
            total_scanned = total_scanned[0] if total_scanned and total_scanned[0] else 0

            # Distinct scan times = number of scans
            scan_count = c.execute(
                "SELECT COUNT(DISTINCT scan_time) FROM breakout_scans"
            ).fetchone()[0]

            # Average flagged per scan
            avg_flagged = c.execute(
                "SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM breakout_scans GROUP BY scan_time)"
            ).fetchone()[0] or 0

            return {
                "total_scans": scan_count,
                "total_flagged": flagged,
                "avg_flagged_per_scan": round(avg_flagged, 1),
                "last_scan_seconds": 0,
                "pairs_tracked": total_scanned,
                "last_scan_time": latest,
            }
    except Exception as e:
        logger.debug(f"breakout_summary error: {e}")
        return _empty_summary()


def _empty_summary() -> dict:
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
    """Current flagged breakout signals from latest scan in DB."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "breakout_scans"):
                return _empty_signals()

            row = c.execute("SELECT MAX(scan_time) FROM breakout_scans").fetchone()
            if not row or not row[0]:
                return _empty_signals()
            latest = row[0]

            rows = c.execute("""
                SELECT product_id, symbol, score, direction, price,
                       volume_24h_usd, price_change_pct,
                       volume_score, price_score, momentum_score,
                       volatility_score, divergence_score, total_scanned
                FROM breakout_scans
                WHERE scan_time = ?
                ORDER BY score DESC
                LIMIT ?
            """, (latest, limit)).fetchall()

            signals = [dict(r) for r in rows]
            total_scanned = signals[0]["total_scanned"] if signals else 0

            # Add computed alias fields expected by frontend
            for sig in signals:
                sig["signal_strength"] = sig.get("score", 0) / 100.0
                sig["volume_surge"] = sig.get("volume_score", 0) / 10.0
                sig["momentum"] = sig.get("momentum_score", 0) / 100.0
                sig["timestamp"] = latest

            return {
                "scan_time": latest,
                "total_scanned": total_scanned,
                "total_flagged": len(signals),
                "signals": signals,
            }
    except Exception as e:
        logger.debug(f"breakout_signals error: {e}")
        return _empty_signals()


def _empty_signals() -> dict:
    return {
        "scan_time": None,
        "total_scanned": 0,
        "total_flagged": 0,
        "signals": [],
    }


@router.get("/history")
async def breakout_history(request: Request, hours: int = 24, limit: int = 100):
    """Breakout signal history across scans."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "breakout_scans"):
                return {"period_hours": hours, "count": 0, "entries": []}

            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            rows = c.execute("""
                SELECT scan_time, product_id, score, direction, price,
                       volume_24h_usd, price_change_pct
                FROM breakout_scans
                WHERE scan_time > ? AND score >= 50
                ORDER BY scan_time DESC, score DESC
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
    """Score heatmap — top 100 pairs from latest scan."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "breakout_scans"):
                return {"pairs": []}

            row = c.execute("SELECT MAX(scan_time) FROM breakout_scans").fetchone()
            if not row or not row[0]:
                return {"pairs": []}
            latest = row[0]

            rows = c.execute("""
                SELECT product_id, symbol, score, direction, price,
                       volume_24h_usd, price_change_pct,
                       volume_score, price_score, momentum_score,
                       volatility_score, divergence_score
                FROM breakout_scans
                WHERE scan_time = ?
                ORDER BY score DESC
                LIMIT 100
            """, (latest,)).fetchall()
            pairs = [dict(r) for r in rows]
            for p in pairs:
                p["tier"] = "T1" if p["score"] >= 70 else "T2" if p["score"] >= 50 else "T3" if p["score"] >= 30 else "T4"
            return {"pairs": pairs}
    except Exception:
        return {"pairs": []}
