"""Polymarket dashboard endpoints — scanner results, bridge signals, edge tracking."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/polymarket", tags=["polymarket"])

BOT_DB = Path(__file__).resolve().parent.parent.parent / "data" / "renaissance_bot.db"
# Directory name kept as "revenue-engine" for Node.js Polymarket bot compatibility
SIGNAL_FILE = Path.home() / "revenue-engine" / "data" / "output" / "renaissance-signal.json"


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
async def polymarket_summary(request: Request):
    """Scanner overview — market counts by type, opportunity count, top edge."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            # Check if table exists
            tables = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "polymarket_scanner" not in tables:
                return {"error": "polymarket_scanner table not found", "markets_by_type": {}}

            # Latest scan time
            latest = c.execute(
                "SELECT MAX(scan_time) as ts FROM polymarket_scanner"
            ).fetchone()
            last_scan = latest["ts"] if latest else None

            if not last_scan:
                return {"last_scan": None, "total_markets": 0, "markets_by_type": {},
                        "opportunities": 0, "top_edge": None}

            # Count by type from latest scan
            type_rows = c.execute("""
                SELECT market_type, COUNT(*) as cnt
                FROM polymarket_scanner
                WHERE scan_time = ?
                GROUP BY market_type
                ORDER BY cnt DESC
            """, (last_scan,)).fetchall()
            markets_by_type = {r["market_type"]: r["cnt"] for r in type_rows}

            total = sum(markets_by_type.values())

            # Opportunities (have edge)
            opp_row = c.execute("""
                SELECT COUNT(*) as cnt, MAX(edge) as max_edge
                FROM polymarket_scanner
                WHERE scan_time = ? AND edge IS NOT NULL AND edge > 0
            """, (last_scan,)).fetchall()
            opp_count = opp_row[0]["cnt"] if opp_row else 0
            max_edge = opp_row[0]["max_edge"] if opp_row else None

            # Top opportunity
            top = c.execute("""
                SELECT asset, direction, edge, confidence, yes_price, question
                FROM polymarket_scanner
                WHERE scan_time = ? AND edge IS NOT NULL AND edge > 0
                ORDER BY edge DESC LIMIT 1
            """, (last_scan,)).fetchone()

            return {
                "last_scan": last_scan,
                "total_markets": total,
                "markets_by_type": markets_by_type,
                "opportunities": opp_count,
                "max_edge": round(float(max_edge), 4) if max_edge else None,
                "top_opportunity": dict(top) if top else None,
            }
    except Exception as e:
        return {"error": str(e), "markets_by_type": {}}


@router.get("/edges")
async def polymarket_edges(request: Request, min_edge: float = 0.0, limit: int = 50):
    """Ranked edge opportunities from latest scan."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            tables = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "polymarket_scanner" not in tables:
                return {"edges": []}

            latest = c.execute(
                "SELECT MAX(scan_time) as ts FROM polymarket_scanner"
            ).fetchone()
            last_scan = latest["ts"] if latest else None
            if not last_scan:
                return {"edges": [], "scan_time": None}

            rows = c.execute("""
                SELECT condition_id, question, slug, market_type, asset,
                       timeframe_minutes, deadline, target_price,
                       yes_price, no_price, volume_24h, liquidity,
                       edge, our_probability, direction, confidence
                FROM polymarket_scanner
                WHERE scan_time = ? AND edge IS NOT NULL AND edge >= ?
                ORDER BY edge DESC
                LIMIT ?
            """, (last_scan, min_edge, limit)).fetchall()
            return {
                "scan_time": last_scan,
                "count": len(rows),
                "edges": [dict(r) for r in rows],
            }
    except Exception as e:
        return {"error": str(e), "edges": []}


@router.get("/markets")
async def polymarket_markets(request: Request, market_type: str | None = None,
                              asset: str | None = None, limit: int = 100):
    """Browse all scanned markets with optional filters."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            tables = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "polymarket_scanner" not in tables:
                return {"markets": []}

            latest = c.execute(
                "SELECT MAX(scan_time) as ts FROM polymarket_scanner"
            ).fetchone()
            last_scan = latest["ts"] if latest else None
            if not last_scan:
                return {"markets": [], "scan_time": None}

            query = """
                SELECT condition_id, question, slug, market_type, asset,
                       timeframe_minutes, deadline, target_price,
                       range_low, range_high, yes_price, no_price,
                       volume_24h, liquidity, edge, our_probability,
                       direction, confidence
                FROM polymarket_scanner
                WHERE scan_time = ?
            """
            params: list = [last_scan]

            if market_type:
                query += " AND market_type = ?"
                params.append(market_type)
            if asset:
                query += " AND asset = ?"
                params.append(asset)

            query += " ORDER BY COALESCE(edge, 0) DESC LIMIT ?"
            params.append(limit)

            rows = c.execute(query, params).fetchall()
            return {
                "scan_time": last_scan,
                "count": len(rows),
                "markets": [dict(r) for r in rows],
            }
    except Exception as e:
        return {"error": str(e), "markets": []}


@router.get("/signal")
async def polymarket_signal(request: Request):
    """Latest bridge signal from the JSON file."""
    try:
        # Try VPS path first, then local
        signal_path = SIGNAL_FILE
        if not signal_path.exists():
            # Check project-relative
            alt = Path(__file__).resolve().parent.parent.parent / "data" / "renaissance-signal.json"
            if alt.exists():
                signal_path = alt
            else:
                return {"signal": None, "note": "No signal file found"}

        data = json.loads(signal_path.read_text())
        return {"signal": data}
    except Exception as e:
        return {"signal": None, "error": str(e)}


@router.get("/history")
async def polymarket_history(request: Request, hours: int = 24, asset: str | None = None):
    """Edge history over time for charting."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            tables = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "polymarket_scanner" not in tables:
                return {"history": []}

            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

            query = """
                SELECT scan_time, COUNT(*) as total_markets,
                       SUM(CASE WHEN edge IS NOT NULL AND edge > 0 THEN 1 ELSE 0 END) as opportunities,
                       MAX(edge) as max_edge,
                       AVG(CASE WHEN edge > 0 THEN edge ELSE NULL END) as avg_edge
                FROM polymarket_scanner
                WHERE scan_time > ?
            """
            params: list = [cutoff]
            if asset:
                query += " AND asset = ?"
                params.append(asset)
            query += " GROUP BY scan_time ORDER BY scan_time"

            rows = c.execute(query, params).fetchall()
            return {
                "period_hours": hours,
                "points": len(rows),
                "history": [dict(r) for r in rows],
            }
    except Exception as e:
        return {"error": str(e), "history": []}


@router.get("/stats")
async def polymarket_stats(request: Request):
    """Bridge + scanner runtime stats."""
    result: Dict[str, Any] = {
        "bridge": None,
        "scanner": None,
    }

    # Try to get bridge stats from bot
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "polymarket_bridge"):
        try:
            result["bridge"] = bot.polymarket_bridge.get_stats()
        except Exception:
            pass
    if bot and hasattr(bot, "polymarket_scanner"):
        try:
            result["scanner"] = bot.polymarket_scanner.get_stats()
        except Exception:
            pass

    return result
