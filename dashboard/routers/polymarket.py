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


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


@router.get("/positions")
async def polymarket_positions(request: Request, status: str | None = None, limit: int = 100):
    """All Polymarket positions — open and resolved."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "polymarket_positions"):
                return {"positions": []}

            query = """
                SELECT position_id, condition_id, slug, question, market_type, asset,
                       direction, entry_price, shares, bet_amount, edge_at_entry,
                       our_prob_at_entry, crowd_prob_at_entry, target_price, deadline,
                       status, exit_price, pnl, opened_at, closed_at, notes
                FROM polymarket_positions
            """
            params: list = []
            if status:
                query += " WHERE status = ?"
                params.append(status)
            query += " ORDER BY opened_at DESC LIMIT ?"
            params.append(limit)

            rows = c.execute(query, params).fetchall()
            return {"positions": [dict(r) for r in rows]}
    except Exception as e:
        return {"error": str(e), "positions": []}


@router.get("/pnl")
async def polymarket_pnl(request: Request):
    """P&L summary: total, 24h, per-asset, per-market-type."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "polymarket_positions"):
                return _empty_pnl()

            # Overall resolved
            overall = c.execute("""
                SELECT
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(SUM(bet_amount), 0) as total_wagered
                FROM polymarket_positions
                WHERE status IN ('won', 'lost')
            """).fetchone()

            # 24h
            pnl_24h = c.execute("""
                SELECT COALESCE(SUM(pnl), 0) as pnl,
                       COUNT(*) as bets,
                       SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins
                FROM polymarket_positions
                WHERE status IN ('won', 'lost')
                  AND closed_at >= datetime('now', '-24 hours')
            """).fetchone()

            # Per asset
            per_asset = c.execute("""
                SELECT asset,
                       COUNT(*) as bets,
                       SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                       COALESCE(SUM(pnl), 0) as pnl
                FROM polymarket_positions
                WHERE status IN ('won', 'lost')
                GROUP BY asset ORDER BY pnl DESC
            """).fetchall()

            # Per market type
            per_type = c.execute("""
                SELECT market_type,
                       COUNT(*) as bets,
                       SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                       COALESCE(SUM(pnl), 0) as pnl
                FROM polymarket_positions
                WHERE status IN ('won', 'lost')
                GROUP BY market_type ORDER BY pnl DESC
            """).fetchall()

            # Open positions summary
            open_summary = c.execute("""
                SELECT COUNT(*) as count,
                       COALESCE(SUM(bet_amount), 0) as exposure
                FROM polymarket_positions
                WHERE status = 'open'
            """).fetchone()

            # Bankroll
            bankroll_row = None
            if _table_exists(c, "polymarket_bankroll_log"):
                bankroll_row = c.execute(
                    "SELECT bankroll FROM polymarket_bankroll_log ORDER BY id DESC LIMIT 1"
                ).fetchone()

            total_bets = overall['total_bets'] or 0
            wins = overall['wins'] or 0
            total_wagered = overall['total_wagered'] or 0

            return {
                "bankroll": bankroll_row['bankroll'] if bankroll_row else 500.0,
                "open_count": open_summary['count'],
                "open_exposure": round(open_summary['exposure'], 2),
                "total_bets": total_bets,
                "wins": wins,
                "losses": overall['losses'] or 0,
                "win_rate": round(wins / total_bets * 100, 1) if total_bets > 0 else 0,
                "total_pnl": round(overall['total_pnl'] or 0, 2),
                "total_wagered": round(total_wagered, 2),
                "roi": round((overall['total_pnl'] or 0) / total_wagered * 100, 1) if total_wagered > 0 else 0,
                "pnl_24h": round(pnl_24h['pnl'] or 0, 2),
                "bets_24h": pnl_24h['bets'] or 0,
                "per_asset": [dict(r) for r in per_asset],
                "per_type": [dict(r) for r in per_type],
            }
    except Exception as e:
        return {"error": str(e), **_empty_pnl()}


def _empty_pnl() -> dict:
    return {
        "bankroll": 500.0, "open_count": 0, "open_exposure": 0,
        "total_bets": 0, "wins": 0, "losses": 0, "win_rate": 0,
        "total_pnl": 0, "total_wagered": 0, "roi": 0,
        "pnl_24h": 0, "bets_24h": 0, "per_asset": [], "per_type": [],
    }


@router.get("/executor")
async def polymarket_executor_status(request: Request):
    """Executor status: bankroll, recent bets."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "polymarket_positions"):
                return {"bankroll": 500.0, "recent_bets": []}

            bankroll_row = None
            if _table_exists(c, "polymarket_bankroll_log"):
                bankroll_row = c.execute(
                    "SELECT bankroll FROM polymarket_bankroll_log ORDER BY id DESC LIMIT 1"
                ).fetchone()

            recent = c.execute("""
                SELECT position_id, question, direction, bet_amount, edge_at_entry,
                       status, pnl, opened_at, closed_at
                FROM polymarket_positions
                ORDER BY opened_at DESC LIMIT 10
            """).fetchall()

            return {
                "bankroll": bankroll_row['bankroll'] if bankroll_row else 500.0,
                "recent_bets": [dict(r) for r in recent],
            }
    except Exception as e:
        return {"error": str(e), "bankroll": 500.0, "recent_bets": []}
