"""Polymarket dashboard endpoints — Strategy A v2."""

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


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


# ─── NEW v2 ENDPOINTS ──────────────────────────────────────────────


@router.get("/overview")
async def polymarket_overview(request: Request):
    """Bankroll, open positions, today's P&L, win rate."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "polymarket_positions"):
                return _empty_overview()

            # Bankroll
            bankroll = 500.0
            if _table_exists(c, "polymarket_bankroll_log"):
                row = c.execute(
                    "SELECT bankroll FROM polymarket_bankroll_log ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    bankroll = row["bankroll"]

            # Open positions
            open_row = c.execute("""
                SELECT COUNT(*) as count, COALESCE(SUM(bet_amount), 0) as exposure
                FROM polymarket_positions WHERE status = 'open'
            """).fetchone()

            # Today's P&L
            today_row = c.execute("""
                SELECT COALESCE(SUM(pnl), 0) as pnl, COUNT(*) as bets,
                       SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins
                FROM polymarket_positions
                WHERE status IN ('won', 'lost', 'sold')
                  AND closed_at >= date('now')
            """).fetchone()

            # Overall stats
            overall = c.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses,
                       COALESCE(SUM(pnl), 0) as total_pnl,
                       COALESCE(SUM(bet_amount), 0) as total_wagered
                FROM polymarket_positions
                WHERE status IN ('won', 'lost', 'sold')
            """).fetchone()

            total = overall["total"] or 0
            wins = overall["wins"] or 0

            return {
                "bankroll": round(bankroll, 2),
                "initial_bankroll": 500.0,
                "open_count": open_row["count"],
                "open_exposure": round(open_row["exposure"], 2),
                "today_pnl": round(today_row["pnl"] or 0, 2),
                "today_bets": today_row["bets"] or 0,
                "today_wins": today_row["wins"] or 0,
                "total_pnl": round(overall["total_pnl"] or 0, 2),
                "total_bets": total,
                "wins": wins,
                "losses": overall["losses"] or 0,
                "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                "total_wagered": round(overall["total_wagered"] or 0, 2),
            }
    except Exception as e:
        return {"error": str(e), **_empty_overview()}


def _empty_overview() -> dict:
    return {
        "bankroll": 500.0, "initial_bankroll": 500.0,
        "open_count": 0, "open_exposure": 0,
        "today_pnl": 0, "today_bets": 0, "today_wins": 0,
        "total_pnl": 0, "total_bets": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "total_wagered": 0,
    }


@router.get("/positions")
async def polymarket_positions(request: Request, status: str | None = None, limit: int = 100):
    """Current open bets with live P&L."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "polymarket_positions"):
                return {"positions": []}

            query = """
                SELECT position_id, slug, question, market_type, asset,
                       direction, entry_price, shares, bet_amount, edge_at_entry,
                       our_prob_at_entry, crowd_prob_at_entry, deadline,
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
            positions = []
            for r in rows:
                p = dict(r)
                # Parse unrealized P&L from notes
                notes = p.get("notes", "") or ""
                unrealized = None
                if "unrealized=" in notes:
                    try:
                        val = notes.split("unrealized=")[1].split(",")[0]
                        unrealized = float(val)
                    except (ValueError, IndexError):
                        pass
                p["unrealized_pnl"] = unrealized
                positions.append(p)

            return {"positions": positions}
    except Exception as e:
        return {"error": str(e), "positions": []}


@router.get("/history")
async def polymarket_history(request: Request, limit: int = 100, asset: str | None = None):
    """Closed bets with outcomes (paginated)."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "polymarket_positions"):
                return {"bets": []}

            query = """
                SELECT position_id, slug, asset, direction, entry_price, exit_price,
                       bet_amount, pnl, status, opened_at, closed_at, notes
                FROM polymarket_positions
                WHERE status IN ('won', 'lost', 'sold', 'expired')
            """
            params: list = []
            if asset:
                query += " AND asset = ?"
                params.append(asset)
            query += " ORDER BY closed_at DESC LIMIT ?"
            params.append(limit)

            rows = c.execute(query, params).fetchall()
            return {"bets": [dict(r) for r in rows]}
    except Exception as e:
        return {"error": str(e), "bets": []}


@router.get("/instruments")
async def polymarket_instruments():
    """4 instruments with enabled status."""
    try:
        from polymarket_strategy_a import INSTRUMENTS
        result = []
        for key, inst in INSTRUMENTS.items():
            result.append({
                "key": key,
                "asset": inst.asset,
                "ml_pair": inst.ml_pair,
                "enabled": inst.enabled,
                "lead_asset": inst.lead_asset,
            })
        return {"instruments": result}
    except Exception as e:
        return {"error": str(e), "instruments": []}


@router.get("/stats")
async def polymarket_stats(request: Request):
    """Aggregate stats: total bets, win rate, avg return, best/worst."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "polymarket_positions"):
                return _empty_stats()

            # Overall
            overall = c.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses,
                       COALESCE(SUM(pnl), 0) as total_pnl,
                       COALESCE(AVG(pnl), 0) as avg_pnl,
                       MAX(pnl) as best,
                       MIN(pnl) as worst
                FROM polymarket_positions
                WHERE status IN ('won', 'lost', 'sold') AND strategy = 'strategy_a'
            """).fetchone()

            # Per asset
            per_asset = c.execute("""
                SELECT asset,
                       COUNT(*) as bets,
                       SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                       COALESCE(SUM(pnl), 0) as pnl,
                       COALESCE(AVG(pnl), 0) as avg_pnl
                FROM polymarket_positions
                WHERE status IN ('won', 'lost', 'sold') AND strategy = 'strategy_a'
                GROUP BY asset ORDER BY pnl DESC
            """).fetchall()

            # Recent activity from polymarket_bets
            activity = []
            if _table_exists(c, "polymarket_bets"):
                bets = c.execute("""
                    SELECT timestamp, asset, direction, action, ml_confidence,
                           token_cost, amount_usd, notes
                    FROM polymarket_bets
                    ORDER BY id DESC LIMIT 10
                """).fetchall()
                activity = [dict(b) for b in bets]

            total = overall["total"] or 0
            wins = overall["wins"] or 0

            return {
                "total_bets": total,
                "wins": wins,
                "losses": overall["losses"] or 0,
                "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                "total_pnl": round(overall["total_pnl"] or 0, 2),
                "avg_return": round(overall["avg_pnl"] or 0, 2),
                "best_trade": round(overall["best"] or 0, 2),
                "worst_trade": round(overall["worst"] or 0, 2),
                "per_asset": [dict(r) for r in per_asset],
                "recent_activity": activity,
            }
    except Exception as e:
        return {"error": str(e), **_empty_stats()}


def _empty_stats() -> dict:
    return {
        "total_bets": 0, "wins": 0, "losses": 0, "win_rate": 0,
        "total_pnl": 0, "avg_return": 0, "best_trade": 0, "worst_trade": 0,
        "per_asset": [], "recent_activity": [],
    }


# ─── OLD ENDPOINTS (backward compat — return empty/defaults) ───────


@router.get("/summary")
async def polymarket_summary(request: Request):
    """Legacy: scanner summary."""
    return {"last_scan": None, "total_markets": 0, "markets_by_type": {},
            "opportunities": 0, "max_edge": None, "top_opportunity": None}


@router.get("/edges")
async def polymarket_edges(request: Request, min_edge: float = 0.0, limit: int = 50):
    """Legacy: edge opportunities."""
    return {"edges": [], "scan_time": None, "count": 0}


@router.get("/markets")
async def polymarket_markets(request: Request, market_type: str | None = None,
                              asset: str | None = None, limit: int = 100):
    """Legacy: scanned markets."""
    return {"markets": [], "scan_time": None, "count": 0}


@router.get("/signal")
async def polymarket_signal(request: Request):
    """Legacy: bridge signal."""
    try:
        signal_path = SIGNAL_FILE
        if not signal_path.exists():
            alt = Path(__file__).resolve().parent.parent.parent / "data" / "renaissance-signal.json"
            if alt.exists():
                signal_path = alt
            else:
                return {"signal": None}
        data = json.loads(signal_path.read_text())
        return {"signal": data}
    except Exception:
        return {"signal": None}


@router.get("/pnl")
async def polymarket_pnl(request: Request):
    """Legacy: redirect to overview."""
    return await polymarket_overview(request)


@router.get("/executor")
async def polymarket_executor_status(request: Request):
    """Legacy: executor status."""
    return {"bankroll": 500.0, "recent_bets": []}


@router.get("/lifecycle")
async def polymarket_lifecycle(request: Request, limit: int = 50):
    """Legacy: lifecycle audit trail."""
    return {"lifecycles": []}


@router.get("/lifecycle/stats")
async def polymarket_lifecycle_stats(request: Request):
    """Legacy: lifecycle stats."""
    return {
        "by_conviction": [], "reversal_rate": {"total": 0, "reversed": 0, "pct": 0},
        "ml_accuracy": {"total": 0, "correct": 0, "pct": 0},
        "crowd_lag_by_asset": [], "by_asset": [],
    }
