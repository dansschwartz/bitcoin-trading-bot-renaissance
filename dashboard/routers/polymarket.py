"""Polymarket dashboard endpoints — Strategy A v3."""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/polymarket", tags=["polymarket"])

BOT_DB = Path(__file__).resolve().parent.parent.parent / "data" / "renaissance_bot.db"
SIGNAL_FILE = Path.home() / "revenue-engine" / "data" / "output" / "renaissance-signal.json"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

# In-memory cache for live markets
_live_markets_cache: Dict[str, Any] = {"data": None, "ts": 0}
_LIVE_CACHE_TTL = 30  # seconds


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


def _has_column(conn, table: str, column: str) -> bool:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return column in cols


# --- NEW v3 ENDPOINTS ---


@router.get("/overview")
async def polymarket_overview(request: Request):
    """Bankroll, open bets, today's P&L, win rate."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            # Bankroll
            bankroll = 500.0
            if _table_exists(c, "polymarket_bankroll_log"):
                row = c.execute(
                    "SELECT bankroll FROM polymarket_bankroll_log ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    bankroll = row["bankroll"]

            # Try new polymarket_bets table first
            if _table_exists(c, "polymarket_bets") and _has_column(c, "polymarket_bets", "entry_side"):
                # Open bets
                open_row = c.execute("""
                    SELECT COUNT(*) as count, COALESCE(SUM(total_invested), 0) as exposure
                    FROM polymarket_bets WHERE status = 'OPEN'
                """).fetchone()

                # Today's P&L
                today_row = c.execute("""
                    SELECT COALESCE(SUM(pnl), 0) as pnl, COUNT(*) as bets,
                           SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) as wins
                    FROM polymarket_bets
                    WHERE status IN ('WON', 'LOST', 'CLOSED')
                      AND exit_at >= date('now')
                """).fetchone()

                # Overall stats
                overall = c.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) as wins,
                           SUM(CASE WHEN status='LOST' THEN 1 ELSE 0 END) as losses,
                           COALESCE(SUM(pnl), 0) as total_pnl,
                           COALESCE(SUM(total_invested), 0) as total_wagered
                    FROM polymarket_bets
                    WHERE status IN ('WON', 'LOST', 'CLOSED')
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
                    "model_info": _model_info(),
                }

            # Fallback to old polymarket_positions table
            if _table_exists(c, "polymarket_positions"):
                open_row = c.execute("""
                    SELECT COUNT(*) as count, COALESCE(SUM(bet_amount), 0) as exposure
                    FROM polymarket_positions WHERE status = 'open'
                """).fetchone()

                today_row = c.execute("""
                    SELECT COALESCE(SUM(pnl), 0) as pnl, COUNT(*) as bets,
                           SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins
                    FROM polymarket_positions
                    WHERE status IN ('won', 'lost', 'sold')
                      AND closed_at >= date('now')
                """).fetchone()

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

            return _empty_overview()
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
    """Current open bets from new polymarket_bets table."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            # Try new table first
            if _table_exists(c, "polymarket_bets") and _has_column(c, "polymarket_bets", "entry_side"):
                where = "WHERE status = 'OPEN'" if status == "open" else ""
                if status and status != "open":
                    where = f"WHERE status = '{status.upper()}'"

                rows = c.execute(f"""
                    SELECT id, slug, asset, entry_side, entry_token_cost, entry_amount,
                           entry_tokens, entry_confidence, adds, total_invested,
                           total_tokens, avg_cost, status, exit_price, exit_reason,
                           exit_at, pnl, return_pct, regime, entry_asset_price,
                           exit_asset_price, opened_at, question
                    FROM polymarket_bets
                    {where}
                    ORDER BY opened_at DESC LIMIT ?
                """, (limit,)).fetchall()

                positions = []
                for r in rows:
                    p = dict(r)
                    # Parse adds count
                    adds_raw = p.get("adds", "[]") or "[]"
                    try:
                        adds = json.loads(adds_raw)
                        p["adds_count"] = len(adds)
                    except (json.JSONDecodeError, TypeError):
                        p["adds_count"] = 0
                    positions.append(p)

                return {"positions": positions}

            # Fallback to old table
            if _table_exists(c, "polymarket_positions"):
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
                return {"positions": [dict(r) for r in rows]}

            return {"positions": []}
    except Exception as e:
        return {"error": str(e), "positions": []}


@router.get("/history")
async def polymarket_history(request: Request, limit: int = 100, asset: str | None = None):
    """Closed bets from new polymarket_bets table."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            # Try new table first
            if _table_exists(c, "polymarket_bets") and _has_column(c, "polymarket_bets", "entry_side"):
                query = """
                    SELECT id, slug, asset, entry_side, entry_token_cost, entry_confidence,
                           adds, total_invested, total_tokens, avg_cost, status,
                           exit_price, exit_reason, exit_at, pnl, return_pct,
                           regime, opened_at, question
                    FROM polymarket_bets
                    WHERE status IN ('WON', 'LOST', 'CLOSED')
                """
                params: list = []
                if asset:
                    query += " AND asset = ?"
                    params.append(asset)
                query += " ORDER BY exit_at DESC LIMIT ?"
                params.append(limit)

                rows = c.execute(query, params).fetchall()
                bets = []
                for r in rows:
                    b = dict(r)
                    adds_raw = b.get("adds", "[]") or "[]"
                    try:
                        adds = json.loads(adds_raw)
                        b["adds_count"] = len(adds)
                    except (json.JSONDecodeError, TypeError):
                        b["adds_count"] = 0
                    bets.append(b)
                return {"bets": bets}

            # Fallback
            if _table_exists(c, "polymarket_positions"):
                query = """
                    SELECT position_id, slug, asset, direction, entry_price, exit_price,
                           bet_amount, pnl, status, opened_at, closed_at, notes
                    FROM polymarket_positions
                    WHERE status IN ('won', 'lost', 'sold', 'expired')
                """
                params = []
                if asset:
                    query += " AND asset = ?"
                    params.append(asset)
                query += " ORDER BY closed_at DESC LIMIT ?"
                params.append(limit)

                rows = c.execute(query, params).fetchall()
                return {"bets": [dict(r) for r in rows]}

            return {"bets": []}
    except Exception as e:
        return {"error": str(e), "bets": []}


@router.get("/instruments")
async def polymarket_instruments():
    """4 instruments with enabled status (no lead_asset)."""
    try:
        from polymarket_strategy_a import INSTRUMENTS
        result = []
        for key, inst in INSTRUMENTS.items():
            result.append({
                "key": key,
                "asset": inst.asset,
                "ml_pair": inst.ml_pair,
                "enabled": inst.enabled,
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
            # Try new table first
            if _table_exists(c, "polymarket_bets") and _has_column(c, "polymarket_bets", "entry_side"):
                overall = c.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) as wins,
                           SUM(CASE WHEN status='LOST' THEN 1 ELSE 0 END) as losses,
                           COALESCE(SUM(pnl), 0) as total_pnl,
                           COALESCE(AVG(pnl), 0) as avg_pnl,
                           MAX(pnl) as best,
                           MIN(pnl) as worst
                    FROM polymarket_bets
                    WHERE status IN ('WON', 'LOST', 'CLOSED')
                """).fetchone()

                per_asset = c.execute("""
                    SELECT asset,
                           COUNT(*) as bets,
                           SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) as wins,
                           COALESCE(SUM(pnl), 0) as pnl,
                           COALESCE(AVG(pnl), 0) as avg_pnl
                    FROM polymarket_bets
                    WHERE status IN ('WON', 'LOST', 'CLOSED')
                    GROUP BY asset ORDER BY pnl DESC
                """).fetchall()

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
                }

            # Fallback to old table
            if _table_exists(c, "polymarket_positions"):
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
                }

            return _empty_stats()
    except Exception as e:
        return {"error": str(e), **_empty_stats()}


def _empty_stats() -> dict:
    return {
        "total_bets": 0, "wins": 0, "losses": 0, "win_rate": 0,
        "total_pnl": 0, "avg_return": 0, "best_trade": 0, "worst_trade": 0,
        "per_asset": [],
    }


# --- 3 NEW ENDPOINTS ---


@router.get("/live-markets")
async def polymarket_live_markets(request: Request):
    """Fetch current 15m markets from Gamma API for all 4 assets. 30s cache."""
    global _live_markets_cache
    now = time.time()

    if _live_markets_cache["data"] is not None and (now - _live_markets_cache["ts"]) < _LIVE_CACHE_TTL:
        return _live_markets_cache["data"]

    try:
        from polymarket_strategy_a import INSTRUMENTS

        cfg = request.app.state.dashboard_config
        now_ts = int(time.time())
        window_ts = (now_ts // 900) * 900
        prev_window_ts = window_ts - 900

        markets = []
        for inst_key, inst in INSTRUMENTS.items():
            if not inst.enabled:
                continue

            for ts in [window_ts, prev_window_ts]:
                slug = inst.slug_pattern.format(ts=ts)
                try:
                    resp = requests.get(GAMMA_MARKETS_URL, params={"slug": slug}, timeout=5)
                    if resp.status_code != 200 or not resp.json():
                        continue
                    m = resp.json()[0]

                    prices = m.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        try:
                            prices = json.loads(prices)
                        except (json.JSONDecodeError, TypeError):
                            prices = []
                    yes_price = float(prices[0]) if prices and len(prices) >= 1 else 0.5
                    no_price = float(prices[1]) if prices and len(prices) >= 2 else 0.5

                    deadline = m.get("endDate", "")
                    mins_left = None
                    if deadline:
                        try:
                            dl = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                            mins_left = round((dl - datetime.now(timezone.utc)).total_seconds() / 60.0, 1)
                        except (ValueError, TypeError):
                            pass

                    # Check if we have a bet on this slug
                    our_bet = None
                    with _conn(cfg.db_path) as c:
                        if _table_exists(c, "polymarket_bets") and _has_column(c, "polymarket_bets", "entry_side"):
                            bet_row = c.execute(
                                "SELECT entry_side, total_invested, avg_cost, status "
                                "FROM polymarket_bets WHERE slug = ? AND status = 'OPEN' LIMIT 1",
                                (slug,)
                            ).fetchone()
                            if bet_row:
                                our_bet = dict(bet_row)

                    markets.append({
                        "asset": inst.asset,
                        "slug": slug,
                        "question": m.get("question", ""),
                        "yes_price": round(yes_price, 3),
                        "no_price": round(no_price, 3),
                        "minutes_left": mins_left,
                        "deadline": deadline,
                        "resolved": m.get("resolved", False),
                        "volume_24h": float(m.get("volume24hr", 0) or 0),
                        "our_bet": our_bet,
                    })
                except Exception:
                    continue

        # Filter out expired/resolved markets
        markets = [m for m in markets if (m.get("minutes_left") or 0) > 0 and not m.get("resolved")]

        result = {"markets": markets, "fetched_at": datetime.now(timezone.utc).isoformat()}
        _live_markets_cache = {"data": result, "ts": now}
        return result
    except Exception as e:
        return {"error": str(e), "markets": []}


@router.get("/calibration")
async def polymarket_calibration(request: Request):
    """Accuracy by confidence bucket from resolved bets.

    Buckets calibrated for crash-regime LightGBM (55-60% range).
    """
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "polymarket_bets") or not _has_column(c, "polymarket_bets", "entry_side"):
                return {"buckets": [], "model_info": _model_info()}

            buckets = []
            for low, high, label in [
                (50, 52, "50-52%"),
                (52, 55, "52-55%"),
                (55, 58, "55-58%"),
                (58, 65, "58-65%"),
                (65, 101, "65%+"),
            ]:
                row = c.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN status = 'WON' THEN 1 ELSE 0 END) as wins
                    FROM polymarket_bets
                    WHERE status IN ('WON', 'LOST')
                      AND entry_confidence >= ? AND entry_confidence < ?
                """, (low, high)).fetchone()

                total = row["total"] or 0
                wins = row["wins"] or 0
                buckets.append({
                    "label": label,
                    "total": total,
                    "wins": wins,
                    "accuracy": round(wins / total * 100, 1) if total > 0 else 0,
                })

            return {"buckets": buckets, "model_info": _model_info()}
    except Exception as e:
        return {"error": str(e), "buckets": [], "model_info": _model_info()}


def _model_info() -> dict:
    """Return crash model metadata for dashboard display."""
    return {
        "source": "crash_lightgbm",
        "test_accuracy": 52.9,
        "test_auc": 0.5432,
        "confidence_threshold": 55.0,
        "kelly_mode": "half-kelly",
        "stop_loss": "40% share price drop",
    }


@router.get("/skip-log")
async def polymarket_skip_log(request: Request, limit: int = 20):
    """Recent skipped opportunities from polymarket_skip_log."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "polymarket_skip_log"):
                return {"skips": []}

            rows = c.execute("""
                SELECT timestamp, asset, slug, reason, ml_confidence,
                       token_cost, ml_direction, minutes_left
                FROM polymarket_skip_log
                ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()

            return {"skips": [dict(r) for r in rows]}
    except Exception as e:
        return {"error": str(e), "skips": []}


# --- OLD ENDPOINTS (backward compat — return empty/defaults) ---


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
