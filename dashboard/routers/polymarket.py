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
                      AND total_invested > 0
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
                    AND total_invested > 0
                """).fetchone()

                total = overall["total"] or 0
                wins = overall["wins"] or 0
                losses = overall["losses"] or 0

                # Bankroll: recalculate from P&L data (bankroll_log can drift
                # due to resolution bugs). Correct = initial + total_pnl - open_exposure.
                total_pnl = round(overall["total_pnl"] or 0, 2)
                open_exposure = round(open_row["exposure"], 2)
                corrected_bankroll = round(500.0 + total_pnl - open_exposure, 2)

                # Win rate: wins / (wins + losses), excluding CLOSED bets
                decided = wins + losses
                win_rate = round(wins / decided * 100, 1) if decided > 0 else 0

                return {
                    "bankroll": corrected_bankroll,
                    "initial_bankroll": 500.0,
                    "open_count": open_row["count"],
                    "open_exposure": open_exposure,
                    "today_pnl": round(today_row["pnl"] or 0, 2),
                    "today_bets": today_row["bets"] or 0,
                    "today_wins": today_row["wins"] or 0,
                    "total_pnl": total_pnl,
                    "total_bets": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": win_rate,
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
                    AND total_invested > 0
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
                    AND total_invested > 0
                """).fetchone()

                per_asset = c.execute("""
                    SELECT asset,
                           COUNT(*) as bets,
                           SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) as wins,
                           COALESCE(SUM(pnl), 0) as pnl,
                           COALESCE(AVG(pnl), 0) as avg_pnl
                    FROM polymarket_bets
                    WHERE status IN ('WON', 'LOST', 'CLOSED')
                    AND total_invested > 0
                    GROUP BY asset ORDER BY pnl DESC
                """).fetchall()

                total = overall["total"] or 0
                wins = overall["wins"] or 0
                losses = overall["losses"] or 0
                decided = wins + losses

                return {
                    "total_bets": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(wins / decided * 100, 1) if decided > 0 else 0,
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
                losses = overall["losses"] or 0
                decided = wins + losses

                return {
                    "total_bets": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(wins / decided * 100, 1) if decided > 0 else 0,
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
        "confidence_threshold": 52.0,
        "kelly_mode": "half-kelly",
        "stop_loss": "disabled",
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


@router.get("/ml-calibration")
async def polymarket_ml_calibration(request: Request):
    """Return ML calibration diagnostics (isotonic/Platt fitted model)."""
    import os
    cal_path = Path(__file__).resolve().parent.parent.parent / "data" / "calibration" / "calibration_model.json"
    if cal_path.exists():
        with open(cal_path) as f:
            return json.load(f)
    return {"error": "No calibration data yet. Run polymarket_calibration.py first."}


@router.get("/simulation")
async def polymarket_simulation(request: Request):
    """Return edge simulation results."""
    import os
    sim_path = Path(__file__).resolve().parent.parent.parent / "data" / "calibration" / "edge_simulation.json"
    if sim_path.exists():
        with open(sim_path) as f:
            return json.load(f)
    return {"error": "No simulation data yet. Run polymarket_edge_simulation.py first."}


@router.get("/history-stats")
async def polymarket_history_stats(request: Request):
    """Return 5-minute market history collection stats."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "polymarket_5m_history"):
                return {"error": "No history data. Run polymarket_history.py first."}

            total = c.execute("SELECT COUNT(*) FROM polymarket_5m_history").fetchone()[0]
            per_asset = c.execute("""
                SELECT asset, COUNT(*) as n,
                       SUM(resolved) as ups,
                       SUM(CASE WHEN crowd_yes_open IS NOT NULL THEN 1 ELSE 0 END) as has_crowd,
                       SUM(CASE WHEN price_start IS NOT NULL THEN 1 ELSE 0 END) as has_price,
                       MIN(window_start) as earliest,
                       MAX(window_start) as latest
                FROM polymarket_5m_history
                GROUP BY asset
            """).fetchall()

            return {
                "total_markets": total,
                "per_asset": [
                    {
                        "asset": r["asset"],
                        "count": r["n"],
                        "ups": r["ups"],
                        "up_pct": round(r["ups"] / r["n"] * 100, 1) if r["n"] > 0 else 0,
                        "has_crowd": r["has_crowd"],
                        "has_price": r["has_price"],
                    }
                    for r in per_asset
                ],
            }
    except Exception as e:
        return {"error": str(e)}


# ── Timing Features Endpoint ──


@router.get("/timing")
def get_timing_features():
    """BTC lead-lag timing features and per-asset edge breakdown."""
    try:
        from polymarket_timing_features import TimingFeatureEngine

        result = {
            "timing_engine": {
                "lead_assets": ["BTC", "ETH"],
                "follower_assets": sorted(TimingFeatureEngine.FOLLOWER_ASSETS),
                "features": [
                    "btc_1bar_ret", "btc_3bar_ret", "btc_vol_ratio",
                    "btc_alt_spread", "btc_volume_z", "lead_momentum",
                ],
            },
            "altcoin_filter": {
                "enabled": True,
                "rule": "5m markets skip BTC/ETH (negative edge per calibration)",
                "reason": "Calibration shows BTC=49.3%, ETH=49.4% on 5m (below 50%)",
            },
            "per_asset_edge": {},
        }

        # Load calibration data for per-asset accuracy
        cal_path = Path(__file__).resolve().parent.parent.parent / "data" / "calibration" / "calibration_model.json"
        if cal_path.exists():
            import json as _json
            with open(cal_path) as f:
                cal_data = _json.load(f)
            for asset, info in cal_data.get("per_asset", {}).items():
                acc = info.get("accuracy", 0) * 100  # Convert to percentage
                result["per_asset_edge"][asset] = {
                    "accuracy_pct": round(acc, 1),
                    "n_samples": info.get("n", 0),
                    "has_edge": acc > 50.0,
                    "is_follower": asset.upper() in TimingFeatureEngine.FOLLOWER_ASSETS,
                }

        # Load recent bet timing data from DB
        with _conn() as c:
            if _table_exists(c, "polymarket_bets"):
                recent = c.execute("""
                    SELECT asset, entry_side, entry_confidence, pnl,
                           opened_at, timeframe
                    FROM polymarket_bets
                    WHERE timeframe = 5
                    ORDER BY opened_at DESC LIMIT 50
                """).fetchall()
                result["recent_5m_bets"] = [
                    {
                        "asset": r["asset"],
                        "side": r["entry_side"],
                        "confidence": r["entry_confidence"],
                        "pnl": r["pnl"],
                        "opened_at": r["opened_at"],
                    }
                    for r in recent
                ]

        return result

    except Exception as e:
        return {"error": str(e)}


# --- REVERSAL STRATEGY ENDPOINTS ---


@router.get("/reversal")
async def polymarket_reversal_stats():
    """Reversal strategy performance and recent bets."""
    try:
        with _conn() as c:
            if not _table_exists(c, "polymarket_reversal_bets"):
                return {
                    "strategy": "BTC Already Told You",
                    "status": "no_data",
                    "total_bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "open_bets": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "recent_bets": [],
                    "message": "No reversal bets yet — table not created",
                }

            stats = c.execute("""
                SELECT
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_bets,
                    COALESCE(SUM(CASE WHEN status IN ('won','lost') THEN pnl ELSE 0 END), 0) as total_pnl,
                    COALESCE(SUM(bet_amount), 0) as total_wagered,
                    AVG(entry_price) as avg_entry_price,
                    AVG(btc_reversal_magnitude_pct) as avg_btc_reversal,
                    AVG(divergence_pct) as avg_divergence
                FROM polymarket_reversal_bets
            """).fetchone()

            recent = c.execute("""
                SELECT asset, direction, entry_price, bet_amount, pnl, status,
                       btc_trend_direction, btc_reversal_direction,
                       altcoin_close_vs_open, opened_at, seconds_remaining
                FROM polymarket_reversal_bets
                ORDER BY opened_at DESC LIMIT 20
            """).fetchall()

            total = stats["total_bets"] or 0
            resolved = (stats["wins"] or 0) + (stats["losses"] or 0)

            return {
                "strategy": "BTC Already Told You",
                "status": "active",
                "total_bets": total,
                "wins": stats["wins"] or 0,
                "losses": stats["losses"] or 0,
                "open_bets": stats["open_bets"] or 0,
                "win_rate": round((stats["wins"] or 0) / resolved * 100, 1) if resolved > 0 else 0,
                "total_pnl": round(stats["total_pnl"], 2),
                "total_wagered": round(stats["total_wagered"] or 0, 2),
                "roi": round(stats["total_pnl"] / (stats["total_wagered"] or 1) * 100, 1),
                "avg_entry_price": round(stats["avg_entry_price"] or 0, 3),
                "avg_btc_reversal_pct": round(stats["avg_btc_reversal"] or 0, 4),
                "avg_divergence_pct": round(stats["avg_divergence"] or 0, 4),
                "breakeven_accuracy": "15-25% depending on entry price",
                "recent_bets": [dict(r) for r in recent],
            }

    except Exception as e:
        return {"error": str(e), "strategy": "BTC Already Told You"}


@router.get("/reversal/live")
async def polymarket_reversal_live():
    """Current reversal strategy configuration and live state."""
    from polymarket_reversal import REVERSAL_ASSETS, MAX_ENTRY_PRICE, MAX_DAILY_REVERSAL_LOSS
    from polymarket_reversal import BASE_BET_SIZE, MAX_BET_SIZE, ENTRY_EARLIEST_SEC, ENTRY_LATEST_SEC

    return {
        "strategy": "BTC Already Told You",
        "entry_window": f"t={ENTRY_EARLIEST_SEC}s to t={ENTRY_LATEST_SEC}s",
        "max_entry_price": MAX_ENTRY_PRICE,
        "assets": list(REVERSAL_ASSETS.keys()),
        "daily_loss_limit": MAX_DAILY_REVERSAL_LOSS,
        "bet_size_range": f"${BASE_BET_SIZE}-${MAX_BET_SIZE}",
        "status": "active (paper)",
        "description": (
            "Detects BTC reversals mid-window, buys contrarian on altcoins "
            "at extreme odds ($0.10-$0.25). Only needs 15-25% accuracy to profit."
        ),
    }


# ─── Live Execution Endpoints ───


@router.get("/live")
async def polymarket_live_stats(request: Request):
    """Live trading statistics — separate from paper."""
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "polymarket_live_executor") and bot.polymarket_live_executor:
        try:
            return bot.polymarket_live_executor.get_stats()
        except Exception as e:
            return {"error": str(e), "live_enabled": False}

    # Fallback: read directly from DB
    try:
        with _conn() as c:
            if not _table_exists(c, "polymarket_live_bets"):
                return {"live_enabled": False, "total_bets": 0, "note": "table not created yet"}
            stats = c.execute("""
                SELECT
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_bets,
                    COALESCE(SUM(CASE WHEN status IN ('won','lost') THEN pnl ELSE 0 END), 0) as total_pnl,
                    COALESCE(SUM(order_size_usd), 0) as total_wagered
                FROM polymarket_live_bets
            """).fetchone()
            recent = c.execute("""
                SELECT asset, direction, order_price, order_size_usd,
                       pnl, status, fill_status, created_at, error_message,
                       timeframe, slug
                FROM polymarket_live_bets ORDER BY created_at DESC LIMIT 20
            """).fetchall()
            resolved = (stats["wins"] or 0) + (stats["losses"] or 0)
            return {
                "live_enabled": "unknown (separate process)",
                "total_bets": stats["total_bets"] or 0,
                "wins": stats["wins"] or 0,
                "losses": stats["losses"] or 0,
                "open_bets": stats["open_bets"] or 0,
                "total_pnl": round(float(stats["total_pnl"]), 2),
                "total_wagered": round(float(stats["total_wagered"] or 0), 2),
                "win_rate": round((stats["wins"] or 0) / resolved * 100, 1) if resolved > 0 else 0,
                "recent_bets": [dict(r) for r in recent],
            }
    except Exception as e:
        return {"error": str(e), "live_enabled": False}


@router.get("/live/kill")
async def polymarket_kill_switch(request: Request):
    """Emergency kill switch — disables all live execution immediately."""
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "polymarket_live_executor") and bot.polymarket_live_executor:
        bot.polymarket_live_executor.live_enabled = False
        logger.warning("[LIVE] *** KILL SWITCH ACTIVATED — live execution disabled ***")
        return {"killed": True, "live_enabled": False}
    return {"killed": False, "error": "Live executor not available in this process"}


# ─── Local Relay Endpoints (for geo-blocked VPS → local Mac execution) ───


@router.get("/live/pending")
async def polymarket_live_pending(request: Request):
    """Return bets awaiting relay execution from a local (non-geo-blocked) machine."""
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "polymarket_live_executor") and bot.polymarket_live_executor:
        return {"pending": bot.polymarket_live_executor.get_pending_relay_bets()}

    # Fallback: read from DB directly
    try:
        with _conn() as c:
            if not _table_exists(c, "polymarket_live_bets"):
                return {"pending": []}
            rows = c.execute("""
                SELECT id, asset, timeframe, slug, question, window_start,
                       direction, predicted_edge, ml_confidence, crowd_price_at_decision,
                       token_id, order_price, order_size_usd, order_shares,
                       order_placed_at, created_at
                FROM polymarket_live_bets
                WHERE status = 'pending_relay'
                ORDER BY created_at ASC
            """).fetchall()
            return {"pending": [dict(r) for r in rows]}
    except Exception as e:
        return {"pending": [], "error": str(e)}


@router.post("/live/confirm")
async def polymarket_live_confirm(request: Request):
    """Confirm relay execution result from local machine."""
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "invalid JSON body"}

    bet_id = body.get("bet_id")
    order_id = body.get("order_id", "")
    fill_status = body.get("fill_status", "error")
    error_msg = body.get("error", "")

    if not bet_id:
        return {"ok": False, "error": "bet_id required"}

    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "polymarket_live_executor") and bot.polymarket_live_executor:
        ok = bot.polymarket_live_executor.confirm_relay_bet(
            int(bet_id), order_id, fill_status, error_msg,
        )
        return {"ok": ok, "bet_id": bet_id, "fill_status": fill_status}

    # Fallback: write to DB directly
    try:
        with _conn() as c:
            now = datetime.now(timezone.utc).isoformat()
            if fill_status in ("placed", "filled"):
                c.execute("""
                    UPDATE polymarket_live_bets
                    SET status='open', order_id=?, fill_status=?,
                        filled_at=?, error_message=NULL
                    WHERE id=? AND status='pending_relay'
                """, (order_id, fill_status, now, int(bet_id)))
            else:
                c.execute("""
                    UPDATE polymarket_live_bets
                    SET status='error', order_id=?, fill_status=?, error_message=?
                    WHERE id=? AND status='pending_relay'
                """, (order_id, fill_status, error_msg[:500], int(bet_id)))
            c.commit()
            return {"ok": True, "bet_id": bet_id, "fill_status": fill_status}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Simple $1 UP Baseline Strategy ──

@router.get("/simple")
async def simple_up_stats():
    """Simple $1 UP baseline strategy stats."""
    try:
        with _conn() as conn:
            if not _table_exists(conn, "simple_up_bets"):
                return {"error": "simple_up_bets table not found — strategy not yet started"}

            total = conn.execute("SELECT COUNT(*) FROM simple_up_bets").fetchone()[0]
            won = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE result='WON'"
            ).fetchone()[0]
            lost = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE result='LOST'"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE result IS NULL"
            ).fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE result='expired'"
            ).fetchone()[0]
            total_pnl = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM simple_up_bets WHERE pnl IS NOT NULL"
            ).fetchone()[0]

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_total = conn.execute(
                "SELECT COUNT(*) FROM simple_up_bets WHERE date(created_at) = ?",
                (today,),
            ).fetchone()[0]
            today_pnl = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM simple_up_bets "
                "WHERE pnl IS NOT NULL AND date(created_at) = ?",
                (today,),
            ).fetchone()[0]

            # Per-asset breakdown
            per_asset = {}
            for asset in ("SOL", "DOGE"):
                row = conn.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN result='WON' THEN 1 ELSE 0 END) as wins,
                           SUM(CASE WHEN result='LOST' THEN 1 ELSE 0 END) as losses,
                           COALESCE(SUM(pnl), 0) as pnl
                    FROM simple_up_bets WHERE asset=?
                """, (asset,)).fetchone()
                per_asset[asset] = {
                    "total": row[0], "wins": row[1],
                    "losses": row[2], "pnl": round(float(row[3]), 4),
                }

            # Recent bets
            recent = conn.execute("""
                SELECT asset, window_ts, slug, entry_price, order_status,
                       result, pnl, created_at
                FROM simple_up_bets ORDER BY id DESC LIMIT 20
            """).fetchall()

            resolved = won + lost
            win_rate = (won / resolved * 100) if resolved > 0 else 0

            return {
                "strategy": "Simple $1 UP",
                "direction": "UP (always)",
                "bet_size": 1.00,
                "assets": ["SOL", "DOGE"],
                "total_bets": total,
                "won": won,
                "lost": lost,
                "pending": pending,
                "expired": expired,
                "win_rate": round(win_rate, 1),
                "total_pnl": round(float(total_pnl), 4),
                "today_bets": today_total,
                "today_pnl": round(float(today_pnl), 4),
                "per_asset": per_asset,
                "recent_bets": [dict(r) for r in recent],
            }
    except Exception as e:
        return {"error": str(e)}
