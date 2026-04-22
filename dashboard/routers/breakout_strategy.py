"""Breakout Strategy dashboard endpoints — separate $2K wallet for parabolic bets."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/breakout-strategy", tags=["breakout-strategy"])

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


# ─── 1. OVERVIEW ──────────────────────────────────────────────────────

@router.get("/overview")
async def overview(request: Request):
    """Wallet, bet size, shots left, open count, total P&L, best win."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "breakout_bets"):
                return _empty_overview()

            # Bankroll
            bankroll = 2000.0
            if _table_exists(c, "breakout_wallet"):
                row = c.execute(
                    "SELECT bankroll_after FROM breakout_wallet ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    bankroll = row["bankroll_after"]

            # Open positions
            open_count = c.execute(
                "SELECT COUNT(*) as cnt FROM breakout_bets WHERE status = 'open'"
            ).fetchone()["cnt"]

            # Closed stats
            closed = c.execute(
                "SELECT pnl_usd, pnl_pct, peak_gain_pct FROM breakout_bets "
                "WHERE status = 'closed'"
            ).fetchall()

            total_pnl = sum(r["pnl_usd"] for r in closed)
            best_win = max((r["pnl_usd"] for r in closed), default=0)
            total_bets = len(closed) + open_count

            bet_size = max(100.0, bankroll / 20)
            shots_left = int(bankroll / bet_size) if bet_size > 0 else 0

            return {
                "bankroll": round(bankroll, 2),
                "bet_size": round(bet_size, 2),
                "shots_left": shots_left,
                "open_count": open_count,
                "total_bets": total_bets,
                "total_pnl": round(total_pnl, 2),
                "best_win": round(best_win, 2),
            }
    except Exception as e:
        logger.error(f"Breakout strategy overview error: {e}")
        return _empty_overview()


def _empty_overview() -> Dict[str, Any]:
    return {
        "bankroll": 2000.0,
        "bet_size": 100.0,
        "shots_left": 20,
        "open_count": 0,
        "total_bets": 0,
        "total_pnl": 0.0,
        "best_win": 0.0,
    }


# ─── 2. OPEN POSITIONS ───────────────────────────────────────────────

@router.get("/positions")
async def positions(request: Request):
    """Open positions with live P&L, peak, hold time."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "breakout_bets"):
                return {"positions": []}

            rows = c.execute(
                "SELECT id, symbol, product_id, entry_price, current_price, "
                "peak_price, peak_gain_pct, pnl_pct, pnl_usd, bet_size_usd, "
                "entry_score, entry_volume_surge, entry_price_change_pct, "
                "opened_at, last_significant_move_at, last_updated "
                "FROM breakout_bets WHERE status = 'open' "
                "ORDER BY opened_at DESC"
            ).fetchall()

            return {"positions": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Breakout strategy positions error: {e}")
        return {"positions": []}


# ─── 3. HISTORY ───────────────────────────────────────────────────────

@router.get("/history")
async def history(request: Request, limit: int = 50):
    """Closed bets with exit reason, peak gain."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "breakout_bets"):
                return {"history": []}

            rows = c.execute(
                "SELECT id, symbol, product_id, entry_price, exit_price, "
                "peak_price, peak_gain_pct, pnl_pct, pnl_usd, bet_size_usd, "
                "entry_score, exit_reason, opened_at, closed_at "
                "FROM breakout_bets WHERE status = 'closed' "
                "ORDER BY closed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

            return {"history": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Breakout strategy history error: {e}")
        return {"history": []}


# ─── 4. WALLET ────────────────────────────────────────────────────────

@router.get("/wallet")
async def wallet(request: Request, limit: int = 50):
    """Wallet event log."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "breakout_wallet"):
                return {"events": []}

            rows = c.execute(
                "SELECT id, event_type, amount, bankroll_after, bet_id, "
                "symbol, detail, timestamp "
                "FROM breakout_wallet ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

            return {"events": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Breakout strategy wallet error: {e}")
        return {"events": []}


# ─── 5. STATS ─────────────────────────────────────────────────────────

@router.get("/stats")
async def stats(request: Request):
    """Win rate, avg winner/loser, biggest win/loss, exit reason breakdown."""
    cfg = request.app.state.dashboard_config
    try:
        with _conn(cfg.db_path) as c:
            if not _table_exists(c, "breakout_bets"):
                return _empty_stats()

            closed = c.execute(
                "SELECT pnl_usd, pnl_pct, peak_gain_pct, exit_reason "
                "FROM breakout_bets WHERE status = 'closed'"
            ).fetchall()

            if not closed:
                return _empty_stats()

            wins = [r for r in closed if r["pnl_usd"] > 0]
            losses = [r for r in closed if r["pnl_usd"] <= 0]

            win_rate = len(wins) / len(closed) * 100

            avg_winner = (
                sum(r["pnl_pct"] for r in wins) / len(wins) if wins else 0
            )
            avg_loser = (
                sum(r["pnl_pct"] for r in losses) / len(losses) if losses else 0
            )
            biggest_win = max(r["pnl_usd"] for r in closed)
            biggest_loss = min(r["pnl_usd"] for r in closed)
            best_peak = max(r["peak_gain_pct"] for r in closed)

            # Exit reason breakdown
            exit_reasons: Dict[str, int] = {}
            for r in closed:
                reason = r["exit_reason"] or "unknown"
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

            return {
                "total_closed": len(closed),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(win_rate, 1),
                "avg_winner_pct": round(avg_winner, 1),
                "avg_loser_pct": round(avg_loser, 1),
                "biggest_win": round(biggest_win, 2),
                "biggest_loss": round(biggest_loss, 2),
                "best_peak_gain_pct": round(best_peak, 1),
                "exit_reasons": exit_reasons,
            }
    except Exception as e:
        logger.error(f"Breakout strategy stats error: {e}")
        return _empty_stats()


def _empty_stats() -> Dict[str, Any]:
    return {
        "total_closed": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_winner_pct": 0.0,
        "avg_loser_pct": 0.0,
        "biggest_win": 0.0,
        "biggest_loss": 0.0,
        "best_peak_gain_pct": 0.0,
        "exit_reasons": {},
    }
