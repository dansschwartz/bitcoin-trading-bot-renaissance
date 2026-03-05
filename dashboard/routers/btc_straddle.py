"""BTC Straddle endpoints — status, history, hourly P&L, aggregate stats."""

import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/straddle", tags=["straddle"])


def _safe_query(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    """Execute query, returning empty list if table doesn't exist."""
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


@router.get("/status")
async def straddle_status(request: Request) -> dict[str, Any]:
    """Straddle engine status — open straddles, mode, config."""
    # Try live status from bot
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "straddle_engine") and bot.straddle_engine:
        try:
            return bot.straddle_engine.get_status()
        except Exception:
            pass

    # Fallback: derive from database
    db = request.app.state.dashboard_config.db_path
    total = _safe_query(db, "SELECT COUNT(*) as cnt FROM straddle_log WHERE status != 'BLOCKED'")
    open_count = _safe_query(db, "SELECT COUNT(*) as cnt FROM straddle_log WHERE status = 'OPEN'")
    closed = _safe_query(
        db,
        """SELECT COUNT(*) as cnt,
                  COALESCE(SUM(CASE WHEN net_pnl_bps > 0 THEN 1 ELSE 0 END), 0) as winners,
                  COALESCE(SUM(net_pnl_usd), 0) as pnl
           FROM straddle_log WHERE status = 'CLOSED'""",
    )
    c = closed[0] if closed else {'cnt': 0, 'winners': 0, 'pnl': 0}
    wr = (c['winners'] / c['cnt'] * 100) if c['cnt'] > 0 else 0

    return {
        'active': False,
        'observation_mode': True,
        'pair': 'BTCUSDT',
        'size_usd': 5,
        'interval_seconds': 60,
        'max_open': 1,
        'open_count': open_count[0]['cnt'] if open_count else 0,
        'open_straddles': [],
        'total_opened': total[0]['cnt'] if total else 0,
        'total_closed': c['cnt'],
        'total_winners': c['winners'],
        'win_rate': round(wr, 1),
        'total_pnl_usd': round(c['pnl'], 4),
        'daily_loss_usd': 0,
        'dead_zone_blocks': 0,
        'config': {},
    }


@router.get("/history")
async def straddle_history(request: Request, limit: int = 100) -> list[dict]:
    """Recent closed straddles."""
    db = request.app.state.dashboard_config.db_path
    return _safe_query(
        db,
        """SELECT id, opened_at, closed_at, entry_price, vol_prediction, status,
                  long_exit_price, short_exit_price, long_exit_reason, short_exit_reason,
                  long_pnl_bps, short_pnl_bps, net_pnl_bps, net_pnl_usd,
                  size_usd, duration_seconds
           FROM straddle_log
           WHERE status = 'CLOSED'
           ORDER BY closed_at DESC
           LIMIT ?""",
        (limit,),
    )


@router.get("/hourly")
async def straddle_hourly(request: Request) -> list[dict]:
    """Hourly P&L aggregation for charting."""
    db = request.app.state.dashboard_config.db_path
    return _safe_query(
        db,
        """SELECT strftime('%Y-%m-%d %H:00', closed_at) as hour,
                  COUNT(*) as straddles,
                  SUM(CASE WHEN net_pnl_bps > 0 THEN 1 ELSE 0 END) as winners,
                  ROUND(SUM(net_pnl_usd), 4) as pnl_usd,
                  ROUND(AVG(net_pnl_bps), 1) as avg_pnl_bps,
                  ROUND(SUM(CASE WHEN net_pnl_bps > 0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate
           FROM straddle_log
           WHERE status = 'CLOSED' AND closed_at IS NOT NULL
           GROUP BY hour
           ORDER BY hour DESC
           LIMIT 48""",
    )


@router.get("/stats")
async def straddle_stats(request: Request) -> dict[str, Any]:
    """Aggregate statistics: totals, win rate, avg P&L, exit reason breakdown."""
    db = request.app.state.dashboard_config.db_path

    agg = _safe_query(
        db,
        """SELECT COUNT(*) as total,
                  SUM(CASE WHEN net_pnl_bps > 0 THEN 1 ELSE 0 END) as winners,
                  ROUND(SUM(net_pnl_usd), 4) as total_pnl_usd,
                  ROUND(AVG(net_pnl_bps), 2) as avg_net_bps,
                  ROUND(AVG(duration_seconds), 1) as avg_duration,
                  ROUND(MAX(net_pnl_bps), 2) as best_bps,
                  ROUND(MIN(net_pnl_bps), 2) as worst_bps
           FROM straddle_log WHERE status = 'CLOSED'""",
    )
    a = agg[0] if agg else {}

    # Exit reason breakdown (long + short combined)
    long_reasons = _safe_query(
        db,
        """SELECT long_exit_reason as reason, COUNT(*) as cnt
           FROM straddle_log WHERE status = 'CLOSED'
           GROUP BY long_exit_reason""",
    )
    short_reasons = _safe_query(
        db,
        """SELECT short_exit_reason as reason, COUNT(*) as cnt
           FROM straddle_log WHERE status = 'CLOSED'
           GROUP BY short_exit_reason""",
    )
    reasons: dict[str, int] = {}
    for row in long_reasons + short_reasons:
        r = row.get('reason', 'unknown') or 'unknown'
        reasons[r] = reasons.get(r, 0) + row.get('cnt', 0)

    dead_zone = _safe_query(
        db, "SELECT COUNT(*) as cnt FROM straddle_log WHERE dead_zone_blocked = 1"
    )

    total = a.get('total', 0) or 0
    winners = a.get('winners', 0) or 0
    return {
        'total': total,
        'winners': winners,
        'win_rate': round(winners / total * 100, 1) if total > 0 else 0,
        'total_pnl_usd': a.get('total_pnl_usd', 0),
        'avg_net_bps': a.get('avg_net_bps', 0),
        'avg_duration': a.get('avg_duration', 0),
        'best_bps': a.get('best_bps', 0),
        'worst_bps': a.get('worst_bps', 0),
        'exit_reasons': reasons,
        'dead_zone_blocks': dead_zone[0]['cnt'] if dead_zone else 0,
    }
