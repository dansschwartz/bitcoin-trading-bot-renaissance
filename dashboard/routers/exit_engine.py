"""Exit Engine endpoints — sub-bar scanner, exit performance, hold time analysis."""

import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/exit-engine", tags=["exit-engine"])


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


@router.get("/summary")
async def exit_engine_summary(request: Request) -> dict[str, Any]:
    """Exit engine performance summary — scanner stats + exit breakdown."""
    db = request.app.state.dashboard_config.db_path

    # Sub-bar scanner status (live from bot if available)
    scanner_status: dict = {}
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "sub_bar_scanner"):
        try:
            scanner_status = bot.sub_bar_scanner.get_status()
        except Exception as e:
            logger.warning(f"bot.sub_bar_scanner.get_status failed: {e}")

    # If no live status, build from DB
    if not scanner_status:
        sub_bar_counts = _safe_query(
            db,
            """SELECT
                   COUNT(*) as total_scans,
                   SUM(CASE WHEN action_taken IS NOT NULL THEN 1 ELSE 0 END) as triggers_fired
               FROM sub_bar_events""",
        )
        scanner_status = {
            "enabled": True,
            "running": True,
            "observation_mode": True,
            "scan_interval_seconds": 10,
            "total_scans": (sub_bar_counts[0]["total_scans"] or 0) if sub_bar_counts else 0,
            "triggers_fired": (sub_bar_counts[0]["triggers_fired"] or 0) if sub_bar_counts else 0,
        }

    # Exit performance by reason (from token_spray_log)
    exit_perf = _safe_query(
        db,
        """SELECT
               exit_reason,
               COUNT(*) as count,
               SUM(CASE WHEN exit_pnl_bps > 0 THEN 1 ELSE 0 END) as winners,
               ROUND(AVG(exit_pnl_bps), 2) as avg_pnl_bps,
               ROUND(SUM(exit_pnl_usd), 4) as total_pnl_usd,
               ROUND(AVG(hold_time_seconds), 1) as avg_hold_sec
           FROM token_spray_log
           WHERE exit_reason IS NOT NULL AND observation_mode = 0
           GROUP BY exit_reason
           ORDER BY total_pnl_usd DESC""",
    )
    for r in exit_perf:
        r["win_rate"] = round(r["winners"] / r["count"] * 100, 1) if r["count"] else 0

    # Sub-bar events breakdown (from sub_bar_events table)
    sub_bar_events = _safe_query(
        db,
        """SELECT trigger_type as reason, COUNT(*) as count,
                  ROUND(AVG(pnl_bps), 2) as avg_pnl_bps
           FROM sub_bar_events
           WHERE observation_mode = 0
           GROUP BY trigger_type""",
    )

    # Reeval events breakdown (if table exists)
    reeval_events = _safe_query(
        db,
        """SELECT reason_code as reason, COUNT(*) as count,
                  ROUND(AVG(pnl_bps), 2) as avg_pnl_bps
           FROM reeval_events
           GROUP BY reason_code""",
    )

    return {
        "scanner": scanner_status,
        "exit_performance": exit_perf,
        "sub_bar_events": sub_bar_events,
        "reeval_events": reeval_events,
    }


@router.get("/recent-exits")
async def recent_exits(request: Request) -> list[dict]:
    """Last 50 exits with details."""
    db = request.app.state.dashboard_config.db_path

    # Try token_spray_log first
    rows = _safe_query(
        db,
        """SELECT timestamp, pair, direction, exit_reason, exit_pnl_bps,
                  exit_pnl_usd, hold_time_seconds, token_size_usd
           FROM token_spray_log
           WHERE exit_reason IS NOT NULL
           ORDER BY timestamp DESC
           LIMIT 50""",
    )
    if rows:
        return rows

    # Fallback: use sub_bar_events
    return _safe_query(
        db,
        """SELECT timestamp, product_id as pair, side as direction,
                  trigger_type as exit_reason, pnl_bps as exit_pnl_bps,
                  NULL as exit_pnl_usd, NULL as hold_time_seconds,
                  NULL as token_size_usd
           FROM sub_bar_events
           ORDER BY timestamp DESC
           LIMIT 50""",
    )


@router.get("/hold-time-distribution")
async def hold_time_distribution(request: Request) -> list[dict]:
    """Distribution of hold times for histogram chart."""
    db = request.app.state.dashboard_config.db_path
    return _safe_query(
        db,
        """SELECT
               CASE
                   WHEN hold_time_seconds < 10 THEN '<10s'
                   WHEN hold_time_seconds < 30 THEN '10-30s'
                   WHEN hold_time_seconds < 60 THEN '30-60s'
                   WHEN hold_time_seconds < 120 THEN '1-2m'
                   WHEN hold_time_seconds < 300 THEN '2-5m'
                   ELSE '>5m'
               END as bucket,
               COUNT(*) as count,
               ROUND(AVG(exit_pnl_bps), 2) as avg_pnl_bps,
               SUM(CASE WHEN exit_pnl_bps > 0 THEN 1 ELSE 0 END) as winners
           FROM token_spray_log
           WHERE hold_time_seconds IS NOT NULL AND observation_mode = 0
           GROUP BY bucket
           ORDER BY MIN(hold_time_seconds)""",
    )
