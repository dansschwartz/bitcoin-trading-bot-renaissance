"""Multi-Asset Straddle endpoints — fleet status, per-asset status, history, hourly P&L, stats.

All P&L is in DOLLARS. BPS is secondary/reference only.
"""

import logging
import sqlite3
import time
from datetime import datetime, timezone
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


@router.get("/fleet")
async def straddle_fleet(request: Request) -> dict[str, Any]:
    """Fleet-wide status across all straddle engines."""
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "straddle_fleet") and bot.straddle_fleet:
        try:
            return bot.straddle_fleet.get_fleet_status()
        except Exception as e:
            logger.warning(f"Failed: return bot.straddle_fleet.get_fleet_status(): {e}")

    # Fallback: derive from database (includes OPEN + CLOSED)
    db = request.app.state.dashboard_config.db_path
    assets = _safe_query(
        db,
        """SELECT COALESCE(asset, 'BTC') as asset,
                  SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_count,
                  SUM(CASE WHEN status = 'OPEN' THEN COALESCE(size_usd, 1000) * 2 ELSE 0 END) as deployed,
                  SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed_count,
                  SUM(CASE WHEN status = 'CLOSED' AND net_pnl_usd > 0 THEN 1 ELSE 0 END) as winners,
                  ROUND(COALESCE(SUM(CASE WHEN status = 'CLOSED' THEN net_pnl_usd ELSE 0 END), 0), 2) as pnl
           FROM straddle_log
           WHERE status IN ('OPEN', 'CLOSED')
           GROUP BY COALESCE(asset, 'BTC')""",
    )
    engines: dict[str, Any] = {}
    total_pnl = 0.0
    total_open = 0
    total_deployed = 0.0
    for row in assets:
        a = row.get('asset', 'BTC')
        open_c = row.get('open_count', 0) or 0
        deployed = row.get('deployed', 0) or 0
        closed_c = row.get('closed_count', 0) or 0
        winners = row.get('winners', 0) or 0
        pnl = row.get('pnl', 0) or 0
        total_pnl += pnl
        total_open += open_c
        total_deployed += deployed
        engines[a] = {
            'open': open_c,
            'deployed': round(deployed, 0),
            'pnl_usd': round(pnl, 2),
            'daily_loss': 0,
            'win_rate': round(winners / closed_c * 100, 1) if closed_c > 0 else 0,
            'total_closed': closed_c,
        }

    for a in ('BTC', 'ETH', 'SOL'):
        if a not in engines:
            engines[a] = {'open': 0, 'deployed': 0, 'pnl_usd': 0, 'daily_loss': 0, 'win_rate': 0, 'total_closed': 0}

    return {
        'halted': False,
        'fleet_daily_loss': 0,
        'fleet_daily_loss_limit': 2100,
        'fleet_max_deployed': 21000,
        'total_deployed': round(total_deployed, 0),
        'total_open': total_open,
        'total_pnl': round(total_pnl, 2),
        'engines': engines,
    }


@router.get("/status")
async def straddle_status(request: Request, asset: str = "") -> dict[str, Any] | list[dict]:
    """Straddle engine status."""
    bot = getattr(request.app.state, "bot", None)

    if asset:
        if bot and hasattr(bot, "straddle_engines"):
            eng = bot.straddle_engines.get(asset.upper())
            if eng:
                try:
                    return eng.get_status()
                except Exception as e:
                    logger.warning(f"Failed: return eng.get_status(): {e}")
        return _db_status(request.app.state.dashboard_config.db_path, asset.upper())

    if bot and hasattr(bot, "straddle_engines") and bot.straddle_engines:
        result = []
        for eng in bot.straddle_engines.values():
            try:
                result.append(eng.get_status())
            except Exception as e:
                logger.warning(f"result.append failed: {e}")
        if result:
            return result

    db = request.app.state.dashboard_config.db_path
    assets = _safe_query(
        db,
        "SELECT DISTINCT COALESCE(asset, 'BTC') as asset FROM straddle_log",
    )
    if not assets:
        assets = [{'asset': 'BTC'}, {'asset': 'ETH'}, {'asset': 'SOL'}]
    return [_db_status(db, row['asset']) for row in assets]


def _db_status(db_path: str, asset: str) -> dict:
    """Build status dict from DB for a given asset."""
    total = _safe_query(
        db_path,
        "SELECT COUNT(*) as cnt FROM straddle_log WHERE status != 'BLOCKED' AND COALESCE(asset, 'BTC') = ?",
        (asset,),
    )
    open_rows = _safe_query(
        db_path,
        """SELECT id, entry_price, size_usd, stop_loss_usd, trail_activation_usd,
                  trail_distance_usd, opened_at, vol_ratio,
                  effective_stop_bps, effective_activation_bps, effective_trail_bps,
                  long_pnl_usd, short_pnl_usd, long_peak_usd, short_peak_usd,
                  long_pnl_bps, short_pnl_bps
           FROM straddle_log WHERE status = 'OPEN' AND COALESCE(asset, 'BTC') = ?
           ORDER BY opened_at DESC""",
        (asset,),
    )
    closed = _safe_query(
        db_path,
        """SELECT COUNT(*) as cnt,
                  COALESCE(SUM(CASE WHEN net_pnl_usd > 0 THEN 1 ELSE 0 END), 0) as winners,
                  COALESCE(SUM(net_pnl_usd), 0) as pnl
           FROM straddle_log WHERE status = 'CLOSED' AND COALESCE(asset, 'BTC') = ?""",
        (asset,),
    )
    c = closed[0] if closed else {'cnt': 0, 'winners': 0, 'pnl': 0}
    wr = (c['winners'] / c['cnt'] * 100) if c['cnt'] > 0 else 0

    now = time.time()
    open_straddles = []
    for row in open_rows:
        age = 0.0
        try:
            opened = datetime.strptime(row['opened_at'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            age = now - opened.timestamp()
        except Exception as e:
            logger.warning(f"datetime.strptime failed: {e}")
        open_straddles.append({
            'straddle_id': row['id'],
            'asset': asset,
            'entry_price': row.get('entry_price', 0),
            'size_usd': row.get('size_usd', 100),
            'stop_loss_usd': row.get('stop_loss_usd', 0.04),
            'trail_activation_usd': row.get('trail_activation_usd', 0.02),
            'trail_distance_usd': row.get('trail_distance_usd', 0.01),
            'stop_loss_bps': row.get('effective_stop_bps') or 4.0,
            'trail_activation_bps': row.get('effective_activation_bps') or 2.0,
            'trail_distance_bps': row.get('effective_trail_bps') or 1.0,
            'vol_ratio': row.get('vol_ratio') or 1.0,
            'age_seconds': round(age, 1),
            'long_pnl_usd': row.get('long_pnl_usd') or 0,
            'short_pnl_usd': row.get('short_pnl_usd') or 0,
            'long_pnl_bps': row.get('long_pnl_bps') or 0,
            'short_pnl_bps': row.get('short_pnl_bps') or 0,
            'long_peak_usd': row.get('long_peak_usd') or 0,
            'short_peak_usd': row.get('short_peak_usd') or 0,
            'long_trail_active': False, 'short_trail_active': False,
        })

    has_activity = len(open_rows) > 0 or (c['cnt'] or 0) > 0

    return {
        'active': has_activity,
        'observation_mode': False,
        'asset': asset,
        'pair': f'{asset}-USD',
        'size_usd': 100,
        'wallet_usd': 7000,
        'interval_seconds': 10,
        'max_open': 35,
        'open_count': len(open_rows),
        'open_straddles': open_straddles,
        'total_opened': total[0]['cnt'] if total else 0,
        'total_closed': c['cnt'],
        'total_winners': c['winners'],
        'win_rate': round(wr, 1),
        'total_pnl_usd': round(c['pnl'], 2),
        'daily_loss_usd': 0,
        'max_daily_loss_usd': 700,
        'config': {
            'stop_loss_bps': 4,
            'trail_activation_bps': 2,
            'trail_distance_bps': 1,
            'max_hold_seconds': 120,
            'vol_scaling': 'proportional',
            'check_interval': 1,
        },
    }


@router.get("/history")
async def straddle_history(request: Request, limit: int = 100, asset: str = "") -> list[dict]:
    """Recent closed straddles. Dollar P&L primary."""
    db = request.app.state.dashboard_config.db_path
    if asset:
        return _safe_query(
            db,
            """SELECT id, COALESCE(asset, 'BTC') as asset, opened_at, closed_at, entry_price,
                      size_usd, stop_loss_usd, trail_activation_usd, trail_distance_usd,
                      status, long_exit_price, short_exit_price,
                      long_exit_reason, short_exit_reason,
                      long_pnl_usd, short_pnl_usd, long_peak_usd, short_peak_usd,
                      long_pnl_bps, short_pnl_bps, net_pnl_usd, net_pnl_bps,
                      duration_seconds
               FROM straddle_log
               WHERE status = 'CLOSED' AND COALESCE(asset, 'BTC') = ?
               ORDER BY closed_at DESC
               LIMIT ?""",
            (asset.upper(), limit),
        )
    return _safe_query(
        db,
        """SELECT id, COALESCE(asset, 'BTC') as asset, opened_at, closed_at, entry_price,
                  size_usd, stop_loss_usd, trail_activation_usd, trail_distance_usd,
                  status, long_exit_price, short_exit_price,
                  long_exit_reason, short_exit_reason,
                  long_pnl_usd, short_pnl_usd, long_peak_usd, short_peak_usd,
                  long_pnl_bps, short_pnl_bps, net_pnl_usd, net_pnl_bps,
                  duration_seconds
           FROM straddle_log
           WHERE status = 'CLOSED'
           ORDER BY closed_at DESC
           LIMIT ?""",
        (limit,),
    )


@router.get("/hourly")
async def straddle_hourly(request: Request, asset: str = "") -> list[dict]:
    """Hourly P&L aggregation. Dollar amounts."""
    db = request.app.state.dashboard_config.db_path
    asset_filter = "AND COALESCE(asset, 'BTC') = ?" if asset else ""
    params = (asset.upper(), 48) if asset else (48,)
    return _safe_query(
        db,
        f"""SELECT strftime('%Y-%m-%d %H:00', closed_at) as hour,
                  COALESCE(asset, 'BTC') as asset,
                  COUNT(*) as straddles,
                  SUM(CASE WHEN net_pnl_usd > 0 THEN 1 ELSE 0 END) as winners,
                  ROUND(SUM(net_pnl_usd), 2) as pnl_usd,
                  ROUND(AVG(net_pnl_usd), 2) as avg_pnl_usd,
                  ROUND(SUM(CASE WHEN net_pnl_usd > 0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate
           FROM straddle_log
           WHERE status = 'CLOSED' AND closed_at IS NOT NULL {asset_filter}
           GROUP BY hour, COALESCE(asset, 'BTC')
           ORDER BY hour DESC
           LIMIT ?""",
        params,
    )


@router.get("/daily")
async def straddle_daily(request: Request, days: int = 30, asset: str = "") -> list[dict]:
    """Daily P&L aggregation. Dollar amounts."""
    db = request.app.state.dashboard_config.db_path
    asset_filter = "AND COALESCE(asset, 'BTC') = ?" if asset else ""
    params = (asset.upper(), days) if asset else (days,)
    return _safe_query(
        db,
        f"""SELECT DATE(closed_at) as day,
                  COUNT(*) as straddles,
                  SUM(CASE WHEN net_pnl_usd > 0 THEN 1 ELSE 0 END) as winners,
                  ROUND(SUM(net_pnl_usd), 2) as pnl_usd,
                  ROUND(AVG(net_pnl_usd), 4) as avg_pnl_usd,
                  ROUND(SUM(CASE WHEN net_pnl_usd > 0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate
           FROM straddle_log
           WHERE status = 'CLOSED' AND closed_at IS NOT NULL {asset_filter}
           GROUP BY day
           ORDER BY day DESC
           LIMIT ?""",
        params,
    )


@router.get("/stats")
async def straddle_stats(request: Request, asset: str = "") -> dict[str, Any]:
    """Aggregate statistics — all dollar-denominated."""
    db = request.app.state.dashboard_config.db_path
    asset_filter = "AND COALESCE(asset, 'BTC') = ?" if asset else ""
    params = (asset.upper(),) if asset else ()

    agg = _safe_query(
        db,
        f"""SELECT COUNT(*) as total,
                  SUM(CASE WHEN net_pnl_usd > 0 THEN 1 ELSE 0 END) as winners,
                  ROUND(SUM(net_pnl_usd), 2) as total_pnl_usd,
                  ROUND(AVG(net_pnl_usd), 2) as avg_net_usd,
                  ROUND(AVG(CASE WHEN net_pnl_usd > 0 THEN net_pnl_usd END), 2) as avg_winner_usd,
                  ROUND(AVG(CASE WHEN net_pnl_usd < 0 THEN net_pnl_usd END), 2) as avg_loser_usd,
                  ROUND(AVG(duration_seconds), 1) as avg_duration,
                  ROUND(MAX(net_pnl_usd), 2) as best_usd,
                  ROUND(MIN(net_pnl_usd), 2) as worst_usd
           FROM straddle_log WHERE status = 'CLOSED' {asset_filter}""",
        params,
    )
    a = agg[0] if agg else {}

    # Exit reason breakdown
    long_reasons = _safe_query(
        db,
        f"""SELECT long_exit_reason as reason, COUNT(*) as cnt
           FROM straddle_log WHERE status = 'CLOSED' {asset_filter}
           GROUP BY long_exit_reason""",
        params,
    )
    short_reasons = _safe_query(
        db,
        f"""SELECT short_exit_reason as reason, COUNT(*) as cnt
           FROM straddle_log WHERE status = 'CLOSED' {asset_filter}
           GROUP BY short_exit_reason""",
        params,
    )
    reasons: dict[str, int] = {}
    for row in long_reasons + short_reasons:
        r = row.get('reason', 'unknown') or 'unknown'
        reasons[r] = reasons.get(r, 0) + row.get('cnt', 0)

    # Per-asset breakdown
    per_asset = _safe_query(
        db,
        """SELECT COALESCE(asset, 'BTC') as asset,
                  COUNT(*) as total,
                  SUM(CASE WHEN net_pnl_usd > 0 THEN 1 ELSE 0 END) as winners,
                  ROUND(SUM(net_pnl_usd), 2) as pnl_usd,
                  ROUND(AVG(net_pnl_usd), 2) as avg_usd
           FROM straddle_log WHERE status = 'CLOSED'
           GROUP BY COALESCE(asset, 'BTC')""",
    )

    total = a.get('total', 0) or 0
    winners = a.get('winners', 0) or 0
    return {
        'total': total,
        'winners': winners,
        'win_rate': round(winners / total * 100, 1) if total > 0 else 0,
        'total_pnl_usd': a.get('total_pnl_usd', 0),
        'avg_net_usd': a.get('avg_net_usd', 0),
        'avg_winner_usd': a.get('avg_winner_usd', 0),
        'avg_loser_usd': a.get('avg_loser_usd', 0),
        'avg_duration': a.get('avg_duration', 0),
        'best_usd': a.get('best_usd', 0),
        'worst_usd': a.get('worst_usd', 0),
        'exit_reasons': reasons,
        'per_asset': per_asset,
    }
