"""Multi-Asset Straddle endpoints — fleet status, per-asset status, history, hourly P&L, stats."""

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


@router.get("/fleet")
async def straddle_fleet(request: Request) -> dict[str, Any]:
    """Fleet-wide status across all straddle engines."""
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "straddle_fleet") and bot.straddle_fleet:
        try:
            return bot.straddle_fleet.get_fleet_status()
        except Exception:
            pass

    # Fallback: derive from database (includes OPEN + CLOSED)
    db = request.app.state.dashboard_config.db_path
    assets = _safe_query(
        db,
        """SELECT COALESCE(asset, 'BTC') as asset,
                  SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_count,
                  SUM(CASE WHEN status = 'OPEN' THEN COALESCE(size_usd, 500) * 2 ELSE 0 END) as deployed,
                  SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed_count,
                  SUM(CASE WHEN status = 'CLOSED' AND net_pnl_bps > 0 THEN 1 ELSE 0 END) as winners,
                  ROUND(COALESCE(SUM(CASE WHEN status = 'CLOSED' THEN net_pnl_usd ELSE 0 END), 0), 4) as pnl
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
            'pnl_usd': round(pnl, 4),
            'daily_loss': 0,
            'win_rate': round(winners / closed_c * 100, 1) if closed_c > 0 else 0,
            'total_closed': closed_c,
        }

    # If no data yet, show both engines as placeholders
    for a in ('BTC', 'ETH'):
        if a not in engines:
            engines[a] = {'open': 0, 'deployed': 0, 'pnl_usd': 0, 'daily_loss': 0, 'win_rate': 0, 'total_closed': 0}

    return {
        'halted': False,
        'fleet_daily_loss': 0,
        'fleet_daily_loss_limit': 1500,
        'fleet_max_deployed': 70000,
        'total_deployed': round(total_deployed, 0),
        'total_open': total_open,
        'total_pnl': round(total_pnl, 4),
        'engines': engines,
    }


@router.get("/status")
async def straddle_status(request: Request, asset: str = "") -> dict[str, Any] | list[dict]:
    """Straddle engine status. If asset specified, returns that engine's status.
    Otherwise returns all engines."""
    bot = getattr(request.app.state, "bot", None)

    if asset:
        # Single asset
        if bot and hasattr(bot, "straddle_engines"):
            eng = bot.straddle_engines.get(asset.upper())
            if eng:
                try:
                    return eng.get_status()
                except Exception:
                    pass
        # Fallback from DB
        return _db_status(request.app.state.dashboard_config.db_path, asset.upper())

    # All engines
    if bot and hasattr(bot, "straddle_engines") and bot.straddle_engines:
        result = []
        for eng in bot.straddle_engines.values():
            try:
                result.append(eng.get_status())
            except Exception:
                pass
        if result:
            return result

    # DB fallback — list all known assets
    db = request.app.state.dashboard_config.db_path
    assets = _safe_query(
        db,
        "SELECT DISTINCT COALESCE(asset, 'BTC') as asset FROM straddle_log",
    )
    if not assets:
        assets = [{'asset': 'BTC'}]
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
        """SELECT id, entry_price, vol_ratio, effective_stop_bps, effective_activation_bps,
                  effective_trail_bps, opened_at, size_usd
           FROM straddle_log WHERE status = 'OPEN' AND COALESCE(asset, 'BTC') = ?
           ORDER BY opened_at DESC""",
        (asset,),
    )
    closed = _safe_query(
        db_path,
        """SELECT COUNT(*) as cnt,
                  COALESCE(SUM(CASE WHEN net_pnl_bps > 0 THEN 1 ELSE 0 END), 0) as winners,
                  COALESCE(SUM(net_pnl_usd), 0) as pnl
           FROM straddle_log WHERE status = 'CLOSED' AND COALESCE(asset, 'BTC') = ?""",
        (asset,),
    )
    c = closed[0] if closed else {'cnt': 0, 'winners': 0, 'pnl': 0}
    wr = (c['winners'] / c['cnt'] * 100) if c['cnt'] > 0 else 0

    import time
    now = time.time()
    open_straddles = []
    for row in open_rows:
        # Parse opened_at to get age
        age = 0.0
        try:
            from datetime import datetime, timezone
            opened = datetime.strptime(row['opened_at'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            age = now - opened.timestamp()
        except Exception:
            pass
        open_straddles.append({
            'straddle_id': row['id'],
            'asset': asset,
            'entry_price': row.get('entry_price', 0),
            'vol_prediction': None,
            'vol_ratio': row.get('vol_ratio', 1.0),
            'effective_stop_bps': row.get('effective_stop_bps', 5.0),
            'effective_activation_bps': row.get('effective_activation_bps', 3.0),
            'effective_trail_bps': row.get('effective_trail_bps', 2.0),
            'age_seconds': round(age, 1),
            'long_trail_active': False,
            'short_trail_active': False,
            'long_peak_bps': 0,
            'short_peak_bps': 0,
        })

    has_activity = len(open_rows) > 0 or (c['cnt'] or 0) > 0

    return {
        'active': has_activity,
        'observation_mode': False,
        'asset': asset,
        'pair': f'{asset}-USD',
        'size_usd': 500,
        'wallet_usd': 35000,
        'interval_seconds': 10,
        'max_open': 35,
        'open_count': len(open_rows),
        'open_straddles': open_straddles,
        'total_opened': total[0]['cnt'] if total else 0,
        'total_closed': c['cnt'],
        'total_winners': c['winners'],
        'win_rate': round(wr, 1),
        'total_pnl_usd': round(c['pnl'], 4),
        'daily_loss_usd': 0,
        'max_daily_loss_usd': 700,
        'dead_zone_blocks': 0,
        'vol_scaling': 'none',
        'config': {},
    }


@router.get("/history")
async def straddle_history(request: Request, limit: int = 100, asset: str = "") -> list[dict]:
    """Recent closed straddles. Optional asset filter."""
    db = request.app.state.dashboard_config.db_path
    if asset:
        return _safe_query(
            db,
            """SELECT id, COALESCE(asset, 'BTC') as asset, opened_at, closed_at, entry_price,
                      vol_prediction, vol_ratio, effective_stop_bps, effective_activation_bps,
                      effective_trail_bps, status,
                      long_exit_price, short_exit_price, long_exit_reason, short_exit_reason,
                      long_pnl_bps, short_pnl_bps, net_pnl_bps, net_pnl_usd,
                      size_usd, duration_seconds
               FROM straddle_log
               WHERE status = 'CLOSED' AND COALESCE(asset, 'BTC') = ?
               ORDER BY closed_at DESC
               LIMIT ?""",
            (asset.upper(), limit),
        )
    return _safe_query(
        db,
        """SELECT id, COALESCE(asset, 'BTC') as asset, opened_at, closed_at, entry_price,
                  vol_prediction, vol_ratio, effective_stop_bps, effective_activation_bps,
                  effective_trail_bps, status,
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
async def straddle_hourly(request: Request, asset: str = "") -> list[dict]:
    """Hourly P&L aggregation for charting. Optional asset filter."""
    db = request.app.state.dashboard_config.db_path
    asset_filter = "AND COALESCE(asset, 'BTC') = ?" if asset else ""
    params = (asset.upper(), 48) if asset else (48,)
    return _safe_query(
        db,
        f"""SELECT strftime('%Y-%m-%d %H:00', closed_at) as hour,
                  COALESCE(asset, 'BTC') as asset,
                  COUNT(*) as straddles,
                  SUM(CASE WHEN net_pnl_bps > 0 THEN 1 ELSE 0 END) as winners,
                  ROUND(SUM(net_pnl_usd), 4) as pnl_usd,
                  ROUND(AVG(net_pnl_bps), 1) as avg_pnl_bps,
                  ROUND(SUM(CASE WHEN net_pnl_bps > 0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate
           FROM straddle_log
           WHERE status = 'CLOSED' AND closed_at IS NOT NULL {asset_filter}
           GROUP BY hour, COALESCE(asset, 'BTC')
           ORDER BY hour DESC
           LIMIT ?""",
        params,
    )


@router.get("/stats")
async def straddle_stats(request: Request, asset: str = "") -> dict[str, Any]:
    """Aggregate statistics: totals, win rate, avg P&L, exit reason breakdown."""
    db = request.app.state.dashboard_config.db_path
    asset_filter = "AND COALESCE(asset, 'BTC') = ?" if asset else ""
    params = (asset.upper(),) if asset else ()

    agg = _safe_query(
        db,
        f"""SELECT COUNT(*) as total,
                  SUM(CASE WHEN net_pnl_bps > 0 THEN 1 ELSE 0 END) as winners,
                  ROUND(SUM(net_pnl_usd), 4) as total_pnl_usd,
                  ROUND(AVG(net_pnl_bps), 2) as avg_net_bps,
                  ROUND(AVG(duration_seconds), 1) as avg_duration,
                  ROUND(MAX(net_pnl_bps), 2) as best_bps,
                  ROUND(MIN(net_pnl_bps), 2) as worst_bps,
                  ROUND(AVG(vol_ratio), 3) as avg_vol_ratio
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

    dead_zone = _safe_query(
        db,
        f"SELECT COUNT(*) as cnt FROM straddle_log WHERE dead_zone_blocked = 1 {asset_filter}",
        params,
    )

    # Per-asset breakdown
    per_asset = _safe_query(
        db,
        """SELECT COALESCE(asset, 'BTC') as asset,
                  COUNT(*) as total,
                  SUM(CASE WHEN net_pnl_bps > 0 THEN 1 ELSE 0 END) as winners,
                  ROUND(SUM(net_pnl_usd), 4) as pnl_usd,
                  ROUND(AVG(net_pnl_bps), 2) as avg_bps
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
        'avg_net_bps': a.get('avg_net_bps', 0),
        'avg_duration': a.get('avg_duration', 0),
        'best_bps': a.get('best_bps', 0),
        'worst_bps': a.get('worst_bps', 0),
        'avg_vol_ratio': a.get('avg_vol_ratio', 1.0),
        'exit_reasons': reasons,
        'dead_zone_blocks': dead_zone[0]['cnt'] if dead_zone else 0,
        'per_asset': per_asset,
    }
