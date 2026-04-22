"""Token Spray endpoints — spray status, live feed, hourly P&L, pair economics, A/B test."""

import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/token-spray", tags=["token-spray"])


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
async def spray_status(request: Request) -> dict[str, Any]:
    """Token spray engine status — open tokens, budget, mode."""
    db = request.app.state.dashboard_config.db_path

    # Try to get live status from bot if available
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "token_spray"):
        try:
            return bot.token_spray.get_status()
        except Exception as e:
            logger.warning(f"Failed: return bot.token_spray.get_status(): {e}")

    # Fallback: derive from database
    total_sprayed = _safe_query(db, "SELECT COUNT(*) as cnt FROM token_spray_log")
    open_tokens = _safe_query(
        db, "SELECT COUNT(*) as cnt FROM token_spray_log WHERE exit_reason IS NULL"
    )
    today_pnl = _safe_query(
        db,
        """SELECT COALESCE(SUM(exit_pnl_usd), 0) as pnl, COUNT(*) as cnt
           FROM token_spray_log
           WHERE exit_pnl_usd IS NOT NULL AND observation_mode = 0
             AND date(timestamp) = date('now')""",
    )

    return {
        "active": True,
        "observation_mode": False,
        "spray_interval_seconds": 5,
        "total_sprayed": total_sprayed[0]["cnt"] if total_sprayed else 0,
        "total_open": open_tokens[0]["cnt"] if open_tokens else 0,
        "max_open": 200,
        "today_pnl_usd": today_pnl[0]["pnl"] if today_pnl else 0,
        "today_tokens": today_pnl[0]["cnt"] if today_pnl else 0,
        "budget": {
            "deployed_usd": (open_tokens[0]["cnt"] if open_tokens else 0) * 10,
            "total_capital": 5000,
            "deployed_pct": min(
                (open_tokens[0]["cnt"] if open_tokens else 0) * 10 / 4000 * 100, 100
            ),
        },
    }


@router.get("/wallets")
async def wallet_stats(request: Request) -> list[dict]:
    """Per-wallet (per-model) P&L and win rate breakdown."""
    db = request.app.state.dashboard_config.db_path

    # Try live status from bot
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "token_spray") and getattr(bot.token_spray, "wallets", None):
        try:
            wallets = bot.token_spray.wallets
            result = []
            for wid, w in wallets.items():
                wr = (w.total_winners / w.total_closed * 100) if w.total_closed > 0 else 0.0
                result.append({
                    "wallet_id": wid,
                    "budget_usd": w.budget_usd,
                    "deployed_usd": round(w.deployed_usd, 2),
                    "total_sprayed": w.total_sprayed,
                    "total_closed": w.total_closed,
                    "total_pnl_usd": round(w.total_pnl_usd, 4),
                    "win_rate": round(wr, 1),
                })
            result.sort(key=lambda x: x["total_pnl_usd"], reverse=True)
            return result
        except Exception as e:
            logger.warning(f"Failed: wallets = bot.token_spray.wallets: {e}")

    # Fallback: DB aggregate
    return _safe_query(
        db,
        """SELECT wallet_id,
                  COUNT(*) as total_tokens,
                  SUM(CASE WHEN exit_pnl_usd IS NOT NULL THEN exit_pnl_usd ELSE 0 END) as total_pnl_usd,
                  SUM(CASE WHEN exit_pnl_bps > 0 THEN 1 ELSE 0 END) as winners,
                  SUM(CASE WHEN exit_pnl_bps IS NOT NULL THEN 1 ELSE 0 END) as closed,
                  ROUND(AVG(exit_pnl_bps), 2) as avg_pnl_bps
           FROM token_spray_log
           WHERE wallet_id != 'default' AND observation_mode = 0
           GROUP BY wallet_id
           ORDER BY total_pnl_usd DESC""",
    )


@router.get("/live-feed")
async def live_feed(request: Request) -> list[dict]:
    """Last 50 token events (sprayed + closed) for the live feed table."""
    db = request.app.state.dashboard_config.db_path
    return _safe_query(
        db,
        """SELECT timestamp, pair, direction, token_size_usd, entry_price,
                  direction_rule, vol_regime, exit_pnl_bps, exit_pnl_usd,
                  exit_reason, hold_time_seconds, observation_mode, wallet_id
           FROM token_spray_log
           ORDER BY timestamp DESC
           LIMIT 50""",
    )


@router.get("/hourly-pnl")
async def hourly_pnl(request: Request) -> list[dict]:
    """Hourly P&L for the spray chart."""
    db = request.app.state.dashboard_config.db_path
    rows = _safe_query(
        db,
        """SELECT
               strftime('%Y-%m-%d %H:00', timestamp) as hour,
               COUNT(*) as tokens,
               SUM(CASE WHEN exit_pnl_bps > 0 THEN 1 ELSE 0 END) as winners,
               ROUND(SUM(exit_pnl_usd), 4) as pnl_usd,
               ROUND(AVG(exit_pnl_bps), 2) as avg_pnl_bps
           FROM token_spray_log
           WHERE exit_pnl_usd IS NOT NULL AND observation_mode = 0
           GROUP BY hour
           ORDER BY hour DESC
           LIMIT 48""",
    )
    for r in rows:
        r["win_rate"] = round(r["winners"] / r["tokens"] * 100, 1) if r["tokens"] else 0
    return rows


@router.get("/pair-economics")
async def pair_economics(request: Request) -> list[dict]:
    """Per-pair EV analysis with live performance data."""
    db = request.app.state.dashboard_config.db_path
    rows = _safe_query(
        db,
        """SELECT
               pair, direction_rule,
               COUNT(*) as total_tokens,
               SUM(CASE WHEN exit_pnl_bps > 0 THEN 1 ELSE 0 END) as winners,
               ROUND(AVG(exit_pnl_bps), 2) as avg_pnl_bps,
               ROUND(SUM(exit_pnl_usd), 4) as total_pnl_usd,
               ROUND(AVG(CASE WHEN exit_pnl_bps > 0 THEN exit_pnl_bps END), 2) as avg_win_bps,
               ROUND(AVG(CASE WHEN exit_pnl_bps <= 0 THEN exit_pnl_bps END), 2) as avg_loss_bps,
               ROUND(AVG(hold_time_seconds), 1) as avg_hold_sec
           FROM token_spray_log
           WHERE exit_pnl_bps IS NOT NULL AND observation_mode = 0
           GROUP BY pair, direction_rule
           ORDER BY total_pnl_usd DESC""",
    )
    for r in rows:
        r["win_rate"] = (
            round(r["winners"] / r["total_tokens"] * 100, 1) if r["total_tokens"] else 0
        )
        r["current_ev"] = r.get("avg_pnl_bps") or 0
        r["current_spread"] = 0
        r["tradeable"] = True
    return rows


@router.get("/ab-test")
async def ab_test(request: Request) -> dict[str, Any]:
    """A/B test results by direction rule."""
    db = request.app.state.dashboard_config.db_path

    # Try live stats from bot
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "token_spray") and hasattr(bot.token_spray, "direction_rule"):
        try:
            return {"ab_results": bot.token_spray.direction_rule.rule_stats}
        except Exception as e:
            logger.warning(f"Failed to get token spray AB test results: {e}")

    # Fallback: derive from DB
    rows = _safe_query(
        db,
        """SELECT
               direction_rule as rule,
               pair,
               COUNT(*) as trades,
               SUM(CASE WHEN exit_pnl_bps > 0 THEN 1 ELSE 0 END) as winners,
               ROUND(SUM(exit_pnl_bps), 2) as total_pnl_bps,
               ROUND(AVG(exit_pnl_bps), 2) as avg_pnl_bps
           FROM token_spray_log
           WHERE exit_pnl_bps IS NOT NULL AND observation_mode = 0
           GROUP BY direction_rule, pair""",
    )
    results = {}
    for r in rows:
        key = f"{r['pair']}_{r['rule']}"
        results[key] = {
            "rule": r["rule"],
            "pair": r["pair"],
            "trades": r["trades"],
            "win_rate": r["winners"] / r["trades"] if r["trades"] else 0,
            "total_pnl_bps": r["total_pnl_bps"],
            "avg_pnl_bps": r["avg_pnl_bps"],
        }
    return {"ab_results": results}


@router.get("/volatility")
async def volatility_current(request: Request) -> dict[str, Any]:
    """Current volatility regime per pair."""
    # Try live from bot's GARCH engine
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "garch_engine"):
        try:
            engine = bot.garch_engine
            result = {}
            for pid in getattr(bot, "product_ids", []):
                forecast = engine.forecast_volatility(pid)
                if forecast:
                    result[pid] = {
                        "regime": forecast.get("vol_regime", "unknown"),
                        "predicted_bps": round(forecast.get("forecast_vol", 0) * 10000, 1),
                        "tradeable": True,
                    }
            return result
        except Exception as e:
            logger.warning(f"Failed: engine = bot.garch_engine: {e}")

    # Fallback: check DB for recent volatility regimes from spray log
    db = request.app.state.dashboard_config.db_path
    rows = _safe_query(
        db,
        """SELECT pair, vol_regime, expected_move_bps
           FROM (
               SELECT pair, vol_regime, expected_move_bps,
                      ROW_NUMBER() OVER (PARTITION BY pair ORDER BY timestamp DESC) as rn
               FROM token_spray_log
               WHERE vol_regime IS NOT NULL
           ) WHERE rn = 1""",
    )
    return {
        r["pair"]: {
            "regime": r["vol_regime"],
            "predicted_bps": round(r["expected_move_bps"] or 0, 1),
            "tradeable": True,
        }
        for r in rows
    } if rows else {}
