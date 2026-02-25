"""Arbitrage dashboard endpoints — reads from data/arbitrage.db and live orchestrator."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/arbitrage", tags=["arbitrage"])

ARB_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "arbitrage.db"


@contextmanager
def _arb_conn():
    """Connect to arbitrage.db (separate from main bot DB)."""
    conn = sqlite3.connect(str(ARB_DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
    finally:
        conn.close()


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


@router.get("/status")
async def arb_status(request: Request):
    """Live arbitrage engine status from orchestrator, with DB fallback."""
    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch and hasattr(orch, "get_full_status"):
        try:
            status = orch.get_full_status()
            # Convert Decimal objects to float for JSON serialization
            return _sanitize_for_json(status)
        except Exception as e:
            logger.debug(f"Live status error: {e}")

    # Fallback: summary from DB — check if bot is running by looking for recent arb activity
    try:
        with _arb_conn() as c:
            total = c.execute("SELECT COUNT(*) as cnt FROM arb_trades").fetchone()["cnt"]
            filled = c.execute(
                "SELECT COUNT(*) as cnt FROM arb_trades WHERE status = 'filled'"
            ).fetchone()["cnt"]
            profit = c.execute(
                "SELECT COALESCE(SUM(actual_profit_usd), 0) as p FROM arb_trades WHERE status = 'filled'"
            ).fetchone()["p"]

            # Check if the bot process is running by looking for the main DB heartbeat
            bot_running = False
            try:
                main_db = ARB_DB_PATH.parent / "renaissance_bot.db"
                if main_db.exists():
                    mc = sqlite3.connect(f"file:{main_db}?mode=ro", uri=True, timeout=5.0)
                    mc.row_factory = sqlite3.Row
                    row = mc.execute(
                        "SELECT MAX(timestamp) as ts FROM decisions"
                    ).fetchone()
                    mc.close()
                    if row and row["ts"]:
                        from datetime import datetime, timedelta, timezone
                        last_decision = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
                        # Bot is "running" if it made a decision in the last 10 minutes
                        if datetime.now(timezone.utc) - last_decision < timedelta(minutes=10):
                            bot_running = True
            except Exception:
                pass

            return {
                "running": bot_running,
                "uptime_seconds": 0,
                "db_summary": {
                    "total_trades": total,
                    "filled_trades": filled,
                    "total_profit_usd": round(float(profit), 4),
                },
            }
    except Exception:
        return {"running": False, "error": "arbitrage.db not found"}


@router.get("/trades")
async def arb_trades(
    request: Request, limit: int = 50, offset: int = 0, strategy: Optional[str] = None
):
    """Recent arbitrage trades from arb_trades table."""
    try:
        with _arb_conn() as c:
            if strategy:
                rows = c.execute(
                    """SELECT * FROM arb_trades
                       WHERE strategy = ?
                       ORDER BY id DESC LIMIT ? OFFSET ?""",
                    (strategy, min(limit, 500), offset),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM arb_trades ORDER BY id DESC LIMIT ? OFFSET ?",
                    (min(limit, 500), offset),
                ).fetchall()
            return _rows_to_dicts(rows)
    except Exception as e:
        return {"error": str(e)}


@router.get("/signals")
async def arb_signals(
    request: Request, limit: int = 100, strategy: Optional[str] = None
):
    """Recent arbitrage signals."""
    try:
        with _arb_conn() as c:
            if strategy:
                rows = c.execute(
                    """SELECT * FROM arb_signals
                       WHERE strategy = ?
                       ORDER BY id DESC LIMIT ?""",
                    (strategy, min(limit, 500)),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM arb_signals ORDER BY id DESC LIMIT ?",
                    (min(limit, 500),),
                ).fetchall()
            return _rows_to_dicts(rows)
    except Exception as e:
        return {"error": str(e)}


@router.get("/summary")
async def arb_summary(request: Request):
    """Aggregate stats from arb_trades: total, P&L by strategy, win rate."""
    try:
        with _arb_conn() as c:
            total = c.execute("SELECT COUNT(*) as cnt FROM arb_trades").fetchone()["cnt"]
            filled = c.execute(
                "SELECT COUNT(*) as cnt FROM arb_trades WHERE status = 'filled'"
            ).fetchone()["cnt"]

            # P&L
            profit_row = c.execute(
                """SELECT
                     COALESCE(SUM(actual_profit_usd), 0) as total_profit,
                     COALESCE(SUM(CASE WHEN actual_profit_usd > 0 THEN 1 ELSE 0 END), 0) as wins,
                     COALESCE(SUM(CASE WHEN actual_profit_usd <= 0 THEN 1 ELSE 0 END), 0) as losses
                   FROM arb_trades WHERE status = 'filled'"""
            ).fetchone()
            wins = profit_row["wins"] or 0
            losses = profit_row["losses"] or 0

            # By strategy
            strategy_rows = c.execute(
                """SELECT strategy,
                          COUNT(*) as trades,
                          COALESCE(SUM(actual_profit_usd), 0) as profit,
                          SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as fills
                   FROM arb_trades GROUP BY strategy"""
            ).fetchall()

            # Signals
            signal_total = 0
            signal_approved = 0
            try:
                sr = c.execute(
                    """SELECT COUNT(*) as total,
                              SUM(CASE WHEN approved = 1 THEN 1 ELSE 0 END) as approved
                       FROM arb_signals"""
                ).fetchone()
                signal_total = sr["total"] or 0
                signal_approved = sr["approved"] or 0
            except Exception:
                pass

            # Today's P&L
            daily_row = c.execute(
                """SELECT COALESCE(SUM(actual_profit_usd), 0) as daily
                   FROM arb_trades
                   WHERE date(timestamp) = date('now') AND status = 'filled'"""
            ).fetchone()

            return {
                "total_trades": total,
                "filled_trades": filled,
                "total_profit_usd": round(float(profit_row["total_profit"]), 4),
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.0,
                "signals_total": signal_total,
                "signals_approved": signal_approved,
                "daily_pnl_usd": round(float(daily_row["daily"]), 4),
                "by_strategy": [
                    {
                        "strategy": r["strategy"],
                        "trades": r["trades"],
                        "fills": r["fills"],
                        "profit_usd": round(float(r["profit"]), 4),
                    }
                    for r in strategy_rows
                ],
            }
    except Exception as e:
        return {"error": str(e)}


@router.get("/wallet")
async def arb_wallet(request: Request):
    """Arbitrage wallet balance and allocation."""
    orch = getattr(request.app.state, "arb_orchestrator", None)
    initial_balance = 10000.0  # Arbitrage wallet starting balance

    # Calculate realized P&L from arb trades
    total_profit = 0.0
    try:
        with _arb_conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(actual_profit_usd), 0) as p FROM arb_trades WHERE status = 'filled'"
            ).fetchone()
            total_profit = float(row["p"])
    except Exception:
        pass

    current_balance = initial_balance + total_profit

    return {
        "initial_balance": initial_balance,
        "current_balance": round(current_balance, 2),
        "total_realized_pnl": round(total_profit, 4),
        "return_pct": round((total_profit / initial_balance) * 100, 4) if initial_balance > 0 else 0.0,
    }


@router.get("/funding")
async def arb_funding(request: Request):
    """Current funding rates and positions from the funding arb module."""
    orch = getattr(request.app.state, "arb_orchestrator", None)

    # Try live data from orchestrator
    if orch and hasattr(orch, 'funding_arb'):
        try:
            stats = orch.funding_arb.get_stats()
            return _sanitize_for_json(stats)
        except Exception as e:
            logger.debug(f"Live funding stats error: {e}")

    # Fallback: read from DB
    try:
        with _arb_conn() as c:
            # Check if tables exist
            tables = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]

            result = {
                "observation_mode": True,
                "current_rates": {},
                "open_positions": [],
                "total_funding_collected": 0.0,
            }

            if "funding_positions" in tables:
                open_pos = c.execute(
                    "SELECT * FROM funding_positions WHERE status='open'"
                ).fetchall()
                result["open_positions"] = _rows_to_dicts(open_pos)

                collected = c.execute(
                    "SELECT COALESCE(SUM(total_funding_collected), 0) FROM funding_positions"
                ).fetchone()[0]
                result["total_funding_collected"] = round(float(collected), 4)

            if "funding_rate_history" in tables:
                # Latest rate for each symbol
                latest = c.execute("""
                    SELECT symbol, funding_rate, annualized_pct, signal, timestamp
                    FROM funding_rate_history
                    WHERE id IN (
                        SELECT MAX(id) FROM funding_rate_history GROUP BY symbol
                    )
                """).fetchall()
                for r in latest:
                    name = r["symbol"].split("/")[0] if "/" in r["symbol"] else r["symbol"]
                    result["current_rates"][name] = {
                        "rate": r["funding_rate"],
                        "annualized": f"{r['annualized_pct']:.1f}%",
                        "signal": r["signal"],
                        "last_update": r["timestamp"],
                    }

            return result
    except Exception as e:
        return {"error": str(e)}


@router.get("/depth-analysis")
async def arb_depth_analysis(request: Request):
    """Analyze book depth data from triangular trades for sizing decisions."""
    try:
        with _arb_conn() as c:
            # Check if leg depth columns exist
            cols = [r[1] for r in c.execute("PRAGMA table_info(arb_trades)").fetchall()]
            if "leg1_depth_usd" not in cols:
                return {"error": "leg depth columns not yet available — deploy latest code"}

            total_with_depth = c.execute(
                "SELECT COUNT(*) FROM arb_trades WHERE strategy='triangular' "
                "AND leg1_depth_usd > 0"
            ).fetchone()[0]

            if total_with_depth == 0:
                return {
                    "avg_leg1_depth": 0.0, "avg_leg2_depth": 0.0, "avg_leg3_depth": 0.0,
                    "min_observed_depth": 0.0,
                    "trades_where_size_exceeded_25pct_depth": 0,
                    "total_trades_with_depth_data": 0,
                    "recommended_max_trade_usd": 2000,
                }

            row = c.execute("""
                SELECT AVG(leg1_depth_usd) as avg1,
                       AVG(leg2_depth_usd) as avg2,
                       AVG(leg3_depth_usd) as avg3,
                       MIN(CASE WHEN leg2_depth_usd > 0 THEN leg2_depth_usd END) as min2,
                       MIN(CASE WHEN leg3_depth_usd > 0 THEN leg3_depth_usd END) as min3
                FROM arb_trades
                WHERE strategy='triangular' AND leg1_depth_usd > 0
            """).fetchone()

            # Leg1 (USDC/USDT) is always huge, so min depth = min(leg2, leg3)
            min2 = row["min2"] or 0
            min3 = row["min3"] or 0
            min_depth = min(min2, min3) if min2 > 0 and min3 > 0 else max(min2, min3)

            # Count trades where size > 25% of thinnest leg
            oversized = c.execute("""
                SELECT COUNT(*) FROM arb_trades
                WHERE strategy='triangular' AND leg1_depth_usd > 0
                AND quantity > 0.25 * MIN(
                    CASE WHEN leg2_depth_usd > 0 THEN leg2_depth_usd ELSE 999999 END,
                    CASE WHEN leg3_depth_usd > 0 THEN leg3_depth_usd ELSE 999999 END
                )
            """).fetchone()[0]

            recommended = min(2000, 0.25 * min_depth) if min_depth > 0 else 2000

            return {
                "avg_leg1_depth": round(float(row["avg1"] or 0), 2),
                "avg_leg2_depth": round(float(row["avg2"] or 0), 2),
                "avg_leg3_depth": round(float(row["avg3"] or 0), 2),
                "min_observed_depth": round(float(min_depth), 2),
                "trades_where_size_exceeded_25pct_depth": oversized,
                "total_trades_with_depth_data": total_with_depth,
                "recommended_max_trade_usd": round(recommended, 2),
            }
    except Exception as e:
        return {"error": str(e)}


@router.get("/sizing-distribution")
async def arb_sizing_distribution(request: Request):
    """Analyze dynamic sizing tier distribution and depth utilization."""
    try:
        with _arb_conn() as c:
            cols = [r[1] for r in c.execute("PRAGMA table_info(arb_trades)").fetchall()]
            if "trade_size_usd" not in cols or "leg_count" not in cols:
                return {"error": "trade_size_usd/leg_count columns not yet available — deploy latest code"}

            # Tier breakdown (based on trade_size_usd)
            tiers = {}
            for tier_name, lo, hi in [("FULL", 1500, 999999), ("MEDIUM", 500, 1500),
                                       ("SMALL", 50, 500)]:
                row = c.execute(
                    "SELECT COUNT(*), AVG(actual_profit_usd), AVG(trade_size_usd) "
                    "FROM arb_trades WHERE strategy='triangular' AND status='filled' "
                    "AND trade_size_usd >= ? AND trade_size_usd < ?",
                    (lo, hi)
                ).fetchone()
                tiers[tier_name] = {
                    "count": row[0] or 0,
                    "avg_profit": round(row[1] or 0, 4),
                    "avg_size": round(row[2] or 0, 0),
                }

            # Skip count (trade_size_usd is NULL or 0 for skipped)
            skip_count = c.execute(
                "SELECT COUNT(*) FROM arb_trades WHERE strategy='triangular' "
                "AND status='skipped'"
            ).fetchone()[0]
            tiers["SKIP"] = {"count": skip_count, "reason": "depth_too_thin"}

            total = c.execute(
                "SELECT COUNT(*) FROM arb_trades WHERE strategy='triangular' "
                "AND trade_size_usd > 0"
            ).fetchone()[0]

            # Depth utilization: how many trades exceed 25% of min leg depth
            exceeding = c.execute("""
                SELECT COUNT(*) FROM arb_trades
                WHERE strategy='triangular' AND trade_size_usd > 0
                AND leg2_depth_usd > 0
                AND trade_size_usd > 0.25 * MIN(
                    CASE WHEN leg2_depth_usd > 0 THEN leg2_depth_usd ELSE 999999 END,
                    CASE WHEN leg3_depth_usd > 0 THEN leg3_depth_usd ELSE 999999 END
                )
            """).fetchone()[0]

            avg_util = c.execute("""
                SELECT AVG(
                    CASE WHEN leg2_depth_usd > 0
                    THEN trade_size_usd / MIN(
                        CASE WHEN leg2_depth_usd > 0 THEN leg2_depth_usd ELSE 999999 END,
                        CASE WHEN leg3_depth_usd > 0 THEN leg3_depth_usd ELSE 999999 END
                    ) * 100
                    ELSE NULL END
                ) FROM arb_trades
                WHERE strategy='triangular' AND trade_size_usd > 0 AND leg2_depth_usd > 0
            """).fetchone()[0]

            # 3-leg vs 4-leg breakdown
            leg_breakdown = {}
            for lc in (3, 4):
                row = c.execute(
                    "SELECT COUNT(*), SUM(actual_profit_usd), AVG(actual_profit_usd) "
                    "FROM arb_trades WHERE strategy='triangular' AND status='filled' "
                    "AND leg_count=?", (lc,)
                ).fetchone()
                detected = c.execute(
                    "SELECT COUNT(*) FROM arb_trades WHERE strategy='triangular' AND leg_count=?",
                    (lc,)
                ).fetchone()[0]
                leg_breakdown[f"{lc}_leg"] = {
                    "detected": detected,
                    "filled": row[0] or 0,
                    "fill_rate": f"{(row[0] or 0) / max(1, detected) * 100:.1f}%",
                    "total_profit": round(row[1] or 0, 2),
                    "avg_per_fill": round(row[2] or 0, 2),
                }

            return {
                "total_trades": total,
                "by_tier": tiers,
                "avg_depth_utilization_pct": round(avg_util or 0, 1),
                "trades_exceeding_25pct_depth": exceeding,
                "pct_exceeding_25pct_depth": round(exceeding / max(1, total) * 100, 1),
                "by_leg_count": leg_breakdown,
            }
    except Exception as e:
        return {"error": str(e)}


@router.get("/path-performance")
async def arb_path_performance(request: Request):
    """Per-path fill rate and performance tracking for triangular arb."""
    try:
        with _arb_conn() as c:
            # Check if path_performance table exists
            tables = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "path_performance" not in tables:
                return {"error": "path_performance table not yet created — deploy latest code"}

            rows = c.execute(
                "SELECT * FROM path_performance ORDER BY priority_score DESC"
            ).fetchall()
            all_paths = _rows_to_dicts(rows)

            deprioritized = [p for p in all_paths if p.get('is_deprioritized')]
            top_paths = [p for p in all_paths if not p.get('is_deprioritized')][:10]
            worst_paths = sorted(
                [p for p in all_paths if p.get('total_attempts', 0) >= 20],
                key=lambda x: x.get('fill_rate', 0),
            )[:5]

            return {
                "total_paths_tracked": len(all_paths),
                "deprioritized_paths": len(deprioritized),
                "top_paths": [
                    {
                        "path": p["path"],
                        "fill_rate": round(p.get("fill_rate", 0), 3),
                        "avg_profit": round(p.get("avg_profit_per_fill", 0), 4),
                        "score": round(p.get("priority_score", 0), 4),
                        "attempts": p.get("total_attempts", 0),
                        "fills": p.get("total_fills", 0),
                    }
                    for p in top_paths
                ],
                "worst_paths": [
                    {
                        "path": p["path"],
                        "fill_rate": round(p.get("fill_rate", 0), 3),
                        "avg_profit": round(p.get("avg_profit_per_fill", 0), 4),
                        "attempts": p.get("total_attempts", 0),
                        "fills": p.get("total_fills", 0),
                    }
                    for p in worst_paths
                ],
            }
    except Exception as e:
        return {"error": str(e)}


@router.get("/fill-rate-optimization")
async def arb_fill_rate_optimization(request: Request):
    """Fill rate optimization dashboard — preflight, staleness, path ranking, adaptive threshold."""
    result = {
        "current_fill_rate": 0.0,
        "target_fill_rate": 0.75,
        "preflight_stats": {},
        "staleness_filter": {},
        "path_ranking": {},
        "adaptive_threshold": {},
    }

    # Try live data from orchestrator's triangular arb
    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch and hasattr(orch, "triangular_arb"):
        try:
            tri_stats = orch.triangular_arb.get_stats()
            result["preflight_stats"] = tri_stats.get("preflight", {})
            result["staleness_filter"] = {
                "staleness_skips": tri_stats.get("staleness_skips", 0),
                "active_cooldowns": tri_stats.get("staleness_active_cooldowns", 0),
            }
            result["adaptive_threshold"] = tri_stats.get("adaptive_threshold", {})

            # Fill rate from executor
            exec_stats = tri_stats.get("executor", {})
            total = exec_stats.get("total_trades", 0)
            fills = exec_stats.get("total_fills", 0)
            result["current_fill_rate"] = fills / max(1, total)
        except Exception as e:
            logger.debug(f"Live fill rate stats error: {e}")

    # DB-based fill rate (always available)
    try:
        with _arb_conn() as c:
            row = c.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) as fills "
                "FROM arb_trades WHERE strategy='triangular'"
            ).fetchone()
            total_db = row["total"] or 0
            fills_db = row["fills"] or 0
            result["db_fill_rate"] = fills_db / max(1, total_db)
            result["db_total_attempts"] = total_db
            result["db_total_fills"] = fills_db

            # Edge bucket analysis
            edge_buckets = c.execute("""
                SELECT
                    CASE
                        WHEN expected_profit_usd < 0.10 THEN '<$0.10'
                        WHEN expected_profit_usd < 0.20 THEN '$0.10-0.20'
                        WHEN expected_profit_usd < 0.50 THEN '$0.20-0.50'
                        WHEN expected_profit_usd < 1.00 THEN '$0.50-1.00'
                        WHEN expected_profit_usd < 2.00 THEN '$1.00-2.00'
                        ELSE '>$2.00'
                    END as edge_bucket,
                    COUNT(*) as attempts,
                    SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) as fills,
                    ROUND(100.0 * SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) / COUNT(*), 1) as fill_rate_pct,
                    ROUND(AVG(CASE WHEN status='filled' THEN actual_profit_usd ELSE NULL END), 4) as avg_actual_profit
                FROM arb_trades
                WHERE strategy = 'triangular'
                  AND expected_profit_usd IS NOT NULL
                  AND expected_profit_usd > 0
                GROUP BY edge_bucket
                ORDER BY MIN(expected_profit_usd)
            """).fetchall()
            result["edge_buckets"] = _rows_to_dicts(edge_buckets)

            # Path ranking from path_performance table
            tables = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "path_performance" in tables:
                paths = c.execute(
                    "SELECT path, fill_rate, avg_profit_per_fill, total_attempts, is_deprioritized "
                    "FROM path_performance ORDER BY priority_score DESC LIMIT 5"
                ).fetchall()
                result["path_ranking"] = {
                    "total_paths": c.execute("SELECT COUNT(*) FROM path_performance").fetchone()[0],
                    "deprioritized": c.execute(
                        "SELECT COUNT(*) FROM path_performance WHERE is_deprioritized=1"
                    ).fetchone()[0],
                    "top_paths": [dict(r) for r in paths],
                }
    except Exception as e:
        logger.debug(f"DB fill rate stats error: {e}")

    return _sanitize_for_json(result)


@router.get("/contract-verification")
async def arb_contract_verification(request: Request):
    """Contract verification status for cross-exchange safety."""
    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch and hasattr(orch, "contract_verifier"):
        try:
            cv = orch.contract_verifier
            stats = cv.get_stats()
            stats["blocked_details"] = cv.get_mismatches()
            return _sanitize_for_json(stats)
        except Exception as e:
            logger.debug(f"Contract verification status error: {e}")

    return {
        "total_checked": 0,
        "verified": 0,
        "blocked": 0,
        "blocklist": [],
        "mismatches": [],
        "note": "Contract verifier not initialized (orchestrator not running)",
    }


@router.get("/sizing")
async def arb_sizing(request: Request):
    """Dynamic depth-based position sizing analytics.

    Returns config, last-hour stats, by-depth-tier breakdown, and top paths.
    """
    result = {
        "config": {},
        "last_hour": {},
        "by_depth_tier": {},
        "top_paths_by_capital": [],
        "top_paths_by_roi": [],
    }

    # Live config from orchestrator
    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch and hasattr(orch, "triangular_arb"):
        try:
            tri_stats = orch.triangular_arb.get_stats()
            sizing = tri_stats.get("sizing", {})
            result["config"] = {
                "max_single_trade_usd": sizing.get("max_single_trade_usd"),
                "depth_fraction": sizing.get("depth_fraction"),
                "max_capital_per_cycle": sizing.get("max_capital_per_cycle"),
                "hourly_capital_limit": sizing.get("hourly_capital_limit"),
                "edge_thresholds": sizing.get("edge_thresholds"),
            }
            result["live_budget"] = {
                "cycle_capital_deployed": sizing.get("cycle_capital_deployed", 0),
                "hourly_capital_deployed": sizing.get("hourly_capital_deployed", 0),
                "trades_skipped_capital": sizing.get("trades_skipped_capital", 0),
            }
        except Exception as e:
            logger.debug(f"Live sizing stats error: {e}")

    # DB-based analytics
    try:
        with _arb_conn() as c:
            # Check columns exist
            cols = [r[1] for r in c.execute("PRAGMA table_info(arb_trades)").fetchall()]
            has_sizing = "bottleneck_depth_usd" in cols and "sizing_reason" in cols

            # Last hour stats
            hour_row = c.execute("""
                SELECT COUNT(*) as trades,
                       AVG(trade_size_usd) as avg_size,
                       MAX(trade_size_usd) as max_size,
                       SUM(trade_size_usd) as capital_deployed,
                       SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) as skipped,
                       SUM(CASE WHEN status='filled' THEN actual_profit_usd ELSE 0 END) as profit
                FROM arb_trades
                WHERE strategy='triangular'
                  AND timestamp >= datetime('now', '-1 hour')
            """).fetchone()
            result["last_hour"] = {
                "trades": hour_row["trades"] or 0,
                "avg_trade_size": round(float(hour_row["avg_size"] or 0), 2),
                "max_trade_size": round(float(hour_row["max_size"] or 0), 2),
                "capital_deployed": round(float(hour_row["capital_deployed"] or 0), 2),
                "trades_skipped": hour_row["skipped"] or 0,
                "profit": round(float(hour_row["profit"] or 0), 4),
            }

            # Median trade size (approximate via subquery)
            median_row = c.execute("""
                SELECT trade_size_usd FROM arb_trades
                WHERE strategy='triangular' AND status='filled'
                  AND timestamp >= datetime('now', '-1 hour')
                  AND trade_size_usd > 0
                ORDER BY trade_size_usd
                LIMIT 1 OFFSET (
                    SELECT COUNT(*) / 2 FROM arb_trades
                    WHERE strategy='triangular' AND status='filled'
                      AND timestamp >= datetime('now', '-1 hour')
                      AND trade_size_usd > 0
                )
            """).fetchone()
            result["last_hour"]["median_trade_size"] = round(
                float(median_row[0]) if median_row else 0, 2
            )

            # By depth tier (bottleneck_depth_usd ranges)
            if has_sizing:
                depth_tiers = [
                    ("$50-1K", 50, 1000),
                    ("$1K-5K", 1000, 5000),
                    ("$5K-20K", 5000, 20000),
                    ("$20K-100K", 20000, 100000),
                    (">$100K", 100000, 999999999),
                ]
                for tier_name, lo, hi in depth_tiers:
                    row = c.execute(
                        "SELECT COUNT(*) as cnt, "
                        "AVG(trade_size_usd) as avg_size, "
                        "AVG(actual_profit_usd) as avg_profit, "
                        "SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) as fills "
                        "FROM arb_trades "
                        "WHERE strategy='triangular' "
                        "AND bottleneck_depth_usd >= ? AND bottleneck_depth_usd < ?",
                        (lo, hi),
                    ).fetchone()
                    cnt = row["cnt"] or 0
                    result["by_depth_tier"][tier_name] = {
                        "count": cnt,
                        "avg_trade_size": round(float(row["avg_size"] or 0), 2),
                        "avg_profit": round(float(row["avg_profit"] or 0), 4),
                        "fill_rate": round(
                            (row["fills"] or 0) / max(1, cnt), 3
                        ),
                    }

            # Top paths by capital deployed (last 24h)
            capital_paths = c.execute("""
                SELECT path,
                       SUM(trade_size_usd) as total_capital,
                       COUNT(*) as trades,
                       SUM(CASE WHEN status='filled' THEN actual_profit_usd ELSE 0 END) as profit
                FROM arb_trades
                WHERE strategy='triangular' AND path != '' AND path IS NOT NULL
                  AND timestamp >= datetime('now', '-24 hours')
                GROUP BY path
                ORDER BY total_capital DESC
                LIMIT 10
            """).fetchall()
            result["top_paths_by_capital"] = [
                {
                    "path": r["path"],
                    "capital_deployed": round(float(r["total_capital"] or 0), 2),
                    "trades": r["trades"],
                    "profit": round(float(r["profit"] or 0), 4),
                }
                for r in capital_paths
            ]

            # Top paths by ROI (profit/capital, min 5 fills)
            roi_paths = c.execute("""
                SELECT path,
                       SUM(trade_size_usd) as total_capital,
                       SUM(CASE WHEN status='filled' THEN actual_profit_usd ELSE 0 END) as profit,
                       COUNT(*) as trades
                FROM arb_trades
                WHERE strategy='triangular' AND path != '' AND path IS NOT NULL
                  AND timestamp >= datetime('now', '-24 hours')
                GROUP BY path
                HAVING SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) >= 5
                   AND total_capital > 0
                ORDER BY (profit * 1.0 / total_capital) DESC
                LIMIT 10
            """).fetchall()
            result["top_paths_by_roi"] = [
                {
                    "path": r["path"],
                    "roi_pct": round(
                        float(r["profit"] or 0) / max(1, float(r["total_capital"] or 1)) * 100, 4
                    ),
                    "capital_deployed": round(float(r["total_capital"] or 0), 2),
                    "profit": round(float(r["profit"] or 0), 4),
                    "trades": r["trades"],
                }
                for r in roi_paths
            ]

    except Exception as e:
        result["error"] = str(e)

    return _sanitize_for_json(result)


@router.get("/cross-exchange/discovery")
async def arb_cross_exchange_discovery(request: Request):
    """Dynamic pair discovery stats — overlapping pairs, promotions, demotions."""
    orch = getattr(request.app.state, "arb_orchestrator", None)

    result = {
        "discovery": {},
        "book_status": {},
        "cross_exchange_24h": {},
    }

    # Live discovery stats
    if orch and hasattr(orch, "pair_discovery"):
        try:
            result["discovery"] = orch.pair_discovery.get_stats()
        except Exception as e:
            logger.debug(f"Discovery stats error: {e}")

    # Live book status
    if orch and hasattr(orch, "book_manager"):
        try:
            result["book_status"] = orch.book_manager.get_status()
        except Exception as e:
            logger.debug(f"Book status error: {e}")

    # 24h cross-exchange performance from DB
    try:
        with _arb_conn() as c:
            row = c.execute("""
                SELECT COUNT(*) as trades,
                       SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) as fills,
                       COALESCE(SUM(CASE WHEN status='filled' THEN actual_profit_usd ELSE 0 END), 0) as paper_profit,
                       COALESCE(SUM(CASE WHEN status='filled' THEN realistic_profit_usd ELSE 0 END), 0) as realistic_profit,
                       SUM(CASE WHEN edge_survived=1 AND status='filled' THEN 1 ELSE 0 END) as edge_survived,
                       SUM(CASE WHEN edge_survived=0 AND status='filled' THEN 1 ELSE 0 END) as edge_lost,
                       COALESCE(SUM(withdrawal_fee_usd), 0) as total_withdrawal_fees,
                       COALESCE(SUM(taker_fee_usd), 0) as total_taker_fees,
                       COALESCE(SUM(adverse_move_usd), 0) as total_adverse_move
                FROM arb_trades
                WHERE strategy='cross_exchange'
                  AND timestamp >= datetime('now', '-24 hours')
            """).fetchone()

            fills = row["fills"] or 0
            edge_survived = row["edge_survived"] or 0
            result["cross_exchange_24h"] = {
                "trades": row["trades"] or 0,
                "fills": fills,
                "paper_profit_usd": round(float(row["paper_profit"]), 4),
                "realistic_profit_usd": round(float(row["realistic_profit"] or 0), 4),
                "edge_survived": edge_survived,
                "edge_lost": row["edge_lost"] or 0,
                "edge_survival_rate": round(edge_survived / max(1, fills), 3),
                "cost_breakdown": {
                    "withdrawal_fees": round(float(row["total_withdrawal_fees"]), 4),
                    "taker_fees": round(float(row["total_taker_fees"]), 4),
                    "adverse_move": round(float(row["total_adverse_move"]), 4),
                },
            }

            # Per-pair breakdown (top 10 by trade count)
            pair_rows = c.execute("""
                SELECT symbol,
                       COUNT(*) as trades,
                       SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) as fills,
                       COALESCE(SUM(CASE WHEN status='filled' THEN actual_profit_usd ELSE 0 END), 0) as paper_profit,
                       COALESCE(SUM(CASE WHEN status='filled' THEN realistic_profit_usd ELSE 0 END), 0) as realistic_profit,
                       SUM(CASE WHEN edge_survived=1 AND status='filled' THEN 1 ELSE 0 END) as survived
                FROM arb_trades
                WHERE strategy='cross_exchange'
                  AND timestamp >= datetime('now', '-24 hours')
                GROUP BY symbol
                ORDER BY trades DESC
                LIMIT 10
            """).fetchall()
            result["cross_exchange_24h"]["by_pair"] = [
                {
                    "symbol": r["symbol"],
                    "trades": r["trades"],
                    "fills": r["fills"],
                    "paper_profit": round(float(r["paper_profit"]), 4),
                    "realistic_profit": round(float(r["realistic_profit"] or 0), 4),
                    "edge_survival_rate": round(
                        (r["survived"] or 0) / max(1, r["fills"]), 3
                    ),
                }
                for r in pair_rows
            ]
    except Exception as e:
        result["cross_exchange_24h"]["error"] = str(e)

    return _sanitize_for_json(result)


@router.get("/cross-exchange/concentration")
async def arb_cross_exchange_concentration(request: Request):
    """Volume participation limiter status — per-pair limits, blocked pairs."""
    orch = getattr(request.app.state, "arb_orchestrator", None)

    if orch and hasattr(orch, "volume_limiter"):
        try:
            stats = orch.volume_limiter.get_stats()
            return _sanitize_for_json(stats)
        except Exception as e:
            return {"error": str(e)}

    # DB fallback: compute from recent trades
    try:
        with _arb_conn() as c:
            # Trades per pair in last hour
            rows = c.execute("""
                SELECT symbol, COUNT(*) as trades, SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) as fills,
                       SUM(CASE WHEN status='filled' THEN COALESCE(trade_size_usd, 0) ELSE 0 END) as volume_usd
                FROM arb_trades WHERE strategy='cross_exchange'
                  AND timestamp > datetime('now', '-1 hour')
                GROUP BY symbol ORDER BY trades DESC
            """).fetchall()
            return {
                "note": "DB fallback — live limiter not available",
                "pairs_tracked": len(rows),
                "blocked_pairs": [],
                "rejections": {"total": 0},
                "pair_activity_1h": {r["symbol"]: {"trades": r["trades"], "fills": r["fills"],
                                                    "volume_usd": round(r["volume_usd"] or 0, 2)} for r in rows},
            }
    except Exception:
        return {"pairs_tracked": 0, "blocked_pairs": [], "rejections": {"total": 0}}


@router.get("/volume-participation")
async def arb_volume_participation(request: Request):
    """Volume participation stats for all tracked pairs — shows capacity utilization."""
    orch = getattr(request.app.state, "arb_orchestrator", None)

    if orch and hasattr(orch, "volume_limiter"):
        try:
            stats = orch.volume_limiter.get_stats()
            pair_details = stats.get("pair_details", {})

            # Sort by participation rate descending (most active first)
            sorted_pairs = dict(
                sorted(
                    pair_details.items(),
                    key=lambda x: x[1].get('participation_rate', 0),
                    reverse=True,
                )
            )

            return _sanitize_for_json({
                "config": stats.get("config", {}),
                "summary": {
                    "pairs_tracked": stats.get("pairs_tracked", 0),
                    "pairs_liquid": stats.get("pairs_liquid", 0),
                    "pairs_excluded": stats.get("pairs_excluded", 0),
                    "blocked_pairs": stats.get("blocked_pairs", []),
                    "total_remaining_capacity_usd": stats.get("total_remaining_capacity_usd", 0),
                    "rejections": stats.get("rejections", {}),
                },
                "pairs": sorted_pairs,
            })
        except Exception as e:
            return {"error": str(e)}

    # DB fallback: show recent trade activity by pair
    try:
        with _arb_conn() as c:
            rows = c.execute("""
                SELECT symbol, COUNT(*) as trades,
                       SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) as fills,
                       SUM(CASE WHEN status='filled' THEN COALESCE(trade_size_usd, 0) ELSE 0 END) as volume_usd,
                       SUM(CASE WHEN status='filled' THEN COALESCE(paper_profit_usd, 0) ELSE 0 END) as paper_pnl,
                       SUM(CASE WHEN status='filled' THEN COALESCE(realistic_profit_usd, 0) ELSE 0 END) as real_pnl
                FROM arb_trades WHERE strategy='cross_exchange'
                  AND timestamp > datetime('now', '-1 hour')
                GROUP BY symbol ORDER BY volume_usd DESC
            """).fetchall()
            return {
                "note": "DB fallback — live volume limiter not available",
                "config": {"max_participation_rate": 0.02, "min_daily_volume_usd": 500000},
                "summary": {"pairs_active_1h": len(rows)},
                "pairs": {r["symbol"]: {
                    "trades_1h": r["trades"], "fills_1h": r["fills"],
                    "volume_usd_1h": round(r["volume_usd"] or 0, 2),
                    "paper_pnl_1h": round(r["paper_pnl"] or 0, 4),
                    "realistic_pnl_1h": round(r["real_pnl"] or 0, 4),
                } for r in rows},
            }
    except Exception as e:
        return {"error": f"DB fallback failed: {e}"}


@router.get("/cross-exchange/realistic")
async def arb_cross_exchange_realistic(request: Request):
    """Paper vs Realistic P&L comparison for cross-exchange arb."""
    result = {
        "live_stats": {},
        "all_time": {},
        "daily": [],
    }

    # Live executor stats (includes realistic fill stats)
    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch and hasattr(orch, "executor"):
        try:
            result["live_stats"] = orch.executor.get_stats()
        except Exception as e:
            logger.debug(f"Live executor stats error: {e}")

    # All-time and daily breakdown from DB
    try:
        with _arb_conn() as c:
            # All-time
            row = c.execute("""
                SELECT COUNT(*) as fills,
                       COALESCE(SUM(actual_profit_usd), 0) as paper_total,
                       COALESCE(SUM(realistic_profit_usd), 0) as realistic_total,
                       COALESCE(SUM(withdrawal_fee_usd), 0) as wfees,
                       COALESCE(SUM(taker_fee_usd), 0) as tfees,
                       COALESCE(SUM(adverse_move_usd), 0) as adverse,
                       SUM(CASE WHEN edge_survived=1 THEN 1 ELSE 0 END) as survived,
                       SUM(CASE WHEN edge_survived=0 THEN 1 ELSE 0 END) as lost
                FROM arb_trades
                WHERE strategy='cross_exchange' AND status='filled'
            """).fetchone()

            fills = row["fills"] or 0
            result["all_time"] = {
                "fills": fills,
                "paper_profit_usd": round(float(row["paper_total"]), 4),
                "realistic_profit_usd": round(float(row["realistic_total"] or 0), 4),
                "phantom_pnl_usd": round(
                    float(row["paper_total"]) - float(row["realistic_total"] or row["paper_total"]), 4
                ),
                "total_withdrawal_fees": round(float(row["wfees"]), 4),
                "total_taker_fees": round(float(row["tfees"]), 4),
                "total_adverse_move": round(float(row["adverse"]), 4),
                "edge_survived": row["survived"] or 0,
                "edge_lost": row["lost"] or 0,
                "edge_survival_rate": round(
                    (row["survived"] or 0) / max(1, fills), 3
                ),
            }

            # Daily breakdown (last 7 days)
            daily_rows = c.execute("""
                SELECT date(timestamp) as day,
                       COUNT(*) as fills,
                       COALESCE(SUM(actual_profit_usd), 0) as paper,
                       COALESCE(SUM(realistic_profit_usd), 0) as realistic,
                       SUM(CASE WHEN edge_survived=1 THEN 1 ELSE 0 END) as survived
                FROM arb_trades
                WHERE strategy='cross_exchange' AND status='filled'
                  AND timestamp >= datetime('now', '-7 days')
                GROUP BY day
                ORDER BY day DESC
            """).fetchall()
            result["daily"] = [
                {
                    "date": r["day"],
                    "fills": r["fills"],
                    "paper_profit": round(float(r["paper"]), 4),
                    "realistic_profit": round(float(r["realistic"] or 0), 4),
                    "edge_survival_rate": round(
                        (r["survived"] or 0) / max(1, r["fills"]), 3
                    ),
                }
                for r in daily_rows
            ]
    except Exception as e:
        result["error"] = str(e)

    return _sanitize_for_json(result)


def _sanitize_for_json(obj):
    """Convert Decimal and other non-JSON types to float/str."""
    from decimal import Decimal
    import math
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj
