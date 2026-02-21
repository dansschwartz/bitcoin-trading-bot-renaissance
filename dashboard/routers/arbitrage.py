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
