"""Arbitrage dashboard endpoints — reads from data/arbitrage.db and live orchestrator."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
import yaml

logger = logging.getLogger(__name__)

_ARB_CFG_PATH = Path(__file__).resolve().parent.parent.parent / "arbitrage" / "config" / "arbitrage.yaml"

def _read_arb_config() -> dict:
    try:
        with open(_ARB_CFG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

router = APIRouter(prefix="/api/arbitrage", tags=["arbitrage"])

ARB_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "arbitrage.db"

_INDEXES_ENSURED = False

def _ensure_indexes(conn):
    """Create indexes for faster GROUP BY queries (idempotent)."""
    global _INDEXES_ENSURED
    if _INDEXES_ENSURED:
        return
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_arb_trades_ts_strategy_status "
            "ON arb_trades(timestamp, strategy, status)"
        )
        conn.commit()
        _INDEXES_ENSURED = True
    except Exception as e:
        logger.debug(f"Index creation: {e}")


@contextmanager
def _arb_conn():
    """Connect to arbitrage.db (separate from main bot DB)."""
    conn = sqlite3.connect(str(ARB_DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        _ensure_indexes(conn)
        yield conn
    finally:
        conn.close()


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def _query_extra_strategy_pnl_daily(conn, days: int) -> Dict[str, Dict[str, Dict]]:
    """Query basis/funding/listing/pairs tables for daily P&L.
    Returns {date: {strategy: {pnl, trades, wins}}}
    """
    result: Dict[str, Dict[str, Dict]] = {}
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    # Basis positions (closed)
    if "basis_positions" in tables:
        try:
            rows = conn.execute(
                """SELECT date(exit_timestamp) as d,
                          COALESCE(SUM(final_pnl), 0) as pnl,
                          COUNT(*) as trades,
                          SUM(CASE WHEN final_pnl > 0 THEN 1 ELSE 0 END) as wins
                   FROM basis_positions
                   WHERE is_open = 0 AND exit_timestamp IS NOT NULL
                     AND date(exit_timestamp) >= date('now', ?)
                   GROUP BY date(exit_timestamp)""",
                (f"-{days} days",),
            ).fetchall()
            for r in rows:
                d = r["d"]
                if d:
                    result.setdefault(d, {})
                    result[d]["basis_trading"] = {
                        "pnl": round(float(r["pnl"]), 4),
                        "trades": r["trades"],
                        "wins": r["wins"],
                    }
        except Exception as e:
            logger.debug(f"basis daily pnl: {e}")

    # Funding positions (closed)
    if "funding_positions" in tables:
        try:
            rows = conn.execute(
                """SELECT date(exit_timestamp) as d,
                          COALESCE(SUM(final_pnl), 0) as pnl,
                          COUNT(*) as trades,
                          SUM(CASE WHEN final_pnl > 0 THEN 1 ELSE 0 END) as wins
                   FROM funding_positions
                   WHERE status = 'closed' AND exit_timestamp IS NOT NULL
                     AND date(exit_timestamp) >= date('now', ?)
                   GROUP BY date(exit_timestamp)""",
                (f"-{days} days",),
            ).fetchall()
            for r in rows:
                d = r["d"]
                if d:
                    result.setdefault(d, {})
                    result[d]["funding_rate"] = {
                        "pnl": round(float(r["pnl"]), 4),
                        "trades": r["trades"],
                        "wins": r["wins"],
                    }
        except Exception as e:
            logger.debug(f"funding daily pnl: {e}")

    # Listing arb positions (closed)
    if "listing_arb_positions" in tables:
        try:
            rows = conn.execute(
                """SELECT date(exit_time) as d,
                          COALESCE(SUM(realized_pnl_usd), 0) as pnl,
                          COUNT(*) as trades,
                          SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins
                   FROM listing_arb_positions
                   WHERE is_open = 0 AND exit_time IS NOT NULL
                     AND date(exit_time) >= date('now', ?)
                   GROUP BY date(exit_time)""",
                (f"-{days} days",),
            ).fetchall()
            for r in rows:
                d = r["d"]
                if d:
                    result.setdefault(d, {})
                    result[d]["listing_arb"] = {
                        "pnl": round(float(r["pnl"]), 4),
                        "trades": r["trades"],
                        "wins": r["wins"],
                    }
        except Exception as e:
            logger.debug(f"listing daily pnl: {e}")

    # Pairs positions (closed)
    if "pairs_positions" in tables:
        try:
            rows = conn.execute(
                """SELECT date(exit_time) as d,
                          COALESCE(SUM(realized_pnl_usd), 0) as pnl,
                          COUNT(*) as trades,
                          SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins
                   FROM pairs_positions
                   WHERE is_open = 0 AND exit_time IS NOT NULL
                     AND date(exit_time) >= date('now', ?)
                   GROUP BY date(exit_time)""",
                (f"-{days} days",),
            ).fetchall()
            for r in rows:
                d = r["d"]
                if d:
                    result.setdefault(d, {})
                    result[d]["pairs_arb"] = {
                        "pnl": round(float(r["pnl"]), 4),
                        "trades": r["trades"],
                        "wins": r["wins"],
                    }
        except Exception as e:
            logger.debug(f"pairs daily pnl: {e}")

    return result


def _query_extra_strategy_pnl_hourly(conn, hours: int) -> Dict[str, Dict[str, float]]:
    """Query basis/funding/listing/pairs tables for hourly P&L.
    Returns {hour: {strategy: pnl}}
    """
    result: Dict[str, Dict[str, float]] = {}
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if "basis_positions" in tables:
        try:
            rows = conn.execute(
                """SELECT strftime('%Y-%m-%d %H:00', exit_timestamp) as h,
                          COALESCE(SUM(final_pnl), 0) as pnl
                   FROM basis_positions
                   WHERE is_open = 0 AND exit_timestamp IS NOT NULL
                     AND exit_timestamp >= datetime('now', ?)
                   GROUP BY strftime('%Y-%m-%d %H:00', exit_timestamp)""",
                (f"-{hours} hours",),
            ).fetchall()
            for r in rows:
                h = r["h"]
                if h:
                    result.setdefault(h, {})
                    result[h]["basis_trading"] = round(float(r["pnl"]), 4)
        except Exception as e:
            logger.debug(f"basis hourly pnl: {e}")

    if "funding_positions" in tables:
        try:
            rows = conn.execute(
                """SELECT strftime('%Y-%m-%d %H:00', exit_timestamp) as h,
                          COALESCE(SUM(final_pnl), 0) as pnl
                   FROM funding_positions
                   WHERE status = 'closed' AND exit_timestamp IS NOT NULL
                     AND exit_timestamp >= datetime('now', ?)
                   GROUP BY strftime('%Y-%m-%d %H:00', exit_timestamp)""",
                (f"-{hours} hours",),
            ).fetchall()
            for r in rows:
                h = r["h"]
                if h:
                    result.setdefault(h, {})
                    result[h]["funding_rate"] = round(float(r["pnl"]), 4)
        except Exception as e:
            logger.debug(f"funding hourly pnl: {e}")

    if "listing_arb_positions" in tables:
        try:
            rows = conn.execute(
                """SELECT strftime('%Y-%m-%d %H:00', exit_time) as h,
                          COALESCE(SUM(realized_pnl_usd), 0) as pnl
                   FROM listing_arb_positions
                   WHERE is_open = 0 AND exit_time IS NOT NULL
                     AND exit_time >= datetime('now', ?)
                   GROUP BY strftime('%Y-%m-%d %H:00', exit_time)""",
                (f"-{hours} hours",),
            ).fetchall()
            for r in rows:
                h = r["h"]
                if h:
                    result.setdefault(h, {})
                    result[h]["listing_arb"] = round(float(r["pnl"]), 4)
        except Exception as e:
            logger.debug(f"listing hourly pnl: {e}")

    if "pairs_positions" in tables:
        try:
            rows = conn.execute(
                """SELECT strftime('%Y-%m-%d %H:00', exit_time) as h,
                          COALESCE(SUM(realized_pnl_usd), 0) as pnl
                   FROM pairs_positions
                   WHERE is_open = 0 AND exit_time IS NOT NULL
                     AND exit_time >= datetime('now', ?)
                   GROUP BY strftime('%Y-%m-%d %H:00', exit_time)""",
                (f"-{hours} hours",),
            ).fetchall()
            for r in rows:
                h = r["h"]
                if h:
                    result.setdefault(h, {})
                    result[h]["pairs_arb"] = round(float(r["pnl"]), 4)
        except Exception as e:
            logger.debug(f"pairs hourly pnl: {e}")

    return result


def _query_extra_strategy_totals(conn) -> Dict[str, Dict[str, Any]]:
    """Query total + today's P&L for each extra strategy.
    Returns {strategy: {profit_usd, trades, fills, today_pnl_usd}}
    """
    result: Dict[str, Dict[str, Any]] = {}
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if "basis_positions" in tables:
        try:
            row = conn.execute(
                """SELECT COUNT(*) as trades,
                          COALESCE(SUM(final_pnl), 0) as profit,
                          SUM(CASE WHEN final_pnl > 0 THEN 1 ELSE 0 END) as wins
                   FROM basis_positions WHERE is_open = 0"""
            ).fetchone()
            today = conn.execute(
                """SELECT COALESCE(SUM(final_pnl), 0) as today
                   FROM basis_positions
                   WHERE is_open = 0 AND date(exit_timestamp) = date('now')"""
            ).fetchone()
            result["basis_trading"] = {
                "profit_usd": round(float(row["profit"]), 4),
                "trades": row["trades"],
                "fills": row["trades"],
                "today_pnl_usd": round(float(today["today"]), 4),
            }
        except Exception as e:
            logger.debug(f"basis totals: {e}")

    if "funding_positions" in tables:
        try:
            row = conn.execute(
                """SELECT COUNT(*) as trades,
                          COALESCE(SUM(final_pnl), 0) as profit,
                          SUM(CASE WHEN final_pnl > 0 THEN 1 ELSE 0 END) as wins
                   FROM funding_positions WHERE status = 'closed'"""
            ).fetchone()
            today = conn.execute(
                """SELECT COALESCE(SUM(final_pnl), 0) as today
                   FROM funding_positions
                   WHERE status = 'closed' AND date(exit_timestamp) = date('now')"""
            ).fetchone()
            result["funding_rate"] = {
                "profit_usd": round(float(row["profit"]), 4),
                "trades": row["trades"],
                "fills": row["trades"],
                "today_pnl_usd": round(float(today["today"]), 4),
            }
        except Exception as e:
            logger.debug(f"funding totals: {e}")

    if "listing_arb_positions" in tables:
        try:
            row = conn.execute(
                """SELECT COUNT(*) as trades,
                          COALESCE(SUM(realized_pnl_usd), 0) as profit,
                          SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins
                   FROM listing_arb_positions WHERE is_open = 0"""
            ).fetchone()
            today = conn.execute(
                """SELECT COALESCE(SUM(realized_pnl_usd), 0) as today
                   FROM listing_arb_positions
                   WHERE is_open = 0 AND date(exit_time) = date('now')"""
            ).fetchone()
            result["listing_arb"] = {
                "profit_usd": round(float(row["profit"]), 4),
                "trades": row["trades"],
                "fills": row["trades"],
                "today_pnl_usd": round(float(today["today"]), 4),
            }
        except Exception as e:
            logger.debug(f"listing totals: {e}")

    if "pairs_positions" in tables:
        try:
            row = conn.execute(
                """SELECT COUNT(*) as trades,
                          COALESCE(SUM(realized_pnl_usd), 0) as profit,
                          SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins
                   FROM pairs_positions WHERE is_open = 0"""
            ).fetchone()
            today = conn.execute(
                """SELECT COALESCE(SUM(realized_pnl_usd), 0) as today
                   FROM pairs_positions
                   WHERE is_open = 0 AND date(exit_time) = date('now')"""
            ).fetchone()
            result["pairs_arb"] = {
                "profit_usd": round(float(row["profit"]), 4),
                "trades": row["trades"],
                "fills": row["trades"],
                "today_pnl_usd": round(float(today["today"]), 4),
            }
        except Exception as e:
            logger.debug(f"pairs totals: {e}")

    return result


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

            # Check if the bot process is running by looking for recent activity
            bot_running = False
            from datetime import datetime, timedelta, timezone
            now = datetime.now(timezone.utc)

            # Check 1: recent arb trades in arb DB
            try:
                last_arb = c.execute(
                    "SELECT MAX(timestamp) as ts FROM arb_trades"
                ).fetchone()
                if last_arb and last_arb["ts"]:
                    last_arb_dt = datetime.fromisoformat(str(last_arb["ts"]).replace("Z", "+00:00"))
                    if now - last_arb_dt < timedelta(minutes=10):
                        bot_running = True
            except Exception:
                pass

            # Check 2: recent decisions in main DB
            if not bot_running:
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
                            last_decision = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
                            if now - last_decision < timedelta(minutes=10):
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
    """Recent arbitrage trades from arb_trades table (excludes skipped)."""
    try:
        with _arb_conn() as c:
            if strategy:
                rows = c.execute(
                    """SELECT * FROM arb_trades
                       WHERE strategy = ? AND status != 'skipped'
                       ORDER BY id DESC LIMIT ? OFFSET ?""",
                    (strategy, min(limit, 500), offset),
                ).fetchall()
            else:
                rows = c.execute(
                    """SELECT * FROM arb_trades
                       WHERE status != 'skipped'
                       ORDER BY id DESC LIMIT ? OFFSET ?""",
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

            # By strategy (arb_trades: cross_exchange + triangular)
            strategy_rows = c.execute(
                """SELECT strategy,
                          COUNT(*) as trades,
                          COALESCE(SUM(actual_profit_usd), 0) as profit,
                          SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as fills
                   FROM arb_trades GROUP BY strategy"""
            ).fetchall()

            # Today's P&L per strategy from arb_trades
            today_by_strategy = {}
            try:
                today_rows = c.execute(
                    """SELECT strategy,
                              COALESCE(SUM(actual_profit_usd), 0) as today_pnl
                       FROM arb_trades
                       WHERE date(timestamp) = date('now') AND status = 'filled'
                       GROUP BY strategy"""
                ).fetchall()
                for r in today_rows:
                    today_by_strategy[r["strategy"]] = round(float(r["today_pnl"]), 4)
            except Exception:
                pass

            by_strategy = [
                {
                    "strategy": r["strategy"],
                    "trades": r["trades"],
                    "fills": r["fills"],
                    "profit_usd": round(float(r["profit"]), 4),
                    "today_pnl_usd": today_by_strategy.get(r["strategy"], 0.0),
                }
                for r in strategy_rows
            ]

            # Add extra strategies from other tables
            extra = _query_extra_strategy_totals(c)
            for strat_name, strat_data in extra.items():
                by_strategy.append({
                    "strategy": strat_name,
                    "trades": strat_data["trades"],
                    "fills": strat_data["fills"],
                    "profit_usd": strat_data["profit_usd"],
                    "today_pnl_usd": strat_data.get("today_pnl_usd", 0.0),
                })

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

            # Today's P&L (all strategies combined)
            daily_arb = c.execute(
                """SELECT COALESCE(SUM(actual_profit_usd), 0) as daily
                   FROM arb_trades
                   WHERE date(timestamp) = date('now') AND status = 'filled'"""
            ).fetchone()
            daily_total = float(daily_arb["daily"])
            # Add extra strategy today P&L
            for strat_data in extra.values():
                daily_total += strat_data.get("today_pnl_usd", 0.0)

            # Total profit including extra strategies
            total_profit = float(profit_row["total_profit"])
            for strat_data in extra.values():
                total_profit += strat_data["profit_usd"]

            return {
                "total_trades": total,
                "filled_trades": filled,
                "total_profit_usd": round(total_profit, 4),
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.0,
                "signals_total": signal_total,
                "signals_approved": signal_approved,
                "daily_pnl_usd": round(daily_total, 4),
                "by_strategy": by_strategy,
            }
    except Exception as e:
        return {"error": str(e)}


@router.get("/daily-pnl")
async def arb_daily_pnl(request: Request):
    """Daily P&L for the last N days (default 10), with per-strategy breakdown."""
    days = int(request.query_params.get("days", 10))
    try:
        with _arb_conn() as c:
            # arb_trades grouped by date + strategy
            rows = c.execute(
                """SELECT date(timestamp) as date, strategy,
                          COALESCE(SUM(actual_profit_usd), 0) as pnl,
                          COUNT(*) as trades,
                          SUM(CASE WHEN actual_profit_usd > 0 THEN 1 ELSE 0 END) as wins
                   FROM arb_trades
                   WHERE status = 'filled'
                     AND date(timestamp) >= date('now', ?)
                   GROUP BY date(timestamp), strategy
                   ORDER BY date(timestamp)""",
                (f"-{days} days",),
            ).fetchall()

            # Build per-date dict
            daily: Dict[str, Dict] = {}
            for r in rows:
                d = r["date"]
                if d not in daily:
                    daily[d] = {"date": d, "pnl": 0.0, "trades": 0, "wins": 0, "by_strategy": {}}
                strat = r["strategy"]
                pnl = round(float(r["pnl"]), 4)
                trades = r["trades"]
                w = r["wins"]
                daily[d]["pnl"] += pnl
                daily[d]["trades"] += trades
                daily[d]["wins"] += w
                daily[d]["by_strategy"][strat] = {"pnl": pnl, "trades": trades, "wins": w}

            # Merge extra strategy data
            extra = _query_extra_strategy_pnl_daily(c, days)
            for d, strats in extra.items():
                if d not in daily:
                    daily[d] = {"date": d, "pnl": 0.0, "trades": 0, "wins": 0, "by_strategy": {}}
                for strat, data in strats.items():
                    daily[d]["pnl"] += data["pnl"]
                    daily[d]["trades"] += data["trades"]
                    daily[d]["wins"] += data["wins"]
                    daily[d]["by_strategy"][strat] = data

            # Round totals
            for d in daily.values():
                d["pnl"] = round(d["pnl"], 4)

            return sorted(daily.values(), key=lambda x: x["date"])
    except Exception as e:
        return {"error": str(e)}


@router.get("/hourly-pnl")
async def arb_hourly_pnl(request: Request):
    """Hourly P&L for the last N hours (default 48), with per-strategy breakdown."""
    hours = int(request.query_params.get("hours", 48))
    try:
        with _arb_conn() as c:
            # arb_trades grouped by hour + strategy
            rows = c.execute(
                """SELECT strftime('%Y-%m-%d %H:00', timestamp) as hour, strategy,
                          COALESCE(SUM(actual_profit_usd), 0) as pnl,
                          COUNT(*) as trades,
                          SUM(CASE WHEN actual_profit_usd > 0 THEN 1 ELSE 0 END) as wins
                   FROM arb_trades
                   WHERE status = 'filled'
                     AND timestamp >= datetime('now', ?)
                   GROUP BY strftime('%Y-%m-%d %H:00', timestamp), strategy
                   ORDER BY hour""",
                (f"-{hours} hours",),
            ).fetchall()

            hourly: Dict[str, Dict] = {}
            for r in rows:
                h = r["hour"]
                if h not in hourly:
                    hourly[h] = {"hour": h, "pnl": 0.0, "trades": 0, "wins": 0, "by_strategy": {}}
                strat = r["strategy"]
                pnl = round(float(r["pnl"]), 4)
                hourly[h]["pnl"] += pnl
                hourly[h]["trades"] += r["trades"]
                hourly[h]["wins"] += r["wins"]
                hourly[h]["by_strategy"][strat] = pnl

            # Merge extra strategy data
            extra = _query_extra_strategy_pnl_hourly(c, hours)
            for h, strats in extra.items():
                if h not in hourly:
                    hourly[h] = {"hour": h, "pnl": 0.0, "trades": 0, "wins": 0, "by_strategy": {}}
                for strat, pnl in strats.items():
                    hourly[h]["pnl"] += pnl
                    hourly[h]["by_strategy"][strat] = pnl

            for h in hourly.values():
                h["pnl"] = round(h["pnl"], 4)

            return sorted(hourly.values(), key=lambda x: x["hour"])
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

            # Add extra strategy P&L
            extra = _query_extra_strategy_totals(c)
            for strat_data in extra.values():
                total_profit += strat_data["profit_usd"]
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
                       SUM(CASE WHEN status='filled' THEN COALESCE(actual_profit_usd, 0) ELSE 0 END) as paper_pnl,
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



@router.get("/basis")
async def arb_basis(request: Request):
    """Basis trading (spot-futures convergence) status and snapshots."""
    orch = getattr(request.app.state, "arb_orchestrator", None)

    _cfg = _read_arb_config()
    result = {
        "observation_mode": _cfg.get("basis_trading", {}).get("observation_mode", True),
        "current_basis": {},
        "recent_snapshots": [],
        "recent_opportunities": [],
        "stats": {},
    }

    # Try live data from orchestrator
    if orch and hasattr(orch, "basis_arb"):
        try:
            stats = orch.basis_arb.get_status()
            result["stats"] = stats
            result["observation_mode"] = stats.get("observation_mode", True)
            result["current_basis"] = stats.get("current_basis", {})
        except Exception as e:
            logger.debug(f"Live basis stats error: {e}")

    # DB data
    try:
        with _arb_conn() as c:
            tables = [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]

            if "basis_snapshots" in tables:
                rows = c.execute(
                    "SELECT * FROM basis_snapshots ORDER BY id DESC LIMIT 50"
                ).fetchall()
                result["recent_snapshots"] = _rows_to_dicts(rows)

                if not result["current_basis"]:
                    latest = c.execute(
                        "SELECT symbol, spot_price, futures_price, basis_bps, "
                        "direction, annualized_basis_pct, funding_rate, timestamp "
                        "FROM basis_snapshots "
                        "WHERE id IN (SELECT MAX(id) FROM basis_snapshots GROUP BY symbol)"
                    ).fetchall()
                    for r in latest:
                        result["current_basis"][r["symbol"]] = {
                            "spot": r["spot_price"],
                            "futures": r["futures_price"],
                            "basis_bps": r["basis_bps"],
                            "direction": r["direction"],
                            "annualized_pct": r["annualized_basis_pct"],
                            "funding_rate": r["funding_rate"],
                            "last_update": r["timestamp"],
                        }

                count = c.execute("SELECT COUNT(*) FROM basis_snapshots").fetchone()[0]
                result["stats"]["total_snapshots"] = count

            if "basis_opportunities" in tables:
                rows = c.execute(
                    "SELECT * FROM basis_opportunities ORDER BY id DESC LIMIT 20"
                ).fetchall()
                result["recent_opportunities"] = _rows_to_dicts(rows)

                opp_count = c.execute("SELECT COUNT(*) FROM basis_opportunities").fetchone()[0]
                result["stats"]["total_opportunities"] = opp_count

            if "basis_positions" in tables:
                open_count = c.execute(
                    "SELECT COUNT(*) FROM basis_positions WHERE is_open=1"
                ).fetchone()[0]
                result["stats"]["open_positions"] = open_count

                # Fetch actual open position rows
                open_rows = c.execute(
                    "SELECT * FROM basis_positions WHERE is_open=1 ORDER BY entry_timestamp DESC"
                ).fetchall()
                result["open_positions"] = _rows_to_dicts(open_rows)

                # Fetch last 20 closed positions
                closed_rows = c.execute(
                    "SELECT * FROM basis_positions WHERE is_open=0 "
                    "ORDER BY exit_timestamp DESC LIMIT 20"
                ).fetchall()
                result["recent_trades"] = _rows_to_dicts(closed_rows)

                # Total P&L from all closed positions
                total_pnl_row = c.execute(
                    "SELECT COALESCE(SUM(final_pnl), 0) as total FROM basis_positions WHERE is_open=0"
                ).fetchone()
                result["total_pnl"] = total_pnl_row[0] if total_pnl_row else 0

    except Exception as e:
        result["error"] = str(e)

    return _sanitize_for_json(result)


@router.get("/listing")
async def arb_listing(request: Request):
    """Listing arbitrage (new MEXC token detection) status."""
    orch = getattr(request.app.state, "arb_orchestrator", None)

    _cfg = _read_arb_config()
    result = {
        "observation_mode": _cfg.get("listing_arbitrage", {}).get("observation_mode", True),
        "listings_evaluated": 0,
        "trades_opened": 0,
        "trades_closed": 0,
        "total_pnl_usd": 0,
        "open_positions": [],
        "recent_listings": [],
        "monitor_stats": {},
        "hard_limits": {
            "max_position_usd": 200,
            "max_concurrent": 2,
            "max_hold_minutes": 60,
        },
    }

    # Try live data from orchestrator
    if orch and hasattr(orch, "listing_arb"):
        try:
            result.update(orch.listing_arb.get_status())
        except Exception as e:
            logger.debug(f"Live listing arb stats error: {e}")

    if orch and hasattr(orch, "listing_monitor"):
        try:
            result["recent_listings"] = orch.listing_monitor.get_recent_listings(20)
            result["monitor_stats"] = orch.listing_monitor.get_stats()
        except Exception as e:
            logger.debug(f"Live listing monitor stats error: {e}")

    # DB fallback for recent listings
    if not result["recent_listings"]:
        try:
            with _arb_conn() as c:
                tables = [r[0] for r in c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]

                if "listing_events" in tables:
                    rows = c.execute(
                        "SELECT symbol, detected_at, is_first_listing, mexc_initial_price, on_binance "
                        "FROM listing_events ORDER BY id DESC LIMIT 20"
                    ).fetchall()
                    result["recent_listings"] = [
                        {
                            "symbol": r["symbol"],
                            "detected_at": r["detected_at"],
                            "is_first_listing": bool(r["is_first_listing"]),
                            "mexc_initial_price": r["mexc_initial_price"],
                            "already_on_binance": bool(r["on_binance"]),
                        }
                        for r in rows
                    ]

                if "listing_arb_evaluations" in tables:
                    eval_count = c.execute(
                        "SELECT COUNT(*) FROM listing_arb_evaluations"
                    ).fetchone()[0]
                    result["listings_evaluated"] = result.get("listings_evaluated", 0) or eval_count

                if "known_symbols_snapshot" in tables:
                    mexc_count = c.execute(
                        "SELECT COUNT(*) FROM known_symbols_snapshot WHERE exchange='mexc'"
                    ).fetchone()[0]
                    binance_count = c.execute(
                        "SELECT COUNT(*) FROM known_symbols_snapshot WHERE exchange='binance'"
                    ).fetchone()[0]
                    if not result["monitor_stats"]:
                        result["monitor_stats"] = {
                            "known_mexc_symbols": mexc_count,
                            "known_binance_symbols": binance_count,
                        }

        except Exception as e:
            result["error"] = str(e)

    return _sanitize_for_json(result)


@router.get("/pairs")
async def arb_pairs(request: Request):
    """Statistical pairs arbitrage (cointegration) status."""
    orch = getattr(request.app.state, "arb_orchestrator", None)

    _cfg = _read_arb_config()
    result = {
        "observation_mode": _cfg.get("statistical_pairs", {}).get("observation_mode", True),
        "cycle_count": 0,
        "opportunities_detected": 0,
        "pair_states": [],
        "open_positions": [],
        "config": {},
    }

    # Try live data from orchestrator
    if orch and hasattr(orch, "pairs_arb"):
        try:
            result.update(orch.pairs_arb.get_status())
        except Exception as e:
            logger.debug(f"Live pairs arb stats error: {e}")

    # DB fallback for signal counts
    if not result.get("pair_states"):
        try:
            with _arb_conn() as c:
                tables = [r[0] for r in c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]

                if "pairs_signals" in tables:
                    sig_count = c.execute("SELECT COUNT(*) FROM pairs_signals").fetchone()[0]
                    result["total_signals"] = sig_count

                    opp_count = c.execute(
                        "SELECT COUNT(*) FROM pairs_signals WHERE action_taken='observation_logged'"
                    ).fetchone()[0]
                    result["opportunities_detected"] = opp_count

                    # Latest z-scores per pair
                    latest = c.execute("""
                        SELECT base_symbol, quote_symbol, z_score, half_life,
                               is_cointegrated, adf_pvalue, signal
                        FROM pairs_signals
                        WHERE id IN (
                            SELECT MAX(id) FROM pairs_signals
                            GROUP BY base_symbol, quote_symbol
                        )
                    """).fetchall()
                    result["pair_states"] = [
                        {
                            "base": r["base_symbol"],
                            "quote": r["quote_symbol"],
                            "z_score": r["z_score"],
                            "half_life_bars": r["half_life"],
                            "is_cointegrated": bool(r["is_cointegrated"]),
                            "adf_pvalue": r["adf_pvalue"],
                            "signal": r["signal"],
                            "has_open_position": False,
                        }
                        for r in latest
                    ]

        except Exception as e:
            result["error"] = str(e)

    return _sanitize_for_json(result)


@router.get("/temporal")
async def arb_temporal(request: Request):
    """Temporal pattern analysis — bias weights by hour/day/funding window."""
    import json
    from pathlib import Path

    # Try live orchestrator first
    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch:
        try:
            report = orch.temporal_analyzer.get_report()
            return _sanitize_for_json(report)
        except Exception as e:
            pass  # Fall through to cache

    # Fallback: read from cached file
    cache_path = Path("data/temporal_profiles.json")
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache_data = json.load(f)
            profiles = cache_data.get("profiles", {})
            report = {
                "status": "from_cache",
                "total_profiles": len(profiles),
                "total_trades": cache_data.get("trade_count", 0),
                "last_refresh": cache_data.get("last_refresh"),
            }
            # Extract aggregate profiles: "cross_exchange:*", "triangular:*", "*:*" (global)
            for strategy_key, label in [("*:*", "global"), ("cross_exchange:*", "cross_exchange"), ("triangular:*", "triangular")]:
                if strategy_key in profiles:
                    p = profiles[strategy_key]
                    by_hour = p.get("by_hour", {})
                    by_dow = p.get("by_dow", {})
                    # Best/worst hours
                    hour_list = [(h, b.get("bias_weight", 1.0), b.get("avg_profit_usd", 0), b.get("total_trades", 0))
                                 for h, b in by_hour.items()]
                    hour_list.sort(key=lambda x: x[1], reverse=True)
                    report[f"{label}_best_hours"] = [
                        {"hour": h, "bias": round(w, 3), "avg_profit": round(p, 4), "trades": t}
                        for h, w, p, t in hour_list[:5]
                    ]
                    report[f"{label}_worst_hours"] = [
                        {"hour": h, "bias": round(w, 3), "avg_profit": round(p, 4), "trades": t}
                        for h, w, p, t in hour_list[-5:]
                    ]
                    # Day of week
                    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    dow_list = [(dow_names[int(d)] if d.isdigit() and int(d) < 7 else d,
                                 b.get("bias_weight", 1.0), b.get("avg_profit_usd", 0))
                                for d, b in by_dow.items()]
                    dow_list.sort(key=lambda x: x[1], reverse=True)
                    report[f"{label}_by_dow"] = [
                        {"day": d, "bias": round(w, 3), "avg_profit": round(p, 4)}
                        for d, w, p in dow_list
                    ]
            return _sanitize_for_json(report)
        except Exception as e:
            return {"error": f"cache read failed: {e}"}

    return {"error": "no temporal data available (orchestrator not running, no cache)"}


@router.get("/pair-expansion")
async def arb_pair_expansion(request: Request):
    """MEXC pair expansion — discovered pairs, tiers, scores."""
    import json
    from pathlib import Path

    # Try live orchestrator first
    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch:
        try:
            result = {}
            result["manager"] = orch.expanded_pair_manager.get_report()
            result["discovery"] = orch.mexc_pair_discovery.get_report()
            return _sanitize_for_json(result)
        except Exception:
            pass  # Fall through to cache

    # Fallback: read from cached file
    cache_path = Path("data/pair_expansion_cache.json")
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            return _sanitize_for_json(data)
        except Exception as e:
            return {"error": f"cache read failed: {e}"}

    return {
        "status": "not_enabled",
        "note": "Pair expansion has not completed a scan yet. "
                "Check arbitrage.yaml pair_expansion.enabled and wait for first scan cycle.",
    }




@router.get("/capital-velocity")
async def arb_capital_velocity(request: Request):
    """Capital velocity — profit per capital-hour by strategy."""
    import json
    from pathlib import Path

    # Try live orchestrator first
    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch:
        try:
            return _sanitize_for_json(orch.velocity_tracker.get_velocity_report())
        except Exception:
            pass

    # Fallback: cached file
    cache_path = Path("data/capital_velocity_cache.json")
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"cache read failed: {e}"}

    return {"strategies": {}, "note": "Velocity tracker has no data yet."}


@router.get("/edge-decay")
async def arb_edge_decay(request: Request):
    """Edge decay — profitability trend per strategy (7-day rolling)."""
    import json
    from pathlib import Path

    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch:
        try:
            return _sanitize_for_json(orch.edge_decay_monitor.get_decay_report())
        except Exception:
            pass

    cache_path = Path("data/edge_decay_cache.json")
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"cache read failed: {e}"}

    return {"strategies": {}, "note": "Edge decay monitor has not run yet."}


@router.get("/strategy-allocation")
async def arb_strategy_allocation(request: Request):
    """Strategy allocation — current and target capital allocation."""
    import json
    from pathlib import Path

    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch:
        try:
            return _sanitize_for_json(orch.strategy_allocator.get_allocation_report())
        except Exception:
            pass

    cache_path = Path("data/strategy_allocation_cache.json")
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"cache read failed: {e}"}

    return {"observation_mode": True, "note": "Strategy allocator has not run yet."}


@router.get("/exhaust-snapshots")
async def arb_exhaust_snapshots(request: Request):
    """Data exhaust — recent order book snapshots around trade events."""
    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch:
        try:
            snapshots = orch.exhaust_capture.get_recent_snapshots(limit=100)
            return _sanitize_for_json({"snapshots": snapshots, "count": len(snapshots)})
        except Exception:
            pass

    # Fallback: query DB directly
    try:
        with _arb_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM arb_signal_snapshots ORDER BY id DESC LIMIT 100"
            ).fetchall()
            snapshots = _rows_to_dicts(rows)
            return {"snapshots": snapshots, "count": len(snapshots)}
    except Exception:
        return {"snapshots": [], "count": 0, "note": "No exhaust data yet."}



@router.get("/capital")
async def arb_capital_allocation(request: Request):
    """Capital allocation — budget splits between triangular arb and market maker."""
    orch = getattr(request.app.state, "arb_orchestrator", None)

    # Try live orchestrator first
    if orch:
        allocator = getattr(orch, "capital_allocator", None)
        if allocator:
            try:
                summary = allocator.get_summary()
                total_usd = await allocator.get_total_usd()
                tri_budget = await allocator.get_available_budget("triangular")
                mm_budget = await allocator.get_available_budget("market_maker")
                return _sanitize_for_json({
                    "total_usd": total_usd,
                    "allocations": summary["allocations"],
                    "deployed": summary["deployed"],
                    "absolute_min_free": summary["absolute_min_free"],
                    "budgets": {
                        "triangular": tri_budget,
                        "market_maker": mm_budget,
                    },
                })
            except Exception as e:
                logger.warning(f"Capital allocator live query failed: {e}")

    # Fallback: return static allocation config
    return {
        "allocations": {"triangular": 0.40, "market_maker": 0.50, "reserve": 0.10},
        "deployed": {"triangular": 0.0, "market_maker": 0.0},
        "absolute_min_free": 50.0,
        "note": "Static config — live orchestrator not in this process. "
                "Live data visible in arb logs (Capital: line).",
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


# ─── Unified Price Feed Health Endpoint ───

price_feed_router = APIRouter(prefix="/api/price-feed", tags=["price-feed"])


@price_feed_router.get("/health")
async def price_feed_health(request: Request):
    """Return Binance unified price feed health status."""
    orch = getattr(request.app.state, "arb_orchestrator", None)
    if orch and hasattr(orch, "price_feed"):
        try:
            health = orch.price_feed.get_health()
            provider_stats = orch.ticker_provider.get_stats() if hasattr(orch, "ticker_provider") else {}
            return _sanitize_for_json({
                **health,
                "provider": provider_stats,
            })
        except Exception as e:
            return {"status": "error", "error": str(e)}
    return {"status": "unavailable", "note": "Orchestrator not in this process"}


# ─── Liquidation Feed Endpoints ───

liquidation_router = APIRouter(prefix="/api/liquidations", tags=["liquidations"])


@liquidation_router.get("")
async def liquidation_stats(request: Request):
    """Real-time liquidation data across all pairs from Binance Futures."""
    orch = getattr(request.app.state, "arb_orchestrator", None)
    feed = getattr(orch, "price_feed", None) if orch else None
    if not feed or not hasattr(feed, "get_all_liquidation_stats"):
        return {"status": "unavailable", "note": "Liquidation feed not available"}

    try:
        import time as _time
        stats = feed.get_all_liquidation_stats()

        # Sort by total liquidation volume
        sorted_stats = dict(sorted(
            stats.items(),
            key=lambda x: x[1].get("long_usd_5m", 0) + x[1].get("short_usd_5m", 0),
            reverse=True,
        ))

        return _sanitize_for_json({
            "timestamp": _time.time(),
            "feed_connected": feed._liq_connected,
            "total_orders_received": feed._liq_count,
            "pairs_with_liquidations": len(sorted_stats),
            "active_cascades": [
                sym for sym, s in sorted_stats.items()
                if s.get("cascade_active")
            ],
            "active_squeezes": [
                sym for sym, s in sorted_stats.items()
                if s.get("squeeze_active")
            ],
            "top_liquidations": dict(list(sorted_stats.items())[:20]),
        })
    except Exception as e:
        return {"status": "error", "error": str(e)}


@liquidation_router.get("/{symbol}")
async def liquidation_symbol(symbol: str, request: Request):
    """Liquidation data for a specific symbol (e.g. BTC/USDT or BTCUSDT)."""
    orch = getattr(request.app.state, "arb_orchestrator", None)
    feed = getattr(orch, "price_feed", None) if orch else None
    if not feed or not hasattr(feed, "get_liquidation_stats"):
        return {"status": "unavailable"}

    try:
        # Accept both BTC/USDT and BTCUSDT formats
        lookup = symbol.replace("_", "/").replace("-", "/")
        if "/" not in lookup:
            # Try to normalize BTCUSDT → BTC/USDT
            for quote in ("USDT", "USDC", "BTC", "ETH", "BNB"):
                if lookup.upper().endswith(quote):
                    base = lookup.upper()[:-len(quote)]
                    if base:
                        lookup = f"{base}/{quote}"
                        break

        stats = feed.get_liquidation_stats(lookup)
        if stats:
            return _sanitize_for_json(stats)
        return {"error": f"No liquidation data for {symbol} (tried {lookup})"}
    except Exception as e:
        return {"error": str(e)}
