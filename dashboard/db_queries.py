"""All SQLite read queries for the dashboard — single file, clean API."""

import json
import logging
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _resolve_db_path(db_path: str) -> str:
    p = Path(db_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    return str(p)


@contextmanager
def _conn(db_path: str):
    path = _resolve_db_path(db_path)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
    finally:
        conn.close()


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def _sanitize_floats(obj):
    """Replace NaN/Inf with None so JSON serialization succeeds."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


# ─── System ───────────────────────────────────────────────────────────────

def get_cycle_count(db_path: str) -> int:
    with _conn(db_path) as c:
        row = c.execute("SELECT COUNT(*) AS cnt FROM decisions").fetchone()
        return row["cnt"] if row else 0


def get_trade_count(db_path: str) -> int:
    with _conn(db_path) as c:
        row = c.execute("SELECT COUNT(*) AS cnt FROM trades").fetchone()
        return row["cnt"] if row else 0


# ─── Decisions / Signals ──────────────────────────────────────────────────

def get_recent_decisions(db_path: str, limit: int = 100) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        results = _rows_to_dicts(rows)
        for r in results:
            if r.get("reasoning"):
                try:
                    r["reasoning"] = _sanitize_floats(json.loads(r["reasoning"]))
                except (json.JSONDecodeError, TypeError):
                    pass
        return results


def get_decision_by_id(db_path: str, decision_id: int) -> Optional[Dict[str, Any]]:
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        if result.get("reasoning"):
            try:
                result["reasoning"] = _sanitize_floats(json.loads(result["reasoning"]))
            except (json.JSONDecodeError, TypeError):
                pass
        return result


# ─── Trades & Positions ──────────────────────────────────────────────────

def get_open_positions(db_path: str) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM open_positions WHERE status = 'OPEN'"
        ).fetchall()
        return _rows_to_dicts(rows)


def get_closed_trades(
    db_path: str, limit: int = 50, offset: int = 0
) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return _rows_to_dicts(rows)


def get_trade_by_id(db_path: str, trade_id: int) -> Optional[Dict[str, Any]]:
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        return dict(row) if row else None


def get_trade_lifecycle(db_path: str, trade_id: int) -> Dict[str, Any]:
    trade = get_trade_by_id(db_path, trade_id)
    if not trade:
        return {"error": "Trade not found"}
    # Find surrounding decisions for context
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT * FROM decisions
               WHERE product_id = ? AND timestamp <= ?
               ORDER BY id DESC LIMIT 5""",
            (trade.get("product_id", "BTC-USD"), trade.get("timestamp", "")),
        ).fetchall()
        decisions = _rows_to_dicts(rows)
        for d in decisions:
            if d.get("reasoning"):
                try:
                    d["reasoning"] = json.loads(d["reasoning"])
                except (json.JSONDecodeError, TypeError):
                    pass
    return {"trade": trade, "context_decisions": decisions}


# ─── Analytics ────────────────────────────────────────────────────────────

def get_equity_curve(db_path: str, hours: int = 24) -> List[Dict[str, Any]]:
    """Approximate equity curve from trades — cumulative PnL over time."""
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT timestamp, side, size, price,
                      CASE WHEN side='SELL' THEN size*price
                           WHEN side='BUY' THEN -size*price
                           ELSE 0 END AS pnl_delta
               FROM trades
               WHERE datetime(timestamp) > datetime('now', ? || ' hours')
               ORDER BY timestamp ASC""",
            (f"-{hours}",),
        ).fetchall()
        data = _rows_to_dicts(rows)
        cumulative = 0.0
        for row in data:
            cumulative += row.get("pnl_delta", 0.0)
            row["cumulative_pnl"] = cumulative
        return data


def get_pnl_summary(db_path: str, hours: int = 24) -> Dict[str, Any]:
    with _conn(db_path) as c:
        row = c.execute(
            """SELECT
                 COUNT(*) as total_trades,
                 COALESCE(SUM(CASE WHEN side='SELL' THEN size*price
                                   WHEN side='BUY' THEN -size*price
                                   ELSE 0 END), 0.0) as realized_pnl,
                 COALESCE(AVG(slippage), 0.0) as avg_slippage
               FROM trades
               WHERE datetime(timestamp) > datetime('now', ? || ' hours')""",
            (f"-{hours}",),
        ).fetchone()
        total = row["total_trades"] or 0

        # Win rate (approximate: SELL trades with positive value)
        wins_row = c.execute(
            """SELECT COUNT(*) as wins FROM trades
               WHERE side='SELL'
                 AND size*price > 0
                 AND datetime(timestamp) > datetime('now', ? || ' hours')""",
            (f"-{hours}",),
        ).fetchone()
        wins = wins_row["wins"] if wins_row else 0
        sells_row = c.execute(
            """SELECT COUNT(*) as cnt FROM trades
               WHERE side='SELL'
                 AND datetime(timestamp) > datetime('now', ? || ' hours')""",
            (f"-{hours}",),
        ).fetchone()
        sells = sells_row["cnt"] if sells_row else 0

        return {
            "total_trades": total,
            "realized_pnl": round(row["realized_pnl"], 2) if row["realized_pnl"] else 0.0,
            "unrealized_pnl": 0.0,  # Would need live price
            "avg_slippage": round(row["avg_slippage"], 6) if row["avg_slippage"] else 0.0,
            "win_rate": round(wins / sells, 4) if sells > 0 else 0.0,
            "total_sells": sells,
            "total_wins": wins,
        }


def get_performance_by_regime(db_path: str) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT
                 d.hmm_regime as regime,
                 d.action,
                 COUNT(*) as count,
                 AVG(d.confidence) as avg_confidence,
                 AVG(d.weighted_signal) as avg_signal
               FROM decisions d
               WHERE d.hmm_regime IS NOT NULL
               GROUP BY d.hmm_regime, d.action
               ORDER BY d.hmm_regime, count DESC"""
        ).fetchall()
        return _rows_to_dicts(rows)


def get_performance_by_execution(db_path: str) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT
                 algo_used,
                 COUNT(*) as count,
                 AVG(slippage) as avg_slippage,
                 AVG(execution_time) as avg_exec_time
               FROM trades
               WHERE algo_used IS NOT NULL
               GROUP BY algo_used
               ORDER BY count DESC"""
        ).fetchall()
        return _rows_to_dicts(rows)


def get_return_distribution(db_path: str) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT
                 CASE WHEN side='SELL' THEN size*price
                      WHEN side='BUY' THEN -size*price
                      ELSE 0 END AS trade_pnl
               FROM trades
               ORDER BY id DESC
               LIMIT 500"""
        ).fetchall()
        return _rows_to_dicts(rows)


def get_calendar_pnl(db_path: str) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT
                 date(timestamp) as date,
                 SUM(CASE WHEN side='SELL' THEN size*price
                          WHEN side='BUY' THEN -size*price
                          ELSE 0 END) as daily_pnl,
                 COUNT(*) as trade_count
               FROM trades
               GROUP BY date(timestamp)
               ORDER BY date DESC
               LIMIT 90"""
        ).fetchall()
        return _rows_to_dicts(rows)


def get_hourly_pnl(db_path: str) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT
                 CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                 AVG(CASE WHEN side='SELL' THEN size*price
                          WHEN side='BUY' THEN -size*price
                          ELSE 0 END) as avg_pnl,
                 COUNT(*) as trade_count
               FROM trades
               GROUP BY strftime('%H', timestamp)
               ORDER BY hour"""
        ).fetchall()
        return _rows_to_dicts(rows)


# ─── Brain (ML & Regime) ─────────────────────────────────────────────────

def get_ml_predictions_history(
    db_path: str, hours: int = 24
) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT * FROM ml_predictions
               WHERE datetime(timestamp) > datetime('now', ? || ' hours')
               ORDER BY timestamp DESC""",
            (f"-{hours}",),
        ).fetchall()
        return _rows_to_dicts(rows)


def get_regime_history(db_path: str, limit: int = 200) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT id, timestamp, hmm_regime, confidence, weighted_signal
               FROM decisions
               WHERE hmm_regime IS NOT NULL
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return _rows_to_dicts(rows)


def get_vae_history(db_path: str, limit: int = 200) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT id, timestamp, vae_loss
               FROM decisions
               WHERE vae_loss IS NOT NULL
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return _rows_to_dicts(rows)


# ─── Risk ─────────────────────────────────────────────────────────────────

def get_risk_metrics(db_path: str) -> Dict[str, Any]:
    """Compute risk metrics from trade history."""
    with _conn(db_path) as c:
        # Daily PnL for drawdown calc
        rows = c.execute(
            """SELECT date(timestamp) as dt,
                      SUM(CASE WHEN side='SELL' THEN size*price
                               WHEN side='BUY' THEN -size*price
                               ELSE 0 END) as daily_pnl
               FROM trades
               GROUP BY date(timestamp)
               ORDER BY dt ASC"""
        ).fetchall()

        daily_pnls = [r["daily_pnl"] for r in rows if r["daily_pnl"] is not None]
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        consecutive_losses = 0
        max_consec_losses = 0

        for pnl in daily_pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / max(peak, 1.0) if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            if pnl < 0:
                consecutive_losses += 1
                max_consec_losses = max(max_consec_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        return {
            "max_drawdown": round(max_dd, 4),
            "cumulative_pnl": round(cumulative, 2),
            "peak_equity": round(peak, 2),
            "max_consecutive_losses": max_consec_losses,
            "total_trading_days": len(daily_pnls),
        }


def get_risk_gateway_log(db_path: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Decisions with VAE loss (risk gateway pass/block proxy)."""
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT id, timestamp, product_id, action, confidence, vae_loss, hmm_regime
               FROM decisions
               WHERE vae_loss IS NOT NULL
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return _rows_to_dicts(rows)


def get_exposure(db_path: str) -> Dict[str, Any]:
    positions = get_open_positions(db_path)
    long_exp = sum(p["size"] * p["entry_price"] for p in positions if p["side"] == "BUY")
    short_exp = sum(p["size"] * p["entry_price"] for p in positions if p["side"] == "SELL")
    return {
        "long_exposure": round(long_exp, 2),
        "short_exposure": round(short_exp, 2),
        "net_exposure": round(long_exp - short_exp, 2),
        "gross_exposure": round(long_exp + short_exp, 2),
        "position_count": len(positions),
        "positions_by_asset": _group_positions_by_asset(positions),
    }


def _group_positions_by_asset(positions: List[Dict]) -> Dict[str, Any]:
    groups: Dict[str, Any] = {}
    for p in positions:
        pid = p.get("product_id", "UNKNOWN")
        if pid not in groups:
            groups[pid] = {"count": 0, "total_size": 0.0, "total_value": 0.0}
        groups[pid]["count"] += 1
        groups[pid]["total_size"] += p.get("size", 0.0)
        groups[pid]["total_value"] += p.get("size", 0.0) * p.get("entry_price", 0.0)
    return groups


# ─── Backtest ─────────────────────────────────────────────────────────────

def ensure_backtest_table(db_path: str) -> None:
    """Create the backtest_runs table if it does not already exist."""
    with _conn(db_path) as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS backtest_runs (
                   id               INTEGER PRIMARY KEY AUTOINCREMENT,
                   timestamp        TEXT    NOT NULL DEFAULT (datetime('now')),
                   config_json      TEXT,
                   total_trades     INTEGER DEFAULT 0,
                   realized_pnl     REAL    DEFAULT 0,
                   sharpe_ratio     REAL    DEFAULT 0,
                   max_drawdown     REAL    DEFAULT 0,
                   win_rate         REAL    DEFAULT 0,
                   duration_seconds REAL    DEFAULT 0,
                   notes            TEXT
               )"""
        )
        c.commit()


def get_backtest_runs(db_path: str) -> List[Dict[str, Any]]:
    """Return recent backtest runs (creates table on first call)."""
    ensure_backtest_table(db_path)
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM backtest_runs ORDER BY id DESC LIMIT 50"
        ).fetchall()
        return _rows_to_dicts(rows)


def get_backtest_result(db_path: str, run_id: int) -> Optional[Dict[str, Any]]:
    """Return a single backtest run by id (creates table on first call)."""
    ensure_backtest_table(db_path)
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT * FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None


# ─── Market Data ──────────────────────────────────────────────────────────

def get_recent_market_data(
    db_path: str, product_id: str = "BTC-USD", limit: int = 100
) -> List[Dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT * FROM market_data
               WHERE product_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (product_id, limit),
        ).fetchall()
        return _rows_to_dicts(rows)


def get_latest_price(db_path: str, product_id: str = "BTC-USD") -> Optional[Dict[str, Any]]:
    with _conn(db_path) as c:
        row = c.execute(
            """SELECT * FROM market_data
               WHERE product_id = ?
               ORDER BY id DESC
               LIMIT 1""",
            (product_id,),
        ).fetchone()
        return dict(row) if row else None
