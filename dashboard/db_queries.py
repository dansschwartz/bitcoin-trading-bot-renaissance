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

# Starting capital for paper trading (matches PaperTradingSimulator default)
INITIAL_CAPITAL = 10_000.0

# HMM regime numeric → human name mapping
REGIME_NAMES = {
    "0": "low_volatility",
    "1": "trending",
    "2": "high_volatility",
    0: "low_volatility",
    1: "trending",
    2: "high_volatility",
}


def _map_regime(raw_regime) -> str:
    """Map numeric HMM regime to human-readable name."""
    if raw_regime is None:
        return "Unknown"
    mapped = REGIME_NAMES.get(raw_regime)
    if mapped:
        return mapped
    # Already a string name?
    s = str(raw_regime).strip()
    mapped = REGIME_NAMES.get(s)
    return mapped if mapped else s if s else "Unknown"


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
    """Equity curve from trades — cumulative realized P&L anchored to starting capital."""
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT timestamp, side, size, price
               FROM trades
               WHERE datetime(timestamp) > datetime('now', ? || ' hours')
               ORDER BY timestamp ASC""",
            (f"-{hours}",),
        ).fetchall()
        data = _rows_to_dicts(rows)

        # Compute realized P&L by pairing BUY/SELL trades per product
        cumulative_pnl = 0.0
        for row in data:
            side = row.get("side", "")
            size = row.get("size", 0.0)
            price = row.get("price", 0.0)
            if side == "SELL":
                cumulative_pnl += size * price
            elif side == "BUY":
                cumulative_pnl -= size * price
            row["pnl_delta"] = cumulative_pnl - (data[data.index(row) - 1].get("cumulative_pnl", 0.0) if data.index(row) > 0 else 0.0)
            row["cumulative_pnl"] = round(cumulative_pnl, 2)
            row["equity"] = round(INITIAL_CAPITAL + cumulative_pnl, 2)

        # Append current unrealized P&L as final point
        unrealized = _compute_total_unrealized_pnl(c)
        if data:
            last = data[-1]
            total_equity = INITIAL_CAPITAL + cumulative_pnl + unrealized
            data.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "side": "MARK",
                "size": 0,
                "price": 0,
                "pnl_delta": unrealized,
                "cumulative_pnl": round(cumulative_pnl + unrealized, 2),
                "equity": round(total_equity, 2),
            })
        elif unrealized != 0.0:
            data.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "side": "MARK",
                "size": 0,
                "price": 0,
                "pnl_delta": unrealized,
                "cumulative_pnl": round(unrealized, 2),
                "equity": round(INITIAL_CAPITAL + unrealized, 2),
            })

        return data


def _is_long_side(side: str) -> bool:
    """Check if position side is long (BUY or LONG)."""
    return side.upper() in ("BUY", "LONG")


def _compute_total_unrealized_pnl(conn) -> float:
    """Compute total unrealized P&L from open positions using latest market prices."""
    positions = conn.execute(
        "SELECT * FROM open_positions WHERE status = 'OPEN'"
    ).fetchall()
    total = 0.0
    price_cache: dict = {}
    for p in positions:
        p = dict(p)
        pid = p.get("product_id", "BTC-USD")
        entry_price = p.get("entry_price", 0.0)
        size = p.get("size", 0.0)
        side = p.get("side", "BUY")
        # Get latest price for this product (cached)
        if pid not in price_cache:
            lp = conn.execute(
                "SELECT price FROM market_data WHERE product_id = ? ORDER BY id DESC LIMIT 1",
                (pid,),
            ).fetchone()
            price_cache[pid] = lp["price"] if lp else entry_price
        current_price = price_cache[pid]
        if _is_long_side(side):
            total += (current_price - entry_price) * size
        else:
            total += (entry_price - current_price) * size
    return round(total, 2)


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

        # Compute unrealized P&L from open positions
        unrealized = _compute_total_unrealized_pnl(c)

        # Win rate: pair BUY→SELL round-trips per product and check if profitable.
        # A "round trip" is a BUY followed by a SELL on the same product.
        # Fall back to counting positions with positive unrealized P&L if no round trips.
        round_trips = _compute_round_trip_stats(c, hours)

        return {
            "total_trades": total,
            "realized_pnl": round(row["realized_pnl"], 2) if row["realized_pnl"] else 0.0,
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round((row["realized_pnl"] or 0.0) + unrealized, 2),
            "avg_slippage": round(row["avg_slippage"], 6) if row["avg_slippage"] else 0.0,
            "win_rate": round(round_trips["win_rate"], 4),
            "total_round_trips": round_trips["total"],
            "winning_round_trips": round_trips["wins"],
            "initial_capital": INITIAL_CAPITAL,
            "current_equity": round(INITIAL_CAPITAL + (row["realized_pnl"] or 0.0) + unrealized, 2),
        }


def _compute_round_trip_stats(conn, hours: int = 24) -> Dict[str, Any]:
    """Pair BUY→SELL trades per product to compute actual win rate."""
    rows = conn.execute(
        """SELECT product_id, side, size, price, timestamp
           FROM trades
           WHERE datetime(timestamp) > datetime('now', ? || ' hours')
           ORDER BY timestamp ASC""",
        (f"-{hours}",),
    ).fetchall()

    # Track cost basis per product using FIFO
    buys: Dict[str, list] = {}  # product_id -> [(size, price), ...]
    wins = 0
    losses = 0

    for r in rows:
        r = dict(r)
        pid = r["product_id"]
        side = r["side"]
        size = r["size"]
        price = r["price"]

        if side == "BUY":
            buys.setdefault(pid, []).append((size, price))
        elif side == "SELL" and pid in buys and buys[pid]:
            # Match against oldest buy (FIFO)
            remaining = size
            cost_basis = 0.0
            while remaining > 0 and buys[pid]:
                buy_size, buy_price = buys[pid][0]
                matched = min(remaining, buy_size)
                cost_basis += matched * buy_price
                remaining -= matched
                if matched >= buy_size:
                    buys[pid].pop(0)
                else:
                    buys[pid][0] = (buy_size - matched, buy_price)
            sell_proceeds = (size - remaining) * price
            if sell_proceeds > cost_basis:
                wins += 1
            else:
                losses += 1

    total = wins + losses
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total if total > 0 else 0.0,
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
        results = _rows_to_dicts(rows)
        for r in results:
            r["regime"] = _map_regime(r.get("regime"))
        return results


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
        results = _rows_to_dicts(rows)
        for r in results:
            r["hmm_regime"] = _map_regime(r.get("hmm_regime"))
        return results


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
    """Compute risk metrics from trade history, anchored to initial capital."""
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

        # Equity starts at INITIAL_CAPITAL, not zero
        equity = INITIAL_CAPITAL
        peak = INITIAL_CAPITAL
        max_dd = 0.0
        consecutive_losses = 0
        max_consec_losses = 0
        daily_returns = []

        for pnl in daily_pnls:
            prev_equity = equity
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            if pnl < 0:
                consecutive_losses += 1
                max_consec_losses = max(max_consec_losses, consecutive_losses)
            else:
                consecutive_losses = 0
            # Daily return as percentage
            if prev_equity > 0:
                daily_returns.append(pnl / prev_equity)

        # Add unrealized P&L to current equity
        unrealized = _compute_total_unrealized_pnl(c)
        current_equity = equity + unrealized

        # Sharpe ratio (annualized, assuming 365 trading days for crypto)
        sharpe = 0.0
        if len(daily_returns) >= 2:
            import statistics
            mean_r = statistics.mean(daily_returns)
            std_r = statistics.stdev(daily_returns)
            if std_r > 0:
                sharpe = round((mean_r / std_r) * (365 ** 0.5), 2)

        cumulative_pnl = equity - INITIAL_CAPITAL + unrealized

        return {
            "max_drawdown": round(max_dd, 4),
            "cumulative_pnl": round(cumulative_pnl, 2),
            "peak_equity": round(peak, 2),
            "current_equity": round(current_equity, 2),
            "initial_capital": INITIAL_CAPITAL,
            "sharpe_ratio": sharpe,
            "max_consecutive_losses": max_consec_losses,
            "total_trading_days": len(daily_pnls),
            "unrealized_pnl": round(unrealized, 2),
        }


def get_risk_gateway_log(db_path: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Decisions with risk gateway context — VAE loss + regime + action taken."""
    with _conn(db_path) as c:
        # Include all trade decisions (BUY/SELL), not just those with vae_loss
        rows = c.execute(
            """SELECT id, timestamp, product_id, action, confidence, vae_loss, hmm_regime
               FROM decisions
               WHERE action IN ('BUY', 'SELL')
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        results = _rows_to_dicts(rows)
        for r in results:
            r["hmm_regime"] = _map_regime(r.get("hmm_regime"))
            # Determine gateway verdict: PASS if trade was executed, BLOCK if not
            # vae_loss being high (> 0.3) means gateway would have blocked
            vae = r.get("vae_loss")
            if vae is not None:
                r["gateway_verdict"] = "PASS" if vae < 0.3 else "BLOCK"
            else:
                r["gateway_verdict"] = "PASS"  # No VAE = no gateway = allowed through
                r["vae_loss"] = 0.0  # Default for display
        return results


def get_exposure(db_path: str) -> Dict[str, Any]:
    """Compute exposure using current market prices (not entry prices)."""
    with _conn(db_path) as c:
        positions = c.execute(
            "SELECT * FROM open_positions WHERE status = 'OPEN'"
        ).fetchall()
        positions = [dict(p) for p in positions]

        # Enrich with current prices
        price_cache: Dict[str, float] = {}
        for p in positions:
            pid = p.get("product_id", "BTC-USD")
            if pid not in price_cache:
                lp = c.execute(
                    "SELECT price FROM market_data WHERE product_id = ? ORDER BY id DESC LIMIT 1",
                    (pid,),
                ).fetchone()
                price_cache[pid] = lp["price"] if lp else p.get("entry_price", 0.0)
            p["current_price"] = price_cache[pid]

        long_exp = sum(
            p["size"] * p["current_price"] for p in positions if _is_long_side(p.get("side", ""))
        )
        short_exp = sum(
            p["size"] * p["current_price"] for p in positions if not _is_long_side(p.get("side", ""))
        )
        return {
            "long_exposure": round(long_exp, 2),
            "short_exposure": round(short_exp, 2),
            "net_exposure": round(long_exp - short_exp, 2),
            "gross_exposure": round(long_exp + short_exp, 2),
            "position_count": len(positions),
            "positions_by_asset": _group_positions_by_asset(positions, price_cache),
        }


def _group_positions_by_asset(
    positions: List[Dict], price_cache: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Group positions by asset and compute netted exposure."""
    groups: Dict[str, Any] = {}
    for p in positions:
        pid = p.get("product_id", "UNKNOWN")
        if pid not in groups:
            groups[pid] = {
                "count": 0,
                "long_size": 0.0,
                "short_size": 0.0,
                "long_value": 0.0,
                "short_value": 0.0,
            }
        g = groups[pid]
        g["count"] += 1
        size = p.get("size", 0.0)
        current_price = (
            price_cache.get(pid, p.get("entry_price", 0.0)) if price_cache else p.get("entry_price", 0.0)
        )
        if _is_long_side(p.get("side", "")):
            g["long_size"] += size
            g["long_value"] += size * current_price
        else:
            g["short_size"] += size
            g["short_value"] += size * current_price

    # Add computed net fields
    for g in groups.values():
        g["net_size"] = round(g["long_size"] - g["short_size"], 8)
        g["net_value"] = round(g["long_value"] - g["short_value"], 2)
        g["total_size"] = round(g["long_size"] + g["short_size"], 8)
        g["total_value"] = round(g["long_value"] + g["short_value"], 2)
    return groups


def evaluate_risk_alerts(db_path: str, thresholds: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Generate risk alerts from current metrics and thresholds."""
    if thresholds is None:
        thresholds = {
            "pnl_threshold": -200,
            "drawdown_threshold": 0.05,
            "consecutive_loss_threshold": 5,
        }

    alerts: List[Dict[str, Any]] = []
    metrics = get_risk_metrics(db_path)
    now = datetime.now(timezone.utc).isoformat()

    # Drawdown alert
    if metrics["max_drawdown"] > thresholds.get("drawdown_threshold", 0.05):
        severity = "CRITICAL" if metrics["max_drawdown"] > 0.10 else "WARNING"
        alerts.append({
            "type": "drawdown",
            "severity": severity,
            "message": f"Max drawdown {metrics['max_drawdown']:.2%} exceeds {thresholds['drawdown_threshold']:.0%} threshold",
            "timestamp": now,
            "value": metrics["max_drawdown"],
        })

    # P&L alert
    if metrics["cumulative_pnl"] < thresholds.get("pnl_threshold", -200):
        alerts.append({
            "type": "pnl",
            "severity": "CRITICAL",
            "message": f"Cumulative P&L ${metrics['cumulative_pnl']:.2f} below ${thresholds['pnl_threshold']} threshold",
            "timestamp": now,
            "value": metrics["cumulative_pnl"],
        })

    # Consecutive losses alert
    if metrics["max_consecutive_losses"] >= thresholds.get("consecutive_loss_threshold", 5):
        alerts.append({
            "type": "consecutive_losses",
            "severity": "WARNING",
            "message": f"{metrics['max_consecutive_losses']} consecutive losing days (threshold: {thresholds['consecutive_loss_threshold']})",
            "timestamp": now,
            "value": metrics["max_consecutive_losses"],
        })

    # Exposure concentration alert
    try:
        exposure = get_exposure(db_path)
        if exposure["gross_exposure"] > 0:
            net_to_gross = abs(exposure["net_exposure"]) / exposure["gross_exposure"]
            if net_to_gross > 0.8:
                alerts.append({
                    "type": "exposure_concentration",
                    "severity": "WARNING",
                    "message": f"Net/Gross exposure ratio {net_to_gross:.0%} — portfolio is heavily directional",
                    "timestamp": now,
                    "value": net_to_gross,
                })
    except Exception:
        pass

    return alerts


def get_benchmark_equity(db_path: str, hours: int = 24, product_id: str = "BTC-USD") -> List[Dict[str, Any]]:
    """Compute buy-and-hold benchmark: what if we'd invested INITIAL_CAPITAL in BTC at the start?"""
    with _conn(db_path) as c:
        rows = c.execute(
            """SELECT timestamp, price FROM market_data
               WHERE product_id = ?
                 AND datetime(timestamp) > datetime('now', ? || ' hours')
               ORDER BY timestamp ASC""",
            (product_id, f"-{hours}"),
        ).fetchall()
        if not rows:
            return []
        data = _rows_to_dicts(rows)
        first_price = data[0]["price"]
        if first_price <= 0:
            return []
        # Sample every Nth point to keep data reasonable (max ~200 points)
        step = max(1, len(data) // 200)
        result = []
        for i in range(0, len(data), step):
            d = data[i]
            benchmark_equity = INITIAL_CAPITAL * (d["price"] / first_price)
            result.append({
                "timestamp": d["timestamp"],
                "benchmark_equity": round(benchmark_equity, 2),
            })
        # Always include the last point
        if len(data) > 0 and (len(data) - 1) % step != 0:
            d = data[-1]
            result.append({
                "timestamp": d["timestamp"],
                "benchmark_equity": round(INITIAL_CAPITAL * (d["price"] / first_price), 2),
            })
        return result


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
