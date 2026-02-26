"""Pre-built SQL queries returning structured results from our database."""
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from knowledge._base import PAIRS, get_db_connection, get_snapshot_db

def weekly_performance(days: int = 7, use_snapshot: bool = True) -> Dict[str, Any]:
    """Complete weekly performance summary."""
    conn = get_snapshot_db() if use_snapshot else get_db_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    result = {"period_days": days}

    try:
        row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), "
            "SUM(pnl), AVG(pnl) FROM decisions WHERE timestamp >= ?", (cutoff,)
        ).fetchone()
        total = row[0] or 0
        result.update({
            "trades": total, "wins": row[1] or 0,
            "win_rate": round((row[1] or 0) / max(total, 1), 4),
            "total_pnl": round(row[2] or 0, 2),
            "avg_pnl": round(row[3] or 0, 4),
        })

        rows = conn.execute(
            "SELECT product_id, COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), "
            "SUM(pnl), AVG(pnl) FROM decisions WHERE timestamp >= ? "
            "GROUP BY product_id ORDER BY SUM(pnl) DESC", (cutoff,)
        ).fetchall()
        result["per_pair"] = {
            r[0]: {"trades": r[1], "wins": r[2],
                   "win_rate": round(r[2]/max(r[1],1), 4),
                   "pnl": round(r[3] or 0, 2)}
            for r in rows
        }
    except Exception as e:
        result["error"] = str(e)
    finally:
        conn.close()
    return result

def model_accuracy_matrix(days: int = 7, use_snapshot: bool = True) -> pd.DataFrame:
    """Accuracy per model per pair. Rows=models, columns=pairs."""
    conn = get_snapshot_db() if use_snapshot else get_db_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        rows = conn.execute(
            "SELECT model_name, product_id, COUNT(*), "
            "SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) "
            "FROM model_predictions WHERE timestamp >= ? "
            "GROUP BY model_name, product_id", (cutoff,)
        ).fetchall()
        data = {}
        for model, pair, total, correct in rows:
            data.setdefault(model, {})[pair] = round(correct/max(total,1), 4)
        return pd.DataFrame(data).T.reindex(columns=PAIRS)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def correlation_matrix(days: int = 7, use_snapshot: bool = True) -> pd.DataFrame:
    """6x6 return correlation matrix from recent bars."""
    conn = get_snapshot_db() if use_snapshot else get_db_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        returns = {}
        for pair in PAIRS:
            rows = conn.execute(
                "SELECT timestamp, close FROM five_minute_bars "
                "WHERE product_id=? AND timestamp>=? ORDER BY timestamp",
                (pair, cutoff)
            ).fetchall()
            if rows:
                prices = pd.Series([r[1] for r in rows],
                                   index=pd.to_datetime([r[0] for r in rows]))
                returns[pair] = np.log(prices / prices.shift(1)).dropna()
        return pd.DataFrame(returns).dropna().corr()
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def devil_tracker_summary(days: int = 7, use_snapshot: bool = True) -> Dict[str, Any]:
    """Devil Tracker cost decomposition by pair and hour."""
    conn = get_snapshot_db() if use_snapshot else get_db_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    result = {}
    try:
        row = conn.execute(
            "SELECT COUNT(*), SUM(devil_cost), AVG(slippage_bps), AVG(fee_bps) "
            "FROM devil_tracker WHERE timestamp >= ?", (cutoff,)
        ).fetchone()
        result.update({
            "entries": row[0] or 0,
            "total_cost": round(row[1] or 0, 2),
            "avg_slippage_bps": round(row[2] or 0, 4),
            "avg_fee_bps": round(row[3] or 0, 4),
        })
        rows = conn.execute(
            "SELECT product_id, SUM(devil_cost), COUNT(*) FROM devil_tracker "
            "WHERE timestamp >= ? GROUP BY product_id ORDER BY SUM(devil_cost) DESC",
            (cutoff,)
        ).fetchall()
        result["by_pair"] = {r[0]: {"cost": round(r[1] or 0, 2), "n": r[2]} for r in rows}
    except Exception as e:
        result["error"] = str(e)
    finally:
        conn.close()
    return result

def regime_history(days: int = 7, use_snapshot: bool = True) -> Dict[str, Any]:
    """Regime distribution, transitions, entropy."""
    conn = get_snapshot_db() if use_snapshot else get_db_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    result = {}
    try:
        rows = conn.execute(
            "SELECT regime FROM regime_history WHERE timestamp >= ? ORDER BY timestamp",
            (cutoff,)
        ).fetchall()
        if not rows:
            return {"error": "No regime data"}
        regimes = [r[0] for r in rows]
        from collections import Counter
        counts = Counter(regimes)
        total = len(regimes)
        result["distribution"] = {k: round(v/total, 4) for k, v in counts.items()}
        transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
        result["transitions"] = transitions
        result["avg_duration_bars"] = round(total / max(transitions, 1), 1)
        result["current_regime"] = regimes[-1]
        probs = np.array([v/total for v in counts.values()])
        probs = probs[probs > 0]
        result["entropy"] = round(float(-np.sum(probs * np.log2(probs))), 4)
        result["normalized_entropy"] = round(result["entropy"] / max(np.log2(len(counts)), 0.001), 4)
    except Exception as e:
        result["error"] = str(e)
    finally:
        conn.close()
    return result
