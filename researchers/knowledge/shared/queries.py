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

def _audit_table_exists(conn) -> bool:
    """Check if decision_audit_log table exists."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='decision_audit_log'"
    ).fetchone()
    return row is not None


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


# ═══════════════════════════════════════════════════════════════
# DECISION AUDIT LOG QUERIES
# ═══════════════════════════════════════════════════════════════

def audit_model_accuracy(db_path: str, days: int = 7) -> Dict[str, Any]:
    """Per-model live accuracy from evaluated predictions.

    Returns dict with ensemble_accuracy, ensemble_n, per_model list,
    and per_model_by_regime cross-tabulation.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    result: Dict[str, Any] = {}

    try:
        if not _audit_table_exists(conn):
            return {"error": "decision_audit_log table not found"}

        # Ensemble accuracy from audit log (correct = outcome same sign as weighted_signal)
        row = conn.execute("""
            SELECT COUNT(*) as n,
                   SUM(CASE WHEN (outcome_6bar > 0 AND weighted_signal > 0)
                             OR (outcome_6bar < 0 AND weighted_signal < 0) THEN 1 ELSE 0 END) as correct
            FROM decision_audit_log
            WHERE outcome_6bar IS NOT NULL
              AND timestamp >= datetime('now', ? || ' days')
        """, (f"-{days}",)).fetchone()
        n = row['n'] or 0
        correct = row['correct'] or 0
        result['ensemble_accuracy'] = round(correct / max(n, 1), 4)
        result['ensemble_n'] = n

        # Per-model from ml_predictions
        rows = conn.execute("""
            SELECT model_name, COUNT(*) as n,
                   SUM(is_correct) as correct,
                   ROUND(1.0 * SUM(is_correct) / COUNT(*), 4) as accuracy
            FROM ml_predictions
            WHERE is_correct IS NOT NULL
              AND timestamp >= datetime('now', ? || ' days')
            GROUP BY model_name
            ORDER BY accuracy DESC
        """, (f"-{days}",)).fetchall()
        result['per_model'] = [dict(r) for r in rows]

        # Per-model by regime (join audit log for regime info)
        rows = conn.execute("""
            SELECT mp.model_name, dal.regime_label as regime,
                   COUNT(*) as n,
                   ROUND(1.0 * SUM(mp.is_correct) / COUNT(*), 4) as accuracy
            FROM ml_predictions mp
            JOIN decision_audit_log dal
              ON mp.product_id = dal.product_id
              AND ABS(JULIANDAY(mp.timestamp) - JULIANDAY(dal.timestamp)) < 0.001
            WHERE mp.is_correct IS NOT NULL
              AND mp.timestamp >= datetime('now', ? || ' days')
            GROUP BY mp.model_name, dal.regime_label
            HAVING COUNT(*) >= 20
            ORDER BY mp.model_name, accuracy DESC
        """, (f"-{days}",)).fetchall()
        result['per_model_by_regime'] = [dict(r) for r in rows]

    except Exception as e:
        result["error"] = str(e)
    finally:
        conn.close()
    return result


def audit_signal_effectiveness(db_path: str, days: int = 7) -> Dict[str, Any]:
    """Which signals predict outcomes?

    Per-signal average value when outcome was positive vs negative.
    Signals with large separation have predictive power.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    result: Dict[str, Any] = {'signals': []}

    try:
        if not _audit_table_exists(conn):
            return {"error": "decision_audit_log table not found"}

        signal_cols = [
            'sig_macd', 'sig_rsi', 'sig_bollinger',
            'sig_order_flow', 'sig_order_book', 'sig_volume',
            'sig_volume_profile', 'sig_lead_lag', 'sig_stat_arb',
            'sig_entropy', 'sig_ml_ensemble', 'sig_garch_vol',
            'sig_fractal', 'sig_correlation_divergence',
        ]

        for col in signal_cols:
            row = conn.execute(f"""
                SELECT
                    '{col}' as signal_name,
                    AVG(CASE WHEN outcome_6bar > 0 THEN {col} END) as avg_when_up,
                    AVG(CASE WHEN outcome_6bar < 0 THEN {col} END) as avg_when_down,
                    AVG(CASE WHEN outcome_6bar > 0 THEN {col} END) -
                    AVG(CASE WHEN outcome_6bar < 0 THEN {col} END) as separation,
                    COUNT(*) as n
                FROM decision_audit_log
                WHERE outcome_6bar IS NOT NULL
                  AND {col} IS NOT NULL
                  AND timestamp >= datetime('now', ? || ' days')
            """, (f"-{days}",)).fetchone()
            if row and (row['n'] or 0) > 50:
                result['signals'].append(dict(row))

        result['signals'].sort(
            key=lambda x: abs(x.get('separation', 0) or 0), reverse=True
        )
    except Exception as e:
        result["error"] = str(e)
    finally:
        conn.close()
    return result


def audit_regime_performance(db_path: str, days: int = 14) -> Dict[str, Any]:
    """P&L and accuracy breakdown by regime.

    Shows decisions, trade rate, accuracy, avg return, confidence,
    and sizing multipliers per regime.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    result: Dict[str, Any] = {}

    try:
        if not _audit_table_exists(conn):
            return {"error": "decision_audit_log table not found"}

        rows = conn.execute("""
            SELECT
                regime_label as regime,
                COUNT(*) as decisions,
                SUM(CASE WHEN final_action != 'HOLD' THEN 1 ELSE 0 END) as trades,
                SUM(CASE WHEN outcome_6bar IS NOT NULL AND (
                    (outcome_6bar > 0 AND weighted_signal > 0) OR
                    (outcome_6bar < 0 AND weighted_signal < 0)
                ) THEN 1 ELSE 0 END) as correct,
                ROUND(AVG(outcome_6bar) * 10000, 2) as avg_return_bps,
                ROUND(AVG(final_confidence), 4) as avg_confidence,
                ROUND(AVG(effective_edge) * 10000, 2) as avg_edge_bps,
                ROUND(AVG(market_impact_bps), 2) as avg_impact_bps
            FROM decision_audit_log
            WHERE timestamp >= datetime('now', ? || ' days')
            GROUP BY regime_label
            ORDER BY decisions DESC
        """, (f"-{days}",)).fetchall()
        result['by_regime'] = [dict(r) for r in rows]

    except Exception as e:
        result["error"] = str(e)
    finally:
        conn.close()
    return result


def audit_sizing_chain_analysis(db_path: str, days: int = 7) -> Dict[str, Any]:
    """How is the sizing chain affecting trade size?

    Shows average Kelly fraction, applied fraction, position size,
    and per-pair breakdown for non-HOLD decisions.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    result: Dict[str, Any] = {}

    try:
        if not _audit_table_exists(conn):
            return {"error": "decision_audit_log table not found"}

        row = conn.execute("""
            SELECT
                ROUND(AVG(kelly_fraction), 4) as avg_kelly_f,
                ROUND(AVG(applied_fraction), 4) as avg_applied_f,
                ROUND(AVG(position_usd), 2) as avg_size_usd,
                ROUND(AVG(market_impact_bps), 2) as avg_impact_bps,
                ROUND(AVG(edge), 6) as avg_edge,
                ROUND(AVG(effective_edge), 6) as avg_effective_edge,
                COUNT(*) as n
            FROM decision_audit_log
            WHERE final_action != 'HOLD'
              AND timestamp >= datetime('now', ? || ' days')
        """, (f"-{days}",)).fetchone()
        result['overall'] = dict(row) if row else {}

        # Per-pair breakdown
        rows = conn.execute("""
            SELECT
                product_id,
                ROUND(AVG(position_usd), 2) as avg_size_usd,
                ROUND(AVG(market_impact_bps), 2) as avg_impact_bps,
                ROUND(AVG(kelly_fraction), 4) as avg_kelly_f,
                COUNT(*) as n
            FROM decision_audit_log
            WHERE final_action != 'HOLD'
              AND timestamp >= datetime('now', ? || ' days')
            GROUP BY product_id
            ORDER BY avg_size_usd DESC
        """, (f"-{days}",)).fetchall()
        result['by_pair'] = [dict(r) for r in rows]

    except Exception as e:
        result["error"] = str(e)
    finally:
        conn.close()
    return result


def audit_cost_vs_edge(db_path: str, days: int = 7) -> Dict[str, Any]:
    """Is the Devil eating the edge?

    Compares predicted edge vs actual return for traded decisions.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    result: Dict[str, Any] = {}

    try:
        if not _audit_table_exists(conn):
            return {"error": "decision_audit_log table not found"}

        rows = conn.execute("""
            SELECT
                product_id,
                COUNT(*) as n,
                ROUND(AVG(edge) * 10000, 2) as avg_predicted_edge_bps,
                ROUND(AVG(effective_edge) * 10000, 2) as avg_effective_edge_bps,
                ROUND(AVG(market_impact_bps), 2) as avg_impact_bps,
                ROUND(AVG(outcome_6bar) * 10000, 2) as avg_actual_return_bps,
                ROUND(AVG(effective_edge * 10000) - AVG(outcome_6bar * 10000), 2)
                    as edge_vs_reality_gap_bps
            FROM decision_audit_log
            WHERE outcome_6bar IS NOT NULL
              AND final_action != 'HOLD'
              AND timestamp >= datetime('now', ? || ' days')
            GROUP BY product_id
            ORDER BY edge_vs_reality_gap_bps DESC
        """, (f"-{days}",)).fetchall()
        result['by_pair'] = [dict(r) for r in rows]

    except Exception as e:
        result["error"] = str(e)
    finally:
        conn.close()
    return result


def audit_confluence_effectiveness(db_path: str, days: int = 14) -> Dict[str, Any]:
    """Does the confluence boost actually help?

    Compares accuracy and returns for boosted vs unboosted trades.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    result: Dict[str, Any] = {}

    try:
        if not _audit_table_exists(conn):
            return {"error": "decision_audit_log table not found"}

        rows = conn.execute("""
            SELECT
                CASE WHEN confluence_boost > 0 THEN 'boosted' ELSE 'unboosted' END as group_name,
                COUNT(*) as n,
                ROUND(AVG(outcome_6bar) * 10000, 2) as avg_return_bps,
                ROUND(AVG(final_confidence), 4) as avg_confidence
            FROM decision_audit_log
            WHERE outcome_6bar IS NOT NULL
              AND final_action != 'HOLD'
              AND timestamp >= datetime('now', ? || ' days')
            GROUP BY group_name
        """, (f"-{days}",)).fetchall()
        result['comparison'] = [dict(r) for r in rows]

    except Exception as e:
        result["error"] = str(e)
    finally:
        conn.close()
    return result


def audit_feature_health(db_path: str, days: int = 3) -> Dict[str, Any]:
    """Feature vector quality and model activity over time.

    Shows daily decision counts, average active models, and
    feature vector hash diversity (unique hashes = diverse inputs).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    result: Dict[str, Any] = {}

    try:
        if not _audit_table_exists(conn):
            return {"error": "decision_audit_log table not found"}

        rows = conn.execute("""
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as decisions,
                ROUND(AVG(ml_model_count), 1) as avg_models_active,
                COUNT(DISTINCT feature_vector_hash) as unique_feature_hashes,
                ROUND(AVG(raw_signal_count), 1) as avg_signals_active
            FROM decision_audit_log
            WHERE timestamp >= datetime('now', ? || ' days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """, (f"-{days}",)).fetchall()
        result['daily'] = [dict(r) for r in rows]

    except Exception as e:
        result["error"] = str(e)
    finally:
        conn.close()
    return result


def audit_raw_decisions(db_path: str, pair: Optional[str] = None,
                        limit: int = 50) -> List[Dict]:
    """Fetch raw audit rows for deep inspection."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    rows_out: List[Dict] = []

    try:
        if not _audit_table_exists(conn):
            return [{"error": "decision_audit_log table not found"}]

        where = "WHERE 1=1"
        params: list = []
        if pair:
            where += " AND product_id = ?"
            params.append(pair)

        rows = conn.execute(f"""
            SELECT * FROM decision_audit_log
            {where}
            ORDER BY timestamp DESC
            LIMIT ?
        """, params + [limit]).fetchall()
        rows_out = [dict(r) for r in rows]

    except Exception as e:
        rows_out = [{"error": str(e)}]
    finally:
        conn.close()
    return rows_out
