"""
Data Flow Diagnostic (Audit 5)
==============================
Queries each module's data source and reports table existence, row count,
latest timestamp, and a sample row.  Run after 1 hour of trading to verify
data is flowing, or invoke manually:

    python -m tests.test_data_flow_diagnostic [--db data/renaissance_bot.db]
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("data_flow_diagnostic")


# ── Table queries for each module ─────────────────────────────────────────────

_MODULE_CHECKS: List[Dict[str, Any]] = [
    {
        "module": "BarAggregator",
        "description": "5-min OHLCV bars (from on_trade feed)",
        "table": "bars",
        "ts_column": "ts_close",
    },
    {
        "module": "DevilTracker",
        "description": "Execution quality tracking (entry/exit/devil computation)",
        "table": "devil_tracker",
        "ts_column": "opened_at",
    },
    {
        "module": "BetaMonitor",
        "description": "Portfolio beta snapshots",
        "table": "market_data",
        "ts_column": "timestamp",
        "extra_note": "BetaMonitor reads market_data + trades; check both exist",
    },
    {
        "module": "CapacityMonitor",
        "description": "Capacity / slippage wall detection",
        "table": "devil_tracker",
        "ts_column": "opened_at",
        "extra_note": "Depends on devil_tracker table being populated",
    },
    {
        "module": "SharpeMonitor",
        "description": "Rolling Sharpe health from daily trade P&L",
        "table": "trades",
        "ts_column": "timestamp",
    },
    {
        "module": "KellyPositionSizer",
        "description": "Fractional Kelly from trade history (algo_used LIKE %signal_type%)",
        "table": "trades",
        "ts_column": "timestamp",
        "extra_note": "Requires 'algo_used' column with signal type info",
    },
    {
        "module": "DailySignalReview (MedallionSignalThrottle)",
        "description": "Daily P&L per signal type",
        "table": "trades",
        "ts_column": "timestamp",
    },
    {
        "module": "MedallionRegimeDetector",
        "description": "HMM regime detection from market data",
        "table": "market_data",
        "ts_column": "timestamp",
    },
    {
        "module": "DatabaseManager — decisions",
        "description": "Trading decisions log",
        "table": "decisions",
        "ts_column": "timestamp",
    },
    {
        "module": "DatabaseManager — trades",
        "description": "Executed trades log",
        "table": "trades",
        "ts_column": "timestamp",
    },
    {
        "module": "DatabaseManager — ml_predictions",
        "description": "ML model predictions",
        "table": "ml_predictions",
        "ts_column": "timestamp",
    },
    {
        "module": "SignalThrottleLog",
        "description": "Signal auto-throttle kill/restore events",
        "table": "signal_throttle_log",
        "ts_column": "timestamp",
    },
]

# Tables created by recovery/database.py ensure_all_tables()
_RECOVERY_TABLES = [
    "system_state_log",
    "signals",
    "cost_observations",
    "balance_snapshots",
    "daily_performance",
    "exchange_health",
    "cascade_signals",
]


def _table_exists(cursor: sqlite3.Cursor, table: str) -> bool:
    cursor.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cursor.fetchone()[0] > 0


def _row_count(cursor: sqlite3.Cursor, table: str) -> int:
    cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
    return cursor.fetchone()[0]


def _latest_ts(cursor: sqlite3.Cursor, table: str, ts_col: str) -> Optional[str]:
    try:
        cursor.execute(f"SELECT MAX([{ts_col}]) FROM [{table}]")
        row = cursor.fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _sample_row(cursor: sqlite3.Cursor, table: str) -> Optional[Dict]:
    try:
        cursor.execute(f"SELECT * FROM [{table}] ORDER BY rowid DESC LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))
    except Exception:
        return None


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    try:
        cursor.execute(f"PRAGMA table_info([{table}])")
        cols = [row[1] for row in cursor.fetchall()]
        return column in cols
    except Exception:
        return False


def diagnose_data_flow(db_path: str = "data/renaissance_bot.db") -> Dict[str, Any]:
    """
    Run data flow diagnostic against the given SQLite database.

    Returns a dict with per-module results and an overall summary.
    """
    results: Dict[str, Any] = {"db_path": db_path, "timestamp": datetime.now(timezone.utc).isoformat()}
    module_results = []
    total_ok = 0
    total_empty = 0
    total_missing = 0

    db = Path(db_path)
    if not db.exists():
        results["error"] = f"Database file not found: {db_path}"
        logger.error(results["error"])
        return results

    conn = sqlite3.connect(str(db))
    cursor = conn.cursor()

    # ── Main module checks ──
    for check in _MODULE_CHECKS:
        entry: Dict[str, Any] = {
            "module": check["module"],
            "description": check["description"],
            "table": check["table"],
        }
        if not _table_exists(cursor, check["table"]):
            entry["status"] = "MISSING"
            entry["detail"] = "Table does not exist"
            total_missing += 1
        else:
            count = _row_count(cursor, check["table"])
            entry["row_count"] = count
            latest = _latest_ts(cursor, check["table"], check["ts_column"])
            entry["latest_timestamp"] = latest
            if count == 0:
                entry["status"] = "EMPTY"
                entry["detail"] = "Table exists but has no rows"
                total_empty += 1
            else:
                entry["status"] = "OK"
                entry["detail"] = f"{count} rows, latest: {latest}"
                total_ok += 1
            sample = _sample_row(cursor, check["table"])
            if sample:
                # Truncate long values for readability
                entry["sample_row"] = {
                    k: (str(v)[:80] + "..." if isinstance(v, str) and len(str(v)) > 80 else v)
                    for k, v in sample.items()
                }
        if "extra_note" in check:
            entry["note"] = check["extra_note"]
        module_results.append(entry)

    # ── Kelly Sizer schema check — does trades.algo_used column exist? ──
    algo_check: Dict[str, Any] = {
        "module": "KellyPositionSizer — schema",
        "description": "Check trades.algo_used column exists for signal-type queries",
        "table": "trades",
    }
    if _table_exists(cursor, "trades"):
        has_algo = _column_exists(cursor, "trades", "algo_used")
        algo_check["status"] = "OK" if has_algo else "WARNING"
        algo_check["detail"] = (
            "algo_used column present" if has_algo
            else "algo_used column MISSING — KellyPositionSizer queries will return empty"
        )
        if not has_algo:
            total_empty += 1
        else:
            total_ok += 1
    else:
        algo_check["status"] = "MISSING"
        algo_check["detail"] = "trades table does not exist"
        total_missing += 1
    module_results.append(algo_check)

    # ── Recovery tables check ──
    for tbl in _RECOVERY_TABLES:
        entry = {
            "module": f"Recovery — {tbl}",
            "description": f"Recovery module table: {tbl}",
            "table": tbl,
        }
        if not _table_exists(cursor, tbl):
            entry["status"] = "MISSING"
            entry["detail"] = "Table not created (recovery module may not have run)"
            total_missing += 1
        else:
            count = _row_count(cursor, tbl)
            entry["row_count"] = count
            entry["status"] = "OK" if count > 0 else "EMPTY"
            entry["detail"] = f"{count} rows" if count > 0 else "Table exists but empty"
            if count > 0:
                total_ok += 1
            else:
                total_empty += 1
        module_results.append(entry)

    conn.close()

    results["modules"] = module_results
    results["summary"] = {
        "ok": total_ok,
        "empty": total_empty,
        "missing": total_missing,
        "total_checks": total_ok + total_empty + total_missing,
    }

    return results


def print_report(results: Dict[str, Any]) -> None:
    """Pretty-print the diagnostic report."""
    print(f"\n{'=' * 70}")
    print(f"  DATA FLOW DIAGNOSTIC — {results.get('timestamp', 'unknown')}")
    print(f"  Database: {results.get('db_path', 'unknown')}")
    print(f"{'=' * 70}\n")

    if "error" in results:
        print(f"  ERROR: {results['error']}\n")
        return

    for m in results.get("modules", []):
        status = m.get("status", "?")
        icon = {"OK": "+", "EMPTY": "~", "MISSING": "!", "WARNING": "?"}.get(status, " ")
        print(f"  [{icon}] {m['module']:<45} {status}")
        print(f"      Table: {m['table']}")
        if "row_count" in m:
            print(f"      Rows: {m['row_count']}")
        if "latest_timestamp" in m and m["latest_timestamp"]:
            print(f"      Latest: {m['latest_timestamp']}")
        print(f"      Detail: {m.get('detail', '')}")
        if "note" in m:
            print(f"      Note: {m['note']}")
        if "sample_row" in m:
            sample = m["sample_row"]
            # Show first 3 columns only
            cols = list(sample.items())[:3]
            preview = ", ".join(f"{k}={v}" for k, v in cols)
            print(f"      Sample: {preview}")
        print()

    s = results.get("summary", {})
    print(f"  SUMMARY: {s.get('ok', 0)} OK, {s.get('empty', 0)} empty, "
          f"{s.get('missing', 0)} missing (out of {s.get('total_checks', 0)} checks)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Data Flow Diagnostic")
    parser.add_argument("--db", default="data/renaissance_bot.db", help="Path to SQLite database")
    args = parser.parse_args()

    results = diagnose_data_flow(args.db)
    print_report(results)

    # Exit code: 0 if all OK, 1 if any issues
    summary = results.get("summary", {})
    sys.exit(0 if summary.get("missing", 0) == 0 else 1)
