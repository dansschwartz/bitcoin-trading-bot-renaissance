"""
Recovery Database Migration Module

Provides functions to ensure all enhanced tables exist in the Renaissance bot
SQLite database. Tables are created with CREATE TABLE IF NOT EXISTS so this
module is safe to call repeatedly (idempotent).

Usage:
    # Programmatic
    from recovery.database import ensure_all_tables
    ensure_all_tables("data/renaissance_bot.db")

    # Command-line
    python -m recovery.database
    python -m recovery.database --db-path data/renaissance_bot.db
"""

import argparse
import logging
import os
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL definitions for every enhanced table
# ---------------------------------------------------------------------------

_MIGRATION_SQL = """
-- ================================================================
-- System state transitions log
-- ================================================================
CREATE TABLE IF NOT EXISTS system_state_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    state TEXT NOT NULL,
    previous_state TEXT,
    reason TEXT,
    metadata TEXT  -- JSON
);
CREATE INDEX IF NOT EXISTS idx_state_log_ts ON system_state_log(timestamp);

-- ================================================================
-- Signals detected (all, not just traded)
-- ================================================================
CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    signal_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    buy_exchange TEXT,
    sell_exchange TEXT,
    buy_price REAL,
    sell_price REAL,
    gross_spread_bps REAL,
    estimated_cost_bps REAL,
    net_spread_bps REAL,
    confidence REAL,
    traded INTEGER DEFAULT 0,
    trade_id TEXT,
    rejection_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type, timestamp);

-- ================================================================
-- Cost model learning data
-- ================================================================
CREATE TABLE IF NOT EXISTS cost_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity REAL NOT NULL,
    estimated_fee_bps REAL,
    estimated_slippage_bps REAL,
    estimated_total_bps REAL,
    realized_fee_bps REAL,
    realized_slippage_bps REAL,
    realized_total_bps REAL,
    book_depth_at_trade REAL,
    spread_at_trade_bps REAL,
    volatility_1h REAL,
    hour_of_day INTEGER,
    day_of_week INTEGER
);
CREATE INDEX IF NOT EXISTS idx_cost_obs_ts ON cost_observations(timestamp);

-- ================================================================
-- Balance snapshots (periodic)
-- ================================================================
CREATE TABLE IF NOT EXISTS balance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    exchange TEXT NOT NULL,
    currency TEXT NOT NULL,
    free REAL NOT NULL,
    locked REAL NOT NULL,
    total REAL NOT NULL,
    usd_value REAL
);
CREATE INDEX IF NOT EXISTS idx_balance_ts ON balance_snapshots(timestamp);

-- ================================================================
-- Daily performance summary
-- ================================================================
CREATE TABLE IF NOT EXISTS daily_performance (
    date TEXT PRIMARY KEY,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    failed_trades INTEGER DEFAULT 0,
    gross_profit_usd REAL DEFAULT 0,
    total_fees_usd REAL DEFAULT 0,
    net_profit_usd REAL DEFAULT 0,
    cross_exchange_pnl REAL DEFAULT 0,
    funding_rate_pnl REAL DEFAULT 0,
    triangular_pnl REAL DEFAULT 0,
    directional_pnl REAL DEFAULT 0,
    max_drawdown_usd REAL,
    max_one_sided_usd REAL,
    emergency_closes INTEGER DEFAULT 0,
    avg_estimated_cost REAL,
    avg_realized_cost REAL,
    cost_estimation_error REAL,
    total_equity_usd REAL,
    coinbase_equity_usd REAL,
    mexc_equity_usd REAL,
    binance_equity_usd REAL
);

-- ================================================================
-- Exchange health log
-- ================================================================
CREATE TABLE IF NOT EXISTS exchange_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    exchange TEXT NOT NULL,
    api_latency_ms INTEGER,
    ws_connected INTEGER,
    book_staleness_ms INTEGER,
    error_count_1h INTEGER DEFAULT 0,
    status TEXT NOT NULL  -- healthy, degraded, down
);
CREATE INDEX IF NOT EXISTS idx_health_ts ON exchange_health(timestamp);

-- ================================================================
-- Liquidation cascade signals
-- ================================================================
CREATE TABLE IF NOT EXISTS cascade_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    risk_score REAL NOT NULL,
    funding_rate REAL,
    funding_rate_percentile REAL,
    open_interest_change_24h REAL,
    long_short_ratio REAL,
    estimated_liquidation_usd REAL,
    recommended_action TEXT,
    acted_upon INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cascade_ts ON cascade_signals(timestamp);
"""


def ensure_all_tables(db_path: str) -> None:
    """Create all enhanced tables in the SQLite database.

    This function is idempotent: every statement uses
    ``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS`` so it is
    safe to call on every application startup.

    Args:
        db_path: Path to the SQLite database file.  Parent directories will
                 be created automatically if they do not exist.

    Raises:
        sqlite3.Error: If the database file cannot be opened or a statement
                       fails for reasons other than the object already existing.
    """
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_MIGRATION_SQL)
        conn.commit()
        logger.info(
            "Enhanced database tables ensured successfully at %s", db_path
        )
    except sqlite3.Error:
        logger.exception("Failed to ensure enhanced database tables at %s", db_path)
        raise
    finally:
        if conn is not None:
            conn.close()


def run_migration() -> None:
    """Entry-point for command-line migration.

    Parses ``--db-path`` from the command line (defaults to
    ``data/renaissance_bot.db``) and calls :func:`ensure_all_tables`.
    """
    parser = argparse.ArgumentParser(
        description="Run Renaissance bot database migration to add enhanced tables."
    )
    parser.add_argument(
        "--db-path",
        default="data/renaissance_bot.db",
        help="Path to the SQLite database file (default: data/renaissance_bot.db)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging output.",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    logger.info("Starting database migration for %s", args.db_path)
    ensure_all_tables(args.db_path)
    logger.info("Migration completed successfully.")


if __name__ == "__main__":
    run_migration()
