"""Database schema for the agent coordination system (Doc 15).

Tables: agent_events, proposals, improvement_log, model_ledger, weekly_reports.
Call ``ensure_agent_tables(db_path)`` on startup to create/migrate.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── CREATE TABLE statements ──

_TABLES = {
    "agent_events": """
        CREATE TABLE IF NOT EXISTS agent_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            agent_name  TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            channel     TEXT NOT NULL,
            payload     TEXT,
            severity    TEXT DEFAULT 'info'
        )
    """,
    "proposals": """
        CREATE TABLE IF NOT EXISTS proposals (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at          TEXT,
            source              TEXT NOT NULL DEFAULT 'quant_researcher',
            category            TEXT NOT NULL,
            title               TEXT NOT NULL,
            description         TEXT,
            code_diff           TEXT,
            config_changes      TEXT,
            deployment_mode     TEXT NOT NULL DEFAULT 'parameter_tune',
            status              TEXT NOT NULL DEFAULT 'pending',
            safety_gate_results TEXT,
            backtest_sharpe     REAL,
            backtest_drawdown   REAL,
            backtest_accuracy   REAL,
            backtest_sample_size INTEGER,
            backtest_p_value    REAL,
            sandbox_start       TEXT,
            sandbox_end         TEXT,
            deployed_at         TEXT,
            rollback_at         TEXT,
            notes               TEXT
        )
    """,
    "improvement_log": """
        CREATE TABLE IF NOT EXISTS improvement_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            proposal_id     INTEGER REFERENCES proposals(id),
            change_type     TEXT NOT NULL,
            description     TEXT,
            metric_before   TEXT,
            metric_after    TEXT,
            config_snapshot TEXT,
            reverted        INTEGER DEFAULT 0
        )
    """,
    "model_ledger": """
        CREATE TABLE IF NOT EXISTS model_ledger (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            model_name      TEXT NOT NULL,
            model_version   TEXT NOT NULL,
            accuracy        REAL,
            sharpe          REAL,
            max_drawdown    REAL,
            file_path       TEXT,
            file_hash       TEXT,
            status          TEXT NOT NULL DEFAULT 'active',
            replaced_by     INTEGER REFERENCES model_ledger(id)
        )
    """,
    "weekly_reports": """
        CREATE TABLE IF NOT EXISTS weekly_reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start      TEXT NOT NULL,
            week_end        TEXT NOT NULL,
            generated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            report_json     TEXT NOT NULL,
            sharpe_7d       REAL,
            total_pnl       REAL,
            total_trades    INTEGER,
            win_rate        REAL
        )
    """,
}

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_agent_events_ts ON agent_events(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_agent_events_agent ON agent_events(agent_name)",
    "CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status)",
    "CREATE INDEX IF NOT EXISTS idx_proposals_category ON proposals(category)",
    "CREATE INDEX IF NOT EXISTS idx_improvement_log_ts ON improvement_log(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_model_ledger_name ON model_ledger(model_name)",
    "CREATE INDEX IF NOT EXISTS idx_weekly_reports_week ON weekly_reports(week_start)",
]


def ensure_agent_tables(db_path: str) -> None:
    """Create agent tables if they don't exist.  Safe to call repeatedly."""
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        for table_name, ddl in _TABLES.items():
            conn.execute(ddl)
        for idx in _INDEXES:
            conn.execute(idx)
        conn.commit()
        conn.close()
        logger.info("Agent tables verified (%d tables, %d indexes)", len(_TABLES), len(_INDEXES))
    except Exception as exc:
        logger.error("Failed to ensure agent tables: %s", exc)
        raise


# ── Convenience writers ──

def log_agent_event(
    db_path: str,
    agent_name: str,
    event_type: str,
    channel: str,
    payload: Optional[Dict[str, Any]] = None,
    severity: str = "info",
) -> None:
    """Insert one row into agent_events."""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute(
            """INSERT INTO agent_events (agent_name, event_type, channel, payload, severity)
               VALUES (?, ?, ?, ?, ?)""",
            (
                agent_name,
                event_type,
                channel,
                json.dumps(payload) if payload else None,
                severity,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("log_agent_event failed: %s", exc)


def insert_proposal(db_path: str, proposal_dict: Dict[str, Any]) -> int:
    """Insert a proposal and return its id."""
    conn = sqlite3.connect(db_path, timeout=5.0)
    cur = conn.execute(
        """INSERT INTO proposals
           (source, category, title, description, code_diff, config_changes,
            deployment_mode, status, safety_gate_results,
            backtest_sharpe, backtest_drawdown, backtest_accuracy,
            backtest_sample_size, backtest_p_value, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            proposal_dict.get("source", "quant_researcher"),
            proposal_dict.get("category", "parameter_tune"),
            proposal_dict.get("title", "Untitled"),
            proposal_dict.get("description"),
            proposal_dict.get("code_diff"),
            json.dumps(proposal_dict.get("config_changes")) if proposal_dict.get("config_changes") else None,
            proposal_dict.get("deployment_mode", "parameter_tune"),
            proposal_dict.get("status", "pending"),
            json.dumps(proposal_dict.get("safety_gate_results")) if proposal_dict.get("safety_gate_results") else None,
            proposal_dict.get("backtest_sharpe"),
            proposal_dict.get("backtest_drawdown"),
            proposal_dict.get("backtest_accuracy"),
            proposal_dict.get("backtest_sample_size"),
            proposal_dict.get("backtest_p_value"),
            proposal_dict.get("notes"),
        ),
    )
    proposal_id = cur.lastrowid
    conn.commit()
    conn.close()
    return proposal_id


def insert_model_ledger(
    db_path: str,
    model_name: str,
    model_version: str,
    accuracy: Optional[float] = None,
    sharpe: Optional[float] = None,
    max_drawdown: Optional[float] = None,
    file_path: Optional[str] = None,
    file_hash: Optional[str] = None,
    status: str = "active",
    proposal_id: Optional[int] = None,
) -> int:
    """Insert one row into model_ledger and return its id."""
    conn = sqlite3.connect(db_path, timeout=5.0)
    cur = conn.execute(
        """INSERT INTO model_ledger
           (model_name, model_version, accuracy, sharpe, max_drawdown,
            file_path, file_hash, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (model_name, model_version, accuracy, sharpe, max_drawdown,
         file_path, file_hash, status),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    logger.debug("model_ledger: inserted %s v=%s (id=%d)", model_name, model_version, row_id)
    return row_id


def log_improvement(
    db_path: str,
    proposal_id: Optional[int],
    change_type: str,
    description: str,
    metric_before: Optional[str] = None,
    metric_after: Optional[str] = None,
    config_snapshot: Optional[str] = None,
) -> None:
    """Insert one row into improvement_log."""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute(
            """INSERT INTO improvement_log
               (proposal_id, change_type, description, metric_before, metric_after, config_snapshot)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (proposal_id, change_type, description, metric_before, metric_after, config_snapshot),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("log_improvement failed: %s", exc)
