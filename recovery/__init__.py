"""
Renaissance Trading Bot — Disaster Recovery & Auto-Restart Module

Provides:
    - StateManager: SQLite-backed state persistence and trade lifecycle tracking
    - RecoveryEngine: Startup reconciliation for incomplete trades
    - Watchdog: Separate process that monitors heartbeat and restarts the bot
    - GracefulShutdownHandler: SIGTERM/SIGINT handling with in-flight trade drainage
    - ensure_all_tables / run_migration: Enhanced database schema migration
"""

from recovery.database import ensure_all_tables, run_migration

try:
    from recovery.state_manager import (
        SystemState,
        TradeLifecycleState,
        ActiveTrade,
        StateManager,
    )
    from recovery.recovery_engine import RecoveryEngine
    from recovery.watchdog import Watchdog
    from recovery.shutdown import GracefulShutdownHandler
except ImportError:
    pass

__all__ = [
    "ensure_all_tables",
    "run_migration",
    "SystemState",
    "TradeLifecycleState",
    "ActiveTrade",
    "StateManager",
    "RecoveryEngine",
    "Watchdog",
    "GracefulShutdownHandler",
]
