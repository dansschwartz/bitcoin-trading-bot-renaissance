"""
State Manager — SQLite-backed system state and trade lifecycle persistence.

Uses a dedicated ``data/recovery_state.db`` database (separate from the main
trading database) so the recovery module is self-contained.  Heartbeat is
file-based: a small file at ``data/.heartbeat`` whose mtime is updated
periodically; the watchdog process checks this mtime to decide if the main
bot is still alive.

All public methods are plain synchronous functions that can be called from
async code via ``asyncio.to_thread``.  Thin async wrappers are provided for
convenience (prefixed with ``a``).

Tables
------
system_state_log
    Append-only log of every state transition.
active_trades
    Current state of every trade that has been registered but not yet
    completed or abandoned.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("recovery.state_manager")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_DEFAULT_DB_PATH = _DATA_DIR / "recovery_state.db"
_HEARTBEAT_PATH = _DATA_DIR / ".heartbeat"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SystemState(Enum):
    """Finite state machine for the overall system."""
    STARTING = "STARTING"
    RECOVERING = "RECOVERING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    HALTED = "HALTED"


class TradeLifecycleState(Enum):
    """Lifecycle states for an individual trade (directional or arbitrage)."""
    PENDING = "PENDING"
    BUY_SUBMITTED = "BUY_SUBMITTED"
    BUY_FILLED = "BUY_FILLED"
    SELL_SUBMITTED = "SELL_SUBMITTED"
    SELL_FILLED = "SELL_FILLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ActiveTrade:
    """Represents a single in-flight trade tracked by the recovery system."""
    trade_id: str
    signal_type: str  # e.g. "directional", "cross_exchange_arb", "funding_arb", "triangular"
    symbol: str
    state: TradeLifecycleState = TradeLifecycleState.PENDING
    created_at: str = ""
    updated_at: str = ""
    buy_exchange: str = ""
    sell_exchange: str = ""
    buy_order_id: str = ""
    sell_order_id: str = ""
    buy_quantity: float = 0.0
    sell_quantity: float = 0.0
    buy_price: float = 0.0
    sell_price: float = 0.0
    buy_filled_qty: float = 0.0
    sell_filled_qty: float = 0.0
    buy_fill_price: float = 0.0
    sell_fill_price: float = 0.0
    expected_profit_usd: float = 0.0
    actual_profit_usd: float = 0.0
    error_message: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

class StateManager:
    """
    Manages persistent system state and trade lifecycle in SQLite.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the recovery SQLite database.  Defaults to ``data/recovery_state.db``.
    heartbeat_path : str | Path | None
        Path to the heartbeat sentinel file.  Defaults to ``data/.heartbeat``.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        heartbeat_path: Optional[str] = None,
    ) -> None:
        self._db_path = str(db_path or _DEFAULT_DB_PATH)
        self._heartbeat_path = str(heartbeat_path or _HEARTBEAT_PATH)

        # Ensure directories exist
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._heartbeat_path), exist_ok=True)

        self._init_db()
        logger.info("StateManager initialised (db=%s)", self._db_path)

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self):
        """Yield a SQLite connection with WAL mode and a 10-second timeout."""
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables if they do not already exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_state_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    state           TEXT    NOT NULL,
                    previous_state  TEXT,
                    reason          TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_trades (
                    trade_id        TEXT PRIMARY KEY,
                    signal_type     TEXT NOT NULL,
                    symbol          TEXT NOT NULL,
                    state           TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    buy_exchange    TEXT DEFAULT '',
                    sell_exchange   TEXT DEFAULT '',
                    buy_order_id    TEXT DEFAULT '',
                    sell_order_id   TEXT DEFAULT '',
                    buy_quantity    REAL DEFAULT 0.0,
                    sell_quantity   REAL DEFAULT 0.0,
                    buy_price       REAL DEFAULT 0.0,
                    sell_price      REAL DEFAULT 0.0,
                    buy_filled_qty  REAL DEFAULT 0.0,
                    sell_filled_qty REAL DEFAULT 0.0,
                    buy_fill_price  REAL DEFAULT 0.0,
                    sell_fill_price REAL DEFAULT 0.0,
                    expected_profit_usd REAL DEFAULT 0.0,
                    actual_profit_usd   REAL DEFAULT 0.0,
                    error_message   TEXT DEFAULT '',
                    raw_data        TEXT DEFAULT '{}'
                )
            """)
            # Index for fast lookup of non-terminal trades
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_active_trades_state
                ON active_trades(state)
            """)
            logger.debug("Database tables verified/created.")

    # ------------------------------------------------------------------
    # System state
    # ------------------------------------------------------------------

    def set_system_state(self, state: SystemState, reason: str = "") -> None:
        """Record a state transition.  The latest row is the current state."""
        now = datetime.now(timezone.utc).isoformat()
        previous = self.get_system_state()
        prev_value = previous.value if previous else None
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO system_state_log (timestamp, state, previous_state, reason) "
                "VALUES (?, ?, ?, ?)",
                (now, state.value, prev_value, reason),
            )
        logger.info(
            "System state: %s -> %s  reason=%s",
            prev_value, state.value, reason,
        )

    def get_system_state(self) -> Optional[SystemState]:
        """Return the most recent system state, or ``None`` if no state has been set."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state FROM system_state_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        try:
            return SystemState(row["state"])
        except (ValueError, KeyError):
            return None

    def get_state_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent state transitions (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM system_state_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Heartbeat (file-based)
    # ------------------------------------------------------------------

    def send_heartbeat(self) -> None:
        """Touch the heartbeat file so the watchdog knows we are alive."""
        try:
            Path(self._heartbeat_path).touch()
        except OSError:
            logger.warning("Failed to update heartbeat file at %s", self._heartbeat_path)

    def heartbeat_age_seconds(self) -> float:
        """Seconds since the heartbeat file was last modified.

        Returns ``float('inf')`` if the file does not exist.
        """
        try:
            mtime = os.path.getmtime(self._heartbeat_path)
            return time.time() - mtime
        except OSError:
            return float("inf")

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def register_trade(self, trade: ActiveTrade) -> None:
        """Insert a new trade into the active_trades table."""
        now = datetime.now(timezone.utc).isoformat()
        if not trade.created_at:
            trade.created_at = now
        if not trade.updated_at:
            trade.updated_at = now
        raw = json.dumps(trade.raw_data) if isinstance(trade.raw_data, dict) else trade.raw_data

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO active_trades (
                    trade_id, signal_type, symbol, state, created_at, updated_at,
                    buy_exchange, sell_exchange, buy_order_id, sell_order_id,
                    buy_quantity, sell_quantity, buy_price, sell_price,
                    buy_filled_qty, sell_filled_qty, buy_fill_price, sell_fill_price,
                    expected_profit_usd, actual_profit_usd, error_message, raw_data
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade.trade_id, trade.signal_type, trade.symbol,
                    trade.state.value, trade.created_at, trade.updated_at,
                    trade.buy_exchange, trade.sell_exchange,
                    trade.buy_order_id, trade.sell_order_id,
                    trade.buy_quantity, trade.sell_quantity,
                    trade.buy_price, trade.sell_price,
                    trade.buy_filled_qty, trade.sell_filled_qty,
                    trade.buy_fill_price, trade.sell_fill_price,
                    trade.expected_profit_usd, trade.actual_profit_usd,
                    trade.error_message, raw,
                ),
            )
        logger.info("Registered trade %s (%s %s)", trade.trade_id, trade.signal_type, trade.symbol)

    def update_trade_state(
        self,
        trade_id: str,
        new_state: TradeLifecycleState,
        **kwargs: Any,
    ) -> None:
        """Update the state (and optional fields) of an existing trade."""
        now = datetime.now(timezone.utc).isoformat()
        # Build SET clause dynamically for optional extra columns
        set_parts = ["state = ?", "updated_at = ?"]
        values: list = [new_state.value, now]

        allowed_columns = {
            "buy_order_id", "sell_order_id",
            "buy_quantity", "sell_quantity",
            "buy_price", "sell_price",
            "buy_filled_qty", "sell_filled_qty",
            "buy_fill_price", "sell_fill_price",
            "expected_profit_usd", "actual_profit_usd",
            "error_message", "raw_data",
        }
        for col, val in kwargs.items():
            if col not in allowed_columns:
                continue
            if col == "raw_data" and isinstance(val, dict):
                val = json.dumps(val)
            set_parts.append(f"{col} = ?")
            values.append(val)

        values.append(trade_id)
        sql = f"UPDATE active_trades SET {', '.join(set_parts)} WHERE trade_id = ?"

        with self._connect() as conn:
            conn.execute(sql, values)
        logger.debug("Trade %s -> %s", trade_id, new_state.value)

    def complete_trade(
        self,
        trade_id: str,
        actual_profit_usd: float = 0.0,
        error_message: str = "",
    ) -> None:
        """Mark a trade as completed (or failed)."""
        final_state = (
            TradeLifecycleState.COMPLETED if not error_message
            else TradeLifecycleState.FAILED
        )
        self.update_trade_state(
            trade_id,
            final_state,
            actual_profit_usd=actual_profit_usd,
            error_message=error_message,
        )
        logger.info(
            "Trade %s completed state=%s profit=%.4f",
            trade_id, final_state.value, actual_profit_usd,
        )

    def get_trade(self, trade_id: str) -> Optional[ActiveTrade]:
        """Fetch a single trade by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM active_trades WHERE trade_id = ?", (trade_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_trade(row)

    def get_active_trades(self) -> List[ActiveTrade]:
        """Return all trades that are NOT in a terminal state."""
        terminal = (
            TradeLifecycleState.COMPLETED.value,
            TradeLifecycleState.FAILED.value,
            TradeLifecycleState.CANCELLED.value,
        )
        placeholders = ",".join("?" for _ in terminal)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM active_trades WHERE state NOT IN ({placeholders})",
                terminal,
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_incomplete_trades(self) -> List[ActiveTrade]:
        """Return trades that started but did not reach a terminal state.

        This is the primary input for the recovery engine at startup.
        """
        return self.get_active_trades()

    def get_all_trades(self, limit: int = 200) -> List[ActiveTrade]:
        """Return the most recent trades regardless of state."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM active_trades ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    # ------------------------------------------------------------------
    # Async convenience wrappers
    # ------------------------------------------------------------------

    async def aset_system_state(self, state: SystemState, reason: str = "") -> None:
        """Async wrapper around :meth:`set_system_state`."""
        await asyncio.to_thread(self.set_system_state, state, reason)

    async def aget_system_state(self) -> Optional[SystemState]:
        """Async wrapper around :meth:`get_system_state`."""
        return await asyncio.to_thread(self.get_system_state)

    async def asend_heartbeat(self) -> None:
        """Async wrapper around :meth:`send_heartbeat`."""
        await asyncio.to_thread(self.send_heartbeat)

    async def aregister_trade(self, trade: ActiveTrade) -> None:
        """Async wrapper around :meth:`register_trade`."""
        await asyncio.to_thread(self.register_trade, trade)

    async def aupdate_trade_state(
        self, trade_id: str, new_state: TradeLifecycleState, **kwargs: Any,
    ) -> None:
        """Async wrapper around :meth:`update_trade_state`."""
        await asyncio.to_thread(self.update_trade_state, trade_id, new_state, **kwargs)

    async def acomplete_trade(
        self, trade_id: str, actual_profit_usd: float = 0.0, error_message: str = "",
    ) -> None:
        """Async wrapper around :meth:`complete_trade`."""
        await asyncio.to_thread(self.complete_trade, trade_id, actual_profit_usd, error_message)

    async def aget_active_trades(self) -> List[ActiveTrade]:
        """Async wrapper around :meth:`get_active_trades`."""
        return await asyncio.to_thread(self.get_active_trades)

    async def aget_incomplete_trades(self) -> List[ActiveTrade]:
        """Async wrapper around :meth:`get_incomplete_trades`."""
        return await asyncio.to_thread(self.get_incomplete_trades)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> ActiveTrade:
        """Convert a database row to an :class:`ActiveTrade` instance."""
        raw = row["raw_data"]
        try:
            raw_dict = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            raw_dict = {}

        try:
            state = TradeLifecycleState(row["state"])
        except ValueError:
            state = TradeLifecycleState.FAILED

        return ActiveTrade(
            trade_id=row["trade_id"],
            signal_type=row["signal_type"],
            symbol=row["symbol"],
            state=state,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            buy_exchange=row["buy_exchange"] or "",
            sell_exchange=row["sell_exchange"] or "",
            buy_order_id=row["buy_order_id"] or "",
            sell_order_id=row["sell_order_id"] or "",
            buy_quantity=float(row["buy_quantity"] or 0),
            sell_quantity=float(row["sell_quantity"] or 0),
            buy_price=float(row["buy_price"] or 0),
            sell_price=float(row["sell_price"] or 0),
            buy_filled_qty=float(row["buy_filled_qty"] or 0),
            sell_filled_qty=float(row["sell_filled_qty"] or 0),
            buy_fill_price=float(row["buy_fill_price"] or 0),
            sell_fill_price=float(row["sell_fill_price"] or 0),
            expected_profit_usd=float(row["expected_profit_usd"] or 0),
            actual_profit_usd=float(row["actual_profit_usd"] or 0),
            error_message=row["error_message"] or "",
            raw_data=raw_dict,
        )
