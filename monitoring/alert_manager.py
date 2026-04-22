"""
Alert Manager — aggregates alerts from all trading sub-systems and routes
them through the :class:`TelegramAlerter`.

Responsibilities:
    * Collect alerts from recovery, trading, arbitrage, and the liquidation
      detector.
    * De-duplicate by content hash so the same alert is not stored/sent twice
      within a configurable window.
    * Persist every alert to the ``system_state_log`` table in the SQLite
      database so the dashboard can query recent history.
    * Expose :meth:`get_recent_alerts` for the dashboard API.

All public methods are ``async`` and catch exceptions internally so that
alerting failures never crash the trading system.
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from monitoring.telegram_bot import TelegramAlerter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEDUP_WINDOW_SECONDS: float = 60.0


class AlertManager:
    """Aggregates, de-duplicates, persists, and routes alerts.

    Args:
        telegram_alerter: A configured :class:`TelegramAlerter` instance.
                          If ``None`` a new console-only alerter is created.
        db_path:          Path to the SQLite database that contains the
                          ``system_state_log`` table.  Defaults to
                          ``data/renaissance_bot.db``.
        config:           Optional configuration dict.  Recognized keys:
                          ``db_path``, ``dedup_window_seconds``.
    """

    def __init__(
        self,
        telegram_alerter: Optional[TelegramAlerter] = None,
        db_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        config = config or {}

        self._telegram: TelegramAlerter = telegram_alerter or TelegramAlerter()

        self._db_path: str = (
            db_path
            or config.get("db_path")
            or os.environ.get("RENAISSANCE_DB_PATH", "data/renaissance_bot.db")
        )

        self._dedup_window: float = float(
            config.get("dedup_window_seconds", DEDUP_WINDOW_SECONDS)
        )

        # In-memory dedup cache: content_hash -> monotonic timestamp
        self._recent_hashes: Dict[str, float] = {}

        # In-memory history kept for quick dashboard access
        self._history: List[Dict[str, Any]] = []

        logger.info(
            "AlertManager initialized (db=%s, dedup_window=%.0fs)",
            self._db_path,
            self._dedup_window,
        )

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for safe SQLite connections with WAL mode."""
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_alert(
        self,
        level: str,
        message: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Process, de-duplicate, persist, and route a single alert.

        Args:
            level:    One of ``INFO``, ``WARNING``, ``CRITICAL``.
            message:  Human-readable alert text.
            source:   Origin subsystem (e.g. ``recovery``, ``arbitrage``).
            metadata: Arbitrary JSON-safe dict attached to the log entry.
        """
        try:
            level = level.upper()
            is_critical = level == "CRITICAL"

            # --- deduplication (critical always passes) ---
            if not is_critical:
                content_hash = self._content_hash(level, message, source)
                now = time.monotonic()
                last = self._recent_hashes.get(content_hash)
                if last is not None and (now - last) < self._dedup_window:
                    logger.debug("Duplicate alert suppressed: %s", message[:80])
                    return
                self._recent_hashes[content_hash] = now
                self._evict_old_hashes(now)

            # --- build record ---
            ts = datetime.now(timezone.utc).isoformat()
            record: Dict[str, Any] = {
                "timestamp": ts,
                "level": level,
                "message": message,
                "source": source,
                "metadata": metadata or {},
            }

            # --- persist to SQLite ---
            self._persist(record)

            # --- keep in memory for dashboard ---
            self._history.append(record)
            # Cap memory usage
            if len(self._history) > 500:
                self._history = self._history[-500:]

            # --- route to Telegram ---
            await self._route_to_telegram(level, message, source)

        except Exception:
            logger.exception("AlertManager.send_alert failed")

    async def send_trade_alert(self, trade_result: dict) -> None:
        """Convenience: forward a trade execution to Telegram and persist it.

        Args:
            trade_result: Trade result dict (same as TelegramAlerter expects).
        """
        try:
            symbol = trade_result.get("symbol", trade_result.get("product_id", ""))
            side = trade_result.get("side", "")
            await self.send_alert(
                level="INFO",
                message=f"Trade executed: {side} {symbol}",
                source="trading",
                metadata=trade_result,
            )
            await self._telegram.send_trade_executed(trade_result)
        except Exception:
            logger.exception("AlertManager.send_trade_alert failed")

    async def send_daily_summary(self, daily: dict) -> None:
        """Convenience: forward a daily summary to Telegram and persist it.

        Args:
            daily: Daily performance dict.
        """
        try:
            date = daily.get("date", "N/A")
            net = daily.get("net_profit_usd", 0.0)
            await self.send_alert(
                level="INFO",
                message=f"Daily summary {date}: net P&L {net:.2f} USD",
                source="performance",
                metadata=daily,
            )
            await self._telegram.send_daily_summary(daily)
        except Exception:
            logger.exception("AlertManager.send_daily_summary failed")

    async def send_warning(self, message: str, source: str = "") -> None:
        """Send a WARNING through all channels.

        Args:
            message: Warning description.
            source:  Origin subsystem.
        """
        try:
            await self.send_alert("WARNING", message, source=source)
            await self._telegram.send_warning(message)
        except Exception:
            logger.exception("AlertManager.send_warning failed")

    async def send_critical(self, message: str, source: str = "") -> None:
        """Send a CRITICAL alert through all channels.

        Args:
            message: Critical event description.
            source:  Origin subsystem.
        """
        try:
            await self.send_alert("CRITICAL", message, source=source)
            await self._telegram.send_critical(message)
        except Exception:
            logger.exception("AlertManager.send_critical failed")

    async def send_system_event(
        self, event: str, details: str, source: str = "system"
    ) -> None:
        """Record and send a system state-change event.

        Args:
            event:   Short event label.
            details: Longer description.
            source:  Origin subsystem.
        """
        try:
            await self.send_alert(
                level="INFO",
                message=f"{event}: {details}",
                source=source,
                metadata={"event": event, "details": details},
            )
            await self._telegram.send_system_event(event, details)
        except Exception:
            logger.exception("AlertManager.send_system_event failed")

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent alerts (from memory + SQLite fallback).

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of alert dicts ordered newest-first.
        """
        try:
            # Try SQLite first for a complete picture
            alerts = self._query_recent_from_db(limit)
            if alerts:
                return alerts
        except Exception:
            logger.debug("SQLite query failed, falling back to in-memory history")

        # Fallback to in-memory
        return list(reversed(self._history[-limit:]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_hash(level: str, message: str, source: str) -> str:
        """Deterministic hash for deduplication."""
        raw = f"{level}|{source}|{message}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def _evict_old_hashes(self, now: float) -> None:
        """Remove dedup entries older than the window."""
        expired = [
            h
            for h, ts in self._recent_hashes.items()
            if (now - ts) > self._dedup_window
        ]
        for h in expired:
            del self._recent_hashes[h]

    def _persist(self, record: Dict[str, Any]) -> None:
        """Insert alert into the system_state_log table.

        Uses the ``state`` column for the alert level, ``reason`` for the
        message text, and ``metadata`` for JSON-encoded extra data.
        """
        try:
            meta = json.dumps(
                {
                    "source": record.get("source", ""),
                    "alert_level": record.get("level", "INFO"),
                    **(record.get("metadata") or {}),
                },
                default=str,
            )
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO system_state_log (timestamp, state, previous_state, reason, metadata)
                    VALUES (?, ?, NULL, ?, ?)
                    """,
                    (
                        record["timestamp"],
                        record.get("level", "INFO"),
                        record.get("message", ""),
                        meta,
                    ),
                )
                conn.commit()
        except Exception:
            # Persistence failure must not block alerting
            logger.debug("Failed to persist alert to system_state_log", exc_info=True)

    def _query_recent_from_db(self, limit: int) -> List[Dict[str, Any]]:
        """Query the most recent alerts from system_state_log."""
        results: List[Dict[str, Any]] = []
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, timestamp, state, previous_state, reason, metadata
                    FROM system_state_log
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                for row in cursor.fetchall():
                    meta_raw = row[5]
                    try:
                        meta = json.loads(meta_raw) if meta_raw else {}
                    except (json.JSONDecodeError, TypeError):
                        meta = {}
                    results.append(
                        {
                            "id": row[0],
                            "timestamp": row[1],
                            "level": row[2],
                            "previous_state": row[3],
                            "message": row[4],
                            "metadata": meta,
                        }
                    )
        except Exception:
            logger.debug("Failed to query system_state_log", exc_info=True)
        return results

    async def _route_to_telegram(
        self, level: str, message: str, source: str
    ) -> None:
        """Dispatch to the appropriate TelegramAlerter method based on level."""
        prefix = f"[{source}] " if source else ""
        full_message = f"{prefix}{message}"
        try:
            if level == "CRITICAL":
                await self._telegram.send_critical(full_message)
            elif level == "WARNING":
                await self._telegram.send_warning(full_message)
            else:
                await self._telegram.send_system_event("Alert", full_message)
        except Exception:
            logger.debug("Telegram routing failed", exc_info=True)
