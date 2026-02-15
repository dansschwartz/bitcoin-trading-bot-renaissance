"""
Devil Tracker -- Execution Quality Measurement (A1)
=====================================================
Measures the gap between *theoretical* P&L (what the signal predicted at
signal-detection price, assuming zero execution cost) and *actual* P&L
(what was realised after slippage, latency, and fees).

    Devil = theoretical_pnl - actual_pnl

A positive Devil means money was "left on the table" during execution.
The goal is to keep the Devil as close to zero as possible.

This module is self-contained: it owns its own SQLite table and does NOT
depend on the broader DatabaseManager.  Any caller can instantiate it
with just a db_path string.

Design principle: "if unexpected, do nothing" -- every public method
catches exceptions internally and logs rather than raising, so a tracker
failure never kills a live trading loop.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeExecution:
    """Full lifecycle record of a single leg (entry OR exit) of a trade."""
    trade_id: str
    signal_type: str
    pair: str
    side: str                          # "BUY" or "SELL"
    exchange: str

    signal_timestamp: Optional[str] = None
    signal_price: Optional[float] = None

    order_timestamp: Optional[str] = None
    order_price: Optional[float] = None

    fill_timestamp: Optional[str] = None
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    fill_fee: Optional[float] = None

    theoretical_pnl: Optional[float] = None
    actual_pnl: Optional[float] = None
    devil: Optional[float] = None

    slippage_bps: Optional[float] = None
    latency_signal_to_fill_ms: Optional[float] = None
    latency_signal_to_order_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# DevilTracker
# ---------------------------------------------------------------------------

class DevilTracker:
    """
    Records every phase of trade execution and computes the Devil metric.

    Usage::

        dt = DevilTracker("data/renaissance_bot.db")
        tid = dt.record_signal_detection("stat_arb", "BTC-USD", "coinbase", 64350.0)
        dt.record_order_submission(tid, 64352.0)
        dt.record_fill(tid, 64355.0, 0.01, 0.12)
        # Later, for round-trip Devil:
        devil = dt.compute_devil_for_round_trip(entry_tid, exit_tid)
        summary = dt.get_devil_summary(window_hours=24)
    """

    TABLE = "devil_tracker"

    # ----- SQL constants -----
    _CREATE_TABLE = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        trade_id               TEXT PRIMARY KEY,
        signal_type            TEXT NOT NULL,
        pair                   TEXT NOT NULL,
        side                   TEXT NOT NULL,
        exchange               TEXT NOT NULL,
        signal_timestamp       TEXT,
        signal_price           REAL,
        order_timestamp        TEXT,
        order_price            REAL,
        fill_timestamp         TEXT,
        fill_price             REAL,
        fill_quantity          REAL,
        fill_fee               REAL,
        theoretical_pnl        REAL,
        actual_pnl             REAL,
        devil                  REAL,
        slippage_bps           REAL,
        latency_signal_to_fill_ms  REAL,
        latency_signal_to_order_ms REAL
    )
    """

    _CREATE_IDX_PAIR = (
        f"CREATE INDEX IF NOT EXISTS idx_devil_pair_ts "
        f"ON {TABLE} (pair, signal_timestamp)"
    )
    _CREATE_IDX_SIGNAL = (
        f"CREATE INDEX IF NOT EXISTS idx_devil_signal_ts "
        f"ON {TABLE} (signal_type, signal_timestamp)"
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_table()
        logger.info("DevilTracker initialised (db=%s)", db_path)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        """Yield a short-lived connection with WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_table(self):
        """Create the table and indexes if they do not exist."""
        try:
            with self._conn() as conn:
                conn.execute(self._CREATE_TABLE)
                conn.execute(self._CREATE_IDX_PAIR)
                conn.execute(self._CREATE_IDX_SIGNAL)
                conn.commit()
        except Exception as exc:
            logger.error("DevilTracker: failed to create table: %s", exc)

    # ------------------------------------------------------------------
    # Phase 1 -- Signal detection
    # ------------------------------------------------------------------

    def record_signal_detection(
        self,
        signal_type: str,
        pair: str,
        exchange: str,
        price: float,
        side: str = "BUY",
    ) -> Optional[str]:
        """
        Record the moment a signal is detected.

        Returns a unique *trade_id* (UUID4 hex) that callers must pass to
        subsequent ``record_order_submission`` / ``record_fill`` calls.
        Returns ``None`` on failure.
        """
        try:
            trade_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            with self._conn() as conn:
                conn.execute(
                    f"""
                    INSERT INTO {self.TABLE}
                        (trade_id, signal_type, pair, side, exchange,
                         signal_timestamp, signal_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (trade_id, signal_type, pair, side, exchange, now, price),
                )
                conn.commit()
            logger.debug(
                "DevilTracker: signal recorded id=%s type=%s pair=%s price=%.2f",
                trade_id, signal_type, pair, price,
            )
            return trade_id
        except Exception as exc:
            logger.error("DevilTracker.record_signal_detection failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Phase 2 -- Order submission
    # ------------------------------------------------------------------

    def record_order_submission(self, trade_id: str, order_price: float) -> bool:
        """
        Record when the order was submitted to the exchange.

        Also computes *latency_signal_to_order_ms* from the signal timestamp.
        Returns True on success.
        """
        try:
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()

            with self._conn() as conn:
                row = conn.execute(
                    f"SELECT signal_timestamp, signal_price FROM {self.TABLE} WHERE trade_id = ?",
                    (trade_id,),
                ).fetchone()
                if row is None:
                    logger.warning("DevilTracker: trade_id %s not found for order", trade_id)
                    return False

                signal_ts_str, signal_price = row
                latency_ms = None
                if signal_ts_str:
                    try:
                        signal_dt = datetime.fromisoformat(signal_ts_str)
                        if signal_dt.tzinfo is None:
                            signal_dt = signal_dt.replace(tzinfo=timezone.utc)
                        latency_ms = (now - signal_dt).total_seconds() * 1000.0
                    except (ValueError, TypeError):
                        pass

                conn.execute(
                    f"""
                    UPDATE {self.TABLE}
                    SET order_timestamp = ?,
                        order_price = ?,
                        latency_signal_to_order_ms = ?
                    WHERE trade_id = ?
                    """,
                    (now_iso, order_price, latency_ms, trade_id),
                )
                conn.commit()
            logger.debug(
                "DevilTracker: order recorded id=%s price=%.2f latency=%.1fms",
                trade_id, order_price, latency_ms or 0.0,
            )
            return True
        except Exception as exc:
            logger.error("DevilTracker.record_order_submission failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Phase 3 -- Fill
    # ------------------------------------------------------------------

    def record_fill(
        self,
        trade_id: str,
        fill_price: float,
        fill_quantity: float,
        fill_fee: float,
    ) -> bool:
        """
        Record that the order was filled.

        Computes:
        - slippage_bps  = |fill_price - signal_price| / signal_price * 10000
        - latency_signal_to_fill_ms
        - theoretical_pnl (per-leg, relative to signal price, ignoring costs)
        - actual_pnl (per-leg, accounting for slippage and fees)
        - devil = theoretical_pnl - actual_pnl

        Returns True on success.
        """
        try:
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()

            with self._conn() as conn:
                row = conn.execute(
                    f"""SELECT signal_timestamp, signal_price, side
                        FROM {self.TABLE} WHERE trade_id = ?""",
                    (trade_id,),
                ).fetchone()
                if row is None:
                    logger.warning("DevilTracker: trade_id %s not found for fill", trade_id)
                    return False

                signal_ts_str, signal_price, side = row

                # --- latency ---
                latency_ms = None
                if signal_ts_str:
                    try:
                        signal_dt = datetime.fromisoformat(signal_ts_str)
                        if signal_dt.tzinfo is None:
                            signal_dt = signal_dt.replace(tzinfo=timezone.utc)
                        latency_ms = (now - signal_dt).total_seconds() * 1000.0
                    except (ValueError, TypeError):
                        pass

                # --- slippage ---
                slippage_bps = 0.0
                if signal_price and signal_price > 0:
                    slippage_bps = abs(fill_price - signal_price) / signal_price * 10_000.0

                # --- P&L computation (per-leg) ---
                # Theoretical: assumes execution at signal_price with zero cost.
                # For a BUY leg the "value" is that we acquired at signal_price;
                # for a SELL leg the "value" is that we disposed at signal_price.
                # We express P&L in quote currency (USD).
                theoretical_pnl = 0.0
                actual_pnl = 0.0
                if signal_price and signal_price > 0:
                    if side and side.upper() == "BUY":
                        # Lower fill is better for a buy.
                        theoretical_pnl = 0.0  # baseline for the leg
                        actual_pnl = (signal_price - fill_price) * fill_quantity - fill_fee
                    else:
                        # Higher fill is better for a sell.
                        theoretical_pnl = 0.0
                        actual_pnl = (fill_price - signal_price) * fill_quantity - fill_fee

                devil = theoretical_pnl - actual_pnl

                conn.execute(
                    f"""
                    UPDATE {self.TABLE}
                    SET fill_timestamp = ?,
                        fill_price = ?,
                        fill_quantity = ?,
                        fill_fee = ?,
                        slippage_bps = ?,
                        latency_signal_to_fill_ms = ?,
                        theoretical_pnl = ?,
                        actual_pnl = ?,
                        devil = ?
                    WHERE trade_id = ?
                    """,
                    (
                        now_iso, fill_price, fill_quantity, fill_fee,
                        slippage_bps, latency_ms,
                        theoretical_pnl, actual_pnl, devil,
                        trade_id,
                    ),
                )
                conn.commit()

            logger.debug(
                "DevilTracker: fill recorded id=%s fill=%.2f slip=%.1fbps devil=%.4f",
                trade_id, fill_price, slippage_bps, devil,
            )
            return True
        except Exception as exc:
            logger.error("DevilTracker.record_fill failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Round-trip Devil
    # ------------------------------------------------------------------

    def compute_devil_for_round_trip(
        self, entry_trade_id: str, exit_trade_id: str
    ) -> Optional[float]:
        """
        Compute the total Devil for a round-trip trade (entry + exit).

        The *theoretical* round-trip P&L assumes both legs executed at
        their respective signal prices with zero cost.

        The *actual* round-trip P&L is the sum of per-leg actual P&L values.

        Returns ``devil = theoretical_pnl - actual_pnl`` in quote
        currency, or ``None`` if data is missing.
        """
        try:
            with self._conn() as conn:
                entry = conn.execute(
                    f"""SELECT signal_price, fill_price, fill_quantity, fill_fee, side
                        FROM {self.TABLE} WHERE trade_id = ?""",
                    (entry_trade_id,),
                ).fetchone()
                exit_ = conn.execute(
                    f"""SELECT signal_price, fill_price, fill_quantity, fill_fee, side
                        FROM {self.TABLE} WHERE trade_id = ?""",
                    (exit_trade_id,),
                ).fetchone()

            if entry is None or exit_ is None:
                logger.warning(
                    "DevilTracker: cannot compute round-trip devil, missing leg(s)"
                )
                return None

            e_sig, e_fill, e_qty, e_fee, e_side = entry
            x_sig, x_fill, x_qty, x_fee, x_side = exit_

            # Validate
            for val in (e_sig, e_fill, e_qty, e_fee, x_sig, x_fill, x_qty, x_fee):
                if val is None:
                    logger.warning("DevilTracker: incomplete data for round-trip devil")
                    return None

            # Theoretical P&L: assumes execution at signal prices, zero fees
            # Entry is BUY at e_sig, exit is SELL at x_sig (typical long trade)
            if (e_side or "").upper() == "BUY":
                theoretical = (x_sig - e_sig) * min(e_qty, x_qty)
            else:
                # Short trade: entry SELL, exit BUY
                theoretical = (e_sig - x_sig) * min(e_qty, x_qty)

            # Actual P&L: uses fill prices, subtracts fees
            if (e_side or "").upper() == "BUY":
                actual = (x_fill - e_fill) * min(e_qty, x_qty) - e_fee - x_fee
            else:
                actual = (e_fill - x_fill) * min(e_qty, x_qty) - e_fee - x_fee

            devil = theoretical - actual
            logger.info(
                "DevilTracker round-trip: theoretical=%.4f actual=%.4f devil=%.4f",
                theoretical, actual, devil,
            )
            return devil

        except Exception as exc:
            logger.error("DevilTracker.compute_devil_for_round_trip failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Summary / analytics
    # ------------------------------------------------------------------

    def get_devil_summary(self, window_hours: int = 24) -> Dict[str, Any]:
        """
        Aggregate Devil statistics over a rolling window.

        Returns a dict with:
        - aggregate_devil_bps: portfolio-level Devil in basis points
        - by_pair: {pair: avg_devil_bps}
        - by_signal: {signal_type: avg_devil_bps}
        - by_exchange: {exchange: avg_devil_bps}
        - avg_slippage_bps
        - avg_latency_ms
        - total_theoretical_pnl
        - total_actual_pnl
        - total_devil
        - devil_trend: list of (timestamp, devil) for sparkline
        """
        empty: Dict[str, Any] = {
            "aggregate_devil_bps": 0.0,
            "by_pair": {},
            "by_signal": {},
            "by_exchange": {},
            "avg_slippage_bps": 0.0,
            "avg_latency_ms": 0.0,
            "total_theoretical_pnl": 0.0,
            "total_actual_pnl": 0.0,
            "total_devil": 0.0,
            "devil_trend": [],
            "trade_count": 0,
        }
        try:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(hours=window_hours)
            ).isoformat()

            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"""
                    SELECT * FROM {self.TABLE}
                    WHERE fill_timestamp IS NOT NULL
                      AND signal_timestamp >= ?
                    ORDER BY signal_timestamp ASC
                    """,
                    (cutoff,),
                ).fetchall()

            if not rows:
                return empty

            # Accumulators
            total_devil = 0.0
            total_theoretical = 0.0
            total_actual = 0.0
            total_slippage = 0.0
            total_latency = 0.0
            latency_count = 0

            pair_devils: Dict[str, List[float]] = {}
            signal_devils: Dict[str, List[float]] = {}
            exchange_devils: Dict[str, List[float]] = {}
            devil_trend: List[tuple] = []

            for r in rows:
                d = r["devil"] if r["devil"] is not None else 0.0
                t = r["theoretical_pnl"] if r["theoretical_pnl"] is not None else 0.0
                a = r["actual_pnl"] if r["actual_pnl"] is not None else 0.0
                s = r["slippage_bps"] if r["slippage_bps"] is not None else 0.0
                lat = r["latency_signal_to_fill_ms"]

                total_devil += d
                total_theoretical += t
                total_actual += a
                total_slippage += s

                if lat is not None:
                    total_latency += lat
                    latency_count += 1

                pair = r["pair"]
                sig = r["signal_type"]
                exch = r["exchange"]

                pair_devils.setdefault(pair, []).append(s)  # use slippage as proxy bps
                signal_devils.setdefault(sig, []).append(s)
                exchange_devils.setdefault(exch, []).append(s)

                devil_trend.append((r["signal_timestamp"], d))

            n = len(rows)

            def _avg(lst):
                return sum(lst) / len(lst) if lst else 0.0

            # Aggregate devil in bps -- ratio of total devil to total notional
            total_notional = 0.0
            for r in rows:
                fp = r["fill_price"] or 0.0
                fq = r["fill_quantity"] or 0.0
                total_notional += fp * fq
            aggregate_devil_bps = (
                (total_devil / total_notional * 10_000.0) if total_notional > 0 else 0.0
            )

            return {
                "aggregate_devil_bps": round(aggregate_devil_bps, 2),
                "by_pair": {k: round(_avg(v), 2) for k, v in pair_devils.items()},
                "by_signal": {k: round(_avg(v), 2) for k, v in signal_devils.items()},
                "by_exchange": {k: round(_avg(v), 2) for k, v in exchange_devils.items()},
                "avg_slippage_bps": round(total_slippage / n, 2) if n else 0.0,
                "avg_latency_ms": round(total_latency / latency_count, 2) if latency_count else 0.0,
                "total_theoretical_pnl": round(total_theoretical, 6),
                "total_actual_pnl": round(total_actual, 6),
                "total_devil": round(total_devil, 6),
                "devil_trend": devil_trend[-100:],   # last 100 points
                "trade_count": n,
            }
        except Exception as exc:
            logger.error("DevilTracker.get_devil_summary failed: %s", exc)
            return empty

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    def should_alert(self, threshold_bps: float = 5.0) -> Optional[str]:
        """
        Check whether the recent Devil warrants a human alert.

        Looks at the last 1-hour window.  Returns an alert message string
        if the aggregate Devil exceeds *threshold_bps*, otherwise ``None``.
        """
        try:
            summary = self.get_devil_summary(window_hours=1)
            if summary["trade_count"] == 0:
                return None

            agg = summary["aggregate_devil_bps"]
            if abs(agg) > threshold_bps:
                return (
                    f"DEVIL ALERT: aggregate execution cost is {agg:.1f} bps "
                    f"over the last hour ({summary['trade_count']} trades). "
                    f"Avg slippage {summary['avg_slippage_bps']:.1f} bps, "
                    f"avg latency {summary['avg_latency_ms']:.0f} ms."
                )

            avg_slip = summary["avg_slippage_bps"]
            if avg_slip > threshold_bps:
                return (
                    f"SLIPPAGE ALERT: avg slippage is {avg_slip:.1f} bps "
                    f"({summary['trade_count']} trades in last hour)."
                )

            return None
        except Exception as exc:
            logger.error("DevilTracker.should_alert failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_trade(self, trade_id: str) -> Optional[TradeExecution]:
        """Fetch a single TradeExecution record, or None."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    f"SELECT * FROM {self.TABLE} WHERE trade_id = ?",
                    (trade_id,),
                ).fetchone()
            if row is None:
                return None
            return TradeExecution(**{k: row[k] for k in row.keys()})
        except Exception as exc:
            logger.error("DevilTracker.get_trade failed: %s", exc)
            return None
