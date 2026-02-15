"""
Five-Minute Bar Aggregator (B3)
Converts raw tick / order-book data into 5-minute OHLCV bars with derived
features (VWAP, log-return, average spread, buy/sell ratio, funding rate)
and persists them to SQLite.
"""

import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BAR_DURATION_S = 300  # 5 minutes


@dataclass
class FiveMinuteBar:
    """A single 5-minute OHLCV bar with derived features."""
    pair: str
    exchange: str
    bar_start: float
    bar_end: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    num_trades: int
    vwap: float
    log_return: float
    avg_spread_bps: float
    buy_sell_ratio: float
    funding_rate: float


class _BarAccumulator:
    """Mutable state for an in-progress bar."""
    __slots__ = (
        "bar_start", "bar_end",
        "open", "high", "low", "close",
        "volume", "num_trades",
        "pv_sum",  # price * volume running sum for VWAP
        "buy_volume", "sell_volume",
        "spread_samples", "spread_sum",
        "funding_rate",
    )

    def __init__(self, bar_start: float):
        self.bar_start: float = bar_start
        self.bar_end: float = bar_start + BAR_DURATION_S
        self.open: Optional[float] = None
        self.high: float = -math.inf
        self.low: float = math.inf
        self.close: Optional[float] = None
        self.volume: float = 0.0
        self.num_trades: int = 0
        self.pv_sum: float = 0.0
        self.buy_volume: float = 0.0
        self.sell_volume: float = 0.0
        self.spread_samples: int = 0
        self.spread_sum: float = 0.0
        self.funding_rate: float = 0.0


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS five_minute_bars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    exchange TEXT NOT NULL,
    bar_start REAL NOT NULL,
    bar_end REAL NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    num_trades INTEGER,
    vwap REAL,
    log_return REAL,
    avg_spread_bps REAL,
    buy_sell_ratio REAL,
    funding_rate REAL,
    UNIQUE(pair, exchange, bar_start)
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_bars_lookup
ON five_minute_bars(pair, exchange, bar_start DESC);
"""


class BarAggregator:
    """
    Accepts streaming trade ticks and order-book snapshots, accumulates
    them into 5-minute OHLCV bars enriched with VWAP, log-return, average
    spread, buy/sell ratio and funding rate, then flushes completed bars to
    a SQLite database.
    """

    def __init__(self, config: Dict[str, Any], db_path: Optional[str] = None):
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = json.load(f)

        if db_path is None:
            db_path = config.get("database", {}).get("path", "data/renaissance_bot.db")

        self._db_path: str = db_path
        self._current_bars: Dict[Tuple[str, str], _BarAccumulator] = {}
        self._last_close: Dict[Tuple[str, str], float] = {}

        # Ensure the database directory exists and create the table
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(_CREATE_TABLE_SQL)
            cursor.execute(_CREATE_INDEX_SQL)
            conn.commit()

        logger.info("BarAggregator initialised  db=%s  bar_duration=%ds", self._db_path, BAR_DURATION_S)

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _get_connection(self):
        """Context manager for safe SQLite connections."""
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
    # Public API -- streaming ingestion
    # ------------------------------------------------------------------

    def on_trade(
        self,
        pair: str,
        exchange: str,
        price: float,
        quantity: float,
        side: str,
        timestamp: float,
    ) -> None:
        """
        Process a single trade tick.  If the tick falls outside the
        current bar window the existing bar is flushed first.
        """
        key = (pair, exchange)
        acc = self._current_bars.get(key)

        bar_start = self._bar_start_for(timestamp)

        # Flush the current bar if the tick belongs to a new bar window
        if acc is not None and bar_start >= acc.bar_end:
            self._flush_bar(pair, exchange)
            acc = None

        if acc is None:
            acc = _BarAccumulator(bar_start)
            self._current_bars[key] = acc

        # Update OHLCV
        if acc.open is None:
            acc.open = price
        acc.high = max(acc.high, price)
        acc.low = min(acc.low, price)
        acc.close = price
        acc.volume += quantity
        acc.num_trades += 1
        acc.pv_sum += price * quantity

        # Buy/sell tracking
        side_lower = side.lower()
        if side_lower == "buy":
            acc.buy_volume += quantity
        elif side_lower == "sell":
            acc.sell_volume += quantity

    def on_orderbook_snapshot(
        self,
        pair: str,
        exchange: str,
        best_bid: float,
        best_ask: float,
        timestamp: float,
    ) -> None:
        """
        Record a spread observation for the current bar.
        """
        key = (pair, exchange)
        acc = self._current_bars.get(key)

        bar_start = self._bar_start_for(timestamp)

        if acc is not None and bar_start >= acc.bar_end:
            self._flush_bar(pair, exchange)
            acc = None

        if acc is None:
            acc = _BarAccumulator(bar_start)
            self._current_bars[key] = acc

        mid = (best_bid + best_ask) / 2.0
        if mid > 0:
            spread_bps = ((best_ask - best_bid) / mid) * 10_000
            acc.spread_sum += spread_bps
            acc.spread_samples += 1

    # ------------------------------------------------------------------
    # Public API -- querying
    # ------------------------------------------------------------------

    def get_bars(
        self, pair: str, exchange: str, n_bars: int = 288
    ) -> List[FiveMinuteBar]:
        """
        Retrieve the most recent *n_bars* completed bars for the given
        pair and exchange from the database.  288 bars = 24 hours of
        5-minute bars.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT pair, exchange, bar_start, bar_end,
                       open, high, low, close, volume, num_trades,
                       vwap, log_return, avg_spread_bps,
                       buy_sell_ratio, funding_rate
                FROM five_minute_bars
                WHERE pair = ? AND exchange = ?
                ORDER BY bar_start DESC
                LIMIT ?
                """,
                (pair, exchange, n_bars),
            )
            rows = cursor.fetchall()

        bars = [
            FiveMinuteBar(
                pair=r[0], exchange=r[1],
                bar_start=r[2], bar_end=r[3],
                open=r[4], high=r[5], low=r[6], close=r[7],
                volume=r[8], num_trades=r[9],
                vwap=r[10], log_return=r[11],
                avg_spread_bps=r[12], buy_sell_ratio=r[13],
                funding_rate=r[14],
            )
            for r in rows
        ]

        # Return in chronological order (oldest first)
        bars.reverse()
        return bars

    def get_latest_bar(
        self, pair: str, exchange: str
    ) -> Optional[FiveMinuteBar]:
        """Return the most recently completed bar, or None."""
        bars = self.get_bars(pair, exchange, n_bars=1)
        return bars[0] if bars else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bar_start_for(timestamp: float) -> float:
        """Compute the bar-window start for a given epoch timestamp."""
        return float(int(timestamp) // BAR_DURATION_S * BAR_DURATION_S)

    def _flush_bar(self, pair: str, exchange: str) -> None:
        """
        Finalise and persist the current bar for *(pair, exchange)*.
        Computes derived features and writes to SQLite.
        """
        key = (pair, exchange)
        acc = self._current_bars.pop(key, None)
        if acc is None:
            return

        # Skip empty bars (no trades received)
        if acc.num_trades == 0 or acc.open is None:
            logger.debug("Skipping empty bar for %s/%s at %.0f", pair, exchange, acc.bar_start)
            return

        # VWAP
        vwap = acc.pv_sum / acc.volume if acc.volume > 0 else acc.close

        # Log return from previous bar's close
        prev_close = self._last_close.get(key)
        if prev_close is not None and prev_close > 0 and acc.close > 0:
            log_return = math.log(acc.close / prev_close)
        else:
            log_return = 0.0

        # Average spread in bps
        avg_spread_bps = (
            acc.spread_sum / acc.spread_samples
            if acc.spread_samples > 0
            else 0.0
        )

        # Buy/sell ratio
        total_side_vol = acc.buy_volume + acc.sell_volume
        if total_side_vol > 0:
            buy_sell_ratio = acc.buy_volume / total_side_vol
        else:
            buy_sell_ratio = 0.5  # neutral default

        bar = FiveMinuteBar(
            pair=pair,
            exchange=exchange,
            bar_start=acc.bar_start,
            bar_end=acc.bar_end,
            open=acc.open,
            high=acc.high,
            low=acc.low,
            close=acc.close,
            volume=acc.volume,
            num_trades=acc.num_trades,
            vwap=vwap,
            log_return=log_return,
            avg_spread_bps=avg_spread_bps,
            buy_sell_ratio=buy_sell_ratio,
            funding_rate=acc.funding_rate,
        )

        # Persist to database
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO five_minute_bars
                        (pair, exchange, bar_start, bar_end,
                         open, high, low, close, volume, num_trades,
                         vwap, log_return, avg_spread_bps,
                         buy_sell_ratio, funding_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bar.pair, bar.exchange,
                        bar.bar_start, bar.bar_end,
                        bar.open, bar.high, bar.low, bar.close,
                        bar.volume, bar.num_trades,
                        bar.vwap, bar.log_return,
                        bar.avg_spread_bps, bar.buy_sell_ratio,
                        bar.funding_rate,
                    ),
                )
                conn.commit()
        except sqlite3.Error as db_err:
            logger.error(
                "Failed to write bar %s/%s at %.0f: %s",
                pair, exchange, acc.bar_start, db_err,
            )
            return

        # Cache close for next bar's log-return calculation
        self._last_close[key] = bar.close

        logger.debug(
            "Flushed bar %s/%s  start=%.0f  O=%.2f H=%.2f L=%.2f C=%.2f  "
            "V=%.6f  trades=%d  vwap=%.2f  logret=%.6f  spread=%.1f bps",
            pair, exchange, bar.bar_start,
            bar.open, bar.high, bar.low, bar.close,
            bar.volume, bar.num_trades, bar.vwap,
            bar.log_return, bar.avg_spread_bps,
        )
