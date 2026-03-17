"""
Data Exhaust Capture — snapshots order book state at detection,
execution, and post-execution for post-hoc analysis.

Captures top 5 price levels from both exchanges at each phase
to understand how the book moves around our trades.
"""
import asyncio
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("arb.analytics.exhaust")


class ExhaustCapture:

    def __init__(self, db_path: str = "data/arbitrage.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS arb_signal_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT,
                phase TEXT,
                timestamp TEXT,
                symbol TEXT,
                exchange TEXT,
                bid_1 REAL, bid_2 REAL, bid_3 REAL, bid_4 REAL, bid_5 REAL,
                bid_qty_1 REAL, bid_qty_2 REAL, bid_qty_3 REAL, bid_qty_4 REAL, bid_qty_5 REAL,
                ask_1 REAL, ask_2 REAL, ask_3 REAL, ask_4 REAL, ask_5 REAL,
                ask_qty_1 REAL, ask_qty_2 REAL, ask_qty_3 REAL, ask_qty_4 REAL, ask_qty_5 REAL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signal_snapshots_signal "
            "ON arb_signal_snapshots(signal_id)"
        )
        conn.commit()
        conn.close()

    def _snapshot_book(self, book) -> Dict:
        """Extract top 5 bid/ask levels from an OrderBook object."""
        result = {}
        bids = book.bids[:5] if book and hasattr(book, 'bids') else []
        asks = book.asks[:5] if book and hasattr(book, 'asks') else []

        for i in range(5):
            if i < len(bids):
                result[f"bid_{i+1}"] = float(bids[i].price)
                result[f"bid_qty_{i+1}"] = float(bids[i].quantity)
            else:
                result[f"bid_{i+1}"] = None
                result[f"bid_qty_{i+1}"] = None

            if i < len(asks):
                result[f"ask_{i+1}"] = float(asks[i].price)
                result[f"ask_qty_{i+1}"] = float(asks[i].quantity)
            else:
                result[f"ask_{i+1}"] = None
                result[f"ask_qty_{i+1}"] = None

        return result

    def _store_snapshot(self, signal_id: str, phase: str, symbol: str,
                        exchange: str, book_data: Dict):
        """Persist a single book snapshot to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """INSERT INTO arb_signal_snapshots
                   (signal_id, phase, timestamp, symbol, exchange,
                    bid_1, bid_2, bid_3, bid_4, bid_5,
                    bid_qty_1, bid_qty_2, bid_qty_3, bid_qty_4, bid_qty_5,
                    ask_1, ask_2, ask_3, ask_4, ask_5,
                    ask_qty_1, ask_qty_2, ask_qty_3, ask_qty_4, ask_qty_5)
                   VALUES (?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?)""",
                (
                    signal_id, phase, datetime.utcnow().isoformat(),
                    symbol, exchange,
                    book_data.get("bid_1"), book_data.get("bid_2"),
                    book_data.get("bid_3"), book_data.get("bid_4"),
                    book_data.get("bid_5"),
                    book_data.get("bid_qty_1"), book_data.get("bid_qty_2"),
                    book_data.get("bid_qty_3"), book_data.get("bid_qty_4"),
                    book_data.get("bid_qty_5"),
                    book_data.get("ask_1"), book_data.get("ask_2"),
                    book_data.get("ask_3"), book_data.get("ask_4"),
                    book_data.get("ask_5"),
                    book_data.get("ask_qty_1"), book_data.get("ask_qty_2"),
                    book_data.get("ask_qty_3"), book_data.get("ask_qty_4"),
                    book_data.get("ask_qty_5"),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Exhaust snapshot persist error: {e}")

    def capture_at_detection(self, signal_id: str, symbol: str,
                              books: Dict):
        """Snapshot books when a signal is first detected.
        books = {"exchange_name": OrderBook, ...}
        """
        for exchange, book in books.items():
            if book is None:
                continue
            data = self._snapshot_book(book)
            self._store_snapshot(signal_id, "detection", symbol, exchange, data)

    def capture_at_execution(self, signal_id: str, symbol: str,
                              books: Dict):
        """Snapshot books at trade execution time."""
        for exchange, book in books.items():
            if book is None:
                continue
            data = self._snapshot_book(book)
            self._store_snapshot(signal_id, "execution", symbol, exchange, data)

    def capture_post_execution(self, signal_id: str, symbol: str,
                                books: Dict):
        """Snapshot books ~1s after execution for slippage analysis."""
        for exchange, book in books.items():
            if book is None:
                continue
            data = self._snapshot_book(book)
            self._store_snapshot(signal_id, "post_execution", symbol, exchange, data)

    def get_recent_snapshots(self, limit: int = 100) -> List[Dict]:
        """Return recent snapshots for dashboard display."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM arb_signal_snapshots
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"Exhaust query error: {e}")
            return []
