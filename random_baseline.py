"""
Random Entry Baseline — Shadow system measuring ML value-add.

Generates random buy/sell entries at the same rate as the real system,
but positions are managed by the same reeval engine. Measures whether
ML signals add value over random entry + smart exit management.

Observation only: no real trades, just DB tracking.
"""

import logging
import random
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RandomEntryBaseline:
    """Shadow system that randomly enters positions to benchmark ML value."""

    def __init__(
        self,
        db_path: str = "data/renaissance_bot.db",
        entry_probability: float = 0.05,  # 5% chance per pair per cycle
        max_shadow_positions: int = 10,
    ) -> None:
        self.db_path = db_path
        self.entry_probability = entry_probability
        self.max_shadow_positions = max_shadow_positions
        self._ensure_table()
        self._open_count = self._count_open()

    def _ensure_table(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS random_baseline (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                exit_reason TEXT,
                pnl_pct REAL,
                status TEXT NOT NULL DEFAULT 'open',
                peak_price REAL,
                trough_price REAL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rb_status ON random_baseline(status)"
        )
        conn.commit()
        conn.close()

    def _count_open(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cnt = conn.execute(
            "SELECT COUNT(*) FROM random_baseline WHERE status = 'open'"
        ).fetchone()[0]
        conn.close()
        return cnt

    def maybe_enter(self, product_id: str, current_price: float) -> Optional[Dict[str, Any]]:
        """Randomly decide whether to shadow-enter a position."""
        if self._open_count >= self.max_shadow_positions:
            return None
        if current_price <= 0:
            return None
        if random.random() > self.entry_probability:
            return None

        # Check if already holding this product
        conn = sqlite3.connect(self.db_path)
        existing = conn.execute(
            "SELECT id FROM random_baseline WHERE product_id = ? AND status = 'open'",
            (product_id,),
        ).fetchone()
        if existing:
            conn.close()
            return None

        direction = random.choice(["BUY", "SELL"])
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO random_baseline "
            "(product_id, direction, entry_price, entry_time, status, peak_price, trough_price) "
            "VALUES (?, ?, ?, ?, 'open', ?, ?)",
            (product_id, direction, current_price, now, current_price, current_price),
        )
        conn.commit()
        conn.close()
        self._open_count += 1

        logger.debug(
            f"RANDOM BASELINE: Shadow {direction} {product_id} @ ${current_price:.4f}"
        )
        return {"product_id": product_id, "direction": direction, "price": current_price}

    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Update open shadow positions. Apply same exit rules as real system.

        Exit rules (mirroring reeval engine):
          - Stop loss: -5%
          - Trailing stop: 15% from peak
          - Max hold: 24h
        """
        exits = []
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT id, product_id, direction, entry_price, entry_time, "
            "peak_price, trough_price FROM random_baseline WHERE status = 'open'"
        ).fetchall()

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        for row in rows:
            rid, pid, direction, entry_price, entry_time, peak, trough = row
            price = current_prices.get(pid)
            if price is None or price <= 0:
                continue

            # Update peak/trough
            new_peak = max(peak or price, price)
            new_trough = min(trough or price, price)

            # P&L calculation
            if direction == "BUY":
                pnl_pct = (price / entry_price - 1.0) * 100
            else:
                pnl_pct = (1.0 - price / entry_price) * 100

            # Exit rules
            exit_reason = None

            # Stop loss: -5%
            if pnl_pct <= -5.0:
                exit_reason = "stop_loss"

            # Trailing stop: 15% from peak (only when profitable)
            if exit_reason is None and pnl_pct > 0:
                if direction == "BUY":
                    drop_from_peak = (new_peak - price) / new_peak * 100 if new_peak > 0 else 0
                else:
                    rise_from_trough = (price - new_trough) / new_trough * 100 if new_trough > 0 else 0
                    drop_from_peak = rise_from_trough
                if drop_from_peak >= 15.0:
                    exit_reason = "trailing_stop"

            # Max hold: 24h
            if exit_reason is None:
                try:
                    entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    hours_held = (now - entry_dt).total_seconds() / 3600
                    if hours_held >= 24:
                        exit_reason = "max_hold_24h"
                except (ValueError, AttributeError):
                    pass

            if exit_reason:
                conn.execute(
                    "UPDATE random_baseline SET status='closed', exit_price=?, "
                    "exit_time=?, exit_reason=?, pnl_pct=?, peak_price=?, trough_price=? "
                    "WHERE id=?",
                    (price, now_iso, exit_reason, round(pnl_pct, 4), new_peak, new_trough, rid),
                )
                exits.append({
                    "product_id": pid, "direction": direction,
                    "pnl_pct": round(pnl_pct, 2), "reason": exit_reason,
                })
                self._open_count -= 1
            else:
                conn.execute(
                    "UPDATE random_baseline SET peak_price=?, trough_price=? WHERE id=?",
                    (new_peak, new_trough, rid),
                )

        conn.commit()
        conn.close()
        return exits

    def get_stats(self) -> Dict[str, Any]:
        """Return baseline stats for dashboard comparison."""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM random_baseline").fetchone()[0]
        closed = conn.execute(
            "SELECT COUNT(*), COALESCE(AVG(pnl_pct), 0), "
            "SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) "
            "FROM random_baseline WHERE status = 'closed'"
        ).fetchone()
        open_cnt = conn.execute(
            "SELECT COUNT(*) FROM random_baseline WHERE status = 'open'"
        ).fetchone()[0]
        conn.close()

        closed_count = closed[0] or 0
        avg_pnl = closed[1] or 0
        wins = closed[2] or 0
        win_rate = (wins / closed_count * 100) if closed_count > 0 else 0

        return {
            "total_trades": total,
            "open_count": open_cnt,
            "closed_count": closed_count,
            "avg_pnl_pct": round(avg_pnl, 4),
            "win_rate": round(win_rate, 1),
            "entry_probability": self.entry_probability,
        }
