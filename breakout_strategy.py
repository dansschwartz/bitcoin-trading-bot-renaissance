"""
Breakout Strategy — Separate $2,000 wallet that catches parabolic moves.

Entry: breakout_score >= 60, volume surge >= 5x, price_change >= 10%, near 24h high.
Exit: -10% stop loss, 48h sideways, 25% trailing stop from peak.
Bet size: max($100, bankroll / 20). Max 10 open positions. 24h cooldown per symbol.
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

INITIAL_BANKROLL = 2000.0
DEFAULT_BET_SIZE = 100.0
MAX_POSITIONS = 10
COOLDOWN_HOURS = 24
STOP_LOSS_PCT = -10.0
SIDEWAYS_HOURS = 48
SIDEWAYS_THRESHOLD_PCT = 5.0
TRAILING_STOP_PCT = 25.0
MIN_BREAKOUT_SCORE = 60.0
MIN_VOLUME_SURGE = 5.0
MIN_PRICE_CHANGE_PCT = 10.0
EXCLUDED_BASES = {"BTC", "ETH"}  # Whole-market movers


@contextmanager
def _conn(db_path: str):
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class BreakoutStrategy:
    """Separate wallet strategy that catches parabolic breakouts with $100 bets."""

    def __init__(
        self,
        db_path: str = "data/renaissance_bot.db",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.db_path = db_path
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        self.bankroll = INITIAL_BANKROLL
        self.open_positions: List[Dict[str, Any]] = []

        self._ensure_tables()
        self._load_state()

    # ══════════════════════════════════════════════════════════════
    # DB SETUP
    # ══════════════════════════════════════════════════════════════

    def _ensure_tables(self) -> None:
        """Create breakout_bets and breakout_wallet tables if they don't exist."""
        with _conn(self.db_path) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS breakout_bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    direction TEXT NOT NULL DEFAULT 'long',
                    status TEXT NOT NULL DEFAULT 'open',
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    bet_size_usd REAL NOT NULL,
                    current_price REAL,
                    peak_price REAL NOT NULL,
                    peak_gain_pct REAL NOT NULL DEFAULT 0.0,
                    pnl_usd REAL DEFAULT 0.0,
                    pnl_pct REAL DEFAULT 0.0,
                    entry_score REAL,
                    entry_volume_surge REAL,
                    entry_price_change_pct REAL,
                    exit_reason TEXT,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    last_significant_move_at TEXT,
                    last_updated TEXT
                )
            """)
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_breakout_bets_status "
                "ON breakout_bets(status)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_breakout_bets_symbol "
                "ON breakout_bets(symbol)"
            )

            c.execute("""
                CREATE TABLE IF NOT EXISTS breakout_wallet (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    bankroll_after REAL NOT NULL,
                    bet_id INTEGER,
                    symbol TEXT,
                    detail TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_breakout_wallet_ts "
                "ON breakout_wallet(timestamp)"
            )
            c.commit()

    def _load_state(self) -> None:
        """Load bankroll and open positions from DB."""
        with _conn(self.db_path) as c:
            # Load bankroll from latest wallet event
            row = c.execute(
                "SELECT bankroll_after FROM breakout_wallet "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                self.bankroll = row["bankroll_after"]
            else:
                # First run — seed the wallet
                self.bankroll = INITIAL_BANKROLL
                now = datetime.now(timezone.utc).isoformat()
                c.execute(
                    "INSERT INTO breakout_wallet "
                    "(event_type, amount, bankroll_after, detail, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("seed", INITIAL_BANKROLL, INITIAL_BANKROLL,
                     "Initial breakout strategy bankroll", now),
                )
                c.commit()

            # Load open positions
            rows = c.execute(
                "SELECT * FROM breakout_bets WHERE status = 'open'"
            ).fetchall()
            self.open_positions = [dict(r) for r in rows]

        self.logger.info(
            f"BREAKOUT STRATEGY loaded: bankroll=${self.bankroll:.2f}, "
            f"{len(self.open_positions)} open positions, "
            f"min_score={MIN_BREAKOUT_SCORE}"
        )

    # ══════════════════════════════════════════════════════════════
    # MAIN CYCLE — called by the bot every cycle
    # ══════════════════════════════════════════════════════════════

    def execute_cycle(
        self,
        breakout_signals: list,
        current_prices: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Run one cycle of the breakout strategy.
        Returns summary dict with actions taken.
        """
        actions = {"exits": [], "entries": [], "managed": 0}

        # Step 1: Manage open positions (exits)
        for pos in list(self.open_positions):
            pid = pos["product_id"]
            price = current_prices.get(pid)
            if price is None:
                # Try symbol format
                sym = pos["symbol"].replace("USDT", "-USD")
                price = current_prices.get(sym)
            if price is None:
                continue

            exit_result = self._manage_position(pos, price)
            actions["managed"] += 1
            if exit_result:
                actions["exits"].append(exit_result)

        # Step 2: Look for new entries
        if len(self.open_positions) < MAX_POSITIONS:
            for signal in breakout_signals:
                if len(self.open_positions) >= MAX_POSITIONS:
                    break
                entry_result = self._try_enter(signal, current_prices)
                if entry_result:
                    actions["entries"].append(entry_result)

        return actions

    # ══════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT — 3 EXIT RULES
    # ══════════════════════════════════════════════════════════════

    def _manage_position(
        self, pos: Dict[str, Any], current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Check 3 exit rules on an open position. Returns exit info or None."""
        entry_price = pos["entry_price"]
        peak_price = pos["peak_price"]
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        # Update peak
        if current_price > peak_price:
            peak_price = current_price
            # This is also a significant move
            with _conn(self.db_path) as c:
                c.execute(
                    "UPDATE breakout_bets SET peak_price = ?, "
                    "peak_gain_pct = ?, last_significant_move_at = ?, "
                    "last_updated = ? WHERE id = ?",
                    (
                        peak_price,
                        round((peak_price / entry_price - 1) * 100, 2),
                        now_iso,
                        now_iso,
                        pos["id"],
                    ),
                )
                c.commit()
            pos["peak_price"] = peak_price
            pos["last_significant_move_at"] = now_iso

        # Current P&L
        pnl_pct = (current_price / entry_price - 1) * 100

        # Update current price in DB
        with _conn(self.db_path) as c:
            c.execute(
                "UPDATE breakout_bets SET current_price = ?, pnl_pct = ?, "
                "pnl_usd = ?, last_updated = ? WHERE id = ?",
                (
                    current_price,
                    round(pnl_pct, 2),
                    round(pos["bet_size_usd"] * pnl_pct / 100, 2),
                    now_iso,
                    pos["id"],
                ),
            )
            c.commit()

        # ── Rule 1: Stop loss — -10% from entry ──
        if pnl_pct <= STOP_LOSS_PCT:
            return self._exit_position(pos, current_price, "stop_loss")

        # ── Rule 2: Sideways — 48h without >5% move from last significant move ──
        last_move_str = pos.get("last_significant_move_at") or pos["opened_at"]
        try:
            last_move = datetime.fromisoformat(last_move_str.replace("Z", "+00:00"))
            if last_move.tzinfo is None:
                last_move = last_move.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            last_move = now

        hours_since_move = (now - last_move).total_seconds() / 3600
        if hours_since_move >= SIDEWAYS_HOURS:
            # Check if price moved significantly from entry
            move_from_entry = abs(pnl_pct)
            if move_from_entry < SIDEWAYS_THRESHOLD_PCT:
                return self._exit_position(pos, current_price, "sideways_48h")

        # ── Rule 3: Trailing stop — 25% drop from peak, only when in profit ──
        if peak_price > entry_price:  # Only when position has been profitable
            drop_from_peak_pct = (peak_price - current_price) / peak_price * 100
            if drop_from_peak_pct >= TRAILING_STOP_PCT:
                return self._exit_position(pos, current_price, "trailing_stop")

        return None

    # ══════════════════════════════════════════════════════════════
    # ENTRY LOGIC
    # ══════════════════════════════════════════════════════════════

    def _try_enter(
        self, signal: Any, current_prices: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Try to enter a breakout position. Returns entry info or None."""
        # Extract signal fields
        symbol = getattr(signal, "symbol", "")
        product_id = getattr(signal, "product_id", "")
        score = getattr(signal, "breakout_score", 0)
        price_change_pct = getattr(signal, "price_change_pct", 0)
        direction = getattr(signal, "direction", "")
        price = getattr(signal, "price", 0)
        details = getattr(signal, "details", {})

        # ── Filter 1: Score >= 75 ──
        if score < MIN_BREAKOUT_SCORE:
            return None

        # ── Filter 2: Bullish only ──
        if direction != "bullish":
            return None

        # ── Filter 3: Price change >= 10% ──
        if abs(price_change_pct) < MIN_PRICE_CHANGE_PCT:
            return None

        # ── Filter 4: Volume surge >= 5x ──
        volume_surge = details.get("volume_surge", 0)
        volume_score = getattr(signal, "volume_score", 0)
        # If volume_surge not in details, infer from volume_score
        # volume_score 30 = 5x, 22 = 3x, 15 = 2x
        if volume_surge < MIN_VOLUME_SURGE and volume_score < 30:
            return None

        # ── Filter 5: Near 24h high (not a dump recovery) ──
        high_24h = details.get("high_24h", 0)
        if high_24h > 0 and price > 0:
            dist_from_high_pct = (high_24h - price) / high_24h * 100
            if dist_from_high_pct > 5:  # More than 5% below 24h high = dump recovery
                return None

        # ── Filter 6: Exclude BTC/ETH ──
        base = symbol.replace("USDT", "")
        if base in EXCLUDED_BASES:
            return None

        # ── Filter 7: Not already holding this symbol ──
        for pos in self.open_positions:
            if pos["symbol"] == symbol:
                return None

        # ── Filter 8: 24h cooldown per symbol ──
        if self._in_cooldown(symbol):
            return None

        # ── Filter 9: Enough bankroll ──
        bet_size = self._calculate_bet_size()
        if bet_size > self.bankroll:
            return None

        return self._enter_position(
            symbol=symbol,
            product_id=product_id,
            price=price,
            bet_size=bet_size,
            score=score,
            volume_surge=volume_surge if volume_surge >= MIN_VOLUME_SURGE else volume_score,
            price_change_pct=price_change_pct,
        )

    def _in_cooldown(self, symbol: str) -> bool:
        """Check if symbol was traded in last 24h."""
        with _conn(self.db_path) as c:
            row = c.execute(
                "SELECT closed_at FROM breakout_bets "
                "WHERE symbol = ? AND status = 'closed' "
                "ORDER BY id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if not row or not row["closed_at"]:
                return False
            try:
                closed = datetime.fromisoformat(
                    row["closed_at"].replace("Z", "+00:00")
                )
                if closed.tzinfo is None:
                    closed = closed.replace(tzinfo=timezone.utc)
                hours_ago = (
                    datetime.now(timezone.utc) - closed
                ).total_seconds() / 3600
                return hours_ago < COOLDOWN_HOURS
            except (ValueError, AttributeError):
                return False

    def _calculate_bet_size(self) -> float:
        """Calculate bet size: max($100, bankroll / 20)."""
        return max(DEFAULT_BET_SIZE, self.bankroll / 20)

    def _enter_position(
        self,
        symbol: str,
        product_id: str,
        price: float,
        bet_size: float,
        score: float,
        volume_surge: float,
        price_change_pct: float,
    ) -> Dict[str, Any]:
        """Open a breakout bet."""
        now = datetime.now(timezone.utc).isoformat()
        quantity = bet_size / price

        with _conn(self.db_path) as c:
            cur = c.execute(
                "INSERT INTO breakout_bets "
                "(symbol, product_id, direction, status, entry_price, quantity, "
                "bet_size_usd, current_price, peak_price, peak_gain_pct, "
                "entry_score, entry_volume_surge, entry_price_change_pct, "
                "opened_at, last_significant_move_at, last_updated) "
                "VALUES (?, ?, 'long', 'open', ?, ?, ?, ?, ?, 0.0, ?, ?, ?, ?, ?, ?)",
                (
                    symbol, product_id, price, quantity, bet_size,
                    price, price, score, volume_surge, price_change_pct,
                    now, now, now,
                ),
            )
            bet_id = cur.lastrowid

            # Debit wallet
            self.bankroll -= bet_size
            c.execute(
                "INSERT INTO breakout_wallet "
                "(event_type, amount, bankroll_after, bet_id, symbol, detail, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "bet_placed", -bet_size, self.bankroll,
                    bet_id, symbol,
                    f"Entry at ${price:.4f}, score={score:.0f}", now,
                ),
            )
            c.commit()

        pos = {
            "id": bet_id,
            "symbol": symbol,
            "product_id": product_id,
            "direction": "long",
            "status": "open",
            "entry_price": price,
            "quantity": quantity,
            "bet_size_usd": bet_size,
            "current_price": price,
            "peak_price": price,
            "peak_gain_pct": 0.0,
            "entry_score": score,
            "entry_volume_surge": volume_surge,
            "entry_price_change_pct": price_change_pct,
            "opened_at": now,
            "last_significant_move_at": now,
            "last_updated": now,
        }
        self.open_positions.append(pos)

        self.logger.info(
            f"BREAKOUT ENTRY: {symbol} @ ${price:.4f} | "
            f"bet=${bet_size:.0f} | score={score:.0f} | "
            f"change={price_change_pct:+.1f}% | bankroll=${self.bankroll:.2f}"
        )

        return {
            "action": "entry",
            "symbol": symbol,
            "price": price,
            "bet_size": bet_size,
            "score": score,
        }

    def _exit_position(
        self, pos: Dict[str, Any], exit_price: float, reason: str
    ) -> Dict[str, Any]:
        """Close a breakout bet and return proceeds to wallet."""
        now = datetime.now(timezone.utc).isoformat()
        entry_price = pos["entry_price"]
        quantity = pos["quantity"]
        bet_size = pos["bet_size_usd"]

        pnl_pct = (exit_price / entry_price - 1) * 100
        proceeds = quantity * exit_price
        pnl_usd = proceeds - bet_size
        peak_gain_pct = (pos["peak_price"] / entry_price - 1) * 100

        with _conn(self.db_path) as c:
            c.execute(
                "UPDATE breakout_bets SET status = 'closed', "
                "exit_price = ?, current_price = ?, pnl_usd = ?, pnl_pct = ?, "
                "peak_gain_pct = ?, exit_reason = ?, closed_at = ?, "
                "last_updated = ? WHERE id = ?",
                (
                    exit_price, exit_price, round(pnl_usd, 2),
                    round(pnl_pct, 2), round(peak_gain_pct, 2),
                    reason, now, now, pos["id"],
                ),
            )

            # Credit wallet with proceeds
            self.bankroll += proceeds
            c.execute(
                "INSERT INTO breakout_wallet "
                "(event_type, amount, bankroll_after, bet_id, symbol, detail, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "bet_closed", proceeds, self.bankroll,
                    pos["id"], pos["symbol"],
                    f"Exit ({reason}) at ${exit_price:.4f}, P&L={pnl_pct:+.1f}%", now,
                ),
            )
            c.commit()

        # Remove from open positions
        self.open_positions = [p for p in self.open_positions if p["id"] != pos["id"]]

        self.logger.info(
            f"BREAKOUT EXIT: {pos['symbol']} @ ${exit_price:.4f} | "
            f"reason={reason} | P&L={pnl_pct:+.1f}% (${pnl_usd:+.2f}) | "
            f"peak={peak_gain_pct:+.1f}% | bankroll=${self.bankroll:.2f}"
        )

        return {
            "action": "exit",
            "symbol": pos["symbol"],
            "reason": reason,
            "pnl_pct": round(pnl_pct, 2),
            "pnl_usd": round(pnl_usd, 2),
            "peak_gain_pct": round(peak_gain_pct, 2),
        }

    # ══════════════════════════════════════════════════════════════
    # STATS — for dashboard
    # ══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Return summary stats for dashboard consumption."""
        with _conn(self.db_path) as c:
            # Closed bets
            closed = c.execute(
                "SELECT pnl_usd, pnl_pct, peak_gain_pct, exit_reason "
                "FROM breakout_bets WHERE status = 'closed'"
            ).fetchall()

            total_bets = len(closed) + len(self.open_positions)
            wins = [r for r in closed if r["pnl_usd"] > 0]
            losses = [r for r in closed if r["pnl_usd"] <= 0]
            win_rate = len(wins) / len(closed) * 100 if closed else 0

            total_pnl = sum(r["pnl_usd"] for r in closed)
            avg_winner = (
                sum(r["pnl_pct"] for r in wins) / len(wins) if wins else 0
            )
            avg_loser = (
                sum(r["pnl_pct"] for r in losses) / len(losses) if losses else 0
            )
            biggest_win = max((r["pnl_usd"] for r in closed), default=0)
            biggest_loss = min((r["pnl_usd"] for r in closed), default=0)
            best_peak = max((r["peak_gain_pct"] for r in closed), default=0)

            # Exit reason breakdown
            exit_reasons: Dict[str, int] = {}
            for r in closed:
                reason = r["exit_reason"] or "unknown"
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        return {
            "bankroll": round(self.bankroll, 2),
            "bet_size": round(self._calculate_bet_size(), 2),
            "shots_left": int(self.bankroll / self._calculate_bet_size())
            if self._calculate_bet_size() > 0 else 0,
            "open_count": len(self.open_positions),
            "total_bets": total_bets,
            "closed_bets": len(closed),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 1),
            "avg_winner_pct": round(avg_winner, 1),
            "avg_loser_pct": round(avg_loser, 1),
            "biggest_win": round(biggest_win, 2),
            "biggest_loss": round(biggest_loss, 2),
            "best_peak_gain_pct": round(best_peak, 1),
            "exit_reasons": exit_reasons,
        }
