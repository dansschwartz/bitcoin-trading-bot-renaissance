"""
Performance Tracking & Analytics — logs every trade, tracks P&L
per strategy, and provides real-time analytics.

Every signal detected, order placed, fill received, and cost estimate
vs actual is logged. This data is how the system improves.
"""
import json
import logging
import sqlite3
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("arb.tracking")


class PerformanceTracker:

    def __init__(self, db_path: str = "data/arbitrage.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # In-memory aggregates
        self._trades: deque = deque(maxlen=10000)
        self._by_strategy: Dict[str, dict] = {
            "cross_exchange": self._empty_stats(),
            "funding_rate": self._empty_stats(),
            "triangular": self._empty_stats(),
        }
        self._by_pair: Dict[str, dict] = {}
        self._hourly_pnl: Dict[str, Decimal] = {}  # hour_key -> pnl
        self._total_profit = Decimal('0')
        self._total_trades = 0
        self._start_time = datetime.utcnow()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS arb_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                strategy TEXT,
                symbol TEXT,
                buy_exchange TEXT,
                sell_exchange TEXT,
                status TEXT,
                buy_price REAL,
                sell_price REAL,
                quantity REAL,
                gross_spread_bps REAL,
                net_spread_bps REAL,
                expected_profit_usd REAL,
                actual_profit_usd REAL,
                buy_fee REAL,
                sell_fee REAL,
                estimated_cost_bps REAL,
                realized_cost_bps REAL,
                confidence REAL,
                timestamp TEXT,
                book_depth_json TEXT
            )
        """)
        # Add book_depth_json column if missing (existing DBs)
        try:
            conn.execute("ALTER TABLE arb_trades ADD COLUMN book_depth_json TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Add individual leg depth columns (for efficient queries)
        for col in ("leg1_depth_usd", "leg2_depth_usd", "leg3_depth_usd"):
            try:
                conn.execute(f"ALTER TABLE arb_trades ADD COLUMN {col} REAL")
            except sqlite3.OperationalError:
                pass  # Column already exists
        try:
            conn.execute("ALTER TABLE arb_trades ADD COLUMN exchange TEXT DEFAULT 'mexc'")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Dynamic sizing and leg count columns
        for col, default in [("trade_size_usd", "REAL"), ("leg_count", "INTEGER DEFAULT 3"),
                              ("bottleneck_depth_usd", "REAL DEFAULT 0"),
                              ("sizing_reason", "TEXT DEFAULT ''")]:
            try:
                conn.execute(f"ALTER TABLE arb_trades ADD COLUMN {col} {default}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS arb_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT,
                strategy TEXT,
                symbol TEXT,
                gross_spread_bps REAL,
                net_spread_bps REAL,
                approved INTEGER,
                executed INTEGER,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS arb_daily_summary (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                filled_trades INTEGER,
                total_profit_usd REAL,
                cross_exchange_profit REAL,
                funding_rate_profit REAL,
                triangular_profit REAL,
                win_rate REAL,
                avg_profit_per_trade REAL,
                max_drawdown_usd REAL
            )
        """)
        # Path performance tracking for fill rate optimization
        conn.execute("""
            CREATE TABLE IF NOT EXISTS path_performance (
                path TEXT PRIMARY KEY,
                total_attempts INTEGER DEFAULT 0,
                total_fills INTEGER DEFAULT 0,
                total_profit REAL DEFAULT 0,
                avg_profit_per_fill REAL DEFAULT 0,
                fill_rate REAL DEFAULT 0,
                last_attempt_ts REAL DEFAULT 0,
                last_fill_ts REAL DEFAULT 0,
                priority_score REAL DEFAULT 1.0,
                is_deprioritized INTEGER DEFAULT 0,
                updated_at REAL DEFAULT 0
            )
        """)
        # Add path column to arb_trades if missing (currency path, e.g. "USDT→BTC→ETH→USDT")
        try:
            conn.execute("ALTER TABLE arb_trades ADD COLUMN path TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()
        conn.close()

    def record_trade(self, execution_result):
        """Record a completed trade execution."""
        signal = execution_result.signal
        trade = {
            'trade_id': execution_result.trade_id,
            'strategy': signal.signal_type,
            'symbol': signal.symbol,
            'buy_exchange': signal.buy_exchange,
            'sell_exchange': signal.sell_exchange,
            'status': execution_result.status,
            'buy_price': float(signal.buy_price),
            'sell_price': float(signal.sell_price),
            'quantity': float(signal.recommended_quantity),
            'gross_spread_bps': float(signal.gross_spread_bps),
            'net_spread_bps': float(signal.net_spread_bps),
            'expected_profit_usd': float(signal.expected_profit_usd),
            'actual_profit_usd': float(execution_result.actual_profit_usd),
            'buy_fee': float(execution_result.buy_result.fee_amount if execution_result.buy_result else 0),
            'sell_fee': float(execution_result.sell_result.fee_amount if execution_result.sell_result else 0),
            'estimated_cost_bps': float(signal.total_cost_bps),
            'realized_cost_bps': float(execution_result.realized_cost_bps),
            'confidence': float(signal.confidence),
            'timestamp': datetime.utcnow().isoformat(),
            'book_depth_json': None,
            'leg1_depth_usd': 0.0,
            'leg2_depth_usd': 0.0,
            'leg3_depth_usd': 0.0,
        }

        self._trades.append(trade)
        self._total_trades += 1

        if execution_result.status == "filled":
            profit = execution_result.actual_profit_usd
            self._total_profit += profit

            # Update strategy stats
            strategy = signal.signal_type
            if strategy in self._by_strategy:
                stats = self._by_strategy[strategy]
                stats['trades'] += 1
                stats['profit_usd'] += float(profit)
                if profit > 0:
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1

            # Update pair stats
            if signal.symbol not in self._by_pair:
                self._by_pair[signal.symbol] = self._empty_stats()
            pair_stats = self._by_pair[signal.symbol]
            pair_stats['trades'] += 1
            pair_stats['profit_usd'] += float(profit)
            if profit > 0:
                pair_stats['wins'] += 1
            else:
                pair_stats['losses'] += 1

            # Hourly PnL
            hour_key = datetime.utcnow().strftime("%Y-%m-%d %H:00")
            self._hourly_pnl[hour_key] = self._hourly_pnl.get(hour_key, Decimal('0')) + profit

        # Persist to SQLite
        self._persist_trade(trade)

    def record_triangular_trade(self, result, opportunity=None) -> None:
        """Record a completed triangular arbitrage execution.

        Maps TriExecutionResult fields to the arb_trades schema so the
        dashboard picks up triangular P&L automatically.
        """
        # Build a human-readable symbol from the leg symbols
        leg_symbols = [leg.symbol for leg in result.legs] if result.legs else []
        symbol = "→".join(leg_symbols) if leg_symbols else "triangular"

        # Build currency path (e.g. "USDT→BTC→ETH→USDT") for path performance tracking
        currency_path = ""
        if opportunity and hasattr(opportunity, 'path'):
            currencies = [opportunity.start_currency]
            for _, _, next_curr in opportunity.path:
                currencies.append(next_curr)
            currency_path = "→".join(currencies)

        # For failed/unwound trades, profit_usd = end(0) - start(500) = -500
        # which is wrong — the unwind recovers capital. Use 0 for non-filled.
        if result.status != "filled":
            effective_profit = Decimal('0')
        else:
            effective_profit = result.profit_usd

        # Spread / cost in bps (relative to start amount)
        start = float(result.start_amount) if result.start_amount else 1.0
        profit = float(effective_profit)
        fees = float(result.total_fees_usd)
        spread_bps = (profit / start) * 10000 if start > 0 else 0.0
        cost_bps = (fees / start) * 10000 if start > 0 else 0.0

        # Expected profit at detection time (from opportunity's profit_bps)
        expected_profit = 0.0
        if opportunity and hasattr(opportunity, 'profit_bps'):
            expected_profit = float(opportunity.profit_bps) / 10000.0 * start
        else:
            expected_profit = profit  # Fallback to actual

        # Extract leg-1 and leg-3 fill prices for buy_price / sell_price
        buy_price = 0.0
        sell_price = 0.0
        if result.legs:
            leg1 = result.legs[0]
            if leg1.order_result and leg1.order_result.average_fill_price:
                buy_price = float(leg1.order_result.average_fill_price)
            if len(result.legs) >= 3:
                leg3 = result.legs[2]
                if leg3.order_result and leg3.order_result.average_fill_price:
                    sell_price = float(leg3.order_result.average_fill_price)

        # Serialize book depth as JSON if available + extract per-leg depths
        depth_json = None
        leg1_depth = 0.0
        leg2_depth = 0.0
        leg3_depth = 0.0
        if hasattr(result, 'book_depth') and result.book_depth:
            try:
                depth_json = json.dumps(result.book_depth)
            except Exception:
                pass
            legs = result.book_depth.get('legs', [])
            if len(legs) >= 1:
                leg1_depth = legs[0].get('depth_usd_top5', 0.0)
            if len(legs) >= 2:
                leg2_depth = legs[1].get('depth_usd_top5', 0.0)
            if len(legs) >= 3:
                leg3_depth = legs[2].get('depth_usd_top5', 0.0)

        # Extract bottleneck depth and sizing reason from result
        bottleneck_depth = getattr(result, 'bottleneck_depth_usd', 0.0)
        sizing_reason_str = getattr(result, 'sizing_reason', '')

        trade = {
            'trade_id': result.trade_id,
            'strategy': 'triangular',
            'symbol': symbol,
            'buy_exchange': result.legs[0].order_result.exchange if result.legs and result.legs[0].order_result else 'mexc',
            'sell_exchange': result.legs[0].order_result.exchange if result.legs and result.legs[0].order_result else 'mexc',
            'status': result.status,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'quantity': float(result.start_amount),
            'gross_spread_bps': spread_bps + cost_bps,  # before fees
            'net_spread_bps': spread_bps,                # after fees
            'expected_profit_usd': expected_profit,
            'actual_profit_usd': profit,
            'buy_fee': fees / 2,
            'sell_fee': fees / 2,
            'estimated_cost_bps': cost_bps,
            'realized_cost_bps': cost_bps,
            'confidence': 1.0,
            'timestamp': result.timestamp.isoformat(),
            'book_depth_json': depth_json,
            'leg1_depth_usd': leg1_depth,
            'leg2_depth_usd': leg2_depth,
            'leg3_depth_usd': leg3_depth,
            'trade_size_usd': float(result.start_amount),
            'leg_count': len(result.legs) if result.legs else 3,
            'path': currency_path,
            'bottleneck_depth_usd': bottleneck_depth,
            'sizing_reason': sizing_reason_str,
        }

        self._trades.append(trade)
        self._total_trades += 1

        if result.status == "filled":
            profit_dec = result.profit_usd
            self._total_profit += profit_dec

            stats = self._by_strategy["triangular"]
            stats['trades'] += 1
            stats['profit_usd'] += profit
            if profit > 0:
                stats['wins'] += 1
            else:
                stats['losses'] += 1

            # Pair stats
            if symbol not in self._by_pair:
                self._by_pair[symbol] = self._empty_stats()
            pair_stats = self._by_pair[symbol]
            pair_stats['trades'] += 1
            pair_stats['profit_usd'] += profit
            if profit > 0:
                pair_stats['wins'] += 1
            else:
                pair_stats['losses'] += 1

            # Hourly PnL
            hour_key = datetime.utcnow().strftime("%Y-%m-%d %H:00")
            self._hourly_pnl[hour_key] = self._hourly_pnl.get(hour_key, Decimal('0')) + profit_dec

        self._persist_trade(trade)

        # Update path performance stats for fill rate optimization
        if currency_path:
            self.update_path_performance(
                currency_path,
                filled=(result.status == "filled"),
                profit=profit,
            )

        logger.info(
            f"Recorded triangular trade {result.trade_id}: "
            f"status={result.status} profit=${profit:.4f} "
            f"expected=${expected_profit:.4f} path={currency_path}"
        )

    def record_signal(self, signal, approved: bool, executed: bool):
        """Record a detected signal for analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO arb_signals (signal_id, strategy, symbol, gross_spread_bps, "
                "net_spread_bps, approved, executed, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    signal.signal_id, signal.signal_type, signal.symbol,
                    float(signal.gross_spread_bps), float(signal.net_spread_bps),
                    int(approved), int(executed), datetime.utcnow().isoformat(),
                )
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Signal record error: {e}")

    def _persist_trade(self, trade: dict):
        try:
            conn = sqlite3.connect(self.db_path)
            cols = ", ".join(trade.keys())
            placeholders = ", ".join("?" * len(trade))
            conn.execute(
                f"INSERT OR REPLACE INTO arb_trades ({cols}) VALUES ({placeholders})",
                tuple(trade.values()),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Trade persist error: {e}")

    def get_summary(self) -> dict:
        uptime = (datetime.utcnow() - self._start_time).total_seconds() / 3600
        filled = [t for t in self._trades if t['status'] == 'filled']
        wins = [t for t in filled if t['actual_profit_usd'] > 0]

        return {
            "uptime_hours": round(uptime, 2),
            "total_trades": self._total_trades,
            "total_fills": len(filled),
            "total_profit_usd": float(self._total_profit),
            "win_rate": len(wins) / max(1, len(filled)),
            "avg_profit_per_fill": float(self._total_profit / max(1, len(filled))),
            "by_strategy": self._by_strategy,
            "top_pairs": dict(sorted(
                self._by_pair.items(),
                key=lambda x: x[1]['profit_usd'],
                reverse=True
            )[:5]),
            "recent_hourly_pnl": dict(sorted(self._hourly_pnl.items())[-24:]),
        }

    def update_path_performance(self, path: str, filled: bool, profit: float = 0.0) -> None:
        """Update rolling path performance stats after every attempt."""
        import time as _time
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # Query last 200 attempts for this path to compute rolling stats
            recent = conn.execute(
                "SELECT status, actual_profit_usd FROM arb_trades "
                "WHERE path = ? ORDER BY id DESC LIMIT 200",
                (path,),
            ).fetchall()

            attempts = len(recent)
            fills = sum(1 for r in recent if r['status'] == 'filled')
            total_profit = sum(
                r['actual_profit_usd'] for r in recent
                if r['status'] == 'filled' and r['actual_profit_usd']
            )
            fill_rate = fills / attempts if attempts > 0 else 0.0
            avg_profit = total_profit / fills if fills > 0 else 0.0

            # Priority score: fill_rate × avg_profit
            # Need 20+ samples for meaningful score
            priority_score = fill_rate * avg_profit if attempts >= 20 else 1.0
            is_deprioritized = 1 if (fill_rate < 0.30 and attempts >= 50) else 0

            now = _time.time()
            conn.execute(
                "INSERT OR REPLACE INTO path_performance "
                "(path, total_attempts, total_fills, total_profit, avg_profit_per_fill, "
                "fill_rate, last_attempt_ts, last_fill_ts, priority_score, "
                "is_deprioritized, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    path, attempts, fills, total_profit, avg_profit,
                    fill_rate, now,
                    now if filled else (conn.execute(
                        "SELECT last_fill_ts FROM path_performance WHERE path = ?", (path,)
                    ).fetchone() or (0.0,))[0],
                    priority_score, is_deprioritized, now,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Path performance update error: {e}")

    def get_path_performance(self, path: str) -> Optional[dict]:
        """Get performance stats for a specific path."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM path_performance WHERE path = ?", (path,)
            ).fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception:
            return None

    def get_all_path_performance(self) -> List[dict]:
        """Get performance stats for all tracked paths."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM path_performance ORDER BY priority_score DESC"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def _empty_stats(self) -> dict:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'profit_usd': 0.0}
