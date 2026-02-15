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
                timestamp TEXT
            )
        """)
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
            conn.execute(
                "INSERT OR REPLACE INTO arb_trades "
                "(trade_id, strategy, symbol, buy_exchange, sell_exchange, status, "
                "buy_price, sell_price, quantity, gross_spread_bps, net_spread_bps, "
                "expected_profit_usd, actual_profit_usd, buy_fee, sell_fee, "
                "estimated_cost_bps, realized_cost_bps, confidence, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def _empty_stats(self) -> dict:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'profit_usd': 0.0}
