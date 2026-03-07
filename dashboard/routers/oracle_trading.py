"""Oracle Trading Engine dashboard endpoints."""

import json
import logging
import sqlite3
from pathlib import Path

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/oracle-trading", tags=["oracle-trading"])

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "renaissance_bot.db"


def _db():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/status")
async def oracle_trading_status():
    """Full status of all oracle trading wallets."""
    try:
        conn = _db()
        # Get latest snapshot per pair
        rows = conn.execute("""
            SELECT pair, capital, position_open, entry_price,
                   total_trades, winning_trades, total_pnl_usd,
                   win_rate, max_drawdown_pct, status, timestamp
            FROM oracle_wallet_snapshots
            WHERE timestamp = (
                SELECT MAX(timestamp) FROM oracle_wallet_snapshots s2
                WHERE s2.pair = oracle_wallet_snapshots.pair
            )
            ORDER BY pair
        """).fetchall()
        conn.close()

        wallets = {}
        total_capital = 0.0
        total_pnl = 0.0
        total_initial = 0.0
        positions_open = 0

        for r in rows:
            pair = r['pair']
            cap = r['capital'] or 0
            pnl = r['total_pnl_usd'] or 0
            initial = 5000.0  # default wallet size
            wr = r['win_rate'] or 0
            wallets[pair] = {
                'pair': pair,
                'capital': round(cap, 2),
                'initial': initial,
                'pnl_usd': round(pnl, 2),
                'pnl_pct': round((cap - initial) / initial * 100, 2) if initial > 0 else 0,
                'position_open': bool(r['position_open']),
                'entry_price': r['entry_price'] or 0,
                'total_trades': r['total_trades'] or 0,
                'winning_trades': r['winning_trades'] or 0,
                'win_rate': round(wr, 1),
                'max_drawdown_pct': round((r['max_drawdown_pct'] or 0) * 100, 2),
                'status': r['status'] or 'observation',
                'last_signal': 'HOLD',
                'current_price': 0,
                'stop_loss_price': 0,
                'unrealized_pnl': 0,
            }
            total_capital += cap
            total_initial += initial
            total_pnl += pnl
            if r['position_open']:
                positions_open += 1

        return {
            'wallets': wallets,
            'summary': {
                'total_capital': round(total_capital, 2),
                'total_initial': total_initial,
                'total_pnl': round(total_pnl, 2),
                'total_return_pct': round(
                    (total_capital - total_initial) / total_initial * 100, 2
                ) if total_initial > 0 else 0,
                'positions_open': positions_open,
                'total_pairs': len(wallets),
                'active_pairs': sum(
                    1 for w in wallets.values() if w['status'] != 'halted'
                ),
            },
        }

    except Exception as e:
        logger.debug(f"Oracle trading status error: {e}")
        return {'wallets': {}, 'summary': {
            'total_capital': 0, 'total_initial': 0, 'total_pnl': 0,
            'total_return_pct': 0, 'positions_open': 0,
            'total_pairs': 0, 'active_pairs': 0,
        }}


@router.get("/trades")
async def oracle_trading_trades(pair: str = '', limit: int = 50):
    """Trade history, optionally filtered by pair."""
    try:
        conn = _db()
        if pair:
            rows = conn.execute("""
                SELECT pair, action, price, capital_before, capital_after,
                       pnl_usd, pnl_pct, fee_paid, exit_reason, hold_bars,
                       signal, signal_confidence, entry_time, exit_time,
                       timestamp
                FROM oracle_trades
                WHERE pair = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (pair.upper(), limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT pair, action, price, capital_before, capital_after,
                       pnl_usd, pnl_pct, fee_paid, exit_reason, hold_bars,
                       signal, signal_confidence, entry_time, exit_time,
                       timestamp
                FROM oracle_trades
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
        conn.close()

        return [{
            'pair': r['pair'],
            'action': r['action'],
            'price': r['price'],
            'capital_before': r['capital_before'],
            'capital_after': r['capital_after'],
            'pnl_usd': r['pnl_usd'],
            'pnl_pct': r['pnl_pct'],
            'fee_paid': r['fee_paid'],
            'exit_reason': r['exit_reason'],
            'hold_bars': r['hold_bars'],
            'signal': r['signal'],
            'confidence': r['signal_confidence'],
            'entry_time': r['entry_time'],
            'exit_time': r['exit_time'],
            'timestamp': r['timestamp'],
        } for r in rows]

    except Exception as e:
        logger.debug(f"Oracle trades error: {e}")
        return []


@router.get("/equity/{pair}")
async def oracle_equity_curve(pair: str):
    """Equity curve data for one pair."""
    try:
        conn = _db()
        rows = conn.execute("""
            SELECT capital_after, pnl_pct, exit_reason, timestamp
            FROM oracle_trades
            WHERE pair = ? AND action = 'CLOSE'
            ORDER BY timestamp
        """, (pair.upper(),)).fetchall()
        conn.close()

        return [{
            'capital': r['capital_after'],
            'return_pct': r['pnl_pct'],
            'exit_reason': r['exit_reason'],
            'timestamp': r['timestamp'],
        } for r in rows]

    except Exception as e:
        logger.debug(f"Oracle equity error: {e}")
        return []
