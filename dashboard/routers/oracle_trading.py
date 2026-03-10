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


@router.get("/open-trades")
async def oracle_open_trades():
    """Open positions with current price and unrealized P&L."""
    try:
        conn = _db()
        # Get wallets with open positions
        snapshots = conn.execute("""
            SELECT pair, capital, entry_price, total_trades, winning_trades,
                   win_rate, status, timestamp
            FROM oracle_wallet_snapshots
            WHERE position_open = 1
            AND timestamp = (
                SELECT MAX(timestamp) FROM oracle_wallet_snapshots s2
                WHERE s2.pair = oracle_wallet_snapshots.pair
            )
            ORDER BY pair
        """).fetchall()

        # Get entry times from the most recent OPEN trade per pair
        open_times = {}
        for snap in snapshots:
            row = conn.execute("""
                SELECT timestamp, signal, signal_confidence
                FROM oracle_trades
                WHERE pair = ? AND action = 'OPEN'
                ORDER BY timestamp DESC LIMIT 1
            """, (snap['pair'],)).fetchone()
            if row:
                open_times[snap['pair']] = {
                    'entry_time': row['timestamp'],
                    'signal': row['signal'],
                    'confidence': row['signal_confidence'],
                }
        conn.close()

        # Fetch current prices from Binance
        import urllib.request
        prices = {}
        try:
            req = urllib.request.Request(
                'https://api.binance.com/api/v3/ticker/price',
                headers={'User-Agent': 'Mozilla/5.0'},
            )
            resp = urllib.request.urlopen(req, timeout=5)
            for item in json.loads(resp.read()):
                prices[item['symbol']] = float(item['price'])
        except Exception as pe:
            logger.debug(f"Binance price fetch error: {pe}")

        result = []
        for snap in snapshots:
            pair = snap['pair']
            entry_price = snap['entry_price'] or 0
            current_price = prices.get(pair, 0)
            unrealized_pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 and current_price > 0 else 0
            # Estimate unrealized $ using 10% of wallet capital as position size (Parente paper)
            capital = snap['capital'] or 5000
            position_size = capital * 0.10
            unrealized_pnl_usd = position_size * (unrealized_pnl_pct / 100) if entry_price > 0 else 0

            entry_info = open_times.get(pair, {})
            entry_time = entry_info.get('entry_time', '')

            # Calculate duration
            hours_open = 0
            if entry_time:
                try:
                    from datetime import datetime, timezone
                    et = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    if et.tzinfo is None:
                        et = et.replace(tzinfo=timezone.utc)
                    hours_open = (datetime.now(timezone.utc) - et).total_seconds() / 3600
                except Exception:
                    pass

            result.append({
                'pair': pair,
                'entry_price': round(entry_price, 6),
                'current_price': round(current_price, 6),
                'unrealized_pnl_usd': round(unrealized_pnl_usd, 2),
                'unrealized_pnl_pct': round(unrealized_pnl_pct, 2),
                'entry_time': entry_time,
                'hours_open': round(hours_open, 1),
                'signal': entry_info.get('signal', ''),
                'confidence': entry_info.get('confidence', 0),
                'capital': round(capital, 2),
                'position_size': round(position_size, 2),
            })

        result.sort(key=lambda x: x['unrealized_pnl_usd'], reverse=True)
        return result

    except Exception as e:
        logger.debug(f"Oracle open trades error: {e}")
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
