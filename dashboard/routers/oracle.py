"""Oracle dashboard endpoints — reads from oracle_signals table."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/oracle", tags=["oracle"])

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "renaissance_bot.db"


@contextmanager
def _db_conn():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@router.get("/status")
async def oracle_status():
    """Get latest oracle signal for all assets + recent history."""
    assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
    signals: Dict[str, Any] = {}

    try:
        with _db_conn() as conn:
            for asset in assets:
                # Latest signal
                row = conn.execute("""
                    SELECT signal, confidence, candle_close, candle_time,
                           timestamp, model_votes
                    FROM oracle_signals
                    WHERE asset = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (asset,)).fetchone()

                if not row:
                    continue

                # History (last 48 hours)
                history = conn.execute("""
                    SELECT timestamp, signal, confidence, candle_close
                    FROM oracle_signals
                    WHERE asset = ? AND timestamp > datetime('now', '-48 hours')
                    ORDER BY timestamp DESC
                """, (asset,)).fetchall()

                signals[asset] = {
                    'current': {
                        'signal': row['signal'],
                        'confidence': row['confidence'],
                        'candle_close': row['candle_close'],
                        'candle_time': row['candle_time'],
                        'timestamp': row['timestamp'],
                        'model_votes': json.loads(row['model_votes'])
                                       if row['model_votes'] else [],
                    },
                    'history': [
                        {
                            'timestamp': h['timestamp'],
                            'signal': h['signal'],
                            'confidence': h['confidence'],
                            'candle_close': h['candle_close'],
                        }
                        for h in history
                    ],
                }

        return {'signals': signals}

    except Exception as e:
        return {'signals': {}, 'error': str(e)}


@router.get("/history")
async def oracle_history(asset: str = 'BTCUSDT', hours: int = 168):
    """Get oracle signal history for a specific asset."""
    try:
        with _db_conn() as conn:
            rows = conn.execute("""
                SELECT timestamp, signal, confidence, candle_close,
                       candle_time, model_votes
                FROM oracle_signals
                WHERE asset = ? AND timestamp > datetime('now', ? || ' hours')
                ORDER BY timestamp DESC
            """, (asset, f"-{hours}")).fetchall()

            return [
                {
                    'timestamp': r['timestamp'],
                    'signal': r['signal'],
                    'confidence': r['confidence'],
                    'candle_close': r['candle_close'],
                    'candle_time': r['candle_time'],
                    'model_votes': json.loads(r['model_votes'])
                                   if r['model_votes'] else [],
                }
                for r in rows
            ]

    except Exception as e:
        return {'error': str(e)}
