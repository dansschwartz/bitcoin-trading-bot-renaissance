# database_manager.py
import sqlite3
import json
import logging
import os
import numpy as np
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class MarketData:
    """Market data structure"""
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    source: str = "coinbase"
    product_id: str = "BTC-USD"


@dataclass
class SentimentData:
    """Sentiment data structure"""
    overall_sentiment: float
    twitter_sentiment: float
    reddit_sentiment: float
    fear_greed_index: int
    confidence: float
    timestamp: datetime
    sources: Dict[str, Any]


class DatabaseManager:
    def __init__(self, config: Dict):
        self.db_path = config['path']
        self.backup_interval = config.get('backup_interval', 3600)
        self.logger = logging.getLogger(f"{__name__}.DatabaseManager")

    @contextmanager
    def _get_connection(self):
        """Context manager for safe SQLite connections with WAL mode and timeout."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def init_database(self):
        """Initialize database with all required tables"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT, price REAL NOT NULL,
                volume REAL NOT NULL, bid REAL NOT NULL, ask REAL NOT NULL,
                spread REAL NOT NULL, timestamp TEXT NOT NULL, source TEXT NOT NULL,
                product_id TEXT DEFAULT 'BTC-USD')''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id INTEGER UNIQUE,
                product_id TEXT NOT NULL, t_entry TEXT NOT NULL, entry_price REAL NOT NULL,
                t_exit TEXT NOT NULL, exit_price REAL NOT NULL, horizon_min INTEGER NOT NULL,
                ret_pct REAL NOT NULL, correct INTEGER NOT NULL,
                FOREIGN KEY (decision_id) REFERENCES decisions(id))''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT, overall_sentiment REAL NOT NULL,
                twitter_sentiment REAL NOT NULL, reddit_sentiment REAL NOT NULL,
                fear_greed_index INTEGER NOT NULL, confidence REAL NOT NULL,
                timestamp TEXT NOT NULL, sources TEXT NOT NULL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS onchain_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT, active_addresses INTEGER,
                transaction_count INTEGER, hash_rate REAL, network_health REAL,
                timestamp TEXT NOT NULL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                product_id TEXT NOT NULL, action TEXT NOT NULL, confidence REAL NOT NULL,
                position_size REAL NOT NULL, weighted_signal REAL NOT NULL,
                reasoning TEXT NOT NULL, feature_vector TEXT, vae_loss REAL,
                hmm_regime TEXT)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                product_id TEXT NOT NULL, side TEXT NOT NULL, size REAL NOT NULL,
                price REAL NOT NULL, status TEXT NOT NULL, algo_used TEXT,
                slippage REAL, execution_time REAL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                product_id TEXT NOT NULL, model_name TEXT NOT NULL,
                prediction REAL NOT NULL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS open_positions (
                position_id TEXT PRIMARY KEY, product_id TEXT NOT NULL,
                side TEXT NOT NULL, size REAL NOT NULL, entry_price REAL NOT NULL,
                stop_loss_price REAL, take_profit_price REAL,
                opened_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'OPEN')''')

            # Medallion Intelligence: Expanded data capture tables
            cursor.execute('''CREATE TABLE IF NOT EXISTS funding_rate_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
                funding_rate REAL NOT NULL, exchange TEXT,
                predicted_rate REAL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS open_interest_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
                open_interest REAL NOT NULL, change_24h_pct REAL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS liquidation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
                direction TEXT NOT NULL, risk_score REAL,
                funding_rate_percentile REAL, long_short_ratio REAL,
                recommended_action TEXT)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS ghost_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, param_set TEXT NOT NULL,
                product_id TEXT NOT NULL, action TEXT NOT NULL,
                entry_price REAL NOT NULL, exit_price REAL,
                pnl_pct REAL, exit_reason TEXT, cycles_held INTEGER)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS signal_throttle_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, action TEXT NOT NULL,
                signal_name TEXT NOT NULL, accuracy REAL,
                sample_count INTEGER, product_id TEXT)''')

            conn.commit()
            self.logger.info("Database initialized successfully with expanded metrics support")

    async def store_market_data(self, data: MarketData):
        """Store market data"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                ts = data.timestamp.isoformat() if hasattr(data.timestamp, 'isoformat') else datetime.now(timezone.utc).isoformat()

                cursor.execute('''
                    INSERT INTO market_data (price, volume, bid, ask, spread, timestamp, source, product_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.price,
                    data.volume,
                    data.bid,
                    data.ask,
                    data.spread,
                    ts,
                    data.source,
                    data.product_id
                ))

                conn.commit()
            self.logger.debug("Market data stored successfully")

        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")

    async def store_sentiment_data(self, data: SentimentData):
        """Store sentiment data"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                def json_serial(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(f"Type {type(obj)} not serializable")

                cursor.execute('''
                    INSERT INTO sentiment_data (overall_sentiment, twitter_sentiment, reddit_sentiment,
                                              fear_greed_index, confidence, timestamp, sources)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.overall_sentiment,
                    data.twitter_sentiment,
                    data.reddit_sentiment,
                    data.fear_greed_index,
                    data.confidence,
                    data.timestamp.isoformat(),
                    json.dumps(data.sources, default=json_serial)
                ))

                conn.commit()
            self.logger.debug("Sentiment data stored successfully")

        except Exception as e:
            self.logger.error(f"Error storing sentiment data: {e}")

    # Tables that may be queried via get_recent_data / cleanup_old_data
    ALLOWED_TABLES = frozenset({
        "market_data", "labels", "sentiment_data", "onchain_data",
        "decisions", "trades", "ml_predictions", "open_positions",
        "daily_candles", "data_refresh_log",
    })

    async def get_recent_data(self, table: str, hours: int = 24) -> List[Dict]:
        """Get recent data from specified table"""
        if table not in self.ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {table}")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    f"SELECT * FROM {table} "
                    "WHERE datetime(timestamp) > datetime('now', ? || ' hours') "
                    "ORDER BY timestamp DESC",
                    (f"-{int(hours)}",)
                )

                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving data: {e}")
            return []

    async def cleanup_old_data(self, days: int = 30):
        """Remove data older than specified days"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                tables = ['market_data', 'sentiment_data', 'onchain_data', 'decisions', 'trades', 'ml_predictions']

                for table in tables:
                    if table not in self.ALLOWED_TABLES:
                        continue
                    cursor.execute(
                        f"DELETE FROM {table} "
                        "WHERE datetime(timestamp) < datetime('now', ? || ' days')",
                        (f"-{int(days)}",)
                    )

                conn.commit()
            self.logger.info(f"Cleaned up data older than {days} days")

        except Exception as e:
            self.logger.error(f"Error cleaning up data: {e}")

    async def store_decision(self, decision_data: Dict[str, Any]):
        """Store trading decision with expanded metrics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Use UTC for persistence consistency
                ts = decision_data.get('timestamp')
                if not ts:
                    ts = datetime.now(timezone.utc).isoformat()
                elif isinstance(ts, datetime):
                    ts = ts.isoformat()

                # Handle non-serializable objects in reasoning
                reasoning = decision_data.get('reasoning', {})
                def json_serial(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    raise TypeError(f"Type {type(obj)} not serializable")

                # Extract expanded metrics
                feature_vector = decision_data.get('feature_vector')
                if isinstance(feature_vector, np.ndarray):
                    feature_vector = json.dumps(feature_vector.tolist())
                elif feature_vector is not None:
                    feature_vector = str(feature_vector)

                vae_loss = decision_data.get('vae_loss')
                hmm_regime = decision_data.get('hmm_regime')

                cursor.execute('''
                    INSERT INTO decisions (timestamp, product_id, action, confidence, position_size, weighted_signal, reasoning, feature_vector, vae_loss, hmm_regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ts,
                    decision_data.get('product_id'),
                    decision_data.get('action'),
                    decision_data.get('confidence'),
                    decision_data.get('position_size'),
                    decision_data.get('weighted_signal'),
                    json.dumps(reasoning, default=json_serial),
                    feature_vector,
                    vae_loss,
                    hmm_regime
                ))

                conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing decision: {e}")

    async def store_trade(self, trade_data: Dict[str, Any]):
        """Store executed trade"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                ts = trade_data.get('timestamp')
                if not ts:
                    ts = datetime.now(timezone.utc).isoformat()
                elif isinstance(ts, datetime):
                    ts = ts.isoformat()

                cursor.execute('''
                    INSERT INTO trades (timestamp, product_id, side, size, price, status, algo_used, slippage, execution_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ts,
                    trade_data.get('product_id'),
                    trade_data.get('side'),
                    trade_data.get('size'),
                    trade_data.get('price'),
                    trade_data.get('status'),
                    trade_data.get('algo_used'),
                    trade_data.get('slippage'),
                    trade_data.get('execution_time')
                ))

                conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing trade: {e}")

    async def store_ml_prediction(self, prediction_data: Dict[str, Any]):
        """Store ML model prediction"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                ts = prediction_data.get('timestamp')
                if not ts:
                    ts = datetime.now(timezone.utc).isoformat()
                elif isinstance(ts, datetime):
                    ts = ts.isoformat()

                cursor.execute('''
                    INSERT INTO ml_predictions (timestamp, product_id, model_name, prediction)
                    VALUES (?, ?, ?, ?)
                ''', (
                    ts,
                    prediction_data.get('product_id'),
                    prediction_data.get('model_name'),
                    prediction_data.get('prediction')
                ))

                conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing ML prediction: {e}")

    # ──────────────────────────────────────────────
    #  Position State Recovery
    # ──────────────────────────────────────────────

    async def save_position(self, position_data: Dict[str, Any]):
        """Upsert an open position for state recovery."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO open_positions
                        (position_id, product_id, side, size, entry_price,
                         stop_loss_price, take_profit_price, opened_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position_data['position_id'],
                    position_data['product_id'],
                    position_data['side'],
                    position_data['size'],
                    position_data['entry_price'],
                    position_data.get('stop_loss_price'),
                    position_data.get('take_profit_price'),
                    position_data.get('opened_at', datetime.now(timezone.utc).isoformat()),
                    position_data.get('status', 'OPEN'),
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving position: {e}")

    async def close_position_record(self, position_id: str):
        """Mark a position as CLOSED in the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE open_positions SET status = 'CLOSED' WHERE position_id = ?",
                    (position_id,)
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error closing position record: {e}")

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Retrieve all OPEN positions for state recovery on restart."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM open_positions WHERE status = 'OPEN'")
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []

    async def get_daily_pnl(self, date_str: str) -> float:
        """Sum realized PnL from today's trades (approximated from trade records)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COALESCE(SUM(
                        CASE WHEN side = 'SELL' THEN size * price
                             WHEN side = 'BUY'  THEN -size * price
                             ELSE 0 END
                    ), 0.0)
                    FROM trades
                    WHERE date(timestamp) = ?
                ''', (date_str,))
                result = cursor.fetchone()
                return float(result[0]) if result else 0.0
        except Exception as e:
            self.logger.error(f"Error getting daily PnL: {e}")
            return 0.0