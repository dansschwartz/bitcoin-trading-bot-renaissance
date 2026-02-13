# database_manager.py
import sqlite3
import json
import logging
import os
import numpy as np
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

    async def init_database(self):
        """Initialize database with all required tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create market_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                spread REAL NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                product_id TEXT DEFAULT 'BTC-USD'
            )
        ''')

        # Create labels table (Step 19)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id INTEGER UNIQUE,
                product_id TEXT NOT NULL,
                t_entry TEXT NOT NULL,
                entry_price REAL NOT NULL,
                t_exit TEXT NOT NULL,
                exit_price REAL NOT NULL,
                horizon_min INTEGER NOT NULL,
                ret_pct REAL NOT NULL,
                correct INTEGER NOT NULL,
                FOREIGN KEY (decision_id) REFERENCES decisions(id)
            )
        ''')

        # Create sentiment_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                overall_sentiment REAL NOT NULL,
                twitter_sentiment REAL NOT NULL,
                reddit_sentiment REAL NOT NULL,
                fear_greed_index INTEGER NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                sources TEXT NOT NULL
            )
        ''')

        # Create onchain_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS onchain_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                active_addresses INTEGER,
                transaction_count INTEGER,
                hash_rate REAL,
                network_health REAL,
                timestamp TEXT NOT NULL
            )
        ''')

        # Create decisions table (Expanded with feature_vector)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                product_id TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                position_size REAL NOT NULL,
                weighted_signal REAL NOT NULL,
                reasoning TEXT NOT NULL,
                feature_vector TEXT,
                vae_loss REAL,
                hmm_regime TEXT
            )
        ''')

        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                product_id TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                price REAL NOT NULL,
                status TEXT NOT NULL,
                algo_used TEXT,
                slippage REAL,
                execution_time REAL
            )
        ''')

        # Create ml_predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                product_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prediction REAL NOT NULL
            )
        ''')

        conn.commit()
        conn.close()
        self.logger.info("Database initialized successfully with expanded metrics support")

    async def store_market_data(self, data: MarketData):
        """Store market data"""
        try:
            conn = sqlite3.connect(self.db_path)
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
            conn.close()
            self.logger.debug("Market data stored successfully")

        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")

    async def store_sentiment_data(self, data: SentimentData):
        """Store sentiment data"""
        try:
            conn = sqlite3.connect(self.db_path)
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
            conn.close()
            self.logger.debug("Sentiment data stored successfully")

        except Exception as e:
            self.logger.error(f"Error storing sentiment data: {e}")

    async def get_recent_data(self, table: str, hours: int = 24) -> List[Dict]:
        """Get recent data from specified table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(f'''
                SELECT * FROM {table} 
                WHERE datetime(timestamp) > datetime('now', '-{hours} hours')
                ORDER BY timestamp DESC
            ''')

            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            result = [dict(zip(columns, row)) for row in rows]

            conn.close()
            return result

        except Exception as e:
            self.logger.error(f"Error retrieving data: {e}")
            return []

    async def cleanup_old_data(self, days: int = 30):
        """Remove data older than specified days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            tables = ['market_data', 'sentiment_data', 'onchain_data', 'decisions', 'trades', 'ml_predictions']

            for table in tables:
                cursor.execute(f'''
                    DELETE FROM {table} 
                    WHERE datetime(timestamp) < datetime('now', '-{days} days')
                ''')

            conn.commit()
            conn.close()
            self.logger.info(f"Cleaned up data older than {days} days")

        except Exception as e:
            self.logger.error(f"Error cleaning up data: {e}")

    async def store_decision(self, decision_data: Dict[str, Any]):
        """Store trading decision with expanded metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
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
            conn.close()
        except Exception as e:
            self.logger.error(f"Error storing decision: {e}")

    async def store_trade(self, trade_data: Dict[str, Any]):
        """Store executed trade"""
        try:
            conn = sqlite3.connect(self.db_path)
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
            conn.close()
        except Exception as e:
            self.logger.error(f"Error storing trade: {e}")

    async def store_ml_prediction(self, prediction_data: Dict[str, Any]):
        """Store ML model prediction"""
        try:
            conn = sqlite3.connect(self.db_path)
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
            conn.close()
        except Exception as e:
            self.logger.error(f"Error storing ML prediction: {e}")