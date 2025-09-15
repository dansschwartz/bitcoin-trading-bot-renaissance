# database_manager.py
import sqlite3
import json
import logging
import os
from datetime import datetime
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
                source TEXT NOT NULL
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

        conn.commit()
        conn.close()
        self.logger.info("Database initialized successfully")

    async def store_market_data(self, data: MarketData):
        """Store market data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO market_data (price, volume, bid, ask, spread, timestamp, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.price,
                data.volume,
                data.bid,
                data.ask,
                data.spread,
                data.timestamp.isoformat(),
                data.source
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
                json.dumps(data.sources)
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

            tables = ['market_data', 'sentiment_data', 'onchain_data']

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