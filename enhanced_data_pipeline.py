# enhanced_data_pipeline.py
import numpy as np
import asyncio
import json
from datetime import datetime
import logging
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import queue
import os

# Import our components
from database_manager import DatabaseManager
from data_validator import DataValidator, DataPipelineMonitor
from coinbase_advanced_client import CoinbaseAdvancedClient
from twitter_client import TwitterClient
from reddit_client import RedditClient
from glassnode_client import GlassnodeClient
from fear_greed_client import FearGreedClient


class DataSource(Enum):
    COINBASE = "coinbase"
    TWITTER = "twitter"
    REDDIT = "reddit"
    GLASSNODE = "glassnode"
    FEAR_GREED = "fear_greed"
    GOOGLE_TRENDS = "google_trends"

@dataclass
class MarketData:
    timestamp: datetime
    source: str
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    order_book: Dict
    recent_trades: List[Dict]

@dataclass
class SentimentData:
    timestamp: datetime
    source: str
    sentiment_score: float
    confidence: float
    volume: int
    raw_data: Dict

@dataclass
class OnChainData:
    timestamp: datetime
    source: str
    metric_type: str
    value: float
    confidence: float
    metadata: Dict

class EnhancedDataPipeline:
    def __init__(self, config_path: str = "config/data_pipeline_config.json"):
        self.config = self._load_config(config_path)
        self.data_queues = {
            'market': queue.Queue(maxsize=1000),
            'sentiment': queue.Queue(maxsize=1000),
            'onchain': queue.Queue(maxsize=1000),
            'system': queue.Queue(maxsize=100)
        }

        # Database connections
        self.db_manager = DatabaseManager(self.config['database'])

        # API clients
        self.coinbase_client = CoinbaseAdvancedClient(self.config['coinbase'])
        self.twitter_client = TwitterClient(self.config['twitter'])
        self.reddit_client = RedditClient(self.config['reddit'])
        self.glassnode_client = GlassnodeClient(self.config['glassnode'])
        self.fear_greed_client = FearGreedClient(self.config.get('fear_greed', {}))

        # Data validation
        self.validator = DataValidator()

        # Monitoring
        self.monitor = DataPipelineMonitor()

        # Status tracking
        self.is_running = False
        self.last_heartbeat = {}

        # Setup logging
        self._setup_logging()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self._create_default_config(config_path)
            with open(config_path, 'r') as f:
                return json.load(f)

    def _create_default_config(self, config_path: str):
        """Create the default configuration file if it doesn't exist"""
        # Implementation would create the default config
        pass

    def _setup_logging(self):
        """Setup comprehensive logging"""
        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def start_pipeline(self):
        """Start the complete data pipeline"""
        self.logger.info("Starting Enhanced Data Pipeline...")
        self.is_running = True

        # Initialize database
        await self.db_manager.initialize()

        # Start all data collection tasks
        tasks = [
            asyncio.create_task(self._run_coinbase_websocket()),
            asyncio.create_task(self._run_sentiment_collection()),
            asyncio.create_task(self._run_onchain_collection()),
            asyncio.create_task(self._run_data_processing()),
            asyncio.create_task(self._run_monitoring()),
            asyncio.create_task(self._run_heartbeat())
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            await self.stop_pipeline()

    async def stop_pipeline(self):
        """Gracefully stop the pipeline"""
        self.logger.info("Stopping Enhanced Data Pipeline...")
        self.is_running = False
        await self.db_manager.close()

    async def _run_coinbase_websocket(self):
        """Run Coinbase WebSocket data collection"""
        while self.is_running:
            try:
                await self.coinbase_client.connect_websocket()
                await self.coinbase_client.listen_for_messages(self.data_queues['market'])
            except Exception as e:
                self.logger.error(f"Coinbase WebSocket error: {e}")
                await asyncio.sleep(5)

    async def _run_sentiment_collection(self):
        """Run sentiment data collection"""
        while self.is_running:
            try:
                # Collect Twitter sentiment
                await self.twitter_client.collect_sentiment_data(self.data_queues['sentiment'])

                # Collect Reddit sentiment
                await asyncio.sleep(300)  # 5 minute intervals
            except Exception as e:
                self.logger.error(f"Sentiment collection error: {e}")
                await asyncio.sleep(60)

    async def _run_onchain_collection(self):
        """Run on-chain data collection"""
        while self.is_running:
            try:
                await self.glassnode_client.collect_onchain_data(self.data_queues['onchain'])
                await asyncio.sleep(3600)  # 1 hour intervals
            except Exception as e:
                self.logger.error(f"On-chain collection error: {e}")
                await asyncio.sleep(300)

    async def _run_data_processing(self):
        """Process incoming data from all queues"""
        while self.is_running:
            try:
                # Process market data
                if not self.data_queues['market'].empty():
                    market_data = self.data_queues['market'].get_nowait()
                    if self.validator.validate_market_data(market_data):
                        await self.db_manager.store_market_data(market_data)

                # Process sentiment data
                if not self.data_queues['sentiment'].empty():
                    sentiment_data = self.data_queues['sentiment'].get_nowait()
                    await self.db_manager.store_sentiment_data(sentiment_data)

                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Data processing error: {e}")
                await asyncio.sleep(5)

    async def collect_all_data(self):
        """Collect data from all sources - compatibility method for startup script"""
        try:
            # Collect from all available sources
            data = {
                'timestamp': datetime.utcnow().isoformat()
            }

            # Get market data from Coinbase
            if hasattr(self, 'coinbase_client'):
                market_data = await self._collect_coinbase_data()
                if market_data:
                    data['market_data'] = market_data

            # Get social sentiment
            social_data = await self._collect_social_sentiment()
            if social_data:
                data['social_sentiment'] = social_data

            return data

        except Exception as e:
            self.logger.error(f"Error collecting data: {e}")
            return None

    async def _collect_coinbase_data(self):
        """Get market data from Coinbase"""
        try:
            # Get market data snapshot for BTC-USD
            market_data = self.coinbase_client._create_market_data_snapshot('BTC-USD')

            if market_data:
                return {
                    'price': market_data.price,
                    'volume': market_data.volume,
                    'bid': market_data.bid,
                    'ask': market_data.ask,
                    'spread': market_data.spread,
                    'timestamp': market_data.timestamp.isoformat()
                }
            return None

        except Exception as e:
            self.logger.error(f"Error collecting Coinbase data: {e}")
            return None

    async def _collect_social_sentiment(self):
        """Get social sentiment data"""
        try:
            # Collect from Twitter, Reddit, Fear & Greed
            twitter_data = await self.twitter_client.get_bitcoin_sentiment()
            fear_greed_data = await self.fear_greed_client.get_fear_greed_index()

            return {
                'overall_sentiment': twitter_data.get('sentiment_score', 0),
                'twitter_sentiment': twitter_data.get('sentiment_score', 0),
                'fear_greed': fear_greed_data.get('fear_greed_value', 50)
            }
        except Exception as e:
            self.logger.error(f"Error collecting sentiment: {e}")
            return None

    async def start_continuous_collection(self):
        """Start continuous data collection - compatibility method"""
        await self.start_pipeline()

    async def stop(self):
        """Stop pipeline - compatibility method"""
        await self.stop_pipeline()

    async def _run_monitoring(self):
        """Run system monitoring"""
        while self.is_running:
            try:
                # Update monitoring metrics
                for queue_name, data_queue in self.data_queues.items():
                    self.monitor.record_queue_depth(queue_name, data_queue.qsize())

                await asyncio.sleep(30)

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)

    async def _run_heartbeat(self):
        """Run system heartbeat"""
        while self.is_running:
            try:
                self.logger.info("Enhanced Data Pipeline heartbeat - System running")
                await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(60)