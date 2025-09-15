# data_validator.py
import logging
from datetime import datetime
from typing import Dict
from dataclasses import dataclass
from typing import Optional, Dict, Any

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

class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataValidator")
        self.price_history = {}

    def validate_market_data(self, data: MarketData) -> bool:
        """Validate market data for quality and consistency"""
        try:
            # Basic data presence checks
            if not data.price or data.price <= 0:
                self.logger.warning(f"Invalid price data: {data.price}")
                return False

            if not data.bid or not data.ask or data.bid >= data.ask:
                self.logger.warning(f"Invalid bid/ask data: {data.bid}/{data.ask}")
                return False

            # Price spike detection
            symbol = data.symbol
            if symbol in self.price_history:
                last_price = self.price_history[symbol][-1] if self.price_history[symbol] else data.price
                price_change = abs(data.price - last_price) / last_price

                if price_change > 0.1:  # 10% price change
                    self.logger.warning(f"Large price movement detected: {price_change:.2%}")
                    # Don't reject, but flag for attention

            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []

            self.price_history[symbol].append(data.price)
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]

            return True

        except Exception as e:
            self.logger.error(f"Error validating market data: {e}")
            return False

    def validate_sentiment_data(self, data: SentimentData) -> bool:
        """Validate sentiment data"""
        try:
            if abs(data.sentiment_score) > 1.0:
                self.logger.warning(f"Sentiment score out of range: {data.sentiment_score}")
                return False

            if data.confidence < 0 or data.confidence > 1:
                self.logger.warning(f"Invalid confidence score: {data.confidence}")
                return False

            if data.volume < 0:
                self.logger.warning(f"Invalid volume: {data.volume}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating sentiment data: {e}")
            return False


class DataPipelineMonitor:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Monitor")
        self.metrics = {
            'data_points_processed': 0,
            'errors_count': 0,
            'last_update': {},
            'queue_depths': {},
            'processing_times': []
        }

    def record_data_point(self, data_type: str):
        """Record a processed data point"""
        self.metrics['data_points_processed'] += 1
        self.metrics['last_update'][data_type] = datetime.now()

    def record_error(self, error_type: str, error_msg: str):
        """Record an error"""
        self.metrics['errors_count'] += 1
        self.logger.error(f"{error_type}: {error_msg}")

    def record_queue_depth(self, queue_name: str, depth: int):
        """Record queue depth"""
        self.metrics['queue_depths'][queue_name] = depth

    def get_status_report(self) -> Dict:
        """Get comprehensive status report"""
        now = datetime.now()
        status = {
            'timestamp': now,
            'uptime': 'calculating...',
            'data_points_processed': self.metrics['data_points_processed'],
            'error_rate': self.metrics['errors_count'] / max(self.metrics['data_points_processed'], 1),
            'queue_status': self.metrics['queue_depths'],
            'last_updates': {}
        }

        # Check data freshness
        for data_type, last_update in self.metrics['last_update'].items():
            age_seconds = (now - last_update).total_seconds()
            status['last_updates'][data_type] = {
                'last_update': last_update,
                'age_seconds': age_seconds,
                'status': 'healthy' if age_seconds < 300 else 'stale'
            }

        return status