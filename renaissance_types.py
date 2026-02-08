from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"
    MICROSTRUCTURE = "MICROSTRUCTURE"
    TECHNICAL = "TECHNICAL"
    ALTERNATIVE = "ALTERNATIVE"
    RSI = "RSI"
    MACD = "MACD"
    BOLLINGER = "BOLLINGER"
    ORDER_FLOW = "ORDER_FLOW"
    VOLUME = "VOLUME"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    TREND = "TREND"
    VOLATILITY = "VOLATILITY"
    SENTIMENT = "SENTIMENT"
    FRACTAL = "FRACTAL"
    QUANTUM = "QUANTUM"
    CONSCIOUSNESS = "CONSCIOUSNESS"

class OrderType(Enum):
    """Types of trading orders"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class MLSignalPackage:
    """Unified container for ML model outputs"""
    primary_signals: List[Any]
    ml_predictions: List[Any]  # Added for compatibility with bridge
    ensemble_score: float
    confidence_score: float
    fractal_insights: Dict[str, Any]
    feature_vector: Optional[Any] = None
    processing_time_ms: float = 0.0
    timestamp: datetime = datetime.now()

@dataclass
class TradingDecision:
    """Final trading decision with Renaissance methodology"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0 to 1
    position_size: float  # Percentage of available capital
    reasoning: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            'action': self.action,
            'confidence': self.confidence,
            'position_size': self.position_size,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }
