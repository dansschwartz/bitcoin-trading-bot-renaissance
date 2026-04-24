"""
Market Making Engine with Consciousness Enhancement
Provides liquidity via two-sided limit orders, manages inventory risk,
and captures the bid-ask spread.
"""

import logging
import numpy as np
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone


class MarketRegime(Enum):
    """Market regime classification for market making"""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    CALM = "calm"
    UNKNOWN = "unknown"


class OrderType(Enum):
    """Order type for quotes"""
    BID = "bid"
    ASK = "ask"


@dataclass
class MarketMakingConfig:
    """Configuration for consciousness-enhanced market making engine"""
    symbol: str = "BTC-USD"
    consciousness_boost: float = 1.0
    base_spread_bps: float = 5.0
    max_position_size: float = 10.0
    quote_layers: int = 3
    min_spread_bps: float = 1.0
    max_spread_bps: float = 50.0
    risk_aversion: float = 0.01
    inventory_skew_factor: float = 0.1


@dataclass
class MarketData:
    """Real-time market data snapshot"""
    timestamp: float
    symbol: str
    mid_price: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    volatility: float
    order_flow_imbalance: float
    microstructure_signal: float


@dataclass
class Quote:
    """A market making quote"""
    quote_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    side: OrderType = OrderType.BID
    price: float = 0.0
    size: float = 0.0
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


class MarketMakingEngine:
    """
    Market Making Engine with Consciousness Enhancement

    Supports two initialization modes:
    - MarketMakingConfig object (new, full-featured)
    - Dict config (legacy, basic)
    """

    def __init__(self, config, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        if isinstance(config, MarketMakingConfig):
            self._init_from_config(config)
        elif isinstance(config, dict):
            self._init_from_dict(config)
        else:
            raise TypeError(f"config must be MarketMakingConfig or dict, got {type(config)}")

    def _init_from_config(self, config: MarketMakingConfig):
        """Initialize from MarketMakingConfig (full-featured mode)"""
        self.config = config
        self.current_inventory = 0.0
        self.active_quotes: Dict[str, Quote] = {}
        self.market_data_history: List[MarketData] = []
        self.start_time = time.time()

        # Regime detection state
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.5

        # Performance tracking
        self.performance_metrics = {
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_fills": 0,
            "total_quotes": 0,
            "avg_spread_captured": 0.0,
        }

    def _init_from_dict(self, config: dict):
        """Initialize from dict config (legacy mode)"""
        self.config = config
        self.enabled = config.get("enabled", False)
        self.max_inventory = config.get("max_inventory", 1.0)
        self.target_spread_bps = config.get("target_spread_bps", 10.0)
        self.current_inventory = 0.0
        self.active_quotes: Dict[str, Quote] = {}
        self.market_data_history: List[MarketData] = []
        self.start_time = time.time()
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.5
        self.performance_metrics = {
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_fills": 0,
            "total_quotes": 0,
            "avg_spread_captured": 0.0,
        }

    # ── New full-featured methods ──────────────────────────────────────

    def update_market_data(self, market_data: MarketData):
        """Process and store market data with consciousness enhancement"""
        consciousness = getattr(self.config, 'consciousness_boost', 1.0)

        # Enhance microstructure signal with consciousness
        enhanced_data = MarketData(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            mid_price=market_data.mid_price,
            bid_price=market_data.bid_price,
            ask_price=market_data.ask_price,
            bid_size=market_data.bid_size,
            ask_size=market_data.ask_size,
            last_price=market_data.last_price,
            volume=market_data.volume,
            volatility=market_data.volatility,
            order_flow_imbalance=market_data.order_flow_imbalance,
            microstructure_signal=market_data.microstructure_signal * consciousness,
        )

        self.market_data_history.append(enhanced_data)

        # Keep history bounded
        if len(self.market_data_history) > 1000:
            self.market_data_history = self.market_data_history[-1000:]

        # Update regime detection
        self._detect_regime()

    def generate_quotes(self, market_data: MarketData) -> List[Quote]:
        """Generate bid/ask quotes across multiple layers"""
        config = self.config
        consciousness = getattr(config, 'consciousness_boost', 1.0)
        quote_layers = getattr(config, 'quote_layers', 3)
        base_spread_bps = getattr(config, 'base_spread_bps', 5.0)

        mid = market_data.mid_price
        vol = market_data.volatility
        imbalance = market_data.order_flow_imbalance

        quotes: List[Quote] = []

        for layer in range(quote_layers):
            layer_multiplier = 1.0 + layer * 0.5

            # Spread in price terms
            half_spread = mid * (base_spread_bps / 10000.0) * layer_multiplier / 2.0

            # Volatility adjustment
            vol_adj = vol * mid * 0.5 * layer_multiplier

            # Inventory skew
            inv_skew = -self.current_inventory * getattr(config, 'inventory_skew_factor', 0.1) * vol * mid

            # Microstructure adjustment
            micro_adj = market_data.microstructure_signal * consciousness * mid * 0.0001

            bid_price = mid - half_spread - vol_adj + inv_skew + micro_adj
            ask_price = mid + half_spread + vol_adj + inv_skew + micro_adj

            # Size decreases with layer
            base_size = 0.1 / layer_multiplier

            # Confidence based on consciousness, layer, and market conditions
            base_confidence = max(0.3, 1.0 - layer * 0.15 - vol * 5.0)
            confidence = min(1.0, base_confidence * consciousness)

            # BID quote
            quotes.append(Quote(
                side=OrderType.BID,
                price=round(bid_price, 2),
                size=round(base_size, 6),
                confidence=confidence,
            ))

            # ASK quote
            quotes.append(Quote(
                side=OrderType.ASK,
                price=round(ask_price, 2),
                size=round(base_size, 6),
                confidence=confidence,
            ))

        self.performance_metrics["total_quotes"] += len(quotes)
        return quotes

    def handle_fill(self, fill_data: Dict[str, Any]):
        """Process a fill event and update PnL"""
        quote_id = fill_data.get("quote_id")
        fill_price = fill_data.get("fill_price", 0.0)
        fill_size = fill_data.get("fill_size", 0.0)

        quote = self.active_quotes.pop(quote_id, None)

        if quote is not None:
            if quote.side == OrderType.BID:
                self.current_inventory += fill_size
                pnl_change = -fill_price * fill_size
            else:
                self.current_inventory -= fill_size
                pnl_change = fill_price * fill_size
        else:
            # Fallback: treat as a buy
            self.current_inventory += fill_size
            pnl_change = -fill_price * fill_size

        consciousness = getattr(self.config, 'consciousness_boost', 1.0)
        enhanced_pnl = pnl_change * consciousness

        self.performance_metrics["realized_pnl"] += enhanced_pnl
        self.performance_metrics["total_fills"] += 1

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        consciousness = getattr(self.config, 'consciousness_boost', 1.0)
        uptime = (time.time() - self.start_time) / 3600.0

        return {
            "current_inventory": self.current_inventory,
            "current_regime": self.current_regime.value,
            "regime_confidence": self.regime_confidence,
            "active_quotes_count": len(self.active_quotes),
            "consciousness_boost": consciousness,
            "uptime_hours": round(uptime, 4),
            "realized_pnl": self.performance_metrics["realized_pnl"],
            "total_fills": self.performance_metrics["total_fills"],
            "total_quotes": self.performance_metrics["total_quotes"],
        }

    def _detect_regime(self):
        """Detect current market regime from recent data"""
        if len(self.market_data_history) < 5:
            return

        recent = self.market_data_history[-10:]
        vols = [d.volatility for d in recent]
        flows = [d.order_flow_imbalance for d in recent]

        avg_vol = np.mean(vols)
        avg_flow = np.mean(np.abs(flows))
        flow_trend = np.mean(flows)

        if avg_vol > 0.05:
            self.current_regime = MarketRegime.VOLATILE
            self.regime_confidence = min(1.0, avg_vol * 10)
        elif abs(flow_trend) > 0.3:
            self.current_regime = MarketRegime.TRENDING
            self.regime_confidence = min(1.0, abs(flow_trend))
        elif avg_vol < 0.01:
            self.current_regime = MarketRegime.CALM
            self.regime_confidence = min(1.0, 1.0 - avg_vol * 50)
        else:
            self.current_regime = MarketRegime.MEAN_REVERTING
            self.regime_confidence = 0.6

    # ── Legacy methods (dict-config mode) ──────────────────────────────

    def calculate_quotes(self, mid_price: float, volatility: float,
                         order_book_imbalance: float, vpin: float = 0.5) -> Dict[str, float]:
        """Legacy: Calculate bid/ask prices based on Avellaneda-Stoikov model."""
        if mid_price <= 0:
            return {"bid": 0.0, "ask": 0.0}

        target_spread = getattr(self, 'target_spread_bps', 10.0)
        toxicity_multiplier = 1.0 + (max(0, vpin - 0.5) * 2.0)
        current_target_spread = target_spread * toxicity_multiplier

        half_spread = mid_price * (current_target_spread / 10000.0) / 2.0

        inventory_risk_aversion = 0.1
        skew = -self.current_inventory * inventory_risk_aversion * volatility

        imbalance_skew = order_book_imbalance * half_spread * 0.5

        bid_price = mid_price - half_spread + skew + imbalance_skew
        ask_price = mid_price + half_spread + skew + imbalance_skew

        return {
            "bid": float(bid_price),
            "ask": float(ask_price),
            "mid": float(mid_price),
            "skew": float(skew),
            "inventory": self.current_inventory,
            "vpin_adjusted_spread": current_target_spread,
        }

    def update_inventory(self, fill_side: str = None, fill_size: float = 0.0,
                         base_change: float = None, quote_change: float = None,
                         market_price: float = None, market_data: dict = None):
        """Update inventory - supports both legacy and new calling conventions."""
        if fill_side is not None:
            if fill_side.upper() == "BUY":
                self.current_inventory += fill_size
            elif fill_side.upper() == "SELL":
                self.current_inventory -= fill_size
            self.logger.info(f"Inventory Updated: {self.current_inventory:.4f} BTC")
