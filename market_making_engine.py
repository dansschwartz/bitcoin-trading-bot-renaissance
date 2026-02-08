"""
⚖️ MARKET MAKING ENGINE
=======================
Transition from Taker to Maker. Provides liquidity via two-sided limit orders, 
manages inventory risk, and captures the bid-ask spread.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

class MarketMakingEngine:
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get("enabled", False)
        self.max_inventory = config.get("max_inventory", 1.0) # BTC
        self.target_spread_bps = config.get("target_spread_bps", 10.0) # 10 bps
        self.current_inventory = 0.0
        
    def calculate_quotes(self, mid_price: float, volatility: float, order_book_imbalance: float, vpin: float = 0.5) -> Dict[str, float]:
        """
        Calculates bid and ask prices based on Avellaneda-Stoikov model logic.
        Adjusts quotes to manage inventory (skewing) and toxicity (VPIN).
        """
        if mid_price <= 0:
            return {"bid": 0.0, "ask": 0.0}
            
        # 1. Base Spread (Adaptive based on VPIN)
        # Toxicity boost: if VPIN is high, widen spread to protect against informed traders
        toxicity_multiplier = 1.0 + (max(0, vpin - 0.5) * 2.0)
        current_target_spread = self.target_spread_bps * toxicity_multiplier
        
        half_spread = mid_price * (current_target_spread / 10000.0) / 2.0
        
        # 2. Inventory Skewing (if inventory is positive, lower bid/ask to encourage selling)
        inventory_risk_aversion = 0.1
        skew = -self.current_inventory * inventory_risk_aversion * volatility
        
        # 3. Microstructure Skewing (if imbalance is positive, raise bid/ask)
        imbalance_skew = order_book_imbalance * half_spread * 0.5
        
        bid_price = mid_price - half_spread + skew + imbalance_skew
        ask_price = mid_price + half_spread + skew + imbalance_skew
        
        return {
            "bid": float(bid_price),
            "ask": float(ask_price),
            "mid": float(mid_price),
            "skew": float(skew),
            "inventory": self.current_inventory,
            "vpin_adjusted_spread": current_target_spread
        }

    def update_inventory(self, fill_side: str, fill_size: float):
        """Update the internal inventory tracker."""
        if fill_side.upper() == "BUY":
            self.current_inventory += fill_size
        elif fill_side.upper() == "SELL":
            self.current_inventory -= fill_size
        self.logger.info(f"Inventory Updated: {self.current_inventory:.4f} BTC")
