"""
🏛️ META-STRATEGY SELECTOR
========================
Dynamic controller that switches the bot's execution mode between 
Taker (Renaissance) and Maker (Citadel) based on market conditions.
"""

import logging
from typing import Dict, Any, Optional

class MetaStrategySelector:
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config
        self.current_mode = "TAKER"  # Default mode
        
        # Thresholds
        self.vpin_maker_threshold = config.get("vpin_maker_threshold", 0.4)
        self.volatility_maker_threshold = config.get("volatility_maker_threshold", 0.015)
        
    def select_mode(self, market_data: Dict[str, Any], regime_data: Dict[str, Any]) -> str:
        """
        Determines the optimal execution mode.
        
        MAKER mode is preferred when:
        1. Toxicity (VPIN) is low (less risk of being 'picked off').
        2. Volatility is stable/low (better spread capture).
        3. Market is sideways/mean-reverting.
        
        TAKER mode is preferred when:
        1. Market is trending strongly.
        2. Toxicity is high.
        3. High conviction signals are present.
        """
        vpin = market_data.get('vpin', 0.5)
        volatility = market_data.get('volatility', 0.02)
        trend_regime = regime_data.get('trend_regime', 'sideways')
        vol_regime = regime_data.get('volatility_regime', 'normal_volatility')
        
        # Logic: 
        # If trend is strong (bullish/bearish), we want to be a TAKER to capture the move.
        if trend_regime in ['bullish', 'bearish']:
            self.current_mode = "TAKER"
        # If toxicity is high, we avoid being a MAKER (informed trading risk).
        elif vpin > 0.7:
            self.current_mode = "TAKER"
        # If sideways and low/normal volatility and low toxicity, we can be a MAKER.
        elif trend_regime == 'sideways' and vol_regime in ['low_volatility', 'normal_volatility'] and vpin < self.vpin_maker_threshold:
            self.current_mode = "MAKER"
        else:
            self.current_mode = "TAKER"
            
        self.logger.info(f"🏛️ META-STRATEGY: {self.current_mode} (Regime: {trend_regime}/{vol_regime}, VPIN: {vpin:.4f})")
        return self.current_mode
