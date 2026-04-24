"""
🏛️ BASIS TRADING ENGINE (Arbitrage & Funding)
============================================
Exploits the price difference between Spot and Futures/Perpetual markets
and harvests funding rates for low-risk yield.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class BasisTradingEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.basis_history = []
        
    def calculate_basis_opportunity(self, spot_price: float, futures_price: float, funding_rate: float) -> Dict[str, Any]:
        """
        Calculates the annualized basis and funding yield.
        Basis = (Futures - Spot) / Spot
        """
        if spot_price <= 0:
            return {"basis_pct": 0.0, "signal": "NONE"}

        basis_raw = futures_price - spot_price
        basis_pct = basis_raw / spot_price
        
        # Funding rate is usually 8-hourly (0.01% = 0.0001). Annualize it: (Rate * 3 * 365)
        # 3 funding payments per day, 365 days a year.
        annualized_funding = funding_rate * 3 * 365
        
        # Total Carry = Basis (if we hold to expiry) + Funding
        # For perpetuals, basis is theoretical; we mostly harvest funding.
        total_carry_annualized = annualized_funding + (basis_pct * 365) # Simple approximation
        
        signal = "NONE"
        # Thresholds: 10% annualized yield for engagement
        if total_carry_annualized > 0.10: 
            signal = "CASH_AND_CARRY" # Buy Spot, Sell Futures (Harvest Positive Funding)
        elif total_carry_annualized < -0.10:
            signal = "REVERSE_CARRY" # Sell Spot, Buy Futures (Harvest Negative Funding)
            
        return {
            "spot_price": spot_price,
            "futures_price": futures_price,
            "basis_raw": basis_raw,
            "basis_pct": basis_pct,
            "funding_rate": funding_rate,
            "annualized_funding": annualized_funding,
            "total_carry_annualized": total_carry_annualized,
            "signal": signal,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_basis_signal(self, market_data: Dict[str, Any]) -> float:
        """Returns a normalized signal for the main fusion engine."""
        # This would ideally pull from a Futures exchange client (e.g. Coinbase Advanced Futures)
        # Mocking values for the architecture skeleton
        ticker = market_data.get('ticker', {})
        spot = ticker.get('price', 0.0)
        
        # Hardening spot to float
        try:
            while hasattr(spot, '__iter__') and not isinstance(spot, (str, bytes, dict)):
                spot = spot[0] if len(spot) > 0 else 0.0
            if hasattr(spot, 'item'): spot = spot.item()
            spot = float(spot)
        except:
            spot = 0.0

        if spot <= 0:
            return 0.0

        futures = spot * 1.0002 # 2bps premium mock
        funding = 0.0001 # 0.01% mock
        
        opp = self.calculate_basis_opportunity(spot, futures, funding)
        
        if opp['signal'] == "CASH_AND_CARRY":
            return 0.5 # Bullish for the strategy (buy spot)
        elif opp['signal'] == "REVERSE_CARRY":
            return -0.5 # Bearish for the strategy
        return 0.0
