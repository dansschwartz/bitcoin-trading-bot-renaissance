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
        
        # Funding rate is usually 8-hourly. Annualize it: (Rate * 3 * 365)
        annualized_funding = funding_rate * 3 * 365
        
        # Total Carry = Basis (if we hold to expiry) + Funding
        total_carry_annualized = annualized_funding # Simplified for Perps
        
        signal = "NONE"
        if basis_pct > 0.005: # > 0.5% premium
            signal = "CASH_AND_CARRY" # Buy Spot, Sell Futures
        elif basis_pct < -0.005: # > 0.5% discount
            signal = "REVERSE_CARRY" # Sell Spot, Buy Futures
            
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
        spot = market_data.get('ticker', {}).get('price', 0.0)
        futures = spot * 1.0002 # 2bps premium mock
        funding = 0.0001 # 0.01% mock
        
        opp = self.calculate_basis_opportunity(spot, futures, funding)
        
        if opp['signal'] == "CASH_AND_CARRY":
            return 0.5 # Bullish for the strategy (buy spot)
        elif opp['signal'] == "REVERSE_CARRY":
            return -0.5 # Bearish for the strategy
        return 0.0
