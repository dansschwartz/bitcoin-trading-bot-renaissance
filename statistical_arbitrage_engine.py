"""
Statistical Arbitrage Engine
Renaissance Technologies-style pairs trading and cointegration analysis.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

class StatisticalArbitrageEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.history = {} # product_id -> prices
        self.window_size = 120 # 2-hour window for cointegration
        
    def update_price(self, product_id: str, price: float):
        if product_id not in self.history:
            self.history[product_id] = []
        self.history[product_id].append(price)
        if len(self.history[product_id]) > self.window_size:
            self.history[product_id].pop(0)
            
    def calculate_pair_signal(self, base_id: str, target_id: str) -> Dict[str, Any]:
        """
        Calculates a statistical arbitrage signal between two assets.
        Uses Z-Score of the log-price spread.
        """
        try:
            if base_id not in self.history or target_id not in self.history:
                return {"z_score": 0.0, "signal": 0.0, "status": "insufficient_data"}
                
            b_prices = self.history[base_id]
            t_prices = self.history[target_id]
            
            if len(b_prices) < 10 or len(t_prices) < 10:
                return {"z_score": 0.0, "signal": 0.0, "status": "insufficient_data"}
                
            # Synchronize
            min_len = min(len(b_prices), len(t_prices))
            b = np.log(np.array(b_prices[-min_len:], dtype=np.float64))
            t = np.log(np.array(t_prices[-min_len:], dtype=np.float64))
            
            # Calculate beta (hedge ratio) using simple linear regression
            # Spread = b - beta * t
            cov_matrix = np.cov(b, t)
            if cov_matrix.shape == (2, 2):
                beta = cov_matrix[0, 1] / np.var(t)
            else:
                beta = 1.0
                
            if np.isnan(beta) or np.isinf(beta):
                beta = 1.0
                
            spread = b - beta * t
            
            # Z-Score of current spread
            mean_spread = np.mean(spread)
            std_spread = np.std(spread)
            
            if std_spread < 1e-9:
                z_score = 0.0
            else:
                z_score = (spread[-1] - mean_spread) / std_spread
                
            # Signal: -Z-score (mean reversion)
            # If Z-score is high (spread too wide), sell base buy target -> negative signal for base
            signal = -float(np.clip(z_score / 2.0, -1.0, 1.0))
            
            return {
                "z_score": float(z_score),
                "signal": signal,
                "beta": float(beta),
                "status": "active",
                "timestamp": datetime.now(timezone.utc)
            }
        except Exception as e:
            self.logger.error(f"Error in pair signal calculation: {e}")
            return {"z_score": 0.0, "signal": 0.0, "status": "error"}
