"""
Market Entropy Engine: Measures market disorder and predictability transitions.
Uses Shannon Entropy and Approximate Entropy (ApEn).
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional

class MarketEntropyEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_entropy(self, price_series: np.array) -> Dict[str, Any]:
        """
        Calculates multiple entropy metrics for the price series.
        """
        if len(price_series) < 20:
            return {"shannon_entropy": 0.0, "predictability": 0.0, "status": "insufficient_data"}
            
        # Calculate returns
        returns = np.diff(np.log(price_series))
        
        # 1. Shannon Entropy (Discretized)
        hist, bin_edges = np.histogram(returns, bins=10, density=True)
        # Avoid log(0)
        hist = hist[hist > 0]
        shannon = -np.sum(hist * np.log2(hist))
        
        # 2. Approximate Entropy (Simplified)
        # Low ApEn = Predictable/Trending
        # High ApEn = Random/Efficient
        apen = self._approximate_entropy(returns, m=2, r=0.2 * np.std(returns))
        
        # Predictability score [0, 1]
        # Inverse of ApEn normalized roughly
        predictability = 1.0 / (1.0 + apen) if not np.isnan(apen) else 0.5
        
        return {
            "shannon_entropy": float(shannon),
            "approximate_entropy": float(apen),
            "predictability": float(predictability),
            "status": "active"
        }
        
    def _approximate_entropy(self, U, m, r):
        """
        Approximate Entropy implementation.
        U: Time series
        m: Embedded dimension
        r: Tolerance
        """
        def _max_dist(x_i, x_j):
            return max([abs(a - b) for a, b in zip(x_i, x_j)])

        def _phi(m):
            x = [[U[j] for j in range(i, i + m)] for i in range(len(U) - m + 1)]
            C = [len([1 for x_j in x if _max_dist(x_i, x_j) <= r]) / (len(U) - m + 1.0) for x_i in x]
            return (len(U) - m + 1.0)**-1 * sum(np.log(C))

        try:
            return abs(_phi(m) - _phi(m + 1))
        except Exception:
            return 0.5
