"""
Fractal Intelligence Engine: Pattern Discovery via Dynamic Time Warping (DTW).
Compares current market patterns against a library of "Golden Fractals".
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from scipy.spatial.distance import cdist

class FractalIntelligenceEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        # Define some "Golden Patterns" (Normalized)
        self.patterns = {
            "v_recovery": np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            "bull_flag": np.array([0.0, 0.5, 1.0, 0.9, 0.8, 0.7, 0.75, 0.8, 1.0, 1.1, 1.2]),
            "double_bottom": np.array([1.0, 0.5, 0.0, 0.3, 0.0, 0.5, 1.0]),
            "consolidation_breakout": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.8, 1.0, 1.2])
        }
        
    def find_best_match(self, price_series: np.array) -> Dict[str, Any]:
        """
        Finds the closest golden pattern to the current price series using DTW.
        """
        if len(price_series) < 10:
            return {"best_pattern": "none", "similarity": 0.0, "signal": 0.0}
            
        # Normalize input series
        norm_series = (price_series - np.min(price_series)) / (np.max(price_series) - np.min(price_series) + 1e-9)
        
        best_pattern = "none"
        best_dist = float('inf')
        
        for name, pattern in self.patterns.items():
            dist = self._dtw_distance(norm_series, pattern)
            if dist < best_dist:
                best_dist = dist
                best_pattern = name
                
        # Convert distance to similarity score [0, 1]
        similarity = 1.0 / (1.0 + best_dist)
        
        return {
            "best_pattern": best_pattern,
            "similarity": float(similarity),
            "signal": self._get_pattern_signal(best_pattern, similarity)
        }
        
    def _dtw_distance(self, s1: np.array, s2: np.array) -> float:
        """
        Calculates DTW distance between two series.
        """
        n, m = len(s1), len(s2)
        dtw_matrix = np.zeros((n + 1, m + 1))
        dtw_matrix[1:, 0] = np.inf
        dtw_matrix[0, 1:] = np.inf
        dtw_matrix[0, 0] = 0
        
        # Use cdist for fast pairwise distance if needed, but for small patterns manual is fine
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i-1] - s2[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
                
        return dtw_matrix[n, m] / (n + m) # Normalized by length
        
    def _get_pattern_signal(self, pattern: str, similarity: float) -> float:
        """
        Maps pattern types to directional signals.
        """
        if similarity < 0.6: # Threshold for significance
            return 0.0
            
        mapping = {
            "v_recovery": 0.8,
            "bull_flag": 1.0,
            "double_bottom": 0.9,
            "consolidation_breakout": 0.7,
            "none": 0.0
        }
        return mapping.get(pattern, 0.0) * similarity
