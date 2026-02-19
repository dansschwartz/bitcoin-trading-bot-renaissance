"""
Step 16: Cross-Asset Correlation & Lead-Lag Engine
Analyzes relationships between BTC and ETH to detect early entry signals.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

class CrossAssetCorrelationEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.history = {} # product_id -> prices
        self.window_size = 120 # 2-hour of 1-min candles (matches preload depth)
        
    def update_price(self, product_id: str, price: float):
        if product_id not in self.history:
            self.history[product_id] = []
        self.history[product_id].append(price)
        if len(self.history[product_id]) > self.window_size:
            self.history[product_id].pop(0)
            
    def calculate_lead_lag(self, base_id: str, target_id: str) -> Dict[str, Any]:
        """
        Calculates if base_id is leading or lagging target_id.
        Positive lag = base leads target.
        """
        if base_id not in self.history or target_id not in self.history:
            return {"correlation": 0.0, "lag_seconds": 0, "status": "insufficient_data"}
            
        base_prices = self.history[base_id]
        target_prices = self.history[target_id]
        
        if len(base_prices) < 6 or len(target_prices) < 6:
            return {"correlation": 0.0, "lag_seconds": 0, "status": "insufficient_data"}
            
        # Synchronize lengths
        min_len = min(len(base_prices), len(target_prices))
        b = np.array(base_prices[-min_len:])
        t = np.array(target_prices[-min_len:])
        
        # Calculate returns
        b_ret = np.diff(np.log(b))
        t_ret = np.diff(np.log(t))
        
        # Simple cross-correlation
        correlation = np.corrcoef(b_ret, t_ret)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
            
        # Detect lead-lag using shifted correlation
        # We try shifting base_id returns by -1, 0, 1 to see which has highest correlation
        lags = [-2, -1, 0, 1, 2]
        corrs = []
        for lag in lags:
            if lag == 0:
                corrs.append(correlation)
            elif lag > 0:
                # b leads t
                if len(b_ret) > lag and len(t_ret) > lag:
                    try:
                        c = np.corrcoef(b_ret[:-lag], t_ret[lag:])[0, 1]
                        corrs.append(c if not np.isnan(c) else 0.0)
                    except Exception:
                        corrs.append(0.0)
                else:
                    corrs.append(0.0)
            else:
                # t leads b
                abs_lag = abs(lag)
                if len(b_ret) > abs_lag and len(t_ret) > abs_lag:
                    try:
                        c = np.corrcoef(b_ret[abs_lag:], t_ret[:-abs_lag])[0, 1]
                        corrs.append(c if not np.isnan(c) else 0.0)
                    except Exception:
                        corrs.append(0.0)
                else:
                    corrs.append(0.0)
                
        best_lag_idx = np.argmax([abs(c) if not np.isnan(c) else 0 for c in corrs])
        best_lag = lags[best_lag_idx]
        best_corr = corrs[best_lag_idx]
        
        # Compute directional signal: if base leads target, use base's recent
        # return (scaled by correlation strength) as a predictor of target's
        # next move.  When base lags target, no actionable signal.
        directional_signal = 0.0
        if best_lag > 0 and abs(best_corr) > 0.2:
            # base leads target — base's recent move predicts target's next move
            recent_base_return = float(b_ret[-1]) if len(b_ret) > 0 else 0.0
            # Scale: correlation-strength × base return, capped to [-1, 1]
            directional_signal = float(np.clip(
                best_corr * recent_base_return * 50.0,  # 50x to normalise ~2% returns
                -1.0, 1.0
            ))

        return {
            "correlation": float(correlation),
            "lead_lag_score": float(best_corr),
            "directional_signal": directional_signal,
            "lag_periods": int(best_lag),
            "is_leading": best_lag > 0 and abs(best_corr) > 0.6,
            "timestamp": datetime.now(timezone.utc)
        }
