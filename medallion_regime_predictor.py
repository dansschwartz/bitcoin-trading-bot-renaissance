"""
Step 17: Medallion Regime Prediction (HMM)
Uses a Hidden Markov Model to predict market regime transitions,
allowing the bot to front-run regime shifts.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from hmmlearn import hmm

class MedallionRegimePredictor:
    def __init__(self, n_regimes: int = 3, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes, 
            covariance_type="full", 
            n_iter=100
        )
        self.is_fitted = False
        self.regime_map = {0: "Low Volatility", 1: "High Volatility", 2: "Trending"}

    def fit(self, price_df: pd.DataFrame):
        """Fits the HMM on historical returns and volatility."""
        if len(price_df) < 100:
            return

        try:
            returns = price_df['close'].pct_change().dropna().values.reshape(-1, 1)
            volatility = price_df['close'].pct_change().rolling(window=10).std().dropna().values.reshape(-1, 1)
            
            # Synchronize lengths
            min_len = min(len(returns), len(volatility))
            features = np.column_stack([returns[-min_len:], volatility[-min_len:]])
            
            self.model.fit(features)
            # Regularize transition matrix: add small epsilon to zero rows
            # so predict() doesn't fail on "transmat_ rows must sum to 1"
            tm = self.model.transmat_.copy()
            row_sums = tm.sum(axis=1)
            for i in range(len(row_sums)):
                if row_sums[i] < 1e-10:
                    tm[i] = 1.0 / self.n_regimes  # uniform fallback
            tm = tm / tm.sum(axis=1, keepdims=True)
            self.model.transmat_ = tm
            self.is_fitted = True
            self.logger.info(f"Medallion HMM fitted with {self.n_regimes} regimes.")
        except Exception as e:
            self.logger.error(f"HMM fitting failed: {e}")

    def predict_next_regime(self, price_df: pd.DataFrame) -> Dict[str, Any]:
        """Predicts the most likely next regime and transition probabilities."""
        if not self.is_fitted or len(price_df) < 20:
            return {"current_regime": "Unknown", "next_regime": "Unknown", "transition_prob": 0.0}

        try:
            returns = price_df['close'].pct_change().dropna().values.reshape(-1, 1)
            volatility = price_df['close'].pct_change().rolling(window=10).std().dropna().values.reshape(-1, 1)
            
            min_len = min(len(returns), len(volatility))
            current_features = np.column_stack([returns[-min_len:], volatility[-min_len:]])
            
            # Current regime
            hidden_states = self.model.predict(current_features)
            current_state = hidden_states[-1]
            
            # Transition matrix
            trans_matrix = self.model.transmat_
            next_state_probs = trans_matrix[current_state]
            next_state = np.argmax(next_state_probs)
            
            return {
                "current_regime": self.regime_map.get(current_state, str(current_state)),
                "next_regime": self.regime_map.get(next_state, str(next_state)),
                "transition_prob": float(next_state_probs[next_state]),
                "all_probs": next_state_probs.tolist()
            }
        except Exception as e:
            self.logger.error(f"HMM prediction failed: {e}")
            return {"current_regime": "Error", "next_regime": "Error", "transition_prob": 0.0}
