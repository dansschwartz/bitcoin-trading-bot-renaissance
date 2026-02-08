"""
Regime Overlay Adapter
Bridges Golden Path technical indicators with Experimental Step 7 Market Regime Detection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from enhanced_technical_indicators import EnhancedTechnicalIndicators as RenaissanceTechnicalIndicators
from medallion_regime_predictor import MedallionRegimePredictor

class RegimeOverlay:
    """
    Adapter that provides market regime intelligence to the Renaissance Trading Bot.
    Uses the experimental RenaissanceTechnicalIndicators for regime detection.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get("enabled", False)
        self.consciousness_boost = config.get("consciousness_boost", 0.142)
        
        # Initialize the experimental regime detector
        self.detector = RenaissanceTechnicalIndicators()
        
        # Initialize Step 17 HMM Predictor
        self.hmm_predictor = MedallionRegimePredictor(n_regimes=3, logger=self.logger)
        
        self.current_regime = None
        self.logger.info(f"RegimeOverlay initialized (Enabled: {self.enabled})")

    def update(self, price_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Update market regime detection using the latest price data.
        Includes Enhanced Regime Detection with trend persistence, volatility clustering, and HMM transition prediction.
        """
        if not self.enabled or price_df.empty:
            return None

        try:
            # Prepare data for the experimental detector
            market_data = {
                'high': price_df['high'].values,
                'low': price_df['low'].values,
                'close': price_df['close'].values,
                'volume': price_df['volume'].values
            }
            
            # Detect base regime using signals summary
            signals = self.detector.get_signals_summary()
            
            self.current_regime = {
                'trend': signals.get('trend', 'unknown'),
                'volatility': signals.get('volatility', 'unknown'),
                'combined_signal': signals.get('combined_signal', 0.0),
                'confidence': signals.get('confidence', 0.0)
            }
            
            if self.current_regime:
                # 1. HMM Regime Transition Prediction (Step 17)
                if not self.hmm_predictor.is_fitted and len(price_df) > 100:
                    self.hmm_predictor.fit(price_df)
                
                hmm_forecast = self.hmm_predictor.predict_next_regime(price_df)
                self.current_regime['hmm_forecast'] = hmm_forecast

                # 2. ENHANCEMENT: Trend Persistence Score
                # Analysis of how long the current trend has lasted
                if len(price_df) >= 30:
                    returns = price_df['close'].pct_change().dropna()
                    persistence = (returns.rolling(window=10).mean().iloc[-1] / 
                                   (returns.rolling(window=10).std().iloc[-1] + 1e-6))
                    self.current_regime['trend_persistence'] = float(np.clip(persistence, -1.0, 1.0))
                else:
                    self.current_regime['trend_persistence'] = 0.0

                # ENHANCEMENT: Volatility Clustering (GARCH-lite)
                # Check if volatility is increasing or decreasing
                if len(price_df) >= 20:
                    recent_vol = price_df['close'].pct_change().tail(5).std()
                    baseline_vol = price_df['close'].pct_change().tail(20).std()
                    vol_acceleration = recent_vol / (baseline_vol + 1e-9)
                    self.current_regime['volatility_acceleration'] = float(vol_acceleration)
                else:
                    self.current_regime['volatility_acceleration'] = 1.0

                self.logger.info(f"Market Regime Detected: {self.current_regime.get('volatility_regime')} / {self.current_regime.get('trend_regime')} "
                                f"(Persistence: {self.current_regime['trend_persistence']:.2f}, VolAccel: {self.current_regime['volatility_acceleration']:.2f})")
            
            return self.current_regime
            
        except Exception as e:
            self.logger.error(f"Regime detection failed in overlay: {e}")
            return None

    def get_adjusted_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust base signal weights based on current market regime.
        """
        if not self.enabled or not self.current_regime:
            return base_weights

        try:
            regime_weights = self.current_regime.get('regime_weights', {})
            vol_w = float(regime_weights.get('volatility_weight', 1.0))
            trend_w = float(regime_weights.get('trend_weight', 1.0))
            liq_w = float(regime_weights.get('liquidity_weight', 1.0))

            adjusted = base_weights.copy()
            
            # Apply trend weights to technical indicators
            if 'macd' in adjusted: adjusted['macd'] = float(adjusted['macd']) * float(trend_w)
            if 'rsi' in adjusted: adjusted['rsi'] = float(adjusted['rsi']) * float(trend_w)
            
            # Apply volatility weights to Bollinger
            if 'bollinger' in adjusted: adjusted['bollinger'] = float(adjusted['bollinger']) * float(vol_w)
            
            # Apply liquidity weights to microstructure
            if 'order_flow' in adjusted: adjusted['order_flow'] = float(adjusted['order_flow']) * float(liq_w)
            if 'order_book' in adjusted: adjusted['order_book'] = float(adjusted['order_book']) * float(liq_w)
            if 'volume' in adjusted: adjusted['volume'] = float(adjusted['volume']) * float(liq_w)

            # Re-normalize weights to sum to 1.0 and ensure standard floats
            total = float(sum(adjusted.values()))
            if total > 0:
                adjusted = {k: float(v) / total for k, v in adjusted.items()}
            else:
                # Fallback to standard weights cast to float
                total_base = float(sum(base_weights.values()))
                adjusted = {k: float(v) / total_base for k, v in base_weights.items()}
            
            return adjusted
            
        except Exception as e:
            self.logger.error(f"Weight adjustment failed: {e}")
            return base_weights

    def get_confidence_boost(self) -> float:
        """Get confidence boost factor from regime analysis."""
        if not self.enabled or not self.current_regime:
            return 0.0
        
        # Use regime confidence to provide a small boost/penalty
        conf_score = self.current_regime.get('confidence_score', 0.5)
        return (conf_score - 0.5) * 0.1  # Max +/- 5% boost
