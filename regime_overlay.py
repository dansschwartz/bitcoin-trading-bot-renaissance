"""
Regime Overlay Adapter
Bridges the Advanced 5-state HMM Regime Detector with the Renaissance Trading Bot.
Falls back to the legacy MedallionRegimePredictor if AdvancedRegimeDetector is unavailable.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from enhanced_technical_indicators import EnhancedTechnicalIndicators as RenaissanceTechnicalIndicators
from medallion_regime_predictor import MedallionRegimePredictor

# Advanced HMM suite (new)
try:
    from advanced_regime_detector import (
        AdvancedRegimeDetector, RegimeState, MarketRegime,
        REGIME_ALPHA_WEIGHTS, ALPHA_TO_SIGNAL_MAP,
    )
    ADVANCED_HMM_AVAILABLE = True
except ImportError:
    ADVANCED_HMM_AVAILABLE = False


class RegimeOverlay:
    """
    Adapter that provides market regime intelligence to the Renaissance Trading Bot.
    Uses the 5-state HMM AdvancedRegimeDetector when available, with legacy fallback.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get("enabled", False)
        self.consciousness_boost = config.get("consciousness_boost", 0.0)

        # Initialize the experimental regime detector (legacy)
        self.detector = RenaissanceTechnicalIndicators()

        # Legacy 3-state HMM
        self.hmm_predictor = MedallionRegimePredictor(n_regimes=3, logger=self.logger)

        # Advanced 5-state HMM (new)
        self._advanced_detector: Optional[Any] = None
        self._advanced_regime_state: Optional[Any] = None
        self._cycle_count = 0
        if ADVANCED_HMM_AVAILABLE:
            hmm_cfg = {
                "n_regimes": config.get("hmm_regimes", 5),
                "refit_interval": config.get("hmm_refit_interval", 50),
                "min_samples": config.get("hmm_min_samples", 200),
                "covariance_type": config.get("hmm_covariance_type", "full"),
                "n_iter": config.get("hmm_n_iter", 150),
            }
            self._advanced_detector = AdvancedRegimeDetector(hmm_cfg, logger=self.logger)
            self.logger.info("Advanced 5-state HMM regime detector initialized")

        self.current_regime = None
        self.logger.info(f"RegimeOverlay initialized (Enabled: {self.enabled}, Advanced HMM: {ADVANCED_HMM_AVAILABLE})")

    def update(self, price_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Update market regime detection using the latest price data.
        Uses the 5-state HMM if available, otherwise falls back to legacy.
        """
        if not self.enabled or price_df.empty:
            return None

        self._cycle_count += 1

        try:
            # Detect base regime using legacy signals summary
            signals = self.detector.get_signals_summary()

            self.current_regime = {
                'trend': signals.get('trend', 'unknown'),
                'volatility': signals.get('volatility', 'unknown'),
                'combined_signal': signals.get('combined_signal', 0.0),
                'confidence': signals.get('confidence', 0.0)
            }

            # --- Advanced 5-state HMM path ---
            if self._advanced_detector is not None:
                # Periodically refit
                self._advanced_detector.maybe_refit(price_df, self._cycle_count)

                # If not fitted yet and we have enough data, do initial fit
                if not self._advanced_detector.is_fitted and len(price_df) >= 200:
                    self._advanced_detector.fit(price_df)

                # Predict current regime
                regime_state = self._advanced_detector.predict(price_df)
                if regime_state is not None:
                    self._advanced_regime_state = regime_state
                    self.current_regime['hmm_regime'] = regime_state.current_regime.value
                    self.current_regime['hmm_confidence'] = regime_state.confidence
                    self.current_regime['regime_probabilities'] = regime_state.regime_probabilities
                    self.current_regime['next_regime_probs'] = regime_state.next_regime_probs
                    self.current_regime['regime_duration'] = regime_state.regime_duration_estimate
                    self.current_regime['alpha_weights'] = regime_state.alpha_weights

            # --- Legacy HMM path ---
            if not self.hmm_predictor.is_fitted and len(price_df) > 100:
                self.hmm_predictor.fit(price_df)

            hmm_forecast = self.hmm_predictor.predict_next_regime(price_df)
            self.current_regime['hmm_forecast'] = hmm_forecast

            # Trend Persistence Score
            if len(price_df) >= 30:
                returns = price_df['close'].pct_change().dropna()
                persistence = (returns.rolling(window=10).mean().iloc[-1] /
                               (returns.rolling(window=10).std().iloc[-1] + 1e-6))
                self.current_regime['trend_persistence'] = float(np.clip(persistence, -1.0, 1.0))
            else:
                self.current_regime['trend_persistence'] = 0.0

            # Volatility Clustering
            if len(price_df) >= 20:
                recent_vol = price_df['close'].pct_change().tail(5).std()
                baseline_vol = price_df['close'].pct_change().tail(20).std()
                vol_acceleration = recent_vol / (baseline_vol + 1e-9)
                self.current_regime['volatility_acceleration'] = float(vol_acceleration)
            else:
                self.current_regime['volatility_acceleration'] = 1.0

            regime_label = self.current_regime.get('hmm_regime', 'unknown')
            self.logger.info(
                f"Market Regime: {regime_label} "
                f"(Persistence: {self.current_regime['trend_persistence']:.2f}, "
                f"VolAccel: {self.current_regime['volatility_acceleration']:.2f})"
            )

            return self.current_regime

        except Exception as e:
            self.logger.error(f"Regime detection failed in overlay: {e}")
            return None

    def get_regime_adjusted_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply regime-specific alpha weight multipliers to base signal weights.
        Uses the ALPHA_TO_SIGNAL_MAP from AdvancedRegimeDetector.
        Falls back to get_adjusted_weights() if advanced HMM is unavailable.
        """
        if not self.enabled or not self.current_regime:
            return base_weights

        alpha_weights = self.current_regime.get('alpha_weights')
        if not alpha_weights or not ADVANCED_HMM_AVAILABLE:
            return self.get_adjusted_weights(base_weights)

        try:
            adjusted = base_weights.copy()

            for alpha_key, signal_keys in ALPHA_TO_SIGNAL_MAP.items():
                multiplier = alpha_weights.get(alpha_key, 1.0)
                for sk in signal_keys:
                    if sk in adjusted:
                        adjusted[sk] = float(adjusted[sk]) * float(multiplier)

            # Re-normalize
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {k: float(v) / total for k, v in adjusted.items()}

            return adjusted

        except Exception as e:
            self.logger.error(f"Regime weight adjustment failed: {e}")
            return base_weights

    def get_adjusted_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Legacy: Adjust base signal weights based on current market regime.
        """
        if not self.enabled or not self.current_regime:
            return base_weights

        try:
            regime_weights = self.current_regime.get('regime_weights', {})
            vol_w = float(regime_weights.get('volatility_weight', 1.0))
            trend_w = float(regime_weights.get('trend_weight', 1.0))
            liq_w = float(regime_weights.get('liquidity_weight', 1.0))

            adjusted = base_weights.copy()

            if 'macd' in adjusted: adjusted['macd'] = float(adjusted['macd']) * float(trend_w)
            if 'rsi' in adjusted: adjusted['rsi'] = float(adjusted['rsi']) * float(trend_w)
            if 'bollinger' in adjusted: adjusted['bollinger'] = float(adjusted['bollinger']) * float(vol_w)
            if 'order_flow' in adjusted: adjusted['order_flow'] = float(adjusted['order_flow']) * float(liq_w)
            if 'order_book' in adjusted: adjusted['order_book'] = float(adjusted['order_book']) * float(liq_w)
            if 'volume' in adjusted: adjusted['volume'] = float(adjusted['volume']) * float(liq_w)

            total = float(sum(adjusted.values()))
            if total > 0:
                adjusted = {k: float(v) / total for k, v in adjusted.items()}
            else:
                total_base = float(sum(base_weights.values()))
                adjusted = {k: float(v) / total_base for k, v in base_weights.items()}

            return adjusted

        except Exception as e:
            self.logger.error(f"Weight adjustment failed: {e}")
            return base_weights

    def get_hmm_regime_label(self) -> str:
        """Return the current HMM regime label string."""
        if self._advanced_regime_state is not None:
            return self._advanced_regime_state.current_regime.value
        if self.current_regime:
            return self.current_regime.get('hmm_regime', 'unknown')
        return 'unknown'

    def get_confidence_boost(self) -> float:
        """Get confidence boost factor from regime analysis."""
        if not self.enabled or not self.current_regime:
            return 0.0

        # Use advanced HMM confidence if available
        if self._advanced_regime_state is not None:
            conf = self._advanced_regime_state.confidence
            return float((conf - 0.5) * 0.1)  # Max +/- 5% boost

        conf_score = self.current_regime.get('confidence_score', 0.5)
        return (conf_score - 0.5) * 0.1
