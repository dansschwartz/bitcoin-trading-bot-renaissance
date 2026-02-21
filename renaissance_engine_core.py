"""
Renaissance Engine Core
Contains base classes and utilities to avoid circular dependencies.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime

from renaissance_types import MLSignalPackage, TradingDecision

logger = logging.getLogger(__name__)

class SignalFusion:
    """Signal fusion results"""

    def __init__(self):
        self.combined_signal = 0.0
        self.confidence = 0.0
        self.contributing_signals = {}
        self.weights = {}
        self._ml_signal_scale = 10.0  # Amplification for small ML predictions

    def set_ml_signal_scale(self, scale: float) -> None:
        """Set ML signal amplification factor (default 10.0)."""
        self._ml_signal_scale = float(scale)

    def fuse_signals(self, signals: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, float, Dict[str, Any]]:
        """Fuse multiple signals with weights"""
        # Resolve weights key naming (lowercase string vs enum)
        processed_weights = {}
        for k, v in weights.items():
            processed_weights[k.lower() if isinstance(k, str) else k] = v

        weighted_sum = 0.0
        total_weight = 0.0
        contributions = {}

        def force_float(val):
            try:
                temp = val
                if temp is None: return 0.0
                while hasattr(temp, '__iter__') and not isinstance(temp, (str, bytes, dict)):
                    if hasattr(temp, '__len__') and len(temp) > 0:
                        temp = temp[0]
                    else:
                        temp = 0.0
                        break
                if hasattr(temp, 'item'): temp = temp.item()
                return float(temp)
            except:
                return 0.0

        for k, v in signals.items():
            w = processed_weights.get(k.lower() if isinstance(k, str) else k, 0.0)
            try:
                val_f = force_float(v)
                weight_f = force_float(w)
                weighted_sum += val_f * weight_f
                total_weight += weight_f
                contributions[str(k)] = float(val_f * weight_f)
            except:
                continue

        combined = float(weighted_sum / total_weight) if total_weight > 0 else 0.0
        confidence = float(min(abs(combined) * 1.5, 1.0)) # Heuristic confidence

        return combined, confidence, {
            'contributions': contributions,
            'weights': weights
        }

    def fuse_signals_with_ml(self, signals: Dict[str, float],
                             weights: Dict[str, float],
                             ml_package: Optional[MLSignalPackage] = None) -> Tuple[float, float, Dict[str, Any]]:
        """Enhanced signal fusion including ML predictions"""

        # Start with traditional signal fusion
        traditional_strength, traditional_confidence, traditional_metadata = self.fuse_signals(signals, weights)

        if not ml_package:
            return traditional_strength, traditional_confidence, traditional_metadata

        # Extract ML insights — scale small predictions to usable range
        ml_scale = self._ml_signal_scale
        ml_strength = float(np.clip(ml_package.ensemble_score * ml_scale, -1.0, 1.0))
        ml_confidence = ml_package.confidence_score

        # Combine traditional and ML signals
        # We give ML a significant but balanced weight
        ml_weight = 0.35 # Institutional default
        trad_weight = 1.0 - ml_weight

        # Weighted combination
        final_strength = (traditional_strength * trad_weight) + (ml_strength * ml_weight)
        
        # Combined confidence with consciousness weighting
        consciousness_factor = ml_package.confidence_score
        final_confidence = (traditional_confidence * trad_weight) + (ml_confidence * ml_weight * consciousness_factor)

        # Enhanced metadata
        enhanced_metadata = {
            **traditional_metadata,
            'ml_ensemble_score': ml_strength,
            'ml_confidence': ml_confidence,
            'consciousness_score': consciousness_factor,
            'fractal_insights': ml_package.fractal_insights,
            'fusion_method': 'ml_enhanced'
        }

        return float(np.clip(final_strength, -1.0, 1.0)), float(np.clip(final_confidence, 0.0, 1.0)), enhanced_metadata


class RiskManager:
    """Risk management component with ML-informed decisions"""
    def __init__(self, daily_loss_limit=500.0, position_limit=1000.0, **kwargs):
        self.max_position_size = float(position_limit)
        self.current_risk = 0.0
        self.daily_pnl = 0.0
        self.daily_loss_limit = float(daily_loss_limit)
        self.position_limit = float(position_limit)
        self.risk_limits = {
            'daily_loss': self.daily_loss_limit,
            'position_limit': self.position_limit
        }
        
        # ML-enhanced risk parameters (Unified from Enhanced Bot)
        self.ml_risk_adjustment = 0.8  # How much ML confidence affects position size
        self.consciousness_threshold = 0.6  # Minimum consciousness score for full position
        self.fractal_volatility_adjustment = 0.9  # Fractal-based volatility adjustment
        
        # Accept any additional arguments without error
        for key, value in kwargs.items():
            setattr(self, key, value)

    def calculate_position_size(self, signal_strength: float, confidence: float, current_price: float) -> float:
        """Calculate base position size in currency units"""
        # Type guard for inputs
        def force_float(val):
            try:
                temp = val
                if temp is None: return 0.0
                while hasattr(temp, '__iter__') and not isinstance(temp, (str, bytes, dict)):
                    if hasattr(temp, '__len__') and len(temp) > 0:
                        temp = temp[0]
                    else:
                        temp = 0.0
                        break
                if hasattr(temp, 'item'): temp = temp.item()
                return float(temp)
            except:
                return 0.0

        try:
            s_f = force_float(signal_strength)
            c_f = force_float(confidence)
            # Base logic: strength * confidence * position_limit
            base_size = abs(s_f) * c_f * self.position_limit
            return float(np.clip(base_size, 0.0, self.position_limit))
        except:
            return 0.0

    def calculate_ml_enhanced_position_size(self, signal_strength: float, confidence: float, 
                                          current_price: float, ml_package: Optional[MLSignalPackage]) -> float:
        """Calculate position size with ML enhancements"""
        # Type guard for inputs
        def force_float(val):
            temp = val
            while hasattr(temp, '__iter__') and not isinstance(temp, (str, bytes, dict)):
                if hasattr(temp, '__len__') and len(temp) > 0:
                    temp = temp[0]
                else:
                    temp = 0.0
                    break
            if hasattr(temp, 'item'): temp = temp.item()
            return float(temp)
        
        # Start with base calculation
        base_size = self.calculate_position_size(signal_strength, confidence, current_price)
        
        if not ml_package:
            return float(base_size)

        try:
            # ML confidence adjustment
            ml_confidence_adj = 1.0
            ml_conf_f = force_float(ml_package.confidence_score)
            if ml_conf_f > 0:
                ml_confidence_adj = (
                    self.ml_risk_adjustment * ml_conf_f + 
                    (1.0 - self.ml_risk_adjustment)
                )

            # Consciousness score adjustment
            consciousness_adj = 1.0
            if ml_conf_f < self.consciousness_threshold:
                consciousness_adj = ml_conf_f / self.consciousness_threshold

            # Fractal regime adjustment
            fractal_adj = 1.0
            regime = ml_package.fractal_insights.get('regime_detection', 'normal')
            if regime == 'chaotic':
                fractal_adj = 0.5  # Reduce position in chaotic markets
            elif regime == 'trending':
                fractal_adj = 1.2  # Increase position in trending markets
            elif regime == 'mean_reverting':
                fractal_adj = 0.8  # Moderate reduction in mean-reverting markets

            # Apply all adjustments
            enhanced_size = float(base_size) * float(ml_confidence_adj) * float(consciousness_adj) * float(fractal_adj)

            # Ensure within limits
            return float(np.clip(enhanced_size, 0.0, self.position_limit))
        except Exception as e:
            return float(base_size)

    def assess_risk_regime(self, ml_package: Optional[MLSignalPackage]) -> Dict[str, Any]:
        """Assess current risk regime using ML insights"""
        risk_assessment = {
            'overall_risk': 'medium',
            'ml_model_health': 'good',
            'consciousness_level': 'normal',
            'fractal_regime': 'stable',
            'recommended_action': 'proceed_normal'
        }
        
        if not ml_package:
            return risk_assessment

        # Consciousness level assessment
        consciousness = ml_package.confidence_score
        if consciousness > 0.8:
            risk_assessment['consciousness_level'] = 'high'
        elif consciousness < 0.4:
            risk_assessment['consciousness_level'] = 'low'
            risk_assessment['recommended_action'] = 'proceed_cautious'

        # Fractal regime assessment
        fractal_regime = ml_package.fractal_insights.get('regime_detection', 'normal')
        risk_assessment['fractal_regime'] = fractal_regime

        if fractal_regime == 'chaotic':
            risk_assessment['overall_risk'] = 'high'
            risk_assessment['recommended_action'] = 'reduce_exposure'
        elif fractal_regime == 'trending':
            risk_assessment['overall_risk'] = 'low'
            risk_assessment['recommended_action'] = 'increase_exposure'

        # Processing time as risk factor
        if ml_package.processing_time_ms > 1000:  # Over 1 second
            risk_assessment['ml_model_health'] = 'slow'
        elif ml_package.processing_time_ms > 5000:  # Over 5 seconds
            risk_assessment['ml_model_health'] = 'degraded'
            risk_assessment['recommended_action'] = 'fallback_mode'

        return risk_assessment
