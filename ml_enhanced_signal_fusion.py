
"""
ML-Enhanced Signal Fusion System
Advanced signal fusion combining traditional Renaissance signals with ML predictions
using consciousness engine and fractal analysis
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Import base components
from renaissance_trading_bot import TradingSignal, SignalType
from ml_integration_bridge import MLSignalPackage

class FusionStrategy(Enum):
    """Different fusion strategies available"""
    WEIGHTED_AVERAGE = "weighted_average"
    ENSEMBLE_VOTING = "ensemble_voting"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE_DYNAMIC = "adaptive_dynamic"
    CONSCIOUSNESS_GUIDED = "consciousness_guided"

@dataclass
class FusionResult:
    """Result of signal fusion process"""
    final_strength: float  # -1.0 to 1.0
    final_confidence: float  # 0.0 to 1.0
    fusion_strategy_used: FusionStrategy

    # Signal contributions
    traditional_contribution: float
    ml_contribution: float
    consciousness_weight: float
    fractal_adjustment: float

    # Quality metrics
    signal_coherence: float  # How well signals agree
    information_content: float  # Information richness
    temporal_consistency: float  # Consistency over time

    # Metadata
    processing_details: Dict[str, Any]
    fusion_timestamp: datetime

class MetaSignalAnalyzer:
    """Analyzes signals at a meta level for quality and coherence"""

    def __init__(self):
        self.signal_history = []
        self.coherence_threshold = 0.6

    def analyze_signal_quality(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Analyze the quality of incoming signals"""
        if not signals:
            return {'strength_consistency': 0.0, 'confidence_distribution': 0.0, 'temporal_quality': 0.0}

        # Strength consistency (how much signals agree on direction)
        strengths = [s.strength for s in signals]
        strength_consistency = self._calculate_consistency(strengths)

        # Confidence distribution (diversity of confidence levels)
        confidences = [s.confidence for s in signals]
        confidence_distribution = np.std(confidences) if len(confidences) > 1 else 0.5

        # Temporal quality (how fresh are the signals)
        current_time = datetime.now()
        ages = [(current_time - s.timestamp).total_seconds() for s in signals]
        avg_age = np.mean(ages)
        temporal_quality = max(0.0, 1.0 - min(avg_age / 300, 1.0))  # 5 minute decay

        return {
            'strength_consistency': strength_consistency,
            'confidence_distribution': confidence_distribution,
            'temporal_quality': temporal_quality
        }

    def analyze_ml_package_quality(self, ml_package: MLSignalPackage) -> Dict[str, float]:
        """Analyze the quality of ML predictions"""
        quality_metrics = {
            'prediction_diversity': 0.0,
            'ensemble_stability': 0.0,
            'processing_efficiency': 0.0,
            'fractal_coherence': 0.0
        }

        # Prediction diversity
        if ml_package.ml_predictions:
            strengths = [p.get('strength', 0) if isinstance(p, dict) else 0 
                        for p in ml_package.ml_predictions]
            quality_metrics['prediction_diversity'] = np.std(strengths) if len(strengths) > 1 else 0.5

        # Ensemble stability (how stable is the ensemble score)
        ensemble_score = abs(ml_package.ensemble_score)
        quality_metrics['ensemble_stability'] = min(ensemble_score, 1.0)

        # Processing efficiency
        processing_time = ml_package.processing_time_ms
        quality_metrics['processing_efficiency'] = max(0.0, 1.0 - min(processing_time / 5000, 1.0))

        # Fractal coherence
        fractal_dimension = ml_package.fractal_insights.get('fractal_dimension', 1.5)
        quality_metrics['fractal_coherence'] = 1.0 - abs(fractal_dimension - 1.5) / 0.5

        return quality_metrics

    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency of values (inverse of normalized standard deviation)"""
        if len(values) <= 1:
            return 1.0

        std_dev = np.std(values)
        mean_abs = np.mean([abs(v) for v in values])

        if mean_abs == 0:
            return 0.5

        # Normalized consistency (0 = inconsistent, 1 = perfectly consistent)
        consistency = max(0.0, 1.0 - (std_dev / (mean_abs + 1e-6)))
        return consistency

class ConsciousnessGuidedFusion:
    """Advanced fusion using consciousness-like attention mechanisms"""

    def __init__(self):
        self.attention_mechanisms = {
            'signal_importance': self._calculate_signal_importance,
            'temporal_attention': self._calculate_temporal_attention,
            'confidence_attention': self._calculate_confidence_attention,
            'coherence_attention': self._calculate_coherence_attention
        }

        # Attention weights (sum to 1.0)
        self.attention_weights = {
            'signal_importance': 0.3,
            'temporal_attention': 0.2,
            'confidence_attention': 0.25,
            'coherence_attention': 0.25
        }

    def apply_consciousness_fusion(self, traditional_signals: List[TradingSignal], 
                                  ml_package: MLSignalPackage,
                                  signal_quality: Dict[str, float],
                                  ml_quality: Dict[str, float]) -> Tuple[float, float, Dict[str, Any]]:
        """Apply consciousness-guided fusion"""

        # Calculate attention scores for different aspects
        attention_scores = {}
        for mechanism_name, mechanism_func in self.attention_mechanisms.items():
            attention_scores[mechanism_name] = mechanism_func(
                traditional_signals, ml_package, signal_quality, ml_quality
            )

        # Weighted attention score
        overall_attention = sum(
            score * self.attention_weights.get(mechanism_name, 0.25)
            for mechanism_name, score in attention_scores.items()
        )

        # Traditional signal processing
        traditional_strength = self._process_traditional_signals(traditional_signals, attention_scores)
        traditional_confidence = self._calculate_traditional_confidence(traditional_signals, attention_scores)

        # ML signal processing
        ml_strength = ml_package.ensemble_score
        ml_confidence = ml_package.confidence_score

        # Consciousness-weighted combination
        consciousness_weight = overall_attention

        # Dynamic weight allocation based on consciousness
        if consciousness_weight > 0.8:  # High consciousness - trust ML more
            ml_weight = 0.6
            traditional_weight = 0.4
        elif consciousness_weight < 0.4:  # Low consciousness - trust traditional more
            ml_weight = 0.3
            traditional_weight = 0.7
        else:  # Balanced consciousness
            ml_weight = 0.5
            traditional_weight = 0.5

        # Final fusion
        final_strength = (
            traditional_strength * traditional_weight * traditional_confidence +
            ml_strength * ml_weight * ml_confidence * consciousness_weight
        )

        final_confidence = (
            traditional_confidence * traditional_weight +
            ml_confidence * ml_weight * consciousness_weight
        )

        fusion_metadata = {
            'attention_scores': attention_scores,
            'overall_attention': overall_attention,
            'consciousness_weight': consciousness_weight,
            'weights_used': {'traditional': traditional_weight, 'ml': ml_weight},
            'fusion_method': 'consciousness_guided'
        }

        return (
            np.clip(final_strength, -1.0, 1.0),
            np.clip(final_confidence, 0.0, 1.0),
            fusion_metadata
        )

    def _calculate_signal_importance(self, traditional_signals: List[TradingSignal], 
                                    ml_package: MLSignalPackage,
                                    signal_quality: Dict[str, float],
                                    ml_quality: Dict[str, float]) -> float:
        """Calculate importance based on signal characteristics"""

        # Traditional signals importance
        traditional_importance = 0.5
        if traditional_signals:
            avg_confidence = np.mean([s.confidence for s in traditional_signals])
            signal_count_factor = min(len(traditional_signals) / 5.0, 1.0)
            traditional_importance = avg_confidence * signal_count_factor

        # ML signals importance
        ml_importance = ml_package.confidence_score * ml_quality.get('ensemble_stability', 0.5)

        # Combined importance
        return (traditional_importance + ml_importance) / 2.0

    def _calculate_temporal_attention(self, traditional_signals: List[TradingSignal], 
                                     ml_package: MLSignalPackage,
                                     signal_quality: Dict[str, float],
                                     ml_quality: Dict[str, float]) -> float:
        """Calculate attention based on temporal factors"""

        # Recency of traditional signals
        current_time = datetime.now()
        if traditional_signals:
            ages = [(current_time - s.timestamp).total_seconds() for s in traditional_signals]
            avg_age = np.mean(ages)
            temporal_score = max(0.0, 1.0 - min(avg_age / 600, 1.0))  # 10 minute decay
        else:
            temporal_score = 0.5

        # ML processing efficiency
        processing_efficiency = ml_quality.get('processing_efficiency', 0.5)

        return (temporal_score + processing_efficiency) / 2.0

    def _calculate_confidence_attention(self, traditional_signals: List[TradingSignal], 
                                       ml_package: MLSignalPackage,
                                       signal_quality: Dict[str, float],
                                       ml_quality: Dict[str, float]) -> float:
        """Calculate attention based on confidence levels"""

        # Traditional confidence
        traditional_conf = 0.5
        if traditional_signals:
            confidences = [s.confidence for s in traditional_signals]
            traditional_conf = np.mean(confidences)

        # ML confidence
        ml_conf = ml_package.confidence_score

        # Weighted by quality
        traditional_weighted = traditional_conf * signal_quality.get('confidence_distribution', 0.5)
        ml_weighted = ml_conf * ml_quality.get('prediction_diversity', 0.5)

        return (traditional_weighted + ml_weighted) / 2.0

    def _calculate_coherence_attention(self, traditional_signals: List[TradingSignal], 
                                      ml_package: MLSignalPackage,
                                      signal_quality: Dict[str, float],
                                      ml_quality: Dict[str, float]) -> float:
        """Calculate attention based on signal coherence"""

        # Traditional signal coherence
        traditional_coherence = signal_quality.get('strength_consistency', 0.5)

        # ML fractal coherence
        ml_coherence = ml_quality.get('fractal_coherence', 0.5)

        # Cross-signal coherence (do traditional and ML agree?)
        cross_coherence = 0.5
        if traditional_signals and ml_package.ensemble_score != 0:
            traditional_direction = np.mean([s.strength for s in traditional_signals])
            ml_direction = ml_package.ensemble_score

            if (traditional_direction > 0 and ml_direction > 0) or (traditional_direction < 0 and ml_direction < 0):
                cross_coherence = 0.8  # They agree
            else:
                cross_coherence = 0.2  # They disagree

        return (traditional_coherence + ml_coherence + cross_coherence) / 3.0

    def _process_traditional_signals(self, signals: List[TradingSignal], 
                                    attention_scores: Dict[str, float]) -> float:
        """Process traditional signals with attention weighting"""
        if not signals:
            return 0.0

        # Weight signals by confidence and attention
        weighted_strength = 0.0
        total_weight = 0.0

        for signal in signals:
            # Attention-based weight
            attention_weight = attention_scores.get('signal_importance', 0.5)
            confidence_weight = signal.confidence
            temporal_weight = attention_scores.get('temporal_attention', 0.5)

            combined_weight = attention_weight * confidence_weight * temporal_weight

            weighted_strength += signal.strength * combined_weight
            total_weight += combined_weight

        return weighted_strength / (total_weight + 1e-6)

    def _calculate_traditional_confidence(self, signals: List[TradingSignal], 
                                         attention_scores: Dict[str, float]) -> float:
        """Calculate confidence for traditional signals"""
        if not signals:
            return 0.0

        # Base confidence from signals
        base_confidence = np.mean([s.confidence for s in signals])

        # Adjust by attention scores
        coherence_factor = attention_scores.get('coherence_attention', 0.5)
        temporal_factor = attention_scores.get('temporal_attention', 0.5)

        adjusted_confidence = base_confidence * coherence_factor * (0.5 + 0.5 * temporal_factor)

        return np.clip(adjusted_confidence, 0.0, 1.0)

class MLEnhancedSignalFusion:
    """Main ML-Enhanced Signal Fusion System"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize components
        self.meta_analyzer = MetaSignalAnalyzer()
        self.consciousness_fusion = ConsciousnessGuidedFusion()

        # Fusion parameters
        self.default_strategy = FusionStrategy.CONSCIOUSNESS_GUIDED
        self.fallback_strategy = FusionStrategy.WEIGHTED_AVERAGE

        # Performance tracking
        self.fusion_history = []
        self.performance_metrics = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'avg_processing_time': 0.0,
            'strategy_performance': {strategy: {'count': 0, 'success': 0} for strategy in FusionStrategy}
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def fuse_signals(self, traditional_signals: List[TradingSignal], 
                    ml_package: MLSignalPackage,
                    strategy: Optional[FusionStrategy] = None) -> FusionResult:
        """Main signal fusion method"""

        start_time = datetime.now()

        # Use default strategy if none specified
        if strategy is None:
            strategy = self.default_strategy

        try:
            # Analyze signal quality
            signal_quality = self.meta_analyzer.analyze_signal_quality(traditional_signals)
            ml_quality = self.meta_analyzer.analyze_ml_package_quality(ml_package)

            # Apply selected fusion strategy
            if strategy == FusionStrategy.CONSCIOUSNESS_GUIDED:
                strength, confidence, metadata = self.consciousness_fusion.apply_consciousness_fusion(
                    traditional_signals, ml_package, signal_quality, ml_quality
                )
            else:
                # Fallback to simpler strategy
                strength, confidence, metadata = self._apply_fallback_fusion(
                    traditional_signals, ml_package, signal_quality, ml_quality
                )

            # Calculate additional quality metrics
            signal_coherence = self._calculate_signal_coherence(traditional_signals, ml_package)
            information_content = self._calculate_information_content(traditional_signals, ml_package)
            temporal_consistency = self._calculate_temporal_consistency(traditional_signals)

            # Create fusion result
            result = FusionResult(
                final_strength=strength,
                final_confidence=confidence,
                fusion_strategy_used=strategy,
                traditional_contribution=metadata.get('weights_used', {}).get('traditional', 0.5),
                ml_contribution=metadata.get('weights_used', {}).get('ml', 0.5),
                consciousness_weight=metadata.get('consciousness_weight', 0.5),
                fractal_adjustment=self._calculate_fractal_adjustment(ml_package),
                signal_coherence=signal_coherence,
                information_content=information_content,
                temporal_consistency=temporal_consistency,
                processing_details={
                    **metadata,
                    'signal_quality': signal_quality,
                    'ml_quality': ml_quality,
                    'traditional_signal_count': len(traditional_signals),
                    'ml_signal_count': len(ml_package.primary_signals)
                },
                fusion_timestamp=datetime.now()
            )

            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(strategy, True, processing_time)

            # Store in history
            self.fusion_history.append(result)
            if len(self.fusion_history) > 100:  # Keep last 100 results
                self.fusion_history.pop(0)

            return result

        except Exception as e:
            self.logger.error(f"Error in signal fusion: {e}")

            # Update performance metrics for failure
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(strategy, False, processing_time)

            # Return safe fallback result
            return self._create_fallback_result(traditional_signals, ml_package, start_time)

    def _apply_fallback_fusion(self, traditional_signals: List[TradingSignal], 
                              ml_package: MLSignalPackage,
                              signal_quality: Dict[str, float],
                              ml_quality: Dict[str, float]) -> Tuple[float, float, Dict[str, Any]]:
        """Apply simple weighted average fusion as fallback"""

        # Simple weighted average
        traditional_strength = 0.0
        traditional_confidence = 0.0

        if traditional_signals:
            strengths = [s.strength * s.confidence for s in traditional_signals]
            confidences = [s.confidence for s in traditional_signals]

            total_conf = sum(confidences)
            if total_conf > 0:
                traditional_strength = sum(strengths) / total_conf
                traditional_confidence = sum(confidences) / len(confidences)

        # ML components
        ml_strength = ml_package.ensemble_score
        ml_confidence = ml_package.confidence_score

        # Simple 50/50 weighting
        final_strength = 0.5 * traditional_strength + 0.5 * ml_strength
        final_confidence = 0.5 * traditional_confidence + 0.5 * ml_confidence

        metadata = {
            'weights_used': {'traditional': 0.5, 'ml': 0.5},
            'consciousness_weight': 0.5,
            'fusion_method': 'weighted_average_fallback'
        }

        return (
            np.clip(final_strength, -1.0, 1.0),
            np.clip(final_confidence, 0.0, 1.0),
            metadata
        )

    def _calculate_signal_coherence(self, traditional_signals: List[TradingSignal], 
                                   ml_package: MLSignalPackage) -> float:
        """Calculate coherence between traditional and ML signals"""

        if not traditional_signals or ml_package.ensemble_score == 0:
            return 0.5

        # Direction agreement
        traditional_direction = np.mean([s.strength for s in traditional_signals])
        ml_direction = ml_package.ensemble_score

        if (traditional_direction > 0 and ml_direction > 0) or (traditional_direction < 0 and ml_direction < 0):
            direction_coherence = 0.8
        elif traditional_direction == 0 or ml_direction == 0:
            direction_coherence = 0.5
        else:
            direction_coherence = 0.2

        # Magnitude coherence
        traditional_magnitude = abs(traditional_direction)
        ml_magnitude = abs(ml_direction)

        magnitude_diff = abs(traditional_magnitude - ml_magnitude)
        magnitude_coherence = max(0.0, 1.0 - magnitude_diff)

        return (direction_coherence + magnitude_coherence) / 2.0

    def _calculate_information_content(self, traditional_signals: List[TradingSignal], 
                                     ml_package: MLSignalPackage) -> float:
        """Calculate information content of the signal fusion"""

        # Traditional information
        traditional_info = 0.0
        if traditional_signals:
            # Diversity of signal types
            signal_types = set(s.signal_type for s in traditional_signals)
            type_diversity = len(signal_types) / len(SignalType)

            # Confidence distribution
            confidences = [s.confidence for s in traditional_signals]
            conf_std = np.std(confidences) if len(confidences) > 1 else 0.5

            traditional_info = (type_diversity + conf_std) / 2.0

        # ML information
        ml_info = 0.0
        if ml_package.primary_signals:
            ml_signal_count = len(ml_package.primary_signals)
            ml_diversity = min(ml_signal_count / 5.0, 1.0)  # Normalize to max 5 signals

            fractal_info = abs(ml_package.fractal_insights.get('fractal_dimension', 1.5) - 1.0)

            ml_info = (ml_diversity + fractal_info) / 2.0

        return (traditional_info + ml_info) / 2.0

    def _calculate_temporal_consistency(self, traditional_signals: List[TradingSignal]) -> float:
        """Calculate temporal consistency of signals"""

        if len(traditional_signals) < 2:
            return 0.7

        # Time span of signals
        timestamps = [s.timestamp for s in traditional_signals]
        time_span = (max(timestamps) - min(timestamps)).total_seconds()

        # Consistency is higher when signals are close in time
        consistency = max(0.0, 1.0 - min(time_span / 300, 1.0))  # 5 minute window

        return consistency

    def _calculate_fractal_adjustment(self, ml_package: MLSignalPackage) -> float:
        """Calculate fractal-based adjustment factor"""

        fractal_insights = ml_package.fractal_insights
        regime = fractal_insights.get('regime_detection', 'normal')

        adjustment_factors = {
            'trending': 1.1,
            'mean_reverting': 0.9,
            'chaotic': 0.8,
            'normal': 1.0
        }

        return adjustment_factors.get(regime, 1.0)

    def _update_performance_metrics(self, strategy: FusionStrategy, success: bool, processing_time: float):
        """Update performance tracking metrics"""

        self.performance_metrics['total_fusions'] += 1
        if success:
            self.performance_metrics['successful_fusions'] += 1

        # Update average processing time
        total = self.performance_metrics['total_fusions']
        current_avg = self.performance_metrics['avg_processing_time']
        self.performance_metrics['avg_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )

        # Update strategy-specific metrics
        strategy_perf = self.performance_metrics['strategy_performance'][strategy]
        strategy_perf['count'] += 1
        if success:
            strategy_perf['success'] += 1

    def _create_fallback_result(self, traditional_signals: List[TradingSignal], 
                               ml_package: MLSignalPackage, start_time: datetime) -> FusionResult:
        """Create a safe fallback result in case of errors"""

        return FusionResult(
            final_strength=0.0,
            final_confidence=0.3,
            fusion_strategy_used=FusionStrategy.WEIGHTED_AVERAGE,
            traditional_contribution=0.5,
            ml_contribution=0.5,
            consciousness_weight=0.5,
            fractal_adjustment=1.0,
            signal_coherence=0.5,
            information_content=0.3,
            temporal_consistency=0.5,
            processing_details={'error': 'fallback_result_created'},
            fusion_timestamp=datetime.now()
        )

    def get_fusion_performance(self) -> Dict[str, Any]:
        """Get comprehensive fusion performance metrics"""

        success_rate = (
            self.performance_metrics['successful_fusions'] / 
            max(self.performance_metrics['total_fusions'], 1)
        )

        strategy_success_rates = {}
        for strategy, perf in self.performance_metrics['strategy_performance'].items():
            if perf['count'] > 0:
                strategy_success_rates[strategy.value] = perf['success'] / perf['count']
            else:
                strategy_success_rates[strategy.value] = 0.0

        return {
            'overall_success_rate': success_rate,
            'total_fusions': self.performance_metrics['total_fusions'],
            'avg_processing_time_ms': self.performance_metrics['avg_processing_time'],
            'strategy_success_rates': strategy_success_rates,
            'recent_fusion_count': len(self.fusion_history),
            'last_fusion_time': self.fusion_history[-1].fusion_timestamp if self.fusion_history else None
        }

    def get_recent_results(self, count: int = 10) -> List[FusionResult]:
        """Get recent fusion results"""
        return self.fusion_history[-count:] if self.fusion_history else []
