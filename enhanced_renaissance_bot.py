"""
Enhanced Renaissance Trading Bot with ML Integration
Extends the base Renaissance bot with revolutionary ML capabilities
"""

import numpy as np
import pandas as pd
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import renaissance_trading_bot

# Import ML configuration (FIXED)
from ml_config import MLPatternConfig

# Import ML integration
from ml_integration_bridge import MLIntegrationBridge, MLSignalPackage

@dataclass
class EnhancedTradingDecision:
    """Enhanced trading decision with ML insights"""
    # Base decision
    action: renaissance_trading_bot.OrderType
    size: float
    strength: float
    confidence: float

    # ML enhancements
    ml_ensemble_score: float
    ml_confidence: float
    fractal_insights: Dict[str, Any]
    consciousness_score: float

    # Signal breakdown
    traditional_signal_count: int
    ml_signal_count: int
    signal_fusion_weights: Dict[str, float]

    # Risk and performance
    risk_adjusted_size: float
    expected_performance: float
    adaptive_confidence: float

    # Metadata
    timestamp: datetime
    processing_time_ms: float
    model_health_status: Dict[str, Any]

class AdaptiveSignalFusion(renaissance_trading_bot.SignalFusion):
    """Enhanced signal fusion with ML integration and adaptive weighting"""

    def __init__(self):
        super().__init__()
        # Enhanced weights that adapt based on performance
        self.adaptive_weights = {
            renaissance_trading_bot.SignalType.MICROSTRUCTURE: 0.35,
            renaissance_trading_bot.SignalType.TECHNICAL: 0.25,
            renaissance_trading_bot.SignalType.MOMENTUM: 0.15,
            renaissance_trading_bot.SignalType.MEAN_REVERSION: 0.1,
            renaissance_trading_bot.SignalType.ALTERNATIVE: 0.05,
            'ml_ensemble': 0.1  # Initial ML weight
        }

        # Performance tracking for adaptive weighting
        self.signal_performance = {
            signal_type: {'correct': 0, 'total': 0, 'avg_return': 0.0}
            for signal_type in self.adaptive_weights.keys()
        }

        self.adaptation_rate = 0.05  # How fast weights adapt

    def fuse_signals_with_ml(self, traditional_signals: List[renaissance_trading_bot.TradingDecision],
                             ml_package: MLSignalPackage) -> Tuple[float, float, Dict[str, Any]]:
        """Enhanced signal fusion including ML predictions"""

        # Start with traditional signal fusion
        traditional_strength, traditional_confidence, traditional_metadata = self.fuse_signals(traditional_signals)

        # Extract ML insights
        ml_strength = ml_package.ensemble_score
        ml_confidence = ml_package.confidence_score

        # Adaptive weight adjustment based on recent performance
        self._update_adaptive_weights()

        # Combine traditional and ML signals
        traditional_weight = sum([
            self.adaptive_weights.get(signal_type, 0.1) 
            for signal_type in [renaissance_trading_bot.SignalType.MICROSTRUCTURE, renaissance_trading_bot.SignalType.TECHNICAL,
                                renaissance_trading_bot.SignalType.MOMENTUM, renaissance_trading_bot.SignalType.MEAN_REVERSION, renaissance_trading_bot.SignalType.ALTERNATIVE]
        ])

        ml_weight = self.adaptive_weights.get('ml_ensemble', 0.1)

        # Normalize weights
        total_weight = traditional_weight + ml_weight
        if total_weight > 0:
            traditional_weight /= total_weight
            ml_weight /= total_weight

        # Weighted combination
        final_strength = (
            traditional_strength * traditional_weight * traditional_confidence +
            ml_strength * ml_weight * ml_confidence
        )

        # Combined confidence with consciousness weighting
        consciousness_factor = ml_package.confidence_score
        final_confidence = (
            traditional_confidence * traditional_weight +
            ml_confidence * ml_weight * consciousness_factor
        )

        # Enhanced metadata
        enhanced_metadata = {
            **traditional_metadata,
            'ml_ensemble_score': ml_strength,
            'ml_confidence': ml_confidence,
            'consciousness_score': consciousness_factor,
            'fractal_insights': ml_package.fractal_insights,
            'adaptive_weights': self.adaptive_weights.copy(),
            'signal_performance': self._get_performance_summary(),
            'fusion_method': 'adaptive_ml_enhanced'
        }

        return np.clip(final_strength, -1.0, 1.0), np.clip(final_confidence, 0.0, 1.0), enhanced_metadata

    def _update_adaptive_weights(self):
        """Update signal weights based on recent performance"""
        for signal_type, performance in self.signal_performance.items():
            if performance['total'] > 10:  # Need sufficient data
                success_rate = performance['correct'] / performance['total']
                avg_return = performance['avg_return']

                # Performance score combining accuracy and returns
                performance_score = (success_rate * 0.6) + (min(max(avg_return, -0.1), 0.1) * 5 + 0.5) * 0.4

                # Adjust weights based on performance
                current_weight = self.adaptive_weights.get(signal_type, 0.1)
                if performance_score > 0.6:  # Good performance
                    new_weight = current_weight * (1 + self.adaptation_rate)
                elif performance_score < 0.4:  # Poor performance
                    new_weight = current_weight * (1 - self.adaptation_rate)
                else:
                    new_weight = current_weight

                self.adaptive_weights[signal_type] = max(0.01, min(new_weight, 0.5))

        # Normalize weights to sum to 1
        total_weight = sum(self.adaptive_weights.values())
        if total_weight > 0:
            for signal_type in self.adaptive_weights:
                self.adaptive_weights[signal_type] /= total_weight

    def update_signal_performance(self, signal_type: str, was_correct: bool, return_achieved: float):
        """Update performance tracking for adaptive weighting"""
        if signal_type in self.signal_performance:
            perf = self.signal_performance[signal_type]
            perf['total'] += 1
            if was_correct:
                perf['correct'] += 1

            # Update running average of returns
            alpha = 0.1  # Smoothing factor
            perf['avg_return'] = (1 - alpha) * perf['avg_return'] + alpha * return_achieved

    def _get_performance_summary(self) -> Dict[str, float]:
        """Get summary of signal performance"""
        summary = {}
        for signal_type, performance in self.signal_performance.items():
            if performance['total'] > 0:
                success_rate = performance['correct'] / performance['total']
                summary[str(signal_type)] = {
                    'success_rate': success_rate,
                    'avg_return': performance['avg_return'],
                    'sample_size': performance['total']
                }
        return summary

class EnhancedRiskManager(renaissance_trading_bot.RiskManager):
    """Enhanced risk management with ML-informed decisions"""

    def __init__(self, max_position_size: float = 0.02, max_drawdown: float = 0.05):
        super().__init__(max_position_size, max_drawdown)

        # ML-enhanced risk parameters
        self.ml_risk_adjustment = 0.8  # How much ML confidence affects position size
        self.consciousness_threshold = 0.6  # Minimum consciousness score for full position
        self.fractal_volatility_adjustment = 0.9  # Fractal-based volatility adjustment

        # Enhanced risk tracking
        self.ml_position_history = []
        self.consciousness_history = []
        self.fractal_regime_history = []

    def calculate_enhanced_position_size(self, signal_strength: float, confidence: float, 
                                       current_price: float, ml_package: MLSignalPackage) -> float:
        """Calculate position size with ML enhancements"""

        # Start with base calculation
        base_size = self.calculate_position_size(signal_strength, confidence, current_price)

        # ML confidence adjustment
        ml_confidence_adj = 1.0
        if ml_package.confidence_score > 0:
            ml_confidence_adj = (
                self.ml_risk_adjustment * ml_package.confidence_score + 
                (1 - self.ml_risk_adjustment)
            )

        # Consciousness score adjustment
        consciousness_adj = 1.0
        if ml_package.confidence_score < self.consciousness_threshold:
            consciousness_adj = ml_package.confidence_score / self.consciousness_threshold

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
        enhanced_size = base_size * ml_confidence_adj * consciousness_adj * fractal_adj

        # Ensure within limits
        enhanced_size = np.clip(enhanced_size, -self.max_position_size, self.max_position_size)

        # Track for analysis
        self.ml_position_history.append({
            'timestamp': datetime.now(),
            'base_size': base_size,
            'enhanced_size': enhanced_size,
            'ml_confidence_adj': ml_confidence_adj,
            'consciousness_adj': consciousness_adj,
            'fractal_adj': fractal_adj
        })

        # Keep history manageable
        if len(self.ml_position_history) > 100:
            self.ml_position_history.pop(0)

        return enhanced_size

    def assess_ml_risk_regime(self, ml_package: MLSignalPackage) -> Dict[str, Any]:
        """Assess current risk regime using ML insights"""

        risk_assessment = {
            'overall_risk': 'medium',
            'ml_model_health': 'good',
            'consciousness_level': 'normal',
            'fractal_regime': 'stable',
            'recommended_action': 'proceed_normal'
        }

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

class EnhancedRenaissanceTradingBot(renaissance_trading_bot.RenaissanceTradingBot):
    """Enhanced Renaissance Trading Bot with ML Integration"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Initialize ML integration bridge
        self.ml_bridge = MLIntegrationBridge(config)

        # Replace components with enhanced versions
        self.signal_fusion = AdaptiveSignalFusion()
        self.risk_manager = EnhancedRiskManager()

        # Enhanced bot state
        self.ml_enabled = True
        self.performance_baseline = None
        self.ml_performance_metrics = {
            'total_trades': 0,
            'ml_enhanced_trades': 0,
            'traditional_only_trades': 0,
            'avg_ml_processing_time': 0.0,
            'ml_success_rate': 0.0
        }

        # Initialize ML bridge
        self.ml_bridge.initialize()

        self.logger.info("Enhanced Renaissance Trading Bot initialized with ML capabilities")

    async def analyze_market_enhanced(self, market_data: pd.DataFrame) -> Tuple[List[renaissance_trading_bot.TradingDecision], MLSignalPackage]:
        """Enhanced market analysis with ML integration"""

        # Get traditional signals
        traditional_signals = await self.analyze_market(market_data)

        # Get ML signals
        ml_package = await self.ml_bridge.generate_ml_signals(market_data, traditional_signals)

        return traditional_signals, ml_package

    def make_enhanced_trading_decision(self, traditional_signals: List[renaissance_trading_bot.TradingDecision],
                                       ml_package: MLSignalPackage,
                                       current_price: float = 50000.0) -> EnhancedTradingDecision:
        """Make enhanced trading decision with ML integration"""

        start_time = datetime.now()

        # Enhanced signal fusion
        strength, confidence, fusion_metadata = self.signal_fusion.fuse_signals_with_ml(
            traditional_signals, ml_package
        )

        # ML-enhanced risk assessment
        risk_assessment = self.risk_manager.assess_ml_risk_regime(ml_package)

        # Apply risk management
        if not self.risk_manager.check_risk_limits() or risk_assessment['recommended_action'] == 'fallback_mode':
            decision = EnhancedTradingDecision(
                action=renaissance_trading_bot.OrderType.HOLD,
                size=0.0,
                strength=0.0,
                confidence=0.0,
                ml_ensemble_score=ml_package.ensemble_score,
                ml_confidence=ml_package.confidence_score,
                fractal_insights=ml_package.fractal_insights,
                consciousness_score=ml_package.confidence_score,
                traditional_signal_count=len(traditional_signals),
                ml_signal_count=len(ml_package.primary_signals),
                signal_fusion_weights=self.signal_fusion.adaptive_weights.copy(),
                risk_adjusted_size=0.0,
                expected_performance=0.0,
                adaptive_confidence=confidence,
                timestamp=datetime.now(),
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                model_health_status=self.ml_bridge.get_integration_status()
            )
            return decision

        # Determine action
        min_confidence = 0.3
        min_strength = 0.1

        # Adjust thresholds based on ML confidence
        if ml_package.confidence_score > 0.7:
            min_confidence *= 0.8  # Lower threshold for high ML confidence
            min_strength *= 0.8
        elif ml_package.confidence_score < 0.4:
            min_confidence *= 1.2  # Higher threshold for low ML confidence
            min_strength *= 1.2

        if confidence < min_confidence or abs(strength) < min_strength:
            action = renaissance_trading_bot.OrderType.HOLD
            size = 0.0
            risk_adjusted_size = 0.0
        else:
            action = renaissance_trading_bot.OrderType.BUY if strength > 0 else renaissance_trading_bot.OrderType.SELL

            # Enhanced position sizing
            base_size = self.risk_manager.calculate_position_size(strength, confidence, current_price)
            risk_adjusted_size = self.risk_manager.calculate_enhanced_position_size(
                strength, confidence, current_price, ml_package
            )
            size = risk_adjusted_size

        # Calculate expected performance
        expected_performance = self._calculate_expected_performance(
            strength, confidence, ml_package, risk_assessment
        )

        # Adaptive confidence adjustment
        adaptive_confidence = self._calculate_adaptive_confidence(
            confidence, ml_package, risk_assessment
        )

        # Create enhanced decision
        decision = EnhancedTradingDecision(
            action=action,
            size=size,
            strength=strength,
            confidence=confidence,
            ml_ensemble_score=ml_package.ensemble_score,
            ml_confidence=ml_package.confidence_score,
            fractal_insights=ml_package.fractal_insights,
            consciousness_score=ml_package.confidence_score,
            traditional_signal_count=len(traditional_signals),
            ml_signal_count=len(ml_package.primary_signals),
            signal_fusion_weights=self.signal_fusion.adaptive_weights.copy(),
            risk_adjusted_size=risk_adjusted_size,
            expected_performance=expected_performance,
            adaptive_confidence=adaptive_confidence,
            timestamp=datetime.now(),
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            model_health_status=self.ml_bridge.get_integration_status()
        )

        # Update performance metrics
        self._update_performance_metrics(decision, ml_package)

        self.last_decision = decision
        return decision

    def _calculate_expected_performance(self, strength: float, confidence: float, 
                                     ml_package: MLSignalPackage, 
                                     risk_assessment: Dict[str, Any]) -> float:
        """Calculate expected performance of the trade"""

        # Base expected return from signal strength and confidence
        base_expectation = strength * confidence

        # ML enhancement factor
        ml_factor = 1.0 + (ml_package.confidence_score - 0.5) * 0.5

        # Fractal regime adjustment
        regime = ml_package.fractal_insights.get('regime_detection', 'normal')
        regime_factor = {
            'trending': 1.2,
            'mean_reverting': 0.9,
            'chaotic': 0.6,
            'normal': 1.0
        }.get(regime, 1.0)

        # Risk assessment factor
        risk_factor = {
            'low': 1.1,
            'medium': 1.0,
            'high': 0.8
        }.get(risk_assessment.get('overall_risk', 'medium'), 1.0)

        expected_performance = base_expectation * ml_factor * regime_factor * risk_factor

        return np.clip(expected_performance, -0.1, 0.1)  # Cap at ±10%

    def _calculate_adaptive_confidence(self, base_confidence: float, 
                                     ml_package: MLSignalPackage, 
                                     risk_assessment: Dict[str, Any]) -> float:
        """Calculate adaptive confidence score"""

        # Start with base confidence
        adaptive_conf = base_confidence

        # Boost confidence if ML and traditional signals agree
        if abs(ml_package.ensemble_score) > 0.1:  # ML has opinion
            traditional_direction = 1 if base_confidence > 0.5 else -1
            ml_direction = 1 if ml_package.ensemble_score > 0 else -1

            if traditional_direction == ml_direction:
                adaptive_conf *= 1.15  # 15% boost for agreement
            else:
                adaptive_conf *= 0.9   # 10% reduction for disagreement

        # Consciousness level adjustment
        consciousness = ml_package.confidence_score
        if consciousness > 0.8:
            adaptive_conf *= 1.1
        elif consciousness < 0.4:
            adaptive_conf *= 0.85

        # Model health adjustment
        if risk_assessment.get('ml_model_health', 'good') == 'degraded':
            adaptive_conf *= 0.9
        elif risk_assessment.get('ml_model_health', 'good') == 'slow':
            adaptive_conf *= 0.95

        return np.clip(adaptive_conf, 0.0, 1.0)

    def _update_performance_metrics(self, decision: EnhancedTradingDecision, ml_package: MLSignalPackage):
        """Update performance tracking metrics"""

        self.ml_performance_metrics['total_trades'] += 1

        if ml_package.ensemble_score != 0.0 or len(ml_package.primary_signals) > 0:
            self.ml_performance_metrics['ml_enhanced_trades'] += 1
        else:
            self.ml_performance_metrics['traditional_only_trades'] += 1

        # Update average processing time
        current_time = ml_package.processing_time_ms
        total_trades = self.ml_performance_metrics['total_trades']
        current_avg = self.ml_performance_metrics['avg_ml_processing_time']

        self.ml_performance_metrics['avg_ml_processing_time'] = (
            (current_avg * (total_trades - 1) + current_time) / total_trades
        )

    async def run_enhanced_trading_cycle(self, market_data: pd.DataFrame, 
                                       current_price: float = 50000.0) -> EnhancedTradingDecision:
        """Run complete enhanced trading cycle with ML integration"""

        self.logger.info("Starting enhanced trading cycle with ML integration...")

        # Enhanced market analysis
        traditional_signals, ml_package = await self.analyze_market_enhanced(market_data)

        # Make enhanced decision
        decision = self.make_enhanced_trading_decision(
            traditional_signals, ml_package, current_price
        )

        self.logger.info(
            f"Enhanced trading decision: {decision.action.value} "
            f"(strength: {decision.strength:.3f}, confidence: {decision.confidence:.3f}, "
            f"ML ensemble: {decision.ml_ensemble_score:.3f}, "
            f"consciousness: {decision.consciousness_score:.3f})"
        )

        return decision

    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced bot status"""

        base_status = self.get_status()

        enhanced_status = {
            **base_status,
            'ml_integration': self.ml_bridge.get_integration_status(),
            'ml_performance_metrics': self.ml_performance_metrics.copy(),
            'adaptive_weights': self.signal_fusion.adaptive_weights.copy(),
            'signal_performance': self.signal_fusion._get_performance_summary(),
            'enhancement_level': 'full_ml_integration',
            'last_enhanced_decision': asdict(self.last_decision) if isinstance(self.last_decision, EnhancedTradingDecision) else None
        }

        return enhanced_status

    def enable_ml_integration(self, enabled: bool = True):
        """Enable or disable ML integration"""
        self.ml_enabled = enabled
        self.ml_bridge.ml_enabled = enabled
        self.logger.info(f"ML integration {'enabled' if enabled else 'disabled'}")

    def update_performance_feedback(self, signal_type: str, was_correct: bool, return_achieved: float):
        """Update performance feedback for adaptive learning"""
        self.signal_fusion.update_signal_performance(signal_type, was_correct, return_achieved)
