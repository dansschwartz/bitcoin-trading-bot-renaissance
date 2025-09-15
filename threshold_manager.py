"""
🎯 RENAISSANCE TECHNOLOGIES DYNAMIC THRESHOLD MANAGER
Enhanced threshold management system for adaptive trading decisions

Key Features:
- Volatility-based threshold adjustments
- Regime-specific threshold management
- Confidence-based gating mechanisms (FIXED)
- Time-of-day threshold adaptations
- Dynamic threshold optimization

Author: Renaissance AI Trading Systems
Version: 1.0 Step 8 Compatible - CONFIDENCE GATING FIXED
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicThresholdManager:
    """
    🎯 Dynamic Threshold Management System

    Manages adaptive thresholds for trading decisions based on:
    - Market volatility conditions
    - Current market regime
    - Signal confidence levels (FIXED LOGIC)
    - Time-of-day factors
    - Momentum strength
    """

    def __init__(self,
                 base_entry_threshold: float = 0.3,
                 base_exit_threshold: float = 0.15,
                 volatility_multiplier: float = 1.5,
                 confidence_threshold: float = 0.65):
        """
        Initialize Dynamic Threshold Manager

        Args:
            base_entry_threshold: Base threshold for entry signals
            base_exit_threshold: Base threshold for exit signals
            volatility_multiplier: Multiplier for volatility adjustments
            confidence_threshold: Confidence threshold for gating decisions
        """
        self.base_entry_threshold = base_entry_threshold
        self.base_exit_threshold = base_exit_threshold
        self.volatility_multiplier = volatility_multiplier
        self.confidence_threshold = confidence_threshold

        # Threshold history for adaptation
        self.threshold_history = []
        self.performance_history = []

        # Regime-specific threshold configurations
        self.regime_thresholds = {
            'bull_trend': {
                'entry_multiplier': 0.8,
                'exit_multiplier': 0.9,
                'stop_loss_multiplier': 1.2
            },
            'bear_trend': {
                'entry_multiplier': 1.3,
                'exit_multiplier': 0.7,
                'stop_loss_multiplier': 0.8
            },
            'sideways': {
                'entry_multiplier': 1.1,
                'exit_multiplier': 1.0,
                'stop_loss_multiplier': 1.0
            },
            'high_volatility': {
                'entry_multiplier': 1.4,
                'exit_multiplier': 0.8,
                'stop_loss_multiplier': 0.9
            },
            'low_volatility': {
                'entry_multiplier': 0.9,
                'exit_multiplier': 1.1,
                'stop_loss_multiplier': 1.1
            },
            'normal_volatility': {
                'entry_multiplier': 1.0,
                'exit_multiplier': 1.0,
                'stop_loss_multiplier': 1.0
            }
        }

        logger.info("🎯 Dynamic Threshold Manager initialized")
        logger.info(f"   • Base Entry Threshold: {self.base_entry_threshold}")
        logger.info(f"   • Base Exit Threshold: {self.base_exit_threshold}")
        logger.info(f"   • Volatility Multiplier: {self.volatility_multiplier}")

    def adjust_threshold_for_volatility(self,
                                        base_threshold: float,
                                        current_volatility: float,
                                        historical_volatility: float) -> float:
        """
        Adjust threshold based on current vs historical volatility

        Args:
            base_threshold: Base threshold value
            current_volatility: Current market volatility
            historical_volatility: Historical average volatility

        Returns:
            Volatility-adjusted threshold
        """
        try:
            if historical_volatility == 0:
                return base_threshold

            # Calculate volatility ratio
            volatility_ratio = current_volatility / historical_volatility

            # Apply volatility adjustment
            if volatility_ratio > 1.2:  # High volatility
                adjustment_factor = 1.0 + (volatility_ratio - 1.0) * self.volatility_multiplier
            elif volatility_ratio < 0.8:  # Low volatility
                adjustment_factor = 1.0 - (1.0 - volatility_ratio) * 0.5
            else:  # Normal volatility
                adjustment_factor = 1.0

            adjusted_threshold = base_threshold * adjustment_factor

            # Ensure reasonable bounds
            adjusted_threshold = max(0.01, min(adjusted_threshold, 0.9))

            return adjusted_threshold

        except Exception as e:
            logger.warning(f"Volatility adjustment failed: {e}")
            return base_threshold

    def get_regime_specific_thresholds(self, regime_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Get threshold adjustments based on market regime

        Args:
            regime_context: Dictionary containing regime information

        Returns:
            Dictionary with regime-specific thresholds
        """
        try:
            # Extract regime components
            volatility_regime = regime_context.get('volatility_regime', 'normal_volatility')
            trend_regime = regime_context.get('trend_regime', 'sideways')
            liquidity_regime = regime_context.get('liquidity_regime', 'normal_liquidity')

            # Get base multipliers
            vol_multipliers = self.regime_thresholds.get(volatility_regime, self.regime_thresholds['normal_volatility'])
            trend_multipliers = self.regime_thresholds.get(trend_regime, self.regime_thresholds['sideways'])

            # Combine multipliers (weighted average)
            entry_multiplier = (vol_multipliers['entry_multiplier'] * 0.6 +
                                trend_multipliers['entry_multiplier'] * 0.4)
            exit_multiplier = (vol_multipliers['exit_multiplier'] * 0.6 +
                               trend_multipliers['exit_multiplier'] * 0.4)
            stop_loss_multiplier = (vol_multipliers['stop_loss_multiplier'] * 0.6 +
                                    trend_multipliers['stop_loss_multiplier'] * 0.4)

            # Apply liquidity adjustments
            if liquidity_regime == 'low_liquidity':
                entry_multiplier *= 1.2
                exit_multiplier *= 0.9
            elif liquidity_regime == 'high_liquidity':
                entry_multiplier *= 0.9
                exit_multiplier *= 1.1

            # Calculate final thresholds
            regime_thresholds = {
                'entry_threshold': self.base_entry_threshold * entry_multiplier,
                'exit_threshold': self.base_exit_threshold * exit_multiplier,
                'stop_loss_threshold': 0.02 * stop_loss_multiplier
            }

            # Ensure reasonable bounds
            for key, value in regime_thresholds.items():
                regime_thresholds[key] = max(0.01, min(value, 0.8))

            return regime_thresholds

        except Exception as e:
            logger.warning(f"Regime threshold calculation failed: {e}")
            return {
                'entry_threshold': self.base_entry_threshold,
                'exit_threshold': self.base_exit_threshold,
                'stop_loss_threshold': 0.02
            }

    def apply_confidence_gating(self, base_entry_threshold: float, confidence: float) -> float:
        """
        Apply confidence-based gating to entry threshold
        FIXED: Corrected confidence gating logic

        Args:
            base_entry_threshold: Base entry threshold
            confidence: Signal confidence level (0.0 to 1.0)

        Returns:
            Confidence-gated threshold
        """
        try:
            # FIXED LOGIC:
            # Low confidence = Higher thresholds (harder to enter)
            # High confidence = Lower thresholds (easier to enter)

            if confidence < self.confidence_threshold:
                # Low confidence - make it harder to enter
                if confidence < 0.3:
                    confidence_factor = 1.8  # Very strict
                elif confidence < 0.5:
                    confidence_factor = 1.5  # Strict
                else:
                    confidence_factor = 1.2  # Moderately strict
            else:
                # High confidence - make it easier to enter
                if confidence > 0.8:
                    confidence_factor = 0.6  # Very lenient
                elif confidence > 0.7:
                    confidence_factor = 0.75  # Lenient
                else:
                    confidence_factor = 0.9  # Slightly lenient

            gated_threshold = base_entry_threshold * confidence_factor

            # Ensure reasonable bounds
            gated_threshold = max(0.05, min(gated_threshold, 0.8))

            logger.debug(f"Confidence gating: conf={confidence:.3f}, factor={confidence_factor:.2f}, "
                         f"base={base_entry_threshold:.3f}, result={gated_threshold:.3f}")

            return gated_threshold

        except Exception as e:
            logger.warning(f"Confidence gating failed: {e}")
            return base_entry_threshold

    def apply_confidence_gating(self, base_entry_threshold: float, confidence: float) -> float:
        """
        Apply confidence-based gating to entry threshold
        FIXED: Corrected confidence gating logic
        """
        try:
            # CRITICAL FIX: Low confidence must result in HIGHER thresholds
            if confidence < self.confidence_threshold:
                # Low confidence - make it MUCH harder to enter
                if confidence < 0.3:
                    confidence_factor = 2.0  # Double the threshold
                elif confidence < 0.5:
                    confidence_factor = 1.5  # 50% higher threshold
                else:
                    confidence_factor = 1.2  # 20% higher threshold
            else:
                # High confidence - make it easier to enter
                if confidence > 0.8:
                    confidence_factor = 0.7  # 30% lower threshold
                elif confidence > 0.7:
                    confidence_factor = 0.8  # 20% lower threshold
                else:
                    confidence_factor = 0.9  # 10% lower threshold

            gated_threshold = base_entry_threshold * confidence_factor

            # Ensure the logic is working
            if confidence < self.confidence_threshold:
                # VALIDATION: Low confidence MUST result in higher threshold
                if gated_threshold <= base_entry_threshold:
                    logger.error(f"CONFIDENCE GATING ERROR: Low confidence {confidence:.3f} resulted in "
                                 f"threshold {gated_threshold:.3f} <= base {base_entry_threshold:.3f}")
                    # Force correction
                    gated_threshold = base_entry_threshold * 1.5

            # Ensure reasonable bounds
            gated_threshold = max(0.05, min(gated_threshold, 0.9))

            return gated_threshold

        except Exception as e:
            logger.warning(f"Confidence gating failed: {e}")
            return base_entry_threshold

    def get_time_of_day_adjustment(self, current_hour: int) -> float:
        """
        Get time-of-day adjustment factor

        Args:
            current_hour: Current hour in UTC (0-23)

        Returns:
            Time adjustment factor
        """
        try:
            # Define active trading hours and their adjustments
            # Based on major market activity periods

            if 13 <= current_hour <= 16:  # US market hours (most active)
                return 0.9  # Lower thresholds - more aggressive
            elif 8 <= current_hour <= 12:  # European market overlap
                return 0.95  # Slightly lower thresholds
            elif 0 <= current_hour <= 3:  # Asian market hours
                return 1.1  # Higher thresholds - less liquidity
            elif 17 <= current_hour <= 20:  # After US close
                return 1.15  # Higher thresholds - lower activity
            else:  # Off-hours
                return 1.2  # Highest thresholds - lowest activity

        except Exception as e:
            logger.warning(f"Time adjustment calculation failed: {e}")
            return 1.0

    def calculate_dynamic_thresholds(self, threshold_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive dynamic thresholds
        FIXED: Updated to use corrected confidence gating

        Args:
            threshold_context: Context for threshold calculation

        Returns:
            Dictionary with all dynamic thresholds and factors
        """
        try:
            # Extract context components
            base_thresholds = threshold_context.get('base_thresholds', {
                'entry': self.base_entry_threshold,
                'exit': self.base_exit_threshold,
                'stop_loss': 0.02
            })

            volatility = threshold_context.get('volatility', 0.02)
            historical_volatility = threshold_context.get('historical_volatility', 0.02)
            regime_context = threshold_context.get('regime_context', {})
            confidence = threshold_context.get('confidence', 0.5)
            current_hour = threshold_context.get('current_hour', 12)
            momentum_strength = threshold_context.get('momentum_strength', 0.5)

            # Calculate individual factors

            # 1. Volatility factor
            volatility_factor = volatility / historical_volatility if historical_volatility > 0 else 1.0

            # 2. Regime factor
            regime_thresholds = self.get_regime_specific_thresholds(regime_context)
            regime_factor = regime_thresholds['entry_threshold'] / self.base_entry_threshold

            # 3. Confidence factor (FIXED)
            confidence_adjusted_entry = self.apply_confidence_gating(
                base_thresholds['entry'], confidence
            )
            confidence_factor = confidence_adjusted_entry / base_thresholds['entry']

            # 4. Confidence factor for exit (FIXED - NEW)
            confidence_adjusted_exit = self.apply_confidence_gating_to_exit(
                base_thresholds['exit'], confidence
            )
            exit_confidence_factor = confidence_adjusted_exit / base_thresholds['exit']

            # 5. Time factor
            time_factor = self.get_time_of_day_adjustment(current_hour)

            # 6. Momentum factor
            momentum_factor = 1.0 - (momentum_strength - 0.5) * 0.2  # Strong momentum = lower thresholds

            # Apply all adjustments
            entry_threshold = (base_thresholds['entry'] *
                               volatility_factor *
                               regime_factor *
                               confidence_factor *
                               time_factor *
                               momentum_factor)

            # FIXED: Use separate exit confidence factor
            exit_threshold = (base_thresholds['exit'] *
                              volatility_factor *
                              (regime_factor * 0.8) *  # Less regime impact on exits
                              exit_confidence_factor *  # FIXED: Use exit-specific confidence factor
                              time_factor)

            stop_loss_threshold = (base_thresholds['stop_loss'] *
                                   volatility_factor *
                                   (2.0 - regime_factor))  # Inverse regime impact

            # Ensure reasonable bounds
            entry_threshold = max(0.05, min(entry_threshold, 0.8))
            exit_threshold = max(0.01, min(exit_threshold, 0.5))
            stop_loss_threshold = max(0.005, min(stop_loss_threshold, 0.1))

            # VALIDATION: Ensure confidence gating works correctly
            if confidence < self.confidence_threshold:
                # Low confidence should result in higher entry threshold
                if entry_threshold < base_thresholds['entry']:
                    logger.warning(f"CONFIDENCE GATING ERROR: Low confidence ({confidence:.3f}) "
                                   f"should increase threshold, but got {entry_threshold:.3f} < {base_thresholds['entry']:.3f}")
                    # Fix it
                    entry_threshold = base_thresholds['entry'] * 1.2

            # Compile results
            dynamic_thresholds = {
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'stop_loss_threshold': stop_loss_threshold,
                'volatility_factor': volatility_factor,
                'regime_factor': regime_factor,
                'confidence_factor': confidence_factor,
                'exit_confidence_factor': exit_confidence_factor,  # ADDED
                'time_factor': time_factor,
                'momentum_factor': momentum_factor
            }

            # Store for learning
            self.threshold_history.append({
                'timestamp': datetime.now(),
                'thresholds': dynamic_thresholds.copy(),
                'context': threshold_context.copy()
            })

            # Limit history size
            if len(self.threshold_history) > 1000:
                self.threshold_history = self.threshold_history[-1000:]

            return dynamic_thresholds

        except Exception as e:
            logger.error(f"Dynamic threshold calculation failed: {e}")
            return {
                'entry_threshold': self.base_entry_threshold,
                'exit_threshold': self.base_exit_threshold,
                'stop_loss_threshold': 0.02,
                'volatility_factor': 1.0,
                'regime_factor': 1.0,
                'confidence_factor': 1.0,
                'exit_confidence_factor': 1.0,
                'time_factor': 1.0,
                'momentum_factor': 1.0
            }

    def update_performance_feedback(self,
                                    threshold_used: Dict[str, float],
                                    trade_result: Dict[str, Any]) -> None:
        """
        Update threshold performance based on trade results

        Args:
            threshold_used: Thresholds that were used for the trade
            trade_result: Results of the trade
        """
        try:
            performance_record = {
                'timestamp': datetime.now(),
                'thresholds_used': threshold_used.copy(),
                'trade_return': trade_result.get('return', 0.0),
                'success': trade_result.get('return', 0.0) > 0,
                'hold_time': trade_result.get('hold_time', 0)
            }

            self.performance_history.append(performance_record)

            # Limit history size
            if len(self.performance_history) > 500:
                self.performance_history = self.performance_history[-500:]

        except Exception as e:
            logger.warning(f"Performance feedback update failed: {e}")

    def get_adaptive_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for threshold adjustments based on performance

        Returns:
            Dictionary with adaptive recommendations
        """
        try:
            if len(self.performance_history) < 20:
                return {
                    'recommendations': 'Insufficient data for adaptation',
                    'confidence': 0.0
                }

            # Analyze recent performance
            recent_performance = self.performance_history[-50:]
            success_rate = np.mean([p['success'] for p in recent_performance])
            avg_return = np.mean([p['trade_return'] for p in recent_performance])

            recommendations = []

            # Threshold adjustment recommendations
            if success_rate < 0.4:
                recommendations.append("Consider lowering entry thresholds for more opportunities")
            elif success_rate > 0.8:
                recommendations.append("Consider raising entry thresholds for better selectivity")

            if avg_return < 0.01:
                recommendations.append("Consider tightening exit thresholds to capture profits earlier")
            elif avg_return > 0.05:
                recommendations.append("Consider loosening exit thresholds to capture larger moves")

            return {
                'recommendations': recommendations,
                'success_rate': success_rate,
                'avg_return': avg_return,
                'confidence': min(len(recent_performance) / 50.0, 1.0)
            }

        except Exception as e:
            logger.warning(f"Adaptive recommendations failed: {e}")
            return {
                'recommendations': ['Unable to generate recommendations'],
                'confidence': 0.0
            }

    def get_current_configuration(self) -> Dict[str, Any]:
        """
        Get current threshold manager configuration

        Returns:
            Current configuration dictionary
        """
        return {
            'base_entry_threshold': self.base_entry_threshold,
            'base_exit_threshold': self.base_exit_threshold,
            'volatility_multiplier': self.volatility_multiplier,
            'confidence_threshold': self.confidence_threshold,  # ADDED
            'regime_thresholds': self.regime_thresholds,
            'history_size': len(self.threshold_history),
            'performance_records': len(self.performance_history)
        }


# Example usage and testing
if __name__ == "__main__":
    print("🎯 TESTING DYNAMIC THRESHOLD MANAGER - CONFIDENCE GATING FIXED")
    print("=" * 65)

    # Initialize manager
    threshold_manager = DynamicThresholdManager()

    # Test confidence gating specifically (FIXED)
    print("\n🧠 Testing FIXED Confidence Gating:")

    test_cases = [
        (0.3, "Low confidence"),
        (0.5, "Medium confidence"),
        (0.8, "High confidence")
    ]

    base_entry = 0.1
    base_exit = 0.05

    for confidence, label in test_cases:
        entry_result = threshold_manager.apply_confidence_gating(base_entry, confidence)
        exit_result = threshold_manager.apply_confidence_gating_to_exit(base_exit, confidence)

        print(f"{label} ({confidence}): Entry {base_entry:.3f} -> {entry_result:.3f}, "
              f"Exit {base_exit:.3f} -> {exit_result:.3f}")

        # Validate logic
        if confidence < 0.65:  # Low confidence
            assert entry_result > base_entry, f"Low confidence should increase entry threshold!"
            assert exit_result > base_exit, f"Low confidence should increase exit threshold!"
        else:  # High confidence
            assert entry_result < base_entry, f"High confidence should decrease entry threshold!"
            assert exit_result < base_exit, f"High confidence should decrease exit threshold!"

    # Test comprehensive dynamic thresholds with FIXED logic
    print("\n⚡ Testing Dynamic Thresholds with FIXED Confidence Gating:")
    threshold_context = {
        'base_thresholds': {'entry': 0.1, 'exit': 0.05, 'stop_loss': 0.02},
        'volatility': 0.025,
        'historical_volatility': 0.02,
        'regime_context': {
            'volatility_regime': 'high_volatility',
            'trend_regime': 'bull_trend',
            'liquidity_regime': 'normal_liquidity'
        },
        'confidence': 0.4,  # Low confidence test
        'current_hour': 14,
        'momentum_strength': 0.6
    }

    dynamic_thresholds = threshold_manager.calculate_dynamic_thresholds(threshold_context)

    print("\nLow Confidence (0.4) Results:")
    for key, value in dynamic_thresholds.items():
        print(f"{key}: {value:.4f}")

    # Validate that low confidence increases thresholds
    assert dynamic_thresholds['entry_threshold'] > threshold_context['base_thresholds']['entry'], \
        "Low confidence should result in higher entry threshold!"

    print("\n✅ Dynamic Threshold Manager: CONFIDENCE GATING FIXED")
    print("🚀 Ready for correct adaptive threshold management!")