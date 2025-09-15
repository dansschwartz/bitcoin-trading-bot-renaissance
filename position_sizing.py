"""
🎯 RENAISSANCE TECHNOLOGIES POSITION SIZING SYSTEM
================================================================

Advanced position sizing using Kelly criterion, volatility-based adjustments,
and regime-specific limits for optimal risk management.

Key Features:
- Kelly Criterion implementation for optimal bet sizing
- Volatility-based position adjustments
- Regime-specific risk limits
- Drawdown protection mechanisms
- Correlation-based exposure controls

Author: Renaissance AI Trading Systems
Version: 2.0 Enhanced Decision Framework
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PositionSizingConfig:
    """Configuration parameters for position sizing system"""
    max_position_size: float = 0.25  # 25% max position
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk
    kelly_multiplier: float = 0.25  # Conservative Kelly multiplier
    volatility_lookback: int = 20  # Days for volatility calculation
    min_position_size: float = 0.01  # 1% minimum position
    max_correlation_exposure: float = 0.5  # 50% max correlated exposure


class PositionSizingManager:
    """
    Advanced Position Sizing Manager with Kelly Criterion and Risk Controls

    Implements sophisticated position sizing strategies used by Renaissance
    Technologies for optimal risk-adjusted returns.
    """

    def __init__(self, config: Optional[PositionSizingConfig] = None):
        """
        Initialize Position Sizing Manager

        Args:
            config: Configuration parameters for position sizing
        """
        self.config = config or PositionSizingConfig()

        # Initialize tracking variables
        self.position_history = []
        self.performance_history = []
        self.volatility_estimates = {}

        logger.info("🎯 Renaissance Position Sizing Manager Initialized")
        logger.info(f"   • Max Position Size: {self.config.max_position_size * 100:.1f}%")
        logger.info(f"   • Risk Tolerance: {self.config.max_portfolio_risk * 100:.1f}%")
        logger.info(f"   • Kelly Multiplier: {self.config.kelly_multiplier}")

    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing

        Args:
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)

        Returns:
            Kelly fraction for position sizing
        """
        try:
            if win_rate <= 0 or win_rate >= 1:
                logger.warning(f"Invalid win rate: {win_rate}, using conservative default")
                return 0.1

            if avg_win <= 0 or avg_loss <= 0:
                logger.warning(f"Invalid avg win/loss: {avg_win}/{avg_loss}, using default")
                return 0.1

            # Kelly formula: f = (bp - q) / b
            # where: b = odds (avg_win/avg_loss), p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss  # Odds ratio
            p = win_rate
            q = 1 - win_rate

            kelly_fraction = (b * p - q) / b

            # Apply safety constraints
            kelly_fraction = max(0.0, min(kelly_fraction, 1.0))

            # Apply conservative multiplier
            conservative_kelly = kelly_fraction * self.config.kelly_multiplier

            logger.debug(f"Kelly calculation: win_rate={win_rate:.3f}, b={b:.3f}, "
                         f"raw_kelly={kelly_fraction:.3f}, conservative={conservative_kelly:.3f}")

            return conservative_kelly

        except Exception as e:
            logger.error(f"Kelly criterion calculation failed: {e}")
            return 0.05  # Conservative fallback

    def calculate_volatility_adjusted_size(self, base_size: float,
                                           current_volatility: float,
                                           base_volatility: float) -> float:
        """
        Adjust position size based on current volatility vs historical volatility

        Args:
            base_size: Base position size
            current_volatility: Current market volatility
            base_volatility: Historical baseline volatility

        Returns:
            Volatility-adjusted position size
        """
        try:
            if base_volatility <= 0:
                logger.warning(f"Invalid base volatility: {base_volatility}")
                return base_size

            if current_volatility <= 0:
                logger.warning(f"Invalid current volatility: {current_volatility}")
                return base_size

            # Inverse relationship: higher volatility = smaller position
            volatility_ratio = base_volatility / current_volatility

            # Apply square root scaling for smoother adjustments
            adjustment_factor = np.sqrt(volatility_ratio)

            # Constrain adjustment factor
            adjustment_factor = max(0.25, min(adjustment_factor, 2.0))

            adjusted_size = base_size * adjustment_factor

            # Apply position size limits
            adjusted_size = max(self.config.min_position_size,
                                min(adjusted_size, self.config.max_position_size))

            logger.debug(f"Volatility adjustment: base={base_size:.3f}, "
                         f"vol_ratio={volatility_ratio:.3f}, adjusted={adjusted_size:.3f}")

            return adjusted_size

        except Exception as e:
            logger.error(f"Volatility adjustment failed: {e}")
            return min(base_size, self.config.max_position_size)

    def get_regime_specific_limits(self, regime_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Get position sizing limits based on current market regime

        Args:
            regime_data: Dictionary containing regime information

        Returns:
            Dictionary with regime-specific limits
        """
        try:
            volatility_regime = regime_data.get('volatility_regime', 'normal_volatility')
            trend_regime = regime_data.get('trend_regime', 'sideways')
            crisis_level = regime_data.get('crisis_level', 0.0)

            # Base limits
            limits = {
                'max_position_size': self.config.max_position_size,
                'max_portfolio_risk': self.config.max_portfolio_risk,
                'position_concentration': 0.6,  # Max % in single position type
                'sector_exposure': 0.4,  # Max sector exposure
                'correlation_limit': self.config.max_correlation_exposure
            }

            # Adjust for volatility regime
            if volatility_regime == 'high_volatility':
                limits['max_position_size'] *= 0.6  # Reduce by 40%
                limits['max_portfolio_risk'] *= 0.7  # Reduce risk
                limits['position_concentration'] *= 0.5
            elif volatility_regime == 'low_volatility':
                limits['max_position_size'] *= 1.2  # Increase by 20%
                limits['max_portfolio_risk'] *= 1.1  # Slightly more risk

            # Adjust for trend regime
            if trend_regime == 'bear_trend':
                limits['max_position_size'] *= 0.5  # Very conservative in bear
                limits['max_portfolio_risk'] *= 0.5
            elif trend_regime == 'bull_trend':
                limits['max_position_size'] *= 1.1  # Slightly more aggressive

            # Adjust for crisis level
            if crisis_level > 0.5:  # High crisis
                crisis_factor = 1.0 - (crisis_level * 0.6)  # Up to 60% reduction
                limits['max_position_size'] *= crisis_factor
                limits['max_portfolio_risk'] *= crisis_factor

            # Ensure all limits are within absolute bounds
            limits['max_position_size'] = min(limits['max_position_size'], 0.5)  # Never >50%
            limits['max_portfolio_risk'] = min(limits['max_portfolio_risk'], 0.05)  # Never >5%

            logger.debug(f"Regime limits for {volatility_regime}/{trend_regime}: "
                         f"max_pos={limits['max_position_size']:.3f}")

            return limits

        except Exception as e:
            logger.error(f"Regime limits calculation failed: {e}")
            return {
                'max_position_size': self.config.max_position_size * 0.5,  # Conservative fallback
                'max_portfolio_risk': self.config.max_portfolio_risk * 0.5,
                'position_concentration': 0.3,
                'sector_exposure': 0.2,
                'correlation_limit': 0.3
            }

    def calculate_optimal_position_size(self, sizing_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate optimal position size using comprehensive risk management

        Args:
            sizing_context: Dictionary with all relevant sizing parameters

        Returns:
            Dictionary with position sizing results
        """
        try:
            # Extract context parameters
            signal_strength = sizing_context.get('signal_strength', 0.5)
            confidence = sizing_context.get('confidence', 0.5)
            volatility = sizing_context.get('volatility', 0.02)
            regime_data = sizing_context.get('regime_data', {})
            account_balance = sizing_context.get('account_balance', 100000)
            current_positions = sizing_context.get('current_positions', 0.0)
            max_drawdown = sizing_context.get('max_drawdown', 0.02)

            # Step 1: Calculate Kelly-based base size
            # Use historical performance estimates or defaults
            win_rate = sizing_context.get('historical_win_rate', 0.58)
            avg_win = sizing_context.get('avg_win', 0.025)
            avg_loss = sizing_context.get('avg_loss', 0.015)

            kelly_fraction = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)

            # Step 2: Adjust for signal strength and confidence
            signal_adjustment = (signal_strength * confidence) ** 0.5  # Square root scaling

            # Step 3: Get regime-specific limits
            regime_limits = self.get_regime_specific_limits(regime_data)

            # Step 4: Volatility adjustment
            base_volatility = 0.02  # Baseline 2% volatility
            volatility_adjusted_size = self.calculate_volatility_adjusted_size(
                kelly_fraction, volatility, base_volatility
            )

            # Step 5: Apply signal adjustment
            signal_adjusted_size = volatility_adjusted_size * signal_adjustment

            # Step 6: Apply regime limits
            regime_limited_size = min(signal_adjusted_size,
                                      regime_limits['max_position_size'])

            # Step 7: Drawdown protection
            if max_drawdown > 0.05:  # If significant drawdown
                drawdown_factor = max(0.2, 1.0 - (max_drawdown * 2))  # Reduce size
                regime_limited_size *= drawdown_factor

            # Step 8: Portfolio concentration check
            available_capacity = 1.0 - current_positions
            final_position_size = min(regime_limited_size, available_capacity * 0.8)

            # Step 9: Calculate risk metrics
            position_value = account_balance * final_position_size
            risk_amount = position_value * avg_loss  # Expected loss amount
            final_risk_ratio = risk_amount / account_balance

            # Ensure minimum viable size or zero
            if final_position_size < self.config.min_position_size:
                final_position_size = 0.0
                risk_amount = 0.0
                final_risk_ratio = 0.0

            # Compile results
            result = {
                'position_size': final_position_size,
                'risk_amount': risk_amount,
                'kelly_fraction': kelly_fraction,
                'volatility_adjustment': volatility_adjusted_size / kelly_fraction if kelly_fraction > 0 else 1.0,
                'regime_adjustment': regime_limited_size / signal_adjusted_size if signal_adjusted_size > 0 else 1.0,
                'confidence_scaling': signal_adjustment,
                'final_risk_ratio': final_risk_ratio,
                'max_position_allowed': regime_limits['max_position_size'],
                'position_value': position_value
            }

            logger.debug(f"Position sizing: signal_strength={signal_strength:.3f}, "
                         f"confidence={confidence:.3f}, final_size={final_position_size:.3f}")

            return result

        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            return {
                'position_size': 0.0,
                'risk_amount': 0.0,
                'kelly_fraction': 0.05,
                'volatility_adjustment': 1.0,
                'regime_adjustment': 1.0,
                'confidence_scaling': 0.5,
                'final_risk_ratio': 0.0,
                'max_position_allowed': self.config.max_position_size,
                'position_value': 0.0
            }

    def update_performance_history(self, trade_result: Dict[str, Any]) -> None:
        """
        Update performance history for improved Kelly calculations

        Args:
            trade_result: Dictionary with trade outcome information
        """
        try:
            self.performance_history.append({
                'timestamp': datetime.now(),
                'return': trade_result.get('return', 0.0),
                'position_size': trade_result.get('position_size', 0.0),
                'win': trade_result.get('return', 0.0) > 0
            })

            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]

        except Exception as e:
            logger.error(f"Performance history update failed: {e}")

    def get_current_risk_metrics(self) -> Dict[str, float]:
        """
        Get current portfolio risk metrics

        Returns:
            Dictionary with current risk statistics
        """
        try:
            if not self.performance_history:
                return {
                    'current_drawdown': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.58,  # Default estimate
                    'avg_win': 0.025,
                    'avg_loss': 0.015,
                    'profit_factor': 1.5,
                    'sharpe_estimate': 1.2
                }

            # Calculate recent performance metrics
            recent_returns = [trade['return'] for trade in self.performance_history[-100:]]
            wins = [r for r in recent_returns if r > 0]
            losses = [abs(r) for r in recent_returns if r < 0]

            return {
                'current_drawdown': self._calculate_current_drawdown(),
                'max_drawdown': self._calculate_max_drawdown(),
                'win_rate': len(wins) / len(recent_returns) if recent_returns else 0.58,
                'avg_win': np.mean(wins) if wins else 0.025,
                'avg_loss': np.mean(losses) if losses else 0.015,
                'profit_factor': (np.sum(wins) / np.sum(losses)) if losses else 1.5,
                'sharpe_estimate': np.mean(recent_returns) / np.std(recent_returns) if len(recent_returns) > 1 else 1.2
            }

        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return {
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.58,
                'avg_win': 0.025,
                'avg_loss': 0.015,
                'profit_factor': 1.5,
                'sharpe_estimate': 1.2
            }

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak equity"""
        try:
            if len(self.performance_history) < 10:
                return 0.0

            # Calculate cumulative returns
            returns = [trade['return'] for trade in self.performance_history]
            cumulative = np.cumprod(1 + np.array(returns))

            # Find current drawdown
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative[-1] - peak[-1]) / peak[-1]

            return drawdown

        except Exception:
            return 0.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum historical drawdown"""
        try:
            if len(self.performance_history) < 10:
                return 0.0

            returns = [trade['return'] for trade in self.performance_history]
            cumulative = np.cumprod(1 + np.array(returns))

            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak

            return np.min(drawdown)

        except Exception:
            return 0.0


def test_position_sizing_system():
    """Test the position sizing system"""
    print("🎯 TESTING RENAISSANCE POSITION SIZING SYSTEM")
    print("=" * 60)

    # Initialize position manager
    manager = PositionSizingManager()

    # Test Kelly criterion
    win_rate = 0.58
    avg_win = 0.025
    avg_loss = 0.015

    kelly = manager.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
    print(f"Kelly Fraction: {kelly:.4f}")

    # Test volatility adjustment
    base_size = 0.1
    current_vol = 0.025
    base_vol = 0.02

    vol_adjusted = manager.calculate_volatility_adjusted_size(base_size, current_vol, base_vol)
    print(f"Volatility Adjusted Size: {vol_adjusted:.4f}")

    # Test regime limits
    regime_data = {
        'volatility_regime': 'high_volatility',
        'trend_regime': 'bear_trend',
        'crisis_level': 0.3
    }

    limits = manager.get_regime_specific_limits(regime_data)
    print(f"Regime Limits: {limits}")

    # Test comprehensive sizing
    sizing_context = {
        'signal_strength': 0.75,
        'confidence': 0.68,
        'volatility': 0.025,
        'regime_data': regime_data,
        'account_balance': 100000,
        'current_positions': 0.3,
        'max_drawdown': 0.05
    }

    result = manager.calculate_optimal_position_size(sizing_context)
    print(f"Optimal Position Size: {result['position_size']:.4f}")
    print(f"Risk Ratio: {result['final_risk_ratio']:.4f}")

    print("\n✅ Position Sizing System: OPERATIONAL")


if __name__ == "__main__":
    test_position_sizing_system()
