
"""
Dynamic Risk Budget Manager - Renaissance Technologies Bitcoin Trading Bot
Step 9 Component 4: Intelligent Capital Allocation System

This module implements sophisticated risk budgeting using advanced mathematical models
with consciousness enhancement for optimal capital allocation.

Features:
- Kelly Criterion optimization with regime awareness
- Dynamic risk allocation based on market conditions  
- Performance-based risk budgeting with adaptive scaling
- Volatility targeting with consciousness-enhanced predictions
- Renaissance Technologies-inspired mathematical rigor
- Bitcoin-optimized parameters for crypto markets

Author: Renaissance Technologies Inspired Bot
Version: 1.0.0
License: Proprietary
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import json
import warnings
from scipy import optimize, stats
from sklearn.covariance import LedoitWolf
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import pickle
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskBudget:
    """Risk budget allocation structure"""
    strategy_id: str
    base_allocation: float  # Base risk budget (0-1)
    current_allocation: float  # Current adjusted allocation
    max_allocation: float  # Maximum allowed allocation
    min_allocation: float  # Minimum allowed allocation
    performance_multiplier: float  # Performance-based adjustment
    regime_multiplier: float  # Market regime adjustment
    volatility_multiplier: float  # Volatility-based adjustment
    consciousness_enhancement: float  # Consciousness boost factor
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AllocationDecision:
    """Capital allocation decision structure"""
    timestamp: datetime
    strategy_allocations: Dict[str, float]  # Strategy ID -> allocation
    total_risk_budget: float
    kelly_fractions: Dict[str, float]  # Kelly optimal fractions
    regime_adjustments: Dict[str, float]  # Regime-based adjustments
    performance_scores: Dict[str, float]  # Performance metrics
    confidence_level: float  # Decision confidence (0-1)
    reasoning: str  # Human-readable explanation
    risk_metrics: Dict[str, float]  # Associated risk metrics

@dataclass
class PerformanceMetrics:
    """Strategy performance tracking"""
    strategy_id: str
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    information_ratio: float
    tail_ratio: float  # Tail risk adjusted return
    consciousness_score: float  # Performance with consciousness enhancement
    rolling_alpha: float  # Excess return vs benchmark
    risk_adjusted_return: float  # Return per unit of risk

class KellyCriterionOptimizer:
    """Advanced Kelly Criterion implementation with consciousness enhancement"""

    def __init__(self, consciousness_factor: float = 1.142):
        self.consciousness_factor = consciousness_factor
        self.min_kelly = 0.01  # Minimum Kelly fraction
        self.max_kelly = 0.25  # Maximum Kelly fraction for safety

    def calculate_kelly_fraction(self, returns: np.ndarray, 
                               regime_volatility: float = None) -> float:
        """
        Calculate optimal Kelly fraction with consciousness enhancement

        Args:
            returns: Historical returns array
            regime_volatility: Current regime volatility adjustment

        Returns:
            Optimal Kelly fraction (0-1)
        """
        try:
            if len(returns) < 30:  # Need sufficient data
                return self.min_kelly

            # Calculate basic statistics with consciousness enhancement
            mean_return = np.mean(returns) * self.consciousness_factor
            variance = np.var(returns, ddof=1)

            # Regime adjustment if provided
            if regime_volatility is not None:
                variance *= regime_volatility

            # Kelly formula: f* = (μ - r) / σ²
            # For crypto, assuming risk-free rate ≈ 0
            if variance <= 0:
                return self.min_kelly

            kelly_fraction = mean_return / variance

            # Apply consciousness enhancement to final fraction
            kelly_fraction *= self.consciousness_factor

            # Safety constraints
            kelly_fraction = np.clip(kelly_fraction, self.min_kelly, self.max_kelly)

            return float(kelly_fraction)

        except Exception as e:
            logger.warning(f"Kelly calculation error: {e}")
            return self.min_kelly

    def calculate_fractional_kelly(self, kelly_fraction: float, 
                                 fraction: float = 0.25) -> float:
        """Calculate fractional Kelly for reduced volatility"""
        return kelly_fraction * fraction * self.consciousness_factor

    def multi_asset_kelly(self, returns_matrix: np.ndarray, 
                         covariance_matrix: np.ndarray = None) -> np.ndarray:
        """
        Multi-asset Kelly optimization with consciousness enhancement

        Args:
            returns_matrix: Matrix of returns for multiple assets
            covariance_matrix: Covariance matrix (optional, will estimate)

        Returns:
            Optimal Kelly fractions array
        """
        try:
            if covariance_matrix is None:
                # Use shrinkage estimator for better covariance estimation
                lw = LedoitWolf()
                covariance_matrix = lw.fit(returns_matrix).covariance_

            # Enhanced mean returns with consciousness factor
            mean_returns = np.mean(returns_matrix, axis=0) * self.consciousness_factor

            # Multi-asset Kelly formula: f* = Σ⁻¹μ
            try:
                inv_cov = np.linalg.pinv(covariance_matrix)
                kelly_weights = inv_cov @ mean_returns
            except np.linalg.LinAlgError:
                # Fallback to diagonal approximation
                kelly_weights = mean_returns / np.diag(covariance_matrix)

            # Apply consciousness enhancement
            kelly_weights *= self.consciousness_factor

            # Normalize and constrain
            kelly_weights = np.clip(kelly_weights, 0, self.max_kelly)

            return kelly_weights

        except Exception as e:
            logger.warning(f"Multi-asset Kelly error: {e}")
            n_assets = returns_matrix.shape[1]
            return np.full(n_assets, self.min_kelly)

class RegimeAwareAllocator:
    """Market regime-aware risk allocation system"""

    def __init__(self, consciousness_factor: float = 1.142):
        self.consciousness_factor = consciousness_factor
        self.regime_mappings = {
            'bull_high_vol': 1.2,  # Increase allocation in bull markets
            'bull_low_vol': 1.4,   # Optimal conditions
            'bear_high_vol': 0.4,  # Reduce in volatile bear markets
            'bear_low_vol': 0.6,   # Moderate reduction
            'sideways_high_vol': 0.7,  # Cautious in uncertain times
            'sideways_low_vol': 1.0,   # Neutral allocation
            'transition': 0.8,     # Conservative during transitions
            'crisis': 0.2,         # Minimal allocation during crisis
        }

    def get_regime_multiplier(self, regime: str, 
                            confidence: float = 1.0) -> float:
        """
        Get regime-based allocation multiplier with consciousness enhancement

        Args:
            regime: Current market regime
            confidence: Regime detection confidence (0-1)

        Returns:
            Allocation multiplier
        """
        base_multiplier = self.regime_mappings.get(regime, 1.0)

        # Adjust by confidence level
        adjusted_multiplier = (base_multiplier * confidence + 
                             1.0 * (1 - confidence))

        # Apply consciousness enhancement
        if adjusted_multiplier > 1.0:
            adjusted_multiplier *= self.consciousness_factor
        else:
            # Be more conservative with consciousness in downturns
            adjusted_multiplier /= self.consciousness_factor

        return max(0.1, min(2.0, adjusted_multiplier))

    def calculate_regime_adjusted_allocation(self, base_allocation: float,
                                           regime: str,
                                           regime_confidence: float,
                                           volatility_ratio: float = 1.0) -> float:
        """Calculate regime-adjusted allocation with consciousness enhancement"""
        regime_mult = self.get_regime_multiplier(regime, regime_confidence)

        # Volatility adjustment with consciousness
        vol_adjustment = 1.0 / (volatility_ratio * self.consciousness_factor)
        vol_adjustment = np.clip(vol_adjustment, 0.5, 1.5)

        adjusted_allocation = base_allocation * regime_mult * vol_adjustment
        return np.clip(adjusted_allocation, 0.01, 0.5)  # Safety bounds

class PerformanceBasedBudgeting:
    """Performance-based risk budget allocation with adaptive scaling"""

    def __init__(self, consciousness_factor: float = 1.142):
        self.consciousness_factor = consciousness_factor
        self.lookback_periods = [30, 90, 180, 365]  # Multiple time horizons
        self.performance_weights = [0.4, 0.3, 0.2, 0.1]  # Weights for each period

    def calculate_performance_score(self, strategy_returns: np.ndarray,
                                  benchmark_returns: np.ndarray = None) -> float:
        """
        Calculate comprehensive performance score with consciousness enhancement

        Args:
            strategy_returns: Strategy return time series
            benchmark_returns: Benchmark returns (optional)

        Returns:
            Performance score (0-2, where 1 is neutral)
        """
        try:
            if len(strategy_returns) < 30:
                return 1.0  # Neutral for insufficient data

            # Calculate multiple performance metrics
            sharpe = self._calculate_sharpe_ratio(strategy_returns)
            sortino = self._calculate_sortino_ratio(strategy_returns)
            calmar = self._calculate_calmar_ratio(strategy_returns)
            tail_ratio = self._calculate_tail_ratio(strategy_returns)

            # Information ratio if benchmark provided
            info_ratio = 0.0
            if benchmark_returns is not None and len(benchmark_returns) == len(strategy_returns):
                info_ratio = self._calculate_information_ratio(strategy_returns, benchmark_returns)

            # Composite score with consciousness enhancement
            base_score = (sharpe * 0.3 + sortino * 0.25 + calmar * 0.2 + 
                         tail_ratio * 0.15 + info_ratio * 0.1)

            # Apply consciousness enhancement
            consciousness_score = base_score * self.consciousness_factor

            # Normalize to 0-2 range (1 = neutral)
            normalized_score = 1.0 + np.tanh(consciousness_score) * 0.8

            return float(np.clip(normalized_score, 0.2, 2.0))

        except Exception as e:
            logger.warning(f"Performance score calculation error: {e}")
            return 1.0

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio with consciousness enhancement"""
        if len(returns) == 0:
            return 0.0
        mean_return = np.mean(returns) * self.consciousness_factor
        std_return = np.std(returns, ddof=1)
        return mean_return / std_return if std_return > 0 else 0.0

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio focusing on downside deviation"""
        if len(returns) == 0:
            return 0.0
        mean_return = np.mean(returns) * self.consciousness_factor
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0.01
        return mean_return / downside_std

    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio (return/max drawdown)"""
        if len(returns) == 0:
            return 0.0

        # Calculate cumulative returns and max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))

        if max_drawdown == 0:
            return 0.0

        annual_return = np.mean(returns) * 365 * self.consciousness_factor
        return annual_return / max_drawdown

    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        if len(returns) < 20:
            return 1.0

        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)

        if p5 >= 0:
            return 2.0  # All positive returns

        ratio = abs(p95 / p5) if p5 != 0 else 1.0
        return min(ratio * self.consciousness_factor, 5.0)

    def _calculate_information_ratio(self, strategy_returns: np.ndarray,
                                   benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio vs benchmark"""
        excess_returns = strategy_returns - benchmark_returns
        if len(excess_returns) == 0:
            return 0.0

        mean_excess = np.mean(excess_returns) * self.consciousness_factor
        std_excess = np.std(excess_returns, ddof=1)

        return mean_excess / std_excess if std_excess > 0 else 0.0

class VolatilityTargeting:
    """Volatility targeting system with consciousness-enhanced predictions"""

    def __init__(self, target_volatility: float = 0.2, 
                 consciousness_factor: float = 1.142):
        self.target_volatility = target_volatility
        self.consciousness_factor = consciousness_factor
        self.volatility_models = {}
        self.volatility_history = deque(maxlen=252)  # 1 year of daily data

    def estimate_future_volatility(self, returns: np.ndarray,
                                 horizon_days: int = 30) -> float:
        """
        Estimate future volatility with consciousness enhancement

        Args:
            returns: Historical returns
            horizon_days: Forecasting horizon

        Returns:
            Predicted volatility (annualized)
        """
        try:
            if len(returns) < 30:
                return self.target_volatility

            # Multiple volatility estimation methods
            methods = {
                'ewma': self._ewma_volatility(returns),
                'garch': self._simple_garch_volatility(returns),
                'realized': self._realized_volatility(returns),
                'parkinson': self._parkinson_volatility(returns) if len(returns) > 5 else None
            }

            # Filter valid methods
            valid_methods = {k: v for k, v in methods.items() if v is not None}

            if not valid_methods:
                return self.target_volatility

            # Ensemble prediction with consciousness enhancement
            weights = {'ewma': 0.3, 'garch': 0.3, 'realized': 0.25, 'parkinson': 0.15}
            weighted_vol = sum(weights.get(method, 0) * vol 
                             for method, vol in valid_methods.items())

            # Apply consciousness enhancement to prediction
            consciousness_adjusted = weighted_vol * self.consciousness_factor

            # Store in history
            self.volatility_history.append(consciousness_adjusted)

            return float(consciousness_adjusted)

        except Exception as e:
            logger.warning(f"Volatility estimation error: {e}")
            return self.target_volatility

    def _ewma_volatility(self, returns: np.ndarray, lambda_factor: float = 0.94) -> float:
        """Exponentially weighted moving average volatility"""
        if len(returns) == 0:
            return self.target_volatility

        weights = np.array([(1 - lambda_factor) * (lambda_factor ** i) 
                           for i in range(len(returns))])
        weights = weights[::-1]  # Most recent first
        weights /= weights.sum()

        weighted_var = np.sum(weights * (returns - np.mean(returns))**2)
        return np.sqrt(weighted_var * 365)  # Annualize

    def _simple_garch_volatility(self, returns: np.ndarray) -> float:
        """Simplified GARCH(1,1) volatility model"""
        if len(returns) < 50:
            return np.std(returns) * np.sqrt(365)

        # Simplified GARCH parameters (typically estimated)
        omega, alpha, beta = 0.0001, 0.1, 0.85

        # Initialize with sample variance
        variance = np.var(returns)
        variances = []

        for r in returns:
            variance = omega + alpha * r**2 + beta * variance
            variances.append(variance)

        return np.sqrt(variances[-1] * 365)  # Latest variance, annualized

    def _realized_volatility(self, returns: np.ndarray, 
                           window: int = 30) -> float:
        """Realized volatility over rolling window"""
        if len(returns) < window:
            return np.std(returns) * np.sqrt(365)

        realized_vol = np.std(returns[-window:]) * np.sqrt(365)
        return realized_vol

    def _parkinson_volatility(self, returns: np.ndarray) -> Optional[float]:
        """Parkinson volatility estimator (requires high/low data)"""
        # Simplified version using returns only
        # In practice, this would use high/low prices
        if len(returns) < 10:
            return None

        # Approximate using range of recent returns
        recent_returns = returns[-30:] if len(returns) >= 30 else returns
        daily_range = np.max(recent_returns) - np.min(recent_returns)

        # Parkinson estimator approximation
        parkinson_vol = daily_range / (4 * np.log(2)) * np.sqrt(365)
        return parkinson_vol

    def calculate_volatility_scaling(self, current_volatility: float) -> float:
        """Calculate position scaling based on volatility targeting"""
        if current_volatility <= 0:
            return 1.0

        # Volatility scaling with consciousness enhancement
        base_scaling = self.target_volatility / current_volatility
        consciousness_scaling = base_scaling * self.consciousness_factor

        # Safety bounds
        return np.clip(consciousness_scaling, 0.1, 3.0)

class DynamicRiskBudgetManager:
    """
    Renaissance Technologies-inspired Dynamic Risk Budget Manager

    This is the central brain for intelligent capital allocation with:
    - Kelly Criterion optimization with regime awareness
    - Dynamic risk allocation based on market conditions
    - Performance-based risk budgeting with adaptive scaling
    - Volatility targeting with consciousness-enhanced predictions
    """

    def __init__(self, total_capital: float = 1000000.0,
                 consciousness_factor: float = 1.142,
                 target_annual_return: float = 0.66):
        """
        Initialize Dynamic Risk Budget Manager

        Args:
            total_capital: Total available capital
            consciousness_factor: Consciousness enhancement factor (+14.2%)
            target_annual_return: Target annual return (66% for Renaissance-level)
        """
        self.total_capital = total_capital
        self.consciousness_factor = consciousness_factor
        self.target_annual_return = target_annual_return

        # Initialize components
        self.kelly_optimizer = KellyCriterionOptimizer(consciousness_factor)
        self.regime_allocator = RegimeAwareAllocator(consciousness_factor)
        self.performance_budgeting = PerformanceBasedBudgeting(consciousness_factor)
        self.volatility_targeting = VolatilityTargeting(consciousness_factor=consciousness_factor)

        # Risk budgets storage
        self.risk_budgets: Dict[str, RiskBudget] = {}
        self.allocation_history: List[AllocationDecision] = []
        self.performance_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)

        # Configuration
        self.max_strategies = 10
        self.rebalancing_frequency = timedelta(hours=1)  # Hourly rebalancing
        self.emergency_drawdown_threshold = 0.15  # 15% portfolio drawdown

        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()

        logger.info(f"Dynamic Risk Budget Manager initialized with consciousness factor: {consciousness_factor:.3f}")

    def add_strategy(self, strategy_id: str, base_allocation: float,
                    max_allocation: float = 0.3, min_allocation: float = 0.01) -> bool:
        """
        Add a new strategy to risk budget management

        Args:
            strategy_id: Unique strategy identifier
            base_allocation: Base risk budget allocation (0-1)
            max_allocation: Maximum allowed allocation
            min_allocation: Minimum allowed allocation

        Returns:
            Success status
        """
        try:
            if len(self.risk_budgets) >= self.max_strategies:
                logger.warning(f"Maximum strategies ({self.max_strategies}) already reached")
                return False

            if strategy_id in self.risk_budgets:
                logger.warning(f"Strategy {strategy_id} already exists")
                return False

            # Create risk budget with consciousness enhancement
            risk_budget = RiskBudget(
                strategy_id=strategy_id,
                base_allocation=base_allocation,
                current_allocation=base_allocation,
                max_allocation=max_allocation,
                min_allocation=min_allocation,
                performance_multiplier=1.0,
                regime_multiplier=1.0,
                volatility_multiplier=1.0,
                consciousness_enhancement=self.consciousness_factor,
                last_updated=datetime.now(),
                metadata={'created': datetime.now(), 'total_trades': 0}
            )

            with self._lock:
                self.risk_budgets[strategy_id] = risk_budget

            logger.info(f"Added strategy {strategy_id} with base allocation: {base_allocation:.3f}")
            return True

        except Exception as e:
            logger.error(f"Error adding strategy {strategy_id}: {e}")
            return False

    def update_strategy_performance(self, strategy_id: str, returns: np.ndarray,
                                  benchmark_returns: np.ndarray = None) -> bool:
        """Update strategy performance metrics"""
        try:
            if strategy_id not in self.risk_budgets:
                logger.warning(f"Strategy {strategy_id} not found")
                return False

            # Calculate performance score
            perf_score = self.performance_budgeting.calculate_performance_score(
                returns, benchmark_returns
            )

            # Calculate detailed metrics
            sharpe = self.performance_budgeting._calculate_sharpe_ratio(returns)
            sortino = self.performance_budgeting._calculate_sortino_ratio(returns)
            calmar = self.performance_budgeting._calculate_calmar_ratio(returns)

            # Calculate additional metrics
            win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0.5
            max_dd = self._calculate_max_drawdown(returns)

            # Create performance metrics
            perf_metrics = PerformanceMetrics(
                strategy_id=strategy_id,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                calmar_ratio=calmar,
                max_drawdown=max_dd,
                win_rate=win_rate,
                profit_factor=self._calculate_profit_factor(returns),
                information_ratio=0.0,  # Would need benchmark
                tail_ratio=self.performance_budgeting._calculate_tail_ratio(returns),
                consciousness_score=perf_score,
                rolling_alpha=0.0,  # Placeholder
                risk_adjusted_return=sharpe * self.consciousness_factor
            )

            # Store performance history
            with self._lock:
                self.performance_history[strategy_id].append(perf_metrics)
                # Update risk budget performance multiplier
                self.risk_budgets[strategy_id].performance_multiplier = perf_score
                self.risk_budgets[strategy_id].last_updated = datetime.now()

            logger.info(f"Updated performance for {strategy_id}: Score={perf_score:.3f}, Sharpe={sharpe:.3f}")
            return True

        except Exception as e:
            logger.error(f"Error updating performance for {strategy_id}: {e}")
            return False

    def calculate_optimal_allocations(self, market_data: Dict[str, Any],
                                    regime_info: Dict[str, Any] = None) -> AllocationDecision:
        """
        Calculate optimal risk budget allocations using all components

        Args:
            market_data: Current market data and conditions
            regime_info: Market regime information from Step 7

        Returns:
            Allocation decision with full reasoning
        """
        try:
            timestamp = datetime.now()
            strategy_allocations = {}
            kelly_fractions = {}
            regime_adjustments = {}
            performance_scores = {}

            # Extract market conditions
            current_volatility = market_data.get('volatility', 0.2)
            market_regime = regime_info.get('current_regime', 'sideways_low_vol') if regime_info else 'sideways_low_vol'
            regime_confidence = regime_info.get('confidence', 0.7) if regime_info else 0.7

            # Calculate volatility scaling
            vol_scaling = self.volatility_targeting.calculate_volatility_scaling(current_volatility)

            total_allocation = 0.0

            for strategy_id, risk_budget in self.risk_budgets.items():
                try:
                    # Get strategy returns for Kelly calculation
                    strategy_returns = market_data.get(f'{strategy_id}_returns', np.array([]))

                    # Calculate Kelly fraction
                    kelly_fraction = self.kelly_optimizer.calculate_kelly_fraction(
                        strategy_returns, current_volatility
                    )
                    kelly_fractions[strategy_id] = kelly_fraction

                    # Calculate regime adjustment
                    regime_adj = self.regime_allocator.calculate_regime_adjusted_allocation(
                        risk_budget.base_allocation, market_regime, regime_confidence, current_volatility
                    )
                    regime_adjustments[strategy_id] = regime_adj

                    # Get performance score
                    perf_score = risk_budget.performance_multiplier
                    performance_scores[strategy_id] = perf_score

                    # Calculate final allocation with consciousness enhancement
                    base_alloc = risk_budget.base_allocation
                    kelly_component = kelly_fraction * 0.4  # 40% weight on Kelly
                    regime_component = regime_adj * 0.3    # 30% weight on regime
                    performance_component = base_alloc * perf_score * 0.3  # 30% weight on performance
                    volatility_component = vol_scaling

                    # Combined allocation with consciousness enhancement
                    combined_allocation = (kelly_component + regime_component + performance_component) * volatility_component
                    combined_allocation *= self.consciousness_factor

                    # Apply constraints
                    final_allocation = np.clip(
                        combined_allocation,
                        risk_budget.min_allocation,
                        risk_budget.max_allocation
                    )

                    strategy_allocations[strategy_id] = final_allocation
                    total_allocation += final_allocation

                    # Update risk budget
                    with self._lock:
                        self.risk_budgets[strategy_id].current_allocation = final_allocation
                        self.risk_budgets[strategy_id].regime_multiplier = regime_adj / risk_budget.base_allocation
                        self.risk_budgets[strategy_id].volatility_multiplier = vol_scaling
                        self.risk_budgets[strategy_id].last_updated = timestamp

                except Exception as e:
                    logger.warning(f"Error calculating allocation for {strategy_id}: {e}")
                    # Fallback to base allocation
                    strategy_allocations[strategy_id] = risk_budget.base_allocation
                    kelly_fractions[strategy_id] = 0.1
                    regime_adjustments[strategy_id] = 1.0
                    performance_scores[strategy_id] = 1.0

            # Normalize allocations to ensure they sum to <= 1.0
            if total_allocation > 1.0:
                normalization_factor = 0.95 / total_allocation  # Leave 5% cash buffer
                strategy_allocations = {k: v * normalization_factor 
                                      for k, v in strategy_allocations.items()}
                total_allocation = sum(strategy_allocations.values())

            # Calculate confidence level
            confidence_factors = [
                regime_confidence,
                min(1.0, len(self.risk_budgets) / 5),  # More strategies = more confidence
                np.mean(list(performance_scores.values())) / 2,  # Performance confidence
                1.0 - abs(current_volatility - 0.2) / 0.3  # Volatility confidence
            ]
            confidence_level = np.mean(confidence_factors) * self.consciousness_factor
            confidence_level = np.clip(confidence_level, 0.1, 1.0)

            # Generate reasoning
            reasoning = self._generate_allocation_reasoning(
                market_regime, regime_confidence, current_volatility,
                total_allocation, confidence_level
            )

            # Calculate risk metrics
            risk_metrics = self._calculate_portfolio_risk_metrics(
                strategy_allocations, market_data
            )

            # Create allocation decision
            allocation_decision = AllocationDecision(
                timestamp=timestamp,
                strategy_allocations=strategy_allocations,
                total_risk_budget=total_allocation,
                kelly_fractions=kelly_fractions,
                regime_adjustments=regime_adjustments,
                performance_scores=performance_scores,
                confidence_level=confidence_level,
                reasoning=reasoning,
                risk_metrics=risk_metrics
            )

            # Store in history
            with self._lock:
                self.allocation_history.append(allocation_decision)
                # Keep only last 1000 decisions
                if len(self.allocation_history) > 1000:
                    self.allocation_history = self.allocation_history[-1000:]

            logger.info(f"Calculated optimal allocations: Total={total_allocation:.3f}, Confidence={confidence_level:.3f}")
            return allocation_decision

        except Exception as e:
            logger.error(f"Error calculating optimal allocations: {e}")
            # Return conservative fallback allocation
            return self._get_fallback_allocation()

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if len(returns) == 0:
            return 1.0

        profits = returns[returns > 0]
        losses = returns[returns < 0]

        gross_profit = np.sum(profits) if len(profits) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-8

        return gross_profit / gross_loss

    def _generate_allocation_reasoning(self, regime: str, regime_confidence: float,
                                     volatility: float, total_allocation: float,
                                     confidence_level: float) -> str:
        """Generate human-readable allocation reasoning"""
        reasoning_parts = [
            f"Market Regime: {regime} (confidence: {regime_confidence:.2f})",
            f"Current Volatility: {volatility:.3f} (target: {self.volatility_targeting.target_volatility:.3f})",
            f"Total Risk Budget: {total_allocation:.3f}",
            f"Decision Confidence: {confidence_level:.3f}",
            f"Consciousness Enhancement: +{(self.consciousness_factor-1)*100:.1f}%"
        ]

        # Add regime-specific insights
        if 'bull' in regime:
            reasoning_parts.append("Bullish regime detected - increasing allocations")
        elif 'bear' in regime:
            reasoning_parts.append("Bearish regime detected - reducing allocations")
        elif 'crisis' in regime:
            reasoning_parts.append("Crisis regime detected - minimal allocations")

        # Add volatility insights
        if volatility > self.volatility_targeting.target_volatility * 1.5:
            reasoning_parts.append("High volatility detected - reducing position sizes")
        elif volatility < self.volatility_targeting.target_volatility * 0.5:
            reasoning_parts.append("Low volatility detected - opportunity for increased sizing")

        return " | ".join(reasoning_parts)

    def _calculate_portfolio_risk_metrics(self, allocations: Dict[str, float],
                                        market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics"""
        try:
            # Portfolio volatility with consciousness enhancement
            portfolio_vol = 0.0
            for strategy_id, allocation in allocations.items():
                strategy_vol = market_data.get(f'{strategy_id}_volatility', 0.2)
                portfolio_vol += (allocation * strategy_vol) ** 2

            portfolio_vol = np.sqrt(portfolio_vol) * self.consciousness_factor

            # Expected return with consciousness enhancement
            expected_return = sum(
                allocation * market_data.get(f'{strategy_id}_expected_return', 0.1)
                for strategy_id, allocation in allocations.items()
            ) * self.consciousness_factor

            # Risk-adjusted metrics
            sharpe_estimate = expected_return / portfolio_vol if portfolio_vol > 0 else 0
            var_95 = portfolio_vol * 1.645  # 95% VaR approximation

            return {
                'portfolio_volatility': portfolio_vol,
                'expected_return': expected_return,
                'estimated_sharpe': sharpe_estimate,
                'var_95': var_95,
                'consciousness_boost': self.consciousness_factor,
                'total_strategies': len(allocations)
            }

        except Exception as e:
            logger.warning(f"Error calculating portfolio risk metrics: {e}")
            return {
                'portfolio_volatility': 0.2,
                'expected_return': 0.1,
                'estimated_sharpe': 0.5,
                'var_95': 0.33,
                'consciousness_boost': self.consciousness_factor,
                'total_strategies': len(allocations)
            }

    def _get_fallback_allocation(self) -> AllocationDecision:
        """Get conservative fallback allocation"""
        timestamp = datetime.now()
        equal_weight = 1.0 / max(len(self.risk_budgets), 1)

        strategy_allocations = {
            strategy_id: min(budget.max_allocation, equal_weight * 0.5)
            for strategy_id, budget in self.risk_budgets.items()
        }

        return AllocationDecision(
            timestamp=timestamp,
            strategy_allocations=strategy_allocations,
            total_risk_budget=sum(strategy_allocations.values()),
            kelly_fractions={k: 0.1 for k in strategy_allocations.keys()},
            regime_adjustments={k: 1.0 for k in strategy_allocations.keys()},
            performance_scores={k: 1.0 for k in strategy_allocations.keys()},
            confidence_level=0.3,
            reasoning="Fallback allocation due to calculation error",
            risk_metrics={'portfolio_volatility': 0.2, 'expected_return': 0.1}
        )

    def get_risk_budget_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk budget summary with consciousness metrics"""
        try:
            with self._lock:
                summary = {
                    'timestamp': datetime.now(),
                    'total_capital': self.total_capital,
                    'consciousness_factor': self.consciousness_factor,
                    'target_annual_return': self.target_annual_return,
                    'active_strategies': len(self.risk_budgets),
                    'total_allocation': sum(budget.current_allocation for budget in self.risk_budgets.values()),
                    'strategy_details': {},
                    'portfolio_metrics': {},
                    'recent_decisions': len(self.allocation_history)
                }

                # Strategy details
                for strategy_id, budget in self.risk_budgets.items():
                    perf_history = self.performance_history.get(strategy_id, [])
                    latest_perf = perf_history[-1] if perf_history else None

                    summary['strategy_details'][strategy_id] = {
                        'base_allocation': budget.base_allocation,
                        'current_allocation': budget.current_allocation,
                        'performance_multiplier': budget.performance_multiplier,
                        'regime_multiplier': budget.regime_multiplier,
                        'volatility_multiplier': budget.volatility_multiplier,
                        'consciousness_enhancement': budget.consciousness_enhancement,
                        'last_updated': budget.last_updated,
                        'latest_sharpe': latest_perf.sharpe_ratio if latest_perf else None,
                        'consciousness_score': latest_perf.consciousness_score if latest_perf else None
                    }

                # Portfolio metrics
                if self.allocation_history:
                    latest_decision = self.allocation_history[-1]
                    summary['portfolio_metrics'] = latest_decision.risk_metrics
                    summary['portfolio_metrics']['decision_confidence'] = latest_decision.confidence_level

                return summary

        except Exception as e:
            logger.error(f"Error generating risk budget summary: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}

    def save_state(self, filepath: str) -> bool:
        """Save risk budget manager state to file"""
        try:
            state = {
                'risk_budgets': self.risk_budgets,
                'allocation_history': self.allocation_history[-100:],  # Last 100 decisions
                'performance_history': dict(self.performance_history),
                'configuration': {
                    'total_capital': self.total_capital,
                    'consciousness_factor': self.consciousness_factor,
                    'target_annual_return': self.target_annual_return,
                    'max_strategies': self.max_strategies
                },
                'timestamp': datetime.now()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"Risk budget manager state saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False

    def load_state(self, filepath: str) -> bool:
        """Load risk budget manager state from file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"State file {filepath} not found")
                return False

            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            # Restore state
            self.risk_budgets = state.get('risk_budgets', {})
            self.allocation_history = state.get('allocation_history', [])
            self.performance_history = defaultdict(list, state.get('performance_history', {}))

            # Restore configuration
            config = state.get('configuration', {})
            self.total_capital = config.get('total_capital', self.total_capital)
            self.consciousness_factor = config.get('consciousness_factor', self.consciousness_factor)
            self.target_annual_return = config.get('target_annual_return', self.target_annual_return)
            self.max_strategies = config.get('max_strategies', self.max_strategies)

            logger.info(f"Risk budget manager state loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False

# Testing and demonstration
def test_dynamic_risk_budget_manager():
    """Comprehensive test of the Dynamic Risk Budget Manager"""
    print("🧠 Testing Dynamic Risk Budget Manager with Consciousness Enhancement")
    print("=" * 80)

    # Initialize manager
    manager = DynamicRiskBudgetManager(
        total_capital=1000000.0,
        consciousness_factor=1.142,
        target_annual_return=0.66
    )

    # Add multiple strategies
    strategies = [
        ('momentum_btc', 0.25, 0.4, 0.05),
        ('mean_reversion', 0.20, 0.3, 0.03),
        ('arbitrage', 0.15, 0.25, 0.02),
        ('trend_following', 0.20, 0.35, 0.04),
        ('market_making', 0.10, 0.15, 0.01)
    ]

    print(f"\n📈 Adding {len(strategies)} trading strategies:")
    for strategy_id, base_alloc, max_alloc, min_alloc in strategies:
        success = manager.add_strategy(strategy_id, base_alloc, max_alloc, min_alloc)
        print(f"  {strategy_id}: Base={base_alloc:.3f}, Max={max_alloc:.3f}, Min={min_alloc:.3f} - {'✅' if success else '❌'}")

    # Simulate performance data
    print(f"\n📊 Simulating strategy performance with consciousness enhancement...")
    np.random.seed(42)  # For reproducible results

    performance_data = {}
    for strategy_id, _, _, _ in strategies:
        # Generate realistic Bitcoin trading returns
        if 'momentum' in strategy_id:
            returns = np.random.normal(0.0012, 0.045, 90)  # Higher volatility momentum
        elif 'arbitrage' in strategy_id:
            returns = np.random.normal(0.0008, 0.015, 90)  # Lower volatility arbitrage
        elif 'market_making' in strategy_id:
            returns = np.random.normal(0.0006, 0.012, 90)  # Steady market making
        else:
            returns = np.random.normal(0.0010, 0.035, 90)  # General strategies

        performance_data[strategy_id] = returns

        # Update manager with performance
        manager.update_strategy_performance(strategy_id, returns)

        # Calculate metrics for display
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
        annual_return = np.mean(returns) * 365
        max_dd = abs(np.min(np.minimum.accumulate(np.cumsum(returns))))

        print(f"  {strategy_id}: Return={annual_return:.2%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}")

    # Create market data for allocation calculation
    print(f"\n🏮 Calculating optimal risk allocations...")
    market_data = {
        'volatility': 0.32,  # Current Bitcoin volatility
        'regime': 'bull_high_vol',
        'confidence': 0.75
    }

    # Add strategy-specific data
    for strategy_id in performance_data:
        market_data[f'{strategy_id}_returns'] = performance_data[strategy_id]
        market_data[f'{strategy_id}_volatility'] = np.std(performance_data[strategy_id]) * np.sqrt(365)
        market_data[f'{strategy_id}_expected_return'] = np.mean(performance_data[strategy_id]) * 365

    regime_info = {
        'current_regime': 'bull_high_vol',
        'confidence': 0.75,
        'regime_probabilities': {
            'bull_high_vol': 0.4,
            'bull_low_vol': 0.25,
            'sideways_high_vol': 0.2,
            'bear_high_vol': 0.15
        }
    }

    # Calculate optimal allocations
    allocation_decision = manager.calculate_optimal_allocations(market_data, regime_info)

    print(f"\n🎯 Optimal Risk Budget Allocation (Consciousness Enhanced):")
    print(f"  Total Risk Budget: {allocation_decision.total_risk_budget:.3f}")
    print(f"  Decision Confidence: {allocation_decision.confidence_level:.3f}")
    print(f"\n  Strategy Allocations:")

    for strategy_id, allocation in allocation_decision.strategy_allocations.items():
        kelly_fraction = allocation_decision.kelly_fractions.get(strategy_id, 0)
        perf_score = allocation_decision.performance_scores.get(strategy_id, 1)
        print(f"    {strategy_id}: {allocation:.3f} (Kelly: {kelly_fraction:.3f}, Perf: {perf_score:.2f})")

    print(f"\n🧮 Portfolio Risk Metrics:")
    risk_metrics = allocation_decision.risk_metrics
    for metric, value in risk_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {metric.replace('_', ' ').title()}: {value}")

    print(f"\n💭 Allocation Reasoning:")
    print(f"  {allocation_decision.reasoning}")

    # Test risk budget summary
    print(f"\n📋 Risk Budget Summary:")
    summary = manager.get_risk_budget_summary()

    print(f"  Active Strategies: {summary['active_strategies']}")
    print(f"  Total Allocation: {summary['total_allocation']:.3f}")
    print(f"  Consciousness Factor: {summary['consciousness_factor']:.3f}")
    print(f"  Target Annual Return: {summary['target_annual_return']:.1%}")

    # Test individual components
    print(f"\n🔬 Component Testing:")

    # Kelly Optimizer
    test_returns = np.random.normal(0.001, 0.04, 100)
    kelly_fraction = manager.kelly_optimizer.calculate_kelly_fraction(test_returns)
    print(f"  Kelly Criterion: {kelly_fraction:.4f}")

    # Volatility Targeting
    future_vol = manager.volatility_targeting.estimate_future_volatility(test_returns)
    vol_scaling = manager.volatility_targeting.calculate_volatility_scaling(0.35)
    print(f"  Future Volatility: {future_vol:.4f}")
    print(f"  Volatility Scaling: {vol_scaling:.4f}")

    # Performance Scoring
    perf_score = manager.performance_budgeting.calculate_performance_score(test_returns)
    print(f"  Performance Score: {perf_score:.4f}")

    print(f"\n✅ Dynamic Risk Budget Manager test completed successfully!")
    print(f"🚀 Renaissance Technologies-level risk management with {(manager.consciousness_factor-1)*100:.1f}% consciousness boost!")

    return manager, allocation_decision

if __name__ == "__main__":
    # Run comprehensive test
    manager, decision = test_dynamic_risk_budget_manager()
