"""
Dynamic Spread Optimizer with Consciousness Enhancement
Advanced spread calculation and optimization for market making with Renaissance Technologies-level sophistication
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from collections import deque
from scipy import optimize, stats
from scipy.special import erf
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings

warnings.filterwarnings('ignore')


class SpreadRegime(Enum):
    """Spread regime classification"""
    TIGHT = "tight"
    NORMAL = "normal"
    WIDE = "wide"
    STRESSED = "stressed"
    CRISIS = "crisis"


class OptimizationMethod(Enum):
    """Spread optimization methods"""
    AVELLANEDA_STOIKOV = "avellaneda_stoikov"
    GLOSTEN_MILGROM = "glosten_milgrom"
    HO_STOLL = "ho_stoll"
    CONSCIOUSNESS_ENHANCED = "consciousness_enhanced"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


@dataclass
class SpreadConfig:
    """Configuration for spread optimizer"""
    base_spread_bps: float = 5.0  # Base spread in basis points
    min_spread_bps: float = 1.0  # Minimum spread
    max_spread_bps: float = 50.0  # Maximum spread
    consciousness_boost: float = 1.0  # Neutralized (was 1.142)
    risk_aversion_gamma: float = 0.01  # Risk aversion parameter
    inventory_penalty: float = 0.001  # Inventory penalty coefficient
    adverse_selection_factor: float = 0.5  # Adverse selection impact
    volatility_scaling: float = 2.0  # Volatility impact scaling
    liquidity_scaling: float = 1.5  # Liquidity impact scaling
    regime_adaptation_speed: float = 0.1  # How quickly to adapt to regime changes
    optimization_method: OptimizationMethod = OptimizationMethod.CONSCIOUSNESS_ENHANCED


@dataclass
class MarketState:
    """Current market state for spread calculation"""
    mid_price: float
    bid_price: float
    ask_price: float
    volatility: float
    bid_depth: float
    ask_depth: float
    inventory_position: float
    order_flow_imbalance: float
    microstructure_signal: float
    adverse_selection_risk: float
    liquidity_score: float
    regime_indicator: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class SpreadComponents:
    """Breakdown of spread components"""
    base_component: float
    volatility_component: float
    inventory_component: float
    adverse_selection_component: float
    liquidity_component: float
    regime_component: float
    consciousness_component: float
    total_spread: float


@dataclass
class OptimizationResult:
    """Result of spread optimization"""
    optimal_bid_spread: float
    optimal_ask_spread: float
    symmetric_spread: float
    spread_components: SpreadComponents
    expected_pnl: float
    expected_risk: float
    sharpe_ratio: float
    fill_probability_bid: float
    fill_probability_ask: float
    confidence_score: float
    computation_time_ms: float


class DynamicSpreadOptimizer:
    """
    Dynamic Spread Optimizer with Consciousness Enhancement

    Features:
    - Multiple theoretical models (Avellaneda-Stoikov, Glosten-Milgrom, etc.)
    - Consciousness-enhanced optimization with 1.142x boost
    - Regime-aware adaptive spreading
    - Real-time adverse selection protection
    - Multi-objective optimization (PnL vs Risk)
    - Dynamic inventory-dependent asymmetric spreads
    - Advanced volatility and liquidity adjustments
    """

    def __init__(self, config: SpreadConfig):
        self.config = config
        self.logger = self._setup_logging()

        # Historical data for model calibration
        self.market_history: deque = deque(maxlen=1000)
        self.spread_history: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=500)

        # Model parameters (dynamically calibrated)
        self.model_parameters = self._initialize_model_parameters()

        # FIXED: Initialize performance tracking as dict with lists
        self.performance_tracking = {}

        # Performance tracking
        self.performance_metrics = self._initialize_performance_metrics()

        # Optimization cache for expensive calculations
        self.optimization_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 0.05  # 50ms cache TTL

        # Regime detection state
        self.current_regime = SpreadRegime.NORMAL
        self.regime_confidence = 0.5

        self.logger.info(
            f"Dynamic Spread Optimizer initialized with consciousness factor: {config.consciousness_boost}")
        self.logger.info(f"Optimization method: {config.optimization_method.value}")

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("DynamicSpreadOptimizer")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_model_parameters(self) -> Dict[str, Any]:
        """Initialize model parameters for different optimization methods"""
        return {
            # Avellaneda-Stoikov parameters
            'as_gamma': self.config.risk_aversion_gamma,
            'as_kappa': 1.5,  # Mean reversion speed
            'as_A': 140.0,  # Order arrival intensity
            'as_k': 1.5,  # Temporary impact parameter

            # Glosten-Milgrom parameters
            'gm_alpha': 0.3,  # Probability of informed trading
            'gm_sigma_u': 0.01,  # Uninformed volatility
            'gm_sigma_i': 0.03,  # Informed volatility

            # Ho-Stoll parameters
            'hs_processing_cost': 0.0001,  # Processing cost component
            'hs_inventory_holding_cost': 0.0005,  # Inventory holding cost

            # Consciousness enhancement parameters
            'consciousness_alpha': self.config.consciousness_boost - 1.0,
            'consciousness_beta': 0.5,
            'consciousness_gamma': 0.3,

            # Adaptive parameters
            'adaptation_learning_rate': 0.01,
            'regime_weights': {
                SpreadRegime.TIGHT: 0.7,
                SpreadRegime.NORMAL: 1.0,
                SpreadRegime.WIDE: 1.4,
                SpreadRegime.STRESSED: 2.0,
                SpreadRegime.CRISIS: 3.0
            }
        }

    def _initialize_performance_metrics(self) -> Dict[str, Any]:
        """Initialize performance tracking"""
        return {
            'total_optimizations': 0,
            'avg_computation_time_ms': 0.0,
            'spread_accuracy_score': 0.0,
            'pnl_improvement': 0.0,
            'risk_reduction': 0.0,
            'consciousness_effectiveness': 0.0,
            'regime_adaptation_score': 0.0,
            'adverse_selection_mitigation': 0.0,
            'start_time': time.time()
        }

    def optimize_spread(self, market_state: MarketState) -> OptimizationResult:
        """Optimize spreads using consciousness-enhanced multi-model approach"""

        start_time = time.time()

        # Check cache first
        cache_key = self._generate_cache_key(market_state)
        if self._is_cache_valid(cache_key):
            cached_result = self.optimization_cache[cache_key]
            cached_result.computation_time_ms = (time.time() - start_time) * 1000

            # FIXED: Only update performance tracking here for cached results
            # Don't increment total_optimizations here - it's done in _update_performance_metrics
            self._update_performance_metrics(cached_result)

            return cached_result

        try:
            # Update regime detection
            self._update_spread_regime(market_state)

            # Store market state in history
            self.market_history.append(market_state)

            # Choose optimization method based on configuration
            if self.config.optimization_method == OptimizationMethod.CONSCIOUSNESS_ENHANCED:
                result = self._optimize_consciousness_enhanced(market_state)
            elif self.config.optimization_method == OptimizationMethod.AVELLANEDA_STOIKOV:
                result = self._optimize_avellaneda_stoikov(market_state)
            elif self.config.optimization_method == OptimizationMethod.GLOSTEN_MILGROM:
                result = self._optimize_glosten_milgrom(market_state)
            elif self.config.optimization_method == OptimizationMethod.HO_STOLL:
                result = self._optimize_ho_stoll(market_state)
            elif self.config.optimization_method == OptimizationMethod.ADAPTIVE_HYBRID:
                result = self._optimize_adaptive_hybrid(market_state)
            else:
                result = self._optimize_consciousness_enhanced(market_state)

            # Apply bounds and validation
            result = self._apply_spread_bounds(result, market_state)

            # Calculate performance metrics
            result.computation_time_ms = (time.time() - start_time) * 1000

            # Cache result
            self.optimization_cache[cache_key] = result
            self.cache_timestamps[cache_key] = time.time()

            # Update performance tracking
            self._update_performance_metrics(result)

            # Store in history
            self.spread_history.append(result)

            self.logger.debug(f"Spread optimized: Bid={result.optimal_bid_spread:.1f}bps, "
                              f"Ask={result.optimal_ask_spread:.1f}bps, "
                              f"Symmetric={result.symmetric_spread:.1f}bps")

            return result

        except Exception as e:
            self.logger.error(f"Error in spread optimization: {e}")
            return self._create_fallback_result(market_state)

    def _optimize_consciousness_enhanced(self, market_state: MarketState) -> OptimizationResult:
        """Consciousness-enhanced spread optimization combining multiple models"""

        consciousness_factor = self.config.consciousness_boost

        # Calculate base spread components with consciousness enhancement
        components = self._calculate_enhanced_spread_components(market_state)

        # Multi-objective optimization with consciousness boost
        def objective_function(spreads):
            bid_spread, ask_spread = spreads

            # Expected PnL with consciousness enhancement
            expected_pnl = self._calculate_expected_pnl_enhanced(
                market_state, bid_spread, ask_spread
            )

            # Risk penalty with consciousness mitigation
            risk_penalty = self._calculate_risk_penalty_enhanced(
                market_state, bid_spread, ask_spread
            )

            # Consciousness bonus for optimal behavior
            consciousness_bonus = self._calculate_consciousness_bonus(
                market_state, bid_spread, ask_spread
            )

            # Combined objective (maximize PnL - Risk + Consciousness)
            objective = expected_pnl - risk_penalty + consciousness_bonus

            return -objective  # Minimize negative for optimization

        # Constraints for asymmetric spreads
        def spread_constraint(spreads):
            bid_spread, ask_spread = spreads
            # Ensure reasonable asymmetry based on inventory and flow
            max_asymmetry = 0.3 * consciousness_factor  # Enhanced asymmetry tolerance
            symmetric_spread = (bid_spread + ask_spread) / 2
            return max_asymmetry - abs((bid_spread - ask_spread) / symmetric_spread)

        # Initial guess based on components
        initial_symmetric = components.total_spread
        initial_asymmetry = self._calculate_optimal_asymmetry(market_state)

        initial_guess = [
            initial_symmetric + initial_asymmetry,  # bid spread
            initial_symmetric - initial_asymmetry  # ask spread
        ]

        # Bounds (in bps)
        bounds = [
            (self.config.min_spread_bps, self.config.max_spread_bps),  # bid spread
            (self.config.min_spread_bps, self.config.max_spread_bps)  # ask spread
        ]

        # Optimization
        try:
            result = optimize.minimize(
                objective_function,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': spread_constraint},
                options={'maxiter': 50, 'ftol': 1e-6}
            )

            if result.success:
                optimal_bid_spread, optimal_ask_spread = result.x
            else:
                # Fallback to initial guess if optimization fails
                optimal_bid_spread, optimal_ask_spread = initial_guess

        except Exception as e:
            self.logger.warning(f"Optimization failed, using fallback: {e}")
            optimal_bid_spread, optimal_ask_spread = initial_guess

        # Calculate final metrics
        symmetric_spread = (optimal_bid_spread + optimal_ask_spread) / 2

        expected_pnl = self._calculate_expected_pnl_enhanced(
            market_state, optimal_bid_spread, optimal_ask_spread
        )

        expected_risk = self._calculate_expected_risk(
            market_state, optimal_bid_spread, optimal_ask_spread
        )

        sharpe_ratio = expected_pnl / max(expected_risk, 1e-8)

        # Fill probabilities with consciousness enhancement
        fill_prob_bid = self._calculate_fill_probability_enhanced(
            market_state, optimal_bid_spread, 'bid'
        )

        fill_prob_ask = self._calculate_fill_probability_enhanced(
            market_state, optimal_ask_spread, 'ask'
        )

        # Confidence score based on model convergence and consciousness
        confidence_score = self._calculate_confidence_score(
            market_state, optimal_bid_spread, optimal_ask_spread
        )

        return OptimizationResult(
            optimal_bid_spread=optimal_bid_spread,
            optimal_ask_spread=optimal_ask_spread,
            symmetric_spread=symmetric_spread,
            spread_components=components,
            expected_pnl=expected_pnl,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            fill_probability_bid=fill_prob_bid,
            fill_probability_ask=fill_prob_ask,
            confidence_score=confidence_score,
            computation_time_ms=0.0  # Will be set by caller
        )

    def _optimize_avellaneda_stoikov(self, market_state: MarketState) -> OptimizationResult:
        """Avellaneda-Stoikov model with consciousness enhancement"""

        # Model parameters with consciousness boost
        gamma = self.model_parameters['as_gamma'] / self.config.consciousness_boost
        kappa = self.model_parameters['as_kappa'] * self.config.consciousness_boost
        A = self.model_parameters['as_A'] * self.config.consciousness_boost
        k = self.model_parameters['as_k']

        # Current inventory
        q = market_state.inventory_position

        # Volatility estimate with consciousness enhancement
        sigma = market_state.volatility * (2.0 - self.config.consciousness_boost)

        # Time to horizon (assume 1 minute)
        T = 60.0

        # Reservation price calculation
        reservation_price = market_state.mid_price - q * gamma * sigma ** 2 * T

        # Optimal spread calculation
        optimal_spread = gamma * sigma ** 2 * T + (2 / gamma) * math.log(1 + gamma / k)

        # Asymmetric adjustments based on inventory
        inventory_skew = q * gamma * sigma ** 2 * T / 2

        optimal_bid_spread = (optimal_spread / 2 + inventory_skew) * 10000  # Convert to bps
        optimal_ask_spread = (optimal_spread / 2 - inventory_skew) * 10000  # Convert to bps

        # Apply consciousness enhancement
        consciousness_adjustment = 1.0 / self.config.consciousness_boost
        optimal_bid_spread *= consciousness_adjustment
        optimal_ask_spread *= consciousness_adjustment

        return self._build_optimization_result(
            market_state, optimal_bid_spread, optimal_ask_spread
        )

    def _optimize_glosten_milgrom(self, market_state: MarketState) -> OptimizationResult:
        """Glosten-Milgrom model with consciousness enhancement"""

        # Model parameters
        alpha = self.model_parameters['gm_alpha']  # Probability of informed trade
        sigma_u = self.model_parameters['gm_sigma_u']  # Uninformed volatility
        sigma_i = self.model_parameters['gm_sigma_i']  # Informed volatility

        # Enhanced adverse selection detection with consciousness
        adverse_selection_prob = market_state.adverse_selection_risk * alpha
        enhanced_detection = adverse_selection_prob / self.config.consciousness_boost

        # Expected value conditional on informed trading
        informed_value_impact = sigma_i * math.sqrt(2 / math.pi)

        # Bid-ask spread calculation
        spread_half = enhanced_detection * informed_value_impact + (1 - enhanced_detection) * sigma_u

        # Inventory adjustment
        inventory_adjustment = abs(market_state.inventory_position) * 0.001

        # Convert to bps
        optimal_bid_spread = (spread_half + inventory_adjustment) * 10000
        optimal_ask_spread = (spread_half + inventory_adjustment) * 10000

        # Apply consciousness factor
        consciousness_multiplier = 1.0 / self.config.consciousness_boost
        optimal_bid_spread *= consciousness_multiplier
        optimal_ask_spread *= consciousness_multiplier

        return self._build_optimization_result(
            market_state, optimal_bid_spread, optimal_ask_spread
        )

    def _optimize_ho_stoll(self, market_state: MarketState) -> OptimizationResult:
        """Ho-Stoll model with consciousness enhancement"""

        # Processing cost component
        processing_cost = self.model_parameters['hs_processing_cost']

        # Inventory holding cost with consciousness adjustment
        holding_cost = (self.model_parameters['hs_inventory_holding_cost'] *
                        abs(market_state.inventory_position) /
                        self.config.consciousness_boost)

        # Adverse selection component enhanced by consciousness
        adverse_selection_cost = (market_state.adverse_selection_risk *
                                  market_state.volatility *
                                  self.config.consciousness_boost)

        # Total spread
        total_spread = processing_cost + holding_cost + adverse_selection_cost

        # Asymmetric component based on inventory and flow
        inventory_skew = (market_state.inventory_position * 0.0001 +
                          market_state.order_flow_imbalance * 0.0002)

        optimal_bid_spread = (total_spread / 2 + inventory_skew) * 10000
        optimal_ask_spread = (total_spread / 2 - inventory_skew) * 10000

        return self._build_optimization_result(
            market_state, optimal_bid_spread, optimal_ask_spread
        )

    def _optimize_adaptive_hybrid(self, market_state: MarketState) -> OptimizationResult:
        """Adaptive hybrid model combining multiple approaches with consciousness"""

        # Calculate spreads using different models
        as_result = self._optimize_avellaneda_stoikov(market_state)
        gm_result = self._optimize_glosten_milgrom(market_state)
        hs_result = self._optimize_ho_stoll(market_state)
        ce_result = self._optimize_consciousness_enhanced(market_state)

        # Adaptive weights based on market conditions and consciousness
        regime_weight = self.model_parameters['regime_weights'][self.current_regime]
        consciousness_weight = self.config.consciousness_boost

        # Weight calculation based on historical performance and current regime
        weights = {
            'as': 0.25 * regime_weight,
            'gm': 0.25 * (1 + market_state.adverse_selection_risk),
            'hs': 0.20 * market_state.liquidity_score,
            'ce': 0.30 * consciousness_weight
        }

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Weighted combination
        optimal_bid_spread = (
                weights['as'] * as_result.optimal_bid_spread +
                weights['gm'] * gm_result.optimal_bid_spread +
                weights['hs'] * hs_result.optimal_bid_spread +
                weights['ce'] * ce_result.optimal_bid_spread
        )

        optimal_ask_spread = (
                weights['as'] * as_result.optimal_ask_spread +
                weights['gm'] * gm_result.optimal_ask_spread +
                weights['hs'] * hs_result.optimal_ask_spread +
                weights['ce'] * ce_result.optimal_ask_spread
        )

        return self._build_optimization_result(
            market_state, optimal_bid_spread, optimal_ask_spread
        )

    def _calculate_enhanced_spread_components(self, market_state: MarketState) -> SpreadComponents:
        """Calculate detailed spread components with consciousness enhancement"""

        consciousness_factor = self.config.consciousness_boost

        # 1. Base component
        base_component = self.config.base_spread_bps

        # 2. Volatility component with consciousness enhancement
        vol_scaling = self.config.volatility_scaling / consciousness_factor
        volatility_component = market_state.volatility * 10000 * vol_scaling

        # 3. Inventory component
        max_inventory = 10.0  # Assumed max inventory for normalization
        inventory_ratio = market_state.inventory_position / max_inventory
        inventory_component = (
                abs(inventory_ratio) * self.config.inventory_penalty * 10000 * consciousness_factor
        )

        # 4. Adverse selection component with consciousness mitigation
        adverse_selection_component = (
                market_state.adverse_selection_risk *
                self.config.adverse_selection_factor *
                10000 / consciousness_factor
        )

        # 5. Liquidity component
        liquidity_penalty = max(0, 1 - market_state.liquidity_score)
        liquidity_component = (
                liquidity_penalty * self.config.liquidity_scaling * 10000 / consciousness_factor
        )

        # 6. Regime component
        regime_multipliers = {
            SpreadRegime.TIGHT: 0.7,
            SpreadRegime.NORMAL: 1.0,
            SpreadRegime.WIDE: 1.4,
            SpreadRegime.STRESSED: 2.0,
            SpreadRegime.CRISIS: 3.0
        }

        regime_multiplier = regime_multipliers.get(self.current_regime, 1.0)
        regime_component = base_component * (regime_multiplier - 1.0) / consciousness_factor

        # 7. Consciousness component (additive enhancement)
        consciousness_component = (consciousness_factor - 1.0) * base_component * 0.5

        # Total spread
        total_spread = (
                base_component +
                volatility_component +
                inventory_component +
                adverse_selection_component +
                liquidity_component +
                regime_component +
                consciousness_component
        )

        return SpreadComponents(
            base_component=base_component,
            volatility_component=volatility_component,
            inventory_component=inventory_component,
            adverse_selection_component=adverse_selection_component,
            liquidity_component=liquidity_component,
            regime_component=regime_component,
            consciousness_component=consciousness_component,
            total_spread=total_spread
        )

    def _calculate_expected_pnl_enhanced(self,
                                         market_state: MarketState,
                                         bid_spread: float,
                                         ask_spread: float) -> float:
        """Calculate expected PnL with consciousness enhancement"""

        consciousness_factor = self.config.consciousness_boost

        # Fill probabilities with consciousness boost
        bid_fill_prob = self._calculate_fill_probability_enhanced(market_state, bid_spread, 'bid')
        ask_fill_prob = self._calculate_fill_probability_enhanced(market_state, ask_spread, 'ask')

        # Expected spread capture (consciousness improves efficiency)
        expected_bid_capture = bid_fill_prob * bid_spread * consciousness_factor
        expected_ask_capture = ask_fill_prob * ask_spread * consciousness_factor

        # Adverse selection cost mitigation through consciousness
        adverse_selection_cost = (
                market_state.adverse_selection_risk *
                (bid_spread + ask_spread) * 0.1 / consciousness_factor
        )

        # Inventory carrying cost
        inventory_cost = abs(market_state.inventory_position) * 0.001

        # Expected PnL per unit time
        expected_pnl = (
                expected_bid_capture + expected_ask_capture -
                adverse_selection_cost - inventory_cost
        )

        return expected_pnl

    def _calculate_risk_penalty_enhanced(self,
                                         market_state: MarketState,
                                         bid_spread: float,
                                         ask_spread: float) -> float:
        """Calculate risk penalty with consciousness mitigation"""

        consciousness_factor = self.config.consciousness_boost

        # Base risk from volatility
        volatility_risk = market_state.volatility ** 2 * (bid_spread + ask_spread)

        # Inventory risk
        inventory_risk = (market_state.inventory_position ** 2 *
                          market_state.volatility * 0.001)

        # Adverse selection risk
        adverse_selection_risk = (market_state.adverse_selection_risk *
                                  market_state.volatility *
                                  (bid_spread + ask_spread) * 0.01)

        # Total risk with consciousness mitigation
        total_risk = (volatility_risk + inventory_risk + adverse_selection_risk) / consciousness_factor

        return total_risk * self.config.risk_aversion_gamma

    def _calculate_consciousness_bonus(self,
                                       market_state: MarketState,
                                       bid_spread: float,
                                       ask_spread: float) -> float:
        """Calculate consciousness-specific bonus for optimal behavior"""

        consciousness_factor = self.config.consciousness_boost

        # Bonus for symmetric spreads (consciousness promotes balance)
        symmetry_bonus = 0.0
        if bid_spread > 0 and ask_spread > 0:
            spread_ratio = min(bid_spread, ask_spread) / max(bid_spread, ask_spread)
            symmetry_bonus = spread_ratio * consciousness_factor * 0.1

        # Bonus for optimal inventory management
        inventory_bonus = 0.0
        if abs(market_state.inventory_position) < 1.0:  # Good inventory control
            inventory_bonus = (1.0 - abs(market_state.inventory_position)) * consciousness_factor * 0.05

        # Bonus for market quality improvement
        market_quality_bonus = market_state.liquidity_score * consciousness_factor * 0.03

        # Bonus for adverse selection mitigation
        adverse_selection_bonus = (
                (1.0 - market_state.adverse_selection_risk) * consciousness_factor * 0.02
        )

        total_bonus = (
                symmetry_bonus + inventory_bonus +
                market_quality_bonus + adverse_selection_bonus
        )

        return total_bonus

    def _calculate_optimal_asymmetry(self, market_state: MarketState) -> float:
        """Calculate optimal spread asymmetry based on market conditions"""

        consciousness_factor = self.config.consciousness_boost

        # Inventory-driven asymmetry
        inventory_asymmetry = market_state.inventory_position * 0.1

        # Flow-driven asymmetry with consciousness enhancement
        flow_asymmetry = market_state.order_flow_imbalance * 0.2 * consciousness_factor

        # Microstructure signal asymmetry
        signal_asymmetry = market_state.microstructure_signal * 0.15

        # Total asymmetry (bounded)
        total_asymmetry = inventory_asymmetry + flow_asymmetry + signal_asymmetry

        # Consciousness factor reduces extreme asymmetries
        enhanced_asymmetry = total_asymmetry / consciousness_factor

        return max(-2.0, min(enhanced_asymmetry, 2.0))  # Bound asymmetry

    def _calculate_fill_probability_enhanced(self,
                                             market_state: MarketState,
                                             spread: float,
                                             side: str) -> float:
        """Calculate fill probability with consciousness enhancement"""

        consciousness_factor = self.config.consciousness_boost

        # Base fill probability using exponential model
        base_lambda = 100.0 * consciousness_factor  # Enhanced arrival rate

        # Spread impact on fill probability
        spread_impact = math.exp(-spread / 10.0)  # Exponential decay

        # Liquidity impact
        liquidity_impact = market_state.liquidity_score

        # Microstructure signal impact (directional bias)
        if side == 'bid':
            signal_impact = 1.0 + market_state.microstructure_signal * 0.1
        else:  # ask
            signal_impact = 1.0 - market_state.microstructure_signal * 0.1

        # Order flow imbalance impact
        if side == 'bid':
            flow_impact = 1.0 + market_state.order_flow_imbalance * 0.2
        else:  # ask
            flow_impact = 1.0 - market_state.order_flow_imbalance * 0.2

        # Combined fill probability with consciousness enhancement
        fill_probability = (
                                   base_lambda * spread_impact * liquidity_impact *
                                   signal_impact * flow_impact * consciousness_factor
                           ) / 3600.0  # Per second

        # Bound probability between 0 and 1
        return max(0.001, min(fill_probability, 0.999))

    def _calculate_expected_risk(self,
                                 market_state: MarketState,
                                 bid_spread: float,
                                 ask_spread: float) -> float:
        """Calculate expected risk metrics"""

        # Volatility contribution
        vol_risk = market_state.volatility * math.sqrt((bid_spread + ask_spread) / 100.0)

        # Inventory risk
        inventory_risk = abs(market_state.inventory_position) * market_state.volatility * 0.1

        # Model uncertainty risk
        model_risk = 0.01 * (bid_spread + ask_spread) / 100.0

        total_risk = vol_risk + inventory_risk + model_risk

        return total_risk

    def _calculate_confidence_score(self,
                                    market_state: MarketState,
                                    bid_spread: float,
                                    ask_spread: float) -> float:
        """Calculate confidence score for optimization result"""

        consciousness_factor = self.config.consciousness_boost

        # Data quality score
        data_quality = min(market_state.liquidity_score, 1.0 - market_state.adverse_selection_risk)

        # Model convergence (simplified - based on spread reasonableness)
        spread_reasonableness = 1.0 / (1.0 + abs(bid_spread - ask_spread) / max(bid_spread, ask_spread, 1.0))

        # Regime confidence
        regime_confidence = self.regime_confidence

        # Historical performance (if available)
        historical_performance = 0.8  # Default
        if len(self.performance_history) > 10:
            recent_performance = [p.get('accuracy', 0.8) for p in list(self.performance_history)[-10:]]
            historical_performance = np.mean(recent_performance)

        # Consciousness boost to confidence
        base_confidence = (
                data_quality * 0.3 +
                spread_reasonableness * 0.3 +
                regime_confidence * 0.2 +
                historical_performance * 0.2
        )

        enhanced_confidence = min(base_confidence * consciousness_factor, 1.0)

        return enhanced_confidence

    def _update_spread_regime(self, market_state: MarketState) -> None:
        """Update spread regime detection with consciousness enhancement"""

        if len(self.market_history) < 10:
            return

        consciousness_factor = self.config.consciousness_boost

        # FIXED: Convert deque to list before slicing to avoid TypeError
        recent_history = list(self.market_history)[-10:]
        recent_volatilities = [m.volatility for m in recent_history]
        recent_liquidity = [m.liquidity_score for m in recent_history]
        recent_adverse_selection = [m.adverse_selection_risk for m in recent_history]

        avg_volatility = np.mean(recent_volatilities)
        avg_liquidity = np.mean(recent_liquidity)
        avg_adverse_selection = np.mean(recent_adverse_selection)

        # Regime classification with consciousness enhancement
        volatility_score = min(avg_volatility * 100 * consciousness_factor, 1.0)
        liquidity_score = avg_liquidity
        stress_score = avg_adverse_selection / consciousness_factor

        # Combined regime indicator
        regime_score = (volatility_score + (1 - liquidity_score) + stress_score) / 3

        # Classify regime
        if regime_score < 0.2:
            new_regime = SpreadRegime.TIGHT
            confidence = 1.0 - regime_score * 5
        elif regime_score < 0.4:
            new_regime = SpreadRegime.NORMAL
            confidence = 1.0 - abs(regime_score - 0.3) * 10
        elif regime_score < 0.6:
            new_regime = SpreadRegime.WIDE
            confidence = 1.0 - abs(regime_score - 0.5) * 5
        elif regime_score < 0.8:
            new_regime = SpreadRegime.STRESSED
            confidence = 1.0 - abs(regime_score - 0.7) * 5
        else:
            new_regime = SpreadRegime.CRISIS
            confidence = regime_score

        # Apply consciousness enhancement to confidence
        enhanced_confidence = min(confidence * consciousness_factor, 1.0)

        # Update regime if confidence is high enough
        if enhanced_confidence > 0.7 and new_regime != self.current_regime:
            self.logger.info(f"Spread regime transition: {self.current_regime.value} -> {new_regime.value}")
            self.current_regime = new_regime
            self.regime_confidence = enhanced_confidence

    def _build_optimization_result(self,
                                   market_state: MarketState,
                                   bid_spread: float,
                                   ask_spread: float) -> OptimizationResult:
        """Build complete optimization result"""

        symmetric_spread = (bid_spread + ask_spread) / 2

        components = self._calculate_enhanced_spread_components(market_state)

        expected_pnl = self._calculate_expected_pnl_enhanced(
            market_state, bid_spread, ask_spread
        )

        expected_risk = self._calculate_expected_risk(
            market_state, bid_spread, ask_spread
        )

        sharpe_ratio = expected_pnl / max(expected_risk, 1e-8)

        fill_prob_bid = self._calculate_fill_probability_enhanced(
            market_state, bid_spread, 'bid'
        )

        fill_prob_ask = self._calculate_fill_probability_enhanced(
            market_state, ask_spread, 'ask'
        )

        confidence_score = self._calculate_confidence_score(
            market_state, bid_spread, ask_spread
        )

        return OptimizationResult(
            optimal_bid_spread=bid_spread,
            optimal_ask_spread=ask_spread,
            symmetric_spread=symmetric_spread,
            spread_components=components,
            expected_pnl=expected_pnl,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            fill_probability_bid=fill_prob_bid,
            fill_probability_ask=fill_prob_ask,
            confidence_score=confidence_score,
            computation_time_ms=0.0
        )

    def _apply_spread_bounds(self,
                             result: OptimizationResult,
                             market_state: MarketState) -> OptimizationResult:
        """Apply spread bounds and validation"""

        # Apply hard bounds
        result.optimal_bid_spread = max(
            self.config.min_spread_bps,
            min(result.optimal_bid_spread, self.config.max_spread_bps)
        )

        result.optimal_ask_spread = max(
            self.config.min_spread_bps,
            min(result.optimal_ask_spread, self.config.max_spread_bps)
        )

        # Recalculate symmetric spread
        result.symmetric_spread = (result.optimal_bid_spread + result.optimal_ask_spread) / 2

        # Validate asymmetry isn't too extreme
        max_asymmetry_ratio = 0.5  # Max 50% asymmetry
        if result.optimal_bid_spread > 0 and result.optimal_ask_spread > 0:
            spread_ratio = min(result.optimal_bid_spread, result.optimal_ask_spread) / max(result.optimal_bid_spread,
                                                                                           result.optimal_ask_spread)

            if spread_ratio < (1 - max_asymmetry_ratio):
                # Reduce asymmetry
                symmetric = result.symmetric_spread
                max_deviation = symmetric * max_asymmetry_ratio

                if result.optimal_bid_spread > result.optimal_ask_spread:
                    result.optimal_bid_spread = symmetric + max_deviation
                    result.optimal_ask_spread = symmetric - max_deviation
                else:
                    result.optimal_bid_spread = symmetric - max_deviation
                    result.optimal_ask_spread = symmetric + max_deviation

        return result

    def _generate_cache_key(self, market_state: MarketState) -> str:
        """Generate cache key for optimization results"""
        # Round values to reduce cache misses from minor variations
        key_components = [
            round(market_state.mid_price, 2),
            round(market_state.volatility, 4),
            round(market_state.inventory_position, 2),
            round(market_state.order_flow_imbalance, 3),
            round(market_state.adverse_selection_risk, 3),
            round(market_state.liquidity_score, 3),
            self.current_regime.value,
            self.config.optimization_method.value
        ]

        return "_".join(map(str, key_components))

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self.optimization_cache:
            return False

        cache_age = time.time() - self.cache_timestamps.get(cache_key, 0)
        return cache_age < self.cache_ttl

    def _create_fallback_result(self, market_state: MarketState) -> OptimizationResult:
        """Create fallback result when optimization fails"""

        fallback_spread = self.config.base_spread_bps * 1.5  # Conservative fallback

        return OptimizationResult(
            optimal_bid_spread=fallback_spread,
            optimal_ask_spread=fallback_spread,
            symmetric_spread=fallback_spread,
            spread_components=SpreadComponents(
                base_component=self.config.base_spread_bps,
                volatility_component=0.0,
                inventory_component=0.0,
                adverse_selection_component=0.0,
                liquidity_component=0.0,
                regime_component=0.0,
                consciousness_component=0.0,
                total_spread=fallback_spread
            ),
            expected_pnl=0.0,
            expected_risk=0.1,
            sharpe_ratio=0.0,
            fill_probability_bid=0.1,
            fill_probability_ask=0.1,
            confidence_score=0.1,
            computation_time_ms=0.0
        )

    def _update_performance_metrics(self, result: OptimizationResult) -> None:
        """Update performance tracking metrics"""

        self.performance_metrics['total_optimizations'] += 1

        # Update average computation time
        n = self.performance_metrics['total_optimizations']
        old_avg = self.performance_metrics['avg_computation_time_ms']
        new_avg = old_avg + (result.computation_time_ms - old_avg) / n
        self.performance_metrics['avg_computation_time_ms'] = new_avg

        # Track confidence scores
        self.performance_metrics['spread_accuracy_score'] = (
                (self.performance_metrics['spread_accuracy_score'] * (n - 1) + result.confidence_score) / n
        )

        # Track consciousness effectiveness
        consciousness_effectiveness = result.confidence_score * self.config.consciousness_boost
        self.performance_metrics['consciousness_effectiveness'] = (
                (self.performance_metrics['consciousness_effectiveness'] * (n - 1) + consciousness_effectiveness) / n
        )

        # Store result for historical analysis
        self.performance_history.append({
            'timestamp': time.time(),
            'accuracy': result.confidence_score,
            'computation_time': result.computation_time_ms,
            'sharpe_ratio': result.sharpe_ratio,
            'regime': self.current_regime.value
        })

    def calculate_optimal_spreads(self, symbol: str, order_book: Dict, liquidity_metrics: Dict) -> Dict[str, float]:
        """
        FIXED: Calculate optimal spreads using ensemble of algorithms.

        Args:
            symbol: Trading symbol
            order_book: Current order book data
            liquidity_metrics: Liquidity analysis results

        Returns:
            Dictionary with optimal spread recommendations
        """
        try:
            # Convert order book and liquidity metrics to MarketState
            market_state = self._convert_to_market_state(symbol, order_book, liquidity_metrics)

            # Optimize spreads
            result = self.optimize_spread(market_state)

            # Track performance with proper accumulation
            self._track_performance(symbol, 'ensemble', result.symmetric_spread, {
                'volatility': market_state.volatility,
                'volume_ratio': 1.0,
                'liquidity_score': market_state.liquidity_score
            })

            return {
                'bid_spread': result.optimal_bid_spread / 10000,  # Convert bps to decimal
                'ask_spread': result.optimal_ask_spread / 10000,
                'algorithm_results': {
                    'consciousness_enhanced': {
                        'bid_spread': result.optimal_bid_spread / 10000,
                        'ask_spread': result.optimal_ask_spread / 10000
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"Error calculating optimal spreads for {symbol}: {e}")
            return self._get_fallback_spreads()

    def _convert_to_market_state(self, symbol: str, order_book: Dict, liquidity_metrics: Dict) -> MarketState:
        """Convert order book and liquidity metrics to MarketState"""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if not bids or not asks:
                # Return default market state
                return MarketState(
                    mid_price=50000.0,
                    bid_price=49995.0,
                    ask_price=50005.0,
                    volatility=0.02,
                    bid_depth=100.0,
                    ask_depth=100.0,
                    inventory_position=0.0,
                    order_flow_imbalance=0.0,
                    microstructure_signal=0.0,
                    adverse_selection_risk=0.3,
                    liquidity_score=0.5,
                    regime_indicator=0.5
                )

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2.0

            # Calculate bid and ask depth (top 5 levels)
            bid_depth = sum([float(level[1]) for level in bids[:5]])
            ask_depth = sum([float(level[1]) for level in asks[:5]])

            return MarketState(
                mid_price=mid_price,
                bid_price=best_bid,
                ask_price=best_ask,
                volatility=liquidity_metrics.get('volatility', 0.02),
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                inventory_position=0.0,  # Would be provided by inventory manager
                order_flow_imbalance=liquidity_metrics.get('volume_imbalance', 0.0),
                microstructure_signal=liquidity_metrics.get('microstructure_signal', 0.0),
                adverse_selection_risk=liquidity_metrics.get('toxicity_score', 0.3),
                liquidity_score=liquidity_metrics.get('liquidity_score', 0.5),
                regime_indicator=liquidity_metrics.get('regime_indicator', 0.5)
            )

        except Exception as e:
            self.logger.error(f"Error converting to market state: {e}")
            # Return default market state on error
            return MarketState(
                mid_price=50000.0,
                bid_price=49995.0,
                ask_price=50005.0,
                volatility=0.02,
                bid_depth=100.0,
                ask_depth=100.0,
                inventory_position=0.0,
                order_flow_imbalance=0.0,
                microstructure_signal=0.0,
                adverse_selection_risk=0.3,
                liquidity_score=0.5,
                regime_indicator=0.5
            )

    def _track_performance(self, symbol: str, spread_type: str, spread_value: float,
                           market_conditions: Dict) -> None:
        """FIXED: Track spread optimization performance with proper accumulation."""
        try:
            # Ensure we accumulate performance records instead of overwriting
            if symbol not in self.performance_tracking:
                self.performance_tracking[symbol] = []

            performance_record = {
                'timestamp': time.time(),
                'spread_type': spread_type,
                'spread_value': spread_value,
                'market_conditions': market_conditions.copy(),
                'volatility': market_conditions.get('volatility', 0),
                'volume_ratio': market_conditions.get('volume_ratio', 1.0)
            }

            # Append to list instead of overwriting
            self.performance_tracking[symbol].append(performance_record)

            # Keep only last 1000 records to prevent memory issues
            if len(self.performance_tracking[symbol]) > 1000:
                self.performance_tracking[symbol] = self.performance_tracking[symbol][-1000:]

        except Exception as e:
            self.logger.error(f"Error tracking performance for {symbol}: {e}")

    def _get_fallback_spreads(self) -> Dict[str, float]:
        """Get fallback spreads when calculations fail."""
        return {
            'bid_spread': 0.001,  # 0.1%
            'ask_spread': 0.001,  # 0.1%
        }

    def get_performance_metrics(self, symbol: str) -> List[Dict]:
        """Get performance metrics for a symbol."""
        return self.performance_tracking.get(symbol, [])

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization performance report"""

        return {
            'configuration': {
                'consciousness_factor': self.config.consciousness_boost,
                'optimization_method': self.config.optimization_method.value,
                'base_spread_bps': self.config.base_spread_bps,
                'risk_aversion_gamma': self.config.risk_aversion_gamma
            },

            'current_state': {
                'regime': self.current_regime.value,
                'regime_confidence': self.regime_confidence,
                'model_parameters': self.model_parameters
            },

            'performance_metrics': self.performance_metrics,

            'recent_optimizations': len(self.spread_history),
            'cache_hit_ratio': len(self.optimization_cache) / max(self.performance_metrics['total_optimizations'], 1),

            'historical_analysis': {
                'avg_sharpe_ratio': np.mean(
                    [p.get('sharpe_ratio', 0) for p in self.performance_history]) if self.performance_history else 0.0,
                'regime_distribution': {
                    regime.value: sum(1 for p in self.performance_history if p.get('regime') == regime.value) for regime
                    in SpreadRegime}
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Create configuration with consciousness enhancement
    config = SpreadConfig(
        base_spread_bps=5.0,
        consciousness_boost=1.0,
        optimization_method=OptimizationMethod.CONSCIOUSNESS_ENHANCED
    )

    # Initialize optimizer
    optimizer = DynamicSpreadOptimizer(config)

    # Create sample market state
    market_state = MarketState(
        mid_price=50000.0,
        bid_price=49995.0,
        ask_price=50005.0,
        volatility=0.02,
        bid_depth=5.0,
        ask_depth=4.8,
        inventory_position=2.5,
        order_flow_imbalance=0.1,
        microstructure_signal=-0.05,
        adverse_selection_risk=0.3,
        liquidity_score=0.8,
        regime_indicator=0.4
    )

    # Optimize spreads
    result = optimizer.optimize_spread(market_state)

    print(f"Optimization Result:")
    print(f"  Bid Spread: {result.optimal_bid_spread:.2f} bps")
    print(f"  Ask Spread: {result.optimal_ask_spread:.2f} bps")
    print(f"  Symmetric: {result.symmetric_spread:.2f} bps")
    print(f"  Expected PnL: {result.expected_pnl:.6f}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"  Confidence: {result.confidence_score:.3f}")
    print(f"  Computation Time: {result.computation_time_ms:.1f}ms")

    # Get performance report
    report = optimizer.get_optimization_report()
    print(f"\nPerformance Report:")
    for section, data in report.items():
        print(f"  {section}: {data}")