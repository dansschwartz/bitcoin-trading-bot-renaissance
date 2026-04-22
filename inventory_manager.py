"""
Advanced Inventory Manager with Consciousness Enhancement
Manages trading inventory, risk assessment, and rebalancing signals.
"""

import logging
import time
import math
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class RebalanceAction(Enum):
    """Rebalancing action types"""
    HOLD = "hold"
    REDUCE = "reduce"
    INCREASE = "increase"
    REBALANCE = "rebalance"
    EMERGENCY_REDUCE = "emergency_reduce"


@dataclass
class InventoryConfig:
    """Configuration for inventory management"""
    max_position_size: float = 10.0
    consciousness_boost: float = 1.0
    target_inventory: float = 0.0
    risk_limit: float = 0.8
    max_drawdown_limit: float = 0.15
    rebalance_threshold: float = 0.6


@dataclass
class InventoryState:
    """Current inventory state"""
    base_position: float = 0.0
    quote_position: float = 0.0
    adverse_selection_exposure: float = 0.0
    microstructure_alpha: float = 0.0
    timestamp: float = field(default_factory=time.time)
    total_value: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    var_1d: float = 0.0
    expected_shortfall: float = 0.0
    inventory_risk_score: float = 0.0
    max_drawdown: float = 0.0
    position_concentration: float = 0.0
    liquidity_risk: float = 0.0


class AdvancedInventoryManager:
    """
    Advanced Inventory Manager with Consciousness Enhancement

    Tracks positions, assesses risk, calculates optimal inventory,
    and generates rebalancing signals.
    """

    def __init__(self, config: InventoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_state = InventoryState()
        self.state_history: List[InventoryState] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def update_inventory(self, base_change: float, quote_change: float,
                         market_price: float, market_data: Dict[str, Any]) -> InventoryState:
        """Update inventory positions with consciousness enhancement"""
        consciousness = self.config.consciousness_boost

        # Update positions
        self.current_state.base_position += base_change
        self.current_state.quote_position += quote_change
        self.current_state.timestamp = time.time()

        # Compute total value
        self.current_state.total_value = (
            self.current_state.base_position * market_price +
            self.current_state.quote_position
        )

        # Compute adverse selection exposure using consciousness enhancement
        volatility = market_data.get('volatility', 0.02)
        order_flow = market_data.get('order_flow_imbalance', 0.0)
        spread = market_data.get('spread', 0.0001)

        self.current_state.adverse_selection_exposure = (
            abs(self.current_state.base_position) *
            volatility *
            (1.0 + abs(order_flow)) *
            consciousness
        )

        # Compute microstructure alpha
        volume_imbalance = market_data.get('volume_imbalance', 0.0)
        tick_direction = market_data.get('tick_direction', 0)
        trade_size_ratio = market_data.get('trade_size_ratio', 1.0)
        time_between = market_data.get('time_between_trades', 1.0)

        self.current_state.microstructure_alpha = (
            order_flow * 0.3 +
            volume_imbalance * 0.2 +
            tick_direction * 0.1 * spread +
            (trade_size_ratio - 1.0) * 0.15 +
            (1.0 / max(time_between, 0.01)) * 0.001
        ) * consciousness

        # Store in history
        self.state_history.append(InventoryState(
            base_position=self.current_state.base_position,
            quote_position=self.current_state.quote_position,
            adverse_selection_exposure=self.current_state.adverse_selection_exposure,
            microstructure_alpha=self.current_state.microstructure_alpha,
            timestamp=self.current_state.timestamp,
            total_value=self.current_state.total_value,
        ))

        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]

        # Record trade
        self.trade_history.append({
            'base_change': base_change,
            'quote_change': quote_change,
            'market_price': market_price,
            'timestamp': time.time(),
        })

        return self.current_state

    def calculate_optimal_inventory(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal inventory position with consciousness enhancement"""
        consciousness = self.config.consciousness_boost
        current_pos = self.current_state.base_position
        target = self.config.target_inventory

        volatility = market_data.get('volatility', 0.02)
        order_flow = market_data.get('order_flow_imbalance', 0.0)
        mid_price = market_data.get('mid_price', 50000.0)

        # Risk-adjusted optimal position
        risk_aversion = 0.01 / max(consciousness, 0.01)
        vol_penalty = volatility * abs(current_pos) * risk_aversion

        # Flow-based adjustment
        flow_signal = order_flow * consciousness * 0.5

        # Optimal position considers target, flow, and risk
        optimal = target + flow_signal - vol_penalty * np.sign(current_pos)
        optimal = np.clip(optimal, -self.config.max_position_size, self.config.max_position_size)

        position_delta = optimal - current_pos

        # Confidence based on data quality and consciousness
        data_quality = min(1.0, len(self.state_history) / 50.0)
        confidence = min(1.0, data_quality * 0.7 + (1.0 - volatility * 10) * 0.3)
        confidence = max(0.0, min(confidence * consciousness, 1.0))

        return {
            'optimal_position': float(optimal),
            'current_position': float(current_pos),
            'position_delta': float(position_delta),
            'confidence': float(confidence),
            'consciousness_applied': True,
            'risk_aversion': risk_aversion,
            'flow_signal': flow_signal,
        }

    def assess_inventory_risk(self, market_data: Dict[str, Any]) -> RiskMetrics:
        """Assess comprehensive inventory risk"""
        consciousness = self.config.consciousness_boost
        pos = self.current_state.base_position
        volatility = market_data.get('volatility', 0.02)
        mid_price = market_data.get('mid_price', 50000.0)

        # Value-at-Risk (1-day, 95% confidence)
        position_value = abs(pos) * mid_price
        var_1d = position_value * volatility * 1.645  # 95% VaR

        # Expected shortfall (CVaR)
        expected_shortfall = var_1d * 1.4  # Approximate ES

        # Inventory risk score (0 to 1)
        position_ratio = abs(pos) / max(self.config.max_position_size, 0.01)
        vol_component = min(1.0, volatility * 20)
        inventory_risk_score = min(1.0, position_ratio * 0.6 + vol_component * 0.4)

        # Max drawdown from history
        max_drawdown = 0.0
        if len(self.state_history) > 1:
            values = [s.total_value for s in self.state_history if s.total_value != 0]
            if values:
                peak = values[0]
                for v in values:
                    if v > peak:
                        peak = v
                    dd = (peak - v) / max(abs(peak), 1.0)
                    if dd > max_drawdown:
                        max_drawdown = dd

        # Position concentration
        position_concentration = position_ratio

        # Liquidity risk
        liquidity_factor = market_data.get('liquidity_factor', 0.8)
        liquidity_risk = max(0.0, 1.0 - liquidity_factor) * position_ratio

        return RiskMetrics(
            var_1d=var_1d,
            expected_shortfall=expected_shortfall,
            inventory_risk_score=inventory_risk_score,
            max_drawdown=max_drawdown,
            position_concentration=position_concentration,
            liquidity_risk=liquidity_risk,
        )

    def generate_rebalance_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rebalancing signal with consciousness enhancement"""
        consciousness = self.config.consciousness_boost
        pos = self.current_state.base_position
        max_pos = self.config.max_position_size
        target = self.config.target_inventory

        # Position utilization
        utilization = abs(pos) / max(max_pos, 0.01)
        deviation = abs(pos - target) / max(max_pos, 0.01)

        # Urgency based on position size and risk
        risk_metrics = self.assess_inventory_risk(market_data)
        urgency = min(1.0, (
            utilization * 0.4 +
            risk_metrics.inventory_risk_score * 0.3 +
            deviation * 0.3
        ) * consciousness)

        # Determine action
        if utilization > 0.9:
            action = RebalanceAction.EMERGENCY_REDUCE
        elif utilization > self.config.rebalance_threshold:
            action = RebalanceAction.REDUCE
        elif deviation > 0.5:
            if abs(pos) < abs(target):
                action = RebalanceAction.INCREASE
            else:
                action = RebalanceAction.REBALANCE
        else:
            action = RebalanceAction.HOLD

        # Recommended size
        if action in (RebalanceAction.REDUCE, RebalanceAction.EMERGENCY_REDUCE):
            recommended_size = abs(pos - target) * min(1.0, urgency)
        elif action == RebalanceAction.INCREASE:
            recommended_size = abs(target - pos) * 0.5
        elif action == RebalanceAction.REBALANCE:
            recommended_size = abs(pos - target)
        else:
            recommended_size = 0.0

        return {
            'action': action,
            'recommended_size': float(recommended_size),
            'urgency_score': float(urgency),
            'consciousness_enhanced': True,
            'current_utilization': float(utilization),
            'risk_score': float(risk_metrics.inventory_risk_score),
        }
