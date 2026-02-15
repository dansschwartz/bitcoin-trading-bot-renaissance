"""
Renaissance Trading Bot - Signals Module
Liquidation cascade detection, derivative market signal generation,
and advanced microstructure signals.
"""

# Liquidation detector (may not be present yet -- import defensively)
try:
    from signals.liquidation_detector import (
        LiquidationCascadeDetector,
        CascadeRiskSignal,
    )
except ImportError:
    LiquidationCascadeDetector = None  # type: ignore[assignment,misc]
    CascadeRiskSignal = None  # type: ignore[assignment,misc]

# Advanced microstructure signals
from signals.microstructure_signals import (
    MicrostructureSignal,
    OrderBookImbalanceSignal,
    TradeFlowImbalanceSignal,
    LargeOrderDetector,
    SpreadDynamicsSignal,
)

from signals.signal_aggregator import (
    CompositeSignal,
    SignalAggregator,
)

__all__ = [
    # Liquidation detector (legacy)
    "LiquidationCascadeDetector",
    "CascadeRiskSignal",
    # Microstructure signals
    "MicrostructureSignal",
    "OrderBookImbalanceSignal",
    "TradeFlowImbalanceSignal",
    "LargeOrderDetector",
    "SpreadDynamicsSignal",
    # Aggregator
    "CompositeSignal",
    "SignalAggregator",
]
