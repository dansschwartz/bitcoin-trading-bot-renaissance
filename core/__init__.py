"""
Core modules for the Renaissance Trading Bot
=============================================

A1 - DevilTracker:        Execution quality measurement (theoretical vs actual P&L).
A2 - PortfolioEngine:     Self-correcting portfolio engine (target vs actual reconciliation).
A3 - KellyPositionSizer:  Fractional Kelly position sizing from historical trade statistics.
E1 - SignalThrottle:      Daily P&L-based signal health monitor and auto-throttle.
E2 - LeverageManager:     Consistency-based leverage control.
"""

from core.devil_tracker import DevilTracker, TradeExecution
from core.portfolio_engine import (
    PortfolioEngine,
    PortfolioTarget,
    PortfolioActual,
    CorrectionOrder,
)
from core.kelly_position_sizer import KellyPositionSizer
from core.signal_throttle import SignalThrottle
from core.leverage_manager import LeverageManager

__all__ = [
    "DevilTracker",
    "TradeExecution",
    "PortfolioEngine",
    "PortfolioTarget",
    "PortfolioActual",
    "CorrectionOrder",
    "KellyPositionSizer",
    "SignalThrottle",
    "LeverageManager",
]
