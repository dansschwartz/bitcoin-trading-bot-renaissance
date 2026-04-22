"""
Signal Aggregator
=================
Central hub that owns all advanced microstructure signal instances,
feeds them data, and produces a single composite signal suitable for
integration into the existing Renaissance trading bot signal dictionary.

Usage example::

    aggregator = SignalAggregator()

    # Every order book update:
    aggregator.update_book(bids, asks)

    # Every trade:
    aggregator.update_trade(price, size, side, timestamp)

    # When the bot needs signals:
    composite = aggregator.get_composite_signal()
    signals_dict['microstructure_advanced'] = composite['direction_score']

Dependencies: stdlib + numpy only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from signals.microstructure_signals import (
    MicrostructureSignal,
    OrderBookImbalanceSignal,
    TradeFlowImbalanceSignal,
    LargeOrderDetector,
    SpreadDynamicsSignal,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composite result container
# ---------------------------------------------------------------------------

@dataclass
class CompositeSignal:
    """Result returned by :meth:`SignalAggregator.get_composite_signal`.

    Attributes:
        direction_score: Signed value in ``[-1, 1]``.
            Positive = net long bias, negative = net short bias.
        confidence: Aggregate confidence in ``[0, 1]``.
        active_signal_count: Number of individual signals that fired.
        component_signals: List of the raw :class:`MicrostructureSignal`
            objects that contributed.
        timestamp: Generation time.
    """
    direction_score: float
    confidence: float
    active_signal_count: int
    component_signals: List[MicrostructureSignal] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class SignalAggregator:
    """Owns all advanced microstructure signal instances and aggregates them.

    Parameters:
        weights: Optional mapping from signal name to numeric weight.
            Missing entries fall back to equal weight (``1.0``).
    """

    # Default equal weights keyed by signal_name produced by each class
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "order_book_imbalance": 1.0,
        "trade_flow_imbalance": 1.0,
        "large_order_detector": 1.0,
        "spread_dynamics": 1.0,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        # Signal instances
        self.order_book_imbalance = OrderBookImbalanceSignal()
        self.trade_flow_imbalance = TradeFlowImbalanceSignal()
        self.large_order_detector = LargeOrderDetector()
        self.spread_dynamics = SpreadDynamicsSignal()

        # Configurable weights
        self.weights: Dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        if weights is not None:
            self.weights.update(weights)

        # Cache of most recently computed signals (refreshed on each update)
        self._latest_book_signals: List[Optional[MicrostructureSignal]] = []
        self._latest_trade_signal: Optional[MicrostructureSignal] = None
        self._latest_spread_signal: Optional[MicrostructureSignal] = None

        logger.info(f"SignalAggregator initialised with weights: {self.weights!r}")

    # -- data ingestion -----------------------------------------------------

    def update_book(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> None:
        """Feed a new order-book snapshot to all book-based signals.

        Args:
            bids: ``(price, size)`` descending from best bid.
            asks: ``(price, size)`` ascending from best ask.
        """
        try:
            # Order book imbalance
            obi_signal = self.order_book_imbalance.calculate(bids, asks)

            # Large order detector
            lod_signal = self.large_order_detector.calculate(bids, asks)

            self._latest_book_signals = [obi_signal, lod_signal]

            # Spread dynamics -- extract best bid/ask
            if bids and asks:
                best_bid_price = bids[0][0]
                best_ask_price = asks[0][0]
                self.spread_dynamics.update(
                    best_bid_price, best_ask_price, datetime.now()
                )
                self._latest_spread_signal = self.spread_dynamics.calculate()
        except Exception as e:
            logger.error("SignalAggregator.update_book failed: %s", e)

    def update_trade(
        self,
        price: float,
        size: float,
        side: str,
        timestamp: datetime,
    ) -> None:
        """Feed a trade to the trade-flow signal.

        Args:
            price: Execution price.
            size: Trade size.
            side: ``"buy"`` or ``"sell"``.
            timestamp: Trade time.
        """
        try:
            self.trade_flow_imbalance.add_trade(price, size, side, timestamp)
            self._latest_trade_signal = self.trade_flow_imbalance.calculate()
        except Exception as e:
            logger.error("SignalAggregator.update_trade failed: %s", e)

    # -- composite calculation ----------------------------------------------

    def get_composite_signal(self) -> CompositeSignal:
        """Aggregate all active signals into a single composite.

        Returns:
            A :class:`CompositeSignal` with direction_score in ``[-1, 1]``,
            confidence in ``[0, 1]``, and the list of contributing signals.
        """
        try:
            # Collect all non-None signals
            active_signals: List[MicrostructureSignal] = []
            for sig in self._latest_book_signals:
                if sig is not None:
                    active_signals.append(sig)
            if self._latest_trade_signal is not None:
                active_signals.append(self._latest_trade_signal)
            if self._latest_spread_signal is not None:
                active_signals.append(self._latest_spread_signal)

            if not active_signals:
                return CompositeSignal(
                    direction_score=0.0,
                    confidence=0.0,
                    active_signal_count=0,
                    component_signals=[],
                    timestamp=datetime.now(),
                )

            # Weighted aggregation
            weighted_sum = 0.0
            weight_total = 0.0
            confidence_weighted_sum = 0.0

            for sig in active_signals:
                w = self.weights.get(sig.signal_name, 1.0)
                numeric = sig.to_numeric()  # in [-1, 1]
                weighted_sum += numeric * w
                confidence_weighted_sum += sig.confidence * w
                weight_total += w

            if weight_total > 0.0:
                direction_score = weighted_sum / weight_total
                aggregate_confidence = confidence_weighted_sum / weight_total
            else:
                direction_score = 0.0
                aggregate_confidence = 0.0

            # Clamp
            direction_score = float(np.clip(direction_score, -1.0, 1.0))
            aggregate_confidence = float(np.clip(aggregate_confidence, 0.0, 1.0))

            # Reduce confidence when signals disagree in direction
            if len(active_signals) >= 2:
                directions = [sig.to_numeric() for sig in active_signals]
                signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in directions]
                non_zero_signs = [s for s in signs if s != 0]
                if len(non_zero_signs) >= 2:
                    agreement = abs(sum(non_zero_signs)) / len(non_zero_signs)
                    # Scale confidence by agreement (1.0 = unanimous, 0.0 = split)
                    aggregate_confidence *= (0.5 + 0.5 * agreement)
                    aggregate_confidence = float(
                        np.clip(aggregate_confidence, 0.0, 1.0)
                    )

            return CompositeSignal(
                direction_score=direction_score,
                confidence=aggregate_confidence,
                active_signal_count=len(active_signals),
                component_signals=active_signals,
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error("SignalAggregator.get_composite_signal failed: %s", e)
            return CompositeSignal(
                direction_score=0.0,
                confidence=0.0,
                active_signal_count=0,
                component_signals=[],
                timestamp=datetime.now(),
            )

    # -- convenience --------------------------------------------------------

    def get_signal_dict_entry(self) -> Dict[str, float]:
        """Return a dict suitable for merging into the bot's signal dictionary.

        Example return value::

            {
                'microstructure_advanced': 0.42,
                'microstructure_advanced_confidence': 0.78,
            }
        """
        composite = self.get_composite_signal()
        return {
            "microstructure_advanced": composite.direction_score,
            "microstructure_advanced_confidence": composite.confidence,
        }
