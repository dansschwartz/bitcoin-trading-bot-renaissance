"""
Real-time temporal bias provider.

Thin wrapper around TemporalAnalyzer that integrates with the arb detectors.
Detectors call get_bias() before deciding whether to act on a signal.
"""

import logging
from typing import Optional
from arbitrage.analytics.temporal_analyzer import TemporalAnalyzer

logger = logging.getLogger("arb.temporal_bias")


class TemporalBias:
    """
    Provides real-time temporal bias weights to arb detectors.

    Usage in detector:
        bias = self.temporal_bias.get_bias("cross_exchange", "BTC/USDT")
        if bias < 0.5:
            logger.debug("Low temporal bias (%.2f), skipping", bias)
            return  # Skip this signal
        adjusted_confidence = signal.confidence * bias
    """

    def __init__(self, analyzer: TemporalAnalyzer):
        self._analyzer = analyzer

    def get_bias(self, strategy: str, pair: Optional[str] = None) -> float:
        """
        Get current temporal bias for strategy/pair.

        Returns float in [0.2, 2.0]:
        - 0.2 = strongly unfavorable time window
        - 1.0 = neutral / average
        - 2.0 = strongly favorable time window
        """
        return self._analyzer.get_current_bias(strategy, pair)

    def should_skip(
        self, strategy: str, pair: Optional[str] = None, threshold: float = 0.4
    ) -> bool:
        """
        Convenience: should we skip trading entirely in this window?

        Default threshold of 0.4 means: skip if this time window is
        historically less than 40% as profitable as average.
        """
        return self.get_bias(strategy, pair) < threshold
