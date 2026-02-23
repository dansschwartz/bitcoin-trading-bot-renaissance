"""
Polymarket Bridge — Converts Renaissance ML predictions to Polymarket signals.

Writes a signal JSON file that the Node.js Polymarket trader reads.
Signal file: ~/revenue-engine/data/output/renaissance-signal.json

Architecture:
  Renaissance Bot (Python) --writes JSON--> Revenue Engine (Node.js) --trades--> Polymarket
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Path to the revenue-engine's signal consumption point
SIGNAL_FILE = Path.home() / "revenue-engine" / "data" / "output" / "renaissance-signal.json"


class PolymarketBridge:
    """
    Translates Renaissance ML ensemble predictions into Polymarket trading signals.

    Every 5-minute cycle:
    1. Collects the latest ML prediction for BTC-USD
    2. Collects regime state, agreement score, breakout score
    3. Computes a confidence-weighted direction signal
    4. Writes JSON for the Node.js Polymarket trader to consume
    """

    def __init__(
        self,
        signal_file: Optional[Path] = None,
        min_prediction: float = 0.03,
        min_agreement: float = 0.55,
        observation_mode: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.signal_file = signal_file or SIGNAL_FILE
        self.min_prediction = min_prediction
        self.min_agreement = min_agreement
        self.observation_mode = observation_mode
        self.logger = logger or logging.getLogger(__name__)
        self._signal_count: int = 0
        self._skip_count: int = 0

    def generate_signal(
        self,
        prediction: float,
        agreement: float,
        regime: Optional[str] = None,
        breakout_score: float = 0.0,
        btc_price: float = 0.0,
        model_confidences: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a Polymarket-compatible signal from ML predictions.

        Args:
            prediction: -1 to +1 from ML ensemble (positive = bullish)
            agreement: 0-1 model agreement ratio
            regime: Current HMM regime label
            breakout_score: 0-100 from breakout scanner
            btc_price: Current BTC price
            model_confidences: Per-model prediction values

        Returns:
            Signal dict (also written to JSON file for Node.js consumption).
        """
        direction = "UP" if prediction > 0 else "DOWN"
        abs_pred = abs(prediction)

        # Skip if signal too weak
        skip_reason: Optional[str] = None
        if abs_pred < self.min_prediction:
            skip_reason = f"Prediction {abs_pred:.4f} below {self.min_prediction} threshold"
        elif agreement < self.min_agreement:
            skip_reason = f"Agreement {agreement:.2f} below {self.min_agreement} threshold"

        # Compute confidence (0-95)
        # |prediction| of 0.05 -> 20, 0.10 -> 40, 0.15 -> 60, 0.25 -> 100 (capped 95)
        confidence = min(95.0, abs_pred * 400)

        # Agreement boost/penalty
        if agreement >= 0.80:
            confidence *= 1.20
        elif agreement >= 0.70:
            confidence *= 1.10
        elif agreement < 0.55:
            confidence *= 0.70

        # Regime alignment
        regime_aligned = False
        if regime:
            bearish_regimes = ('bear_trending', 'bear_mean_reverting', 'high_volatility')
            bullish_regimes = ('bull_trending', 'bull_mean_reverting')
            if (regime in bearish_regimes and direction == "DOWN") or \
               (regime in bullish_regimes and direction == "UP"):
                confidence *= 1.15
                regime_aligned = True
            elif (regime in bearish_regimes and direction == "UP") or \
                 (regime in bullish_regimes and direction == "DOWN"):
                confidence *= 0.75

        # Breakout boost
        if breakout_score >= 50:
            confidence *= 1.10

        confidence = min(95.0, max(0.0, confidence))

        # Count agreeing/disagreeing models
        mc = model_confidences or {}
        agreeing = sum(1 for v in mc.values() if (v > 0) == (prediction > 0))
        disagreeing = sum(1 for v in mc.values() if (v > 0) != (prediction > 0) and v != 0)

        signal = {
            "source": "renaissance_ml",
            "direction": direction,
            "confidence": round(confidence),
            "rawScore": round(prediction, 6),
            "agreement": round(agreement, 4),
            "regime": regime or "unknown",
            "regimeAligned": regime_aligned,
            "breakoutScore": round(breakout_score, 1),
            "btcPrice": round(btc_price, 2),
            "skipReason": skip_reason,
            "observationMode": self.observation_mode,
            "meta": {
                "activeSignals": len(mc) if mc else 7,
                "agreeingSignals": agreeing,
                "disagreingSignals": disagreeing,
                "latestPrice": btc_price,
                "modelConfidences": {k: round(v, 6) for k, v in mc.items()},
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Write to JSON file
        try:
            self.signal_file.parent.mkdir(parents=True, exist_ok=True)
            self.signal_file.write_text(json.dumps(signal, indent=2))

            if skip_reason:
                self._skip_count += 1
                self.logger.debug(f"POLYMARKET SIGNAL: SKIP — {skip_reason}")
            else:
                self._signal_count += 1
                self.logger.info(
                    f"POLYMARKET SIGNAL: {direction} conf={confidence:.0f}% "
                    f"pred={prediction:+.4f} agree={agreement:.2f} "
                    f"regime={regime} aligned={regime_aligned} "
                    f"{'[OBSERVATION]' if self.observation_mode else '[LIVE]'}"
                )
        except Exception as e:
            self.logger.error(f"Failed to write Polymarket signal: {e}")

        return signal

    def get_stats(self) -> Dict[str, Any]:
        """Return bridge statistics."""
        return {
            "signals_emitted": self._signal_count,
            "signals_skipped": self._skip_count,
            "observation_mode": self.observation_mode,
            "signal_file": str(self.signal_file),
        }
