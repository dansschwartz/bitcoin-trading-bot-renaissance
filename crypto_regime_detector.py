"""Crypto Regime Detector — Crypto-specific market regime classifier.

Classifies the crypto market into one of four regimes:
  BULL_TREND    — Strong uptrend: EMA stack bullish, funding positive, OI rising
  DISTRIBUTION  — Topping: Price high but momentum fading, divergences appearing
  CRASH         — Severe drawdown: Price below key EMAs, funding extreme, VIX spiking
  ACCUMULATION  — Bottoming: Price stabilizing, smart money accumulating

Uses BTC price data (EMAs, drawdown, RSI) and derivatives data (funding, OI, L/S ratio).
36-cycle hysteresis (~3 hours at 5-min intervals) prevents whipsawing.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class CryptoRegime(str, Enum):
    BULL_TREND = "BULL_TREND"
    DISTRIBUTION = "DISTRIBUTION"
    CRASH = "CRASH"
    ACCUMULATION = "ACCUMULATION"
    UNKNOWN = "UNKNOWN"


@dataclass
class CryptoSnapshot:
    """Point-in-time crypto regime reading."""
    regime: CryptoRegime
    confidence: float
    btc_price: float = 0.0
    ema_21: float = 0.0
    ema_55: float = 0.0
    ema_200: float = 0.0
    drawdown_24h: float = 0.0
    rsi_14: float = 50.0
    funding_rate: float = 0.0
    oi_change_1h: float = 0.0
    timestamp: float = 0.0
    reasons: List[str] = field(default_factory=list)


class CryptoRegimeDetector:
    """Crypto-specific regime detector with hysteresis.

    EMA stack: price > EMA21 > EMA55 > EMA200 = bullish
    Drawdown thresholds: >5% = correction, >10% = crash
    Funding rate: >0.01% = overleveraged longs, <-0.005% = shorts dominant
    """

    # EMA periods (in 5-min bars)
    EMA_FAST = 21
    EMA_MED = 55
    EMA_SLOW = 200

    # Drawdown thresholds
    DRAWDOWN_CORRECTION = -0.05  # -5%
    DRAWDOWN_CRASH = -0.10       # -10%

    # RSI thresholds
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    # Funding rate thresholds (per 8h)
    FUNDING_HIGH = 0.0003    # 0.03% per 8h = bullish/overleveraged
    FUNDING_EXTREME = 0.001  # 0.1% per 8h = very overleveraged
    FUNDING_NEGATIVE = -0.0001  # Shorts dominant

    # OI change thresholds (1h)
    OI_SURGE = 0.03    # +3% in 1h
    OI_DECLINE = -0.03  # -3% in 1h (deleveraging)

    # Hysteresis: 36 cycles * 5 min = 3 hours
    HYSTERESIS_CYCLES = 36

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.current_regime: CryptoRegime = CryptoRegime.UNKNOWN
        self.current_confidence: float = 0.0
        self._pending_regime: Optional[CryptoRegime] = None
        self._pending_count: int = 0
        self._history: List[CryptoSnapshot] = []
        self._last_classify_time: float = 0.0

    def classify(
        self,
        price_df,
        derivatives_data: Optional[Dict] = None,
    ) -> CryptoSnapshot:
        """Classify crypto regime from BTC price data and derivatives.

        Args:
            price_df: DataFrame with columns [close, high, low, volume] (5-min bars).
                      Needs >= 200 rows for full EMA stack.
            derivatives_data: Dict with funding_rate, open_interest, long_short_ratio.

        Returns:
            CryptoSnapshot with regime classification.
        """
        now = time.time()
        deriv = derivatives_data or {}

        # --- Compute EMAs ---
        if price_df is None or len(price_df) < self.EMA_FAST:
            return CryptoSnapshot(
                regime=self.current_regime if self.current_regime != CryptoRegime.UNKNOWN
                    else CryptoRegime.UNKNOWN,
                confidence=0.0,
                timestamp=now,
                reasons=["Insufficient data"],
            )

        close = price_df['close'].values.astype(float)
        btc_price = float(close[-1])

        ema_21 = self._ema(close, self.EMA_FAST)
        ema_55 = self._ema(close, min(self.EMA_MED, len(close)))
        ema_200 = self._ema(close, min(self.EMA_SLOW, len(close))) if len(close) >= 100 else ema_55

        # --- 24h drawdown ---
        bars_24h = min(288, len(close))  # 288 * 5min = 24h
        high_24h = float(np.max(close[-bars_24h:]))
        drawdown_24h = (btc_price / high_24h - 1.0) if high_24h > 0 else 0.0

        # --- RSI-14 ---
        rsi_14 = self._compute_rsi(close, 14)

        # --- Derivatives ---
        funding_rate = float(deriv.get('funding_rate', 0.0) or 0.0)
        oi_change_1h = float(deriv.get('oi_change_1h', 0.0) or 0.0)
        ls_ratio = float(deriv.get('long_short_ratio', 1.0) or 1.0)

        # --- Score each regime ---
        scores: Dict[CryptoRegime, float] = {
            CryptoRegime.BULL_TREND: 0.0,
            CryptoRegime.DISTRIBUTION: 0.0,
            CryptoRegime.CRASH: 0.0,
            CryptoRegime.ACCUMULATION: 0.0,
        }
        reasons: List[str] = []

        # --- BULL_TREND signals ---
        # EMA stack: price > EMA21 > EMA55 > EMA200
        if btc_price > ema_21 > ema_55:
            scores[CryptoRegime.BULL_TREND] += 2.5
            reasons.append("EMA stack bullish")
            if ema_55 > ema_200:
                scores[CryptoRegime.BULL_TREND] += 1.0
                reasons.append("Full EMA alignment")

        if rsi_14 > 55 and rsi_14 < self.RSI_OVERBOUGHT:
            scores[CryptoRegime.BULL_TREND] += 1.0

        if funding_rate > 0 and funding_rate < self.FUNDING_HIGH:
            scores[CryptoRegime.BULL_TREND] += 0.5
            reasons.append(f"Funding healthy +{funding_rate*100:.4f}%")

        if oi_change_1h > 0.01:  # Rising OI in uptrend
            scores[CryptoRegime.BULL_TREND] += 0.5

        # --- DISTRIBUTION signals ---
        if rsi_14 > self.RSI_OVERBOUGHT:
            scores[CryptoRegime.DISTRIBUTION] += 2.0
            reasons.append(f"RSI={rsi_14:.0f} overbought")

        if funding_rate > self.FUNDING_EXTREME:
            scores[CryptoRegime.DISTRIBUTION] += 2.5
            reasons.append(f"Funding extreme +{funding_rate*100:.4f}%")
        elif funding_rate > self.FUNDING_HIGH:
            scores[CryptoRegime.DISTRIBUTION] += 1.5
            reasons.append(f"Funding elevated +{funding_rate*100:.4f}%")

        if btc_price > ema_21 and btc_price < ema_55:
            # Price between short and medium EMA — losing momentum
            scores[CryptoRegime.DISTRIBUTION] += 1.5
            reasons.append("Price losing EMA support")

        if ls_ratio > 2.0:
            scores[CryptoRegime.DISTRIBUTION] += 1.0
            reasons.append(f"L/S ratio={ls_ratio:.2f} (crowded longs)")

        # --- CRASH signals ---
        if drawdown_24h < self.DRAWDOWN_CRASH:
            scores[CryptoRegime.CRASH] += 3.0
            reasons.append(f"Drawdown {drawdown_24h*100:.1f}% (crash)")
        elif drawdown_24h < self.DRAWDOWN_CORRECTION:
            scores[CryptoRegime.CRASH] += 1.5
            reasons.append(f"Drawdown {drawdown_24h*100:.1f}% (correction)")

        if btc_price < ema_55 and btc_price < ema_200:
            scores[CryptoRegime.CRASH] += 2.0
            reasons.append("Below EMA55 and EMA200")
        elif btc_price < ema_55:
            scores[CryptoRegime.CRASH] += 1.0

        if oi_change_1h < self.OI_DECLINE:
            scores[CryptoRegime.CRASH] += 1.5
            reasons.append(f"OI deleveraging {oi_change_1h*100:.1f}%")

        if rsi_14 < self.RSI_OVERSOLD:
            scores[CryptoRegime.CRASH] += 1.0
            reasons.append(f"RSI={rsi_14:.0f} oversold")

        # --- ACCUMULATION signals ---
        if rsi_14 < 40 and rsi_14 > self.RSI_OVERSOLD:
            scores[CryptoRegime.ACCUMULATION] += 1.5

        if drawdown_24h < self.DRAWDOWN_CORRECTION and btc_price > ema_21:
            # Bouncing from correction
            scores[CryptoRegime.ACCUMULATION] += 2.0
            reasons.append("Bouncing from correction, above EMA21")

        if funding_rate < self.FUNDING_NEGATIVE:
            scores[CryptoRegime.ACCUMULATION] += 1.5
            reasons.append(f"Negative funding {funding_rate*100:.4f}% (shorts paying)")

        if self.current_regime == CryptoRegime.CRASH and btc_price > ema_21:
            scores[CryptoRegime.ACCUMULATION] += 1.5
            reasons.append("Recovering from crash, above EMA21")

        if oi_change_1h > self.OI_SURGE and btc_price < ema_55:
            # OI building at low prices = accumulation
            scores[CryptoRegime.ACCUMULATION] += 1.0
            reasons.append("OI building at low prices")

        # --- Select winner ---
        raw_regime = max(scores, key=scores.get)  # type: ignore[arg-type]
        total = sum(scores.values())
        confidence = scores[raw_regime] / total if total > 0 else 0.0

        # --- Hysteresis ---
        confirmed_regime = self._apply_hysteresis(raw_regime)

        snapshot = CryptoSnapshot(
            regime=confirmed_regime,
            confidence=round(confidence, 3),
            btc_price=btc_price,
            ema_21=round(ema_21, 2),
            ema_55=round(ema_55, 2),
            ema_200=round(ema_200, 2),
            drawdown_24h=round(drawdown_24h, 4),
            rsi_14=round(rsi_14, 1),
            funding_rate=funding_rate,
            oi_change_1h=oi_change_1h,
            timestamp=now,
            reasons=reasons,
        )

        self._history.append(snapshot)
        if len(self._history) > 500:
            self._history = self._history[-250:]

        self.current_confidence = confidence
        self._last_classify_time = now
        return snapshot

    def _apply_hysteresis(self, raw_regime: CryptoRegime) -> CryptoRegime:
        """Require HYSTERESIS_CYCLES consecutive signals before switching."""
        if raw_regime == self.current_regime:
            self._pending_regime = None
            self._pending_count = 0
            return self.current_regime

        if raw_regime == CryptoRegime.UNKNOWN:
            return self.current_regime if self.current_regime != CryptoRegime.UNKNOWN else raw_regime

        if self._pending_regime == raw_regime:
            self._pending_count += 1
        else:
            self._pending_regime = raw_regime
            self._pending_count = 1

        if self._pending_count >= self.HYSTERESIS_CYCLES:
            old = self.current_regime
            self.current_regime = raw_regime
            self.current_confidence = 0.0
            self._pending_regime = None
            self._pending_count = 0
            self.logger.info(
                f"CRYPTO REGIME CHANGE: {old.value} -> {raw_regime.value} "
                f"(after {self.HYSTERESIS_CYCLES} consecutive cycles / "
                f"~{self.HYSTERESIS_CYCLES * 5 / 60:.0f}h)"
            )
            return raw_regime

        self.logger.debug(
            f"Crypto regime pending: {raw_regime.value} "
            f"({self._pending_count}/{self.HYSTERESIS_CYCLES})"
        )
        return self.current_regime

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Compute exponential moving average, return last value."""
        if len(data) < period:
            return float(np.mean(data))
        alpha = 2.0 / (period + 1)
        ema = float(data[0])
        for val in data[1:]:
            ema = alpha * float(val) + (1 - alpha) * ema
        return ema

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        """Compute RSI from close prices."""
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def get_state(self) -> Dict:
        """Return serializable state for dashboard/logging."""
        last = self._history[-1] if self._history else None
        return {
            "regime": self.current_regime.value,
            "confidence": round(self.current_confidence, 3),
            "pending": self._pending_regime.value if self._pending_regime else None,
            "pending_count": self._pending_count,
            "hysteresis_needed": self.HYSTERESIS_CYCLES,
            "btc_price": last.btc_price if last else 0.0,
            "ema_21": last.ema_21 if last else 0.0,
            "ema_55": last.ema_55 if last else 0.0,
            "ema_200": last.ema_200 if last else 0.0,
            "drawdown_24h": last.drawdown_24h if last else 0.0,
            "rsi_14": last.rsi_14 if last else 50.0,
            "funding_rate": last.funding_rate if last else 0.0,
            "reasons": last.reasons if last else [],
            "last_classify": self._last_classify_time,
        }

    @property
    def history(self) -> List[CryptoSnapshot]:
        return self._history
