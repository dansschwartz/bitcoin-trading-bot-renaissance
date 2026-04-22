"""Macro Regime Detector — Rule-based macro environment classifier.

Classifies the macro environment into one of four regimes:
  EXPANSION  — Risk-on: SPX rising, VIX calm, yields stable
  LATE_CYCLE — Caution: SPX rising but VIX elevated or yields spiking
  CRISIS     — Risk-off: SPX falling, VIX spiking, flight to safety
  RECOVERY   — Transition: SPX stabilizing after drawdown, VIX declining

Uses yfinance daily data (SPX, VIX, DXY, 10Y yield) cached by MacroDataCache.
3-day hysteresis prevents whipsawing between regimes.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class MacroRegime(str, Enum):
    EXPANSION = "EXPANSION"
    LATE_CYCLE = "LATE_CYCLE"
    CRISIS = "CRISIS"
    RECOVERY = "RECOVERY"
    UNKNOWN = "UNKNOWN"


@dataclass
class MacroSnapshot:
    """Point-in-time macro reading."""
    regime: MacroRegime
    confidence: float
    spx_return_1d: float = 0.0
    vix_level: float = 0.0
    dxy_return_1d: float = 0.0
    yield_level: float = 0.0
    btc_spx_corr: float = 0.0
    timestamp: float = 0.0
    reasons: List[str] = field(default_factory=list)


class MacroRegimeDetector:
    """Rule-based macro regime classifier with hysteresis.

    Thresholds derived from historical market regime studies.
    VIX < 20 = calm, 20-30 = elevated, > 30 = crisis.
    SPX daily return: > +0.3% = strong up, < -0.5% = strong down.
    """

    # Classification thresholds
    VIX_CALM = 20.0
    VIX_ELEVATED = 25.0
    VIX_CRISIS = 30.0

    SPX_STRONG_UP = 0.003     # +0.3% daily
    SPX_WEAK = -0.002         # -0.2% daily
    SPX_STRONG_DOWN = -0.005  # -0.5% daily

    DXY_STRONG_MOVE = 0.005   # 0.5% daily move
    YIELD_SPIKE = 0.10        # 10bp daily move

    # Hysteresis: require N consecutive cycles of new regime before switching
    HYSTERESIS_CYCLES = 3  # At daily granularity = 3 days

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.current_regime: MacroRegime = MacroRegime.UNKNOWN
        self.current_confidence: float = 0.0
        self._pending_regime: Optional[MacroRegime] = None
        self._pending_count: int = 0
        self._history: List[MacroSnapshot] = []
        self._last_classify_time: float = 0.0

    def classify(self, macro_data: Dict[str, float]) -> MacroSnapshot:
        """Classify current macro environment from cached macro data.

        Args:
            macro_data: Dict from MacroDataCache.get() with keys like
                        spx_return_1d, vix_norm, dxy_return_1d, etc.

        Returns:
            MacroSnapshot with regime classification.
        """
        now = time.time()

        # Extract features (default to zero if missing)
        spx_ret = macro_data.get('spx_return_1d', 0.0)
        spx_vs_sma = macro_data.get('spx_vs_sma', 0.0)

        # VIX: stored as (vix - 20) / 20, so actual = vix_norm * 20 + 20
        vix_norm = macro_data.get('vix_norm', 0.0)
        vix_level = vix_norm * 20.0 + 20.0
        vix_change = macro_data.get('vix_change', 0.0)
        vix_extreme = macro_data.get('vix_extreme', 0.0)

        dxy_ret = macro_data.get('dxy_return_1d', 0.0)
        dxy_trend = macro_data.get('dxy_trend', 0.0)

        # Yield: stored as (yield - 4.0) / 2.0, so actual = yield_level * 2 + 4
        yield_norm = macro_data.get('yield_level', 0.0)
        yield_actual = yield_norm * 2.0 + 4.0
        yield_change = macro_data.get('yield_change', 0.0)

        # Score each regime
        scores: Dict[MacroRegime, float] = {
            MacroRegime.EXPANSION: 0.0,
            MacroRegime.LATE_CYCLE: 0.0,
            MacroRegime.CRISIS: 0.0,
            MacroRegime.RECOVERY: 0.0,
        }
        reasons: List[str] = []

        # --- CRISIS signals ---
        if vix_level > self.VIX_CRISIS:
            scores[MacroRegime.CRISIS] += 3.0
            reasons.append(f"VIX={vix_level:.1f}>30 (crisis)")
        elif vix_level > self.VIX_ELEVATED:
            scores[MacroRegime.CRISIS] += 1.0
            scores[MacroRegime.LATE_CYCLE] += 1.5
            reasons.append(f"VIX={vix_level:.1f}>25 (elevated)")

        if spx_ret < self.SPX_STRONG_DOWN:
            scores[MacroRegime.CRISIS] += 2.5
            reasons.append(f"SPX={spx_ret*100:.2f}% (strong down)")
        elif spx_ret < self.SPX_WEAK:
            scores[MacroRegime.CRISIS] += 1.0
            scores[MacroRegime.LATE_CYCLE] += 0.5
            reasons.append(f"SPX={spx_ret*100:.2f}% (weak)")

        if vix_change > 0.10:  # VIX up >10% in a day
            scores[MacroRegime.CRISIS] += 1.5
            reasons.append(f"VIX spike +{vix_change*100:.1f}%")

        # --- EXPANSION signals ---
        if vix_level < self.VIX_CALM:
            scores[MacroRegime.EXPANSION] += 2.0
            reasons.append(f"VIX={vix_level:.1f}<20 (calm)")

        if spx_ret > self.SPX_STRONG_UP:
            scores[MacroRegime.EXPANSION] += 2.0
            reasons.append(f"SPX=+{spx_ret*100:.2f}% (strong up)")

        if spx_vs_sma > 0.01:  # SPX above 5-day SMA
            scores[MacroRegime.EXPANSION] += 1.0

        if abs(dxy_ret) < 0.002:  # Stable dollar = risk-on friendly
            scores[MacroRegime.EXPANSION] += 0.5

        # --- LATE_CYCLE signals ---
        if spx_ret > 0 and vix_level > self.VIX_CALM:
            scores[MacroRegime.LATE_CYCLE] += 2.0
            reasons.append("SPX up but VIX elevated")

        if abs(yield_change) > self.YIELD_SPIKE:
            scores[MacroRegime.LATE_CYCLE] += 1.5
            reasons.append(f"Yield move {yield_change:+.2f}")

        if dxy_ret > self.DXY_STRONG_MOVE:
            scores[MacroRegime.LATE_CYCLE] += 1.0
            reasons.append(f"DXY strength +{dxy_ret*100:.2f}%")

        # --- RECOVERY signals ---
        if spx_ret > 0 and vix_change < -0.05:
            scores[MacroRegime.RECOVERY] += 2.0
            reasons.append("SPX up + VIX declining")

        if spx_vs_sma < -0.005 and spx_ret > 0:
            # Below SMA but bouncing
            scores[MacroRegime.RECOVERY] += 1.5
            reasons.append("SPX below SMA but bouncing")

        if self.current_regime == MacroRegime.CRISIS and vix_level < self.VIX_CRISIS:
            scores[MacroRegime.RECOVERY] += 1.0
            reasons.append("Exiting crisis (VIX declining)")

        # --- Select winner ---
        if not macro_data:
            raw_regime = MacroRegime.UNKNOWN
            confidence = 0.0
            reasons = ["No macro data available"]
        else:
            raw_regime = max(scores, key=scores.get)  # type: ignore[arg-type]
            total = sum(scores.values())
            confidence = scores[raw_regime] / total if total > 0 else 0.0

        # --- Hysteresis ---
        confirmed_regime = self._apply_hysteresis(raw_regime)

        snapshot = MacroSnapshot(
            regime=confirmed_regime,
            confidence=round(confidence, 3),
            spx_return_1d=spx_ret,
            vix_level=vix_level,
            dxy_return_1d=dxy_ret,
            yield_level=yield_actual,
            timestamp=now,
            reasons=reasons,
        )

        self._history.append(snapshot)
        if len(self._history) > 200:
            self._history = self._history[-100:]

        self._last_classify_time = now
        return snapshot

    def _apply_hysteresis(self, raw_regime: MacroRegime) -> MacroRegime:
        """Require HYSTERESIS_CYCLES consecutive signals before switching."""
        if raw_regime == self.current_regime:
            self._pending_regime = None
            self._pending_count = 0
            return self.current_regime

        if raw_regime == MacroRegime.UNKNOWN:
            return self.current_regime if self.current_regime != MacroRegime.UNKNOWN else raw_regime

        if self._pending_regime == raw_regime:
            self._pending_count += 1
        else:
            self._pending_regime = raw_regime
            self._pending_count = 1

        if self._pending_count >= self.HYSTERESIS_CYCLES:
            old = self.current_regime
            self.current_regime = raw_regime
            self.current_confidence = 0.0  # Will be set by next classify
            self._pending_regime = None
            self._pending_count = 0
            self.logger.info(
                f"MACRO REGIME CHANGE: {old.value} -> {raw_regime.value} "
                f"(after {self.HYSTERESIS_CYCLES} consecutive signals)"
            )
            return raw_regime

        self.logger.debug(
            f"Macro regime pending: {raw_regime.value} "
            f"({self._pending_count}/{self.HYSTERESIS_CYCLES})"
        )
        return self.current_regime

    def get_state(self) -> Dict:
        """Return serializable state for dashboard/logging."""
        last = self._history[-1] if self._history else None
        return {
            "regime": self.current_regime.value,
            "confidence": round(self.current_confidence, 3) if self.current_confidence else 0.0,
            "pending": self._pending_regime.value if self._pending_regime else None,
            "pending_count": self._pending_count,
            "hysteresis_needed": self.HYSTERESIS_CYCLES,
            "vix_level": last.vix_level if last else 0.0,
            "spx_return_1d": last.spx_return_1d if last else 0.0,
            "reasons": last.reasons if last else [],
            "last_classify": self._last_classify_time,
        }

    @property
    def history(self) -> List[MacroSnapshot]:
        return self._history
