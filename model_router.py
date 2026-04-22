"""Model Router — Maps regime tuple to model configuration.

Routes (macro_regime, crypto_regime, micro_regime) -> ModelConfig via
a priority-ordered lookup table with wildcard matching.

Phase 1: Observation mode only — logs recommended model but doesn't enforce.
Phase 2 (future): Active routing, adjusting signal weights per regime combo.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from macro_regime_detector import MacroRegime
from crypto_regime_detector import CryptoRegime


@dataclass
class ModelConfig:
    """Configuration for a specific regime combination."""
    model_name: str            # Which model to prioritize
    kelly_multiplier: float    # Scale Kelly sizing (0.5 = half Kelly)
    max_position_pct: float    # Max position size as fraction of portfolio
    confidence_threshold: float  # Min confidence to act
    description: str = ""      # Human-readable description


# Priority-ordered model matrix.
# First match wins. Use "*" as wildcard for any regime.
# (macro, crypto, micro) -> ModelConfig
MODEL_MATRIX: List[Tuple[str, str, str, ModelConfig]] = [
    # ── CRISIS overrides (highest priority) ──
    ("CRISIS", "CRASH", "*", ModelConfig(
        model_name="CrashRegime",
        kelly_multiplier=0.3,
        max_position_pct=0.02,
        confidence_threshold=0.70,
        description="Full crisis: crash model only, minimal sizing",
    )),
    ("CRISIS", "*", "*", ModelConfig(
        model_name="CrashRegime",
        kelly_multiplier=0.4,
        max_position_pct=0.03,
        confidence_threshold=0.65,
        description="Macro crisis: crash model with reduced sizing",
    )),
    ("*", "CRASH", "*", ModelConfig(
        model_name="CrashRegime",
        kelly_multiplier=0.4,
        max_position_pct=0.03,
        confidence_threshold=0.65,
        description="Crypto crash: crash model with reduced sizing",
    )),

    # ── LATE_CYCLE + DISTRIBUTION (caution) ──
    ("LATE_CYCLE", "DISTRIBUTION", "*", ModelConfig(
        model_name="MetaEnsemble",
        kelly_multiplier=0.5,
        max_position_pct=0.04,
        confidence_threshold=0.60,
        description="Late cycle distribution: ensemble with half Kelly",
    )),
    ("LATE_CYCLE", "*", "*", ModelConfig(
        model_name="MetaEnsemble",
        kelly_multiplier=0.6,
        max_position_pct=0.05,
        confidence_threshold=0.55,
        description="Late cycle: ensemble, cautious sizing",
    )),

    # ── EXPANSION (risk-on) ──
    ("EXPANSION", "BULL_TREND", "*", ModelConfig(
        model_name="MetaEnsemble",
        kelly_multiplier=1.0,
        max_position_pct=0.10,
        confidence_threshold=0.40,
        description="Expansion + bull trend: full sizing, aggressive",
    )),
    ("EXPANSION", "*", "*", ModelConfig(
        model_name="MetaEnsemble",
        kelly_multiplier=0.8,
        max_position_pct=0.08,
        confidence_threshold=0.45,
        description="Expansion: near-full sizing",
    )),

    # ── RECOVERY (transitional) ──
    ("RECOVERY", "ACCUMULATION", "*", ModelConfig(
        model_name="MetaEnsemble",
        kelly_multiplier=0.7,
        max_position_pct=0.06,
        confidence_threshold=0.50,
        description="Recovery + accumulation: moderate sizing, building",
    )),
    ("RECOVERY", "*", "*", ModelConfig(
        model_name="MetaEnsemble",
        kelly_multiplier=0.6,
        max_position_pct=0.05,
        confidence_threshold=0.55,
        description="Recovery: cautious rebuild",
    )),

    # ── ACCUMULATION (bottom fishing) ──
    ("*", "ACCUMULATION", "*", ModelConfig(
        model_name="MetaEnsemble",
        kelly_multiplier=0.6,
        max_position_pct=0.05,
        confidence_threshold=0.55,
        description="Accumulation phase: moderate sizing, patient",
    )),

    # ── DISTRIBUTION (topping) ──
    ("*", "DISTRIBUTION", "*", ModelConfig(
        model_name="MetaEnsemble",
        kelly_multiplier=0.5,
        max_position_pct=0.04,
        confidence_threshold=0.60,
        description="Distribution phase: reduced sizing",
    )),

    # ── Default fallback ──
    ("*", "*", "*", ModelConfig(
        model_name="MetaEnsemble",
        kelly_multiplier=0.7,
        max_position_pct=0.06,
        confidence_threshold=0.50,
        description="Default: standard ensemble with moderate sizing",
    )),
]


class ModelRouter:
    """Routes regime combinations to model configurations.

    Phase 1: Observation mode — logs recommendations, doesn't enforce.
    Phase 2: Active mode — modifies signal weights and position sizing.
    """

    def __init__(
        self,
        observation_mode: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.observation_mode = observation_mode
        self.logger = logger or logging.getLogger(__name__)
        self._last_config: Optional[ModelConfig] = None
        self._route_count: int = 0
        self._route_history: List[Dict] = []

    def route(
        self,
        macro: MacroRegime,
        crypto: CryptoRegime,
        micro: str = "*",
    ) -> ModelConfig:
        """Look up model config for given regime combination.

        Args:
            macro: Current macro regime (EXPANSION, LATE_CYCLE, CRISIS, RECOVERY)
            crypto: Current crypto regime (BULL_TREND, DISTRIBUTION, CRASH, ACCUMULATION)
            micro: Current micro regime from HMM (e.g., 'mean_reverting', 'trending')

        Returns:
            ModelConfig with recommended model, sizing, and thresholds.
        """
        config = self._lookup(macro.value, crypto.value, micro)
        self._route_count += 1

        # Track changes
        changed = (
            self._last_config is None
            or self._last_config.model_name != config.model_name
            or self._last_config.kelly_multiplier != config.kelly_multiplier
        )

        if changed:
            self.logger.info(
                f"MODEL ROUTE: ({macro.value}, {crypto.value}, {micro}) "
                f"-> {config.model_name} "
                f"[kelly={config.kelly_multiplier:.1f}, "
                f"max_pos={config.max_position_pct:.0%}, "
                f"conf_thresh={config.confidence_threshold:.0%}] "
                f"{'(OBSERVATION)' if self.observation_mode else '(ACTIVE)'}"
            )

        self._last_config = config

        # Record in history
        entry = {
            "macro": macro.value,
            "crypto": crypto.value,
            "micro": micro,
            "model": config.model_name,
            "kelly": config.kelly_multiplier,
            "max_pos": config.max_position_pct,
            "conf_thresh": config.confidence_threshold,
            "description": config.description,
        }
        self._route_history.append(entry)
        if len(self._route_history) > 200:
            self._route_history = self._route_history[-100:]

        return config

    def _lookup(self, macro: str, crypto: str, micro: str) -> ModelConfig:
        """Find first matching entry in MODEL_MATRIX."""
        for m_macro, m_crypto, m_micro, config in MODEL_MATRIX:
            if (m_macro == "*" or m_macro == macro) and \
               (m_crypto == "*" or m_crypto == crypto) and \
               (m_micro == "*" or m_micro == micro):
                return config

        # Should never reach here due to ("*", "*", "*") fallback
        return MODEL_MATRIX[-1][3]

    def get_state(self) -> Dict:
        """Return serializable state for dashboard/logging."""
        return {
            "observation_mode": self.observation_mode,
            "route_count": self._route_count,
            "current_config": {
                "model": self._last_config.model_name,
                "kelly_multiplier": self._last_config.kelly_multiplier,
                "max_position_pct": self._last_config.max_position_pct,
                "confidence_threshold": self._last_config.confidence_threshold,
                "description": self._last_config.description,
            } if self._last_config else None,
            "recent_routes": self._route_history[-10:],
        }
