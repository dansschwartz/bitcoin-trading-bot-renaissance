"""
Regime Config Registry — SAND-inspired unified regime parameters.

Every regime-conditional parameter in the system is registered here
as a named entry with bit-vector tags. Modules read from this registry
at runtime. Council researchers propose changes to specific entries.

Usage by modules:
    from knowledge.regime_registry import REGIME_CONFIG
    scalar = REGIME_CONFIG.get("sizing.regime_scalar", regime="trending")
    weights = REGIME_CONFIG.get_all("signal_weight", regime="high_volatility")

Usage by researchers (proposals):
    from knowledge.regime_registry import REGIME_CONFIG
    # See current value
    print(REGIME_CONFIG.get("sizing.regime_scalar", regime="trending"))
    # Propose a change (writes to proposals list, does NOT change runtime)
    REGIME_CONFIG.propose_change(
        key="sizing.regime_scalar",
        regime="trending",
        current_value=1.20,
        proposed_value=1.05,
        rationale="Audit shows trending regime returns don't justify 1.2x sizing"
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# BIT-VECTOR DIMENSIONS
# ═══════════════════════════════════════════════════════════════

class Regime(IntFlag):
    LOW_VOL        = 0x01
    TRENDING       = 0x02
    MEAN_REVERTING = 0x04
    HIGH_VOL       = 0x08
    CHAOTIC        = 0x10
    TRANSITION     = 0x20
    NORMAL         = 0x40
    ALL            = 0x7F


class ParamType(IntFlag):
    SIZING         = 0x01
    SIGNAL_WEIGHT  = 0x02
    THRESHOLD      = 0x04
    RISK_LIMIT     = 0x08
    VOLATILITY     = 0x10
    EXECUTION      = 0x20
    ALL            = 0x3F


class Pair(IntFlag):
    BTC  = 0x01
    ETH  = 0x02
    SOL  = 0x04
    DOGE = 0x08
    AVAX = 0x10
    LINK = 0x20
    ALL  = 0x3F


# ═══════════════════════════════════════════════════════════════
# REGISTRY ENTRY
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConfigEntry:
    """A single regime-conditional parameter."""
    key: str
    value: Any
    regime: int = Regime.ALL
    param_type: int = ParamType.ALL
    pairs: int = Pair.ALL
    source_file: str = ""
    source_line: str = ""
    description: str = ""
    unit: str = ""
    min_value: Any = None
    max_value: Any = None
    editable: bool = True


# ═══════════════════════════════════════════════════════════════
# THE REGISTRY
# ═══════════════════════════════════════════════════════════════

class RegimeConfigRegistry:
    """Unified registry of all regime-conditional parameters."""

    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, int], ConfigEntry] = {}
        self._proposals: List[Dict[str, Any]] = []

    def register(self, entry: ConfigEntry) -> None:
        self._entries[(entry.key, entry.regime)] = entry

    def register_many(self, entries: List[ConfigEntry]) -> None:
        for e in entries:
            self.register(e)

    def get(self, key: str, regime: Optional[str] = None,
            regime_flag: Optional[int] = None) -> Any:
        """Get a parameter value for a specific regime.

        Lookup order: exact match → bitwise overlap → ALL fallback → None.
        """
        if regime_flag is None and regime is not None:
            regime_flag = self._regime_str_to_flag(regime)

        # Exact match
        entry = self._entries.get((key, regime_flag))
        if entry is not None:
            return entry.value

        # Check if any registered entry's regime flags include this regime
        if regime_flag is not None:
            for (k, r), entry in self._entries.items():
                if k == key and (r & regime_flag):
                    return entry.value

        # Fallback to ALL
        entry = self._entries.get((key, Regime.ALL))
        if entry is not None:
            return entry.value

        return None

    def get_entry(self, key: str, regime: Optional[str] = None) -> Optional[ConfigEntry]:
        """Get the full ConfigEntry (not just value)."""
        flag = self._regime_str_to_flag(regime) if regime else Regime.ALL
        entry = self._entries.get((key, flag))
        if entry is None:
            entry = self._entries.get((key, Regime.ALL))
        return entry

    def get_all(self, param_type: int = ParamType.ALL,
                regime: Optional[int] = None) -> List[ConfigEntry]:
        """Query all entries matching type and/or regime."""
        results = []
        for (_key, r), entry in self._entries.items():
            if param_type != ParamType.ALL and not (entry.param_type & param_type):
                continue
            if regime is not None and not (r & regime):
                continue
            results.append(entry)
        return results

    def manifest(self, param_type: int = ParamType.ALL) -> str:
        """Human-readable listing of all entries."""
        lines = ["KEY | REGIME | VALUE | UNIT | SOURCE"]
        lines.append("-" * 80)
        for (key, regime), entry in sorted(self._entries.items()):
            if param_type != ParamType.ALL and not (entry.param_type & param_type):
                continue
            regime_name = self._flag_to_regime_str(regime)
            lines.append(
                f"{key} | {regime_name} | {entry.value} | "
                f"{entry.unit} | {entry.source_file}:{entry.source_line}"
            )
        return "\n".join(lines)

    def propose_change(
        self,
        key: str,
        regime: str,
        current_value: Any,
        proposed_value: Any,
        rationale: str,
        researcher: str = "unknown",
        evidence: str = "",
    ) -> Dict[str, Any]:
        """Create a structured change proposal.

        Returns a proposal dict for inclusion in proposals.json.
        Does NOT change any runtime values.
        """
        entry = self.get_entry(key, regime)

        # Safety checks
        if entry and not entry.editable:
            return {
                "error": f"Parameter '{key}' is marked non-editable (safety parameter)",
                "key": key,
                "regime": regime,
            }

        if entry and entry.min_value is not None and proposed_value < entry.min_value:
            return {
                "error": f"Proposed value {proposed_value} below minimum {entry.min_value}",
                "key": key,
                "regime": regime,
            }

        if entry and entry.max_value is not None and proposed_value > entry.max_value:
            return {
                "error": f"Proposed value {proposed_value} above maximum {entry.max_value}",
                "key": key,
                "regime": regime,
            }

        proposal = {
            "type": "regime_config_change",
            "key": key,
            "regime": regime,
            "current_value": current_value,
            "proposed_value": proposed_value,
            "rationale": rationale,
            "researcher": researcher,
            "evidence": evidence,
            "source_file": entry.source_file if entry else "unknown",
            "source_line": entry.source_line if entry else "unknown",
            "description": entry.description if entry else "",
            "unit": entry.unit if entry else "",
        }

        self._proposals.append(proposal)
        return proposal

    def get_proposals(self) -> List[Dict[str, Any]]:
        return list(self._proposals)

    def clear_proposals(self) -> None:
        self._proposals = []

    def _regime_str_to_flag(self, regime: str) -> int:
        mapping = {
            "low_volatility": Regime.LOW_VOL,
            "low_vol": Regime.LOW_VOL,
            "trending": Regime.TRENDING,
            "bull_trending": Regime.TRENDING,
            "bear_trending": Regime.TRENDING,
            "mean_reverting": Regime.MEAN_REVERTING,
            "bull_mean_reverting": Regime.MEAN_REVERTING,
            "high_volatility": Regime.HIGH_VOL,
            "high_vol": Regime.HIGH_VOL,
            "volatile": Regime.HIGH_VOL,
            "chaotic": Regime.CHAOTIC,
            "transition": Regime.TRANSITION,
            "normal": Regime.NORMAL,
            "neutral_sideways": Regime.NORMAL,
        }
        return mapping.get(regime.lower(), Regime.ALL)

    def _flag_to_regime_str(self, flag: int) -> str:
        if flag == Regime.ALL:
            return "ALL"
        names = []
        for r in Regime:
            if r != Regime.ALL and flag & r:
                names.append(r.name.lower())
        return "|".join(names) if names else "ALL"


# ═══════════════════════════════════════════════════════════════
# POPULATE THE REGISTRY — sourced from actual codebase values
# ═══════════════════════════════════════════════════════════════

REGIME_CONFIG = RegimeConfigRegistry()

# ── From position_sizer.py: regime_scalars ──
_SIZING_SCALARS = [
    ("trending",       1.20, "Scale up in trends — momentum carries"),
    ("mean_reverting", 1.00, "Normal sizing — regime is stable"),
    ("volatile",       0.60, "Cut size — high adverse selection risk"),
    ("chaotic",        0.30, "Minimal sizing — regime unpredictable"),
    ("normal",         0.80, "Default conservative sizing"),
]
for _regime, _value, _desc in _SIZING_SCALARS:
    REGIME_CONFIG.register(ConfigEntry(
        key="sizing.regime_scalar",
        value=_value,
        regime=REGIME_CONFIG._regime_str_to_flag(_regime),
        param_type=ParamType.SIZING,
        source_file="position_sizer.py",
        source_line="regime_scalars",
        description=_desc,
        unit="multiplier",
        min_value=0.10,
        max_value=2.00,
    ))

# ── From position_sizer.py: other sizing params ──
REGIME_CONFIG.register(ConfigEntry(
    key="sizing.kelly_fraction",
    value=0.25,
    regime=Regime.ALL,
    param_type=ParamType.SIZING,
    source_file="position_sizer.py",
    source_line="kelly_fraction",
    description="Fraction of full Kelly to use (quarter Kelly default)",
    unit="fraction",
    min_value=0.05,
    max_value=0.50,
))

REGIME_CONFIG.register(ConfigEntry(
    key="sizing.cost_gate_ratio",
    value=0.50,
    regime=Regime.ALL,
    param_type=ParamType.SIZING,
    source_file="position_sizer.py",
    source_line="cost_gate_ratio",
    description="Block trades where cost > this fraction of expected profit",
    unit="ratio",
    min_value=0.20,
    max_value=0.90,
))

REGIME_CONFIG.register(ConfigEntry(
    key="sizing.max_participation_rate",
    value=0.10,
    regime=Regime.ALL,
    param_type=ParamType.SIZING,
    source_file="position_sizer.py",
    source_line="max_participation_rate",
    description="Max fraction of daily volume for single trade",
    unit="fraction",
    min_value=0.01,
    max_value=0.25,
))

# ── From regime_overlay.py: signal weight adjustments ──
_SIGNAL_REGIME_MULTS = {
    "trending": {
        "macd": 1.3, "rsi": 1.3, "bollinger": 0.8,
        "order_flow": 0.9, "order_book": 0.9, "volume": 0.9,
    },
    "high_volatility": {
        "macd": 0.7, "rsi": 0.7, "bollinger": 1.3,
        "order_flow": 1.3, "order_book": 1.3, "volume": 1.3,
    },
    "mean_reverting": {
        "macd": 1.0, "rsi": 1.0, "bollinger": 1.2,
        "order_flow": 1.0, "order_book": 1.0, "volume": 1.0,
    },
}
for _regime, _weights in _SIGNAL_REGIME_MULTS.items():
    for _signal, _mult in _weights.items():
        REGIME_CONFIG.register(ConfigEntry(
            key=f"signal_weight.{_signal}",
            value=_mult,
            regime=REGIME_CONFIG._regime_str_to_flag(_regime),
            param_type=ParamType.SIGNAL_WEIGHT,
            source_file="regime_overlay.py",
            source_line="get_adjusted_weights",
            description=f"{_signal} weight multiplier in {_regime} regime",
            unit="multiplier",
            min_value=0.0,
            max_value=3.0,
        ))

# ── From intelligence/regime_predictor.py: volatility estimates ──
_REGIME_VOLS = {
    "low_volatility": 8, "mean_reverting": 15, "trending": 20,
    "high_volatility": 40, "chaotic": 80,
}
for _regime, _vol_bps in _REGIME_VOLS.items():
    REGIME_CONFIG.register(ConfigEntry(
        key="volatility.regime_vol_bps",
        value=_vol_bps,
        regime=REGIME_CONFIG._regime_str_to_flag(_regime),
        param_type=ParamType.VOLATILITY,
        source_file="intelligence/regime_predictor.py",
        source_line="_DEFAULT_REGIME_VOL",
        description=f"Expected per-bar volatility in {_regime} (bps per 5min bar)",
        unit="bps",
        min_value=1,
        max_value=200,
    ))

# ── From enhanced_config_manager.py: trading thresholds ──
_REGIME_THRESHOLDS = {
    "trending": {"buy": 0.60, "sell": -0.55, "confidence": 0.55, "size_mult": 1.0},
    "high_volatility": {"buy": 0.75, "sell": -0.75, "confidence": 0.70, "size_mult": 0.6},
    "mean_reverting": {"buy": 0.55, "sell": -0.50, "confidence": 0.50, "size_mult": 0.9},
}
for _regime, _thresholds in _REGIME_THRESHOLDS.items():
    for _name, _val in _thresholds.items():
        _ptype = ParamType.THRESHOLD if "size" not in _name else ParamType.SIZING
        REGIME_CONFIG.register(ConfigEntry(
            key=f"threshold.{_name}",
            value=_val,
            regime=REGIME_CONFIG._regime_str_to_flag(_regime),
            param_type=_ptype,
            source_file="enhanced_config_manager.py",
            source_line=f"RegimeConfiguration.{_regime}",
            description=f"{_name} threshold in {_regime} regime",
            unit="signal_strength" if "buy" in _name or "sell" in _name else "multiplier",
            min_value=-1.0 if "sell" in _name else 0.0,
            max_value=1.0 if "confidence" in _name else 2.0,
        ))

# ── SAFETY PARAMETERS — NOT EDITABLE ──
_SAFETY = [
    ("safety.max_position_usd", 10000, "ABSOLUTE_MAX_POSITION_USD"),
    ("safety.max_drawdown_pct", 0.10, "ABSOLUTE_MAX_DRAWDOWN_PCT"),
    ("safety.max_leverage", 3.0, "ABSOLUTE_MAX_LEVERAGE"),
    ("safety.max_daily_trades", 200, "ABSOLUTE_MAX_DAILY_TRADES"),
    ("safety.paper_trading", True, "PAPER_TRADING_ONLY"),
]
for _key, _value, _source_line in _SAFETY:
    REGIME_CONFIG.register(ConfigEntry(
        key=_key,
        value=_value,
        regime=Regime.ALL,
        param_type=ParamType.RISK_LIMIT,
        source_file="risk_gateway.py / safety_gate.py",
        source_line=_source_line,
        description=f"SAFETY LIMIT — NEVER CHANGE: {_source_line}",
        editable=False,
    ))
