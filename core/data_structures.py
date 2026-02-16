"""
Core Data Structures for Continuous Position Re-evaluation & MHPE
==================================================================
PositionContext    — full context for an open position (frozen at entry + live state)
ReEvalResult      — output of one re-evaluation cycle for one position
REASON_CODES      — machine-readable codes for every re-evaluation decision
HorizonEstimate   — probability estimate for a single time horizon
ProbabilityCone   — complete probability cone for one position (all horizons)
ConeAnalysis      — actionable interpretation of a ProbabilityCone
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Position Context
# ---------------------------------------------------------------------------

@dataclass
class PositionContext:
    """
    Complete context for an open position.
    Created at entry, updated every evaluation cycle.

    This is the "memory" of why the position was entered and how
    conditions have evolved since. The re-evaluator uses this to
    decide whether to hold, trim, add, or close.
    """

    # ── IDENTITY ──
    position_id: str
    pair: str                          # e.g., "BTC-USD"
    exchange: str                      # e.g., "coinbase"
    side: str                          # "long" or "short"
    strategy: str                      # which strategy created this position

    # ── ENTRY CONDITIONS (frozen at entry, never change) ──
    entry_price: Decimal
    entry_size: Decimal                # original position size in base asset
    entry_size_usd: Decimal
    entry_timestamp: float
    entry_confidence: float            # signal confidence at entry (e.g., 0.58)
    entry_expected_move_bps: float     # how many bps the signal predicted
    entry_cost_estimate_bps: float     # estimated roundtrip cost at entry
    entry_net_edge_bps: float          # expected_move - cost = net edge at entry
    entry_regime: str                  # regime at time of entry
    entry_volatility: float            # ATR or stdev at entry
    entry_book_depth_usd: Decimal      # orderbook depth at entry
    entry_spread_bps: float            # bid-ask spread at entry
    entry_funding_rate: Optional[float] = None  # funding rate at entry (perps only)
    signal_ttl_seconds: int = 300      # how long the signal is valid
    signal_id: str = "unknown"
    signal_source: str = "unknown"     # "cross_exchange", "funding_rate", etc.

    # ── CURRENT STATE (updated every cycle) ──
    current_size: Decimal = Decimal("0")
    current_size_usd: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    unrealized_pnl_usd: Decimal = Decimal("0")
    unrealized_pnl_bps: float = 0.0
    realized_move_bps: float = 0.0     # how many bps price moved in our favor
    remaining_edge_bps: float = 0.0    # estimated remaining edge after cost
    current_confidence: float = 0.5    # re-scored confidence (updated each cycle)
    current_optimal_size: Decimal = Decimal("0")
    current_cost_to_exit_bps: float = 0.0
    current_spread_bps: float = 0.0
    current_book_depth_usd: Decimal = Decimal("10000")
    current_regime: str = "unknown"
    current_volatility: float = 0.0
    current_funding_rate: Optional[float] = None

    # ── MANAGEMENT HISTORY ──
    adjustments: List[Dict] = field(default_factory=list)
    total_trimmed_usd: Decimal = Decimal("0")
    total_added_usd: Decimal = Decimal("0")
    realized_pnl_from_trims_usd: Decimal = Decimal("0")
    times_reevaluated: int = 0
    last_reevaluation_timestamp: float = 0.0

    # ── DERIVED PROPERTIES ──

    @property
    def age_seconds(self) -> float:
        return time.time() - self.entry_timestamp

    @property
    def time_elapsed_pct(self) -> float:
        """How much of the signal TTL has been consumed. 0.0 to 1.0+"""
        if self.signal_ttl_seconds <= 0:
            return 1.0
        return self.age_seconds / self.signal_ttl_seconds

    @property
    def move_completion_pct(self) -> float:
        """How much of the expected move has been captured. Can exceed 1.0."""
        if self.entry_expected_move_bps == 0:
            return 0.0
        return self.realized_move_bps / self.entry_expected_move_bps

    @property
    def edge_consumed_pct(self) -> float:
        """How much of the original net edge has been consumed."""
        if self.entry_net_edge_bps == 0:
            return 1.0
        return 1.0 - (self.remaining_edge_bps / self.entry_net_edge_bps)

    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl_bps > 0

    @property
    def is_expired(self) -> bool:
        return self.time_elapsed_pct >= 1.0

    @property
    def size_vs_optimal_ratio(self) -> float:
        """
        Ratio of current size to what Kelly says it should be.
        < 0.8 means we're too small
        0.8-1.5 = dead zone, no action needed
        > 1.5 means conditions improved, could add
        < 0.3 means we should close
        """
        if self.current_optimal_size == 0:
            return 0.0
        return float(self.current_size / self.current_optimal_size)


# ---------------------------------------------------------------------------
# Re-evaluation Result
# ---------------------------------------------------------------------------

@dataclass
class ReEvalResult:
    """Output of one re-evaluation cycle for one position."""

    position_id: str
    timestamp: float
    action: str                        # "hold", "trim", "close", "add"
    reason: str                        # human-readable explanation
    reason_code: str                   # machine-readable code for logging

    # Scoring details
    rescored_confidence: float
    remaining_edge_bps: float
    optimal_size_usd: Decimal
    current_size_usd: Decimal
    size_ratio: float                  # current / optimal

    # Action details (only relevant if action != "hold")
    trim_amount_usd: Optional[Decimal] = None
    add_amount_usd: Optional[Decimal] = None
    new_stop_price: Optional[Decimal] = None

    # Urgency
    urgency: str = "normal"            # "normal", "high", "critical"


# ---------------------------------------------------------------------------
# Reason Codes
# ---------------------------------------------------------------------------

REASON_CODES = {
    # Edge exhaustion
    "EDGE_CONSUMED":        "Move captured most of expected edge",
    "EDGE_NEGATIVE":        "Remaining edge is negative after costs",
    "EDGE_COST_EXCEEDED":   "Exit cost now exceeds remaining edge",

    # Confidence decay
    "CONFIDENCE_DECAYED":   "Re-scored confidence dropped below threshold",
    "CONFIDENCE_TIME":      "Time decay reduced confidence below threshold",
    "CONFIDENCE_DIVERGED":  "Correlated asset diverged from expectation",

    # Cost changes
    "COST_SPREAD_WIDENED":  "Spread widened significantly since entry",
    "COST_DEPTH_THINNED":   "Orderbook depth dropped, exit would move market",
    "COST_FUNDING_ADVERSE": "Funding rate turned adverse to position",

    # Risk changes
    "RISK_VOL_SPIKE":       "Volatility spiked since entry",
    "RISK_REGIME_CHANGE":   "Market regime changed since entry",
    "RISK_CORRELATION":     "Portfolio correlation shifted unfavorably",
    "RISK_AGGREGATE":       "Aggregate portfolio risk exceeds limits",

    # Profitable management
    "PROFIT_TRIM":          "Captured sufficient edge, trimming to lock in",
    "PROFIT_CLOSE":         "Captured most of expected move, closing",
    "PROFIT_TRAIL":         "Moving stop to lock in partial gains",

    # Opportunity cost
    "OPPORTUNITY_BETTER":   "Better signal available, freeing capital",

    # Hard stops (original exit triggers — emergency guardrails)
    "HARD_TARGET_HIT":      "Full target price reached",
    "HARD_TIME_EXPIRED":    "Signal TTL fully elapsed",
    "HARD_RISK_BUDGET":     "Position risk budget exhausted",
    "HARD_DAILY_LOSS":      "Daily loss limit hit",
    "HARD_SYSTEM_HALT":     "System-wide halt triggered",

    # Internal / churn prevention
    "CHURN_PREVENTION":     "Too soon since last adjustment",
    "MAX_ADJUSTMENTS":      "Max adjustments per position reached",
    "WITHIN_TOLERANCE":     "Size within dead zone, no action needed",
    "TRIM_TOO_SMALL":       "Trim amount below minimum threshold",
    "ADD_TOO_SMALL":        "Add amount below minimum threshold",
    "CONFIDENCE_NOT_IMPROVED": "Confidence did not improve, skipping add",
    "REEVAL_ADD":           "Conditions improved, adding to position",
    "FALLBACK":             "No specific action determined",

    # Cone-driven (MHPE / Doc 11)
    "CONE_CLOSE_NOW":       "Probability cone indicates EV negative at all horizons",
    "CONE_TAIL_RISK":       "Tail risk warning active with marginal P&L",
}


# ---------------------------------------------------------------------------
# Multi-Horizon Probability Estimator (MHPE) Data Structures  —  Doc 11
# ---------------------------------------------------------------------------

@dataclass
class HorizonEstimate:
    """Probability estimate for a single time horizon."""

    horizon_seconds: int               # 1, 5, 30, 120, 300, 900, 3600
    horizon_label: str                 # "1s", "5s", "30s", "2m", "5m", "15m", "60m"

    # Core probabilities
    p_profit: float                    # P(price moves in our favor) [0.0 - 1.0]
    p_loss: float                      # P(price moves against us) = 1 - p_profit

    # Expected moves (in basis points)
    e_favorable_bps: float             # Expected favorable move (positive number)
    e_adverse_bps: float               # Expected adverse move (positive number)
    e_net_bps: float                   # Risk-adjusted EV = p*fav - (1-p)*adv - cost
    sigma_bps: float                   # Standard deviation of expected move

    # Tail risk
    p_adverse_10bps: float             # P(adverse move > 10 bps)
    p_adverse_25bps: float             # P(adverse move > 25 bps)
    p_adverse_50bps: float             # P(adverse move > 50 bps)

    # Meta-confidence: how reliable is THIS estimate?
    estimate_confidence: float         # High for 1s (lots of data), low for 60m

    # Which data source contributed most at this horizon
    dominant_signal: str               # "imbalance", "flow", "vwap", "regime", etc.


@dataclass
class ProbabilityCone:
    """
    Complete probability cone for one position.
    Contains estimates across all time horizons.
    Computed every evaluation cycle by the MHPE.
    """

    position_id: str
    pair: str
    side: str                          # "long" or "short"
    timestamp: float
    computation_time_ms: float

    # The cone — one estimate per horizon, ordered shortest to longest
    horizons: List[HorizonEstimate]

    # Derived analysis
    peak_ev_horizon_seconds: int       # horizon with highest E[net]
    peak_ev_bps: float                 # E[net] at peak horizon
    ev_zero_crossing_seconds: int      # horizon where E[net] turns negative
    optimal_hold_remaining_seconds: int
    recommended_action: str            # "hold_to_peak", "close_now", "trim_now", …
    action_urgency: str                # "none", "low", "medium", "high", "critical"

    # Risk summary
    max_horizon_with_positive_ev: int
    worst_case_5min_bps: float         # 95th percentile adverse at 5 min
    worst_case_15min_bps: float        # 95th percentile adverse at 15 min

    @property
    def is_ev_positive_short_term(self) -> bool:
        return self.horizons[0].e_net_bps > 0 if self.horizons else False

    @property
    def is_ev_decaying(self) -> bool:
        if len(self.horizons) < 2:
            return False
        return self.horizons[-1].e_net_bps < self.horizons[0].e_net_bps

    @property
    def is_risk_accelerating(self) -> bool:
        if len(self.horizons) < 3:
            return False
        short = self.horizons[0]
        long_ = self.horizons[-2]
        if short.e_favorable_bps == 0:
            return True
        short_ratio = short.e_adverse_bps / short.e_favorable_bps
        long_ratio = long_.e_adverse_bps / long_.e_favorable_bps
        return long_ratio > short_ratio * 1.5


@dataclass
class ConeAnalysis:
    """
    Actionable interpretation of a ProbabilityCone.
    This is what the re-evaluator (Doc 10) consumes.
    """

    position_id: str

    # Timing recommendation
    optimal_hold_seconds: int
    close_by_seconds: int              # must close before this (EV negative)
    urgency: str                       # "none", "low", "medium", "high", "immediate"

    # Sizing recommendation (based on cone)
    size_multiplier: float             # 1.0 = keep, 0.5 = trim half, 0.0 = close
    reason: str

    # Risk flags
    tail_risk_warning: bool
    volatility_expanding: bool
    regime_transition_risk: bool

    # Edge timing
    edge_front_loaded: bool
    edge_back_loaded: bool
    edge_uniformly_distributed: bool
