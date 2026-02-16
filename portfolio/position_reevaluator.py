"""
Continuous Position Re-evaluation Engine
=========================================
Renaissance-inspired core that continuously optimizes every open position
every evaluation cycle. Instead of waiting for static exit triggers, it asks:
"Given current conditions, should this position exist at this size?"

Called by PortfolioEngine every evaluation cycle.

The philosophy: ENTRY and EXIT are the SAME PROCESS. The model continuously
calculates the optimal portfolio. The difference between optimal and actual
becomes orders.

IMPORTANT: This does NOT replace the hard exit triggers from Doc 1.
Those remain as emergency guardrails. This is a LAYER ABOVE them that
handles the profitable, intelligent management of positions.
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from core.data_structures import PositionContext, ReEvalResult, REASON_CODES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "evaluation_interval_seconds": 1.0,
    "min_edge_bps": 0.5,
    "confidence_floor": 0.505,
    "trim_threshold": 0.8,
    "close_threshold": 0.3,
    "add_threshold": 1.5,
    "churn_prevention_seconds": 10,
    "max_adjustments_per_position": 5,
    "min_trim_usd": 5.0,
    "cost_spike_multiplier": 2.0,
    "spread_spike_multiplier": 3.0,
    "max_position_loss_bps": 50,
}

# Regime transition confidence impact
_DESTRUCTIVE_TRANSITIONS = {
    ("trending_up", "chaotic"): 0.2,
    ("trending_down", "chaotic"): 0.2,
    ("low_volatility", "high_volatility"): 0.4,
    ("trending_up", "trending_down"): 0.3,
    ("trending_down", "trending_up"): 0.3,
}


# ---------------------------------------------------------------------------
# Remaining Edge helper
# ---------------------------------------------------------------------------

class _RemainingEdge:
    __slots__ = ("remaining_move_bps", "exit_cost_bps", "net_remaining_bps", "edge_ratio")

    def __init__(self, remaining_move_bps: float, exit_cost_bps: float,
                 net_remaining_bps: float, edge_ratio: float):
        self.remaining_move_bps = remaining_move_bps
        self.exit_cost_bps = exit_cost_bps
        self.net_remaining_bps = net_remaining_bps
        self.edge_ratio = edge_ratio


# ---------------------------------------------------------------------------
# PositionReEvaluator
# ---------------------------------------------------------------------------

class PositionReEvaluator:
    """
    Continuously re-evaluates every open position.

    Dependencies (passed via PortfolioEngine or at construction):
      - cost_model:       CostModel instance (to re-estimate exit costs)
      - kelly_sizer:      KellyPositionSizer (to recalculate optimal size)
      - regime_detector:  RegimeDetector (to check for regime changes)
      - devil_tracker:    DevilTracker (to log cost evolution)

    Config keys — see _DEFAULT_CONFIG above.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cost_model: Any = None,
        kelly_sizer: Any = None,
        regime_detector: Any = None,
        devil_tracker: Any = None,
        mhpe: Any = None,
    ):
        raw = config or {}
        self.config: Dict[str, Any] = {**_DEFAULT_CONFIG, **raw}
        self._enabled = self.config.get("enabled", True)

        self.cost_model = cost_model
        self.kelly_sizer = kelly_sizer
        self.regime_detector = regime_detector
        self.devil_tracker = devil_tracker
        self.mhpe = mhpe  # MultiHorizonEstimator (Doc 11)

        # Strategy-specific overrides
        self._strategy_overrides: Dict[str, Dict[str, Any]] = raw.get(
            "strategy_overrides", {}
        )

        # Metrics
        self._total_reevaluations = 0
        self._total_trims = 0
        self._total_early_closes = 0
        self._total_adds = 0
        self._total_holds = 0
        self._profit_from_trims_usd = Decimal("0")

        logger.info(
            "PositionReEvaluator init  enabled=%s  interval=%.1fs  "
            "min_edge=%.1fbps  confidence_floor=%.3f  "
            "trim=%.1f  close=%.1f  add=%.1f",
            self._enabled,
            self.config["evaluation_interval_seconds"],
            self.config["min_edge_bps"],
            self.config["confidence_floor"],
            self.config["trim_threshold"],
            self.config["close_threshold"],
            self.config["add_threshold"],
        )

    # ═══════════════════════════════════════════════════════════
    # Config helpers
    # ═══════════════════════════════════════════════════════════

    def _cfg(self, pos: PositionContext, key: str) -> Any:
        """Get config value with strategy-specific override support."""
        overrides = self._strategy_overrides.get(pos.strategy, {})
        return overrides.get(key, self.config[key])

    # ═══════════════════════════════════════════════════════════
    # MAIN ENTRY POINT — called by PortfolioEngine every cycle
    # ═══════════════════════════════════════════════════════════

    def reevaluate_all(
        self,
        positions: List[PositionContext],
        portfolio_state: Dict[str, Any],
        market_state: Dict[str, Any],
    ) -> List[ReEvalResult]:
        """
        Re-evaluate every open position. Returns a list of actions.

        Called by PortfolioEngine.reconciliation_loop() every cycle.
        The PortfolioEngine executes the returned actions.
        """
        if not self._enabled:
            return []

        results = []
        for position in positions:
            try:
                result = self._reevaluate_single(position, portfolio_state, market_state)
                results.append(result)

                position.times_reevaluated += 1
                position.last_reevaluation_timestamp = time.time()
                position.current_confidence = result.rescored_confidence
                position.remaining_edge_bps = result.remaining_edge_bps
                position.current_optimal_size = result.optimal_size_usd

                self._total_reevaluations += 1
                if result.action == "hold":
                    self._total_holds += 1
                elif result.action == "close":
                    self._total_early_closes += 1
            except Exception as exc:
                logger.error(
                    "ReEval failed for %s: %s", position.position_id, exc
                )
        return results

    # ═══════════════════════════════════════════════════════════
    # SINGLE POSITION RE-EVALUATION — The Core Logic
    # ═══════════════════════════════════════════════════════════

    def _reevaluate_single(
        self,
        pos: PositionContext,
        portfolio_state: Dict[str, Any],
        market_state: Dict[str, Any],
    ) -> ReEvalResult:
        """
        The heart of the system. Re-evaluates one position.

        Steps:
        1. Churn prevention
        2. Check hard stops (emergency guardrails)
        3. Update current market conditions
        4. Re-score signal confidence
        5. Re-estimate remaining edge after costs
        6. Check cost changes
        7. Edge exhaustion check
        8. Confidence check
        9. Recalculate optimal size via Kelly
        10. Compare optimal to actual and decide action
        """
        timestamp = time.time()

        # ── STEP 0: CHURN PREVENTION ──
        time_since_last = timestamp - self._last_adjustment_time(pos)
        churn_seconds = self._cfg(pos, "churn_prevention_seconds")
        if time_since_last < churn_seconds and len(pos.adjustments) > 0:
            return self._hold_result(pos, "CHURN_PREVENTION")

        # ── STEP 1: HARD STOPS (Emergency guardrails) ──
        hard_stop = self._check_hard_stops(pos, portfolio_state)
        if hard_stop is not None:
            return hard_stop

        # ── STEP 2: UPDATE CURRENT CONDITIONS ──
        self._update_position_market_data(pos, market_state)

        # ── STEP 3: RE-SCORE SIGNAL CONFIDENCE ──
        rescored_confidence = self._rescore_signal(pos, market_state)

        # ── STEP 4: RE-ESTIMATE REMAINING EDGE ──
        remaining_edge = self._calculate_remaining_edge(pos)

        # ── STEP 5: CHECK COST CHANGES ──
        cost_action = self._check_cost_changes(pos)
        if cost_action is not None:
            return cost_action

        # ── STEP 5.5: CONSULT PROBABILITY CONE (MHPE — Doc 11) ──
        cone_analysis = None
        if self.mhpe is not None:
            try:
                cone_analysis = self.mhpe.analyze_for_reevaluator(pos)

                # OVERRIDE 1: Cone says close with high urgency
                if (cone_analysis.urgency in ("high", "immediate")
                        and cone_analysis.size_multiplier == 0.0):
                    return self._close_result(
                        pos, "CONE_CLOSE_NOW",
                        cone_analysis.reason,
                        rescored_confidence, remaining_edge.net_remaining_bps,
                        urgency=cone_analysis.urgency,
                    )

                # OVERRIDE 2: Tail risk active + marginal P&L
                tail_threshold = self.config.get(
                    "cone_tail_risk_threshold_bps", 3.0
                )
                if (cone_analysis.tail_risk_warning
                        and pos.unrealized_pnl_bps < tail_threshold):
                    return self._close_result(
                        pos, "CONE_TAIL_RISK",
                        f"tail risk warning, P&L {pos.unrealized_pnl_bps:.1f}bps",
                        rescored_confidence, remaining_edge.net_remaining_bps,
                        urgency="medium",
                    )
            except Exception as exc:
                logger.debug("MHPE cone analysis failed for %s: %s", pos.position_id, exc)

        # ── STEP 6: EDGE EXHAUSTION CHECK ──
        min_edge = self._cfg(pos, "min_edge_bps")
        if remaining_edge.net_remaining_bps <= min_edge:
            return self._close_result(
                pos, "EDGE_CONSUMED",
                f"remaining={remaining_edge.net_remaining_bps:.1f}bps, "
                f"min={min_edge}bps",
                rescored_confidence, remaining_edge.net_remaining_bps,
                urgency="normal" if pos.is_profitable else "high",
            )

        # ── STEP 7: CONFIDENCE CHECK ──
        confidence_floor = self._cfg(pos, "confidence_floor")
        if rescored_confidence < confidence_floor:
            return self._close_result(
                pos, "CONFIDENCE_DECAYED",
                f"confidence={rescored_confidence:.4f}, "
                f"floor={confidence_floor}",
                rescored_confidence, remaining_edge.net_remaining_bps,
                urgency="normal" if pos.is_profitable else "high",
            )

        # ── STEP 8: RECALCULATE OPTIMAL SIZE VIA KELLY ──
        optimal_size_usd = self._recalculate_optimal_size(
            pos, rescored_confidence, remaining_edge, portfolio_state
        )

        # ── STEP 8.5: APPLY CONE SIZE MULTIPLIER (Doc 11) ──
        if (cone_analysis is not None
                and self.config.get("cone_size_multiplier_enabled", True)):
            optimal_size_usd = Decimal(
                str(float(optimal_size_usd) * cone_analysis.size_multiplier)
            )

        # ── STEP 9: APPLY CONSTRAINTS ──
        optimal_size_usd = self._apply_constraints(
            optimal_size_usd, pos, portfolio_state
        )

        # ── STEP 10: COMPARE AND DECIDE ──
        # Dynamic close threshold: if edge is front-loaded and past peak, close faster
        dynamic_close_threshold = None
        if (cone_analysis is not None
                and cone_analysis.edge_front_loaded
                and cone_analysis.optimal_hold_seconds <= 0):
            dynamic_close_threshold = self.config.get(
                "front_loaded_close_threshold", 0.5
            )

        return self._compare_and_decide(
            pos, optimal_size_usd, rescored_confidence, remaining_edge, timestamp,
            dynamic_close_threshold=dynamic_close_threshold,
        )

    # ═══════════════════════════════════════════════════════════
    # STEP 1: HARD STOPS — Emergency guardrails
    # ═══════════════════════════════════════════════════════════

    def _check_hard_stops(
        self, pos: PositionContext, portfolio_state: Dict[str, Any]
    ) -> Optional[ReEvalResult]:
        """Non-negotiable exits — fire regardless of re-evaluation."""

        # Signal expired
        if pos.is_expired:
            urgency = "normal" if pos.is_profitable else "high"
            return self._close_result(
                pos, "HARD_TIME_EXPIRED",
                f"TTL={pos.signal_ttl_seconds}s elapsed",
                pos.current_confidence, pos.remaining_edge_bps,
                urgency=urgency,
            )

        # Risk budget exhausted
        max_loss_bps = self._cfg(pos, "max_position_loss_bps")
        if pos.unrealized_pnl_bps < -max_loss_bps:
            return self._close_result(
                pos, "HARD_RISK_BUDGET",
                f"loss={pos.unrealized_pnl_bps:.1f}bps, max={max_loss_bps}bps",
                pos.current_confidence, pos.remaining_edge_bps,
                urgency="critical",
            )

        # Daily loss limit
        if portfolio_state.get("daily_loss_limit_hit", False):
            return self._close_result(
                pos, "HARD_DAILY_LOSS",
                "portfolio daily loss limit reached",
                pos.current_confidence, pos.remaining_edge_bps,
                urgency="critical",
            )

        # System halt
        if portfolio_state.get("system_halted", False):
            return self._close_result(
                pos, "HARD_SYSTEM_HALT",
                "system halt triggered",
                pos.current_confidence, pos.remaining_edge_bps,
                urgency="critical",
            )

        return None

    # ═══════════════════════════════════════════════════════════
    # STEP 2: UPDATE CURRENT MARKET DATA ON POSITION
    # ═══════════════════════════════════════════════════════════

    def _update_position_market_data(
        self, pos: PositionContext, market_state: Dict[str, Any]
    ) -> None:
        """Refresh the position's current market data from live feeds."""
        pair_data = market_state.get(pos.pair, {})

        last_price = pair_data.get("last_price")
        if last_price is not None and float(last_price) > 0:
            pos.current_price = Decimal(str(last_price))

        # Update P&L
        if pos.current_price > 0 and pos.entry_price > 0:
            if pos.side == "long":
                move = pos.current_price - pos.entry_price
            else:
                move = pos.entry_price - pos.current_price
            pos.unrealized_pnl_usd = move * pos.current_size
            pos.unrealized_pnl_bps = float(move / pos.entry_price) * 10000
            pos.realized_move_bps = max(0, pos.unrealized_pnl_bps)

        pos.current_size_usd = pos.current_size * pos.current_price

        # Market microstructure from pair_data
        spread = pair_data.get("spread_bps")
        if spread is not None:
            pos.current_spread_bps = float(spread)
        depth = pair_data.get("book_depth_usd")
        if depth is not None:
            pos.current_book_depth_usd = Decimal(str(depth))

        # Regime
        if self.regime_detector is not None:
            regime = getattr(self.regime_detector, "current_regime", None)
            if regime is not None:
                pos.current_regime = str(regime)

        vol = pair_data.get("current_atr") or pair_data.get("volatility")
        if vol is not None:
            pos.current_volatility = float(vol)

        funding = pair_data.get("funding_rate")
        if funding is not None:
            pos.current_funding_rate = float(funding)

    # ═══════════════════════════════════════════════════════════
    # STEP 3: RE-SCORE SIGNAL CONFIDENCE
    # ═══════════════════════════════════════════════════════════

    def _rescore_signal(
        self, pos: PositionContext, market_state: Dict[str, Any]
    ) -> float:
        """
        Recalculate the probability that this position will be profitable
        given CURRENT conditions. This is the Bayesian update step.

        Factors: time decay, move completion, spread change, volatility
        change, regime change, funding rate change.

        Returns: float between 0.50 and entry_confidence.
        """
        base_confidence = pos.entry_confidence

        # ── TIME DECAY ── (quadratic: slow at first, accelerating toward expiry)
        elapsed_pct = min(pos.time_elapsed_pct, 1.0)
        time_factor = max(0.0, 1.0 - elapsed_pct ** 2)

        # ── MOVE COMPLETION ──
        move_pct = pos.move_completion_pct
        if move_pct >= 1.0:
            move_factor = 0.1
        elif move_pct >= 0.7:
            move_factor = 0.3 + 0.7 * (1.0 - move_pct) / 0.3
        elif move_pct >= 0.0:
            move_factor = 0.7 + 0.3 * (1.0 - move_pct)
        else:
            # Price went against us
            move_factor = max(0.3, 1.0 + move_pct * 2)

        # ── SPREAD CHANGE ──
        spread_factor = 1.0
        if pos.entry_spread_bps > 0 and pos.current_spread_bps > 0:
            spread_ratio = pos.current_spread_bps / pos.entry_spread_bps
            if spread_ratio > 2.0:
                spread_factor = 0.5
            elif spread_ratio > 1.5:
                spread_factor = 0.7

        # ── VOLATILITY CHANGE ──
        vol_factor = 1.0
        if pos.entry_volatility > 0 and pos.current_volatility > 0:
            vol_ratio = pos.current_volatility / pos.entry_volatility
            if vol_ratio > 3.0:
                vol_factor = 0.3
            elif vol_ratio > 2.0:
                vol_factor = 0.5
            elif vol_ratio > 1.5:
                vol_factor = 0.75

        # ── REGIME CHANGE ──
        regime_factor = 1.0
        if pos.current_regime != pos.entry_regime:
            regime_factor = self._regime_transition_factor(
                pos.entry_regime, pos.current_regime, pos.signal_source
            )

        # ── FUNDING RATE CHANGE (for funding arb positions) ──
        funding_factor = 1.0
        if (pos.signal_source == "funding_rate"
                and pos.entry_funding_rate is not None
                and pos.current_funding_rate is not None):
            if abs(pos.current_funding_rate) < abs(pos.entry_funding_rate) * 0.3:
                funding_factor = 0.2
            elif abs(pos.current_funding_rate) < abs(pos.entry_funding_rate) * 0.5:
                funding_factor = 0.5

        # ── COMBINE ──
        rescored = (
            base_confidence
            * time_factor
            * move_factor
            * spread_factor
            * vol_factor
            * regime_factor
            * funding_factor
        )

        # Floor at 0.50 (no edge), ceiling at entry confidence
        return max(0.50, min(rescored, pos.entry_confidence))

    @staticmethod
    def _regime_transition_factor(
        entry_regime: str, current_regime: str, signal_source: str
    ) -> float:
        """How much does a regime change affect this signal's confidence?"""
        key = (entry_regime, current_regime)
        if key in _DESTRUCTIVE_TRANSITIONS:
            return _DESTRUCTIVE_TRANSITIONS[key]
        return 0.8  # Default: mild reduction for any regime change

    # ═══════════════════════════════════════════════════════════
    # STEP 4: RE-ESTIMATE REMAINING EDGE
    # ═══════════════════════════════════════════════════════════

    def _calculate_remaining_edge(self, pos: PositionContext) -> _RemainingEdge:
        """
        How much profitable movement remains, net of exit costs?
        remaining_edge = (expected_total_move - already_captured) - cost_to_exit
        """
        remaining_move_bps = max(0.0, pos.entry_expected_move_bps - pos.realized_move_bps)

        # Exit cost from cost model or fallback
        exit_cost_bps = pos.current_cost_to_exit_bps
        if self.cost_model is not None:
            try:
                if hasattr(self.cost_model, "estimate_exit_cost"):
                    exit_cost_bps = self.cost_model.estimate_exit_cost(
                        pair=pos.pair,
                        size_usd=float(pos.current_size_usd),
                        exchange=pos.exchange,
                        current_spread_bps=pos.current_spread_bps,
                        current_book_depth_usd=float(pos.current_book_depth_usd),
                    )
                elif hasattr(self.cost_model, "calculate_cost"):
                    # Fall back to SimTransactionCostModel
                    cost = self.cost_model.calculate_cost(
                        trade_size_usd=float(pos.current_size_usd),
                        price=float(pos.current_price),
                    )
                    exit_cost_bps = self.cost_model.cost_in_bps(
                        cost, float(pos.current_size_usd)
                    )
            except Exception:
                pass  # Use cached value

        pos.current_cost_to_exit_bps = exit_cost_bps
        net_remaining = remaining_move_bps - exit_cost_bps
        edge_ratio = (
            net_remaining / pos.entry_net_edge_bps
            if pos.entry_net_edge_bps > 0 else 0.0
        )

        return _RemainingEdge(
            remaining_move_bps=remaining_move_bps,
            exit_cost_bps=exit_cost_bps,
            net_remaining_bps=net_remaining,
            edge_ratio=edge_ratio,
        )

    # ═══════════════════════════════════════════════════════════
    # STEP 5: CHECK FOR COST CHANGES
    # ═══════════════════════════════════════════════════════════

    def _check_cost_changes(self, pos: PositionContext) -> Optional[ReEvalResult]:
        """Did The Devil get worse since we entered?"""

        # Spread spike check
        if pos.entry_spread_bps > 0 and pos.current_spread_bps > 0:
            spread_ratio = pos.current_spread_bps / pos.entry_spread_bps
            spike_mult = self._cfg(pos, "spread_spike_multiplier")
            if spread_ratio > spike_mult:
                return self._close_result(
                    pos, "COST_SPREAD_WIDENED",
                    f"spread={pos.current_spread_bps:.1f}bps "
                    f"(was {pos.entry_spread_bps:.1f}bps, "
                    f"{spread_ratio:.1f}x increase)",
                    pos.current_confidence, pos.remaining_edge_bps,
                    urgency="high",
                )

        # Depth dried up
        if pos.entry_book_depth_usd > 0 and pos.current_book_depth_usd > 0:
            depth_ratio = float(pos.current_book_depth_usd / pos.entry_book_depth_usd)
            if depth_ratio < 0.2:
                return self._close_result(
                    pos, "COST_DEPTH_THINNED",
                    f"depth=${float(pos.current_book_depth_usd):.0f} "
                    f"(was ${float(pos.entry_book_depth_usd):.0f}, "
                    f"{depth_ratio:.0%} remaining)",
                    pos.current_confidence, pos.remaining_edge_bps,
                    urgency="high",
                )

        # Funding rate flipped (for funding arb)
        if (pos.signal_source == "funding_rate"
                and pos.entry_funding_rate is not None
                and pos.current_funding_rate is not None):
            if ((pos.entry_funding_rate > 0 and pos.current_funding_rate < 0) or
                    (pos.entry_funding_rate < 0 and pos.current_funding_rate > 0)):
                return self._close_result(
                    pos, "COST_FUNDING_ADVERSE",
                    f"funding flipped from {pos.entry_funding_rate:.4%} "
                    f"to {pos.current_funding_rate:.4%}",
                    pos.current_confidence, pos.remaining_edge_bps,
                    urgency="high",
                )

        return None

    # ═══════════════════════════════════════════════════════════
    # STEP 8: RECALCULATE OPTIMAL SIZE
    # ═══════════════════════════════════════════════════════════

    def _recalculate_optimal_size(
        self,
        pos: PositionContext,
        rescored_confidence: float,
        remaining_edge: _RemainingEdge,
        portfolio_state: Dict[str, Any],
    ) -> Decimal:
        """What does Kelly say the position size should be now?"""

        if remaining_edge.net_remaining_bps <= 0:
            return Decimal("0")

        # Use Kelly sizer if available
        if self.kelly_sizer is not None:
            try:
                equity = float(portfolio_state.get("equity", 10000))
                # Build a signal dict that KellyPositionSizer expects
                signal_dict = {
                    "signal_type": pos.strategy,
                    "pair": pos.pair,
                    "confidence": rescored_confidence,
                    "expected_edge_bps": remaining_edge.net_remaining_bps,
                }
                size = self.kelly_sizer.get_position_size(signal_dict, equity)
                return Decimal(str(max(0, size)))
            except Exception:
                pass

        # Fallback: simple proportional sizing
        # If confidence dropped by X%, optimal size drops proportionally
        confidence_ratio = rescored_confidence / max(pos.entry_confidence, 0.501)
        edge_ratio = max(0, remaining_edge.edge_ratio)
        scale = confidence_ratio * edge_ratio
        optimal = float(pos.entry_size_usd) * scale
        return Decimal(str(max(0, optimal)))

    # ═══════════════════════════════════════════════════════════
    # STEP 9: APPLY CONSTRAINTS
    # ═══════════════════════════════════════════════════════════

    def _apply_constraints(
        self,
        optimal_size: Decimal,
        pos: PositionContext,
        portfolio_state: Dict[str, Any],
    ) -> Decimal:
        """Apply regime and portfolio constraints to the optimal size."""

        # Regime adjustment
        if self.regime_detector is not None:
            multiplier = getattr(self.regime_detector, "get_size_multiplier", None)
            if multiplier is not None:
                try:
                    regime_mult = multiplier()
                    optimal_size = Decimal(str(float(optimal_size) * regime_mult))
                except Exception:
                    pass

        # Never allow size to exceed the original entry size
        # (re-evaluation can trim but shouldn't grow beyond entry without add logic)
        if optimal_size > pos.entry_size_usd * Decimal("1.5"):
            optimal_size = pos.entry_size_usd * Decimal("1.5")

        return max(Decimal("0"), optimal_size)

    # ═══════════════════════════════════════════════════════════
    # STEP 10: COMPARE AND DECIDE
    # ═══════════════════════════════════════════════════════════

    def _compare_and_decide(
        self,
        pos: PositionContext,
        optimal_size_usd: Decimal,
        rescored_confidence: float,
        remaining_edge: _RemainingEdge,
        timestamp: float,
        dynamic_close_threshold: Optional[float] = None,
    ) -> ReEvalResult:
        """
        The final decision: what to do with this position.

        Dead zones prevent churn:
          - Ratio 0.0-0.3:  Close entirely
          - Ratio 0.3-0.8:  TRIM to optimal size
          - Ratio 0.8-1.5:  HOLD (dead zone)
          - Ratio 1.5+:     ADD (only if confidence also improved)
        """
        if optimal_size_usd == 0 or pos.current_size_usd == 0:
            size_ratio = 0.0
        else:
            size_ratio = float(pos.current_size_usd / optimal_size_usd)

        close_threshold = dynamic_close_threshold or self._cfg(pos, "close_threshold")
        trim_threshold = self._cfg(pos, "trim_threshold")
        add_threshold = self._cfg(pos, "add_threshold")
        max_adjustments = self._cfg(pos, "max_adjustments_per_position")

        adjustment_count = len(pos.adjustments)
        adjustments_exhausted = adjustment_count >= max_adjustments

        # ── CLOSE: Optimal size is tiny or zero ──
        if size_ratio < close_threshold or optimal_size_usd == 0:
            reason_code = "EDGE_CONSUMED" if pos.is_profitable else "CONFIDENCE_DECAYED"
            return self._close_result(
                pos, reason_code,
                f"ratio={size_ratio:.2f}, optimal=${float(optimal_size_usd):.0f}, "
                f"confidence={rescored_confidence:.4f}",
                rescored_confidence, remaining_edge.net_remaining_bps,
                urgency="normal" if pos.is_profitable else "high",
            )

        # ── TRIM: Optimal is 30-80% of current ──
        if close_threshold <= size_ratio < trim_threshold:
            if adjustments_exhausted:
                if size_ratio < 0.5:
                    return self._close_result(
                        pos, "PROFIT_CLOSE",
                        f"max adjustments reached, ratio={size_ratio:.2f}",
                        rescored_confidence, remaining_edge.net_remaining_bps,
                        urgency="normal",
                    )
                return self._hold_result(pos, "MAX_ADJUSTMENTS")

            trim_amount = pos.current_size_usd - optimal_size_usd
            min_trim = self._cfg(pos, "min_trim_usd")
            if float(trim_amount) < min_trim:
                return self._hold_result(pos, "TRIM_TOO_SMALL")

            return ReEvalResult(
                position_id=pos.position_id,
                timestamp=timestamp,
                action="trim",
                reason=f"Optimal size ${float(optimal_size_usd):.0f} < "
                       f"current ${float(pos.current_size_usd):.0f}",
                reason_code="PROFIT_TRIM",
                rescored_confidence=rescored_confidence,
                remaining_edge_bps=remaining_edge.net_remaining_bps,
                optimal_size_usd=optimal_size_usd,
                current_size_usd=pos.current_size_usd,
                size_ratio=size_ratio,
                trim_amount_usd=trim_amount,
                urgency="normal",
            )

        # ── HOLD: Dead zone 0.8-1.5 ──
        if trim_threshold <= size_ratio <= add_threshold:
            return self._hold_result(pos, "WITHIN_TOLERANCE")

        # ── ADD: Optimal is 150%+ of current ──
        if size_ratio > add_threshold:
            if adjustments_exhausted:
                return self._hold_result(pos, "MAX_ADJUSTMENTS")

            # Only add if confidence ALSO improved
            if rescored_confidence <= pos.entry_confidence:
                return self._hold_result(pos, "CONFIDENCE_NOT_IMPROVED")

            add_amount = optimal_size_usd - pos.current_size_usd
            min_trim = self._cfg(pos, "min_trim_usd")
            if float(add_amount) < min_trim:
                return self._hold_result(pos, "ADD_TOO_SMALL")

            return ReEvalResult(
                position_id=pos.position_id,
                timestamp=timestamp,
                action="add",
                reason=f"Optimal size ${float(optimal_size_usd):.0f} > "
                       f"current ${float(pos.current_size_usd):.0f}, "
                       f"confidence improved to {rescored_confidence:.4f}",
                reason_code="REEVAL_ADD",
                rescored_confidence=rescored_confidence,
                remaining_edge_bps=remaining_edge.net_remaining_bps,
                optimal_size_usd=optimal_size_usd,
                current_size_usd=pos.current_size_usd,
                size_ratio=size_ratio,
                add_amount_usd=add_amount,
                urgency="normal",
            )

        # Fallback
        return self._hold_result(pos, "FALLBACK")

    # ═══════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════

    def _hold_result(self, pos: PositionContext, reason_code: str) -> ReEvalResult:
        opt_size = pos.current_optimal_size if pos.current_optimal_size > 0 else pos.current_size_usd
        return ReEvalResult(
            position_id=pos.position_id,
            timestamp=time.time(),
            action="hold",
            reason=REASON_CODES.get(reason_code, reason_code),
            reason_code=reason_code,
            rescored_confidence=pos.current_confidence,
            remaining_edge_bps=pos.remaining_edge_bps,
            optimal_size_usd=opt_size,
            current_size_usd=pos.current_size_usd,
            size_ratio=float(pos.current_size_usd / opt_size) if opt_size > 0 else 1.0,
            urgency="normal",
        )

    def _close_result(
        self, pos: PositionContext, reason_code: str, detail: str,
        rescored_confidence: float, remaining_edge_bps: float,
        urgency: str = "normal",
    ) -> ReEvalResult:
        return ReEvalResult(
            position_id=pos.position_id,
            timestamp=time.time(),
            action="close",
            reason=f"{REASON_CODES.get(reason_code, reason_code)}: {detail}",
            reason_code=reason_code,
            rescored_confidence=rescored_confidence,
            remaining_edge_bps=remaining_edge_bps,
            optimal_size_usd=Decimal("0"),
            current_size_usd=pos.current_size_usd,
            size_ratio=0.0,
            urgency=urgency,
        )

    @staticmethod
    def _last_adjustment_time(pos: PositionContext) -> float:
        if not pos.adjustments:
            return 0.0
        return pos.adjustments[-1].get("timestamp", 0.0)

    # ═══════════════════════════════════════════════════════════
    # METRICS
    # ═══════════════════════════════════════════════════════════

    def get_metrics(self) -> Dict[str, Any]:
        """For Telegram reporting and monitoring."""
        return {
            "total_reevaluations": self._total_reevaluations,
            "total_trims": self._total_trims,
            "total_early_closes": self._total_early_closes,
            "total_adds": self._total_adds,
            "total_holds": self._total_holds,
            "profit_from_trims_usd": float(self._profit_from_trims_usd),
            "trim_rate_pct": (
                self._total_trims / self._total_reevaluations * 100
                if self._total_reevaluations > 0 else 0
            ),
        }
