"""
Self-Correcting Portfolio Engine (A2)
======================================
Maintains an *ideal* portfolio state derived from active signals, reads the
*actual* state from the position manager / exchange, computes the drift
between them, and generates correction orders to close the gap.

This is NOT a replacement for ``unified_portfolio_engine.py`` (which handles
cross-product correlation-aware sizing).  This module sits *downstream*:
it consumes already-sized signals and ensures that what the system *wants*
to hold actually matches what it *does* hold.

The reconciliation loop runs on a configurable interval and is designed
to be awaited from the main asyncio event loop.

Config lives under ``medallion_portfolio_engine`` in config.json.

Design principle: "if unexpected, do nothing" -- every public method
catches exceptions internally and logs rather than raising, so a
portfolio engine failure never kills a live trading loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from core.data_structures import PositionContext, ReEvalResult, REASON_CODES

logger = logging.getLogger(__name__)

# Try to import PositionReEvaluator (may not be available yet during bootstrap)
try:
    from portfolio.position_reevaluator import PositionReEvaluator
    REEVALUATOR_AVAILABLE = True
except ImportError:
    REEVALUATOR_AVAILABLE = False

# Try to import MHPE components (Doc 11)
try:
    from intelligence.microstructure_predictor import MicrostructurePredictor
    from intelligence.statistical_predictor import StatisticalPredictor
    from intelligence.regime_predictor import RegimePredictor
    from intelligence.multi_horizon_estimator import MultiHorizonEstimator
    MHPE_AVAILABLE = True
except ImportError:
    MHPE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default configuration (used when keys are missing from config.json)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "reconciliation_interval_seconds": 5,
    "drift_threshold_pct": 2.0,
    "max_correction_cost_bps": 3.0,
    "max_corrections_per_cycle": 5,
    "target_net_exposure": 0.0,
    "max_single_position_pct": 15.0,
    "max_leverage": 3.0,
    "signal_ttl_default_seconds": 60,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PortfolioTarget:
    """The ideal portfolio the engine wants to hold right now."""
    positions: Dict[str, float]        # pair -> signed target size (USD)
    net_exposure: float                # sum of signed positions
    max_position_pct: float            # hard cap per position
    max_leverage: float                # hard cap on leverage
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class PortfolioActual:
    """Snapshot of what we actually hold (read from position manager)."""
    positions: Dict[str, float]        # pair -> signed size (USD)
    pending_orders: Dict[str, Any]     # pair -> order info
    cash: float
    equity: float                      # cash + unrealised
    unrealized_pnl: float
    net_exposure: float


@dataclass
class CorrectionOrder:
    """A single order that the engine wants to submit to close drift."""
    pair: str
    side: str                          # "BUY" or "SELL"
    quantity: float                    # in USD notional
    reason: str
    priority: int = 0                  # lower = more urgent
    max_cost_bps: float = 3.0         # max acceptable execution cost


# ---------------------------------------------------------------------------
# PortfolioEngine
# ---------------------------------------------------------------------------

class PortfolioEngine:
    """
    Self-correcting portfolio engine.

    Typical integration::

        engine = PortfolioEngine(config, devil_tracker=dt)
        engine.ingest_signal({"pair": "BTC-USD", "side": "BUY",
                              "strength": 0.12, "confidence": 0.7,
                              "signal_type": "stat_arb"})
        # In the async event loop:
        await engine.reconciliation_loop()
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cost_model: Optional[Any] = None,
        devil_tracker: Optional[Any] = None,
        position_manager: Optional[Any] = None,
        kelly_sizer: Optional[Any] = None,
        regime_detector: Optional[Any] = None,
    ):
        full_config = config or {}
        raw = full_config.get("medallion_portfolio_engine", {})
        self.cfg: Dict[str, Any] = {**_DEFAULT_CONFIG, **raw}

        self.cost_model = cost_model
        self.devil_tracker = devil_tracker
        self.position_manager = position_manager
        self.kelly_sizer = kelly_sizer
        self.regime_detector = regime_detector

        # Active signals -- keyed by (pair, signal_type)
        # Each value is the full signal dict plus an ``_expires_at`` epoch.
        self._signals: Dict[tuple, Dict[str, Any]] = {}

        # Last known actual state (populated by fetch_actual)
        self._last_actual: Optional[PortfolioActual] = None

        # Safety: track corrections per cycle to cap runaway loops
        self._corrections_this_cycle: int = 0

        # Flag for graceful shutdown
        self._running: bool = False

        # ── Multi-Horizon Probability Estimator (Doc 11) ──
        self.mhpe: Optional[Any] = None
        mhpe_cfg = full_config.get("multi_horizon_estimator", {})
        if mhpe_cfg.get("enabled", False) and MHPE_AVAILABLE:
            try:
                micro_pred = MicrostructurePredictor(
                    mhpe_cfg.get("microstructure_predictor", {})
                )
                stat_pred = StatisticalPredictor(
                    mhpe_cfg.get("statistical_predictor", {})
                )
                regime_pred = RegimePredictor(
                    mhpe_cfg.get("regime_predictor", {}),
                    regime_detector=regime_detector,
                )
                self.mhpe = MultiHorizonEstimator(
                    config=mhpe_cfg,
                    micro_predictor=micro_pred,
                    stat_predictor=stat_pred,
                    regime_predictor=regime_pred,
                )
                logger.info("MHPE: ACTIVE — probability cones across 7 horizons")
            except Exception as exc:
                logger.warning("MHPE init failed: %s", exc)

        # ── Continuous Position Re-evaluation (Doc 10) ──
        self.reevaluator: Optional[Any] = None
        self.open_positions: Dict[str, PositionContext] = {}
        reeval_cfg = full_config.get("reevaluation", {})
        if reeval_cfg.get("enabled", False) and REEVALUATOR_AVAILABLE:
            try:
                self.reevaluator = PositionReEvaluator(
                    config=reeval_cfg,
                    cost_model=cost_model,
                    kelly_sizer=kelly_sizer,
                    regime_detector=regime_detector,
                    devil_tracker=devil_tracker,
                    mhpe=self.mhpe,
                )
                logger.info("PositionReEvaluator: ACTIVE — continuous position management")
            except Exception as exc:
                logger.warning("PositionReEvaluator init failed: %s", exc)

        logger.info(
            "PortfolioEngine initialised: recon_interval=%ss, drift_thresh=%.1f%%, "
            "max_corrections=%d, reevaluator=%s",
            self.cfg["reconciliation_interval_seconds"],
            self.cfg["drift_threshold_pct"],
            self.cfg["max_corrections_per_cycle"],
            "active" if self.reevaluator else "disabled",
        )

    # ------------------------------------------------------------------
    # Signal ingestion
    # ------------------------------------------------------------------

    def ingest_signal(self, signal_dict: Dict[str, Any]) -> bool:
        """
        Receive a trading signal and store it until it expires.

        Expected keys in *signal_dict*:
            pair, side, strength, confidence, signal_type
        Optional:
            ttl_seconds (overrides signal_ttl_default_seconds)
            notional_usd (pre-sized position amount)
        """
        try:
            pair = signal_dict.get("pair")
            signal_type = signal_dict.get("signal_type", "unknown")
            if not pair:
                logger.warning("PortfolioEngine: signal missing 'pair', ignoring")
                return False

            ttl = signal_dict.get(
                "ttl_seconds", self.cfg["signal_ttl_default_seconds"]
            )
            signal_dict["_expires_at"] = time.time() + ttl
            signal_dict["_ingested_at"] = datetime.now(timezone.utc).isoformat()

            key = (pair, signal_type)
            self._signals[key] = signal_dict
            logger.debug(
                "PortfolioEngine: ingested signal pair=%s type=%s strength=%.3f",
                pair, signal_type, signal_dict.get("strength", 0),
            )
            return True
        except Exception as exc:
            logger.error("PortfolioEngine.ingest_signal failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Target computation
    # ------------------------------------------------------------------

    def compute_target(self) -> PortfolioTarget:
        """
        Build the ideal portfolio from all active (non-expired) signals.

        Steps:
          1. Prune expired signals.
          2. For each pair, aggregate across signal types (sum of signed
             strengths weighted by confidence).
          3. Rank by magnitude and cap each position at max_single_position_pct.
          4. Enforce max_leverage on total notional.
        """
        try:
            self._prune_expired_signals()

            # Aggregate per-pair target notional.
            # If the signal supplies ``notional_usd`` use it directly;
            # otherwise fall back to strength * confidence as a *fraction*
            # of equity (needs equity from actual, default to 10 000).
            equity = 10_000.0
            if self._last_actual and self._last_actual.equity > 0:
                equity = self._last_actual.equity

            pair_agg: Dict[str, float] = {}
            for (pair, _), sig in self._signals.items():
                notional = sig.get("notional_usd")
                if notional is None:
                    strength = sig.get("strength", 0.0)
                    confidence = sig.get("confidence", 0.5)
                    side_mult = 1.0 if (sig.get("side", "BUY")).upper() == "BUY" else -1.0
                    notional = side_mult * abs(strength) * confidence * equity
                else:
                    side_mult = 1.0 if (sig.get("side", "BUY")).upper() == "BUY" else -1.0
                    notional = side_mult * abs(notional)

                pair_agg[pair] = pair_agg.get(pair, 0.0) + notional

            max_pos_usd = equity * (self.cfg["max_single_position_pct"] / 100.0)
            max_leverage = self.cfg["max_leverage"]

            # Cap per-position
            for pair in pair_agg:
                if abs(pair_agg[pair]) > max_pos_usd:
                    sign = 1.0 if pair_agg[pair] >= 0 else -1.0
                    pair_agg[pair] = sign * max_pos_usd

            # Cap total leverage
            total_abs = sum(abs(v) for v in pair_agg.values())
            if total_abs > equity * max_leverage and total_abs > 0:
                scale = (equity * max_leverage) / total_abs
                pair_agg = {k: v * scale for k, v in pair_agg.items()}

            net_exposure = sum(pair_agg.values())

            target = PortfolioTarget(
                positions=pair_agg,
                net_exposure=net_exposure,
                max_position_pct=self.cfg["max_single_position_pct"],
                max_leverage=max_leverage,
            )
            logger.debug(
                "PortfolioEngine: target computed, %d positions, net_exposure=%.2f",
                len(pair_agg), net_exposure,
            )
            return target

        except Exception as exc:
            logger.error("PortfolioEngine.compute_target failed: %s", exc)
            return PortfolioTarget(
                positions={}, net_exposure=0.0,
                max_position_pct=self.cfg["max_single_position_pct"],
                max_leverage=self.cfg["max_leverage"],
            )

    # ------------------------------------------------------------------
    # Actual state
    # ------------------------------------------------------------------

    def fetch_actual(self) -> PortfolioActual:
        """
        Read the actual portfolio state from the position manager.

        If no position_manager is wired in, returns a zero-state snapshot
        so the rest of the pipeline still works (useful in paper mode).
        """
        try:
            if self.position_manager is None:
                actual = PortfolioActual(
                    positions={}, pending_orders={},
                    cash=0.0, equity=10_000.0,
                    unrealized_pnl=0.0, net_exposure=0.0,
                )
                self._last_actual = actual
                return actual

            pm = self.position_manager
            all_positions = pm.get_all_positions() if hasattr(pm, "get_all_positions") else []

            positions: Dict[str, float] = {}
            unrealized = 0.0
            for pos in all_positions:
                pair = getattr(pos, "product_id", None) or pos.get("product_id", "")
                size = getattr(pos, "size", 0.0) if hasattr(pos, "size") else pos.get("size", 0.0)
                entry = getattr(pos, "entry_price", 0.0) if hasattr(pos, "entry_price") else pos.get("entry_price", 0.0)
                side = getattr(pos, "side", None)
                if side is not None:
                    side_val = side.value if hasattr(side, "value") else str(side)
                else:
                    side_val = "LONG"
                sign = 1.0 if side_val == "LONG" else -1.0
                notional = sign * size * entry
                positions[pair] = positions.get(pair, 0.0) + notional

                upnl = getattr(pos, "unrealized_pnl", 0.0)
                if isinstance(upnl, (int, float)):
                    unrealized += upnl

            net_exp = sum(positions.values())
            # Cash/equity from summary if available
            equity = 10_000.0
            cash = 0.0
            if hasattr(pm, "get_position_summary"):
                try:
                    summ = pm.get_position_summary()
                    equity = getattr(summ, "total_exposure_usd", 10_000.0) or 10_000.0
                except Exception:
                    pass

            actual = PortfolioActual(
                positions=positions,
                pending_orders={},
                cash=cash,
                equity=equity,
                unrealized_pnl=unrealized,
                net_exposure=net_exp,
            )
            self._last_actual = actual
            return actual

        except Exception as exc:
            logger.error("PortfolioEngine.fetch_actual failed: %s", exc)
            actual = PortfolioActual(
                positions={}, pending_orders={},
                cash=0.0, equity=10_000.0,
                unrealized_pnl=0.0, net_exposure=0.0,
            )
            self._last_actual = actual
            return actual

    # ------------------------------------------------------------------
    # Drift computation
    # ------------------------------------------------------------------

    def compute_drift(self) -> Dict[str, float]:
        """
        Compute the difference between target and actual for every pair.

        Returns ``{pair: drift_usd}`` where positive means we need to buy
        more and negative means we need to sell.  Only pairs whose drift
        exceeds *drift_threshold_pct* of equity are included.
        """
        try:
            target = self.compute_target()
            actual = self.fetch_actual()

            equity = actual.equity if actual.equity > 0 else 10_000.0
            threshold = equity * (self.cfg["drift_threshold_pct"] / 100.0)

            all_pairs = set(target.positions.keys()) | set(actual.positions.keys())
            drift: Dict[str, float] = {}

            for pair in all_pairs:
                tgt = target.positions.get(pair, 0.0)
                act = actual.positions.get(pair, 0.0)
                d = tgt - act
                if abs(d) >= threshold:
                    drift[pair] = round(d, 2)

            if drift:
                logger.info(
                    "PortfolioEngine: drift detected in %d pair(s): %s",
                    len(drift),
                    ", ".join(f"{k}={v:+.0f}" for k, v in drift.items()),
                )
            return drift

        except Exception as exc:
            logger.error("PortfolioEngine.compute_drift failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Correction generation
    # ------------------------------------------------------------------

    def generate_corrections(
        self, drift: Dict[str, float]
    ) -> List[CorrectionOrder]:
        """
        Turn a drift map into a prioritised list of correction orders.

        Rules:
        - Skip pairs where the estimated correction cost exceeds
          *max_correction_cost_bps*.
        - Limit to *max_corrections_per_cycle* orders.
        - Priority is set by absolute drift magnitude (biggest first).
        """
        try:
            if not drift:
                return []

            max_cost = self.cfg["max_correction_cost_bps"]
            max_corrections = self.cfg["max_corrections_per_cycle"]

            corrections: List[CorrectionOrder] = []
            sorted_pairs = sorted(drift.items(), key=lambda x: abs(x[1]), reverse=True)

            for pair, d in sorted_pairs:
                if len(corrections) >= max_corrections:
                    break

                side = "BUY" if d > 0 else "SELL"
                quantity = abs(d)

                # Estimate cost if a cost model is available
                est_cost_bps = 0.0
                if self.cost_model and hasattr(self.cost_model, "estimate_round_trip_cost"):
                    try:
                        est_cost_bps = self.cost_model.estimate_round_trip_cost() * 10_000.0
                    except Exception:
                        est_cost_bps = 0.0

                if est_cost_bps > max_cost and max_cost > 0:
                    logger.info(
                        "PortfolioEngine: skipping correction for %s, "
                        "cost %.1f bps > cap %.1f bps",
                        pair, est_cost_bps, max_cost,
                    )
                    continue

                corrections.append(CorrectionOrder(
                    pair=pair,
                    side=side,
                    quantity=quantity,
                    reason=f"drift_correction({d:+.0f} USD)",
                    priority=len(corrections),
                    max_cost_bps=max_cost,
                ))

            if corrections:
                logger.info(
                    "PortfolioEngine: generated %d correction(s)", len(corrections),
                )
            return corrections

        except Exception as exc:
            logger.error("PortfolioEngine.generate_corrections failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Correction execution
    # ------------------------------------------------------------------

    def execute_corrections(self, corrections: List[CorrectionOrder]) -> int:
        """
        Log each correction to the devil tracker (if wired in).

        Actual order placement is the caller's responsibility (the engine
        should not talk directly to an exchange).  This method records the
        intent so the devil tracker can later compare it against the fill.

        Returns the number of corrections logged.
        """
        logged = 0
        for corr in corrections:
            try:
                if self.devil_tracker is not None:
                    trade_id = self.devil_tracker.record_signal_detection(
                        signal_type="portfolio_correction",
                        pair=corr.pair,
                        exchange="internal",
                        price=0.0,  # will be filled in by the order path
                        side=corr.side,
                    )
                    if trade_id:
                        logger.info(
                            "PortfolioEngine: correction logged devil_id=%s %s %s %.0f USD (%s)",
                            trade_id, corr.side, corr.pair, corr.quantity, corr.reason,
                        )
                        logged += 1
                else:
                    logger.info(
                        "PortfolioEngine: correction (no tracker) %s %s %.0f USD (%s)",
                        corr.side, corr.pair, corr.quantity, corr.reason,
                    )
                    logged += 1
            except Exception as exc:
                logger.error(
                    "PortfolioEngine.execute_corrections: error on %s: %s",
                    corr.pair, exc,
                )
        return logged

    # ------------------------------------------------------------------
    # Main reconciliation loop
    # ------------------------------------------------------------------

    async def reconciliation_loop(self) -> None:
        """
        Async loop that periodically reconciles target vs. actual.

        Now includes continuous re-evaluation of all open positions (Doc 10).
        Call ``engine.stop()`` to break the loop gracefully.
        """
        self._running = True
        interval = self.cfg["reconciliation_interval_seconds"]
        logger.info(
            "PortfolioEngine: reconciliation loop started (every %ss)", interval,
        )

        while self._running:
            try:
                self._corrections_this_cycle = 0

                # ═══ CONTINUOUS RE-EVALUATION (Doc 10) ═══
                if self.reevaluator and self.open_positions:
                    try:
                        portfolio_state = self._get_portfolio_state()
                        market_state = self._get_market_state()
                        reeval_results = self.reevaluator.reevaluate_all(
                            positions=list(self.open_positions.values()),
                            portfolio_state=portfolio_state,
                            market_state=market_state,
                        )
                        for result in reeval_results:
                            self._execute_reeval_action(result)
                    except Exception as exc:
                        logger.error(
                            "PortfolioEngine: re-evaluation error: %s", exc,
                        )

                # ═══ EXISTING: Drift correction ═══
                drift = self.compute_drift()
                if drift:
                    corrections = self.generate_corrections(drift)
                    if corrections:
                        self._corrections_this_cycle = self.execute_corrections(
                            corrections
                        )
            except Exception as exc:
                logger.error(
                    "PortfolioEngine: reconciliation cycle error: %s", exc,
                )

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logger.info("PortfolioEngine: reconciliation loop cancelled")
                break

        logger.info("PortfolioEngine: reconciliation loop stopped")

    def stop(self) -> None:
        """Signal the reconciliation loop to exit on the next iteration."""
        self._running = False

    # ------------------------------------------------------------------
    # Re-evaluation helpers (Doc 10)
    # ------------------------------------------------------------------

    def _get_portfolio_state(self) -> Dict[str, Any]:
        """Build portfolio state dict for the re-evaluator."""
        actual = self._last_actual
        equity = actual.equity if actual else 10_000.0
        return {
            "equity": equity,
            "available_capital": equity,
            "daily_loss_limit_hit": False,
            "system_halted": False,
        }

    def _get_market_state(self) -> Dict[str, Any]:
        """Build market state dict for the re-evaluator from position manager."""
        market: Dict[str, Any] = {}
        try:
            if self.position_manager is not None and hasattr(self.position_manager, "get_all_positions"):
                for pos in self.position_manager.get_all_positions():
                    pair = getattr(pos, "product_id", None) or pos.get("product_id", "")
                    if pair:
                        market[pair] = {
                            "last_price": getattr(pos, "current_price", 0) or pos.get("current_price", 0),
                        }
        except Exception:
            pass
        return market

    def _execute_reeval_action(self, result: ReEvalResult) -> None:
        """Execute the action decided by the re-evaluator."""
        pos = self.open_positions.get(result.position_id)
        if pos is None:
            return

        if result.action == "hold":
            logger.debug(
                "REEVAL HOLD %s | %s | confidence=%.4f | edge=%.1fbps",
                pos.pair, result.reason_code,
                result.rescored_confidence, result.remaining_edge_bps,
            )
            return

        if result.action == "close":
            logger.info(
                "REEVAL CLOSE %s | %s | pnl=%.1fbps | %s",
                pos.pair, result.reason_code,
                pos.unrealized_pnl_bps, result.reason,
            )
            # Record in DevilTracker
            if self.devil_tracker and hasattr(self.devil_tracker, "record_exit"):
                self.devil_tracker.record_exit(
                    position_id=pos.position_id,
                    pair=pos.pair,
                    side=pos.side,
                    entry_price=float(pos.entry_price),
                    exit_price=float(pos.current_price),
                    size=float(pos.current_size),
                    estimated_cost_bps=pos.entry_cost_estimate_bps,
                    actual_cost_bps=pos.current_cost_to_exit_bps,
                    reason_code=result.reason_code,
                    hold_time_seconds=pos.age_seconds,
                    adjustments=len(pos.adjustments),
                    pnl_bps=pos.unrealized_pnl_bps,
                )
            pos.adjustments.append({
                "timestamp": time.time(),
                "action": "close",
                "amount_usd": float(pos.current_size_usd),
                "reason": result.reason_code,
                "pnl_bps": pos.unrealized_pnl_bps,
            })
            del self.open_positions[pos.position_id]

        elif result.action == "trim":
            logger.info(
                "REEVAL TRIM %s | $%.0f of $%.0f | ratio=%.2f | %s",
                pos.pair, float(result.trim_amount_usd or 0),
                float(pos.current_size_usd), result.size_ratio, result.reason,
            )
            trim_usd = result.trim_amount_usd or Decimal("0")
            pos.current_size_usd -= trim_usd
            if pos.current_price > 0:
                pos.current_size = pos.current_size_usd / pos.current_price
            pos.total_trimmed_usd += trim_usd
            pos.adjustments.append({
                "timestamp": time.time(),
                "action": "trim",
                "amount_usd": float(trim_usd),
                "reason": result.reason_code,
                "pnl_bps": pos.unrealized_pnl_bps,
            })
            if self.devil_tracker and hasattr(self.devil_tracker, "record_trim"):
                self.devil_tracker.record_trim(
                    position_id=pos.position_id,
                    pair=pos.pair,
                    trim_size=float(trim_usd / pos.current_price) if pos.current_price > 0 else 0,
                    trim_price=float(pos.current_price),
                    actual_cost_bps=pos.current_cost_to_exit_bps,
                )
            if self.reevaluator:
                self.reevaluator._total_trims += 1

        elif result.action == "add":
            logger.info(
                "REEVAL ADD %s | +$%.0f to $%.0f | confidence=%.4f | %s",
                pos.pair, float(result.add_amount_usd or 0),
                float(pos.current_size_usd), result.rescored_confidence, result.reason,
            )
            add_usd = result.add_amount_usd or Decimal("0")
            pos.current_size_usd += add_usd
            if pos.current_price > 0:
                pos.current_size = pos.current_size_usd / pos.current_price
            pos.total_added_usd += add_usd
            pos.adjustments.append({
                "timestamp": time.time(),
                "action": "add",
                "amount_usd": float(add_usd),
                "reason": result.reason_code,
                "pnl_bps": 0,
            })
            if self.reevaluator:
                self.reevaluator._total_adds += 1

    def register_position(self, pos: PositionContext) -> None:
        """Register a new position for continuous re-evaluation."""
        self.open_positions[pos.position_id] = pos
        logger.info(
            "Position registered for re-evaluation: %s %s %s $%.0f",
            pos.position_id, pos.pair, pos.side, float(pos.entry_size_usd),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prune_expired_signals(self) -> int:
        """Remove signals past their TTL.  Returns the count removed."""
        now = time.time()
        expired_keys = [
            k for k, v in self._signals.items()
            if v.get("_expires_at", 0) < now
        ]
        for k in expired_keys:
            del self._signals[k]
        if expired_keys:
            logger.debug(
                "PortfolioEngine: pruned %d expired signal(s)", len(expired_keys),
            )
        return len(expired_keys)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a JSON-safe status snapshot for dashboards."""
        try:
            status = {
                "active_signals": len(self._signals),
                "corrections_last_cycle": self._corrections_this_cycle,
                "running": self._running,
                "config": {k: v for k, v in self.cfg.items()},
                "last_actual": {
                    "positions": self._last_actual.positions if self._last_actual else {},
                    "net_exposure": self._last_actual.net_exposure if self._last_actual else 0.0,
                    "equity": self._last_actual.equity if self._last_actual else 0.0,
                } if self._last_actual else None,
                "open_positions": len(self.open_positions),
                "reevaluator_active": self.reevaluator is not None,
            }
            if self.reevaluator:
                status["reevaluation_metrics"] = self.reevaluator.get_metrics()
            return status
        except Exception as exc:
            logger.error("PortfolioEngine.get_status failed: %s", exc)
            return {"error": str(exc)}
