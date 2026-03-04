"""Token Spray Engine — rapid-fire small positions with fast exits.

Replaces the legacy ML decision path (large positions, long holds) with many
small ($10-50) tokens that enter on signal and exit fast at 50% of expected
move or within 5 minutes.  Edge decays exponentially with hold time — this
engine harvests the <5-minute sweet spot.
"""

import asyncio
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SprayToken — one micro-position
# ---------------------------------------------------------------------------

@dataclass
class SprayToken:
    """A single micro-position opened by the spray engine."""
    token_id: str
    pair: str
    side: str                       # "LONG" / "SHORT"
    entry_price: float
    entry_time: float               # time.time() epoch
    size_usd: float
    size_units: float
    direction_rule: str             # "ml_ensemble" / "momentum" / "mean_reversion" / "stat_arb" / "mixed"
    vol_regime: str                 # "low" / "medium" / "high"
    expected_move_bps: float
    target_bps: float
    stop_bps: float
    max_hold_seconds: float
    observation_mode: bool
    weighted_signal: float = 0.0
    confidence: float = 0.0
    # Exit fields — filled on close
    status: str = "open"            # "open" / "closed"
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_pnl_bps: Optional[float] = None
    exit_pnl_usd: Optional[float] = None
    hold_time_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# TokenSprayEngine
# ---------------------------------------------------------------------------

class TokenSprayEngine:
    """Rapid-fire micro-position engine.

    Entry: called once per pair per cycle via ``spray()``.
    Exit:  background loop every ``exit_check_interval_seconds`` via
           ``check_exits()`` which evaluates target / stop / time / edge.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        db_path: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.log = logger or logging.getLogger(__name__)
        self.db_path = db_path
        self._cfg = config

        # Tunable parameters (all from config with defaults)
        self.token_size_usd: float = config.get("token_size_usd", 20.0)
        self.max_budget_usd: float = config.get("max_budget_usd", 2000.0)
        self.max_open_tokens: int = config.get("max_open_tokens", 100)
        self.max_tokens_per_pair: int = config.get("max_tokens_per_pair", 5)
        self.min_signal_strength: float = config.get("min_signal_strength", 0.02)
        self.min_confidence: float = config.get("min_confidence", 0.42)
        self.target_fraction: float = config.get("target_fraction", 0.5)
        self.stop_bps: float = config.get("stop_bps", 50.0)
        self.max_hold_seconds: float = config.get("max_hold_seconds", 300.0)
        self.edge_consumed_min_age: float = config.get("edge_consumed_min_age_seconds", 120.0)
        self.cooldown_seconds: float = config.get("cooldown_seconds", 30.0)
        self.exit_check_interval: float = config.get("exit_check_interval_seconds", 5.0)
        self.vol_scaling: Dict[str, float] = config.get("vol_scaling", {
            "low": 1.0, "medium": 0.7, "high": 0.4,
        })
        self.observation_mode: bool = config.get("observation_mode", True)

        # Runtime state
        self.open_tokens: Dict[str, SprayToken] = {}
        self.budget_deployed_usd: float = 0.0
        self._pair_last_spray: Dict[str, float] = {}   # pair → epoch of last spray
        self._exit_task: Optional[asyncio.Task] = None
        self._running = False

        # Lifetime counters
        self._total_sprayed: int = 0
        self._total_closed: int = 0
        self._total_pnl_usd: float = 0.0

        self.log.info(
            f"TokenSprayEngine: size=${self.token_size_usd} | "
            f"budget=${self.max_budget_usd} | max_tokens={self.max_open_tokens} | "
            f"obs_mode={self.observation_mode}"
        )

    # ------------------------------------------------------------------
    # spray() — Entry point called from per-pair trading loop
    # ------------------------------------------------------------------

    async def spray(
        self,
        pair: str,
        weighted_signal: float,
        contributions: Dict[str, float],
        ml_package: Any,
        market_data: Dict[str, Any],
        confidence: Optional[float] = None,
    ) -> Optional[SprayToken]:
        """Evaluate whether to open a micro-position for *pair*.

        Returns the SprayToken if one was opened, else None.
        """
        # ── Gate 1: signal strength ──
        if abs(weighted_signal) < self.min_signal_strength:
            return None

        # ── Gate 2: confidence (from ml_package if not provided) ──
        if confidence is None and ml_package and hasattr(ml_package, "confidence_score"):
            confidence = ml_package.confidence_score
        if confidence is None:
            confidence = 0.5
        if confidence < self.min_confidence:
            return None

        # ── Gate 3: budget / capacity ──
        if len(self.open_tokens) >= self.max_open_tokens:
            return None
        if self.budget_deployed_usd + self.token_size_usd > self.max_budget_usd:
            return None

        # ── Gate 4: per-pair limit ──
        pair_count = sum(1 for t in self.open_tokens.values() if t.pair == pair)
        if pair_count >= self.max_tokens_per_pair:
            return None

        # ── Gate 5: cooldown ──
        now = time.time()
        last = self._pair_last_spray.get(pair, 0.0)
        if now - last < self.cooldown_seconds:
            return None

        # ── Direction ──
        side = "LONG" if weighted_signal > 0 else "SHORT"

        # ── Classify direction rule (for A/B testing) ──
        direction_rule = self._classify_direction_rule(contributions, weighted_signal)

        # ── Volatility regime & size scaling ──
        vol_regime = self._calc_vol_regime(market_data)
        vol_scale = self.vol_scaling.get(vol_regime, 1.0)
        size_usd = self.token_size_usd * vol_scale

        # ── Expected move → target ──
        expected_move_bps = self._estimate_expected_move(weighted_signal, market_data)
        target_bps = expected_move_bps * self.target_fraction

        # ── Compute size in units ──
        ticker = market_data.get("ticker", {})
        current_price = float(ticker.get("price", 0.0))
        if current_price <= 0:
            return None
        size_units = size_usd / current_price

        # ── Build token ──
        token = SprayToken(
            token_id=uuid.uuid4().hex[:12],
            pair=pair,
            side=side,
            entry_price=current_price,
            entry_time=now,
            size_usd=size_usd,
            size_units=size_units,
            direction_rule=direction_rule,
            vol_regime=vol_regime,
            expected_move_bps=expected_move_bps,
            target_bps=target_bps,
            stop_bps=self.stop_bps,
            max_hold_seconds=self.max_hold_seconds,
            observation_mode=self.observation_mode,
            weighted_signal=weighted_signal,
            confidence=confidence,
        )

        # Register
        self.open_tokens[token.token_id] = token
        self.budget_deployed_usd += size_usd
        self._pair_last_spray[pair] = now
        self._total_sprayed += 1

        # Persist entry
        self._log_to_db(token)

        self.log.info(
            f"SPRAY {'(OBS) ' if self.observation_mode else ''}"
            f"{side} {pair} ${size_usd:.0f} @ {current_price:.4f} | "
            f"rule={direction_rule} vol={vol_regime} "
            f"target={target_bps:.1f}bps stop={self.stop_bps:.0f}bps | "
            f"signal={weighted_signal:.4f} conf={confidence:.3f} "
            f"[{len(self.open_tokens)} open, ${self.budget_deployed_usd:.0f} deployed]"
        )
        return token

    # ------------------------------------------------------------------
    # check_exits() — called every N seconds
    # ------------------------------------------------------------------

    async def check_exits(self, price_fetcher: Callable) -> List[SprayToken]:
        """Evaluate all open tokens for exit conditions.

        *price_fetcher* is ``async def(pairs: List[str]) -> Dict[str, float]``
        returning {pair: current_price}.
        """
        if not self.open_tokens:
            return []

        pairs = list({t.pair for t in self.open_tokens.values()})
        try:
            prices = await price_fetcher(pairs)
        except Exception as e:
            self.log.warning(f"Spray price fetch failed: {e}")
            return []

        closed: List[SprayToken] = []
        now = time.time()

        for token in list(self.open_tokens.values()):
            price = prices.get(token.pair)
            if price is None or price <= 0:
                continue

            age = now - token.entry_time
            pnl_bps = self._calc_pnl_bps(token, price)

            exit_reason: Optional[str] = None

            # Priority 1: Target hit
            if pnl_bps >= token.target_bps:
                exit_reason = "target"
            # Priority 2: Stop hit
            elif pnl_bps <= -token.stop_bps:
                exit_reason = "stop"
            # Priority 3: Time expired
            elif age >= token.max_hold_seconds:
                exit_reason = "time_expired"
            # Priority 4: Edge consumed (profitable after min age)
            elif pnl_bps > 0 and age >= self.edge_consumed_min_age:
                exit_reason = "edge_consumed"

            if exit_reason:
                self._close_token(token, price, now, exit_reason, pnl_bps)
                closed.append(token)

        return closed

    # ------------------------------------------------------------------
    # Background exit loop
    # ------------------------------------------------------------------

    async def start_exit_loop(self, price_fetcher: Callable) -> None:
        """Start the background exit check loop."""
        if self._running:
            return
        self._running = True
        self._exit_task = asyncio.ensure_future(self._exit_loop(price_fetcher))
        self.log.info(
            f"TokenSprayEngine exit loop started (interval={self.exit_check_interval}s)"
        )

    async def stop_exit_loop(self) -> None:
        """Stop the background exit check loop."""
        self._running = False
        if self._exit_task:
            self._exit_task.cancel()
            try:
                await self._exit_task
            except asyncio.CancelledError:
                pass
            self._exit_task = None
        self.log.info("TokenSprayEngine exit loop stopped")

    async def _exit_loop(self, price_fetcher: Callable) -> None:
        """Internal loop that periodically calls check_exits."""
        while self._running:
            try:
                await self.check_exits(price_fetcher)
            except Exception as e:
                self.log.warning(f"Spray exit loop error: {e}")
            await asyncio.sleep(self.exit_check_interval)

    # ------------------------------------------------------------------
    # Status (for dashboard)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine status dict for the /api/token-spray/status endpoint."""
        open_count = len(self.open_tokens)
        return {
            "active": True,
            "observation_mode": self.observation_mode,
            "spray_interval_seconds": self.exit_check_interval,
            "total_sprayed": self._total_sprayed,
            "total_open": open_count,
            "total_closed": self._total_closed,
            "max_open": self.max_open_tokens,
            "today_pnl_usd": round(self._total_pnl_usd, 4),
            "today_tokens": self._total_closed,
            "budget": {
                "deployed_usd": round(self.budget_deployed_usd, 2),
                "total_capital": self.max_budget_usd,
                "deployed_pct": round(
                    self.budget_deployed_usd / self.max_budget_usd * 100
                    if self.max_budget_usd > 0 else 0, 1
                ),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_pnl_bps(token: SprayToken, current_price: float) -> float:
        """Calculate P&L in basis points for a token."""
        if token.entry_price <= 0:
            return 0.0
        move = (current_price - token.entry_price) / token.entry_price * 10000
        return move if token.side == "LONG" else -move

    def _close_token(
        self,
        token: SprayToken,
        exit_price: float,
        exit_epoch: float,
        reason: str,
        pnl_bps: float,
    ) -> None:
        """Close a token and update state."""
        pnl_usd = pnl_bps / 10000 * token.size_usd
        hold_seconds = exit_epoch - token.entry_time

        token.status = "closed"
        token.exit_price = exit_price
        token.exit_time = exit_epoch
        token.exit_reason = reason
        token.exit_pnl_bps = round(pnl_bps, 2)
        token.exit_pnl_usd = round(pnl_usd, 4)
        token.hold_time_seconds = round(hold_seconds, 1)

        # Update engine state
        self.budget_deployed_usd = max(0.0, self.budget_deployed_usd - token.size_usd)
        self._total_closed += 1
        if not token.observation_mode:
            self._total_pnl_usd += pnl_usd

        # Remove from open
        self.open_tokens.pop(token.token_id, None)

        # Persist exit
        self._update_db_exit(token)

        self.log.info(
            f"SPRAY EXIT {'(OBS) ' if token.observation_mode else ''}"
            f"{token.pair} {token.side} → {reason} | "
            f"pnl={pnl_bps:+.1f}bps ${pnl_usd:+.4f} | "
            f"hold={hold_seconds:.0f}s | "
            f"[{len(self.open_tokens)} open, ${self.budget_deployed_usd:.0f} deployed]"
        )

    def _classify_direction_rule(
        self,
        contributions: Dict[str, float],
        weighted_signal: float,
    ) -> str:
        """Classify trade driver for A/B testing."""
        if not contributions:
            return "mixed"

        # Find dominant contributor
        abs_contribs = {k: abs(v) for k, v in contributions.items() if v != 0}
        if not abs_contribs:
            return "mixed"

        top_key = max(abs_contribs, key=abs_contribs.get)  # type: ignore[arg-type]
        top_val = abs_contribs[top_key]
        total_abs = sum(abs_contribs.values())

        # If one signal contributes >50% of total, label it
        if total_abs > 0 and top_val / total_abs > 0.5:
            key_lower = top_key.lower()
            if "ml" in key_lower or "ensemble" in key_lower or "cnn" in key_lower or "lstm" in key_lower:
                return "ml_ensemble"
            if "momentum" in key_lower or "trend" in key_lower:
                return "momentum"
            if "reversion" in key_lower or "mean" in key_lower:
                return "mean_reversion"
            if "arb" in key_lower or "stat" in key_lower or "basis" in key_lower:
                return "stat_arb"

        return "mixed"

    def _calc_vol_regime(self, market_data: Dict[str, Any]) -> str:
        """Determine volatility regime from market data."""
        # Try GARCH forecast first
        garch = market_data.get("garch_forecast", {})
        if garch:
            vol = garch.get("forecast_vol", 0.0)
            if vol > 0.04:
                return "high"
            if vol > 0.015:
                return "medium"
            return "low"

        # Fallback: volatility_prediction
        vp = market_data.get("volatility_prediction", {})
        if vp:
            regime = vp.get("vol_regime", "normal")
            if regime in ("explosive", "high_volatility"):
                return "high"
            if regime in ("active",):
                return "medium"
            return "low"

        return "medium"

    def _estimate_expected_move(
        self,
        weighted_signal: float,
        market_data: Dict[str, Any],
    ) -> float:
        """Estimate expected move in bps from signal and vol data."""
        # Base: signal magnitude × 100 (so signal=0.05 → 5bps expected)
        base_bps = abs(weighted_signal) * 100.0

        # Scale by GARCH if available
        garch = market_data.get("garch_forecast", {})
        if garch:
            vol = garch.get("forecast_vol", 0.02)
            # Normalize: 2% daily vol → 1.0x, scale linearly
            vol_mult = vol / 0.02
            base_bps *= vol_mult

        # Floor at 3bps (don't set targets below noise)
        return max(base_bps, 3.0)

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _log_to_db(self, token: SprayToken) -> None:
        """Insert a new token entry row into token_spray_log."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """INSERT INTO token_spray_log
                   (timestamp, token_id, pair, direction, direction_rule,
                    token_size_usd, entry_price, vol_regime,
                    expected_move_bps, target_bps, stop_bps,
                    max_hold_seconds, observation_mode,
                    weighted_signal, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    token.token_id,
                    token.pair,
                    token.side,
                    token.direction_rule,
                    token.size_usd,
                    token.entry_price,
                    token.vol_regime,
                    token.expected_move_bps,
                    token.target_bps,
                    token.stop_bps,
                    token.max_hold_seconds,
                    1 if token.observation_mode else 0,
                    token.weighted_signal,
                    token.confidence,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.log.warning(f"Spray DB insert failed: {e}")

    def _update_db_exit(self, token: SprayToken) -> None:
        """Update the token_spray_log row with exit data."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """UPDATE token_spray_log
                   SET exit_price = ?, exit_time = ?, exit_reason = ?,
                       exit_pnl_bps = ?, exit_pnl_usd = ?, hold_time_seconds = ?
                   WHERE token_id = ?""",
                (
                    token.exit_price,
                    datetime.fromtimestamp(token.exit_time, tz=timezone.utc).isoformat()
                    if token.exit_time else None,
                    token.exit_reason,
                    token.exit_pnl_bps,
                    token.exit_pnl_usd,
                    token.hold_time_seconds,
                    token.token_id,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.log.warning(f"Spray DB exit update failed: {e}")
