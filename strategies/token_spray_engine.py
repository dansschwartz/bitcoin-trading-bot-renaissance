"""Token Spray Engine — rapid-fire small positions with fast exits.

Replaces the legacy ML decision path (large positions, long holds) with many
small ($10-100) tokens that enter on signal and exit fast via trailing stop.
Edge decays exponentially with hold time — this engine harvests the
<5-minute sweet spot.

Exit logic v2 (3 rules, priority order):
  1. Hard stop:  pnl <= -stop_loss_bps  → cut immediately
  2. Trailing stop: once pnl >= trail_activation_bps, track peak and close
     when pnl drops trail_distance_bps below peak
  3. Opportunity cost timeout: after max_hold_seconds, exit and sub-classify
     as timeout_flat / timeout_profitable / timeout_loss
"""

import asyncio
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-pair exit config (tighter for low-vol majors, wider for alts)
# ---------------------------------------------------------------------------

DEFAULT_PAIR_EXIT = {
    "stop_loss_bps": 8,
    "trail_activation_bps": 3,
    "trail_distance_bps": 5,
    "min_hold_for_trail_seconds": 30,  # trail can't activate before this age
}

# Loaded from config; these are the defaults
PAIR_EXIT_DEFAULTS: Dict[str, Dict[str, float]] = {
    "BTC": {"stop_loss_bps": 7, "trail_activation_bps": 3, "trail_distance_bps": 5, "min_hold_for_trail_seconds": 20},
    "ETH": {"stop_loss_bps": 8, "trail_activation_bps": 3, "trail_distance_bps": 5, "min_hold_for_trail_seconds": 20},
    "SOL": {"stop_loss_bps": 10, "trail_activation_bps": 3, "trail_distance_bps": 6, "min_hold_for_trail_seconds": 30},
    "AVAX": {"stop_loss_bps": 10, "trail_activation_bps": 3, "trail_distance_bps": 6, "min_hold_for_trail_seconds": 30},
    "LINK": {"stop_loss_bps": 10, "trail_activation_bps": 3, "trail_distance_bps": 6, "min_hold_for_trail_seconds": 30},
    "DOGE": {"stop_loss_bps": 12, "trail_activation_bps": 4, "trail_distance_bps": 7, "min_hold_for_trail_seconds": 30},
    "XRP": {"stop_loss_bps": 9, "trail_activation_bps": 3, "trail_distance_bps": 5, "min_hold_for_trail_seconds": 25},
}


def _get_pair_exit_config(pair: str, cfg_overrides: Dict[str, Any]) -> Dict[str, float]:
    """Resolve exit parameters for a specific pair.

    Priority: config/config.json pair_exit_config > PAIR_EXIT_DEFAULTS > DEFAULT_PAIR_EXIT
    """
    # Extract base asset from pair like "BTC-USD" or "BTCUSDT"
    base = pair.split("-")[0].upper() if "-" in pair else pair.replace("USDT", "").replace("USD", "").upper()

    # Check config overrides first
    pair_cfg = cfg_overrides.get(base, {})
    if pair_cfg:
        return {
            "stop_loss_bps": pair_cfg.get("stop_loss_bps", DEFAULT_PAIR_EXIT["stop_loss_bps"]),
            "trail_activation_bps": pair_cfg.get("trail_activation_bps", DEFAULT_PAIR_EXIT["trail_activation_bps"]),
            "trail_distance_bps": pair_cfg.get("trail_distance_bps", DEFAULT_PAIR_EXIT["trail_distance_bps"]),
            "min_hold_for_trail_seconds": pair_cfg.get("min_hold_for_trail_seconds", DEFAULT_PAIR_EXIT["min_hold_for_trail_seconds"]),
        }

    # Fall back to built-in defaults for known assets
    if base in PAIR_EXIT_DEFAULTS:
        return dict(PAIR_EXIT_DEFAULTS[base])

    return dict(DEFAULT_PAIR_EXIT)


# ---------------------------------------------------------------------------
# SprayToken — one micro-position
# ---------------------------------------------------------------------------

@dataclass
class Wallet:
    """Per-model budget + tracking for independent A/B testing."""
    wallet_id: str                  # model name: 'lightgbm', 'cnn', 'weighted_signal', etc.
    enabled: bool = True
    budget_usd: float = 0.0
    deployed_usd: float = 0.0
    total_pnl_usd: float = 0.0
    total_sprayed: int = 0
    total_closed: int = 0
    total_winners: int = 0
    max_tokens_per_pair: int = 1
    pair_last_spray: Dict[str, float] = field(default_factory=dict)


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
    direction_rule: str             # top contributor key or "mixed"
    vol_regime: str                 # "low" / "medium" / "high"
    stop_loss_bps: float            # per-pair hard stop
    trail_activation_bps: float     # bps profit before trail activates
    trail_distance_bps: float       # bps below peak to trigger exit
    min_hold_for_trail: float       # seconds before trail can activate
    max_hold_seconds: float
    observation_mode: bool
    weighted_signal: float = 0.0
    confidence: float = 0.0
    wallet_id: str = "default"
    # Trailing stop tracking
    peak_pnl_bps: float = 0.0
    peak_price: float = 0.0
    trail_active: bool = False
    # Legacy compat (kept for DB; always 0 now)
    expected_move_bps: float = 0.0
    target_bps: float = 0.0
    stop_bps: float = 0.0          # deprecated, use stop_loss_bps
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
           ``check_exits()`` which evaluates hard stop / trailing stop / timeout.
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
        self.cooldown_seconds: float = config.get("cooldown_seconds", 30.0)
        self.exit_check_interval: float = config.get("exit_check_interval_seconds", 5.0)
        self.vol_scaling: Dict[str, float] = config.get("vol_scaling", {
            "low": 1.0, "medium": 0.7, "high": 0.4, "extreme": 0.2,
        })
        self.observation_mode: bool = config.get("observation_mode", True)

        # Exit config (new v2 trailing stop)
        exit_cfg = config.get("exit_config", {})
        self.default_stop_loss_bps: float = exit_cfg.get("stop_loss_bps", 12.0)
        self.default_trail_activation_bps: float = exit_cfg.get("trail_activation_bps", 3.0)
        self.default_trail_distance_bps: float = exit_cfg.get("trail_distance_bps", 5.0)
        self.max_hold_seconds: float = exit_cfg.get("max_hold_seconds", 600.0)
        self.min_move_bps: float = exit_cfg.get("min_move_bps", 3.0)

        # Per-pair exit config overrides from config.json
        self._pair_exit_config: Dict[str, Any] = config.get("pair_exit_config", {})

        # Blacklist + spread gate (blocks wide-spread micro-cap pairs)
        self.blacklisted_pairs: Set[str] = set(config.get("blacklisted_pairs", []))
        self.max_entry_spread_bps: float = config.get("max_entry_spread_bps", 4.0)

        # Runtime state
        self.open_tokens: Dict[str, SprayToken] = {}
        self.budget_deployed_usd: float = 0.0
        self._pair_last_spray: Dict[str, float] = {}   # pair → epoch of last spray
        self._exit_task: Optional[asyncio.Task] = None
        self._running = False

        # Price cache populated by exit loop (used for fresh entry prices)
        self.last_prices: Dict[str, float] = {}

        # Lifetime counters
        self._total_sprayed: int = 0
        self._total_closed: int = 0
        self._total_pnl_usd: float = 0.0

        self.log.info(
            f"TokenSprayEngine v2: size=${self.token_size_usd} | "
            f"budget=${self.max_budget_usd} | max_tokens={self.max_open_tokens} | "
            f"stop={self.default_stop_loss_bps}bps trail_act={self.default_trail_activation_bps}bps "
            f"trail_dist={self.default_trail_distance_bps}bps "
            f"min_hold_trail={DEFAULT_PAIR_EXIT['min_hold_for_trail_seconds']}s "
            f"timeout={self.max_hold_seconds}s | "
            f"obs_mode={self.observation_mode}"
        )

        # Close orphaned tokens from previous runs
        self._close_orphaned_tokens()

        # Per-model wallets (optional; backward compat when absent)
        self.wallets: Dict[str, Wallet] = {}
        wallets_cfg = config.get("wallets", {})
        if wallets_cfg.get("enabled", False):
            self._init_wallets(wallets_cfg)

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
        fresh_price: Optional[float] = None,
    ) -> Optional[SprayToken]:
        """Evaluate whether to open a micro-position for *pair*.

        Returns the SprayToken if one was opened, else None.

        *fresh_price*: If provided, use this instead of market_data ticker
        price (which may be stale by 15-45s from ML processing).
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

        # ── Gate 2b: blacklist ──
        if pair in self.blacklisted_pairs:
            return None

        # ── Gate 2c: spread check at entry time (proportional to stop_loss) ──
        _pair_exit = _get_pair_exit_config(pair, self._pair_exit_config)
        _pair_stop = _pair_exit["stop_loss_bps"]
        _proportional_limit = _pair_stop * 0.5
        _ticker = market_data.get("ticker", {})
        _bid = float(_ticker.get("bid", 0) or 0)
        _ask = float(_ticker.get("ask", 0) or 0)
        if _bid > 0 and _ask > 0:
            _spread_bps = ((_ask - _bid) / ((_ask + _bid) / 2)) * 10000
            if _spread_bps > _proportional_limit:
                self.log.debug(
                    f"SPRAY SPREAD GATE: {pair} spread={_spread_bps:.1f}bps "
                    f"> {_proportional_limit:.1f}bps (stop={_pair_stop}bp * 0.5) — skip"
                )
                return None

        # ── Gate 3: capacity ──
        if len(self.open_tokens) >= self.max_open_tokens:
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

        # ── Gate 3b: budget check with actual vol-scaled size (Fix 3) ──
        if self.budget_deployed_usd + size_usd > self.max_budget_usd:
            return None

        # ── Entry price: prefer fresh_price > exit-loop cache > ticker ──
        current_price = 0.0
        if fresh_price and fresh_price > 0:
            current_price = fresh_price
        elif self.last_prices.get(pair, 0) > 0:
            current_price = self.last_prices[pair]
        else:
            ticker = market_data.get("ticker", {})
            current_price = float(ticker.get("price", 0.0))
        if current_price <= 0:
            return None

        # ── Compute size in units ──
        size_units = size_usd / current_price

        # ── Per-pair exit parameters ──
        pair_exit = _get_pair_exit_config(pair, self._pair_exit_config)
        stop_loss_bps = pair_exit["stop_loss_bps"]
        trail_activation_bps = pair_exit["trail_activation_bps"]
        trail_distance_bps = pair_exit["trail_distance_bps"]
        min_hold_for_trail = pair_exit.get("min_hold_for_trail_seconds", 30)

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
            stop_loss_bps=stop_loss_bps,
            trail_activation_bps=trail_activation_bps,
            trail_distance_bps=trail_distance_bps,
            min_hold_for_trail=min_hold_for_trail,
            max_hold_seconds=self.max_hold_seconds,
            observation_mode=self.observation_mode,
            weighted_signal=weighted_signal,
            confidence=confidence,
            peak_pnl_bps=0.0,
            peak_price=current_price,
            trail_active=False,
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
            f"stop={stop_loss_bps:.0f}bps trail_act={trail_activation_bps:.0f}bps "
            f"trail_dist={trail_distance_bps:.0f}bps | "
            f"signal={weighted_signal:.4f} conf={confidence:.3f} "
            f"[{len(self.open_tokens)} open, ${self.budget_deployed_usd:.0f} deployed]"
        )
        return token

    # ------------------------------------------------------------------
    # Per-model wallets
    # ------------------------------------------------------------------

    def _init_wallets(self, wallets_cfg: Dict[str, Any]) -> None:
        """Create one Wallet per enabled model, splitting budget equally."""
        models_cfg = wallets_cfg.get("models", {})
        enabled_ids = [mid for mid, mcfg in models_cfg.items()
                       if mcfg.get("enabled", True)]
        if not enabled_ids:
            self.log.warning("Wallets enabled but no models enabled — skipping")
            return

        per_wallet_budget = self.max_budget_usd / len(enabled_ids)
        for mid in enabled_ids:
            self.wallets[mid] = Wallet(
                wallet_id=mid,
                enabled=True,
                budget_usd=round(per_wallet_budget, 2),
                max_tokens_per_pair=1,
            )
        self.log.info(
            f"Wallets initialized: {len(self.wallets)} models × "
            f"${per_wallet_budget:.0f}/wallet = ${self.max_budget_usd:.0f} total | "
            f"models={list(self.wallets.keys())}"
        )

    # Mapping from CrashRegime keys to wallet IDs
    _CRASH_KEY_MAP = {
        "CrashRegime_2bar": "crash_2bar",
        "CrashRegime_1bar": "crash_1bar",
        "CrashRegime": "crash_2bar",
    }

    async def spray_wallets(
        self,
        pair: str,
        ml_predictions: list,
        crash_predictions: Dict[str, float],
        weighted_signal: float,
        contributions: Dict[str, float],
        market_data: Dict[str, Any],
        fresh_price: Optional[float] = None,
    ) -> List[SprayToken]:
        """Spray independent tokens per enabled wallet/model.

        Each model's raw prediction is used as its signal (no ensemble fusion).
        The 'weighted_signal' wallet uses the fused blend as a control.
        Returns list of tokens opened (may be empty).
        """
        if not self.wallets:
            return []

        # Build lookup: {wallet_id: (prediction, confidence)}
        model_signals: Dict[str, tuple] = {}

        # ML base models + meta_ensemble from ml_predictions list
        if ml_predictions:
            for mp in ml_predictions:
                if isinstance(mp, dict):
                    name = mp.get("model", "")
                    pred = mp.get("prediction", 0.0)
                    conf = mp.get("confidence")
                elif isinstance(mp, tuple) and len(mp) >= 2:
                    name = mp[0]
                    pred = mp[1]
                    conf = mp[2] if len(mp) > 2 else None
                else:
                    continue
                if name and name in self.wallets:
                    model_signals[name] = (float(pred), float(conf) if conf else 0.5)

        # Crash model predictions
        for crash_key, crash_pred in crash_predictions.items():
            wallet_id = self._CRASH_KEY_MAP.get(crash_key)
            if wallet_id and wallet_id in self.wallets:
                model_signals[wallet_id] = (float(crash_pred), 0.5)

        # Control wallet: use the fused weighted_signal
        if "weighted_signal" in self.wallets:
            conf_val = 0.5
            if contributions:
                # Use the confidence from contributions as proxy
                conf_val = min(1.0, abs(weighted_signal) * 10)
            model_signals["weighted_signal"] = (weighted_signal, conf_val)

        # Spray for each wallet that has a signal
        tokens: List[SprayToken] = []
        for wallet_id, (prediction, confidence) in model_signals.items():
            wallet = self.wallets.get(wallet_id)
            if not wallet or not wallet.enabled:
                continue
            token = await self._spray_for_wallet(
                wallet=wallet,
                pair=pair,
                prediction=prediction,
                confidence=confidence,
                market_data=market_data,
                fresh_price=fresh_price,
            )
            if token:
                tokens.append(token)

        return tokens

    async def _spray_for_wallet(
        self,
        wallet: Wallet,
        pair: str,
        prediction: float,
        confidence: float,
        market_data: Dict[str, Any],
        fresh_price: Optional[float] = None,
    ) -> Optional[SprayToken]:
        """Evaluate and open a token for one wallet. Same gates as spray() but wallet-scoped."""

        # ── Gate 1: signal strength ──
        if abs(prediction) < self.min_signal_strength:
            return None

        # ── Gate 2: confidence ──
        if confidence < self.min_confidence:
            return None

        # ── Gate 2b: blacklist ──
        if pair in self.blacklisted_pairs:
            return None

        # ── Gate 2c: spread check at entry time (proportional to stop_loss) ──
        _pair_exit = _get_pair_exit_config(pair, self._pair_exit_config)
        _pair_stop = _pair_exit["stop_loss_bps"]
        _proportional_limit = _pair_stop * 0.5
        _ticker = market_data.get("ticker", {})
        _bid = float(_ticker.get("bid", 0) or 0)
        _ask = float(_ticker.get("ask", 0) or 0)
        if _bid > 0 and _ask > 0:
            _spread_bps = ((_ask - _bid) / ((_ask + _bid) / 2)) * 10000
            if _spread_bps > _proportional_limit:
                self.log.debug(
                    f"SPRAY SPREAD GATE: {pair} spread={_spread_bps:.1f}bps "
                    f"> {_proportional_limit:.1f}bps (stop={_pair_stop}bp * 0.5) — skip"
                )
                return None

        # ── Gate 3: global capacity ──
        if len(self.open_tokens) >= self.max_open_tokens:
            return None

        # ── Gate 4: per-wallet per-pair limit ──
        pair_count = sum(
            1 for t in self.open_tokens.values()
            if t.pair == pair and t.wallet_id == wallet.wallet_id
        )
        if pair_count >= wallet.max_tokens_per_pair:
            return None

        # ── Gate 5: per-wallet cooldown ──
        now = time.time()
        last = wallet.pair_last_spray.get(pair, 0.0)
        if now - last < self.cooldown_seconds:
            return None

        # ── Direction ──
        side = "LONG" if prediction > 0 else "SHORT"

        # ── Volatility regime & size scaling ──
        vol_regime = self._calc_vol_regime(market_data)
        vol_scale = self.vol_scaling.get(vol_regime, 1.0)
        size_usd = self.token_size_usd * vol_scale

        # ── Gate 6: wallet budget check ──
        if wallet.deployed_usd + size_usd > wallet.budget_usd:
            return None

        # ── Entry price: prefer fresh_price > exit-loop cache > ticker ──
        current_price = 0.0
        if fresh_price and fresh_price > 0:
            current_price = fresh_price
        elif self.last_prices.get(pair, 0) > 0:
            current_price = self.last_prices[pair]
        else:
            ticker = market_data.get("ticker", {})
            current_price = float(ticker.get("price", 0.0))
        if current_price <= 0:
            return None

        size_units = size_usd / current_price

        # ── Per-pair exit parameters ──
        pair_exit = _get_pair_exit_config(pair, self._pair_exit_config)

        # ── Build token ──
        token = SprayToken(
            token_id=uuid.uuid4().hex[:12],
            pair=pair,
            side=side,
            entry_price=current_price,
            entry_time=now,
            size_usd=size_usd,
            size_units=size_units,
            direction_rule=wallet.wallet_id,
            vol_regime=vol_regime,
            stop_loss_bps=pair_exit["stop_loss_bps"],
            trail_activation_bps=pair_exit["trail_activation_bps"],
            trail_distance_bps=pair_exit["trail_distance_bps"],
            min_hold_for_trail=pair_exit.get("min_hold_for_trail_seconds", 30),
            max_hold_seconds=self.max_hold_seconds,
            observation_mode=self.observation_mode,
            weighted_signal=prediction,
            confidence=confidence,
            wallet_id=wallet.wallet_id,
            peak_pnl_bps=0.0,
            peak_price=current_price,
            trail_active=False,
        )

        # Register in global + wallet state
        self.open_tokens[token.token_id] = token
        self.budget_deployed_usd += size_usd
        wallet.deployed_usd += size_usd
        wallet.pair_last_spray[pair] = now
        wallet.total_sprayed += 1
        self._total_sprayed += 1

        # Persist entry
        self._log_to_db(token)

        self.log.info(
            f"SPRAY [{wallet.wallet_id}] {'(OBS) ' if self.observation_mode else ''}"
            f"{side} {pair} ${size_usd:.0f} @ {current_price:.4f} | "
            f"pred={prediction:.4f} conf={confidence:.3f} "
            f"[{len(self.open_tokens)} open, wallet ${wallet.deployed_usd:.0f}/"
            f"${wallet.budget_usd:.0f}]"
        )
        return token

    # ------------------------------------------------------------------
    # check_exits() — called every N seconds (v2 trailing stop logic)
    # ------------------------------------------------------------------

    async def check_exits(self, price_fetcher: Callable) -> List[SprayToken]:
        """Evaluate all open tokens for exit conditions.

        *price_fetcher* is ``async def(pairs: List[str]) -> Dict[str, float]``
        returning {pair: current_price}.

        Exit rules (priority order):
          1. Hard stop:  pnl_bps <= -stop_loss_bps
          2. Trailing stop:  trail_active AND pnl_bps <= peak_pnl_bps - trail_distance_bps
          3. Timeout:  age >= max_hold_seconds → sub-classify
        """
        if not self.open_tokens:
            return []

        pairs = list({t.pair for t in self.open_tokens.values()})
        try:
            prices = await price_fetcher(pairs)
        except Exception as e:
            self.log.warning(f"Spray price fetch failed: {e}")
            return []

        # Update price cache for fresh entry prices (Fix 4)
        for pair, price in prices.items():
            if price > 0:
                self.last_prices[pair] = price

        closed: List[SprayToken] = []
        now = time.time()

        for token in list(self.open_tokens.values()):
            price = prices.get(token.pair)
            if price is None or price <= 0:
                continue

            age = now - token.entry_time
            pnl_bps = self._calc_pnl_bps(token, price)

            # ── Track peak P&L for trailing stop ──
            if pnl_bps > token.peak_pnl_bps:
                token.peak_pnl_bps = pnl_bps
                token.peak_price = price

            # ── Activate trail when threshold crossed AND min hold met ──
            if not token.trail_active and pnl_bps >= token.trail_activation_bps:
                if age >= token.min_hold_for_trail:
                    token.trail_active = True

            exit_reason: Optional[str] = None

            # Rule 1: Hard stop
            if pnl_bps <= -token.stop_loss_bps:
                exit_reason = "stop"

            # Rule 2: Trailing stop (only if trail is active)
            elif token.trail_active and pnl_bps <= (token.peak_pnl_bps - token.trail_distance_bps):
                exit_reason = "trail_stop"

            # Rule 3: Opportunity cost timeout
            elif age >= token.max_hold_seconds:
                if abs(pnl_bps) < self.min_move_bps:
                    exit_reason = "timeout_flat"
                elif pnl_bps > 0:
                    exit_reason = "timeout_profitable"
                else:
                    exit_reason = "timeout_loss"

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
            except asyncio.CancelledError as e:
                self.log.warning(f"Spray exit task cancelled: {e}")
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
        status: Dict[str, Any] = {
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
            "exit_config": {
                "stop_loss_bps": self.default_stop_loss_bps,
                "trail_activation_bps": self.default_trail_activation_bps,
                "trail_distance_bps": self.default_trail_distance_bps,
                "max_hold_seconds": self.max_hold_seconds,
            },
        }
        if self.wallets:
            wallet_status = {}
            for wid, w in self.wallets.items():
                wr = (w.total_winners / w.total_closed * 100) if w.total_closed > 0 else 0.0
                wallet_status[wid] = {
                    "budget_usd": w.budget_usd,
                    "deployed_usd": round(w.deployed_usd, 2),
                    "total_sprayed": w.total_sprayed,
                    "total_closed": w.total_closed,
                    "total_pnl_usd": round(w.total_pnl_usd, 4),
                    "win_rate": round(wr, 1),
                }
            status["wallets"] = wallet_status
        return status

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

        # Update wallet stats if applicable
        wallet = self.wallets.get(token.wallet_id)
        if wallet:
            wallet.deployed_usd = max(0.0, wallet.deployed_usd - token.size_usd)
            wallet.total_closed += 1
            if not token.observation_mode:
                wallet.total_pnl_usd += pnl_usd
                if pnl_usd > 0:
                    wallet.total_winners += 1

        # Remove from open
        self.open_tokens.pop(token.token_id, None)

        # Persist exit
        self._update_db_exit(token)

        self.log.info(
            f"SPRAY EXIT {'(OBS) ' if token.observation_mode else ''}"
            f"{token.pair} {token.side} → {reason} | "
            f"pnl={pnl_bps:+.1f}bps ${pnl_usd:+.4f} peak={token.peak_pnl_bps:+.1f}bps "
            f"trail={'ON' if token.trail_active else 'off'} | "
            f"hold={hold_seconds:.0f}s | "
            f"[{len(self.open_tokens)} open, ${self.budget_deployed_usd:.0f} deployed]"
        )

    def _classify_direction_rule(
        self,
        contributions: Dict[str, float],
        weighted_signal: float,
    ) -> str:
        """Classify trade driver for A/B testing.

        Fix 6: lowered dominance threshold from >50% to >30%, and returns
        the actual top contributor key name directly (e.g. 'rsi', 'entropy',
        'ml_ensemble') instead of bucketing into generic categories.
        """
        if not contributions:
            return "mixed"

        # Find dominant contributor
        abs_contribs = {k: abs(v) for k, v in contributions.items() if v != 0}
        if not abs_contribs:
            return "mixed"

        top_key = max(abs_contribs, key=abs_contribs.get)  # type: ignore[arg-type]
        top_val = abs_contribs[top_key]
        total_abs = sum(abs_contribs.values())

        # If one signal contributes >30% of total, use its key name directly
        if total_abs > 0 and top_val / total_abs > 0.30:
            return top_key

        return "mixed"

    def _calc_vol_regime(self, market_data: Dict[str, Any]) -> str:
        """Determine volatility regime from market data."""
        # Try GARCH forecast first
        garch = market_data.get("garch_forecast", {})
        if garch:
            vol = garch.get("forecast_vol", 0.0)
            if vol > 0.08:
                return "extreme"
            if vol > 0.04:
                return "high"
            if vol > 0.015:
                return "medium"
            return "low"

        # Fallback: volatility_prediction
        vp = market_data.get("volatility_prediction", {})
        if vp:
            regime = vp.get("vol_regime", "normal")
            if regime in ("explosive",):
                return "extreme"
            if regime in ("high_volatility",):
                return "high"
            if regime in ("active",):
                return "medium"
            return "low"

        return "medium"

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _close_orphaned_tokens(self) -> None:
        """Close any DB tokens left open from a previous process.

        On restart, in-memory state is empty but the DB may have tokens
        whose exit_reason is NULL.  Mark them as 'restart_orphan' with
        0 P&L (we can't know the exit price retroactively).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            orphans = conn.execute(
                "SELECT token_id, pair, entry_price FROM token_spray_log "
                "WHERE exit_reason IS NULL"
            ).fetchall()
            if not orphans:
                conn.close()
                return
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.executemany(
                """UPDATE token_spray_log
                   SET exit_price = entry_price, exit_time = ?,
                       exit_reason = 'restart_orphan',
                       exit_pnl_bps = 0, exit_pnl_usd = 0,
                       hold_time_seconds = 0
                   WHERE token_id = ?""",
                [(now_iso, row[0]) for row in orphans],
            )
            conn.commit()
            conn.close()
            self.log.info(
                f"TokenSprayEngine: closed {len(orphans)} orphaned tokens from previous run"
            )
        except Exception as e:
            self.log.warning(f"Spray orphan cleanup failed: {e}")

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
                    weighted_signal, confidence, wallet_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    token.token_id,
                    token.pair,
                    token.side,
                    token.direction_rule,
                    token.size_usd,
                    token.entry_price,
                    token.vol_regime,
                    0.0,                # expected_move_bps (removed)
                    0.0,                # target_bps (removed, now trailing)
                    token.stop_loss_bps,  # stop_bps column = hard stop
                    token.max_hold_seconds,
                    1 if token.observation_mode else 0,
                    token.weighted_signal,
                    token.confidence,
                    token.wallet_id,
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
                       exit_pnl_bps = ?, exit_pnl_usd = ?, hold_time_seconds = ?,
                       peak_pnl_bps = ?
                   WHERE token_id = ?""",
                (
                    token.exit_price,
                    datetime.fromtimestamp(token.exit_time, tz=timezone.utc).isoformat()
                    if token.exit_time else None,
                    token.exit_reason,
                    token.exit_pnl_bps,
                    token.exit_pnl_usd,
                    token.hold_time_seconds,
                    round(token.peak_pnl_bps, 2),
                    token.token_id,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.log.warning(f"Spray DB exit update failed: {e}")
