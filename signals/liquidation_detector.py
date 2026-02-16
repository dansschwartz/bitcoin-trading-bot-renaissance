"""
Renaissance Technologies Liquidation Cascade Detector
=====================================================

Monitors Binance perpetual futures markets for conditions that precede
liquidation cascades (forced long liquidations or short squeezes).

Tracked indicators:
  - Funding rates and their historical percentile
  - Open interest absolute level and 24-hour delta
  - Global long/short account ratio
  - Top-trader long/short position ratio

When the composite risk score exceeds a configurable threshold the detector
emits a ``CascadeRiskSignal`` that downstream strategy modules can consume
to adjust position sizing, set entries, or hedge.

All Binance endpoints used are *public* and require no API key.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Deque, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BINANCE_FUTURES_BASE = "https://fapi.binance.com"

_ENDPOINT_FUNDING_RATE = "/fapi/v1/fundingRate"
_ENDPOINT_OPEN_INTEREST = "/fapi/v1/openInterest"
_ENDPOINT_GLOBAL_LS_RATIO = "/futures/data/globalLongShortAccountRatio"
_ENDPOINT_TOP_TRADER_POS_RATIO = "/futures/data/topLongShortPositionRatio"

# Maximum number of data-points kept per symbol for percentile calculations.
# At 3 readings per 8-hour funding period, 30 days ~ 90 entries.  We keep a
# generous buffer so that irregular scan intervals still yield valid stats.
_HISTORY_MAXLEN = 2160  # 30 days * 24 hours * 3 (one per 20-min scan slot)

_DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "scan_interval_seconds": 60,
    "risk_threshold": 0.5,
    "extreme_funding_percentile": 0.90,
    "high_oi_change_pct": 10.0,
    "extreme_ls_ratio": 2.0,
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CascadeRiskSignal:
    """Emitted when liquidation-cascade risk exceeds the configured threshold."""

    signal_id: str
    timestamp: datetime
    symbol: str
    direction: str  # "long_liquidation" or "short_squeeze"
    risk_score: float  # 0-1
    funding_rate: float
    funding_rate_percentile: float
    open_interest_change_24h: float
    long_short_ratio: float
    estimated_liquidation_usd: float
    recommended_action: str
    entry_trigger: str
    expected_move_pct: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the signal to a plain dictionary."""
        return asdict(self)


@dataclass
class _SymbolSnapshot:
    """Internal per-symbol snapshot gathered during a single scan cycle."""

    symbol: str
    timestamp: datetime
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    long_short_ratio: Optional[float] = None
    top_trader_ls_ratio: Optional[float] = None


@dataclass
class _SymbolHistory:
    """In-memory rolling history for a single symbol."""

    funding_rates: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_HISTORY_MAXLEN)
    )
    open_interests: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_HISTORY_MAXLEN)
    )
    timestamps: Deque[datetime] = field(
        default_factory=lambda: deque(maxlen=_HISTORY_MAXLEN)
    )


@dataclass
class _RealtimeState:
    """Per-symbol real-time price/volume/spread state for fast evaluation."""

    price_window: Deque[tuple] = field(
        default_factory=lambda: deque(maxlen=120)
    )  # (timestamp, price)
    volume_window: Deque[tuple] = field(
        default_factory=lambda: deque(maxlen=120)
    )  # (timestamp, volume)
    spread_window: Deque[tuple] = field(
        default_factory=lambda: deque(maxlen=120)
    )  # (timestamp, spread_bps)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class LiquidationCascadeDetector:
    """Continuously scans Binance Futures for liquidation-cascade risk.

    Parameters
    ----------
    config : dict, optional
        Override any key from ``_DEFAULT_CONFIG``.  Unknown keys are silently
        ignored so that the detector can be embedded in a larger config tree.

    Usage
    -----
    ::

        detector = LiquidationCascadeDetector({"symbols": ["BTCUSDT"]})
        await detector.start()          # begins background scanning
        signals = await detector.get_signals()
        risk    = await detector.get_current_risk()
        await detector.stop()
    """

    # ----- construction / config ------------------------------------------

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = {**_DEFAULT_CONFIG, **(config or {})}
        self._enabled: bool = bool(merged["enabled"])
        self._symbols: List[str] = list(merged["symbols"])
        self._scan_interval: int = int(merged["scan_interval_seconds"])
        self._risk_threshold: float = float(merged["risk_threshold"])
        self._extreme_funding_pctl: float = float(merged["extreme_funding_percentile"])
        self._high_oi_change_pct: float = float(merged["high_oi_change_pct"])
        self._extreme_ls_ratio: float = float(merged["extreme_ls_ratio"])

        # Per-symbol rolling history
        self._history: Dict[str, _SymbolHistory] = {
            sym: _SymbolHistory() for sym in self._symbols
        }

        # Latest risk assessment per symbol (populated after each scan)
        self._current_risk: Dict[str, Dict[str, Any]] = {}

        # Queue of emitted signals
        self._signal_queue: asyncio.Queue[CascadeRiskSignal] = asyncio.Queue()

        # Scan-loop control
        self._scan_task: Optional[asyncio.Task[None]] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

        # Rate-limit back-off state
        self._backoff_until: float = 0.0

        # Fast evaluation loop (1s real-time cascade detection)
        self._fast_eval_enabled: bool = bool(merged.get("fast_eval_enabled", False))
        self._fast_eval_interval: float = float(merged.get("fast_eval_interval_seconds", 1))
        self._fast_eval_task: Optional[asyncio.Task[None]] = None
        self._realtime_state: Dict[str, _RealtimeState] = {
            sym: _RealtimeState() for sym in self._symbols
        }

        logger.info(
            "LiquidationCascadeDetector initialised  symbols=%s  interval=%ds  "
            "threshold=%.2f  enabled=%s  fast_eval=%s",
            self._symbols,
            self._scan_interval,
            self._risk_threshold,
            self._enabled,
            self._fast_eval_enabled,
        )

    # ----- public properties ----------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the detector is logically enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        logger.info("LiquidationCascadeDetector enabled=%s", value)

    # ----- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Start the background scan loop.

        If the detector is disabled via config the loop will still run but
        will skip API calls, allowing it to be hot-enabled later.
        """
        if self._running:
            logger.warning("LiquidationCascadeDetector already running")
            return

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        )
        self._running = True
        self._scan_task = asyncio.create_task(
            self._scan_loop(), name="liquidation-cascade-scan"
        )
        if self._fast_eval_enabled:
            self._fast_eval_task = asyncio.create_task(
                self._fast_eval_loop(), name="liquidation-cascade-fast-eval"
            )
            logger.info(
                "LiquidationCascadeDetector fast eval loop started (%.1fs)",
                self._fast_eval_interval,
            )
        logger.info("LiquidationCascadeDetector scan loop started")

    async def stop(self) -> None:
        """Gracefully shut down the scan loop and close the HTTP session."""
        self._running = False
        for task_attr in ("_scan_task", "_fast_eval_task"):
            task = getattr(self, task_attr, None)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self, task_attr, None)
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        logger.info("LiquidationCascadeDetector stopped")

    # ----- public query interface -----------------------------------------

    async def get_current_risk(self) -> Dict[str, Dict[str, Any]]:
        """Return the most recent risk assessment for every tracked symbol.

        Returns
        -------
        dict
            Mapping of *symbol* to a dict with keys ``risk_score``,
            ``direction``, ``funding_rate``, ``funding_rate_percentile``,
            ``open_interest_change_24h``, ``long_short_ratio``, and
            ``timestamp``.
        """
        return dict(self._current_risk)

    async def get_signals(self) -> List[CascadeRiskSignal]:
        """Drain and return all queued ``CascadeRiskSignal`` objects.

        Non-blocking -- returns an empty list when the queue is empty.
        """
        signals: List[CascadeRiskSignal] = []
        while not self._signal_queue.empty():
            try:
                signals.append(self._signal_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return signals

    # ----- real-time price feed (fast eval) --------------------------------

    def on_price_update(
        self,
        symbol: str,
        price: float,
        volume: float,
        spread_bps: float,
        timestamp: float,
    ) -> None:
        """Called by the main bot to feed real-time data for fast evaluation."""
        if not self._fast_eval_enabled:
            return
        state = self._realtime_state.get(symbol)
        if state is None:
            # Only track configured symbols
            if symbol not in self._symbols:
                return
            state = _RealtimeState()
            self._realtime_state[symbol] = state

        state.price_window.append((timestamp, price))
        state.volume_window.append((timestamp, volume))
        state.spread_window.append((timestamp, spread_bps))

    # ----- fast evaluation loop -------------------------------------------

    async def _fast_eval_loop(self) -> None:
        """Run every 1s: combine cached REST risk with real-time metrics."""
        logger.info("Fast eval loop started (%.1fs interval)", self._fast_eval_interval)
        while self._running:
            try:
                if not self._enabled:
                    await asyncio.sleep(self._fast_eval_interval)
                    continue

                now = time.time()
                for symbol in self._symbols:
                    state = self._realtime_state.get(symbol)
                    if state is None or len(state.price_window) < 5:
                        continue

                    # --- Compute real-time metrics ---
                    # Price drop: % decline in last 10s vs rolling 60s mean
                    recent_prices = [
                        p for ts, p in state.price_window if ts > now - 10
                    ]
                    rolling_prices = [
                        p for ts, p in state.price_window if ts > now - 60
                    ]
                    if not recent_prices or not rolling_prices:
                        continue

                    recent_mean = sum(recent_prices) / len(recent_prices)
                    rolling_mean = sum(rolling_prices) / len(rolling_prices)

                    price_drop_pct = 0.0
                    if rolling_mean > 0:
                        price_drop_pct = (rolling_mean - recent_mean) / rolling_mean * 100.0

                    # Volume surge: recent 10s volume vs rolling 60s average
                    recent_vols = [v for ts, v in state.volume_window if ts > now - 10]
                    rolling_vols = [v for ts, v in state.volume_window if ts > now - 60]
                    volume_surge = 1.0
                    if rolling_vols:
                        avg_rolling_vol = sum(rolling_vols) / len(rolling_vols)
                        avg_recent_vol = sum(recent_vols) / len(recent_vols) if recent_vols else 0
                        if avg_rolling_vol > 0:
                            volume_surge = avg_recent_vol / avg_rolling_vol

                    # Spread widening
                    recent_spreads = [s for ts, s in state.spread_window if ts > now - 10]
                    rolling_spreads = [s for ts, s in state.spread_window if ts > now - 60]
                    spread_ratio = 1.0
                    if rolling_spreads:
                        avg_rolling_spread = sum(rolling_spreads) / len(rolling_spreads)
                        avg_recent_spread = sum(recent_spreads) / len(recent_spreads) if recent_spreads else 0
                        if avg_rolling_spread > 0:
                            spread_ratio = avg_recent_spread / avg_rolling_spread

                    # --- Combine with base REST risk ---
                    base_risk = self._current_risk.get(symbol, {})
                    base_score = float(base_risk.get("risk_score", 0.0))

                    realtime_boost = 0.0
                    # Cascade detection: sharp price drop + volume surge
                    if price_drop_pct > 0.5 and volume_surge > 3.0:
                        realtime_boost += 0.3
                    elif price_drop_pct > 0.3 and volume_surge > 2.0:
                        realtime_boost += 0.15
                    # Spread widening adds minor boost
                    if spread_ratio > 2.0:
                        realtime_boost += 0.1

                    enhanced_score = min(base_score + realtime_boost, 1.0)

                    # Update risk dict if enhanced score differs meaningfully
                    if realtime_boost > 0.05:
                        enhanced_risk = dict(base_risk) if base_risk else {
                            "direction": "long_liquidation",
                            "funding_rate": 0.0,
                            "funding_rate_percentile": 0.5,
                            "open_interest_change_24h": 0.0,
                            "long_short_ratio": 1.0,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        enhanced_risk["risk_score"] = round(enhanced_score, 4)
                        enhanced_risk["realtime_boost"] = round(realtime_boost, 4)
                        enhanced_risk["price_drop_pct"] = round(price_drop_pct, 4)
                        enhanced_risk["volume_surge"] = round(volume_surge, 2)
                        self._current_risk[symbol] = enhanced_risk

                        if enhanced_score > self._risk_threshold:
                            logger.warning(
                                "FAST CASCADE DETECT  symbol=%s  base=%.3f  "
                                "boost=+%.3f  total=%.3f  price_drop=%.2f%%  "
                                "vol_surge=%.1fx",
                                symbol, base_score, realtime_boost,
                                enhanced_score, price_drop_pct, volume_surge,
                            )

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Fast eval cycle failed")

            await asyncio.sleep(self._fast_eval_interval)

    # ----- internal scan loop ---------------------------------------------

    async def _scan_loop(self) -> None:
        """Main loop: scan all symbols, score risk, emit signals, sleep."""
        while self._running:
            try:
                if not self._enabled:
                    await asyncio.sleep(self._scan_interval)
                    continue

                # Respect rate-limit back-off
                now = time.monotonic()
                if now < self._backoff_until:
                    wait = self._backoff_until - now
                    logger.debug("Rate-limit back-off: sleeping %.1fs", wait)
                    await asyncio.sleep(wait)

                cycle_start = time.monotonic()
                scan_results: Dict[str, _SymbolSnapshot] = {}

                for symbol in self._symbols:
                    snapshot = await self._fetch_symbol_data(symbol)
                    scan_results[symbol] = snapshot
                    self._update_history(snapshot)

                # Score each symbol and potentially emit a signal
                emitted_count = 0
                for symbol, snapshot in scan_results.items():
                    risk = self._score_risk(symbol, snapshot)
                    self._current_risk[symbol] = risk

                    if risk["risk_score"] > self._risk_threshold:
                        signal = self._build_signal(symbol, snapshot, risk)
                        await self._signal_queue.put(signal)
                        emitted_count += 1

                        if risk["risk_score"] > 0.7:
                            logger.warning(
                                "HIGH CASCADE RISK  symbol=%s  score=%.3f  "
                                "direction=%s  funding=%.6f  OI_chg=%.2f%%  "
                                "LS=%.3f",
                                symbol,
                                risk["risk_score"],
                                risk["direction"],
                                risk.get("funding_rate", 0.0),
                                risk.get("open_interest_change_24h", 0.0),
                                risk.get("long_short_ratio", 0.0),
                            )

                elapsed = time.monotonic() - cycle_start
                logger.info(
                    "Scan cycle complete  symbols=%d  signals_emitted=%d  "
                    "elapsed=%.2fs",
                    len(self._symbols),
                    emitted_count,
                    elapsed,
                )

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scan cycle failed -- will retry next interval")

            await asyncio.sleep(self._scan_interval)

    # ----- data fetching ---------------------------------------------------

    async def _fetch_symbol_data(self, symbol: str) -> _SymbolSnapshot:
        """Fetch funding rate, OI, and LS ratios for *symbol* from Binance."""
        snap = _SymbolSnapshot(
            symbol=symbol, timestamp=datetime.now(timezone.utc)
        )

        snap.funding_rate = await self._fetch_funding_rate(symbol)
        snap.open_interest = await self._fetch_open_interest(symbol)
        snap.long_short_ratio = await self._fetch_global_ls_ratio(symbol)
        snap.top_trader_ls_ratio = await self._fetch_top_trader_position_ratio(
            symbol
        )

        return snap

    async def _api_get(
        self, path: str, params: Dict[str, Any]
    ) -> Optional[Any]:
        """Issue a GET against Binance Futures and return parsed JSON.

        Handles HTTP-429 (rate limit) by setting a back-off timer.
        Returns ``None`` on any error so callers never crash.
        """
        if self._session is None or self._session.closed:
            return None

        url = f"{_BINANCE_FUTURES_BASE}{path}"
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", "60"))
                    self._backoff_until = time.monotonic() + retry_after
                    logger.warning(
                        "Binance 429 rate-limit on %s -- backing off %ds",
                        path,
                        retry_after,
                    )
                    return None
                if resp.status == 418:
                    # IP ban -- long back-off
                    self._backoff_until = time.monotonic() + 300
                    logger.error(
                        "Binance 418 IP-ban on %s -- backing off 300s", path
                    )
                    return None
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning(
                        "Binance %s returned %d: %s", path, resp.status, text[:200]
                    )
                    return None
                return await resp.json()
        except asyncio.TimeoutError:
            logger.warning("Timeout fetching %s", path)
            return None
        except aiohttp.ClientError as exc:
            logger.warning("HTTP error fetching %s: %s", path, exc)
            return None

    async def _fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """GET /fapi/v1/fundingRate -- latest funding rate for *symbol*."""
        data = await self._api_get(
            _ENDPOINT_FUNDING_RATE, {"symbol": symbol, "limit": 1}
        )
        if data and isinstance(data, list) and len(data) > 0:
            try:
                return float(data[-1]["fundingRate"])
            except (KeyError, ValueError, TypeError):
                pass
        return None

    async def _fetch_open_interest(self, symbol: str) -> Optional[float]:
        """GET /fapi/v1/openInterest -- current open interest for *symbol*."""
        data = await self._api_get(
            _ENDPOINT_OPEN_INTEREST, {"symbol": symbol}
        )
        if data and isinstance(data, dict):
            try:
                return float(data["openInterest"])
            except (KeyError, ValueError, TypeError):
                pass
        return None

    async def _fetch_global_ls_ratio(self, symbol: str) -> Optional[float]:
        """GET /futures/data/globalLongShortAccountRatio -- latest ratio."""
        data = await self._api_get(
            _ENDPOINT_GLOBAL_LS_RATIO,
            {"symbol": symbol, "period": "5m", "limit": 1},
        )
        if data and isinstance(data, list) and len(data) > 0:
            try:
                return float(data[-1]["longShortRatio"])
            except (KeyError, ValueError, TypeError):
                pass
        return None

    async def _fetch_top_trader_position_ratio(
        self, symbol: str
    ) -> Optional[float]:
        """GET /futures/data/topLongShortPositionRatio -- latest ratio."""
        data = await self._api_get(
            _ENDPOINT_TOP_TRADER_POS_RATIO,
            {"symbol": symbol, "period": "5m", "limit": 1},
        )
        if data and isinstance(data, list) and len(data) > 0:
            try:
                return float(data[-1]["longShortRatio"])
            except (KeyError, ValueError, TypeError):
                pass
        return None

    # ----- history management ---------------------------------------------

    def _update_history(self, snap: _SymbolSnapshot) -> None:
        """Append the latest snapshot values to the rolling history."""
        hist = self._history.setdefault(snap.symbol, _SymbolHistory())

        if snap.funding_rate is not None:
            hist.funding_rates.append(snap.funding_rate)
        if snap.open_interest is not None:
            hist.open_interests.append(snap.open_interest)
        hist.timestamps.append(snap.timestamp)

    # ----- risk scoring ----------------------------------------------------

    def _score_risk(
        self, symbol: str, snap: _SymbolSnapshot
    ) -> Dict[str, Any]:
        """Compute a 0-1 composite cascade-risk score for *symbol*.

        Scoring components
        ------------------
        1. Extreme funding rate (>90th percentile of history) : +0.35
        2. High OI change (>10 % in 24 h)                    : +0.25
        3. Extreme long/short ratio (>2:1 or <0.5:1)         : +0.25
        4. Funding rate divergence across time                : +0.15
        """
        score = 0.0
        hist = self._history.get(symbol, _SymbolHistory())

        # ---- 1. Funding rate percentile ----------------------------------
        funding_pctl = 0.5
        if snap.funding_rate is not None and len(hist.funding_rates) >= 10:
            abs_rate = abs(snap.funding_rate)
            abs_history = [abs(r) for r in hist.funding_rates]
            abs_history_sorted = sorted(abs_history)
            count_below = sum(1 for v in abs_history_sorted if v < abs_rate)
            funding_pctl = count_below / len(abs_history_sorted)

            if funding_pctl >= self._extreme_funding_pctl:
                score += 0.35
        elif snap.funding_rate is not None and abs(snap.funding_rate) > 0.001:
            # Fallback: if history is short but rate is objectively extreme
            score += 0.20
            funding_pctl = 0.85

        # ---- 2. Open interest 24h change ---------------------------------
        oi_change_pct = 0.0
        if snap.open_interest is not None and len(hist.open_interests) >= 2:
            # Find the OI reading closest to 24 hours ago.  Each reading is
            # separated by ~scan_interval seconds.  We approximate 24h worth
            # of readings.
            readings_per_day = max(1, int(86400 / max(self._scan_interval, 1)))
            lookback_idx = min(readings_per_day, len(hist.open_interests) - 1)
            oi_24h_ago = hist.open_interests[-1 - lookback_idx]
            if oi_24h_ago > 0:
                oi_change_pct = (
                    (snap.open_interest - oi_24h_ago) / oi_24h_ago
                ) * 100.0

            if abs(oi_change_pct) > self._high_oi_change_pct:
                score += 0.25

        # ---- 3. Extreme long/short ratio ---------------------------------
        ls_ratio = snap.long_short_ratio if snap.long_short_ratio is not None else 1.0
        inverse_extreme = 1.0 / self._extreme_ls_ratio if self._extreme_ls_ratio > 0 else 0.5

        if ls_ratio >= self._extreme_ls_ratio or ls_ratio <= inverse_extreme:
            score += 0.25

        # ---- 4. Funding rate divergence ----------------------------------
        if snap.funding_rate is not None and len(hist.funding_rates) >= 20:
            recent_window = list(hist.funding_rates)[-10:]
            older_window = list(hist.funding_rates)[-20:-10]
            recent_mean = sum(recent_window) / len(recent_window)
            older_mean = sum(older_window) / len(older_window)

            # Divergence: recent mean has moved significantly from older mean
            if older_mean != 0:
                divergence_ratio = abs(recent_mean - older_mean) / abs(older_mean)
            else:
                divergence_ratio = abs(recent_mean) * 100  # scale up from zero

            if divergence_ratio > 0.5:
                score += 0.15

        # Clamp
        score = min(score, 1.0)

        # Direction inference
        direction = self._infer_direction(snap)

        return {
            "risk_score": round(score, 4),
            "direction": direction,
            "funding_rate": snap.funding_rate if snap.funding_rate is not None else 0.0,
            "funding_rate_percentile": round(funding_pctl, 4),
            "open_interest_change_24h": round(oi_change_pct, 4),
            "long_short_ratio": round(ls_ratio, 4),
            "timestamp": snap.timestamp.isoformat(),
        }

    # ----- direction inference --------------------------------------------

    @staticmethod
    def _infer_direction(snap: _SymbolSnapshot) -> str:
        """Determine whether the dominant risk is a long-liquidation cascade
        or a short-squeeze based on the current snapshot.

        Heuristic:
          - Positive funding + LS ratio > 1  =>  longs pay shorts, crowded
            long => long_liquidation risk.
          - Negative funding + LS ratio < 1  =>  shorts pay longs, crowded
            short => short_squeeze risk.
        """
        funding = snap.funding_rate if snap.funding_rate is not None else 0.0
        ls = snap.long_short_ratio if snap.long_short_ratio is not None else 1.0

        if funding > 0 and ls >= 1.0:
            return "long_liquidation"
        if funding < 0 and ls < 1.0:
            return "short_squeeze"
        # Ambiguous -- fall back to funding sign
        if funding >= 0:
            return "long_liquidation"
        return "short_squeeze"

    # ----- signal construction --------------------------------------------

    def _build_signal(
        self,
        symbol: str,
        snap: _SymbolSnapshot,
        risk: Dict[str, Any],
    ) -> CascadeRiskSignal:
        """Construct and return a ``CascadeRiskSignal`` for emission."""
        direction: str = risk["direction"]
        risk_score: float = risk["risk_score"]
        funding: float = risk.get("funding_rate", 0.0)
        funding_pctl: float = risk.get("funding_rate_percentile", 0.5)
        oi_change: float = risk.get("open_interest_change_24h", 0.0)
        ls_ratio: float = risk.get("long_short_ratio", 1.0)

        # Rough estimate of the USD value at risk for liquidation.
        # OI (in contracts / coins) * assumed average price * leverage factor.
        # This is intentionally coarse -- the detector is a *signal*, not a
        # precise accounting tool.
        oi_value = snap.open_interest if snap.open_interest is not None else 0.0
        estimated_liq_usd = oi_value * 0.05  # assume ~5 % of OI at risk

        # Expected move heuristic: higher risk => larger expected move
        expected_move = round(risk_score * 8.0, 2)  # up to ~8 % at score=1

        # Recommended action and entry trigger
        if risk_score >= 0.8:
            recommended_action = (
                "open_short" if direction == "long_liquidation" else "open_long"
            )
            entry_trigger = "immediate_market_order"
        elif risk_score >= 0.6:
            recommended_action = (
                "prepare_short" if direction == "long_liquidation" else "prepare_long"
            )
            entry_trigger = "limit_order_at_key_level"
        else:
            recommended_action = "monitor_closely"
            entry_trigger = "wait_for_confirmation"

        return CascadeRiskSignal(
            signal_id=str(uuid.uuid4()),
            timestamp=snap.timestamp,
            symbol=symbol,
            direction=direction,
            risk_score=risk_score,
            funding_rate=funding,
            funding_rate_percentile=funding_pctl,
            open_interest_change_24h=oi_change,
            long_short_ratio=ls_ratio,
            estimated_liquidation_usd=round(estimated_liq_usd, 2),
            recommended_action=recommended_action,
            entry_trigger=entry_trigger,
            expected_move_pct=expected_move,
        )
