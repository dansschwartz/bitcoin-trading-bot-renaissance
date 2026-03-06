"""Multi-Asset Straddle Engine — Direction-Free Trading via Paired LONG+SHORT Positions.

Opens simultaneous LONG and SHORT legs on an asset every interval.  One side always
wins.  Edge comes from exit asymmetry: tight stop on the loser, trailing stop on
the winner.  Vol-proportional scaling adapts stop/activation/trail to realized volatility.

Supports BTC, ETH, or any asset — each instance operates independently with its
own wallet, limits, and config.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StraddleLeg:
    """One side of a straddle (LONG or SHORT)."""

    side: str                                     # "LONG" or "SHORT"
    entry_price: float
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None             # stop_loss, trail_stop, timeout
    pnl_bps: float = 0.0
    pnl_usd: float = 0.0
    peak_favorable_bps: float = 0.0
    trail_active: bool = False
    trail_peak_bps: float = 0.0                   # best pnl since trail activated
    opened_at: float = 0.0                        # time.time()
    closed_at: Optional[float] = None
    closed: bool = False


@dataclass
class Straddle:
    """A paired LONG + SHORT position on the same asset at the same price."""

    straddle_id: int                              # DB row id
    entry_price: float
    asset: str = "BTC"                            # asset tag for DB/logging
    long_leg: StraddleLeg = field(default_factory=StraddleLeg)
    short_leg: StraddleLeg = field(default_factory=StraddleLeg)
    opened_at: float = 0.0
    size_usd: float = 0.0
    vol_prediction: Optional[float] = None
    vol_ratio: float = 1.0                        # vol-scaling ratio applied
    effective_stop_bps: float = 0.0
    effective_activation_bps: float = 0.0
    effective_trail_bps: float = 0.0


class StraddleEngine:
    """Core straddle engine with paired entry/exit logic.

    Each instance manages one asset (e.g. BTC or ETH) with its own wallet,
    safety limits, and vol-scaling config.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        db_path: str,
        logger: Optional[logging.Logger] = None,
        fleet_controller: Optional[Any] = None,
    ) -> None:
        self.log = logger or logging.getLogger(__name__)
        self.db_path = db_path
        self.fleet = fleet_controller

        # Asset identity
        self.asset: str = config.get('asset', 'BTC')
        self.pair: str = config.get('pair', f'{self.asset}-USD')

        # Sizing
        self.size_usd: float = float(config.get('size_usd', 500))
        self.wallet_usd: float = float(config.get('wallet_usd', 35000))

        # Timing
        self.interval_seconds: float = float(config.get('interval_seconds', 10))
        self.max_open: int = int(config.get('max_open', 35))
        self.max_hold_seconds: float = float(config.get('max_hold_seconds', 300))
        self.exit_check_interval: float = float(config.get('exit_check_interval', 2.0))

        # Base exit params (before vol scaling)
        self.stop_loss_bps: float = float(config.get('stop_loss_bps', 5))
        self.trail_activation_bps: float = float(config.get('trail_activation_bps', 3))
        self.trail_distance_bps: float = float(config.get('trail_distance_bps', 2))

        # Vol-proportional scaling
        self.vol_scaling: str = config.get('vol_scaling', 'none')
        self.vol_baseline_bps: float = float(config.get('vol_baseline_bps', 15.0))
        self.vol_floor: float = float(config.get('vol_floor', 0.3))
        self.vol_ceiling: float = float(config.get('vol_ceiling', 3.0))

        # Dead zone
        self.dead_zone_bps: float = float(config.get('dead_zone_bps', 2.0))
        self.use_vol_model: bool = config.get('use_vol_model', True)

        # Mode
        self.observation_mode: bool = config.get('observation_mode', False)

        # Safety limits (per-asset)
        self.max_deployed_usd: float = self.wallet_usd
        self.max_daily_loss_usd: float = float(config.get('daily_loss_limit_usd', 700))

        # Runtime state
        self.open_straddles: List[Straddle] = []
        self._last_open_time: float = 0.0
        self._running: bool = False
        self._exit_task: Optional[asyncio.Task] = None

        # Daily counters (reset at midnight UTC)
        self._daily_loss_usd: float = 0.0
        self._daily_date: str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self._total_opened: int = 0
        self._total_closed: int = 0
        self._total_pnl_usd: float = 0.0
        self._total_winners: int = 0
        self._dead_zone_blocks: int = 0

        # Rolling vol tracker (60s window of recent 1s returns in bps)
        self._recent_returns: List[float] = []
        self._last_price: float = 0.0
        self._last_price_time: float = 0.0

        # Ensure tables exist
        self._ensure_schema()

        # Close any stale OPEN straddles from a previous run
        self._cleanup_stale_straddles()

        self.log.info(
            f"StraddleEngine[{self.asset}] init: pair={self.pair} size=${self.size_usd} "
            f"wallet=${self.wallet_usd} stop={self.stop_loss_bps}bp "
            f"trail_act={self.trail_activation_bps}bp trail_dist={self.trail_distance_bps}bp "
            f"max_hold={self.max_hold_seconds}s max_open={self.max_open} "
            f"vol_scaling={self.vol_scaling} obs_mode={self.observation_mode}"
        )

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Ensure straddle tables exist with all required columns."""
        try:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS straddle_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT DEFAULT 'BTC',
                    opened_at TEXT,
                    closed_at TEXT,
                    entry_price REAL,
                    vol_prediction REAL,
                    vol_ratio REAL DEFAULT 1.0,
                    effective_stop_bps REAL,
                    effective_activation_bps REAL,
                    effective_trail_bps REAL,
                    entry_spread_bps REAL,
                    concurrent_at_entry INTEGER,
                    status TEXT DEFAULT 'OPEN',
                    long_exit_price REAL,
                    short_exit_price REAL,
                    long_exit_reason TEXT,
                    short_exit_reason TEXT,
                    long_pnl_bps REAL,
                    short_pnl_bps REAL,
                    net_pnl_bps REAL,
                    net_pnl_usd REAL,
                    size_usd REAL,
                    duration_seconds REAL,
                    dead_zone_blocked INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS straddle_legs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    straddle_id INTEGER,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl_bps REAL,
                    pnl_usd REAL,
                    peak_favorable_bps REAL,
                    opened_at TEXT,
                    closed_at TEXT,
                    FOREIGN KEY (straddle_id) REFERENCES straddle_log(id)
                )
            """)
            # Migration: add columns if missing (for existing DBs)
            for col, default in [
                ('asset', "'BTC'"),
                ('vol_ratio', '1.0'),
                ('effective_stop_bps', 'NULL'),
                ('effective_activation_bps', 'NULL'),
                ('effective_trail_bps', 'NULL'),
                ('entry_spread_bps', 'NULL'),
                ('concurrent_at_entry', 'NULL'),
            ]:
                try:
                    conn.execute(f"ALTER TABLE straddle_log ADD COLUMN {col} DEFAULT {default}")
                except sqlite3.OperationalError:
                    pass  # column already exists

            conn.commit()
            conn.close()
        except Exception as e:
            self.log.warning(f"StraddleEngine[{self.asset}] schema ensure error: {e}")

    def _cleanup_stale_straddles(self) -> None:
        """Close any OPEN straddles for this asset left from a previous bot run."""
        try:
            conn = self._get_conn()
            stale = conn.execute(
                "SELECT id, entry_price, opened_at FROM straddle_log "
                "WHERE status = 'OPEN' AND asset = ?",
                (self.asset,),
            ).fetchall()
            if not stale:
                conn.close()
                return
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            for row in stale:
                conn.execute(
                    "UPDATE straddle_log SET status='CLOSED', closed_at=?, "
                    "long_exit_reason='restart', short_exit_reason='restart', "
                    "net_pnl_bps=0, net_pnl_usd=0, duration_seconds=0 "
                    "WHERE id=?",
                    (ts, row['id']),
                )
                conn.execute(
                    "UPDATE straddle_legs SET exit_price=entry_price, exit_reason='restart', "
                    "pnl_bps=0, pnl_usd=0, closed_at=? WHERE straddle_id=?",
                    (ts, row['id']),
                )
            conn.commit()
            conn.close()
            self.log.info(
                f"StraddleEngine[{self.asset}] startup cleanup: closed {len(stale)} stale OPEN straddle(s)"
            )
        except Exception as e:
            self.log.warning(f"StraddleEngine[{self.asset}] startup cleanup error: {e}")

    def _reset_daily_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if today != self._daily_date:
            self._daily_loss_usd = 0.0
            self._daily_date = today

    # ------------------------------------------------------------------
    # Vol-proportional scaling
    # ------------------------------------------------------------------

    def _compute_vol_ratio(self, vol_pred: Optional[float] = None) -> float:
        """Compute vol scaling ratio.

        ratio = clamp(vol / baseline, floor, ceiling)
        In calm markets (mean vol ~0.73 bps), ratio clamps at floor (0.3x)
        effectively tightening params. During spikes, params widen up to ceiling.
        """
        if self.vol_scaling != 'proportional':
            return 1.0

        vol = vol_pred
        if vol is None or vol <= 0:
            # Use rolling realized vol from recent returns
            if len(self._recent_returns) >= 10:
                import math
                vol = (sum(r * r for r in self._recent_returns[-60:]) / len(self._recent_returns[-60:])) ** 0.5
            else:
                vol = self.vol_baseline_bps  # Default to baseline = ratio 1.0

        ratio = vol / self.vol_baseline_bps
        return max(self.vol_floor, min(ratio, self.vol_ceiling))

    def _scaled_params(self, vol_ratio: float) -> tuple[float, float, float]:
        """Return (stop, activation, trail) scaled by vol ratio."""
        return (
            self.stop_loss_bps * vol_ratio,
            self.trail_activation_bps * vol_ratio,
            self.trail_distance_bps * vol_ratio,
        )

    # ------------------------------------------------------------------
    # Price tracking for vol estimation
    # ------------------------------------------------------------------

    def update_price(self, price: float) -> None:
        """Feed a price tick for rolling vol estimation."""
        now = time.time()
        if self._last_price > 0 and price > 0:
            ret_bps = (price - self._last_price) / self._last_price * 10000
            self._recent_returns.append(ret_bps)
            # Keep last 120 returns (~2 min at 1s)
            if len(self._recent_returns) > 120:
                self._recent_returns = self._recent_returns[-120:]
        self._last_price = price
        self._last_price_time = now

    # ------------------------------------------------------------------
    # Open straddle
    # ------------------------------------------------------------------

    def open_straddle(
        self,
        price: float,
        vol_pred: Optional[float] = None,
    ) -> Optional[Straddle]:
        """Open a new straddle at the given price.

        Returns the Straddle object if opened, None if blocked by any gate.
        """
        self._reset_daily_if_needed()
        now = time.time()
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        # Gate: dead zone
        if self.use_vol_model and vol_pred is not None and vol_pred < self.dead_zone_bps:
            self._dead_zone_blocks += 1
            self.log.debug(
                f"STRADDLE[{self.asset}] DEAD-ZONE: vol_pred={vol_pred:.1f}bps < {self.dead_zone_bps}bps — skip"
            )
            try:
                conn = self._get_conn()
                conn.execute(
                    "INSERT INTO straddle_log (asset, opened_at, entry_price, vol_prediction, status, dead_zone_blocked, size_usd) "
                    "VALUES (?, ?, ?, ?, 'BLOCKED', 1, ?)",
                    (self.asset, ts, price, vol_pred, self.size_usd),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                self.log.warning(f"Straddle[{self.asset}] DB dead-zone log error: {e}")
            return None

        # Gate: max open
        if len(self.open_straddles) >= self.max_open:
            self.log.debug(f"STRADDLE[{self.asset}] MAX-OPEN: {len(self.open_straddles)}/{self.max_open}")
            return None

        # Gate: cooldown
        if now - self._last_open_time < self.interval_seconds:
            return None

        # Gate: max deployed USD (wallet limit)
        deployed = sum(s.size_usd * 2 for s in self.open_straddles)
        if deployed + self.size_usd * 2 > self.max_deployed_usd:
            self.log.warning(
                f"STRADDLE[{self.asset}] WALLET: deployed=${deployed:.0f} + ${self.size_usd * 2:.0f} > ${self.max_deployed_usd}"
            )
            return None

        # Gate: daily loss limit
        if self._daily_loss_usd >= self.max_daily_loss_usd:
            self.log.warning(
                f"STRADDLE[{self.asset}] DAILY LOSS: ${self._daily_loss_usd:.2f} >= ${self.max_daily_loss_usd}"
            )
            return None

        # Gate: fleet-wide limits
        if self.fleet and not self.fleet.allow_open(self.asset, self.size_usd):
            self.log.warning(f"STRADDLE[{self.asset}] FLEET LIMIT: blocked by fleet controller")
            return None

        # Compute vol scaling
        vol_ratio = self._compute_vol_ratio(vol_pred)
        eff_stop, eff_activation, eff_trail = self._scaled_params(vol_ratio)

        # Create legs
        long_leg = StraddleLeg(side="LONG", entry_price=price, opened_at=now)
        short_leg = StraddleLeg(side="SHORT", entry_price=price, opened_at=now)

        concurrent = len(self.open_straddles)

        # Insert into DB
        straddle_id = 0
        try:
            conn = self._get_conn()
            cur = conn.execute(
                """INSERT INTO straddle_log
                    (asset, opened_at, entry_price, vol_prediction, vol_ratio,
                     effective_stop_bps, effective_activation_bps, effective_trail_bps,
                     concurrent_at_entry, status, size_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)""",
                (
                    self.asset, ts, price, vol_pred, round(vol_ratio, 4),
                    round(eff_stop, 2), round(eff_activation, 2), round(eff_trail, 2),
                    concurrent, self.size_usd,
                ),
            )
            straddle_id = cur.lastrowid
            conn.execute(
                "INSERT INTO straddle_legs (straddle_id, side, entry_price, opened_at) VALUES (?, 'LONG', ?, ?)",
                (straddle_id, price, ts),
            )
            conn.execute(
                "INSERT INTO straddle_legs (straddle_id, side, entry_price, opened_at) VALUES (?, 'SHORT', ?, ?)",
                (straddle_id, price, ts),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.log.error(f"Straddle[{self.asset}] DB insert error: {e}")
            return None

        straddle = Straddle(
            straddle_id=straddle_id,
            entry_price=price,
            asset=self.asset,
            long_leg=long_leg,
            short_leg=short_leg,
            opened_at=now,
            size_usd=self.size_usd,
            vol_prediction=vol_pred,
            vol_ratio=vol_ratio,
            effective_stop_bps=eff_stop,
            effective_activation_bps=eff_activation,
            effective_trail_bps=eff_trail,
        )

        self.open_straddles.append(straddle)
        self._last_open_time = now
        self._total_opened += 1

        mode_label = "OBS" if self.observation_mode else "LIVE"
        self.log.info(
            f"STRADDLE[{self.asset}] OPEN [{mode_label}] id={straddle_id} price=${price:.2f} "
            f"size=${self.size_usd} vol_r={vol_ratio:.2f} "
            f"eff_stop={eff_stop:.1f} eff_act={eff_activation:.1f} eff_trail={eff_trail:.1f} "
            f"concurrent={concurrent}"
        )
        return straddle

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    def _compute_leg_pnl_bps(self, leg: StraddleLeg, current_price: float) -> float:
        """Compute P&L in basis points for a leg."""
        if leg.side == "LONG":
            return (current_price - leg.entry_price) / leg.entry_price * 10000
        else:  # SHORT
            return (leg.entry_price - current_price) / leg.entry_price * 10000

    def _check_leg_exit(
        self, leg: StraddleLeg, current_price: float, now: float,
        eff_stop: float, eff_activation: float, eff_trail: float,
    ) -> None:
        """Check and apply exit rules to a single leg using effective (vol-scaled) params."""
        if leg.closed:
            return

        pnl_bps = self._compute_leg_pnl_bps(leg, current_price)

        # Track peak favorable
        if pnl_bps > leg.peak_favorable_bps:
            leg.peak_favorable_bps = pnl_bps

        # Stop loss (uses effective stop)
        if pnl_bps <= -eff_stop:
            leg.exit_reason = "stop_loss"
            leg.pnl_bps = pnl_bps
            leg.exit_price = current_price
            leg.closed = True
            leg.closed_at = now
            return

        # Trail activation (uses effective activation)
        if not leg.trail_active and pnl_bps >= eff_activation:
            leg.trail_active = True
            leg.trail_peak_bps = pnl_bps

        # Trail peak update
        if leg.trail_active and pnl_bps > leg.trail_peak_bps:
            leg.trail_peak_bps = pnl_bps

        # Trail stop (uses effective trail distance)
        if leg.trail_active and (leg.trail_peak_bps - pnl_bps) >= eff_trail:
            leg.exit_reason = "trail_stop"
            leg.pnl_bps = pnl_bps
            leg.exit_price = current_price
            leg.closed = True
            leg.closed_at = now
            return

        # Timeout
        age = now - leg.opened_at
        if age >= self.max_hold_seconds:
            if abs(leg.peak_favorable_bps) < self.dead_zone_bps:
                leg.exit_reason = "timeout_flat"
            elif pnl_bps > 0:
                leg.exit_reason = "timeout_profitable"
            else:
                leg.exit_reason = "timeout_loss"
            leg.pnl_bps = pnl_bps
            leg.exit_price = current_price
            leg.closed = True
            leg.closed_at = now

    def check_exits(self, current_price: float) -> List[Straddle]:
        """Check all open straddles for exit conditions.

        Returns list of straddles that were fully closed this call.
        """
        self._reset_daily_if_needed()
        now = time.time()
        closed_straddles: List[Straddle] = []

        # Update price tracker for vol estimation
        self.update_price(current_price)

        for straddle in list(self.open_straddles):
            # Use the straddle's stored effective params
            eff_stop = straddle.effective_stop_bps
            eff_activation = straddle.effective_activation_bps
            eff_trail = straddle.effective_trail_bps

            # Check both legs with effective params
            self._check_leg_exit(straddle.long_leg, current_price, now, eff_stop, eff_activation, eff_trail)
            self._check_leg_exit(straddle.short_leg, current_price, now, eff_stop, eff_activation, eff_trail)

            # Both legs closed → straddle is done
            if straddle.long_leg.closed and straddle.short_leg.closed:
                # Compute net P&L
                net_pnl_bps = straddle.long_leg.pnl_bps + straddle.short_leg.pnl_bps
                net_pnl_usd = net_pnl_bps / 10000 * straddle.size_usd

                # Compute per-leg USD P&L
                straddle.long_leg.pnl_usd = straddle.long_leg.pnl_bps / 10000 * straddle.size_usd
                straddle.short_leg.pnl_usd = straddle.short_leg.pnl_bps / 10000 * straddle.size_usd

                duration = max(
                    (straddle.long_leg.closed_at or now) - straddle.opened_at,
                    (straddle.short_leg.closed_at or now) - straddle.opened_at,
                )

                # Update counters
                self._total_closed += 1
                self._total_pnl_usd += net_pnl_usd
                if net_pnl_bps > 0:
                    self._total_winners += 1
                if net_pnl_usd < 0:
                    self._daily_loss_usd += abs(net_pnl_usd)

                # Notify fleet controller
                if self.fleet:
                    self.fleet.report_close(self.asset, net_pnl_usd)

                # Update DB
                ts_closed = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                try:
                    conn = self._get_conn()
                    conn.execute(
                        """UPDATE straddle_log SET
                            closed_at = ?, status = 'CLOSED',
                            long_exit_price = ?, short_exit_price = ?,
                            long_exit_reason = ?, short_exit_reason = ?,
                            long_pnl_bps = ?, short_pnl_bps = ?,
                            net_pnl_bps = ?, net_pnl_usd = ?,
                            duration_seconds = ?
                        WHERE id = ?""",
                        (
                            ts_closed,
                            straddle.long_leg.exit_price,
                            straddle.short_leg.exit_price,
                            straddle.long_leg.exit_reason,
                            straddle.short_leg.exit_reason,
                            round(straddle.long_leg.pnl_bps, 2),
                            round(straddle.short_leg.pnl_bps, 2),
                            round(net_pnl_bps, 2),
                            round(net_pnl_usd, 4),
                            round(duration, 1),
                            straddle.straddle_id,
                        ),
                    )
                    # Update legs
                    for leg in (straddle.long_leg, straddle.short_leg):
                        leg_closed_ts = (
                            datetime.fromtimestamp(leg.closed_at, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                            if leg.closed_at else ts_closed
                        )
                        conn.execute(
                            """UPDATE straddle_legs SET
                                exit_price = ?, exit_reason = ?,
                                pnl_bps = ?, pnl_usd = ?,
                                peak_favorable_bps = ?, closed_at = ?
                            WHERE straddle_id = ? AND side = ?""",
                            (
                                leg.exit_price,
                                leg.exit_reason,
                                round(leg.pnl_bps, 2),
                                round(leg.pnl_usd, 4),
                                round(leg.peak_favorable_bps, 2),
                                leg_closed_ts,
                                straddle.straddle_id,
                                leg.side,
                            ),
                        )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    self.log.error(f"Straddle[{self.asset}] DB update error: {e}")

                mode_label = "OBS" if self.observation_mode else "LIVE"
                self.log.info(
                    f"STRADDLE[{self.asset}] CLOSED [{mode_label}] id={straddle.straddle_id} "
                    f"net={net_pnl_bps:+.1f}bp ${net_pnl_usd:+.4f} "
                    f"L={straddle.long_leg.exit_reason}/{straddle.long_leg.pnl_bps:+.1f}bp "
                    f"S={straddle.short_leg.exit_reason}/{straddle.short_leg.pnl_bps:+.1f}bp "
                    f"dur={duration:.0f}s vol_r={straddle.vol_ratio:.2f}"
                )

                self.open_straddles.remove(straddle)
                closed_straddles.append(straddle)

        return closed_straddles

    # ------------------------------------------------------------------
    # Background exit loop
    # ------------------------------------------------------------------

    async def start_exit_loop(self, price_fn: Callable) -> None:
        """Start the background exit check loop.

        Args:
            price_fn: async callable that returns Dict[str, float] of prices.
                      Expected key is self.pair (e.g. 'BTC-USD').
        """
        if self._running:
            return
        self._running = True
        self._exit_task = asyncio.ensure_future(self._exit_loop(price_fn))
        self.log.info(
            f"StraddleEngine[{self.asset}] exit loop started "
            f"(interval={self.exit_check_interval}s)"
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
        self.log.info(f"StraddleEngine[{self.asset}] exit loop stopped")

    async def _exit_loop(self, price_fn: Callable) -> None:
        """Internal loop: fetch price, check exits, and open new straddles.

        Runs every exit_check_interval (2s). The entry cooldown (interval_seconds=10s)
        is enforced inside open_straddle() so entries happen at the right cadence.
        """
        while self._running:
            try:
                prices = await price_fn()
                price = prices.get(self.pair)
                if price and price > 0:
                    p = float(price)
                    # Check exits on all open straddles
                    if self.open_straddles:
                        self.check_exits(p)
                    # Try to open a new straddle (cooldown gate inside)
                    self.open_straddle(p, vol_pred=None)
            except Exception as e:
                self.log.warning(f"Straddle[{self.asset}] loop error: {e}")
            await asyncio.sleep(self.exit_check_interval)

    # ------------------------------------------------------------------
    # Status (for dashboard)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine status dict for dashboard endpoint."""
        self._reset_daily_if_needed()

        open_info: List[Dict[str, Any]] = []
        now = time.time()
        for s in self.open_straddles:
            open_info.append({
                'straddle_id': s.straddle_id,
                'asset': s.asset,
                'entry_price': s.entry_price,
                'vol_prediction': s.vol_prediction,
                'vol_ratio': round(s.vol_ratio, 3),
                'effective_stop_bps': round(s.effective_stop_bps, 1),
                'effective_activation_bps': round(s.effective_activation_bps, 1),
                'effective_trail_bps': round(s.effective_trail_bps, 1),
                'age_seconds': round(now - s.opened_at, 1),
                'long_trail_active': s.long_leg.trail_active,
                'short_trail_active': s.short_leg.trail_active,
                'long_peak_bps': round(s.long_leg.peak_favorable_bps, 1),
                'short_peak_bps': round(s.short_leg.peak_favorable_bps, 1),
            })

        win_rate = (self._total_winners / self._total_closed * 100) if self._total_closed > 0 else 0.0

        return {
            'active': self._running,
            'observation_mode': self.observation_mode,
            'asset': self.asset,
            'pair': self.pair,
            'size_usd': self.size_usd,
            'wallet_usd': self.wallet_usd,
            'interval_seconds': self.interval_seconds,
            'max_open': self.max_open,
            'open_count': len(self.open_straddles),
            'open_straddles': open_info,
            'total_opened': self._total_opened,
            'total_closed': self._total_closed,
            'total_winners': self._total_winners,
            'win_rate': round(win_rate, 1),
            'total_pnl_usd': round(self._total_pnl_usd, 4),
            'daily_loss_usd': round(self._daily_loss_usd, 4),
            'max_daily_loss_usd': self.max_daily_loss_usd,
            'dead_zone_blocks': self._dead_zone_blocks,
            'vol_scaling': self.vol_scaling,
            'config': {
                'stop_loss_bps': self.stop_loss_bps,
                'trail_activation_bps': self.trail_activation_bps,
                'trail_distance_bps': self.trail_distance_bps,
                'max_hold_seconds': self.max_hold_seconds,
                'dead_zone_bps': self.dead_zone_bps,
                'use_vol_model': self.use_vol_model,
                'vol_scaling': self.vol_scaling,
                'vol_baseline_bps': self.vol_baseline_bps,
                'vol_floor': self.vol_floor,
                'vol_ceiling': self.vol_ceiling,
            },
        }


class StraddleFleetController:
    """Coordinates multiple StraddleEngine instances for fleet-wide safety.

    Enforces:
    - Fleet-wide daily loss limit ($1,500)
    - Fleet-wide max deployed capital ($15,000)
    - Cross-asset circuit breaker
    """

    def __init__(
        self,
        fleet_daily_loss_limit: float = 1500.0,
        fleet_max_deployed: float = 15000.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.log = logger or logging.getLogger(__name__)
        self.fleet_daily_loss_limit = fleet_daily_loss_limit
        self.fleet_max_deployed = fleet_max_deployed
        self.engines: Dict[str, StraddleEngine] = {}
        self._fleet_daily_loss: float = 0.0
        self._fleet_daily_date: str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self._halted: bool = False

    def register(self, engine: StraddleEngine) -> None:
        """Register a StraddleEngine instance."""
        self.engines[engine.asset] = engine
        self.log.info(f"StraddleFleet: registered {engine.asset} engine")

    def _reset_daily_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if today != self._fleet_daily_date:
            self._fleet_daily_loss = 0.0
            self._fleet_daily_date = today
            self._halted = False

    def allow_open(self, asset: str, size_usd: float) -> bool:
        """Check if a new straddle is allowed at the fleet level."""
        self._reset_daily_if_needed()

        if self._halted:
            return False

        # Check fleet daily loss
        if self._fleet_daily_loss >= self.fleet_daily_loss_limit:
            self.log.warning(
                f"StraddleFleet HALT: daily loss ${self._fleet_daily_loss:.2f} "
                f">= ${self.fleet_daily_loss_limit}"
            )
            self._halted = True
            return False

        # Check fleet-wide deployed capital
        total_deployed = 0.0
        for eng in self.engines.values():
            total_deployed += sum(s.size_usd * 2 for s in eng.open_straddles)
        if total_deployed + size_usd * 2 > self.fleet_max_deployed:
            self.log.warning(
                f"StraddleFleet DEPLOYED: ${total_deployed:.0f} + ${size_usd * 2:.0f} "
                f"> ${self.fleet_max_deployed}"
            )
            return False

        return True

    def report_close(self, asset: str, net_pnl_usd: float) -> None:
        """Report a closed straddle for fleet-wide tracking."""
        self._reset_daily_if_needed()
        if net_pnl_usd < 0:
            self._fleet_daily_loss += abs(net_pnl_usd)

    def get_fleet_status(self) -> Dict[str, Any]:
        """Return fleet-wide status for dashboard."""
        self._reset_daily_if_needed()

        total_deployed = 0.0
        total_open = 0
        total_pnl = 0.0
        per_asset: Dict[str, Any] = {}

        for asset, eng in self.engines.items():
            deployed = sum(s.size_usd * 2 for s in eng.open_straddles)
            total_deployed += deployed
            total_open += len(eng.open_straddles)
            total_pnl += eng._total_pnl_usd
            per_asset[asset] = {
                'open': len(eng.open_straddles),
                'deployed': round(deployed, 2),
                'pnl_usd': round(eng._total_pnl_usd, 4),
                'daily_loss': round(eng._daily_loss_usd, 4),
                'win_rate': round(
                    (eng._total_winners / eng._total_closed * 100) if eng._total_closed > 0 else 0, 1
                ),
                'total_closed': eng._total_closed,
            }

        return {
            'halted': self._halted,
            'fleet_daily_loss': round(self._fleet_daily_loss, 4),
            'fleet_daily_loss_limit': self.fleet_daily_loss_limit,
            'fleet_max_deployed': self.fleet_max_deployed,
            'total_deployed': round(total_deployed, 2),
            'total_open': total_open,
            'total_pnl': round(total_pnl, 4),
            'engines': per_asset,
        }
