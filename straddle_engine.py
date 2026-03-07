"""Multi-Asset Straddle Engine — BPS-Based Exits with Dollar Display.

Opens simultaneous LONG and SHORT legs on an asset every interval. One side always
wins. Edge comes from exit asymmetry: tight stop on the loser (4bp), trailing
stop on the winner (1bp giveback from peak).

All exit DECISIONS are in BASIS POINTS (validated by optimizer over 52,650 simulations).
Dollar amounts are computed for logging and dashboard display only.

Optimizer parameters (do not change without re-running optimizer):
  stop_loss_bps=4, trail_activation_bps=2, trail_distance_bps=1
  leg_size_usd=100, max_hold=120s, entry_interval=10s, check_interval=1s
  vol_scaling=proportional (0.3x-3.0x base on 15bps)
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

    # BPS P&L — used for EXIT DECISIONS
    pnl_bps: float = 0.0
    peak_pnl_bps: float = 0.0                    # highest bps gain seen
    trail_active: bool = False

    # Dollar P&L — for logging/display only
    pnl_usd: float = 0.0
    peak_pnl_usd: float = 0.0

    # Per-leg BPS thresholds (vol-scaled at entry)
    stop_loss_bps: float = 4.0
    trail_activation_bps: float = 2.0
    trail_distance_bps: float = 1.0

    opened_at: float = 0.0                        # time.time()
    closed_at: Optional[float] = None
    closed: bool = False


@dataclass
class Straddle:
    """A paired LONG + SHORT position on the same asset at the same price."""

    straddle_id: int                              # DB row id
    entry_price: float
    asset: str = "BTC"
    long_leg: StraddleLeg = field(default_factory=StraddleLeg)
    short_leg: StraddleLeg = field(default_factory=StraddleLeg)
    opened_at: float = 0.0
    size_usd: float = 0.0                         # per-leg size
    vol_ratio: float = 1.0                         # vol scaling ratio at entry


class StraddleEngine:
    """Core straddle engine with BPS-based exit decisions.

    Each instance manages one asset (e.g. BTC or ETH) with its own wallet,
    safety limits, and config. Exit logic uses basis points validated by
    the optimizer. Dollar amounts are derived for display only.
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
        self.binance_symbol: str = config.get('binance_symbol', self.pair.replace('-', '').replace('USD', 'USDT'))

        # ── OPTIMIZER VALUES (do not change without re-running optimizer) ──
        self.leg_size_usd: float = float(config.get('leg_size_usd', 100))
        self.straddle_interval: float = float(config.get('straddle_interval_seconds', 10))
        self.exit_check_interval: float = float(config.get('check_interval_seconds', 1))
        self.max_hold_seconds: float = float(config.get('max_hold_seconds', 120))
        self.stop_loss_bps: float = float(config.get('stop_loss_bps', 4))
        self.trail_activation_bps: float = float(config.get('trail_activation_bps', 2))
        self.trail_distance_bps: float = float(config.get('trail_distance_bps', 1))

        # ── VOL SCALING ──
        self.vol_scaling: str = config.get('vol_scaling', 'proportional')
        self.vol_scaling_base_vol: float = float(config.get('vol_scaling_base_vol', 15.0))
        self.vol_scaling_min_ratio: float = float(config.get('vol_scaling_min_ratio', 0.3))
        self.vol_scaling_max_ratio: float = float(config.get('vol_scaling_max_ratio', 3.0))

        # ── CAPACITY ──
        self.max_open: int = int(config.get('max_open_straddles', 35))
        self.max_capital_deployed: float = float(config.get('max_capital_deployed', 7000))
        self.max_daily_loss_usd: float = float(config.get('max_daily_loss_usd', 700))

        # Mode
        self.observation_mode: bool = config.get('observation_mode', False)

        # Runtime state
        self.open_straddles: List[Straddle] = []
        self._last_open_time: float = 0.0
        self._running: bool = False
        self._exit_task: Optional[asyncio.Task] = None

        # Vol ratio cache (refreshed every 30s)
        self._cached_vol_ratio: float = 1.0
        self._last_vol_check: float = 0.0

        # Recent price history for vol computation (list of (timestamp, price))
        self._price_history: List[tuple[float, float]] = []
        self._vol_history_max: int = 120  # 120 seconds of 1s prices

        # Daily counters (reset at midnight UTC)
        self._daily_loss_usd: float = 0.0
        self._daily_date: str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self._total_opened: int = 0
        self._total_closed: int = 0
        self._total_pnl_usd: float = 0.0
        self._total_winners: int = 0

        # Ensure tables exist
        self._ensure_schema()

        # Close any stale OPEN straddles from a previous run
        self._cleanup_stale_straddles()

        self.log.info(
            f"StraddleEngine[{self.asset}] init: pair={self.pair} "
            f"leg=${self.leg_size_usd:.0f} "
            f"stop={self.stop_loss_bps}bp act={self.trail_activation_bps}bp "
            f"trail={self.trail_distance_bps}bp "
            f"hold={self.max_hold_seconds:.0f}s entry={self.straddle_interval:.0f}s "
            f"check={self.exit_check_interval:.0f}s "
            f"vol_scaling={self.vol_scaling} max_open={self.max_open} "
            f"obs_mode={self.observation_mode}"
        )

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5)
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
                    status TEXT DEFAULT 'OPEN',
                    size_usd REAL,
                    stop_loss_usd REAL,
                    trail_activation_usd REAL,
                    trail_distance_usd REAL,
                    concurrent_at_entry INTEGER,
                    long_exit_price REAL,
                    short_exit_price REAL,
                    long_exit_reason TEXT,
                    short_exit_reason TEXT,
                    long_pnl_usd REAL,
                    short_pnl_usd REAL,
                    long_peak_usd REAL,
                    short_peak_usd REAL,
                    long_pnl_bps REAL,
                    short_pnl_bps REAL,
                    net_pnl_usd REAL,
                    net_pnl_bps REAL,
                    duration_seconds REAL,
                    dead_zone_blocked INTEGER DEFAULT 0,
                    vol_prediction REAL,
                    vol_ratio REAL DEFAULT 1.0,
                    effective_stop_bps REAL,
                    effective_activation_bps REAL,
                    effective_trail_bps REAL,
                    entry_spread_bps REAL
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
                    pnl_usd REAL,
                    pnl_bps REAL,
                    peak_pnl_usd REAL,
                    peak_favorable_bps REAL,
                    opened_at TEXT,
                    closed_at TEXT,
                    FOREIGN KEY (straddle_id) REFERENCES straddle_log(id)
                )
            """)
            # Migration: add columns if missing (for existing DBs)
            for col, default in [
                ('asset', "'BTC'"),
                ('stop_loss_usd', 'NULL'),
                ('trail_activation_usd', 'NULL'),
                ('trail_distance_usd', 'NULL'),
                ('long_pnl_usd', 'NULL'),
                ('short_pnl_usd', 'NULL'),
                ('long_peak_usd', 'NULL'),
                ('short_peak_usd', 'NULL'),
                ('vol_ratio', '1.0'),
                ('effective_stop_bps', 'NULL'),
                ('effective_activation_bps', 'NULL'),
                ('effective_trail_bps', 'NULL'),
                ('entry_spread_bps', 'NULL'),
                ('concurrent_at_entry', 'NULL'),
                ('vol_prediction', 'NULL'),
            ]:
                try:
                    conn.execute(f"ALTER TABLE straddle_log ADD COLUMN {col} DEFAULT {default}")
                except sqlite3.OperationalError:
                    pass
            for col, default in [
                ('peak_pnl_usd', 'NULL'),
            ]:
                try:
                    conn.execute(f"ALTER TABLE straddle_legs ADD COLUMN {col} DEFAULT {default}")
                except sqlite3.OperationalError:
                    pass

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
                    "net_pnl_usd=0, net_pnl_bps=0, duration_seconds=0 "
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
    # Vol scaling
    # ------------------------------------------------------------------

    def _compute_vol_bps(self) -> float:
        """Compute realized 1-second volatility in bps from recent price history."""
        if len(self._price_history) < 30:
            return self.vol_scaling_base_vol  # Not enough data, return base

        # Use last 60 prices (or whatever we have)
        recent = self._price_history[-60:]
        if len(recent) < 2:
            return self.vol_scaling_base_vol

        returns_bps: list[float] = []
        for i in range(1, len(recent)):
            if recent[i - 1][1] > 0:
                ret = (recent[i][1] - recent[i - 1][1]) / recent[i - 1][1] * 10000
                returns_bps.append(ret)

        if not returns_bps:
            return self.vol_scaling_base_vol

        # Standard deviation of 1s returns in bps
        mean = sum(returns_bps) / len(returns_bps)
        variance = sum((r - mean) ** 2 for r in returns_bps) / len(returns_bps)
        return variance ** 0.5

    def _get_vol_ratio(self) -> float:
        """Get cached vol scaling ratio. Refreshes every 30 seconds."""
        if self.vol_scaling != 'proportional':
            return 1.0

        now = time.time()
        if now - self._last_vol_check > 30:
            vol_bps = self._compute_vol_bps()
            if vol_bps > 0:
                ratio = vol_bps / self.vol_scaling_base_vol
                self._cached_vol_ratio = max(
                    self.vol_scaling_min_ratio,
                    min(ratio, self.vol_scaling_max_ratio),
                )
            else:
                self._cached_vol_ratio = 1.0
            self._last_vol_check = now

        return self._cached_vol_ratio

    def _record_price(self, price: float) -> None:
        """Record a price tick for vol computation."""
        now = time.time()
        self._price_history.append((now, price))
        # Trim to max window
        if len(self._price_history) > self._vol_history_max:
            self._price_history = self._price_history[-self._vol_history_max:]

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

        # Gate: max open
        if len(self.open_straddles) >= self.max_open:
            return None

        # Gate: cooldown
        if now - self._last_open_time < self.straddle_interval:
            return None

        # Gate: max deployed USD
        deployed = sum(s.size_usd * 2 for s in self.open_straddles)
        if deployed + self.leg_size_usd * 2 > self.max_capital_deployed:
            return None

        # Gate: daily loss limit
        if self._daily_loss_usd >= self.max_daily_loss_usd:
            self.log.warning(
                f"STRADDLE[{self.asset}] DAILY LOSS: ${self._daily_loss_usd:.2f} >= ${self.max_daily_loss_usd}"
            )
            return None

        # Gate: fleet-wide limits
        if self.fleet and not self.fleet.allow_open(self.asset, self.leg_size_usd):
            return None

        # Compute vol-scaled thresholds for this straddle
        vol_ratio = self._get_vol_ratio()
        eff_stop = self.stop_loss_bps * vol_ratio
        eff_act = self.trail_activation_bps * vol_ratio
        eff_trail = self.trail_distance_bps * vol_ratio

        # Create legs with per-leg scaled thresholds
        long_leg = StraddleLeg(
            side="LONG", entry_price=price, opened_at=now,
            stop_loss_bps=eff_stop, trail_activation_bps=eff_act,
            trail_distance_bps=eff_trail,
        )
        short_leg = StraddleLeg(
            side="SHORT", entry_price=price, opened_at=now,
            stop_loss_bps=eff_stop, trail_activation_bps=eff_act,
            trail_distance_bps=eff_trail,
        )
        concurrent = len(self.open_straddles)

        # Dollar equivalents for DB/display
        stop_usd = eff_stop / 10000 * self.leg_size_usd
        act_usd = eff_act / 10000 * self.leg_size_usd
        trail_usd = eff_trail / 10000 * self.leg_size_usd

        # Insert into DB
        straddle_id = 0
        try:
            conn = self._get_conn()
            cur = conn.execute(
                """INSERT INTO straddle_log
                    (asset, opened_at, entry_price, size_usd,
                     stop_loss_usd, trail_activation_usd, trail_distance_usd,
                     vol_ratio, effective_stop_bps, effective_activation_bps,
                     effective_trail_bps, concurrent_at_entry, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
                (
                    self.asset, ts, price, self.leg_size_usd,
                    round(stop_usd, 4), round(act_usd, 4), round(trail_usd, 4),
                    round(vol_ratio, 3), round(eff_stop, 2), round(eff_act, 2),
                    round(eff_trail, 2), concurrent,
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
            size_usd=self.leg_size_usd,
            vol_ratio=vol_ratio,
        )

        self.open_straddles.append(straddle)
        self._last_open_time = now
        self._total_opened += 1

        mode_label = "OBS" if self.observation_mode else "LIVE"
        self.log.info(
            f"STRADDLE[{self.asset}] OPEN [{mode_label}] id={straddle_id} "
            f"price=${price:,.2f} leg=${self.leg_size_usd:.0f} "
            f"stop={eff_stop:.1f}bp act={eff_act:.1f}bp trail={eff_trail:.1f}bp "
            f"vol_r={vol_ratio:.2f} concurrent={concurrent}"
        )
        return straddle

    # ------------------------------------------------------------------
    # Exit checks — ALL IN BPS
    # ------------------------------------------------------------------

    def _check_leg_exit(
        self, leg: StraddleLeg, current_price: float, now: float,
        straddle: Straddle,
    ) -> None:
        """Check and apply exit rules to a single leg. All decisions in BPS."""
        if leg.closed:
            return

        # Compute P&L in BPS (used for exit decisions)
        if leg.side == "LONG":
            leg.pnl_bps = (current_price - leg.entry_price) / leg.entry_price * 10000
        else:
            leg.pnl_bps = (leg.entry_price - current_price) / leg.entry_price * 10000

        # Compute dollar P&L (for display only)
        leg.pnl_usd = leg.pnl_bps / 10000 * straddle.size_usd

        # Track peak in BPS
        if leg.pnl_bps > leg.peak_pnl_bps:
            leg.peak_pnl_bps = leg.pnl_bps
            leg.peak_pnl_usd = leg.pnl_usd

        # ═══ RULE 1: STOP LOSS — down Xbp → close immediately ═══
        if leg.pnl_bps <= -leg.stop_loss_bps:
            leg.exit_reason = "stop_loss"
            leg.exit_price = current_price
            leg.closed = True
            leg.closed_at = now
            return

        # ═══ RULE 2: TRAIL ACTIVATION — up Xbp → start trailing ═══
        if not leg.trail_active and leg.pnl_bps >= leg.trail_activation_bps:
            leg.trail_active = True

        # ═══ RULE 3: TRAIL STOP — gave back Xbp from peak → close ═══
        if leg.trail_active:
            drawdown_bps = leg.peak_pnl_bps - leg.pnl_bps
            if drawdown_bps >= leg.trail_distance_bps:
                leg.exit_reason = "trail_stop"
                leg.exit_price = current_price
                leg.closed = True
                leg.closed_at = now
                return

        # ═══ RULE 4: TIMEOUT ═══
        age = now - leg.opened_at
        if age >= self.max_hold_seconds:
            if leg.pnl_bps > 0.1:
                leg.exit_reason = "timeout_profitable"
            elif leg.pnl_bps < -0.1:
                leg.exit_reason = "timeout_loss"
            else:
                leg.exit_reason = "timeout_flat"
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

        for straddle in list(self.open_straddles):
            # Check both legs with BPS thresholds
            self._check_leg_exit(straddle.long_leg, current_price, now, straddle)
            self._check_leg_exit(straddle.short_leg, current_price, now, straddle)

            # Both legs closed → straddle is done
            if straddle.long_leg.closed and straddle.short_leg.closed:
                net_pnl_bps = straddle.long_leg.pnl_bps + straddle.short_leg.pnl_bps
                net_pnl_usd = net_pnl_bps / 10000 * straddle.size_usd

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
                            long_pnl_usd = ?, short_pnl_usd = ?,
                            long_peak_usd = ?, short_peak_usd = ?,
                            long_pnl_bps = ?, short_pnl_bps = ?,
                            net_pnl_usd = ?, net_pnl_bps = ?,
                            duration_seconds = ?
                        WHERE id = ?""",
                        (
                            ts_closed,
                            straddle.long_leg.exit_price,
                            straddle.short_leg.exit_price,
                            straddle.long_leg.exit_reason,
                            straddle.short_leg.exit_reason,
                            round(straddle.long_leg.pnl_usd, 4),
                            round(straddle.short_leg.pnl_usd, 4),
                            round(straddle.long_leg.peak_pnl_usd, 4),
                            round(straddle.short_leg.peak_pnl_usd, 4),
                            round(straddle.long_leg.pnl_bps, 2),
                            round(straddle.short_leg.pnl_bps, 2),
                            round(net_pnl_usd, 4),
                            round(net_pnl_bps, 2),
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
                                pnl_usd = ?, pnl_bps = ?,
                                peak_pnl_usd = ?, peak_favorable_bps = ?,
                                closed_at = ?
                            WHERE straddle_id = ? AND side = ?""",
                            (
                                leg.exit_price,
                                leg.exit_reason,
                                round(leg.pnl_usd, 4),
                                round(leg.pnl_bps, 2),
                                round(leg.peak_pnl_usd, 4),
                                round(leg.peak_pnl_bps, 2),
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
                    f"net=${net_pnl_usd:+.4f} ({net_pnl_bps:+.1f}bp) "
                    f"L={straddle.long_leg.exit_reason}/${straddle.long_leg.pnl_usd:+.4f} "
                    f"S={straddle.short_leg.exit_reason}/${straddle.short_leg.pnl_usd:+.4f} "
                    f"dur={duration:.0f}s vol_r={straddle.vol_ratio:.2f}"
                )

                self.open_straddles.remove(straddle)
                closed_straddles.append(straddle)

        return closed_straddles

    # ------------------------------------------------------------------
    # Background exit loop
    # ------------------------------------------------------------------

    async def start_exit_loop(self, price_fn: Callable) -> None:
        """Start the background exit check loop."""
        if self._running:
            return
        self._running = True
        self._exit_task = asyncio.ensure_future(self._exit_loop(price_fn))
        self.log.info(
            f"StraddleEngine[{self.asset}] exit loop started "
            f"(check_interval={self.exit_check_interval}s)"
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

        Runs every check_interval (1s). The entry cooldown (straddle_interval=10s)
        is enforced inside open_straddle() so entries happen at the right cadence.
        """
        while self._running:
            try:
                prices = await price_fn()
                price = prices.get(self.pair)
                if price and price > 0:
                    p = float(price)
                    # Record price for vol computation
                    self._record_price(p)
                    # Check exits on all open straddles
                    if self.open_straddles:
                        self.check_exits(p)
                        self._sync_open_pnl_to_db()
                    # Try to open a new straddle (cooldown gate inside)
                    self.open_straddle(p, vol_pred=None)
            except Exception as e:
                self.log.warning(f"Straddle[{self.asset}] loop error: {e}")
            await asyncio.sleep(self.exit_check_interval)

    def _sync_open_pnl_to_db(self) -> None:
        """Write live P&L for all open straddles to DB so dashboard can read it."""
        n = len(self.open_straddles)
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            for s in self.open_straddles:
                conn.execute(
                    """UPDATE straddle_log SET
                        long_pnl_usd = ?, short_pnl_usd = ?,
                        long_peak_usd = ?, short_peak_usd = ?,
                        long_pnl_bps = ?, short_pnl_bps = ?,
                        net_pnl_usd = ?, net_pnl_bps = ?
                    WHERE id = ? AND status = 'OPEN'""",
                    (
                        round(s.long_leg.pnl_usd, 4),
                        round(s.short_leg.pnl_usd, 4),
                        round(s.long_leg.peak_pnl_usd, 4),
                        round(s.short_leg.peak_pnl_usd, 4),
                        round(s.long_leg.pnl_bps, 2),
                        round(s.short_leg.pnl_bps, 2),
                        round(s.long_leg.pnl_usd + s.short_leg.pnl_usd, 4),
                        round(s.long_leg.pnl_bps + s.short_leg.pnl_bps, 2),
                        s.straddle_id,
                    ),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            self.log.warning(f"Straddle[{self.asset}] pnl sync error ({n} straddles): {e}")

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
                'size_usd': s.size_usd,
                'stop_loss_bps': s.long_leg.stop_loss_bps,
                'trail_activation_bps': s.long_leg.trail_activation_bps,
                'trail_distance_bps': s.long_leg.trail_distance_bps,
                'stop_loss_usd': round(s.long_leg.stop_loss_bps / 10000 * s.size_usd, 4),
                'trail_activation_usd': round(s.long_leg.trail_activation_bps / 10000 * s.size_usd, 4),
                'trail_distance_usd': round(s.long_leg.trail_distance_bps / 10000 * s.size_usd, 4),
                'vol_ratio': round(s.vol_ratio, 2),
                'age_seconds': round(now - s.opened_at, 1),
                'long_pnl_bps': round(s.long_leg.pnl_bps, 2),
                'short_pnl_bps': round(s.short_leg.pnl_bps, 2),
                'long_pnl_usd': round(s.long_leg.pnl_usd, 4),
                'short_pnl_usd': round(s.short_leg.pnl_usd, 4),
                'long_peak_usd': round(s.long_leg.peak_pnl_usd, 4),
                'short_peak_usd': round(s.short_leg.peak_pnl_usd, 4),
                'long_trail_active': s.long_leg.trail_active,
                'short_trail_active': s.short_leg.trail_active,
            })

        win_rate = (self._total_winners / self._total_closed * 100) if self._total_closed > 0 else 0.0

        return {
            'active': self._running,
            'observation_mode': self.observation_mode,
            'asset': self.asset,
            'pair': self.pair,
            'size_usd': self.leg_size_usd,
            'wallet_usd': self.max_capital_deployed,
            'interval_seconds': self.straddle_interval,
            'max_open': self.max_open,
            'open_count': len(self.open_straddles),
            'open_straddles': open_info,
            'total_opened': self._total_opened,
            'total_closed': self._total_closed,
            'total_winners': self._total_winners,
            'win_rate': round(win_rate, 1),
            'total_pnl_usd': round(self._total_pnl_usd, 2),
            'daily_loss_usd': round(self._daily_loss_usd, 2),
            'max_daily_loss_usd': self.max_daily_loss_usd,
            'vol_ratio': round(self._cached_vol_ratio, 2),
            'config': {
                'stop_loss_bps': self.stop_loss_bps,
                'trail_activation_bps': self.trail_activation_bps,
                'trail_distance_bps': self.trail_distance_bps,
                'max_hold_seconds': self.max_hold_seconds,
                'vol_scaling': self.vol_scaling,
                'check_interval': self.exit_check_interval,
            },
        }


class StraddleFleetController:
    """Coordinates multiple StraddleEngine instances for fleet-wide safety."""

    def __init__(
        self,
        fleet_daily_loss_limit: float = 1400.0,
        fleet_max_deployed: float = 14000.0,
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

        if self._fleet_daily_loss >= self.fleet_daily_loss_limit:
            self.log.warning(
                f"StraddleFleet HALT: daily loss ${self._fleet_daily_loss:.2f} "
                f">= ${self.fleet_daily_loss_limit}"
            )
            self._halted = True
            return False

        total_deployed = 0.0
        for eng in self.engines.values():
            total_deployed += sum(s.size_usd * 2 for s in eng.open_straddles)
        if total_deployed + size_usd * 2 > self.fleet_max_deployed:
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
                'deployed': round(deployed, 0),
                'pnl_usd': round(eng._total_pnl_usd, 2),
                'daily_loss': round(eng._daily_loss_usd, 2),
                'win_rate': round(
                    (eng._total_winners / eng._total_closed * 100) if eng._total_closed > 0 else 0, 1
                ),
                'total_closed': eng._total_closed,
            }

        return {
            'halted': self._halted,
            'fleet_daily_loss': round(self._fleet_daily_loss, 2),
            'fleet_daily_loss_limit': self.fleet_daily_loss_limit,
            'fleet_max_deployed': self.fleet_max_deployed,
            'total_deployed': round(total_deployed, 0),
            'total_open': total_open,
            'total_pnl': round(total_pnl, 2),
            'engines': per_asset,
        }
