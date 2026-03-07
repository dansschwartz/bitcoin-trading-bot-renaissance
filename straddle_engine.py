"""Multi-Asset Straddle Engine — Dollar-Denominated Exits.

Opens simultaneous LONG and SHORT legs on an asset every interval. One side always
wins. Edge comes from exit asymmetry: tight stop on the loser ($1 loss), trailing
stop on the winner ($1 giveback from peak).

All exit decisions are in DOLLARS, not basis points.
BPS are computed for logging/reference only.
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

    # Dollar P&L — used for exit decisions
    pnl_usd: float = 0.0
    peak_pnl_usd: float = 0.0                    # highest dollar gain seen
    trail_active: bool = False

    # BPS — for logging/reference only (NOT used for exit decisions)
    pnl_bps: float = 0.0
    peak_favorable_bps: float = 0.0

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

    # Dollar thresholds stored per-straddle
    stop_loss_usd: float = 1.0
    trail_activation_usd: float = 1.0
    trail_distance_usd: float = 1.0


class StraddleEngine:
    """Core straddle engine with dollar-denominated exits.

    Each instance manages one asset (e.g. BTC or ETH) with its own wallet,
    safety limits, and config.
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
        self.size_usd: float = float(config.get('size_usd', 1000))
        self.wallet_usd: float = float(config.get('wallet_usd', 70000))

        # Timing
        self.interval_seconds: float = float(config.get('interval_seconds', 10))
        self.max_open: int = int(config.get('max_open', 35))
        self.max_hold_seconds: float = float(config.get('max_hold_seconds', 300))
        self.exit_check_interval: float = float(config.get('exit_check_interval', 2.0))

        # Exit params — IN DOLLARS (not bps)
        self.stop_loss_usd: float = float(config.get('stop_loss_usd', 1.00))
        self.trail_activation_usd: float = float(config.get('trail_activation_usd', 1.00))
        self.trail_distance_usd: float = float(config.get('trail_distance_usd', 1.00))

        # Mode
        self.observation_mode: bool = config.get('observation_mode', False)

        # Safety limits (per-asset)
        self.max_deployed_usd: float = self.wallet_usd
        self.max_daily_loss_usd: float = float(config.get('daily_loss_limit_usd', 7000))

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

        # Ensure tables exist
        self._ensure_schema()

        # Close any stale OPEN straddles from a previous run
        self._cleanup_stale_straddles()

        self.log.info(
            f"StraddleEngine[{self.asset}] init: pair={self.pair} "
            f"leg=${self.size_usd:.0f} wallet=${self.wallet_usd:.0f} "
            f"stop=${self.stop_loss_usd:.2f} trail_at=${self.trail_activation_usd:.2f} "
            f"trail_dist=${self.trail_distance_usd:.2f} "
            f"max_hold={self.max_hold_seconds:.0f}s max_open={self.max_open} "
            f"obs_mode={self.observation_mode}"
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
            # Legs migration
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
                    (asset, opened_at, entry_price, size_usd,
                     stop_loss_usd, trail_activation_usd, trail_distance_usd,
                     concurrent_at_entry, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
                (
                    self.asset, ts, price, self.size_usd,
                    self.stop_loss_usd, self.trail_activation_usd, self.trail_distance_usd,
                    concurrent,
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
            stop_loss_usd=self.stop_loss_usd,
            trail_activation_usd=self.trail_activation_usd,
            trail_distance_usd=self.trail_distance_usd,
        )

        self.open_straddles.append(straddle)
        self._last_open_time = now
        self._total_opened += 1

        mode_label = "OBS" if self.observation_mode else "LIVE"
        self.log.info(
            f"STRADDLE[{self.asset}] OPEN [{mode_label}] id={straddle_id} "
            f"price=${price:,.2f} leg=${self.size_usd:.0f} "
            f"stop=${self.stop_loss_usd:.2f} trail_at=${self.trail_activation_usd:.2f} "
            f"trail_dist=${self.trail_distance_usd:.2f} "
            f"concurrent={concurrent}"
        )
        return straddle

    # ------------------------------------------------------------------
    # Exit checks — ALL IN DOLLARS
    # ------------------------------------------------------------------

    def _check_leg_exit(
        self, leg: StraddleLeg, current_price: float, now: float,
        straddle: Straddle,
    ) -> None:
        """Check and apply exit rules to a single leg. All decisions in DOLLARS."""
        if leg.closed:
            return

        # Compute P&L in DOLLARS
        if leg.side == "LONG":
            leg.pnl_usd = (current_price - leg.entry_price) / leg.entry_price * straddle.size_usd
        else:
            leg.pnl_usd = (leg.entry_price - current_price) / leg.entry_price * straddle.size_usd

        # Also compute bps for logging (NOT for decisions)
        if leg.side == "LONG":
            leg.pnl_bps = (current_price - leg.entry_price) / leg.entry_price * 10000
        else:
            leg.pnl_bps = (leg.entry_price - current_price) / leg.entry_price * 10000

        # Track peak in dollars
        if leg.pnl_usd > leg.peak_pnl_usd:
            leg.peak_pnl_usd = leg.pnl_usd
        if leg.pnl_bps > leg.peak_favorable_bps:
            leg.peak_favorable_bps = leg.pnl_bps

        # ═══ RULE 1: STOP LOSS — down $X → close immediately ═══
        if leg.pnl_usd <= -straddle.stop_loss_usd:
            leg.exit_reason = "stop_loss"
            leg.exit_price = current_price
            leg.closed = True
            leg.closed_at = now
            return

        # ═══ RULE 2: TRAIL ACTIVATION — up $X → start trailing ═══
        if not leg.trail_active and leg.pnl_usd >= straddle.trail_activation_usd:
            leg.trail_active = True

        # ═══ RULE 3: TRAIL STOP — gave back $X from peak → close ═══
        if leg.trail_active:
            drawdown_usd = leg.peak_pnl_usd - leg.pnl_usd
            if drawdown_usd >= straddle.trail_distance_usd:
                leg.exit_reason = "trail_stop"
                leg.exit_price = current_price
                leg.closed = True
                leg.closed_at = now
                return

        # ═══ RULE 4: TIMEOUT — 5 minutes ═══
        age = now - leg.opened_at
        if age >= self.max_hold_seconds:
            if leg.pnl_usd > 0.01:
                leg.exit_reason = "timeout_profitable"
            elif leg.pnl_usd < -0.01:
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
            # Check both legs with dollar thresholds
            self._check_leg_exit(straddle.long_leg, current_price, now, straddle)
            self._check_leg_exit(straddle.short_leg, current_price, now, straddle)

            # Both legs closed → straddle is done
            if straddle.long_leg.closed and straddle.short_leg.closed:
                # Net P&L — simple dollar sum
                net_pnl_usd = straddle.long_leg.pnl_usd + straddle.short_leg.pnl_usd
                net_pnl_bps = straddle.long_leg.pnl_bps + straddle.short_leg.pnl_bps

                duration = max(
                    (straddle.long_leg.closed_at or now) - straddle.opened_at,
                    (straddle.short_leg.closed_at or now) - straddle.opened_at,
                )

                # Update counters
                self._total_closed += 1
                self._total_pnl_usd += net_pnl_usd
                if net_pnl_usd > 0:
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
                    f"net=${net_pnl_usd:+.2f} "
                    f"L={straddle.long_leg.exit_reason}/${straddle.long_leg.pnl_usd:+.2f} "
                    f"S={straddle.short_leg.exit_reason}/${straddle.short_leg.pnl_usd:+.2f} "
                    f"dur={duration:.0f}s"
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
        tick = 0
        while self._running:
            try:
                prices = await price_fn()
                price = prices.get(self.pair)
                if price and price > 0:
                    p = float(price)
                    # Check exits on all open straddles
                    if self.open_straddles:
                        self.check_exits(p)
                    # Sync live P&L to DB every ~10s so dashboard can display it
                    tick += 1
                    if tick % 5 == 0 and self.open_straddles:
                        self._sync_open_pnl_to_db()
                    # Try to open a new straddle (cooldown gate inside)
                    self.open_straddle(p, vol_pred=None)
            except Exception as e:
                self.log.warning(f"Straddle[{self.asset}] loop error: {e}")
            await asyncio.sleep(self.exit_check_interval)

    def _sync_open_pnl_to_db(self) -> None:
        """Write live P&L for all open straddles to DB so dashboard can read it."""
        try:
            conn = self._get_conn()
            for s in self.open_straddles:
                conn.execute(
                    """UPDATE straddle_log SET
                        long_pnl_usd = ?, short_pnl_usd = ?,
                        long_peak_usd = ?, short_peak_usd = ?,
                        long_pnl_bps = ?, short_pnl_bps = ?,
                        net_pnl_usd = ?
                    WHERE id = ? AND status = 'OPEN'""",
                    (
                        round(s.long_leg.pnl_usd, 4),
                        round(s.short_leg.pnl_usd, 4),
                        round(s.long_leg.peak_pnl_usd, 4),
                        round(s.short_leg.peak_pnl_usd, 4),
                        round(s.long_leg.pnl_bps, 2),
                        round(s.short_leg.pnl_bps, 2),
                        round(s.long_leg.pnl_usd + s.short_leg.pnl_usd, 4),
                        s.straddle_id,
                    ),
                )
                for leg in (s.long_leg, s.short_leg):
                    conn.execute(
                        """UPDATE straddle_legs SET
                            pnl_usd = ?, peak_pnl_usd = ?,
                            pnl_bps = ?, peak_favorable_bps = ?
                        WHERE straddle_id = ? AND side = ?""",
                        (
                            round(leg.pnl_usd, 4),
                            round(leg.peak_pnl_usd, 4),
                            round(leg.pnl_bps, 2),
                            round(leg.peak_favorable_bps, 2),
                            s.straddle_id,
                            leg.side,
                        ),
                    )
            conn.commit()
            conn.close()
        except Exception as e:
            self.log.debug(f"Straddle[{self.asset}] pnl sync error: {e}")

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
                'stop_loss_usd': s.stop_loss_usd,
                'trail_activation_usd': s.trail_activation_usd,
                'trail_distance_usd': s.trail_distance_usd,
                'age_seconds': round(now - s.opened_at, 1),
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
            'total_pnl_usd': round(self._total_pnl_usd, 2),
            'daily_loss_usd': round(self._daily_loss_usd, 2),
            'max_daily_loss_usd': self.max_daily_loss_usd,
            'config': {
                'stop_loss_usd': self.stop_loss_usd,
                'trail_activation_usd': self.trail_activation_usd,
                'trail_distance_usd': self.trail_distance_usd,
                'max_hold_seconds': self.max_hold_seconds,
            },
        }


class StraddleFleetController:
    """Coordinates multiple StraddleEngine instances for fleet-wide safety."""

    def __init__(
        self,
        fleet_daily_loss_limit: float = 15000.0,
        fleet_max_deployed: float = 150000.0,
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
