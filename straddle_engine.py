"""BTC Straddle Engine — Direction-Free Trading via Paired LONG+SHORT Positions.

Opens simultaneous LONG and SHORT legs on BTC every interval.  One side always
wins.  Edge comes from exit asymmetry: tight stop on the loser, trailing stop
on the winner.  The vol model's dead_zone filter blocks entries when predicted
movement is too small to overcome the stop cost.
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
    long_leg: StraddleLeg = field(default_factory=StraddleLeg)
    short_leg: StraddleLeg = field(default_factory=StraddleLeg)
    opened_at: float = 0.0
    size_usd: float = 0.0
    vol_prediction: Optional[float] = None


class StraddleEngine:
    """Core straddle engine with paired entry/exit logic."""

    # Hard safety limits — not configurable
    MAX_STRADDLES = 20
    MAX_DEPLOYED_USD = 500
    MAX_PER_LEG_USD = 25
    MAX_DAILY_LOSS_USD = 250

    def __init__(
        self,
        config: Dict[str, Any],
        db_path: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.log = logger or logging.getLogger(__name__)
        self.db_path = db_path

        # Config with defaults
        self.pair: str = config.get('pair', 'BTCUSDT')
        self.size_usd: float = min(float(config.get('size_usd', 5.0)), self.MAX_PER_LEG_USD)
        self.interval_seconds: float = float(config.get('interval_seconds', 60))
        self.max_open: int = int(config.get('max_open', 1))
        self.stop_loss_bps: float = float(config.get('stop_loss_bps', 6))
        self.trail_activation_bps: float = float(config.get('trail_activation_bps', 4))
        self.trail_distance_bps: float = float(config.get('trail_distance_bps', 4))
        self.max_hold_seconds: float = float(config.get('max_hold_seconds', 600))
        self.dead_zone_bps: float = float(config.get('dead_zone_bps', 2.0))
        self.use_vol_model: bool = config.get('use_vol_model', True)
        self.exit_check_interval: float = float(config.get('exit_check_interval', 1.0))
        self.observation_mode: bool = config.get('observation_mode', True)

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

        # Close any stale OPEN straddles from a previous run
        self._cleanup_stale_straddles()

        self.log.info(
            f"StraddleEngine init: pair={self.pair} size=${self.size_usd} "
            f"stop={self.stop_loss_bps}bp trail_act={self.trail_activation_bps}bp "
            f"trail_dist={self.trail_distance_bps}bp max_hold={self.max_hold_seconds}s "
            f"obs_mode={self.observation_mode}"
        )

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _cleanup_stale_straddles(self) -> None:
        """Close any OPEN straddles left from a previous bot run."""
        try:
            conn = self._get_conn()
            stale = conn.execute(
                "SELECT id, entry_price, opened_at FROM straddle_log WHERE status = 'OPEN'"
            ).fetchall()
            if not stale:
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
            self.log.info(f"Straddle startup cleanup: closed {len(stale)} stale OPEN straddle(s)")
        except Exception as e:
            self.log.warning(f"Straddle startup cleanup error: {e}")

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

        # Gate: dead zone
        if self.use_vol_model and vol_pred is not None and vol_pred < self.dead_zone_bps:
            self._dead_zone_blocks += 1
            self.log.debug(
                f"STRADDLE DEAD-ZONE: vol_pred={vol_pred:.1f}bps < {self.dead_zone_bps}bps — skip"
            )
            # Log dead-zone block to DB
            try:
                conn = self._get_conn()
                conn.execute(
                    "INSERT INTO straddle_log (opened_at, entry_price, vol_prediction, status, dead_zone_blocked, size_usd) "
                    "VALUES (?, ?, ?, 'BLOCKED', 1, ?)",
                    (ts, price, vol_pred, self.size_usd),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                self.log.warning(f"Straddle DB dead-zone log error: {e}")
            return None

        # Gate: max open
        if len(self.open_straddles) >= self.max_open:
            self.log.debug(f"STRADDLE MAX-OPEN: {len(self.open_straddles)}/{self.max_open}")
            return None

        # Gate: cooldown
        if now - self._last_open_time < self.interval_seconds:
            return None

        # Gate: max deployed USD
        deployed = sum(s.size_usd * 2 for s in self.open_straddles)
        if deployed + self.size_usd * 2 > self.MAX_DEPLOYED_USD:
            self.log.warning(f"STRADDLE BUDGET: deployed=${deployed:.0f} + ${self.size_usd * 2:.0f} > ${self.MAX_DEPLOYED_USD}")
            return None

        # Gate: daily loss limit
        if self._daily_loss_usd >= self.MAX_DAILY_LOSS_USD:
            self.log.warning(f"STRADDLE DAILY LOSS: ${self._daily_loss_usd:.2f} >= ${self.MAX_DAILY_LOSS_USD}")
            return None

        # Gate: hard max straddles
        if self._total_opened >= self.MAX_STRADDLES and len(self.open_straddles) > 0:
            pass  # Allow opening if no straddles are currently open

        # Create legs
        long_leg = StraddleLeg(side="LONG", entry_price=price, opened_at=now)
        short_leg = StraddleLeg(side="SHORT", entry_price=price, opened_at=now)

        # Insert into DB
        straddle_id = 0
        try:
            conn = self._get_conn()
            cur = conn.execute(
                "INSERT INTO straddle_log (opened_at, entry_price, vol_prediction, status, size_usd) "
                "VALUES (?, ?, ?, 'OPEN', ?)",
                (ts, price, vol_pred, self.size_usd),
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
            self.log.error(f"Straddle DB insert error: {e}")
            return None

        straddle = Straddle(
            straddle_id=straddle_id,
            entry_price=price,
            long_leg=long_leg,
            short_leg=short_leg,
            opened_at=now,
            size_usd=self.size_usd,
            vol_prediction=vol_pred,
        )

        self.open_straddles.append(straddle)
        self._last_open_time = now
        self._total_opened += 1

        mode_label = "OBS" if self.observation_mode else "LIVE"
        self.log.info(
            f"STRADDLE OPEN [{mode_label}] id={straddle_id} price=${price:.2f} "
            f"size=${self.size_usd} vol_pred={vol_pred}"
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

    def _check_leg_exit(self, leg: StraddleLeg, current_price: float, now: float) -> None:
        """Check and apply exit rules to a single leg."""
        if leg.closed:
            return

        pnl_bps = self._compute_leg_pnl_bps(leg, current_price)

        # Track peak favorable
        if pnl_bps > leg.peak_favorable_bps:
            leg.peak_favorable_bps = pnl_bps

        # Stop loss
        if pnl_bps <= -self.stop_loss_bps:
            leg.exit_reason = "stop_loss"
            leg.pnl_bps = pnl_bps
            leg.exit_price = current_price
            leg.closed = True
            leg.closed_at = now
            return

        # Trail activation
        if not leg.trail_active and pnl_bps >= self.trail_activation_bps:
            leg.trail_active = True
            leg.trail_peak_bps = pnl_bps

        # Trail peak update
        if leg.trail_active and pnl_bps > leg.trail_peak_bps:
            leg.trail_peak_bps = pnl_bps

        # Trail stop
        if leg.trail_active and (leg.trail_peak_bps - pnl_bps) >= self.trail_distance_bps:
            leg.exit_reason = "trail_stop"
            leg.pnl_bps = pnl_bps
            leg.exit_price = current_price
            leg.closed = True
            leg.closed_at = now
            return

        # Timeout
        age = now - leg.opened_at
        if age >= self.max_hold_seconds:
            leg.exit_reason = "timeout"
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

        for straddle in list(self.open_straddles):
            # Check both legs
            self._check_leg_exit(straddle.long_leg, current_price, now)
            self._check_leg_exit(straddle.short_leg, current_price, now)

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
                        leg_closed_ts = datetime.fromtimestamp(leg.closed_at, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if leg.closed_at else ts_closed
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
                    self.log.error(f"Straddle DB update error: {e}")

                mode_label = "OBS" if self.observation_mode else "LIVE"
                self.log.info(
                    f"STRADDLE CLOSED [{mode_label}] id={straddle.straddle_id} "
                    f"net={net_pnl_bps:+.1f}bp ${net_pnl_usd:+.4f} "
                    f"L={straddle.long_leg.exit_reason}/{straddle.long_leg.pnl_bps:+.1f}bp "
                    f"S={straddle.short_leg.exit_reason}/{straddle.short_leg.pnl_bps:+.1f}bp "
                    f"dur={duration:.0f}s"
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
                      Expected key is self.pair (e.g. 'BTCUSDT').
        """
        if self._running:
            return
        self._running = True
        self._exit_task = asyncio.ensure_future(self._exit_loop(price_fn))
        self.log.info(
            f"StraddleEngine exit loop started (interval={self.exit_check_interval}s)"
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
        self.log.info("StraddleEngine exit loop stopped")

    async def _exit_loop(self, price_fn: Callable) -> None:
        """Internal loop: fetch price + check exits every interval."""
        while self._running:
            try:
                if self.open_straddles:
                    prices = await price_fn()
                    price = prices.get(self.pair)
                    if price and price > 0:
                        self.check_exits(float(price))
            except Exception as e:
                self.log.warning(f"Straddle exit loop error: {e}")
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
                'entry_price': s.entry_price,
                'vol_prediction': s.vol_prediction,
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
            'pair': self.pair,
            'size_usd': self.size_usd,
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
            'dead_zone_blocks': self._dead_zone_blocks,
            'config': {
                'stop_loss_bps': self.stop_loss_bps,
                'trail_activation_bps': self.trail_activation_bps,
                'trail_distance_bps': self.trail_distance_bps,
                'max_hold_seconds': self.max_hold_seconds,
                'dead_zone_bps': self.dead_zone_bps,
                'use_vol_model': self.use_vol_model,
            },
        }
