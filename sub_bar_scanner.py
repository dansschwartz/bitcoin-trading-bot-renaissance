"""
Sub-Bar Scanner - 10-Second Early Exit Monitor
===============================================

Monitors open positions between the normal 5-minute trading cycles for early
exit triggers. This scanner NEVER opens positions — it only flags or closes
positions that meet one of four exit criteria:

1. EDGE_CONSUMED_EARLY  — 80% of expected move already captured
2. STOP_LOSS_EARLY      — loss exceeds stop threshold
3. DIRECTION_REVERSAL   — 30+ bps reversal in 60 seconds
4. VOLATILITY_SPIKE     — 2-min range exceeds 3x expected move

Starts in OBSERVATION mode: logs triggers but does not execute exits.
"""

import asyncio
import logging
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: Dict[str, Any] = {
    'enabled': True,
    'observation_mode': True,           # Log only, don't execute
    'scan_interval_seconds': 10,
    'edge_consumed_pct': 0.80,          # 80% of expected move captured
    'stop_loss_bps': 100,               # 1% stop loss
    'reversal_threshold_bps': 30,       # 30bps reversal triggers exit
    'reversal_window_seconds': 60,
    'vol_spike_multiplier': 3.0,        # 2-min range > 3x expected
    'vol_window_seconds': 120,
    'min_position_age_seconds': 30,     # Don't exit brand new positions
    'max_exits_per_scan': 3,            # Rate limit exits per scan cycle
    'cooldown_after_exit_seconds': 60,  # Don't re-trigger same position
}

# Maximum price history entries per symbol (50 minutes at 10-second intervals)
_MAX_PRICE_HISTORY = 300

# Default expected magnitude in bps when position has no prediction
_DEFAULT_EXPECTED_MAG_BPS = 100.0

# Trigger type constants
TRIGGER_EDGE_CONSUMED = 'EDGE_CONSUMED_EARLY'
TRIGGER_STOP_LOSS = 'STOP_LOSS_EARLY'
TRIGGER_DIRECTION_REVERSAL = 'DIRECTION_REVERSAL'
TRIGGER_VOLATILITY_SPIKE = 'VOLATILITY_SPIKE'

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sub_bar_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    product_id TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    side TEXT,
    entry_price REAL,
    current_price REAL,
    pnl_bps REAL,
    expected_magnitude_bps REAL,
    observation_mode INTEGER DEFAULT 1,
    action_taken TEXT,
    details TEXT
)
"""


class SubBarScanner:
    """10-second async loop monitoring open positions for early exit triggers.

    Four exit triggers:
    1. EDGE_CONSUMED_EARLY  -- 80% of expected move already captured
    2. STOP_LOSS_EARLY      -- loss exceeds stop threshold
    3. DIRECTION_REVERSAL   -- 30+ bps reversal in 60 seconds
    4. VOLATILITY_SPIKE     -- 2-min range exceeds 3x expected move

    Starts in OBSERVATION mode -- logs triggers but doesn't act.
    """

    def __init__(self, config: Dict[str, Any], db_path: str) -> None:
        """Initialize with config and database path.

        Args:
            config: Configuration dict. Keys from _DEFAULT_CONFIG are used;
                    unrecognised keys are ignored. Missing keys fall back to
                    defaults.
            db_path: Absolute path to the SQLite database file.
        """
        # Merge caller config over defaults
        merged = dict(_DEFAULT_CONFIG)
        if config:
            for key in _DEFAULT_CONFIG:
                if key in config:
                    merged[key] = config[key]
        self._cfg = merged

        # Expose frequently-accessed settings as attributes for speed
        self._enabled: bool = bool(self._cfg['enabled'])
        self._observation_mode: bool = bool(self._cfg['observation_mode'])
        self._scan_interval: int = int(self._cfg['scan_interval_seconds'])
        self._edge_consumed_pct: float = float(self._cfg['edge_consumed_pct'])
        self._stop_loss_bps: float = float(self._cfg['stop_loss_bps'])
        self._reversal_threshold_bps: float = float(self._cfg['reversal_threshold_bps'])
        self._reversal_window_seconds: int = int(self._cfg['reversal_window_seconds'])
        self._vol_spike_multiplier: float = float(self._cfg['vol_spike_multiplier'])
        self._vol_window_seconds: int = int(self._cfg['vol_window_seconds'])
        self._min_position_age_seconds: int = int(self._cfg['min_position_age_seconds'])
        self._max_exits_per_scan: int = int(self._cfg['max_exits_per_scan'])
        self._cooldown_seconds: int = int(self._cfg['cooldown_after_exit_seconds'])

        self._db_path: str = db_path

        # Rolling price buffer: symbol -> deque of (unix_timestamp, price)
        self._price_history: Dict[str, Deque[Tuple[float, float]]] = {}

        # Cooldown tracking: product_id -> unix timestamp of last trigger
        self._cooldowns: Dict[str, float] = {}

        # Stats counters
        self._stats: Dict[str, int] = {
            'total_scans': 0,
            'triggers_fired': 0,
            'positions_exited': 0,
            'trigger_counts': {
                TRIGGER_EDGE_CONSUMED: 0,
                TRIGGER_STOP_LOSS: 0,
                TRIGGER_DIRECTION_REVERSAL: 0,
                TRIGGER_VOLATILITY_SPIKE: 0,
            },
        }

        # Async loop control
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._lock: asyncio.Lock = asyncio.Lock()

        # Callbacks — set via start()
        self._position_getter: Optional[Callable] = None
        self._price_getter: Optional[Callable] = None
        self._exit_callback: Optional[Callable] = None

        # Ensure DB table exists
        self._init_db()

        logger.info(
            "SubBarScanner initialised: observation_mode=%s, interval=%ds, "
            "edge_pct=%.0f%%, stop=%dbps, reversal=%dbps/%ds, vol_spike=%.1fx/%ds",
            self._observation_mode,
            self._scan_interval,
            self._edge_consumed_pct * 100,
            self._stop_loss_bps,
            self._reversal_threshold_bps,
            self._reversal_window_seconds,
            self._vol_spike_multiplier,
            self._vol_window_seconds,
        )

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the sub_bar_events table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
            conn.close()
            logger.info("SubBarScanner: sub_bar_events table ready at %s", self._db_path)
        except Exception as exc:
            logger.error("SubBarScanner: failed to init DB table: %s", exc)

    def _record_event(self, event: Dict[str, Any]) -> None:
        """Write a trigger event to the sub_bar_events table."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                INSERT INTO sub_bar_events
                    (timestamp, product_id, trigger_type, side, entry_price,
                     current_price, pnl_bps, expected_magnitude_bps,
                     observation_mode, action_taken, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    event.get('product_id', ''),
                    event.get('trigger', ''),
                    event.get('side', ''),
                    event.get('entry_price'),
                    event.get('current_price'),
                    event.get('pnl_bps'),
                    event.get('expected_magnitude_bps'),
                    1 if self._observation_mode else 0,
                    event.get('action_taken', 'OBSERVED' if self._observation_mode else 'EXIT_REQUESTED'),
                    event.get('details', ''),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error("SubBarScanner: failed to record event: %s", exc)

    # ------------------------------------------------------------------
    # Price history buffer
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float, timestamp: float = None) -> None:
        """Update the rolling price buffer for a symbol.

        Can be called externally by the data feed or internally by _scan_once.

        Args:
            symbol: Trading pair / product id (e.g. 'BTCUSDT').
            price: Current price as a float.
            timestamp: Unix timestamp. Defaults to time.time() if omitted.
        """
        if price <= 0:
            return
        ts = timestamp if timestamp is not None else time.time()
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=_MAX_PRICE_HISTORY)
        self._price_history[symbol].append((ts, price))

    def _get_price_n_seconds_ago(self, symbol: str, seconds: int) -> Optional[float]:
        """Return the price closest to N seconds ago, or None if unavailable."""
        buf = self._price_history.get(symbol)
        if not buf or len(buf) == 0:
            return None

        target_ts = time.time() - seconds
        best_price: Optional[float] = None
        best_delta: float = float('inf')

        for ts, price in buf:
            delta = abs(ts - target_ts)
            if delta < best_delta:
                best_delta = delta
                best_price = price

        # Only return if we have something within 2x the scan interval of
        # the target timestamp (avoids using stale data)
        max_acceptable_delta = max(seconds * 0.5, self._scan_interval * 2)
        if best_delta > max_acceptable_delta:
            return None
        return best_price

    def _get_prices_in_window(self, symbol: str, window_seconds: int) -> List[float]:
        """Return all prices recorded in the last window_seconds."""
        buf = self._price_history.get(symbol)
        if not buf:
            return []
        cutoff = time.time() - window_seconds
        return [price for ts, price in buf if ts >= cutoff]

    # ------------------------------------------------------------------
    # Trigger checks
    # ------------------------------------------------------------------

    def _check_position(self, position: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """Check a single position against all four exit triggers.

        Args:
            position: Dict with keys: product_id, side, entry_price, size_usd,
                      open_timestamp, predicted_magnitude_bps (optional).
            current_price: Latest price for the position's symbol.

        Returns:
            A trigger dict if any trigger fires, or None.
            Trigger dict keys: trigger, product_id, side, entry_price,
            current_price, pnl_bps, expected_magnitude_bps, details.
        """
        try:
            product_id: str = position.get('product_id', '')
            side: str = str(position.get('side', 'BUY')).upper()
            entry_price: float = float(position.get('entry_price', 0))

            if entry_price <= 0 or current_price <= 0:
                return None

            # Compute P&L in basis points
            pnl_bps: float = (current_price - entry_price) / entry_price * 10000.0
            if side == 'SELL':
                pnl_bps = -pnl_bps

            expected_mag: float = float(position.get('predicted_magnitude_bps', _DEFAULT_EXPECTED_MAG_BPS))
            if expected_mag <= 0:
                expected_mag = _DEFAULT_EXPECTED_MAG_BPS

            # Skip positions that are too new
            open_ts = position.get('open_timestamp')
            if open_ts is not None:
                try:
                    age_seconds = time.time() - float(open_ts)
                    if age_seconds < self._min_position_age_seconds:
                        return None
                except (ValueError, TypeError):
                    pass  # If we can't parse the timestamp, proceed with checks

            # Check cooldown
            now = time.time()
            last_trigger_ts = self._cooldowns.get(product_id, 0)
            if (now - last_trigger_ts) < self._cooldown_seconds:
                return None

            # Build base result dict (trigger field filled per-check)
            base = {
                'product_id': product_id,
                'side': side,
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl_bps': round(pnl_bps, 2),
                'expected_magnitude_bps': round(expected_mag, 2),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }

            # ---- Trigger 1: EDGE_CONSUMED_EARLY ----
            if pnl_bps > 0 and pnl_bps >= expected_mag * self._edge_consumed_pct:
                return {
                    **base,
                    'trigger': TRIGGER_EDGE_CONSUMED,
                    'details': (
                        f"P&L {pnl_bps:.1f}bps >= {self._edge_consumed_pct*100:.0f}% "
                        f"of expected {expected_mag:.1f}bps"
                    ),
                }

            # ---- Trigger 2: STOP_LOSS_EARLY ----
            if pnl_bps < -self._stop_loss_bps:
                return {
                    **base,
                    'trigger': TRIGGER_STOP_LOSS,
                    'details': (
                        f"P&L {pnl_bps:.1f}bps exceeds stop loss "
                        f"of -{self._stop_loss_bps:.0f}bps"
                    ),
                }

            # ---- Trigger 3: DIRECTION_REVERSAL ----
            price_ago = self._get_price_n_seconds_ago(product_id, self._reversal_window_seconds)
            if price_ago is not None and price_ago > 0:
                recent_move_bps = (current_price - price_ago) / price_ago * 10000.0
                if side == 'BUY' and recent_move_bps < -self._reversal_threshold_bps:
                    return {
                        **base,
                        'trigger': TRIGGER_DIRECTION_REVERSAL,
                        'details': (
                            f"BUY position saw {recent_move_bps:.1f}bps move in "
                            f"last {self._reversal_window_seconds}s "
                            f"(threshold: -{self._reversal_threshold_bps}bps)"
                        ),
                    }
                elif side == 'SELL' and recent_move_bps > self._reversal_threshold_bps:
                    return {
                        **base,
                        'trigger': TRIGGER_DIRECTION_REVERSAL,
                        'details': (
                            f"SELL position saw +{recent_move_bps:.1f}bps move in "
                            f"last {self._reversal_window_seconds}s "
                            f"(threshold: +{self._reversal_threshold_bps}bps)"
                        ),
                    }

            # ---- Trigger 4: VOLATILITY_SPIKE ----
            prices_window = self._get_prices_in_window(product_id, self._vol_window_seconds)
            if len(prices_window) >= 2:
                min_price = min(prices_window)
                max_price = max(prices_window)
                if min_price > 0:
                    range_bps = (max_price - min_price) / min_price * 10000.0
                    vol_threshold = expected_mag * self._vol_spike_multiplier
                    if range_bps > vol_threshold:
                        return {
                            **base,
                            'trigger': TRIGGER_VOLATILITY_SPIKE,
                            'details': (
                                f"{self._vol_window_seconds}s range {range_bps:.1f}bps > "
                                f"{self._vol_spike_multiplier:.1f}x expected "
                                f"{expected_mag:.1f}bps (threshold: {vol_threshold:.1f}bps)"
                            ),
                        }

            return None

        except Exception as exc:
            logger.error(
                "SubBarScanner: error checking position %s: %s",
                position.get('product_id', '?'), exc,
            )
            return None

    # ------------------------------------------------------------------
    # Scan loop
    # ------------------------------------------------------------------

    async def start(
        self,
        position_getter: Callable,
        price_getter: Callable,
        exit_callback: Callable = None,
    ) -> None:
        """Start the 10-second scan loop.

        Args:
            position_getter: Async callable returning a list of open position
                dicts. Each dict has: product_id, side, entry_price, size_usd,
                open_timestamp, predicted_magnitude_bps (optional).
            price_getter: Async callable(symbol) -> float returning the
                current price for a symbol.
            exit_callback: Optional async callable(product_id, reason, details)
                invoked when a trigger fires and observation_mode is False.
        """
        if not self._enabled:
            logger.info("SubBarScanner: disabled via config, not starting")
            return

        self._position_getter = position_getter
        self._price_getter = price_getter
        self._exit_callback = exit_callback
        self._running = True

        self._task = asyncio.ensure_future(self._run_loop())
        logger.info(
            "SubBarScanner: started (observation_mode=%s, interval=%ds)",
            self._observation_mode, self._scan_interval,
        )

    async def stop(self) -> None:
        """Stop the scan loop gracefully."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info(
            "SubBarScanner: stopped. Stats: %d scans, %d triggers, %d exits",
            self._stats['total_scans'],
            self._stats['triggers_fired'],
            self._stats['positions_exited'],
        )

    async def _run_loop(self) -> None:
        """Main async loop — runs _scan_once every scan_interval_seconds."""
        logger.info("SubBarScanner: scan loop running")
        while self._running:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("SubBarScanner: scan loop error: %s", exc)

            try:
                await asyncio.sleep(self._scan_interval)
            except asyncio.CancelledError:
                break

        logger.info("SubBarScanner: scan loop exited")

    async def _scan_once(self) -> None:
        """Run one scan cycle across all open positions."""
        if self._position_getter is None or self._price_getter is None:
            return

        self._stats['total_scans'] += 1

        # Fetch open positions
        try:
            positions: List[Dict[str, Any]] = await self._position_getter()
        except Exception as exc:
            logger.error("SubBarScanner: position_getter failed: %s", exc)
            return

        if not positions:
            return

        exits_this_scan = 0
        triggered_events: List[Dict[str, Any]] = []

        for position in positions:
            if exits_this_scan >= self._max_exits_per_scan:
                break

            product_id = position.get('product_id', '')
            if not product_id:
                continue

            # Fetch current price
            try:
                current_price = await self._price_getter(product_id)
            except Exception as exc:
                logger.debug(
                    "SubBarScanner: price_getter failed for %s: %s",
                    product_id, exc,
                )
                continue

            if current_price is None or current_price <= 0:
                continue

            # Update price buffer with latest data point
            async with self._lock:
                self.update_price(product_id, current_price)

            # Run trigger checks
            trigger_result = self._check_position(position, current_price)
            if trigger_result is None:
                continue

            # A trigger fired
            trigger_type = trigger_result['trigger']
            self._stats['triggers_fired'] += 1
            trigger_counts = self._stats.get('trigger_counts', {})
            trigger_counts[trigger_type] = trigger_counts.get(trigger_type, 0) + 1

            # Record cooldown
            async with self._lock:
                self._cooldowns[product_id] = time.time()

            # Log it
            mode_label = "OBSERVATION" if self._observation_mode else "LIVE"
            logger.warning(
                "SubBarScanner [%s] %s on %s (%s): pnl=%.1fbps | %s",
                mode_label,
                trigger_type,
                product_id,
                trigger_result.get('side', '?'),
                trigger_result.get('pnl_bps', 0),
                trigger_result.get('details', ''),
            )

            # Persist to DB
            self._record_event(trigger_result)

            triggered_events.append(trigger_result)

            # Execute exit if not in observation mode
            if not self._observation_mode and self._exit_callback is not None:
                try:
                    await self._exit_callback(
                        product_id,
                        trigger_type,
                        trigger_result,
                    )
                    self._stats['positions_exited'] += 1
                    exits_this_scan += 1
                    logger.info(
                        "SubBarScanner: exit_callback invoked for %s (%s)",
                        product_id, trigger_type,
                    )
                except Exception as exc:
                    logger.error(
                        "SubBarScanner: exit_callback failed for %s: %s",
                        product_id, exc,
                    )

        if triggered_events:
            logger.info(
                "SubBarScanner: scan #%d complete — %d trigger(s) across %d position(s)",
                self._stats['total_scans'],
                len(triggered_events),
                len(positions),
            )

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return scanner status dict for dashboard or API consumption."""
        return {
            'enabled': self._enabled,
            'running': self._running,
            'observation_mode': self._observation_mode,
            'scan_interval_seconds': self._scan_interval,
            'total_scans': self._stats['total_scans'],
            'triggers_fired': self._stats['triggers_fired'],
            'positions_exited': self._stats['positions_exited'],
            'trigger_counts': dict(self._stats.get('trigger_counts', {})),
            'symbols_tracked': len(self._price_history),
            'active_cooldowns': sum(
                1 for ts in self._cooldowns.values()
                if (time.time() - ts) < self._cooldown_seconds
            ),
            'config': {
                'edge_consumed_pct': self._edge_consumed_pct,
                'stop_loss_bps': self._stop_loss_bps,
                'reversal_threshold_bps': self._reversal_threshold_bps,
                'reversal_window_seconds': self._reversal_window_seconds,
                'vol_spike_multiplier': self._vol_spike_multiplier,
                'vol_window_seconds': self._vol_window_seconds,
                'min_position_age_seconds': self._min_position_age_seconds,
                'max_exits_per_scan': self._max_exits_per_scan,
                'cooldown_after_exit_seconds': self._cooldown_seconds,
            },
        }
