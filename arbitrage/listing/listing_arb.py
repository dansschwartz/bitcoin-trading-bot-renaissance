"""
Listing Arbitrage — exploits price gaps at new MEXC token listings.

This module receives ListingEvent callbacks from ListingMonitor and evaluates
whether to trade the listing gap.

OBSERVATION MODE ONLY. Position caps are hard-coded and cannot be overridden.

Risk profile: HIGH (new tokens can drop 90%+)
Max position: $200 per listing event (NON-NEGOTIABLE)
Max concurrent listing positions: 2
"""
import asyncio
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional

from .listing_monitor import ListingEvent

logger = logging.getLogger("arb.listing")

_DB_PATH = str(Path("data") / "arbitrage.db")

# ─── HARD LIMITS — DO NOT CHANGE THESE ───────────────────────────────────────

# ABSOLUTE maximum USD risk per listing event.
# New tokens are extremely volatile. This is not conservative — it's rational.
ABSOLUTE_MAX_POSITION_USD = Decimal('200')

# Maximum simultaneous open listing positions.
# Two is already aggressive for this risk profile.
ABSOLUTE_MAX_CONCURRENT = 2

# Maximum hold time before force-closing regardless of P&L.
# Listing arb is a short-duration strategy. Don't bag-hold new tokens.
ABSOLUTE_MAX_HOLD_MINUTES = 60

# Minimum price on MEXC before considering (filters tokens with near-zero liquidity)
MIN_TOKEN_PRICE_USDT = 0.000001

# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ListingPosition:
    position_id: str
    symbol: str
    base_currency: str
    entry_price: Decimal
    quantity: Decimal
    position_usd: Decimal
    entry_time: datetime
    max_hold_until: datetime
    is_open: bool = True
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    realized_pnl_usd: Optional[Decimal] = None
    exit_reason: Optional[str] = None


class ListingArbitrage:
    """
    Evaluates and (in live mode) trades new MEXC listing events.

    Strategy:
    1. ListingMonitor fires on_new_listing callback with ListingEvent
    2. Check if the token meets minimum criteria
    3. In observation_mode: log what we WOULD do, don't act
    4. In live mode: buy immediately on MEXC (price discovery happening)
       Target: sell within 5-30 minutes as price discovery stabilizes

    Exit conditions (all checked every 60 seconds):
    - Price has risen 10%+: take profit
    - Price has dropped 8%: stop loss
    - 60 minutes elapsed: force close regardless
    """

    TAKE_PROFIT_PCT = Decimal('0.10')
    STOP_LOSS_PCT = Decimal('0.08')
    MIN_VOLUME_USD_5MIN = 1000
    MONITOR_INTERVAL_SECONDS = 60

    def __init__(
        self,
        mexc_spot_client,
        observation_mode: bool = True,
        config: Optional[dict] = None,
    ):
        self.spot = mexc_spot_client
        self.observation_mode = observation_mode

        cfg = (config or {}).get("listing_arbitrage", {})
        self.observation_mode = cfg.get("observation_mode", observation_mode)

        self._positions: Dict[str, ListingPosition] = {}
        self._running = False

        # Stats
        self._listings_evaluated = 0
        self._trades_opened = 0
        self._trades_closed = 0
        self._total_pnl_usd = Decimal('0')

        self._ensure_db_tables()
        logger.info(
            f"ListingArbitrage initialized | observation_mode={self.observation_mode} | "
            f"max_position=${ABSOLUTE_MAX_POSITION_USD} | max_concurrent={ABSOLUTE_MAX_CONCURRENT}"
        )

    def _ensure_db_tables(self):
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS listing_arb_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    mexc_price REAL,
                    volume_5min_usd REAL,
                    position_size_usd REAL,
                    action TEXT,
                    skip_reason TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS listing_arb_positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    base_currency TEXT,
                    entry_price REAL,
                    quantity REAL,
                    position_usd REAL,
                    entry_time TEXT,
                    max_hold_until TEXT,
                    is_open INTEGER DEFAULT 1,
                    exit_price REAL,
                    exit_time TEXT,
                    realized_pnl_usd REAL,
                    exit_reason TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    async def on_new_listing(self, event: ListingEvent):
        """
        Called by ListingMonitor when a new MEXC-first listing is detected.
        This is the entry point for evaluation.
        """
        self._listings_evaluated += 1
        symbol = event.symbol

        logger.info(f"LISTING EVENT: evaluating {symbol} | price={event.mexc_initial_price}")

        # Reject if already at max concurrent positions
        open_count = sum(1 for p in self._positions.values() if p.is_open)
        if open_count >= ABSOLUTE_MAX_CONCURRENT:
            await self._log_evaluation(symbol, event.mexc_initial_price, None, "skipped",
                                       f"at max concurrent positions ({ABSOLUTE_MAX_CONCURRENT})")
            return

        # Reject if already in this symbol
        if symbol in self._positions and self._positions[symbol].is_open:
            await self._log_evaluation(symbol, event.mexc_initial_price, None, "skipped",
                                       "already have open position in this symbol")
            return

        # Retry price fetch if not available (new listings sometimes have None price)
        if not event.mexc_initial_price:
            try:
                ticker = await self.spot.get_ticker(symbol)
                if ticker:
                    price = ticker.get('last_price') or ticker.get('lastPrice')
                    if price:
                        event.mexc_initial_price = float(price)
                        logger.info(f"LISTING: retried price for {symbol}: ${event.mexc_initial_price}")
            except Exception as e:
                logger.debug(f"LISTING: price retry failed for {symbol}: {e}")

        # Validate minimum price
        if not event.mexc_initial_price or event.mexc_initial_price < MIN_TOKEN_PRICE_USDT:
            await self._log_evaluation(symbol, event.mexc_initial_price, None, "skipped",
                                       f"price too low: {event.mexc_initial_price}")
            return

        # Check volume (confirm there's actual trading happening)
        volume_usd = await self._estimate_recent_volume(symbol)
        if volume_usd < self.MIN_VOLUME_USD_5MIN:
            await self._log_evaluation(symbol, event.mexc_initial_price, volume_usd, "skipped",
                                       f"insufficient volume: ${volume_usd:.0f} < ${self.MIN_VOLUME_USD_5MIN}")
            return

        # Passed all checks
        position_usd = ABSOLUTE_MAX_POSITION_USD

        action = "observation_logged" if self.observation_mode else "trade_opened"
        await self._log_evaluation(symbol, event.mexc_initial_price, volume_usd, action, None)

        logger.info(
            f"LISTING OPPORTUNITY [{action.upper()}]: {symbol} | "
            f"price={event.mexc_initial_price} | volume_5min=${volume_usd:.0f} | "
            f"position_size=${position_usd}"
        )

        if self.observation_mode:
            return  # STOP HERE

        await self._open_position(event, position_usd)

    async def _estimate_recent_volume(self, symbol: str) -> float:
        """Estimate 5-minute trading volume in USD."""
        try:
            ticker = await self.spot.get_ticker(symbol)
            # Most exchanges provide 24h volume — approximate 5min as 24h/288
            volume_24h = float(ticker.get('quote_volume_24h', 0))
            return volume_24h / 288
        except Exception:
            return 0.0

    async def _log_evaluation(self, symbol, price, volume, action, skip_reason):
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        try:
            conn.execute(
                """INSERT INTO listing_arb_evaluations
                   (symbol, mexc_price, volume_5min_usd, position_size_usd, action, skip_reason, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol, price, volume,
                    float(ABSOLUTE_MAX_POSITION_USD) if action != "skipped" else None,
                    action, skip_reason,
                    datetime.utcnow().isoformat(),
                )
            )
            conn.commit()
        finally:
            conn.close()

    async def _open_position(self, event: ListingEvent, position_usd: Decimal):
        """Open a long position on the newly listed token. TAKER order for speed."""
        try:
            price = Decimal(str(event.mexc_initial_price))
            quantity = position_usd / price

            result = await self.spot.place_order(
                symbol=event.symbol,
                side="BUY",
                quantity=quantity,
                order_type="MARKET",
            )

            if isinstance(result, Exception):
                logger.error(f"Failed to open listing position for {event.symbol}: {result}")
                return

            position_id = str(uuid.uuid4())[:8]
            entry_time = datetime.utcnow()
            max_hold = entry_time + timedelta(minutes=ABSOLUTE_MAX_HOLD_MINUTES)

            position = ListingPosition(
                position_id=position_id,
                symbol=event.symbol,
                base_currency=event.base_currency,
                entry_price=price,
                quantity=quantity,
                position_usd=position_usd,
                entry_time=entry_time,
                max_hold_until=max_hold,
            )
            self._positions[event.symbol] = position
            self._trades_opened += 1

            conn = sqlite3.connect(_DB_PATH, timeout=10)
            try:
                conn.execute(
                    """INSERT INTO listing_arb_positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        position_id, event.symbol, event.base_currency,
                        float(price), float(quantity), float(position_usd),
                        entry_time.isoformat(), max_hold.isoformat(),
                        1, None, None, None, None,
                    )
                )
                conn.commit()
            finally:
                conn.close()

            logger.info(
                f"LISTING POSITION OPENED: {event.symbol} | id={position_id} | "
                f"qty={float(quantity):.4f} @ ${float(price):.6f} | "
                f"max_hold_until={max_hold.isoformat()}"
            )

        except Exception as e:
            logger.error(f"Error opening listing position for {event.symbol}: {e}", exc_info=True)

    async def run(self):
        """Background loop monitoring open positions."""
        self._running = True
        logger.info("ListingArbitrage position monitor started")
        while self._running:
            try:
                await self._monitor_positions()
            except Exception as e:
                logger.error(f"ListingArbitrage monitor error: {e}", exc_info=True)
            await asyncio.sleep(self.MONITOR_INTERVAL_SECONDS)

    def stop(self):
        self._running = False

    async def _monitor_positions(self):
        """Check all open listing positions for exit conditions."""
        for symbol, position in list(self._positions.items()):
            if not position.is_open:
                continue

            # Force close if max hold time exceeded
            now = datetime.utcnow()
            if now >= position.max_hold_until:
                await self._close_position(position, "MAX_HOLD_TIME_EXCEEDED")
                continue

            # Check current price
            try:
                ticker = await self.spot.get_ticker(symbol)
                current_price = Decimal(str(ticker['last_price']))
            except Exception:
                continue

            pnl_pct = (current_price - position.entry_price) / position.entry_price

            # Take profit
            if pnl_pct >= self.TAKE_PROFIT_PCT:
                await self._close_position(
                    position,
                    f"TAKE_PROFIT: {float(pnl_pct)*100:.1f}% gain",
                    current_price
                )

            # Stop loss
            elif pnl_pct <= -self.STOP_LOSS_PCT:
                await self._close_position(
                    position,
                    f"STOP_LOSS: {float(pnl_pct)*100:.1f}% loss",
                    current_price
                )

    async def _close_position(
        self,
        position: ListingPosition,
        reason: str,
        current_price: Optional[Decimal] = None,
    ):
        """Sell the full listing position. Market order for speed."""
        if not position.is_open:
            return

        if self.observation_mode:
            logger.info(f"[OBSERVATION] Would close {position.symbol}: {reason}")
            return

        try:
            await self.spot.place_order(
                symbol=position.symbol,
                side="SELL",
                quantity=position.quantity,
                order_type="MARKET",
            )

            exit_price = current_price or position.entry_price
            pnl = (exit_price - position.entry_price) * position.quantity

            position.is_open = False
            position.exit_price = exit_price
            position.exit_time = datetime.utcnow()
            position.realized_pnl_usd = pnl
            position.exit_reason = reason
            self._total_pnl_usd += pnl
            self._trades_closed += 1

            conn = sqlite3.connect(_DB_PATH, timeout=10)
            try:
                conn.execute(
                    """UPDATE listing_arb_positions SET
                       is_open=0, exit_price=?, exit_time=?, realized_pnl_usd=?, exit_reason=?
                       WHERE position_id=?""",
                    (
                        float(exit_price), position.exit_time.isoformat(),
                        float(pnl), reason, position.position_id
                    )
                )
                conn.commit()
            finally:
                conn.close()

            logger.info(
                f"LISTING POSITION CLOSED: {position.symbol} | "
                f"pnl=${float(pnl):.4f} | reason={reason}"
            )

        except Exception as e:
            logger.error(f"Error closing listing position {position.symbol}: {e}", exc_info=True)

    def get_status(self) -> dict:
        """For dashboard endpoint GET /api/arbitrage/listing"""
        open_positions = [
            {
                "symbol": p.symbol,
                "entry_price": float(p.entry_price),
                "position_usd": float(p.position_usd),
                "entry_time": p.entry_time.isoformat(),
                "max_hold_until": p.max_hold_until.isoformat(),
            }
            for p in self._positions.values()
            if p.is_open
        ]
        return {
            "observation_mode": self.observation_mode,
            "listings_evaluated": self._listings_evaluated,
            "trades_opened": self._trades_opened,
            "trades_closed": self._trades_closed,
            "total_pnl_usd": float(self._total_pnl_usd),
            "open_positions": open_positions,
            "hard_limits": {
                "max_position_usd": float(ABSOLUTE_MAX_POSITION_USD),
                "max_concurrent": ABSOLUTE_MAX_CONCURRENT,
                "max_hold_minutes": ABSOLUTE_MAX_HOLD_MINUTES,
            }
        }
