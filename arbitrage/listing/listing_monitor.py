"""
Monitors MEXC for new token listings in real time.

Detection method:
- Polls MEXC's /api/v3/exchangeInfo every 60 seconds
- Compares current symbol list against known symbols (cached in DB)
- Any NEW symbol that appears = potential listing arbitrage opportunity
- Cross-references against Binance to confirm it's NOT already there
  (if it's on Binance too, it's not a MEXC-first listing)

Emits: ListingEvent whenever a new symbol appears on MEXC for the first time.
"""
import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Set

import aiohttp

logger = logging.getLogger("arb.listing.monitor")

_DB_PATH = str(Path("data") / "arbitrage.db")


@dataclass
class ListingEvent:
    symbol: str                     # e.g. "NEWTOKEN/USDT"
    base_currency: str              # e.g. "NEWTOKEN"
    quote_currency: str             # e.g. "USDT"
    detected_at: datetime
    on_mexc: bool = True
    on_binance: bool = False        # True = not a MEXC-first listing
    on_okx: bool = False
    mexc_initial_price: Optional[float] = None
    is_first_listing: bool = True   # False if already on major exchange


class ListingMonitor:
    """
    Polls MEXC every 60 seconds for new symbols.
    Calls on_new_listing callback when a new MEXC-first listing is detected.
    """

    POLL_INTERVAL_SECONDS = 60

    # Symbols to always ignore (stablecoins, wrapped, always there)
    IGNORE_PREFIXES = [
        "USDT", "USDC", "BUSD", "DAI", "TUSD",  # Stablecoins
        "WBTC", "WETH",                            # Wrapped
        "BTC", "ETH", "BNB",                       # Always listed everywhere
    ]

    def __init__(
        self,
        mexc_client,
        on_new_listing: Optional[Callable] = None,
        config: Optional[dict] = None,
    ):
        self.mexc = mexc_client
        self.on_new_listing = on_new_listing

        cfg = (config or {}).get("listing_arbitrage", {})
        self.poll_interval = cfg.get("poll_interval_seconds", self.POLL_INTERVAL_SECONDS)

        self._known_mexc_symbols: Set[str] = set()
        self._known_binance_symbols: Set[str] = set()
        self._running = False
        self._initialized = False

        # Stats
        self._polls_completed = 0
        self._listings_detected = 0
        self._first_listings_detected = 0

        self._ensure_db_tables()

    def _ensure_db_tables(self):
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS listing_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    base_currency TEXT,
                    quote_currency TEXT,
                    detected_at TEXT NOT NULL,
                    on_mexc INTEGER DEFAULT 1,
                    on_binance INTEGER DEFAULT 0,
                    mexc_initial_price REAL,
                    is_first_listing INTEGER DEFAULT 1,
                    notes TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS known_symbols_snapshot (
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    PRIMARY KEY (exchange, symbol)
                )
            """)
            conn.commit()

            # Load known symbols from DB on startup
            rows = conn.execute(
                "SELECT symbol FROM known_symbols_snapshot WHERE exchange='mexc'"
            ).fetchall()
            self._known_mexc_symbols = {row[0] for row in rows}

            rows = conn.execute(
                "SELECT symbol FROM known_symbols_snapshot WHERE exchange='binance'"
            ).fetchall()
            self._known_binance_symbols = {row[0] for row in rows}

            logger.info(
                f"ListingMonitor loaded: {len(self._known_mexc_symbols)} MEXC symbols, "
                f"{len(self._known_binance_symbols)} Binance symbols from DB"
            )
        finally:
            conn.close()

    async def run(self):
        self._running = True
        logger.info("ListingMonitor started")

        # On first run, populate known symbols without triggering alerts
        await self._initialize_symbol_sets()
        self._initialized = True

        while self._running:
            try:
                await self._poll_for_new_listings()
                self._polls_completed += 1
            except Exception as e:
                logger.error(f"ListingMonitor poll error: {e}", exc_info=True)
            await asyncio.sleep(self.poll_interval)

    def stop(self):
        self._running = False

    async def _initialize_symbol_sets(self):
        """
        On first startup: populate known symbols without triggering callbacks.
        This prevents false alarms from symbols that existed before the bot started.
        """
        logger.info("ListingMonitor: initializing symbol sets (no alerts this pass)")
        mexc_symbols = await self._fetch_mexc_symbols()
        binance_symbols = await self._fetch_binance_symbols()

        new_mexc = mexc_symbols - self._known_mexc_symbols
        new_binance = binance_symbols - self._known_binance_symbols

        conn = sqlite3.connect(_DB_PATH, timeout=10)
        try:
            if new_mexc:
                logger.info(f"Initializing {len(new_mexc)} new MEXC symbols into known set")
                self._known_mexc_symbols.update(new_mexc)
                for sym in new_mexc:
                    conn.execute(
                        "INSERT OR IGNORE INTO known_symbols_snapshot VALUES (?, ?, ?)",
                        ("mexc", sym, datetime.utcnow().isoformat())
                    )

            if new_binance:
                logger.info(f"Initializing {len(new_binance)} Binance symbols into known set")
                self._known_binance_symbols.update(new_binance)
                for sym in new_binance:
                    conn.execute(
                        "INSERT OR IGNORE INTO known_symbols_snapshot VALUES (?, ?, ?)",
                        ("binance", sym, datetime.utcnow().isoformat())
                    )

            conn.commit()
        finally:
            conn.close()

        logger.info(
            f"Symbol sets initialized: {len(self._known_mexc_symbols)} MEXC, "
            f"{len(self._known_binance_symbols)} Binance"
        )

    async def _poll_for_new_listings(self):
        """Check for new symbols since last poll."""
        mexc_symbols = await self._fetch_mexc_symbols()
        binance_symbols = await self._fetch_binance_symbols()

        conn = sqlite3.connect(_DB_PATH, timeout=10)
        try:
            # Update Binance known set (silently)
            new_binance = binance_symbols - self._known_binance_symbols
            if new_binance:
                self._known_binance_symbols.update(new_binance)
                for sym in new_binance:
                    conn.execute(
                        "INSERT OR IGNORE INTO known_symbols_snapshot VALUES (?, ?, ?)",
                        ("binance", sym, datetime.utcnow().isoformat())
                    )

            # Check for NEW MEXC symbols
            new_mexc = mexc_symbols - self._known_mexc_symbols

            for symbol in new_mexc:
                # Filter out non-USDT pairs
                if not symbol.endswith("/USDT"):
                    continue

                base = symbol.replace("/USDT", "")

                # Skip stablecoins and known majors
                if any(base.startswith(p) for p in self.IGNORE_PREFIXES):
                    continue

                # Check if it's already on Binance (not a MEXC-first listing)
                binance_symbol = base + "USDT"
                is_first_listing = binance_symbol not in binance_symbols

                # Fetch initial price on MEXC
                initial_price = None
                try:
                    ticker = await self.mexc.get_ticker(symbol)
                    initial_price = float(ticker.get('last_price', 0))
                except Exception:
                    pass

                event = ListingEvent(
                    symbol=symbol,
                    base_currency=base,
                    quote_currency="USDT",
                    detected_at=datetime.utcnow(),
                    on_mexc=True,
                    on_binance=not is_first_listing,
                    mexc_initial_price=initial_price,
                    is_first_listing=is_first_listing,
                )

                # Persist to DB
                conn.execute(
                    """INSERT INTO listing_events
                       (symbol, base_currency, quote_currency, detected_at,
                        on_mexc, on_binance, mexc_initial_price, is_first_listing)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        symbol, base, "USDT", event.detected_at.isoformat(),
                        1, 0 if is_first_listing else 1,
                        initial_price, 1 if is_first_listing else 0,
                    )
                )

                self._listings_detected += 1
                if is_first_listing:
                    self._first_listings_detected += 1

                logger.info(
                    f"NEW MEXC LISTING DETECTED: {symbol} | "
                    f"price={initial_price} | first_on_mexc={is_first_listing}"
                )

                # Fire callback
                if self.on_new_listing and is_first_listing:
                    try:
                        await self.on_new_listing(event)
                    except Exception as e:
                        logger.error(f"on_new_listing callback error: {e}")

            # Update known set
            self._known_mexc_symbols.update(new_mexc)
            for sym in new_mexc:
                conn.execute(
                    "INSERT OR IGNORE INTO known_symbols_snapshot VALUES (?, ?, ?)",
                    ("mexc", sym, datetime.utcnow().isoformat())
                )
            conn.commit()
        finally:
            conn.close()

    async def _fetch_mexc_symbols(self) -> Set[str]:
        """Fetch all currently tradeable MEXC spot symbols via REST API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.mexc.com/api/v3/defaultSymbols",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        # Fallback: try exchangeInfo
                        raise Exception(f"defaultSymbols returned {resp.status}")
                    data = await resp.json()
                    # defaultSymbols returns list of symbol strings like "BTCUSDT"
                    symbols = set()
                    if isinstance(data, dict) and 'data' in data:
                        for s in data['data']:
                            # Convert BTCUSDT -> BTC/USDT
                            if s.endswith("USDT"):
                                base = s[:-4]
                                symbols.add(f"{base}/USDT")
                    return symbols if symbols else self._known_mexc_symbols
        except Exception:
            pass

        # Fallback: use MEXC exchangeInfo
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.mexc.com/api/v3/exchangeInfo",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    data = await resp.json()
                    symbols = set()
                    for s in data.get('symbols', []):
                        status = s.get('status', '')
                        if status in ('ENABLED', '1', 'TRADING'):
                            base = s.get('baseAsset', '')
                            quote = s.get('quoteAsset', '')
                            if base and quote:
                                symbols.add(f"{base}/{quote}")
                    return symbols if symbols else self._known_mexc_symbols
        except Exception as e:
            logger.error(f"Failed to fetch MEXC symbols: {e}")
            return self._known_mexc_symbols

    async def _fetch_binance_symbols(self) -> Set[str]:
        """Fetch all Binance spot symbols (in BTCUSDT format for comparison).
        Uses /api/v3/ticker/price (~150KB) instead of /exchangeInfo (~16MB).
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.binance.com/api/v3/ticker/price",
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    data = await resp.json()
                    symbols = set()
                    for s in data:
                        symbols.add(s['symbol'])  # e.g. "BTCUSDT"
                    return symbols
        except Exception as e:
            logger.error(f"Failed to fetch Binance symbols: {type(e).__name__}: {e}")
            return self._known_binance_symbols

    def get_recent_listings(self, limit: int = 20) -> List[dict]:
        """Return recent listing events for dashboard."""
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """SELECT symbol, detected_at, is_first_listing, mexc_initial_price, on_binance
                   FROM listing_events ORDER BY id DESC LIMIT ?""",
                (limit,)
            ).fetchall()
            return [
                {
                    "symbol": r["symbol"],
                    "detected_at": r["detected_at"],
                    "is_first_listing": bool(r["is_first_listing"]),
                    "mexc_initial_price": r["mexc_initial_price"],
                    "already_on_binance": bool(r["on_binance"]),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Return monitoring stats for orchestrator/dashboard."""
        return {
            "running": self._running,
            "initialized": self._initialized,
            "polls_completed": self._polls_completed,
            "known_mexc_symbols": len(self._known_mexc_symbols),
            "known_binance_symbols": len(self._known_binance_symbols),
            "listings_detected": self._listings_detected,
            "first_listings_detected": self._first_listings_detected,
        }
