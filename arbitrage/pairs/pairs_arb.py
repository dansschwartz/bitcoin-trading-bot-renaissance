"""
Statistical Pairs Arbitrage — wraps AdvancedMeanReversionEngine with execution.

Lifecycle:
1. Every cycle: feed current prices into AdvancedMeanReversionEngine
2. Engine discovers cointegrated pairs (tests periodically, caches otherwise)
3. Engine returns PairState with z_score per pair
4. If z_score crosses entry threshold: log opportunity (observation) or trade (live)
5. Monitor open positions: close at exit_z or stop_loss_z

observation_mode=True always. DO NOT change to False in this implementation.
"""
import asyncio
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import the existing engine from project root
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from analysis.advanced_mean_reversion_engine import AdvancedMeanReversionEngine, PairState

logger = logging.getLogger("arb.pairs")

_DB_PATH = str(Path("data") / "arbitrage.db")

# Candidate pairs — high correlation, DeFi relationships
CANDIDATE_PAIRS: List[Tuple[str, str]] = [
    ("ETH/USDT", "BTC/USDT"),
    ("LINK/USDT", "ETH/USDT"),
    ("AVAX/USDT", "ETH/USDT"),
    ("SOL/USDT", "ETH/USDT"),
    ("LINK/USDT", "BTC/USDT"),
    ("AVAX/USDT", "BTC/USDT"),
    ("DOGE/USDT", "BTC/USDT"),
]

ALL_SYMBOLS = list({s for pair in CANDIDATE_PAIRS for s in pair})


@dataclass
class PairsPosition:
    position_id: str
    base_symbol: str
    quote_symbol: str
    direction: str              # "short_base_long_quote" | "long_base_short_quote"
    base_quantity: Decimal
    quote_quantity: Decimal
    hedge_ratio: float
    entry_z_score: float
    entry_base_price: Decimal
    entry_quote_price: Decimal
    entry_time: datetime
    is_open: bool = True
    exit_z_score: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl_usd: Optional[Decimal] = None


class PairsArbitrage:
    """
    Wraps AdvancedMeanReversionEngine with trade execution logic.
    Starts in observation_mode — all thresholds and logic run,
    but no orders are placed.
    """

    ENTRY_Z = 2.0
    EXIT_Z = 0.5
    STOP_Z = 3.5
    MAX_POSITION_USD = Decimal('1500')
    MAX_CONCURRENT_POSITIONS = 3
    SCAN_INTERVAL_SECONDS = 300  # Every 5 min

    def __init__(
        self,
        mexc_spot_client,
        observation_mode: bool = True,
        config: Optional[dict] = None,
    ):
        self.spot = mexc_spot_client
        self.observation_mode = observation_mode

        cfg = (config or {}).get("statistical_pairs", {})
        self.observation_mode = cfg.get("observation_mode", observation_mode)
        self.ENTRY_Z = cfg.get("entry_z_score", self.ENTRY_Z)
        self.EXIT_Z = cfg.get("exit_z_score", self.EXIT_Z)
        self.STOP_Z = cfg.get("stop_z_score", self.STOP_Z)
        self.SCAN_INTERVAL_SECONDS = cfg.get("scan_interval_seconds", self.SCAN_INTERVAL_SECONDS)

        # The existing engine — reuse as-is
        self.engine = AdvancedMeanReversionEngine(config={
            "window_size": cfg.get("window_size_bars", 240),
            "min_history": cfg.get("min_history_bars", 60),
            "coint_pvalue": cfg.get("cointegration_pvalue", 0.05),
            "retest_interval": cfg.get("retest_interval_cycles", 288),
            "entry_z": self.ENTRY_Z,
            "exit_z": self.EXIT_Z,
            "max_half_life": cfg.get("max_half_life_bars", 120),
            "min_half_life": cfg.get("min_half_life_bars", 5),
            "max_pairs": 5,
        })

        self._positions: Dict[str, PairsPosition] = {}
        self._latest_signals: Dict[str, PairState] = {}
        self._cycle_count = 0
        self._running = False

        # Stats
        self._opportunities_detected = 0
        self._trades_opened = 0
        self._trades_closed = 0
        self._total_pnl_usd = Decimal('0')

        self._ensure_db_tables()
        logger.info(
            f"PairsArbitrage initialized | observation_mode={self.observation_mode} | "
            f"entry_z={self.ENTRY_Z} | exit_z={self.EXIT_Z} | stop_z={self.STOP_Z}"
        )

    def _ensure_db_tables(self):
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pairs_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    base_symbol TEXT NOT NULL,
                    quote_symbol TEXT NOT NULL,
                    z_score REAL,
                    hedge_ratio REAL,
                    half_life REAL,
                    is_cointegrated INTEGER,
                    adf_pvalue REAL,
                    signal REAL,
                    action_taken TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pairs_positions (
                    position_id TEXT PRIMARY KEY,
                    base_symbol TEXT NOT NULL,
                    quote_symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    base_quantity REAL,
                    quote_quantity REAL,
                    hedge_ratio REAL,
                    entry_z_score REAL,
                    entry_base_price REAL,
                    entry_quote_price REAL,
                    entry_time TEXT,
                    is_open INTEGER DEFAULT 1,
                    exit_z_score REAL,
                    exit_time TEXT,
                    realized_pnl_usd REAL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    async def run(self):
        """Main loop. Feed prices, run engine, evaluate signals, manage positions."""
        self._running = True
        logger.info("PairsArbitrage started")

        # Run initial cointegration discovery after accumulating some prices
        await asyncio.sleep(10)

        while self._running:
            try:
                await self._run_cycle()
            except Exception as e:
                logger.error(f"PairsArbitrage cycle error: {e}", exc_info=True)
            await asyncio.sleep(self.SCAN_INTERVAL_SECONDS)

    def stop(self):
        self._running = False

    async def _run_cycle(self):
        """Single scan cycle: fetch prices -> update engine -> evaluate -> act."""
        self._cycle_count += 1
        logger.info(f"PAIRS CYCLE #{self._cycle_count} starting")

        # 1. Fetch current prices for all required symbols
        prices = await self._fetch_prices()
        if not prices:
            logger.warning(f"PAIRS CYCLE #{self._cycle_count}: no prices fetched")
            return

        # 2. Feed prices into engine
        for symbol, price in prices.items():
            self.engine.update_price(symbol, price)

        # 3. Run discovery periodically
        if self._cycle_count % 288 == 1 or not self.engine._active_pairs:
            self.engine.discover_pairs(ALL_SYMBOLS, self._cycle_count)

        # 4. Get signals for all candidate pairs
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        try:
            for base, quote in CANDIDATE_PAIRS:
                if base not in prices or quote not in prices:
                    continue

                pair_state = self.engine.calculate_pair_signal(base, quote)
                pair_key = f"{base}_{quote}"
                self._latest_signals[pair_key] = pair_state

                # Only log signals where we have enough data
                if pair_state.half_life == float("inf") and pair_state.z_score == 0.0:
                    continue

                # Persist signal to DB
                conn.execute(
                    """INSERT INTO pairs_signals
                       (base_symbol, quote_symbol, z_score, hedge_ratio, half_life,
                        is_cointegrated, adf_pvalue, signal, action_taken, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        base, quote,
                        pair_state.z_score,
                        pair_state.hedge_ratio,
                        pair_state.half_life if pair_state.half_life != float("inf") else None,
                        1 if pair_state.is_cointegrated else 0,
                        pair_state.adf_pvalue,
                        pair_state.signal,
                        "logged",
                        datetime.utcnow().isoformat(),
                    )
                )

                # 5. Evaluate for entry
                if pair_key not in self._positions or not self._positions[pair_key].is_open:
                    if pair_state.is_cointegrated:
                        self._evaluate_entry(base, quote, pair_state, prices, conn)

            # 6. Monitor open positions for exit
            self._monitor_positions(prices, conn)

            conn.commit()
        finally:
            conn.close()

    async def _fetch_prices(self) -> Dict[str, float]:
        """Fetch current prices via MEXC REST API in thread pool (avoids event loop blocking)."""
        import json
        import urllib.request
        prices = {}
        try:
            def _do_fetch():
                req = urllib.request.Request(
                    "https://api.mexc.com/api/v3/ticker/price",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read())

            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, _do_fetch)

            # Build lookup: BTCUSDT -> price
            price_map = {item["symbol"]: float(item["price"]) for item in data}

            for symbol in ALL_SYMBOLS:
                mexc_sym = symbol.replace("/", "")
                if mexc_sym in price_map:
                    prices[symbol] = price_map[mexc_sym]

            logger.info(f"PAIRS prices fetched: {len(prices)}/{len(ALL_SYMBOLS)} symbols")
        except Exception as e:
            logger.warning(f"PAIRS price fetch error: {type(e).__name__}: {e}", exc_info=True)
        return prices

    def _evaluate_entry(
        self,
        base: str,
        quote: str,
        state: PairState,
        prices: Dict[str, float],
        conn,
    ):
        """Evaluate whether to enter a pairs trade."""
        abs_z = abs(state.z_score)

        if abs_z < self.ENTRY_Z:
            return

        if state.half_life == float("inf") or state.half_life > 120 or state.half_life < 5:
            return

        open_count = sum(1 for p in self._positions.values() if p.is_open)
        if open_count >= self.MAX_CONCURRENT_POSITIONS:
            return

        self._opportunities_detected += 1

        # Direction: if z_score > 0, base is expensive relative to quote
        if state.z_score > 0:
            direction = "short_base_long_quote"
        else:
            direction = "long_base_short_quote"

        logger.info(
            f"PAIRS SIGNAL [{('OBSERVATION' if self.observation_mode else 'TRADING')}]: "
            f"{base}/{quote} | z={state.z_score:.2f} | half_life={state.half_life:.1f} bars | "
            f"hedge={state.hedge_ratio:.4f} | direction={direction}"
        )

        # Update action on last signal row
        conn.execute(
            """UPDATE pairs_signals SET action_taken = ?
               WHERE id = (SELECT MAX(id) FROM pairs_signals WHERE base_symbol=? AND quote_symbol=?)""",
            (
                "observation_logged" if self.observation_mode else "trade_opened",
                base, quote,
            )
        )

        if self.observation_mode:
            return  # STOP HERE

        # Live trading would go here but we never enter this branch

    def _monitor_positions(self, prices: Dict[str, float], conn):
        """Check all open positions for exit conditions."""
        for pair_key, position in list(self._positions.items()):
            if not position.is_open:
                continue

            if position.base_symbol not in prices or position.quote_symbol not in prices:
                continue

            current_state = self._latest_signals.get(pair_key)
            if current_state is None:
                continue

            current_z = current_state.z_score
            should_close = False
            close_reason = ""

            if abs(current_z) <= self.EXIT_Z:
                should_close = True
                close_reason = f"SPREAD REVERTED: z={current_z:.2f}"

            elif abs(current_z) >= self.STOP_Z:
                should_close = True
                close_reason = f"STOP LOSS: z={current_z:.2f}"

            if should_close:
                if self.observation_mode:
                    logger.info(f"[OBSERVATION] Would close {pair_key}: {close_reason}")
                else:
                    logger.info(f"CLOSING PAIRS POSITION {pair_key}: {close_reason}")

    def get_status(self) -> dict:
        """For dashboard endpoint GET /api/arbitrage/pairs"""
        active_pairs = []
        for (base, quote) in CANDIDATE_PAIRS:
            pair_key = f"{base}_{quote}"
            state = self._latest_signals.get(pair_key)
            active_pairs.append({
                "base": base,
                "quote": quote,
                "z_score": round(state.z_score, 3) if state else None,
                "half_life_bars": (
                    round(state.half_life, 1)
                    if state and state.half_life != float("inf")
                    else None
                ),
                "is_cointegrated": state.is_cointegrated if state else False,
                "adf_pvalue": round(state.adf_pvalue, 4) if state else None,
                "signal": round(state.signal, 3) if state else None,
                "has_open_position": (
                    pair_key in self._positions and self._positions[pair_key].is_open
                ),
            })

        open_positions = [
            {
                "pair": f"{p.base_symbol}/{p.quote_symbol}",
                "direction": p.direction,
                "entry_z": p.entry_z_score,
                "entry_time": p.entry_time.isoformat(),
            }
            for p in self._positions.values()
            if p.is_open
        ]

        return {
            "observation_mode": self.observation_mode,
            "cycle_count": self._cycle_count,
            "opportunities_detected": self._opportunities_detected,
            "trades_opened": self._trades_opened,
            "trades_closed": self._trades_closed,
            "total_pnl_usd": float(self._total_pnl_usd),
            "open_positions": open_positions,
            "pair_states": active_pairs,
            "config": {
                "entry_z": self.ENTRY_Z,
                "exit_z": self.EXIT_Z,
                "stop_z": self.STOP_Z,
                "max_position_usd": float(self.MAX_POSITION_USD),
                "max_concurrent": self.MAX_CONCURRENT_POSITIONS,
                "scan_interval_seconds": self.SCAN_INTERVAL_SECONDS,
            }
        }
