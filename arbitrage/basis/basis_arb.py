"""
Basis Arbitrage — spot-futures convergence strategy on MEXC.

Monitors the basis (price gap) between MEXC spot and MEXC perpetual
futures contracts. When basis exceeds threshold, logs the opportunity.

Starts in observation_mode=True — no trades, just data collection.

Futures prices fetched directly from MEXC contract REST API:
  https://contract.mexc.com/api/v1/contract/ticker?symbol={SYM}_USDT

Follows the same pattern as FundingRateArbitrage: aiohttp for external
API calls, sqlite3.connect() per-operation for DB writes.
"""
import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

from .basis_calculator import (
    BasisSnapshot,
    BasisOpportunity,
    calculate_basis,
    evaluate_opportunity,
)

logger = logging.getLogger("arb.basis")

MEXC_CONTRACT_API = "https://contract.mexc.com/api/v1/contract/ticker"

# Symbols to monitor (MEXC contract format: BTC_USDT)
BASIS_SYMBOLS = {
    "BTC": "BTC_USDT",
    "ETH": "ETH_USDT",
    "SOL": "SOL_USDT",
}


class BasisArbitrage:
    """Monitors MEXC spot vs futures basis and logs opportunities."""

    def __init__(
        self,
        mexc_client,
        config: Optional[dict] = None,
        tracker=None,
    ):
        self.mexc = mexc_client
        self.tracker = tracker

        # Config
        cfg = (config or {}).get("basis_trading", {})
        self.observation_mode = cfg.get("observation_mode", True)
        self.min_basis_bps = Decimal(str(cfg.get("min_basis_bps", "5")))
        self.max_position_usd = cfg.get("max_position_usd", 1000)
        self.max_total_positions = cfg.get("max_total_positions", 3)
        self.scan_interval = cfg.get("scan_interval_seconds", 120)
        self.symbols_config = cfg.get("symbols", list(BASIS_SYMBOLS.keys()))

        # Build symbol map from config
        self._symbols: Dict[str, str] = {}
        for sym in self.symbols_config:
            sym = sym.upper()
            if sym in BASIS_SYMBOLS:
                self._symbols[sym] = BASIS_SYMBOLS[sym]
            else:
                # Allow custom symbols
                self._symbols[sym] = f"{sym}_USDT"

        # State
        self._running = False
        self._latest_snapshots: Dict[str, BasisSnapshot] = {}
        self._opportunities_found = 0
        self._scans_completed = 0
        self._errors = 0

        # DB
        self._db_path = str(Path("data") / "arbitrage.db")
        self._init_db()

    def _init_db(self):
        """Create basis-specific tables in arbitrage.db."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS basis_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    spot_price REAL NOT NULL,
                    futures_price REAL NOT NULL,
                    basis_abs REAL,
                    basis_pct REAL,
                    basis_bps REAL,
                    direction TEXT,
                    annualized_basis_pct REAL,
                    funding_rate REAL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS basis_opportunities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    spot_price REAL,
                    futures_price REAL,
                    basis_bps REAL,
                    direction TEXT,
                    signal TEXT,
                    edge_bps REAL,
                    estimated_daily_yield_usd REAL,
                    annualized_yield_pct REAL,
                    is_profitable INTEGER,
                    risk_notes TEXT,
                    observation_mode INTEGER DEFAULT 1,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS basis_positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_basis_bps REAL,
                    entry_spot_price REAL,
                    entry_futures_price REAL,
                    entry_timestamp TEXT,
                    size_usd REAL,
                    current_basis_bps REAL,
                    unrealized_pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    is_open INTEGER DEFAULT 1,
                    exit_timestamp TEXT,
                    exit_reason TEXT,
                    final_pnl REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_basis_snap_ts
                ON basis_snapshots(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_basis_snap_sym
                ON basis_snapshots(symbol, timestamp)
            """)
            conn.commit()
            conn.close()
            logger.info("Basis trading DB tables initialized")
        except Exception as e:
            logger.warning(f"Basis DB init error: {e}")

    async def run(self):
        """Main monitoring loop."""
        self._running = True
        mode = "OBSERVATION" if self.observation_mode else "LIVE"
        logger.info(
            f"BasisArbitrage started [{mode}] — "
            f"symbols={list(self._symbols.keys())} | "
            f"min_basis={self.min_basis_bps}bps | "
            f"interval={self.scan_interval}s"
        )

        while self._running:
            try:
                await self._scan_cycle()
            except Exception as e:
                self._errors += 1
                logger.error(f"Basis scan error: {e}")

            await asyncio.sleep(self.scan_interval)

    def stop(self):
        self._running = False
        logger.info("BasisArbitrage stopped")

    async def _scan_cycle(self):
        """Single scan: fetch prices, calculate basis, evaluate, persist."""
        # Fetch futures prices from MEXC contract API
        futures_prices = await self._fetch_futures_prices()
        if not futures_prices:
            logger.debug("No futures prices fetched — skipping cycle")
            return

        # Fetch spot prices from MEXC client
        spot_prices = await self._fetch_spot_prices()
        if not spot_prices:
            logger.debug("No spot prices fetched — skipping cycle")
            return

        self._scans_completed += 1
        snapshots: List[BasisSnapshot] = []
        opportunities: List[BasisOpportunity] = []

        for name, contract_sym in self._symbols.items():
            spot_sym = f"{name}/USDT"

            spot = spot_prices.get(spot_sym)
            futures_data = futures_prices.get(contract_sym)

            if spot is None or futures_data is None:
                continue

            futures = futures_data["price"]
            funding_rate = futures_data.get("funding_rate")

            try:
                snap = calculate_basis(
                    symbol=spot_sym,
                    spot_price=spot,
                    futures_price=futures,
                    funding_rate=funding_rate,
                )
                snapshots.append(snap)
                self._latest_snapshots[spot_sym] = snap

                opp = evaluate_opportunity(
                    snap,
                    min_basis_bps=self.min_basis_bps,
                    position_size_usd=self.max_position_usd,
                )

                if opp.is_profitable:
                    opportunities.append(opp)
                    self._opportunities_found += 1

                    # Check for double signal with funding rate
                    self._check_double_signal(name, snap, funding_rate)

                    logger.info(
                        f"BASIS OPP: {spot_sym} | "
                        f"spot={float(spot):.2f} fut={float(futures):.2f} | "
                        f"basis={float(snap.basis_bps):.1f}bps {snap.direction} | "
                        f"signal={opp.signal} | "
                        f"edge={float(opp.edge_bps):.1f}bps | "
                        f"APR={opp.annualized_yield_pct:.1f}%"
                        f"{' [OBSERVATION]' if self.observation_mode else ''}"
                    )

            except Exception as e:
                logger.debug(f"Basis calc error for {spot_sym}: {e}")
                continue

        # Persist snapshots and opportunities
        self._persist_snapshots(snapshots)
        if opportunities:
            self._persist_opportunities(opportunities)

        # Log summary every scan
        if snapshots:
            summary_parts = []
            for s in snapshots:
                summary_parts.append(
                    f"{s.symbol.split('/')[0]}={float(s.basis_bps):+.1f}bps"
                )
            logger.info(
                f"BASIS SCAN #{self._scans_completed}: {' | '.join(summary_parts)}"
            )

    def _check_double_signal(
        self, name: str, snap: BasisSnapshot, funding_rate: Optional[Decimal]
    ):
        """Log when basis and funding rate both signal on same symbol."""
        if funding_rate is None:
            return

        # Both contango + positive funding = strong sell_basis signal
        # Both backwardation + negative funding = strong buy_basis signal
        if snap.direction == "contango" and funding_rate > Decimal("0.0001"):
            logger.info(
                f"BASIS + FUNDING double signal: {snap.symbol} — "
                f"contango {float(snap.basis_bps):.1f}bps + "
                f"funding +{float(funding_rate)*100:.4f}% — "
                f"strong sell_basis convergence expected"
            )
        elif snap.direction == "backwardation" and funding_rate < Decimal("-0.0001"):
            logger.info(
                f"BASIS + FUNDING double signal: {snap.symbol} — "
                f"backwardation {float(snap.basis_bps):.1f}bps + "
                f"funding {float(funding_rate)*100:.4f}% — "
                f"strong buy_basis convergence expected"
            )

    async def _fetch_futures_prices(self) -> Dict[str, dict]:
        """Fetch futures/perpetual prices from MEXC contract REST API."""
        result = {}
        try:
            async with aiohttp.ClientSession() as session:
                for name, contract_sym in self._symbols.items():
                    try:
                        url = f"{MEXC_CONTRACT_API}?symbol={contract_sym}"
                        async with session.get(
                            url,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            if resp.status != 200:
                                logger.debug(
                                    f"MEXC contract API {resp.status} for {contract_sym}"
                                )
                                continue
                            data = await resp.json()

                        # MEXC contract API response structure:
                        # {"success": true, "data": {"lastPrice": ..., "fundingRate": ...}}
                        if not data.get("success"):
                            continue

                        ticker = data.get("data", {})
                        last_price = ticker.get("lastPrice")
                        if last_price is None:
                            continue

                        funding_rate_raw = ticker.get("fundingRate")
                        funding_rate = None
                        if funding_rate_raw is not None:
                            try:
                                funding_rate = Decimal(str(funding_rate_raw))
                            except (InvalidOperation, ValueError):
                                pass

                        result[contract_sym] = {
                            "price": Decimal(str(last_price)),
                            "funding_rate": funding_rate,
                        }

                    except asyncio.TimeoutError:
                        logger.debug(f"MEXC contract timeout for {contract_sym}")
                    except Exception as e:
                        logger.debug(f"MEXC contract fetch error {contract_sym}: {e}")

        except Exception as e:
            logger.warning(f"MEXC contract API session error: {e}")

        return result

    async def _fetch_spot_prices(self) -> Dict[str, Decimal]:
        """Fetch spot prices from MEXC client (already connected)."""
        result = {}
        for name in self._symbols:
            spot_sym = f"{name}/USDT"
            try:
                # Use the MEXC client's get_ticker or equivalent
                ticker = await self.mexc.get_ticker(spot_sym)
                if ticker and "last_price" in ticker:
                    result[spot_sym] = Decimal(str(ticker["last_price"]))
                elif ticker and "last" in ticker:
                    result[spot_sym] = Decimal(str(ticker["last"]))
            except Exception as e:
                logger.debug(f"Spot price fetch error for {spot_sym}: {e}")
        return result

    def _persist_snapshots(self, snapshots: List[BasisSnapshot]):
        """Write basis snapshots to DB."""
        if not snapshots:
            return
        try:
            conn = sqlite3.connect(self._db_path)
            now = datetime.utcnow().isoformat()
            for s in snapshots:
                conn.execute(
                    "INSERT INTO basis_snapshots "
                    "(symbol, spot_price, futures_price, basis_abs, basis_pct, "
                    "basis_bps, direction, annualized_basis_pct, funding_rate, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        s.symbol,
                        float(s.spot_price),
                        float(s.futures_price),
                        float(s.basis_abs),
                        float(s.basis_pct),
                        float(s.basis_bps),
                        s.direction,
                        float(s.annualized_basis_pct),
                        float(s.funding_rate) if s.funding_rate is not None else None,
                        now,
                    ),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Basis snapshot persist error: {e}")

    def _persist_opportunities(self, opportunities: List[BasisOpportunity]):
        """Write basis opportunities to DB."""
        try:
            conn = sqlite3.connect(self._db_path)
            now = datetime.utcnow().isoformat()
            for opp in opportunities:
                s = opp.snapshot
                conn.execute(
                    "INSERT INTO basis_opportunities "
                    "(symbol, spot_price, futures_price, basis_bps, direction, "
                    "signal, edge_bps, estimated_daily_yield_usd, annualized_yield_pct, "
                    "is_profitable, risk_notes, observation_mode, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        s.symbol,
                        float(s.spot_price),
                        float(s.futures_price),
                        float(s.basis_bps),
                        s.direction,
                        opp.signal,
                        float(opp.edge_bps),
                        opp.estimated_daily_yield_usd,
                        opp.annualized_yield_pct,
                        1 if opp.is_profitable else 0,
                        ",".join(opp.risk_notes) if opp.risk_notes else None,
                        1 if self.observation_mode else 0,
                        now,
                    ),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Basis opportunity persist error: {e}")

    def get_status(self) -> dict:
        """Return current status for dashboard/orchestrator."""
        return {
            "observation_mode": self.observation_mode,
            "running": self._running,
            "scans_completed": self._scans_completed,
            "opportunities_found": self._opportunities_found,
            "errors": self._errors,
            "symbols": list(self._symbols.keys()),
            "min_basis_bps": float(self.min_basis_bps),
            "scan_interval_seconds": self.scan_interval,
            "current_basis": {
                sym: {
                    "spot": float(snap.spot_price),
                    "futures": float(snap.futures_price),
                    "basis_bps": float(snap.basis_bps),
                    "direction": snap.direction,
                    "annualized_pct": float(snap.annualized_basis_pct),
                    "funding_rate": float(snap.funding_rate) if snap.funding_rate is not None else None,
                }
                for sym, snap in self._latest_snapshots.items()
            },
        }

    def get_stats(self) -> dict:
        """Alias for get_status (matches other strategy modules)."""
        return self.get_status()
