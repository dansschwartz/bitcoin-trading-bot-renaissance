"""
Funding Rate Arbitrage — delta-neutral strategy capturing persistent
funding rate differentials on Binance perpetual futures.

HOW IT WORKS:
  When funding rate > +0.01%: longs pay shorts every 8h.
  → Go LONG spot (MEXC) + SHORT perp (Binance) → collect funding.
  When funding rate < -0.01%: shorts pay longs every 8h.
  → Go SHORT spot + LONG perp → collect funding.

The two legs hedge each other — net market exposure ≈ zero.
Profit comes purely from collecting funding payments.

Binance funding rate API is public (no auth), settles at 00:00, 08:00, 16:00 UTC.
"""
import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

from ..exchanges.base import OrderSide

logger = logging.getLogger("arb.funding")

# Symbols we monitor for funding arb
FUNDING_SYMBOLS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
    'DOGE': 'DOGEUSDT',
    'AVAX': 'AVAXUSDT',
    'LINK': 'LINKUSDT',
}

BINANCE_FAPI = "https://fapi.binance.com"


@dataclass
class FundingOpportunity:
    symbol: str
    rate: Decimal
    direction: str  # 'long_spot_short_perp' or 'short_spot_long_perp'
    estimated_daily_yield_usd: float
    annualized_yield_pct: float
    next_funding_time: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FundingPosition:
    position_id: str
    symbol: str
    direction: str
    entry_funding_rate: Decimal
    entry_timestamp: datetime
    entry_spot_price: float
    size_usd: float
    total_funding_collected: float = 0.0
    funding_payments_count: int = 0
    status: str = 'open'
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None
    final_pnl: Optional[float] = None


class FundingRateArbitrage:
    """Monitors Binance perp funding rates and manages hedged positions."""

    def __init__(self, mexc_client, binance_client, risk_engine,
                 config: Optional[dict] = None, tracker=None):
        self.mexc = mexc_client
        self.binance = binance_client
        self.risk = risk_engine
        self.tracker = tracker

        # Config
        cfg = (config or {}).get('funding_rate', {})
        self.observation_mode = cfg.get('observation_mode', True)
        self.entry_threshold = Decimal(str(cfg.get('entry_threshold', '0.0001')))
        self.exit_threshold = Decimal(str(cfg.get('exit_threshold', '0.00005')))
        self.max_position_usd = cfg.get('max_position_usd', 1000)
        self.max_total_positions = cfg.get('max_total_positions', 3)
        self.check_interval = cfg.get('check_interval_seconds', 60)

        # State
        self._positions: Dict[str, FundingPosition] = {}
        self._current_rates: Dict[str, dict] = {}
        self._opportunities_found = 0
        self._running = False
        self._last_funding_collect: Optional[datetime] = None

        # DB
        self._db_path = str(Path("data") / "arbitrage.db")
        self._init_db()

    def _init_db(self):
        """Create funding-specific tables."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS funding_positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_funding_rate REAL,
                    entry_timestamp TEXT,
                    entry_spot_price REAL,
                    size_usd REAL,
                    total_funding_collected REAL DEFAULT 0,
                    funding_payments_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    exit_timestamp TEXT,
                    exit_reason TEXT,
                    final_pnl REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS funding_payments (
                    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT,
                    symbol TEXT,
                    funding_rate REAL,
                    payment_usd REAL,
                    timestamp TEXT,
                    FOREIGN KEY (position_id) REFERENCES funding_positions(position_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS funding_rate_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    funding_rate REAL,
                    annualized_pct REAL,
                    signal TEXT,
                    timestamp TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Funding DB init error: {e}")

    async def run(self):
        """Main monitoring loop."""
        self._running = True
        mode = "OBSERVATION" if self.observation_mode else "LIVE"
        logger.info(f"FundingRateArbitrage started [{mode}] — "
                    f"threshold={float(self.entry_threshold)*100:.3f}%")

        while self._running:
            try:
                # Fetch current rates from Binance
                rates = await self.fetch_current_rates()
                self._current_rates = rates

                # Evaluate opportunities
                opportunities = self.evaluate_opportunities(rates)

                # Record rate history
                self._persist_rate_snapshot(rates)

                # Log opportunities
                for opp in opportunities:
                    self._opportunities_found += 1
                    logger.info(
                        f"FUNDING OPP: {opp.symbol} | rate={float(opp.rate)*100:.4f}% | "
                        f"direction={opp.direction} | "
                        f"daily_yield=${opp.estimated_daily_yield_usd:.2f} | "
                        f"APR={opp.annualized_yield_pct:.1f}%"
                        f"{' [OBSERVATION]' if self.observation_mode else ''}"
                    )

                    if not self.observation_mode:
                        if opp.symbol not in self._positions:
                            await self._open_position(opp)

                # Check exits for open positions
                await self._check_exits(rates)

                # Collect funding near settlement times (00:05, 08:05, 16:05 UTC)
                await self._maybe_collect_funding(rates)

            except Exception as e:
                logger.error(f"Funding rate scan error: {e}")

            await asyncio.sleep(self.check_interval)

    def stop(self):
        self._running = False

    async def fetch_current_rates(self) -> Dict[str, dict]:
        """Fetch current funding rates from Binance premiumIndex (public, no auth)."""
        result = {}
        try:
            url = f"{BINANCE_FAPI}/fapi/v1/premiumIndex"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning(f"Binance funding API returned {resp.status}")
                        return result
                    data = await resp.json()

            for item in data:
                sym = item.get('symbol', '')
                if sym not in FUNDING_SYMBOLS.values():
                    continue
                # Reverse lookup: BTCUSDT -> BTC
                name = next((k for k, v in FUNDING_SYMBOLS.items() if v == sym), sym)
                rate = Decimal(str(item.get('lastFundingRate', '0')))
                next_time_ms = item.get('nextFundingTime', 0)
                next_time = datetime.fromtimestamp(next_time_ms / 1000, tz=timezone.utc) if next_time_ms else None
                mark_price = float(item.get('markPrice', 0))

                result[name] = {
                    'symbol': f"{name}/USDT",
                    'binance_symbol': sym,
                    'rate': rate,
                    'annualized_pct': float(abs(rate) * 3 * 365 * 100),
                    'next_funding_time': next_time,
                    'mark_price': mark_price,
                }
        except Exception as e:
            logger.warning(f"Binance funding rate fetch error: {e}")

        return result

    def evaluate_opportunities(self, rates: Dict[str, dict]) -> List[FundingOpportunity]:
        """Evaluate which symbols have actionable funding rates."""
        opportunities = []

        for name, info in rates.items():
            rate = info['rate']
            abs_rate = abs(rate)

            if abs_rate < self.entry_threshold:
                continue

            # Positive rate: longs pay shorts → go long spot + short perp
            # Negative rate: shorts pay longs → go short spot + long perp
            if rate > 0:
                direction = 'long_spot_short_perp'
            else:
                direction = 'short_spot_long_perp'

            # Estimate daily yield (3 funding periods per day)
            daily_yield = float(abs_rate) * 3 * self.max_position_usd
            annual_pct = float(abs_rate) * 3 * 365 * 100

            next_time = info.get('next_funding_time')

            opportunities.append(FundingOpportunity(
                symbol=info['symbol'],
                rate=rate,
                direction=direction,
                estimated_daily_yield_usd=daily_yield,
                annualized_yield_pct=annual_pct,
                next_funding_time=next_time,
            ))

        opportunities.sort(key=lambda x: abs(float(x.rate)), reverse=True)
        return opportunities

    async def _open_position(self, opp: FundingOpportunity):
        """Open a hedged funding arb position (paper trading only for now)."""
        if len([p for p in self._positions.values() if p.status == 'open']) >= self.max_total_positions:
            return

        position = FundingPosition(
            position_id=f"fund_{opp.symbol.replace('/', '')}_{int(time.time())}",
            symbol=opp.symbol,
            direction=opp.direction,
            entry_funding_rate=opp.rate,
            entry_timestamp=datetime.utcnow(),
            entry_spot_price=0.0,  # Would come from MEXC spot price
            size_usd=self.max_position_usd,
        )
        self._positions[opp.symbol] = position
        self._persist_position(position)

        logger.info(
            f"FUNDING OPEN: {opp.symbol} | {opp.direction} | "
            f"size=${self.max_position_usd} | rate={float(opp.rate)*100:.4f}%"
        )

    async def _check_exits(self, rates: Dict[str, dict]):
        """Close positions where funding rate has normalized."""
        to_close = []
        for symbol, pos in self._positions.items():
            if pos.status != 'open':
                continue

            name = symbol.split('/')[0]
            info = rates.get(name)
            if not info:
                continue

            current_rate = abs(info['rate'])
            if current_rate < self.exit_threshold:
                to_close.append((symbol, 'rate_normalized'))
                logger.info(
                    f"FUNDING CLOSE: {symbol} | rate normalized to "
                    f"{float(info['rate'])*100:.4f}% | "
                    f"collected=${pos.total_funding_collected:.4f} over "
                    f"{pos.funding_payments_count} periods"
                )

        for symbol, reason in to_close:
            pos = self._positions[symbol]
            pos.status = 'closed'
            pos.exit_timestamp = datetime.utcnow()
            pos.exit_reason = reason
            pos.final_pnl = pos.total_funding_collected
            self._persist_position(pos)

    async def _maybe_collect_funding(self, rates: Dict[str, dict]):
        """Simulate funding collection at settlement times."""
        now = datetime.now(timezone.utc)
        # Settle at 00:00, 08:00, 16:00 UTC — check within 5 min window
        hour = now.hour
        minute = now.minute
        is_settlement = hour in (0, 8, 16) and minute < 5

        if not is_settlement:
            return

        # Don't double-collect
        if self._last_funding_collect and \
           (now - self._last_funding_collect).total_seconds() < 3600:
            return

        self._last_funding_collect = now

        for symbol, pos in self._positions.items():
            if pos.status != 'open':
                continue

            name = symbol.split('/')[0]
            info = rates.get(name)
            if not info:
                continue

            rate = info['rate']
            # Payment = position_size * |rate|
            payment = float(abs(rate)) * pos.size_usd
            pos.total_funding_collected += payment
            pos.funding_payments_count += 1

            self._persist_payment(pos.position_id, symbol, float(rate), payment)
            self._persist_position(pos)

            logger.info(
                f"FUNDING COLLECT: {symbol} | rate={float(rate)*100:.4f}% | "
                f"payment=${payment:.4f} | total=${pos.total_funding_collected:.4f}"
            )

    def _persist_position(self, pos: FundingPosition):
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT OR REPLACE INTO funding_positions "
                "(position_id, symbol, direction, entry_funding_rate, entry_timestamp, "
                "entry_spot_price, size_usd, total_funding_collected, funding_payments_count, "
                "status, exit_timestamp, exit_reason, final_pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pos.position_id, pos.symbol, pos.direction,
                 float(pos.entry_funding_rate), pos.entry_timestamp.isoformat(),
                 pos.entry_spot_price, pos.size_usd,
                 pos.total_funding_collected, pos.funding_payments_count,
                 pos.status,
                 pos.exit_timestamp.isoformat() if pos.exit_timestamp else None,
                 pos.exit_reason, pos.final_pnl),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Funding position persist error: {e}")

    def _persist_payment(self, position_id: str, symbol: str,
                         rate: float, payment: float):
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO funding_payments "
                "(position_id, symbol, funding_rate, payment_usd, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (position_id, symbol, rate, payment, datetime.utcnow().isoformat()),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Funding payment persist error: {e}")

    def _persist_rate_snapshot(self, rates: Dict[str, dict]):
        """Record current rates for historical analysis."""
        try:
            conn = sqlite3.connect(self._db_path)
            now = datetime.utcnow().isoformat()
            for name, info in rates.items():
                rate = float(info['rate'])
                annual = info['annualized_pct']
                signal = 'none'
                if abs(info['rate']) >= self.entry_threshold:
                    signal = 'long_spot_short_perp' if info['rate'] > 0 else 'short_spot_long_perp'
                conn.execute(
                    "INSERT INTO funding_rate_history "
                    "(symbol, funding_rate, annualized_pct, signal, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (info['symbol'], rate, annual, signal, now),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Funding rate history persist error: {e}")

    def get_current_rates(self) -> Dict[str, dict]:
        """Return last fetched rates for dashboard."""
        return self._current_rates

    def get_stats(self) -> dict:
        open_positions = {s: p for s, p in self._positions.items() if p.status == 'open'}
        total_collected = sum(p.total_funding_collected for p in self._positions.values())
        return {
            "observation_mode": self.observation_mode,
            "opportunities_found": self._opportunities_found,
            "open_positions": len(open_positions),
            "closed_positions": len(self._positions) - len(open_positions),
            "total_funding_collected_usd": round(total_collected, 4),
            "current_rates": {
                name: {
                    "rate": float(info['rate']),
                    "annualized": f"{info['annualized_pct']:.1f}%",
                    "signal": 'long_spot_short_perp' if info['rate'] > 0 and abs(info['rate']) >= float(self.entry_threshold)
                             else ('short_spot_long_perp' if info['rate'] < 0 and abs(info['rate']) >= float(self.entry_threshold)
                             else 'none'),
                }
                for name, info in self._current_rates.items()
            },
            "positions": {
                s: {
                    "direction": p.direction,
                    "size_usd": p.size_usd,
                    "collected_usd": round(p.total_funding_collected, 4),
                    "payments": p.funding_payments_count,
                    "age_hours": round((datetime.utcnow() - p.entry_timestamp).total_seconds() / 3600, 1),
                }
                for s, p in open_positions.items()
            },
        }
