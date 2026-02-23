"""
Breakout Scanner — Scans ALL Binance USDT pairs in one API call.

Architecture:
  Phase 1 (lightweight): GET /ticker/24hr -> score all 600+ pairs -> flag top N
  Phase 2 (deep): Fetch candles only for flagged pairs -> ML inference -> trade

Breakout signals detected:
  1. Volume spike — current volume >> historical average
  2. Price breakout — price near 24h high or low (range expansion)
  3. Momentum surge — large % move in short window
  4. Volatility expansion — ATR spike relative to recent history
  5. Cross-asset divergence — pair moving much more than BTC/ETH

Each pair gets a composite breakout_score (0-100). Top N are flagged for deep scan.
"""

import asyncio
import logging
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Pairs to always exclude (stablecoins, wrapped tokens, leveraged tokens)
EXCLUDED_BASES = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD', 'USDP', 'PYUSD',
    'WBTC', 'WETH', 'STETH', 'CBETH', 'RETH', 'WBETH',
}

# Substrings that indicate leveraged/derivative tokens
EXCLUDED_SUBSTRINGS = ['UP', 'DOWN', 'BULL', 'BEAR', '3L', '3S', '2L', '2S']

# Always deep-scan these regardless of breakout score (Tier 1 majors)
ALWAYS_SCAN = {
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT',
    'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'MATICUSDT', 'UNIUSDT', 'ATOMUSDT',
    'LTCUSDT', 'BCHUSDT', 'NEARUSDT',
}

BINANCE_ENDPOINTS = [
    "https://api.binance.com/api/v3",   # Global (600+ USDT pairs, blocked in US)
    "https://api.binance.us/api/v3",    # US fallback (230+ USDT pairs)
]


@dataclass
class BreakoutSignal:
    """Scored breakout signal for a single pair."""
    symbol: str                    # Binance symbol (e.g. VANRYUSDT)
    product_id: str                # Our format (e.g. VANRY-USD)
    breakout_score: float          # Composite score 0-100
    volume_score: float            # Volume spike component
    price_score: float             # Range breakout component
    momentum_score: float          # % move magnitude
    volatility_score: float        # ATR expansion
    divergence_score: float        # vs BTC/ETH movement
    direction: str                 # 'bullish' or 'bearish'
    price: float                   # Current price
    volume_24h_usd: float          # 24h quote volume
    price_change_pct: float        # 24h price change %
    details: Dict[str, Any] = field(default_factory=dict)


class BreakoutScanner:
    """
    Scans all Binance USDT pairs for breakout signals in a single API call.

    Usage:
        scanner = BreakoutScanner()
        flagged = await scanner.scan()  # Returns top N breakout signals

        # Feed flagged pair list to the ML pipeline:
        for signal in flagged:
            deep_scan(signal.product_id)
    """

    def __init__(
        self,
        max_flagged: int = 30,
        min_volume_usd: float = 500_000,
        min_breakout_score: float = 25.0,
        history_size: int = 288,            # 24h of 5-min snapshots
        logger: Optional[logging.Logger] = None,
    ):
        self.max_flagged = max_flagged
        self.min_volume_usd = min_volume_usd
        self.min_breakout_score = min_breakout_score
        self.history_size = history_size
        self.logger = logger or logging.getLogger(__name__)

        # Historical ticker data for computing moving averages
        # symbol -> deque of {volume, price, high, low, timestamp}
        self._history: Dict[str, deque] = {}

        # Track BTC/ETH returns for divergence calculation
        self._btc_change: float = 0.0
        self._eth_change: float = 0.0

        # Session for async requests
        self._session = None

        # Scan stats
        self._last_scan_time: float = 0
        self._total_scans: int = 0
        self._total_flagged: int = 0

    async def _ensure_session(self) -> None:
        """Create aiohttp session if needed."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
            )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    # ══════════════════════════════════════════════════════════════
    # MAIN SCAN — called every cycle
    # ══════════════════════════════════════════════════════════════

    async def scan(self) -> List[BreakoutSignal]:
        """
        Scan ALL Binance USDT pairs in one API call.
        Returns top N breakout signals sorted by score.

        This is the Phase 1 lightweight scan. Takes ~1-2 seconds.
        """
        scan_start = time.time()
        await self._ensure_session()

        # ── Single API call: get all 600+ tickers (try global, fallback to US) ──
        all_tickers = None
        for base_url in BINANCE_ENDPOINTS:
            try:
                async with self._session.get(f"{base_url}/ticker/24hr") as resp:
                    if resp.status == 200:
                        all_tickers = await resp.json()
                        break
                    self.logger.debug(f"Binance ticker {base_url} returned {resp.status}")
            except Exception as e:
                self.logger.debug(f"Binance ticker {base_url} failed: {e}")
        if not all_tickers:
            self.logger.warning("All Binance ticker endpoints failed")
            return []

        # ── Filter to USDT pairs, exclude junk ──
        usdt_tickers = []
        for t in all_tickers:
            sym = t.get('symbol', '')
            if not sym.endswith('USDT'):
                continue
            base = sym.replace('USDT', '')
            if base in EXCLUDED_BASES:
                continue
            if any(sub in base for sub in EXCLUDED_SUBSTRINGS):
                continue
            quote_vol = float(t.get('quoteVolume', 0))
            if quote_vol < self.min_volume_usd:
                continue
            usdt_tickers.append(t)

        # ── Extract BTC/ETH benchmarks for divergence ──
        for t in all_tickers:
            if t.get('symbol') == 'BTCUSDT':
                self._btc_change = float(t.get('priceChangePercent', 0))
            elif t.get('symbol') == 'ETHUSDT':
                self._eth_change = float(t.get('priceChangePercent', 0))

        # ── Score every pair ──
        signals: List[BreakoutSignal] = []
        for t in usdt_tickers:
            try:
                signal = self._score_ticker(t)
                if signal and signal.breakout_score >= self.min_breakout_score:
                    signals.append(signal)
            except Exception:
                pass  # Skip broken tickers silently

        # ── Update history for next cycle ──
        for t in usdt_tickers:
            self._update_history(t)

        # ── Sort by score, take top N ──
        signals.sort(key=lambda s: s.breakout_score, reverse=True)
        flagged = signals[:self.max_flagged]

        # ── Stats ──
        scan_elapsed = time.time() - scan_start
        self._total_scans += 1
        self._total_flagged += len(flagged)
        self._last_scan_time = scan_elapsed
        self.last_scan_count = len(usdt_tickers)

        if flagged:
            top3 = ', '.join(f"{s.product_id}({s.breakout_score:.0f})" for s in flagged[:3])
            self.logger.info(
                f"BREAKOUT SCAN: {len(usdt_tickers)} pairs in {scan_elapsed:.1f}s | "
                f"{len(flagged)} flagged | top: {top3}"
            )
        else:
            self.logger.debug(
                f"BREAKOUT SCAN: {len(usdt_tickers)} pairs in {scan_elapsed:.1f}s | "
                f"0 flagged (quiet market)"
            )

        return flagged

    # ══════════════════════════════════════════════════════════════
    # SCORING ENGINE
    # ══════════════════════════════════════════════════════════════

    def _score_ticker(self, ticker: dict) -> Optional[BreakoutSignal]:
        """
        Score a single ticker for breakout potential.
        Returns BreakoutSignal or None if pair is uninteresting.
        """
        sym = ticker['symbol']
        base = sym.replace('USDT', '')
        product_id = f"{base}-USD"

        price = float(ticker.get('lastPrice', 0))
        if price <= 0:
            return None

        price_change_pct = float(ticker.get('priceChangePercent', 0))
        high_24h = float(ticker.get('highPrice', 0))
        low_24h = float(ticker.get('lowPrice', 0))
        volume_24h = float(ticker.get('quoteVolume', 0))

        # ── 1. Volume Score (0-30 points) ──
        volume_score = self._calc_volume_score(sym, volume_24h)

        # ── 2. Price Breakout Score (0-25 points) ──
        price_score = 0.0
        if high_24h > low_24h > 0:
            range_24h = high_24h - low_24h
            range_pct = range_24h / low_24h

            dist_from_high = (high_24h - price) / (range_24h + 1e-10)
            dist_from_low = (price - low_24h) / (range_24h + 1e-10)

            if dist_from_high < 0.05:  # Within 5% of 24h high
                price_score = 25.0 * (1.0 - dist_from_high / 0.05)
            elif dist_from_low < 0.05:  # Within 5% of 24h low
                price_score = 25.0 * (1.0 - dist_from_low / 0.05)

            # Bonus for wide range (volatility expansion)
            if range_pct > 0.10:  # >10% intraday range
                price_score = min(25.0, price_score * 1.5)

        # ── 3. Momentum Score (0-25 points) ──
        abs_change = abs(price_change_pct)
        if abs_change > 20:
            momentum_score = 25.0
        elif abs_change > 10:
            momentum_score = 20.0
        elif abs_change > 5:
            momentum_score = 15.0
        elif abs_change > 3:
            momentum_score = 10.0
        elif abs_change > 1.5:
            momentum_score = 5.0
        else:
            momentum_score = abs_change / 1.5 * 5.0

        # ── 4. Volatility Score (0-10 points) ──
        volatility_score = self._calc_volatility_score(sym, high_24h, low_24h, price)

        # ── 5. Divergence Score (0-10 points) ──
        divergence_score = 0.0
        btc_abs = abs(self._btc_change) + 0.1
        pair_abs = abs(price_change_pct)

        divergence_ratio = pair_abs / btc_abs
        if divergence_ratio > 5:
            divergence_score = 10.0
        elif divergence_ratio > 3:
            divergence_score = 7.0
        elif divergence_ratio > 2:
            divergence_score = 4.0

        # ── Composite score ──
        breakout_score = volume_score + price_score + momentum_score + volatility_score + divergence_score

        # ── Direction ──
        direction = 'bullish' if price_change_pct > 0 else 'bearish'

        return BreakoutSignal(
            symbol=sym,
            product_id=product_id,
            breakout_score=breakout_score,
            volume_score=volume_score,
            price_score=price_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            divergence_score=divergence_score,
            direction=direction,
            price=price,
            volume_24h_usd=volume_24h,
            price_change_pct=price_change_pct,
            details={
                'high_24h': high_24h,
                'low_24h': low_24h,
                'btc_change': self._btc_change,
                'eth_change': self._eth_change,
            },
        )

    def _calc_volume_score(self, symbol: str, current_volume: float) -> float:
        """Score volume spike relative to historical average. 0-30 points."""
        hist = self._history.get(symbol)
        if not hist or len(hist) < 3:
            if current_volume > 100_000_000:
                return 15.0
            elif current_volume > 50_000_000:
                return 10.0
            elif current_volume > 10_000_000:
                return 5.0
            return 0.0

        avg_vol = np.mean([h['volume'] for h in hist])
        if avg_vol <= 0:
            return 0.0

        vol_ratio = current_volume / avg_vol
        if vol_ratio > 5.0:
            return 30.0
        elif vol_ratio > 3.0:
            return 22.0
        elif vol_ratio > 2.0:
            return 15.0
        elif vol_ratio > 1.5:
            return 8.0
        return 0.0

    def _calc_volatility_score(self, symbol: str, high: float, low: float, price: float) -> float:
        """Score volatility expansion. 0-10 points."""
        if price <= 0 or high <= low:
            return 0.0

        current_range_pct = (high - low) / price * 100

        hist = self._history.get(symbol)
        if not hist or len(hist) < 3:
            if current_range_pct > 15:
                return 10.0
            elif current_range_pct > 8:
                return 6.0
            elif current_range_pct > 4:
                return 3.0
            return 0.0

        avg_range = np.mean([h.get('range_pct', 2.0) for h in hist])
        if avg_range <= 0:
            return 0.0

        range_ratio = current_range_pct / avg_range
        if range_ratio > 3.0:
            return 10.0
        elif range_ratio > 2.0:
            return 7.0
        elif range_ratio > 1.5:
            return 4.0
        return 0.0

    def _update_history(self, ticker: dict) -> None:
        """Store ticker snapshot for historical comparison."""
        sym = ticker['symbol']
        if sym not in self._history:
            self._history[sym] = deque(maxlen=self.history_size)

        price = float(ticker.get('lastPrice', 0))
        high = float(ticker.get('highPrice', 0))
        low = float(ticker.get('lowPrice', 0))

        self._history[sym].append({
            'volume': float(ticker.get('quoteVolume', 0)),
            'price': price,
            'high': high,
            'low': low,
            'range_pct': ((high - low) / price * 100) if price > 0 else 0,
            'timestamp': time.time(),
        })

    # ══════════════════════════════════════════════════════════════
    # UTILITY
    # ══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Return scanner statistics."""
        return {
            'total_scans': self._total_scans,
            'total_flagged': self._total_flagged,
            'avg_flagged_per_scan': self._total_flagged / max(1, self._total_scans),
            'last_scan_seconds': self._last_scan_time,
            'pairs_tracked': len(self._history),
        }

    def get_always_scan_pairs(self) -> List[str]:
        """Return the Tier 1 major pairs that always get deep-scanned."""
        return [s.replace('USDT', '-USD') for s in ALWAYS_SCAN]
