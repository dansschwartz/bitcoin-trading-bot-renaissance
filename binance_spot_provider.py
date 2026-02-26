"""
Binance Spot Data Provider — public market data for expanded trading universe.

Fetches OHLCV candles, tickers, and order books from Binance spot API.
No authentication required — uses public endpoints only.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any

import aiohttp


# ── Symbol mapping ──────────────────────────────────────────────

def to_binance_symbol(product_id: str) -> str:
    """BTC-USD → BTCUSDT"""
    base = product_id.split('-')[0]
    return f"{base}USDT"


def from_binance_symbol(binance_sym: str) -> str:
    """BTCUSDT → BTC-USD"""
    base = binance_sym.replace('USDT', '')
    return f"{base}-USD"


# ── Stablecoins and wrapped tokens to exclude ──────────────────

EXCLUDED_BASES: Set[str] = {
    # Stablecoins
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD', 'USDP', 'USDD', 'PYUSD',
    'USD1', 'USDJ', 'EURC', 'EURT', 'GBP', 'EUR', 'AEUR',
    # Wrapped / synthetic tokens
    'WBTC', 'WETH', 'STETH', 'CBETH', 'RETH', 'WBETH', 'WBNB', 'WBETH',
    # Gold / commodity backed
    'PAXG', 'XAUT',
    # Extremely low price / huge qty noise
    'BTTC', 'LUNC', 'SHIB',  # SHIB kept out — too many decimal places
    # Leverage tokens
    'BTCUP', 'BTCDOWN', 'ETHUP', 'ETHDOWN', 'BNBUP', 'BNBDOWN',
    # Problematic / delisted
    'LUNA', 'UST', 'FTT',
    # Additional stablecoins / stable-pegged
    'USDE', 'XUSD', 'FRAX', 'LUSD', 'SUSD', 'MIM', 'CRVUSD',
    # Promotional / low-quality tokens with wash volume on Binance
    'KITE', 'WLFI', 'ENSO', 'ASTER', 'ZAMA', 'ESP', 'SAPIEN',
    'PUMP', 'ALLO', 'SOMI', 'XPL', 'AT', 'BEL',
    # High-spread pairs where Devil > edge (council decision 2026-02-26)
    'AVAX',  # 15.12 bps avg spread vs ~2.7 bps gross edge
}


class BinanceSpotProvider:
    """Fetch spot OHLCV, ticker, and order book data from Binance public API."""

    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self._available_symbols: Set[str] = set()

        # Rate limit tracking (Binance allows 1200 req/min on public endpoints)
        self._api_calls_this_minute: int = 0
        self._api_minute_start: float = time.time()

    async def _ensure_session(self) -> None:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=15)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()

    async def _rate_limited_get(self, url: str, params: Optional[Dict] = None) -> Any:
        """GET with rate limit tracking. Returns parsed JSON."""
        await self._ensure_session()

        # Reset counter every minute
        now = time.time()
        if now - self._api_minute_start > 60:
            self._api_calls_this_minute = 0
            self._api_minute_start = now

        # Throttle if approaching Binance limit
        if self._api_calls_this_minute > 1000:
            self.logger.warning("BINANCE RATE LIMIT: throttling (>1000 req/min)")
            await asyncio.sleep(5)
            self._api_calls_this_minute = 0
            self._api_minute_start = time.time()

        self._api_calls_this_minute += 1

        async with self.session.get(url, params=params) as resp:
            if resp.status == 429:
                self.logger.warning("BINANCE 429: rate limited, backing off 10s")
                await asyncio.sleep(10)
                return None
            if resp.status != 200:
                self.logger.debug(f"Binance API {resp.status} for {url}")
                return None
            return await resp.json()

    # ── Exchange Info & Universe Discovery ──────────────────────

    async def fetch_available_pairs(self) -> List[str]:
        """Get all actively trading USDT spot pairs on Binance."""
        data = await self._rate_limited_get(f"{self.BASE_URL}/exchangeInfo")
        if not data:
            return []

        usdt_pairs = []
        for s in data.get('symbols', []):
            if (s.get('quoteAsset') == 'USDT'
                    and s.get('status') == 'TRADING'
                    and s.get('isSpotTradingAllowed', False)):
                usdt_pairs.append(s['symbol'])

        self._available_symbols = set(usdt_pairs)
        self.logger.info(f"Binance: {len(usdt_pairs)} USDT spot pairs available")
        return usdt_pairs

    async def fetch_all_tickers_24h(self) -> Dict[str, Dict[str, float]]:
        """Fetch 24h ticker stats for ALL symbols in a single request.
        Returns {symbol: {price, bid, ask, volume_24h, quote_volume_24h}}
        """
        data = await self._rate_limited_get(f"{self.BASE_URL}/ticker/24hr")
        if not data:
            return {}

        result = {}
        for t in data:
            sym = t.get('symbol', '')
            if not sym.endswith('USDT'):
                continue
            try:
                result[sym] = {
                    'price': float(t.get('lastPrice', 0)),
                    'bid': float(t.get('bidPrice', 0)),
                    'ask': float(t.get('askPrice', 0)),
                    'volume_24h': float(t.get('volume', 0)),
                    'quote_volume_24h': float(t.get('quoteVolume', 0)),
                    'price_change_pct': float(t.get('priceChangePercent', 0)),
                }
            except (ValueError, TypeError):
                continue
        return result

    # ── Per-Symbol Data Fetching ───────────────────────────────

    async def fetch_candles(self, symbol: str, interval: str = '5m',
                            limit: int = 200) -> List[Dict[str, Any]]:
        """Fetch OHLCV candles. Returns list of dicts with timestamp/OHLCV."""
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        raw = await self._rate_limited_get(f"{self.BASE_URL}/klines", params=params)
        if not raw:
            return []

        candles = []
        for k in raw:
            try:
                candles.append({
                    'timestamp': k[0] / 1000,  # ms → sec
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                })
            except (IndexError, ValueError, TypeError):
                continue
        return candles

    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, float]]:
        """Fetch current ticker for a single symbol."""
        data = await self._rate_limited_get(
            f"{self.BASE_URL}/ticker/24hr", params={'symbol': symbol}
        )
        if not data:
            return None
        try:
            return {
                'price': float(data.get('lastPrice', 0)),
                'bid': float(data.get('bidPrice', 0)),
                'ask': float(data.get('askPrice', 0)),
                'volume_24h': float(data.get('volume', 0)),
                'quote_volume_24h': float(data.get('quoteVolume', 0)),
            }
        except (ValueError, TypeError):
            return None

    async def fetch_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Fetch order book snapshot."""
        data = await self._rate_limited_get(
            f"{self.BASE_URL}/depth", params={'symbol': symbol, 'limit': limit}
        )
        if not data:
            return None
        try:
            return {
                'bids': [(float(b[0]), float(b[1])) for b in data.get('bids', [])],
                'asks': [(float(a[0]), float(a[1])) for a in data.get('asks', [])],
            }
        except (IndexError, ValueError, TypeError):
            return None

    # ── Batch Fetch (for parallel data collection) ─────────────

    async def fetch_pair_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch ticker + latest candle for one pair. Used in parallel fetch phase."""
        try:
            ticker = await self.fetch_ticker(symbol)
            if not ticker or ticker.get('price', 0) <= 0:
                return None
            return {
                'symbol': symbol,
                'ticker': ticker,
            }
        except Exception as e:
            self.logger.debug(f"Binance fetch_pair_data failed for {symbol}: {e}")
            return None

    # ── Universe Builder ───────────────────────────────────────

    async def build_trading_universe(self, min_volume_usd: float = 2_000_000,
                                      max_pairs: int = 150) -> List[Dict[str, Any]]:
        """Build trading universe: top Binance USDT pairs by 24h volume.

        Returns list of dicts with: binance_symbol, product_id, daily_volume_usd, tier
        """
        # Fetch all available pairs
        if not self._available_symbols:
            await self.fetch_available_pairs()

        # Fetch all 24h tickers in one request
        all_tickers = await self.fetch_all_tickers_24h()
        if not all_tickers:
            self.logger.error("UNIVERSE BUILD FAILED: no ticker data from Binance")
            return []

        # Filter by availability, volume, and exclusions
        candidates = []
        for sym in self._available_symbols:
            if not sym.endswith('USDT'):
                continue
            base = sym.replace('USDT', '')
            if base in EXCLUDED_BASES:
                continue
            # Skip tokens with non-ASCII names or very short/long names
            if not base.isascii() or len(base) > 10 or len(base) < 2:
                continue
            ticker = all_tickers.get(sym)
            if not ticker:
                continue
            vol = ticker.get('quote_volume_24h', 0)
            if vol < min_volume_usd:
                continue
            # Skip tokens with suspiciously high volume but very low price
            price = ticker.get('price', 0)
            if price <= 0:
                continue
            candidates.append({
                'binance_symbol': sym,
                'product_id': f"{base}-USD",
                'daily_volume_usd': vol,
            })

        # Sort by volume descending, take top N
        candidates.sort(key=lambda x: x['daily_volume_usd'], reverse=True)
        candidates = candidates[:max_pairs]

        # Assign 4 tiers based on volume rank
        for i, c in enumerate(candidates):
            if i < 15:
                c['tier'] = 1       # Top 15: every cycle
            elif i < 50:
                c['tier'] = 2       # 16-50: every 2nd cycle
            elif i < 100:
                c['tier'] = 3       # 51-100: every 3rd cycle
            else:
                c['tier'] = 4       # 101-150: every 4th cycle

        tier_counts = {}
        for c in candidates:
            tier_counts[c['tier']] = tier_counts.get(c['tier'], 0) + 1

        self.logger.info(
            f"UNIVERSE: {len(candidates)} pairs — "
            + ", ".join(f"Tier {t}: {n}" for t, n in sorted(tier_counts.items()))
        )

        return candidates
