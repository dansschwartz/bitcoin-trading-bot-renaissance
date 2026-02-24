"""
Dynamic Pair Discovery Engine — discovers profitable cross-exchange pairs
by scanning ALL overlapping USDT pairs between MEXC and Binance.

Replaces the static 30-pair list with a dynamic top-40 ranked by gross spread.
Promotes/demotes pairs in UnifiedBookManager automatically.

Uses direct REST calls (aiohttp) to fetch all tickers — bypasses ccxt which
fails on VPS due to geo-blocking / missing market data.

Flow (every 60s):
  1. Direct REST to both exchanges (1 call each, all pairs)
  2. Intersect to find overlapping USDT pairs (~300-500)
  3. Exclude stablecoins, wrapped tokens, blocklisted
  4. Compute gross spread from bid/ask in both directions
  5. Rank by best spread, filter by min threshold
  6. Promote top MAX_ACTIVE_PAIRS into book_manager
  7. Demote stale pairs (with grace period to avoid churn)
"""
import asyncio
import logging
import time
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Set

import aiohttp

logger = logging.getLogger("arb.pair_discovery")

# Tokens to exclude from cross-exchange arb (stablecoins, wrapped, gold, fiat, junk)
EXCLUDED_BASES: Set[str] = {
    # Stablecoins
    'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD', 'UST', 'USDD', 'USDP', 'PYUSD',
    # Fiat-backed
    'EUR', 'GBP', 'JPY', 'AUD', 'BRL', 'TRY',
    # Wrapped tokens
    'WBTC', 'WETH', 'WBNB', 'WTRX', 'WMATIC',
    # Gold-backed
    'PAXG', 'XAUT',
}


class PairDiscoveryEngine:
    """Discovers profitable cross-exchange pairs by scanning all overlapping
    USDT pairs between MEXC and Binance. Promotes/demotes pairs dynamically
    in the UnifiedBookManager."""

    def __init__(
        self,
        mexc,
        binance,
        book_manager,
        contract_verifier=None,
        config: Optional[dict] = None,
    ):
        self.mexc = mexc
        self.binance = binance
        self.book_manager = book_manager
        self.contract_verifier = contract_verifier

        # Load config with defaults
        cfg = (config or {}).get('pair_discovery', {})
        self.scan_interval = cfg.get('scan_interval_sec', 60)
        self.max_active_pairs = cfg.get('max_active_pairs', 40)
        self.min_gross_spread_bps = Decimal(str(cfg.get('min_gross_spread_bps', 3.0)))
        self.min_volume_usdt = Decimal(str(cfg.get('min_volume_usdt', 10_000)))
        self.demotion_grace_scans = cfg.get('demotion_grace_scans', 3)

        # Merge config exclusions with hardcoded set
        extra_excluded = cfg.get('excluded_bases', [])
        self.excluded_bases = EXCLUDED_BASES | set(extra_excluded)

        self._running = False
        self._scan_count = 0
        self._grace_tracker: Dict[str, int] = {}  # pair -> consecutive scans below threshold
        self._promoted_pairs: Set[str] = set()  # currently promoted by discovery

        # Stats
        self._stats = {
            'total_overlapping': 0,
            'above_threshold': 0,
            'promoted': 0,
            'demoted': 0,
            'last_scan_time': 0.0,
            'last_scan_ts': 0,
        }

    async def run(self) -> None:
        """Main discovery loop. Runs until stop() is called."""
        self._running = True
        logger.info(
            f"PAIR_DISCOVERY: Started — scan every {self.scan_interval}s, "
            f"max {self.max_active_pairs} pairs, "
            f"min spread {self.min_gross_spread_bps}bps"
        )

        while self._running:
            try:
                await self._scan_once()
            except Exception as e:
                logger.error(f"PAIR_DISCOVERY: Scan error: {e}")
            await asyncio.sleep(self.scan_interval)

    def stop(self) -> None:
        self._running = False
        logger.info("PAIR_DISCOVERY: Stopped")

    async def _scan_once(self) -> None:
        """Single discovery scan cycle."""
        t0 = time.monotonic()

        # 1. Fetch all tickers from both exchanges via direct REST (parallel)
        mexc_tickers, binance_tickers = await asyncio.gather(
            self._fetch_mexc_tickers_direct(),
            self._fetch_binance_tickers_direct(),
        )

        if not mexc_tickers or not binance_tickers:
            logger.warning("PAIR_DISCOVERY: One or both ticker fetches returned empty")
            return

        # 2. Find overlapping USDT pairs
        mexc_usdt = {s for s in mexc_tickers if s.endswith('/USDT')}
        binance_usdt = {s for s in binance_tickers if s.endswith('/USDT')}
        overlapping = mexc_usdt & binance_usdt

        # 3. Filter exclusions
        candidates: List[dict] = []
        for pair in overlapping:
            base = pair.split('/')[0]
            if base in self.excluded_bases:
                continue

            m = mexc_tickers[pair]
            b = binance_tickers[pair]

            m_bid = m.get('bid', Decimal('0'))
            m_ask = m.get('ask', Decimal('0'))
            b_bid = b.get('bid', Decimal('0'))
            b_ask = b.get('ask', Decimal('0'))

            # Skip if either side has no valid quotes
            if m_bid <= 0 or m_ask <= 0 or b_bid <= 0 or b_ask <= 0:
                continue

            # 4. Volume filter — use Binance volume (MEXC bookTicker lacks it)
            b_volume = b.get('volume_24h', Decimal('0'))
            b_last = b.get('last_price', Decimal('0'))
            volume_usdt = b_volume * b_last if b_last > 0 else Decimal('0')
            if volume_usdt < self.min_volume_usdt:
                continue

            # 5. Compute gross spread in both directions
            # Direction 1: buy MEXC (ask), sell Binance (bid)
            spread_1 = ((b_bid - m_ask) / m_ask) * 10000 if m_ask > 0 else Decimal('0')
            # Direction 2: buy Binance (ask), sell MEXC (bid)
            spread_2 = ((m_bid - b_ask) / b_ask) * 10000 if b_ask > 0 else Decimal('0')

            best_spread = max(spread_1, spread_2)

            candidates.append({
                'pair': pair,
                'best_spread_bps': best_spread,
                'volume_usdt': volume_usdt,
                'direction': 'buy_mexc_sell_binance' if spread_1 >= spread_2 else 'buy_binance_sell_mexc',
            })

        # 6. Rank by best spread, filter by threshold
        above_threshold = [c for c in candidates if c['best_spread_bps'] >= self.min_gross_spread_bps]
        above_threshold.sort(key=lambda c: c['best_spread_bps'], reverse=True)

        # 7. Determine promotion set (top N)
        promotion_set = {c['pair'] for c in above_threshold[:self.max_active_pairs]}

        # 8. Promote new pairs
        newly_promoted = promotion_set - self._promoted_pairs
        for pair in newly_promoted:
            added = await self.book_manager.add_pair(pair)
            if added:
                logger.info(f"PAIR_DISCOVERY: PROMOTED {pair} (spread={self._find_spread(pair, above_threshold):.1f}bps)")
                # Reset grace counter
                self._grace_tracker.pop(pair, None)
                # Trigger contract verification in background (non-blocking)
                if self.contract_verifier:
                    asyncio.create_task(self._verify_pair(pair))

        # 9. Demote with grace period
        currently_active = set(self._promoted_pairs)
        newly_demoted: List[str] = []
        for pair in currently_active:
            if pair not in promotion_set:
                # Increment grace counter
                self._grace_tracker[pair] = self._grace_tracker.get(pair, 0) + 1
                if self._grace_tracker[pair] >= self.demotion_grace_scans:
                    removed = await self.book_manager.remove_pair(pair)
                    if removed:
                        newly_demoted.append(pair)
                        logger.info(f"PAIR_DISCOVERY: DEMOTED {pair} (below threshold for {self.demotion_grace_scans} scans)")
                    self._grace_tracker.pop(pair, None)
            else:
                # Back above threshold — reset grace
                self._grace_tracker.pop(pair, None)

        # Update promoted set
        self._promoted_pairs = (self._promoted_pairs | newly_promoted) - set(newly_demoted)

        # 10. Stats
        elapsed = time.monotonic() - t0
        self._scan_count += 1
        self._stats.update({
            'total_overlapping': len(overlapping),
            'above_threshold': len(above_threshold),
            'promoted': len(newly_promoted),
            'demoted': len(newly_demoted),
            'last_scan_time': round(elapsed, 2),
            'last_scan_ts': int(time.time()),
            'scan_count': self._scan_count,
            'active_discovered_pairs': len(self._promoted_pairs),
        })

        logger.info(
            f"PAIR_DISCOVERY: [{self._scan_count}] "
            f"{len(overlapping)} overlapping, "
            f"{len(above_threshold)} above {float(self.min_gross_spread_bps)}bps, "
            f"+{len(newly_promoted)} promoted, -{len(newly_demoted)} demoted, "
            f"{len(self._promoted_pairs)} active | "
            f"{elapsed:.1f}s"
        )

        # Log top 5 spreads for visibility
        if above_threshold and self._scan_count <= 3 or self._scan_count % 10 == 0:
            top5 = above_threshold[:5]
            for c in top5:
                logger.info(
                    f"  TOP: {c['pair']:15s} spread={float(c['best_spread_bps']):6.1f}bps "
                    f"vol=${float(c['volume_usdt']):>12,.0f} {c['direction']}"
                )

    async def _fetch_mexc_tickers_direct(self) -> Dict[str, dict]:
        """Fetch all MEXC tickers via direct REST (bypasses ccxt which fails on VPS)."""
        url = "https://api.mexc.com/api/v3/ticker/bookTicker"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status != 200:
                        logger.warning(f"PAIR_DISCOVERY: MEXC bookTicker HTTP {resp.status}")
                        return {}
                    data = await resp.json()
        except Exception as e:
            logger.warning(f"PAIR_DISCOVERY: MEXC direct REST failed: {type(e).__name__}: {e}")
            return {}

        result: Dict[str, dict] = {}
        for t in data:
            raw_sym = t.get('symbol', '')
            # Normalize: BTCUSDT → BTC/USDT
            symbol = self._normalize_symbol(raw_sym)
            if not symbol:
                continue
            try:
                bid = Decimal(str(t.get('bidPrice', '0') or '0'))
                ask = Decimal(str(t.get('askPrice', '0') or '0'))
            except (InvalidOperation, ValueError):
                continue
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else Decimal('0')
            result[symbol] = {
                'symbol': symbol,
                'last_price': mid,
                'bid': bid,
                'ask': ask,
                'volume_24h': Decimal('0'),  # bookTicker doesn't include volume
            }
        return result

    async def _fetch_binance_tickers_direct(self) -> Dict[str, dict]:
        """Fetch all Binance tickers via direct REST (bypasses ccxt)."""
        url = "https://api.binance.com/api/v3/ticker/24hr"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        logger.warning(f"PAIR_DISCOVERY: Binance 24hr HTTP {resp.status}")
                        return {}
                    data = await resp.json()
        except Exception as e:
            logger.warning(f"PAIR_DISCOVERY: Binance direct REST failed: {type(e).__name__}: {e}")
            return {}

        result: Dict[str, dict] = {}
        for t in data:
            raw_sym = t.get('symbol', '')
            symbol = self._normalize_symbol(raw_sym)
            if not symbol:
                continue
            try:
                bid = Decimal(str(t.get('bidPrice', '0') or '0'))
                ask = Decimal(str(t.get('askPrice', '0') or '0'))
                last = Decimal(str(t.get('lastPrice', '0') or '0'))
                volume = Decimal(str(t.get('volume', '0') or '0'))
            except (InvalidOperation, ValueError):
                continue
            result[symbol] = {
                'symbol': symbol,
                'last_price': last,
                'bid': bid,
                'ask': ask,
                'volume_24h': volume,
            }
        return result

    @staticmethod
    def _normalize_symbol(raw: str) -> Optional[str]:
        """Convert exchange symbol (BTCUSDT) to normalized format (BTC/USDT)."""
        raw = raw.upper()
        for quote in ('USDT', 'USDC', 'BTC', 'ETH'):
            if raw.endswith(quote):
                base = raw[:-len(quote)]
                if base:
                    return f"{base}/{quote}"
        return None

    async def _verify_pair(self, pair: str) -> None:
        """Background contract verification for a newly promoted pair."""
        try:
            verified = self.contract_verifier.is_verified(pair)
            if not verified:
                logger.info(f"PAIR_DISCOVERY: {pair} failed contract verification (will be skipped by detector)")
        except Exception as e:
            logger.debug(f"PAIR_DISCOVERY: Contract verify error for {pair}: {e}")

    def _find_spread(self, pair: str, candidates: List[dict]) -> float:
        """Find the spread for a pair in the candidates list."""
        for c in candidates:
            if c['pair'] == pair:
                return float(c['best_spread_bps'])
        return 0.0

    def get_stats(self) -> dict:
        """Return discovery statistics for dashboard/monitoring."""
        return {
            **self._stats,
            'promoted_pairs': sorted(self._promoted_pairs),
            'grace_pending': {p: c for p, c in self._grace_tracker.items()},
        }
