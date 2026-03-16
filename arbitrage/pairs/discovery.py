"""
Pair Discovery — Scans ALL MEXC spot pairs and scores them for
arbitrage opportunity quality.

Runs every 6 hours. Fetches exchange info + 24h tickers in 2 API calls.
Scores each pair on:
  - 24h volume (liquidity proxy)
  - Bid-ask spread (cost proxy)
  - Quote currency (USDT preferred, BTC/ETH for triangular)
  - Binance cross-listing (enables cross-exchange arb)
  - Historical arb profitability (from temporal analyzer, if available)

Output: ranked list of pairs with scores and tier assignments.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger("arb.discovery")


@dataclass
class PairScore:
    """Scoring result for a single trading pair."""
    symbol: str                  # e.g. "BTC/USDT"
    base_currency: str           # e.g. "BTC"
    quote_currency: str          # e.g. "USDT"
    exchange: str                # "mexc"

    # Raw metrics
    volume_24h_usd: float = 0.0
    bid_ask_spread_bps: float = 0.0
    last_price: float = 0.0
    price_change_pct_24h: float = 0.0
    num_trades_24h: int = 0

    # Cross-listing
    on_binance: bool = False

    # Triangular connectivity
    triangular_paths: int = 0

    # Composite score (0-100)
    score: float = 0.0

    # Tier assignment
    tier: int = 3                # 1=full depth, 2=ticker, 3=triangular-only

    # Historical performance (from TemporalAnalyzer)
    historical_avg_profit: Optional[float] = None
    historical_trade_count: int = 0


class MexcPairDiscovery:
    """
    Discovers and scores all available pairs on MEXC for triangular arb expansion.

    Uses 2 API calls:
    1. load_markets() — all pairs with trading rules (cached by ccxt)
    2. fetch_tickers() — all 24h ticker stats in ONE call
    """

    # Scoring weights
    WEIGHT_VOLUME = 0.30
    WEIGHT_SPREAD = 0.25
    WEIGHT_CROSS_LISTED = 0.20
    WEIGHT_TRIANGULAR = 0.15
    WEIGHT_HISTORICAL = 0.10

    # Minimum thresholds
    MIN_VOLUME_24H_USD = 50_000
    MIN_TRADES_24H = 100
    MAX_SPREAD_BPS = 50
    ALLOWED_QUOTE_CURRENCIES = {"USDT", "BTC", "ETH", "USDC"}

    # Tier thresholds
    TIER_1_MIN_SCORE = 70
    TIER_2_MIN_SCORE = 50
    TIER_1_MAX = 10
    TIER_2_MAX = 20
    TIER_3_MAX = 70

    REFRESH_INTERVAL_HOURS = 6

    def __init__(
        self,
        mexc_client,
        binance_client=None,
        temporal_analyzer=None,
        config: Optional[dict] = None,
    ):
        self.mexc = mexc_client
        self.binance = binance_client
        self.temporal_analyzer = temporal_analyzer
        self._last_scan: Optional[datetime] = None
        self._scores: Dict[str, PairScore] = {}
        self._binance_symbols: Set[str] = set()

        # Apply config overrides
        if config:
            pd_cfg = config.get('pair_expansion', {})
            self.MIN_VOLUME_24H_USD = pd_cfg.get('min_volume_24h_usd', self.MIN_VOLUME_24H_USD)
            self.MIN_TRADES_24H = pd_cfg.get('min_trades_24h', self.MIN_TRADES_24H)
            self.MAX_SPREAD_BPS = pd_cfg.get('max_spread_bps', self.MAX_SPREAD_BPS)
            self.TIER_1_MAX = pd_cfg.get('tier_1_max', self.TIER_1_MAX)
            self.TIER_2_MAX = pd_cfg.get('tier_2_max', self.TIER_2_MAX)
            self.TIER_3_MAX = pd_cfg.get('tier_3_max', self.TIER_3_MAX)
            self.REFRESH_INTERVAL_HOURS = pd_cfg.get('rescan_interval_hours', self.REFRESH_INTERVAL_HOURS)
            if 'allowed_quote_currencies' in pd_cfg:
                self.ALLOWED_QUOTE_CURRENCIES = set(pd_cfg['allowed_quote_currencies'])

    async def scan(self) -> List[PairScore]:
        """Full pair discovery scan."""
        logger.info("Starting MEXC pair discovery scan...")

        # Step 1: Get all MEXC markets
        mexc_markets = getattr(self.mexc, '_exchange', self.mexc)
        if hasattr(mexc_markets, 'markets'):
            markets = mexc_markets.markets
            if not markets:
                try:
                    await mexc_markets.load_markets()
                    markets = mexc_markets.markets
                except Exception as e:
                    logger.error("Failed to load MEXC markets: %s", e)
                    return []
        else:
            logger.error("Cannot access MEXC markets")
            return []

        logger.info("MEXC has %d total markets", len(markets))

        # Step 2: Fetch all tickers (single API call)
        try:
            if hasattr(mexc_markets, 'fetch_tickers'):
                all_tickers = await mexc_markets.fetch_tickers()
            else:
                all_tickers = await self.mexc.fetch_all_tickers()
        except Exception as e:
            logger.error("Failed to fetch MEXC tickers: %s", e)
            return []

        # Step 3: Get Binance symbol list
        await self._load_binance_symbols()

        # Step 4: Score each pair
        scored: List[PairScore] = []
        triangular_graph = self._build_triangular_graph(markets)

        for symbol, market in markets.items():
            if not market.get('active', False):
                continue
            if market.get('type') != 'spot':
                continue

            base = market.get('base', '')
            quote = market.get('quote', '')

            if quote not in self.ALLOWED_QUOTE_CURRENCIES:
                continue

            ticker = all_tickers.get(symbol, {})
            volume_usd = float(ticker.get('quoteVolume', 0) or 0)
            num_trades = int(ticker.get('info', {}).get('count', 0) or 0)
            bid = float(ticker.get('bid', 0) or 0)
            ask = float(ticker.get('ask', 0) or 0)
            last = float(ticker.get('last', 0) or 0)
            change_pct = float(ticker.get('percentage', 0) or 0)

            if volume_usd < self.MIN_VOLUME_24H_USD:
                continue
            if num_trades < self.MIN_TRADES_24H:
                continue

            if bid > 0 and ask > 0:
                spread_bps = ((ask - bid) / ((ask + bid) / 2)) * 10000
            else:
                spread_bps = 999

            if spread_bps > self.MAX_SPREAD_BPS:
                continue

            binance_symbol = f"{base}/{quote}"
            on_binance = binance_symbol in self._binance_symbols

            tri_paths = triangular_graph.get(symbol, 0)

            # Historical performance from temporal analyzer
            hist_profit = None
            hist_trades = 0
            if self.temporal_analyzer:
                for strategy in ['cross_exchange', 'triangular']:
                    profile = self.temporal_analyzer._profiles.get(f"{strategy}:{symbol}")
                    if profile:
                        total = sum(b.total_profit_usd for b in profile.by_hour.values())
                        total_t = sum(b.total_trades for b in profile.by_hour.values())
                        if hist_profit is None:
                            hist_profit = total
                            hist_trades = total_t
                        else:
                            hist_profit += total
                            hist_trades += total_t

            pair_score = PairScore(
                symbol=symbol,
                base_currency=base,
                quote_currency=quote,
                exchange="mexc",
                volume_24h_usd=volume_usd,
                bid_ask_spread_bps=spread_bps,
                last_price=last,
                price_change_pct_24h=change_pct,
                num_trades_24h=num_trades,
                on_binance=on_binance,
                triangular_paths=tri_paths,
                historical_avg_profit=hist_profit,
                historical_trade_count=hist_trades,
            )

            pair_score.score = self._compute_score(pair_score)
            scored.append(pair_score)

        # Step 5: Rank and assign tiers
        scored.sort(key=lambda p: p.score, reverse=True)
        self._assign_tiers(scored)

        self._scores = {p.symbol: p for p in scored}
        self._last_scan = datetime.utcnow()

        tier_counts = {1: 0, 2: 0, 3: 0}
        for p in scored:
            if p.tier in tier_counts:
                tier_counts[p.tier] += 1

        logger.info(
            "Pair discovery complete — %d eligible pairs. "
            "Tier 1: %d, Tier 2: %d, Tier 3: %d",
            len(scored), tier_counts[1], tier_counts[2], tier_counts[3],
        )

        return scored

    def _compute_score(self, pair: PairScore) -> float:
        """Compute composite score (0-100) for a pair."""
        vol_log = math.log10(max(pair.volume_24h_usd, 1))
        vol_score = min(100, max(0, (vol_log - 4.7) / (7 - 4.7) * 100))

        spread_score = max(0, 100 - (pair.bid_ask_spread_bps / 50 * 100))
        cross_score = 100 if pair.on_binance else 0
        tri_score = min(100, pair.triangular_paths * 5)

        if pair.historical_avg_profit is not None and pair.historical_trade_count >= 10:
            hist_score = min(100, pair.historical_avg_profit * 1000) if pair.historical_avg_profit > 0 else 0
        else:
            hist_score = 50

        composite = (
            vol_score * self.WEIGHT_VOLUME
            + spread_score * self.WEIGHT_SPREAD
            + cross_score * self.WEIGHT_CROSS_LISTED
            + tri_score * self.WEIGHT_TRIANGULAR
            + hist_score * self.WEIGHT_HISTORICAL
        )
        return round(composite, 2)

    def _assign_tiers(self, scored: List[PairScore]):
        """Assign tiers based on score and constraints."""
        t1, t2, t3 = 0, 0, 0
        for pair in scored:
            if pair.score >= self.TIER_1_MIN_SCORE and t1 < self.TIER_1_MAX:
                pair.tier = 1
                t1 += 1
            elif pair.score >= self.TIER_2_MIN_SCORE and t2 < self.TIER_2_MAX:
                pair.tier = 2
                t2 += 1
            elif t3 < self.TIER_3_MAX:
                pair.tier = 3
                t3 += 1
            else:
                pair.tier = 0

    def _build_triangular_graph(self, markets: dict) -> Dict[str, int]:
        """Count how many triangular paths include each pair."""
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        pair_lookup: Dict[Tuple[str, str], str] = {}

        for symbol, market in markets.items():
            if not market.get('active', False) or market.get('type') != 'spot':
                continue
            base = market.get('base', '')
            quote = market.get('quote', '')
            if not base or not quote:
                continue
            adjacency[base].add(quote)
            adjacency[quote].add(base)
            pair_lookup[(base, quote)] = symbol
            pair_lookup[(quote, base)] = symbol

        path_counts: Dict[str, int] = {}
        start_currencies = {"USDT", "BTC", "ETH"}

        for symbol, market in markets.items():
            if not market.get('active', False):
                continue
            base = market.get('base', '')
            quote = market.get('quote', '')
            count = 0
            for start in start_currencies:
                if (start, base) in pair_lookup and (quote, start) in pair_lookup:
                    count += 1
                if (start, quote) in pair_lookup and (base, start) in pair_lookup:
                    count += 1
            path_counts[symbol] = count

        return path_counts

    async def _load_binance_symbols(self):
        """Load Binance symbol list for cross-listing check."""
        if self._binance_symbols:
            return
        try:
            if self.binance:
                exchange = getattr(self.binance, '_exchange', self.binance)
                if hasattr(exchange, 'markets'):
                    if not exchange.markets:
                        await exchange.load_markets()
                    self._binance_symbols = set(exchange.markets.keys())
                    logger.info(
                        "Loaded %d Binance symbols for cross-listing check",
                        len(self._binance_symbols),
                    )
        except Exception as e:
            logger.warning("Could not load Binance symbols: %s", e)
            self._binance_symbols = set()

    async def maybe_rescan(self) -> Optional[List[PairScore]]:
        """Rescan if enough time has passed."""
        if self._last_scan is None:
            return await self.scan()
        elapsed_hours = (datetime.utcnow() - self._last_scan).total_seconds() / 3600
        if elapsed_hours >= self.REFRESH_INTERVAL_HOURS:
            return await self.scan()
        return None

    def get_tier_pairs(self, tier: int) -> List[str]:
        return [p.symbol for p in self._scores.values() if p.tier == tier]

    def get_all_active_pairs(self) -> List[str]:
        return [p.symbol for p in self._scores.values() if p.tier in (1, 2, 3)]

    def get_cross_exchange_eligible(self) -> List[str]:
        return [p.symbol for p in self._scores.values() if p.on_binance and p.tier in (1, 2)]

    def get_report(self) -> dict:
        if not self._scores:
            return {"status": "no_scan_yet"}

        return {
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "total_eligible": len(self._scores),
            "tier_1": [
                {"symbol": p.symbol, "score": p.score, "volume_24h": p.volume_24h_usd,
                 "spread_bps": round(p.bid_ask_spread_bps, 1), "on_binance": p.on_binance,
                 "triangular_paths": p.triangular_paths}
                for p in sorted(self._scores.values(), key=lambda x: x.score, reverse=True)
                if p.tier == 1
            ],
            "tier_2": [
                {"symbol": p.symbol, "score": p.score, "volume_24h": p.volume_24h_usd,
                 "spread_bps": round(p.bid_ask_spread_bps, 1), "on_binance": p.on_binance}
                for p in sorted(self._scores.values(), key=lambda x: x.score, reverse=True)
                if p.tier == 2
            ],
            "tier_3_count": sum(1 for p in self._scores.values() if p.tier == 3),
            "top_triangular": [
                {"symbol": p.symbol, "paths": p.triangular_paths, "score": p.score}
                for p in sorted(self._scores.values(), key=lambda x: x.triangular_paths, reverse=True)[:10]
            ],
        }
