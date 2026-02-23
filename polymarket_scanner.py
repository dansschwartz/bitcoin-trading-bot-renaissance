"""
Polymarket Scanner — Discovers and classifies ALL active crypto prediction markets.

Fetches from Gamma API, classifies by type (DIRECTION, THRESHOLD, RANGE, HIT_PRICE,
VOLATILITY, OTHER), computes edges using Renaissance ML ensemble predictions, and
persists to SQLite for the bridge/dashboard to consume.

Architecture:
  Gamma API → paginated fetch → classify → edge detect → persist to polymarket_scanner table
  Scanner runs every 5 minutes (not every pair cycle).
"""

import asyncio
import logging
import math
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gamma API
# ---------------------------------------------------------------------------
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PolymarketMarket:
    """A classified Polymarket prediction market."""
    condition_id: str
    question: str
    slug: str
    market_type: str            # DIRECTION|THRESHOLD|RANGE|HIT_PRICE|VOLATILITY|OTHER
    asset: Optional[str]        # BTC, ETH, SOL, etc.
    timeframe_minutes: Optional[int]
    deadline: Optional[str]     # ISO timestamp
    target_price: Optional[float]
    range_low: Optional[float]
    range_high: Optional[float]
    yes_price: float
    no_price: float
    volume_24h: float
    liquidity: float
    active: bool
    fetched_at: str             # ISO timestamp


@dataclass
class PolymarketOpportunity:
    """An edge-detected opportunity on a Polymarket market."""
    market: PolymarketMarket
    direction: str              # UP or DOWN
    our_probability: float
    market_probability: float
    edge: float
    confidence: float
    source: str                 # "ml_ensemble" or "statistical"


# ---------------------------------------------------------------------------
# Classification engine
# ---------------------------------------------------------------------------

# Regex patterns: (compiled_regex, market_type)
# Applied to lowercased question text
_DIRECTION_RE = re.compile(
    r'(?:go\s+)?(up|down|increase|decrease|rise|fall|higher|lower)'
    r'.*?(?:next|in)\s+(\d+)\s*(m(?:in(?:ute)?s?)?|h(?:(?:ou)?rs?)?|d(?:ays?)?)',
    re.IGNORECASE,
)
_THRESHOLD_RE = re.compile(
    r'(above|below|over|under)\s+\$?([\d,]+(?:\.\d+)?)\s*([kKmM](?![a-z])|[bB](?!y\b))?',
    re.IGNORECASE,
)
_HIT_PRICE_RE = re.compile(
    r'(?:hit|reach|touch|break)\s+\$?([\d,]+(?:\.\d+)?)\s*([kKmM](?![a-z])|[bB](?!y\b))?',
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r'between\s+\$?([\d,]+(?:\.\d+)?)\s*([kKmMbB])?\s*(?:and|-)\s*\$?([\d,]+(?:\.\d+)?)\s*([kKmMbB])?',
    re.IGNORECASE,
)
_VOLATILITY_RE = re.compile(
    r'(?:move|change|swing|fluctuate)\s+(?:more|less)\s+than\s+(\d+(?:\.\d+)?)\s*%',
    re.IGNORECASE,
)

# Slug-based direction patterns (many Polymarket slugs encode the market type)
_SLUG_DIRECTION_RE = re.compile(
    r'(btc|eth|sol|bitcoin|ethereum|solana|doge|xrp|ada|avax|link|matic|dot|bnb|sui|apt|arb|op)-'
    r'(?:updown|up-or-down|price)-(\d+)(m|min|h|hr|d)',
    re.IGNORECASE,
)

CRYPTO_ASSETS: Dict[str, str] = {
    'btc': 'BTC', 'bitcoin': 'BTC',
    'eth': 'ETH', 'ethereum': 'ETH',
    'sol': 'SOL', 'solana': 'SOL',
    'doge': 'DOGE', 'dogecoin': 'DOGE',
    'xrp': 'XRP', 'ripple': 'XRP',
    'ada': 'ADA', 'cardano': 'ADA',
    'avax': 'AVAX', 'avalanche': 'AVAX',
    'link': 'LINK', 'chainlink': 'LINK',
    'matic': 'MATIC', 'polygon': 'MATIC',
    'dot': 'DOT', 'polkadot': 'DOT',
    'bnb': 'BNB', 'binance coin': 'BNB',
    'sui': 'SUI',
    'apt': 'APT', 'aptos': 'APT',
    'arb': 'ARB', 'arbitrum': 'ARB',
    'op': 'OP', 'optimism': 'OP',
    'ltc': 'LTC', 'litecoin': 'LTC',
    'atom': 'ATOM', 'cosmos': 'ATOM',
    'near': 'NEAR',
    'fil': 'FIL', 'filecoin': 'FIL',
    'uni': 'UNI', 'uniswap': 'UNI',
    'aave': 'AAVE',
    'mkr': 'MKR', 'maker': 'MKR',
    'pepe': 'PEPE',
    'shib': 'SHIB', 'shiba': 'SHIB',
    'wif': 'WIF',
    'bonk': 'BONK',
    'render': 'RENDER', 'rndr': 'RENDER',
    'injective': 'INJ', 'inj': 'INJ',
    'sei': 'SEI',
    'tia': 'TIA', 'celestia': 'TIA',
    'jup': 'JUP', 'jupiter': 'JUP',
    'crypto': None,  # generic crypto keyword, no specific asset
}

# Keywords that indicate a market is crypto-related
_CRYPTO_KEYWORDS = re.compile(
    r'\b(?:' + '|'.join(k for k in CRYPTO_ASSETS if k != 'crypto' and len(k) >= 3) +
    r'|bitcoin|ethereum|solana|crypto(?:currency)?|btc|eth|sol|doge|xrp'
    r'|defi|nft|blockchain|altcoin|token|coin)\b',
    re.IGNORECASE,
)

# False-positive blocklist — sports teams, companies, etc. that match crypto tickers
_FALSE_POSITIVE_RE = re.compile(
    r'\b(?:avalanche|nuggets|broncos|rapids)\b.*\b(?:nhl|nba|nfl|mls|stanley\s+cup|'
    r'playoffs|division|conference|championship|season|trophy|league)\b'
    r'|\b(?:nhl|nba|nfl|mls|stanley\s+cup|playoffs|division|conference|championship|'
    r'season|trophy|league)\b.*\b(?:avalanche|nuggets|broncos|rapids)\b',
    re.IGNORECASE,
)


def _parse_price_suffix(value_str: str, suffix: Optional[str]) -> float:
    """Parse a numeric string with optional k/m/b suffix."""
    v = float(value_str.replace(',', ''))
    if suffix:
        s = suffix.lower()
        if s == 'k':
            v *= 1_000
        elif s == 'm':
            v *= 1_000_000
        elif s == 'b':
            v *= 1_000_000_000
    return v


def _parse_timeframe(amount: str, unit: str) -> int:
    """Convert amount+unit to minutes."""
    n = int(amount)
    u = unit.lower().rstrip('s')
    if u.startswith('m'):
        return n
    elif u.startswith('h'):
        return n * 60
    elif u.startswith('d'):
        return n * 1440
    return n


def _extract_asset(text: str) -> Optional[str]:
    """Extract crypto asset symbol from text."""
    lower = text.lower()
    for keyword, symbol in CRYPTO_ASSETS.items():
        if symbol and re.search(r'\b' + re.escape(keyword) + r'\b', lower):
            return symbol
    return None


def classify_market(raw: dict) -> Optional[PolymarketMarket]:
    """
    Classify a raw Gamma API market dict into a PolymarketMarket.
    Returns None if the market is not crypto-relevant.
    """
    question = raw.get('question', '') or ''
    slug = raw.get('slug', '') or ''
    condition_id = raw.get('conditionId') or raw.get('condition_id', '')

    if not condition_id:
        return None

    combined = f"{question} {slug}"

    # Must be crypto-relevant
    if not _CRYPTO_KEYWORDS.search(combined):
        return None

    # Filter out false positives (sports teams matching crypto tickers)
    if _FALSE_POSITIVE_RE.search(combined):
        return None

    asset = _extract_asset(combined)
    market_type = 'OTHER'
    timeframe_minutes: Optional[int] = None
    target_price: Optional[float] = None
    range_low: Optional[float] = None
    range_high: Optional[float] = None

    # Try slug-based direction first (most reliable)
    slug_m = _SLUG_DIRECTION_RE.search(slug)
    if slug_m:
        market_type = 'DIRECTION'
        slug_asset_key = slug_m.group(1).lower()
        if slug_asset_key in CRYPTO_ASSETS and CRYPTO_ASSETS[slug_asset_key]:
            asset = CRYPTO_ASSETS[slug_asset_key]
        timeframe_minutes = _parse_timeframe(slug_m.group(2), slug_m.group(3))
    else:
        # Try question-based patterns in priority order
        m = _DIRECTION_RE.search(question)
        if m:
            market_type = 'DIRECTION'
            timeframe_minutes = _parse_timeframe(m.group(2), m.group(3))
        else:
            m = _RANGE_RE.search(question)
            if m:
                market_type = 'RANGE'
                range_low = _parse_price_suffix(m.group(1), m.group(2))
                range_high = _parse_price_suffix(m.group(3), m.group(4))
            else:
                m = _THRESHOLD_RE.search(question)
                if m:
                    market_type = 'THRESHOLD'
                    target_price = _parse_price_suffix(m.group(2), m.group(3))
                else:
                    m = _HIT_PRICE_RE.search(question)
                    if m:
                        market_type = 'HIT_PRICE'
                        target_price = _parse_price_suffix(m.group(1), m.group(2))
                    else:
                        m = _VOLATILITY_RE.search(question)
                        if m:
                            market_type = 'VOLATILITY'

    # Extract deadline from endDate or expirationDate
    deadline = raw.get('endDate') or raw.get('expirationDate')

    # Extract prices — Gamma API may nest these differently
    yes_price = 0.0
    no_price = 0.0
    try:
        # outcomePrices is typically a JSON string like "[\"0.55\",\"0.45\"]"
        outcome_prices = raw.get('outcomePrices')
        if isinstance(outcome_prices, str):
            import json
            prices = json.loads(outcome_prices)
            if len(prices) >= 2:
                yes_price = float(prices[0])
                no_price = float(prices[1])
        elif isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
            yes_price = float(outcome_prices[0])
            no_price = float(outcome_prices[1])
    except (ValueError, TypeError, IndexError):
        pass

    # If no outcomePrices, try bestBid/bestAsk or other fields
    if yes_price == 0.0:
        yes_price = _safe_float(raw.get('bestBid', 0))
        no_price = 1.0 - yes_price if yes_price > 0 else 0.0

    volume_24h = _safe_float(raw.get('volume24hr', 0)) or _safe_float(raw.get('volume', 0))
    liquidity = _safe_float(raw.get('liquidity', 0))

    return PolymarketMarket(
        condition_id=condition_id,
        question=question,
        slug=slug,
        market_type=market_type,
        asset=asset,
        timeframe_minutes=timeframe_minutes,
        deadline=deadline,
        target_price=target_price,
        range_low=range_low,
        range_high=range_high,
        yes_price=yes_price,
        no_price=no_price,
        volume_24h=volume_24h,
        liquidity=liquidity,
        active=bool(raw.get('active', True)),
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def _safe_float(v: Any) -> float:
    """Safely convert to float, returning 0.0 on failure."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# GBM probability model for HIT_PRICE edge detection
# ---------------------------------------------------------------------------

# Annualized volatility defaults (conservative estimates)
DEFAULT_ANNUAL_VOL: Dict[str, float] = {
    'BTC': 0.60,
    'ETH': 0.80,
    'SOL': 1.00,
    'DOGE': 1.20,
    'XRP': 1.00,
    'BNB': 0.75,
    'ADA': 1.00,
    'AVAX': 1.10,
    'LINK': 1.00,
    'DOT': 1.00,
    'UNI': 1.10,
    'BONK': 1.50,
}
_DEFAULT_VOL_FALLBACK = 1.00  # For assets not in the table


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _gbm_hit_probability(
    current_price: float,
    target_price: float,
    T_years: float,
    sigma: float,
    mu: float = 0.0,
) -> float:
    """
    Probability that price reaches target by time T under geometric Brownian motion.

    For target > current (upside): P(S_T >= K) = N(d2)
    For target < current (downside): P(S_T <= K) = N(-d2)

    Where d2 = [ln(S/K) + (mu - sigma^2/2)*T] / (sigma * sqrt(T))

    Args:
        current_price: Current asset price
        target_price: Target price to hit
        T_years: Time to deadline in years
        sigma: Annualized volatility (e.g. 0.60 for 60%)
        mu: Annualized drift (informed by ML prediction)

    Returns:
        Probability between 0 and 1.
    """
    if current_price <= 0 or target_price <= 0 or T_years <= 0 or sigma <= 0:
        return 0.0

    # Already at or past the target
    if target_price <= current_price:
        return 0.95  # Near-certain but not 1.0 (market could reverse)

    ln_ratio = math.log(current_price / target_price)
    drift_term = (mu - 0.5 * sigma * sigma) * T_years
    vol_term = sigma * math.sqrt(T_years)

    d2 = (ln_ratio + drift_term) / vol_term
    prob = _norm_cdf(d2)

    # Clamp to reasonable range
    return max(0.01, min(0.99, prob))


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class PolymarketScanner:
    """
    Discovers and classifies all active crypto prediction markets on Polymarket.
    Computes edges using ML ensemble predictions for DIRECTION markets.
    Persists scan results to SQLite for bridge/dashboard consumption.
    """

    def __init__(
        self,
        db_path: str = "data/renaissance_bot.db",
        cache_ttl: int = 300,
        logger: Optional[logging.Logger] = None,
    ):
        self.db_path = db_path
        self.cache_ttl = cache_ttl
        self.logger = logger or logging.getLogger(__name__)
        self._cached_markets: List[dict] = []
        self._cache_time: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._supported_assets = {'BTC', 'ETH', 'SOL'}  # Phase 1: ML edge for these
        self._last_scan_stats: Dict[str, Any] = {}
        self._init_db()

    def _init_db(self) -> None:
        """Create the polymarket_scanner table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS polymarket_scanner (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_id TEXT NOT NULL,
                question TEXT,
                slug TEXT,
                market_type TEXT NOT NULL,
                asset TEXT,
                timeframe_minutes INTEGER,
                deadline TEXT,
                target_price REAL,
                range_low REAL,
                range_high REAL,
                yes_price REAL,
                no_price REAL,
                volume_24h REAL,
                liquidity REAL,
                edge REAL,
                our_probability REAL,
                direction TEXT,
                confidence REAL,
                scan_time TEXT NOT NULL,
                UNIQUE(condition_id, scan_time)
            );

            CREATE INDEX IF NOT EXISTS idx_scanner_type ON polymarket_scanner(market_type);
            CREATE INDEX IF NOT EXISTS idx_scanner_asset ON polymarket_scanner(asset);
            CREATE INDEX IF NOT EXISTS idx_scanner_edge ON polymarket_scanner(edge);
            CREATE INDEX IF NOT EXISTS idx_scanner_time ON polymarket_scanner(scan_time);
        """)
        conn.close()
        self.logger.info("PolymarketScanner: DB table initialized")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _cache_valid(self) -> bool:
        """Check if cached markets are still fresh."""
        return (time.time() - self._cache_time) < self.cache_ttl and len(self._cached_markets) > 0

    async def fetch_markets(self) -> List[dict]:
        """Paginated fetch of all active markets from Gamma API, with 5-min cache."""
        if self._cache_valid():
            self.logger.debug(f"PolymarketScanner: using cached {len(self._cached_markets)} markets")
            return self._cached_markets

        session = await self._ensure_session()
        all_markets: List[dict] = []
        offset = 0
        limit = 100
        max_pages = 50  # Safety limit: 50 * 100 = 5000 markets max

        for _ in range(max_pages):
            params = {
                "limit": str(limit),
                "offset": str(offset),
                "active": "true",
                "closed": "false",
            }
            try:
                async with session.get(GAMMA_MARKETS_URL, params=params) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"Gamma API HTTP {resp.status} at offset {offset}")
                        break
                    page = await resp.json()
            except Exception as e:
                self.logger.warning(f"Gamma API fetch error at offset {offset}: {e}")
                break

            if not page or not isinstance(page, list) or len(page) == 0:
                break

            all_markets.extend(page)
            offset += limit

            if len(page) < limit:
                break

            await asyncio.sleep(0.1)  # Rate limiting

        self._cached_markets = all_markets
        self._cache_time = time.time()
        self.logger.info(f"PolymarketScanner: fetched {len(all_markets)} total markets from Gamma API")
        return all_markets

    def compute_edge(
        self,
        market: PolymarketMarket,
        ml_prediction: float,
        agreement: float,
        regime: str,
        current_price: float = 0.0,
    ) -> Optional[PolymarketOpportunity]:
        """Compute edge for a market using ML predictions.

        Dispatches to type-specific edge methods:
        - DIRECTION: ML prediction → probability comparison
        - HIT_PRICE: GBM probability model with ML-informed drift
        """
        if not market.asset or market.asset not in self._supported_assets:
            return None
        if market.yes_price <= 0.01 or market.yes_price >= 0.99:
            return None  # Illiquid / already resolved

        if market.market_type == 'DIRECTION':
            return self._compute_edge_direction(market, ml_prediction, agreement, regime)
        elif market.market_type == 'HIT_PRICE':
            return self._compute_edge_hit_price(market, ml_prediction, agreement, regime, current_price)
        return None

    def _compute_edge_direction(
        self,
        market: PolymarketMarket,
        ml_prediction: float,
        agreement: float,
        regime: str,
    ) -> Optional[PolymarketOpportunity]:
        """Compute edge for a DIRECTION market using ML predictions."""
        # Convert ML prediction to probability via calibrated sigmoid
        raw = abs(ml_prediction)
        our_prob = 0.50 + min(0.30, raw * 2.5)

        # Agreement adjustment
        if agreement >= 0.80:
            our_prob = min(0.85, our_prob * 1.05)
        elif agreement < 0.55:
            our_prob = max(0.50, our_prob * 0.90)

        # Regime alignment
        bullish_regimes = ('bull_trending', 'bull_mean_reverting')
        bearish_regimes = ('bear_trending', 'bear_mean_reverting', 'high_volatility')
        ml_direction = "UP" if ml_prediction > 0 else "DOWN"

        if (regime in bullish_regimes and ml_direction == "UP") or \
           (regime in bearish_regimes and ml_direction == "DOWN"):
            our_prob = min(0.85, our_prob + 0.03)

        # Edge calculation
        market_prob = market.yes_price  # YES = UP
        if ml_direction == "UP":
            edge = our_prob - market_prob
        else:
            edge = our_prob - (1.0 - market_prob)  # Compare against NO price

        min_edge = 0.03  # 3% minimum
        if edge < min_edge:
            return None

        confidence = min(95.0, edge * 400 + agreement * 20)

        return PolymarketOpportunity(
            market=market,
            direction=ml_direction,
            our_probability=round(our_prob, 4),
            market_probability=round(market_prob, 4),
            edge=round(edge, 4),
            confidence=round(confidence, 1),
            source="ml_ensemble",
        )

    def _compute_edge_hit_price(
        self,
        market: PolymarketMarket,
        ml_prediction: float,
        agreement: float,
        regime: str,
        current_price: float,
    ) -> Optional[PolymarketOpportunity]:
        """
        Compute edge for a HIT_PRICE market using GBM probability model.

        Uses log-normal model: P(S_T >= K) = N(d2)
        Drift mu is informed by ML prediction direction + regime.
        Sigma is asset-specific annualized volatility.

        Edge = our_probability - market_yes_price (for YES bets)
        or   = (1 - our_probability) - market_no_price (for NO bets)
        """
        if not market.target_price or market.target_price <= 0:
            return None
        if current_price <= 0:
            return None
        if not market.deadline:
            return None

        # Parse deadline and compute time to expiry in years
        try:
            deadline_dt = datetime.fromisoformat(market.deadline.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            T_seconds = (deadline_dt - now).total_seconds()
            if T_seconds <= 0:
                return None  # Already expired
            T_years = T_seconds / (365.25 * 24 * 3600)
        except (ValueError, TypeError):
            return None

        # Skip very long-dated markets (>2 years) — too uncertain for ML edge
        if T_years > 2.0:
            return None

        # Get asset volatility
        sigma = DEFAULT_ANNUAL_VOL.get(market.asset, _DEFAULT_VOL_FALLBACK)

        # Compute ML-informed drift
        # Base drift: 0 (risk-neutral). ML prediction nudges it.
        # ml_prediction ~0.05 maps to ~15% annualized drift
        mu = float(ml_prediction) * 3.0  # Scale: 0.05 pred -> 15% annual drift

        # Regime adjustment to drift
        bullish_regimes = ('bull_trending', 'bull_mean_reverting')
        bearish_regimes = ('bear_trending', 'bear_mean_reverting', 'high_volatility')
        if regime in bullish_regimes:
            mu += 0.05  # Add 5% annual drift in bull regimes
        elif regime in bearish_regimes:
            mu -= 0.05

        # Agreement scaling: low agreement → pull drift toward zero
        if agreement < 0.55:
            mu *= 0.5
        elif agreement >= 0.80:
            mu *= 1.2

        # GBM probability
        our_prob = _gbm_hit_probability(current_price, market.target_price, T_years, sigma, mu)

        market_prob = market.yes_price  # Market's implied probability of hitting target

        # Determine which side to bet: YES (will hit) or NO (won't hit)
        yes_edge = our_prob - market_prob
        no_edge = (1.0 - our_prob) - (1.0 - market_prob)  # Simplifies to market_prob - our_prob

        if yes_edge >= 0.03:
            # We think it's MORE likely to hit than market does → bet YES
            edge = yes_edge
            direction = "YES"
        elif no_edge >= 0.03:
            # We think it's LESS likely to hit than market does → bet NO
            edge = no_edge
            direction = "NO"
        else:
            return None  # No actionable edge

        confidence = min(95.0, edge * 300 + agreement * 15)

        return PolymarketOpportunity(
            market=market,
            direction=direction,
            our_probability=round(our_prob, 4),
            market_probability=round(market_prob, 4),
            edge=round(edge, 4),
            confidence=round(confidence, 1),
            source="gbm_ml_drift",
        )

    def _persist(
        self,
        classified: List[PolymarketMarket],
        opportunities: List[PolymarketOpportunity],
    ) -> None:
        """Persist scan results to SQLite."""
        scan_time = datetime.now(timezone.utc).isoformat()

        # Build edge lookup from opportunities
        edge_map: Dict[str, PolymarketOpportunity] = {}
        for opp in opportunities:
            edge_map[opp.market.condition_id] = opp

        conn = sqlite3.connect(self.db_path)
        try:
            rows = []
            for m in classified:
                opp = edge_map.get(m.condition_id)
                rows.append((
                    m.condition_id,
                    m.question,
                    m.slug,
                    m.market_type,
                    m.asset,
                    m.timeframe_minutes,
                    m.deadline,
                    m.target_price,
                    m.range_low,
                    m.range_high,
                    m.yes_price,
                    m.no_price,
                    m.volume_24h,
                    m.liquidity,
                    opp.edge if opp else None,
                    opp.our_probability if opp else None,
                    opp.direction if opp else None,
                    opp.confidence if opp else None,
                    scan_time,
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO polymarket_scanner (
                    condition_id, question, slug, market_type, asset,
                    timeframe_minutes, deadline, target_price, range_low, range_high,
                    yes_price, no_price, volume_24h, liquidity,
                    edge, our_probability, direction, confidence, scan_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            self.logger.debug(f"PolymarketScanner: persisted {len(rows)} markets to DB")
        except Exception as e:
            self.logger.error(f"PolymarketScanner: DB persist error: {e}")
        finally:
            conn.close()

    async def scan(
        self,
        ml_predictions: Optional[Dict[str, float]] = None,
        agreement: float = 0.5,
        regime: str = "unknown",
        current_prices: Optional[Dict[str, float]] = None,
    ) -> List[PolymarketOpportunity]:
        """
        Full scan cycle: fetch -> classify -> edge detect -> persist.

        Args:
            ml_predictions: Asset -> prediction mapping (e.g. {"BTC": 0.05, "ETH": -0.02})
            agreement: Model agreement ratio (0-1)
            regime: Current HMM regime label
            current_prices: Asset -> current price (e.g. {"BTC": 96000, "ETH": 2800})

        Returns:
            List of opportunities sorted by edge (highest first).
        """
        ml_predictions = ml_predictions or {}
        current_prices = current_prices or {}

        try:
            raw_markets = await self.fetch_markets()
        except Exception as e:
            self.logger.warning(f"PolymarketScanner: fetch failed, skipping cycle: {e}")
            return []

        classified: List[PolymarketMarket] = []
        for raw in raw_markets:
            m = classify_market(raw)
            if m is not None:
                classified.append(m)

        # Compute edges for DIRECTION and HIT_PRICE markets
        opportunities: List[PolymarketOpportunity] = []
        for market in classified:
            pred = ml_predictions.get(market.asset, 0.0) if market.asset else 0.0
            price = current_prices.get(market.asset, 0.0) if market.asset else 0.0

            # DIRECTION needs a nonzero prediction; HIT_PRICE needs a current price
            if market.market_type == 'DIRECTION' and pred == 0.0:
                continue
            if market.market_type == 'HIT_PRICE' and price <= 0:
                continue

            opp = self.compute_edge(market, pred, agreement, regime, current_price=price)
            if opp:
                opportunities.append(opp)

        self._persist(classified, opportunities)
        opportunities.sort(key=lambda o: o.edge, reverse=True)

        # Compute stats by type
        type_counts: Dict[str, int] = {}
        for m in classified:
            type_counts[m.market_type] = type_counts.get(m.market_type, 0) + 1

        self._last_scan_stats = {
            "raw_total": len(raw_markets),
            "crypto_classified": len(classified),
            "opportunities": len(opportunities),
            "type_counts": type_counts,
            "top_edge": opportunities[0].edge if opportunities else 0.0,
            "scan_time": datetime.now(timezone.utc).isoformat(),
        }

        if opportunities:
            top = opportunities[0]
            self.logger.info(
                f"POLYMARKET SCAN: {len(raw_markets)} raw -> {len(classified)} crypto "
                f"({', '.join(f'{k}:{v}' for k, v in sorted(type_counts.items()))}) "
                f"-> {len(opportunities)} opportunities "
                f"(top: {top.edge:.1%} edge on {top.market.asset} {top.direction})"
            )
        else:
            self.logger.info(
                f"POLYMARKET SCAN: {len(raw_markets)} raw -> {len(classified)} crypto "
                f"({', '.join(f'{k}:{v}' for k, v in sorted(type_counts.items()))}) "
                f"-> 0 opportunities"
            )

        return opportunities

    def get_opportunities(self, min_edge: float = 0.03, limit: int = 20) -> List[dict]:
        """Query ranked opportunities from DB for dashboard/trading."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Get the latest scan time
            row = conn.execute("SELECT MAX(scan_time) FROM polymarket_scanner").fetchone()
            if not row or not row[0]:
                return []
            latest_scan = row[0]

            cursor = conn.execute("""
                SELECT condition_id, question, slug, market_type, asset,
                       timeframe_minutes, deadline, target_price, range_low, range_high,
                       yes_price, no_price, volume_24h, liquidity,
                       edge, our_probability, direction, confidence, scan_time
                FROM polymarket_scanner
                WHERE edge >= ? AND scan_time = ?
                ORDER BY edge DESC LIMIT ?
            """, (min_edge, latest_scan, limit))

            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, r)) for r in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"PolymarketScanner: get_opportunities error: {e}")
            return []
        finally:
            conn.close()

    def get_market_summary(self) -> Dict[str, Any]:
        """Return latest scan statistics for dashboard/logging."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT MAX(scan_time) FROM polymarket_scanner").fetchone()
            if not row or not row[0]:
                return {"total": 0, "by_type": {}, "by_asset": {}, "with_edge": 0}
            latest = row[0]

            total = conn.execute(
                "SELECT COUNT(*) FROM polymarket_scanner WHERE scan_time = ?", (latest,)
            ).fetchone()[0]

            type_rows = conn.execute(
                "SELECT market_type, COUNT(*) FROM polymarket_scanner WHERE scan_time = ? GROUP BY market_type",
                (latest,)
            ).fetchall()

            asset_rows = conn.execute(
                "SELECT asset, COUNT(*) FROM polymarket_scanner WHERE scan_time = ? AND asset IS NOT NULL GROUP BY asset",
                (latest,)
            ).fetchall()

            edge_count = conn.execute(
                "SELECT COUNT(*) FROM polymarket_scanner WHERE scan_time = ? AND edge IS NOT NULL AND edge >= 0.03",
                (latest,)
            ).fetchone()[0]

            return {
                "total": total,
                "by_type": {r[0]: r[1] for r in type_rows},
                "by_asset": {r[0]: r[1] for r in asset_rows},
                "with_edge": edge_count,
                "scan_time": latest,
            }
        except Exception as e:
            self.logger.error(f"PolymarketScanner: get_market_summary error: {e}")
            return {"total": 0, "by_type": {}, "by_asset": {}, "with_edge": 0}
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Return in-memory scan stats from last run."""
        return dict(self._last_scan_stats)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
