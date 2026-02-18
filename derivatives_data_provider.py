"""
Derivatives Data Provider — fetches Binance Futures + Fear & Greed data for ML features.

Free data sources (no API key needed):
  - Binance Futures: funding rate, open interest, long/short ratio, taker volume
  - Alternative.me: Fear & Greed Index

These feed 7 new features into build_feature_sequence():
  1. funding_rate_z      — funding rate z-scored over 50 bars
  2. oi_change_pct       — 5-bar percentage change in open interest
  3. long_short_ratio    — raw long/short account ratio
  4. taker_buy_sell_ratio — taker buy vol / sell vol
  5. has_derivatives_data — binary flag (1.0 when data available)
  6. fear_greed_norm      — Fear & Greed index / 100
  7. fear_greed_roc       — 3-day rate of change

Usage (live):
    provider = DerivativesDataProvider()
    deriv = await provider.get_derivatives_snapshot("BTC-USD")

Usage (historical for training):
    provider = DerivativesDataProvider()
    df = provider.fetch_historical_derivatives("BTC-USD", days_back=730)
    fng_df = provider.fetch_historical_fear_greed()
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

logger = logging.getLogger(__name__)

# ── Symbol mapping: Coinbase product IDs → Binance futures symbols ────────────
PAIR_TO_BINANCE = {
    'BTC-USD': 'BTCUSDT',
    'ETH-USD': 'ETHUSDT',
    'SOL-USD': 'SOLUSDT',
    'DOGE-USD': 'DOGEUSDT',
    'AVAX-USD': 'AVAXUSDT',
    'LINK-USD': 'LINKUSDT',
}

BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_FUTURES_DATA = "https://fapi.binance.com/futures/data"
FNG_API_BASE = "https://api.alternative.me/fng/"

# Rate limit: Binance allows 1200 req/min for public endpoints
_REQUEST_DELAY = 0.15  # seconds between requests


class DerivativesDataProvider:
    """Fetches and caches Binance Futures derivatives + Fear & Greed data."""

    def __init__(self, cache_ttl_seconds: int = 60):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: Dict[str, float] = {}
        self._cache_ttl = cache_ttl_seconds
        self._fng_cache: Optional[Dict[str, Any]] = None
        self._fng_cache_ts: float = 0.0
        self._fng_cache_ttl = 3600  # 1 hour (FnG updates daily)

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE SNAPSHOT (for real-time inference)
    # ══════════════════════════════════════════════════════════════════════════

    async def get_derivatives_snapshot(self, pair: str) -> Dict[str, float]:
        """Get latest derivatives + FnG data for a pair.

        Returns dict with keys: funding_rate, open_interest, long_short_ratio,
        taker_buy_vol, taker_sell_vol, fear_greed.
        Missing values are NaN.
        """
        cache_key = f"snapshot_{pair}"
        now = time.time()
        if cache_key in self._cache and (now - self._cache_ts.get(cache_key, 0)) < self._cache_ttl:
            return self._cache[cache_key]

        symbol = PAIR_TO_BINANCE.get(pair, pair.replace('-', ''))

        # Fetch all in parallel
        results = await asyncio.gather(
            asyncio.to_thread(self._fetch_funding_rate, symbol, 1),
            asyncio.to_thread(self._fetch_open_interest_live, symbol),
            asyncio.to_thread(self._fetch_long_short_ratio, symbol, "5m", 1),
            asyncio.to_thread(self._fetch_taker_volume, symbol, "5m", 1),
            asyncio.to_thread(self._fetch_fear_greed_live),
            return_exceptions=True,
        )

        funding, oi, ls_ratio, taker, fng = [
            r if not isinstance(r, BaseException) else None for r in results
        ]

        snapshot = {
            'funding_rate': float(funding[0]['fundingRate']) if funding else float('nan'),
            'open_interest': float(oi.get('openInterest', 'nan')) if oi else float('nan'),
            'long_short_ratio': float(ls_ratio[0]['longShortRatio']) if ls_ratio else float('nan'),
            'taker_buy_vol': float(taker[0]['buyVol']) if taker else float('nan'),
            'taker_sell_vol': float(taker[0]['sellVol']) if taker else float('nan'),
            'fear_greed': float(fng) if fng is not None else float('nan'),
        }

        self._cache[cache_key] = snapshot
        self._cache_ts[cache_key] = now
        return snapshot

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORICAL DATA (for training)
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_historical_derivatives(
        self,
        pair: str,
        days_back: int = 730,
        period: str = "5m",
    ) -> pd.DataFrame:
        """Fetch historical derivatives data from Binance Futures.

        Returns DataFrame with columns:
            timestamp, funding_rate, open_interest, long_short_ratio,
            taker_buy_vol, taker_sell_vol

        Args:
            pair: Coinbase-style pair (e.g. 'BTC-USD')
            days_back: How many days of history to fetch
            period: Granularity for OI/LS/taker data ('5m', '15m', '1h', etc.)
        """
        symbol = PAIR_TO_BINANCE.get(pair, pair.replace('-', ''))
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - (days_back * 86400 * 1000)

        logger.info(f"Fetching {days_back}d derivatives history for {symbol}...")

        # Fetch each data type
        funding_df = self._fetch_funding_history(symbol, start_ms, end_ms)
        oi_df = self._fetch_oi_history(symbol, period, start_ms, end_ms)
        ls_df = self._fetch_ls_history(symbol, period, start_ms, end_ms)
        taker_df = self._fetch_taker_history(symbol, period, start_ms, end_ms)

        # Merge all on timestamp
        dfs = []
        if not funding_df.empty:
            dfs.append(funding_df.rename(columns={'value': 'funding_rate'}))
        if not oi_df.empty:
            dfs.append(oi_df.rename(columns={'value': 'open_interest'}))
        if not ls_df.empty:
            dfs.append(ls_df.rename(columns={'value': 'long_short_ratio'}))
        if not taker_df.empty:
            dfs.append(taker_df)

        if not dfs:
            logger.warning(f"No derivatives data fetched for {symbol}")
            return pd.DataFrame()

        # Merge all on timestamp using outer join, forward-fill funding rate
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on='timestamp', how='outer')
        result = result.sort_values('timestamp').reset_index(drop=True)

        # Forward-fill funding rate (it's 8-hourly, others are 5-min)
        if 'funding_rate' in result.columns:
            result['funding_rate'] = result['funding_rate'].ffill()

        logger.info(f"  {symbol}: {len(result)} rows, "
                     f"range: {result['timestamp'].min()} → {result['timestamp'].max()}")
        return result

    def fetch_historical_fear_greed(self, limit: int = 0) -> pd.DataFrame:
        """Fetch full Fear & Greed history from Alternative.me.

        Args:
            limit: Number of days (0 = all available, back to ~2018)

        Returns:
            DataFrame with columns: timestamp, fear_greed
            timestamp is Unix seconds (midnight UTC of each day)
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available; cannot fetch Fear & Greed")
            return pd.DataFrame()

        try:
            params = {'format': 'json', 'limit': 0 if limit <= 0 else limit}

            resp = requests.get(FNG_API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get('data', [])

            if not data:
                return pd.DataFrame()

            rows = []
            for entry in data:
                rows.append({
                    'timestamp': int(entry['timestamp']),
                    'fear_greed': int(entry['value']),
                })

            df = pd.DataFrame(rows)
            df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
            logger.info(f"  Fear & Greed: {len(df)} daily values "
                         f"({datetime.fromtimestamp(df['timestamp'].iloc[0], tz=timezone.utc).strftime('%Y-%m-%d')} → "
                         f"{datetime.fromtimestamp(df['timestamp'].iloc[-1], tz=timezone.utc).strftime('%Y-%m-%d')})")
            return df

        except Exception as e:
            logger.error(f"Fear & Greed fetch failed: {e}")
            return pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════════════
    # INTERNAL — Binance Futures API calls
    # ══════════════════════════════════════════════════════════════════════════

    def _fetch_funding_rate(self, symbol: str, limit: int = 10) -> Optional[List[dict]]:
        """GET /fapi/v1/fundingRate — latest funding rates."""
        if not REQUESTS_AVAILABLE:
            return None
        try:
            resp = requests.get(
                f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate",
                params={'symbol': symbol, 'limit': limit},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"Funding rate {resp.status_code}: {resp.text[:200]}")
            return None
        except Exception as e:
            logger.warning(f"Funding rate fetch error: {e}")
            return None

    def _fetch_open_interest_live(self, symbol: str) -> Optional[dict]:
        """GET /fapi/v1/openInterest — current open interest."""
        if not REQUESTS_AVAILABLE:
            return None
        try:
            resp = requests.get(
                f"{BINANCE_FAPI_BASE}/fapi/v1/openInterest",
                params={'symbol': symbol},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.warning(f"OI fetch error: {e}")
            return None

    def _fetch_long_short_ratio(self, symbol: str, period: str, limit: int) -> Optional[List[dict]]:
        """GET /futures/data/globalLongShortAccountRatio"""
        if not REQUESTS_AVAILABLE:
            return None
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES_DATA}/globalLongShortAccountRatio",
                params={'symbol': symbol, 'period': period, 'limit': limit},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.warning(f"LS ratio fetch error: {e}")
            return None

    def _fetch_taker_volume(self, symbol: str, period: str, limit: int) -> Optional[List[dict]]:
        """GET /futures/data/takeBuySellVol"""
        if not REQUESTS_AVAILABLE:
            return None
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES_DATA}/takeBuySellVol",
                params={'symbol': symbol, 'period': period, 'limit': limit},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.warning(f"Taker volume fetch error: {e}")
            return None

    def _fetch_fear_greed_live(self) -> Optional[float]:
        """Fetch current Fear & Greed value with caching."""
        now = time.time()
        if self._fng_cache is not None and (now - self._fng_cache_ts) < self._fng_cache_ttl:
            return self._fng_cache.get('value')

        if not REQUESTS_AVAILABLE:
            return None
        try:
            resp = requests.get(FNG_API_BASE, params={'limit': 1, 'format': 'json'}, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                if data:
                    val = int(data[0]['value'])
                    self._fng_cache = {'value': val}
                    self._fng_cache_ts = now
                    return float(val)
            return None
        except Exception as e:
            logger.warning(f"FnG fetch error: {e}")
            return None

    # ── Historical pagination helpers ──────────────────────────────────────

    def _fetch_funding_history(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Paginate through funding rate history (8-hourly data, max 1000/request)."""
        if not REQUESTS_AVAILABLE:
            return pd.DataFrame()

        all_rows = []
        current_start = start_ms

        while current_start < end_ms:
            try:
                resp = requests.get(
                    f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate",
                    params={'symbol': symbol, 'startTime': current_start, 'endTime': end_ms, 'limit': 1000},
                    timeout=15,
                )
                if resp.status_code != 200:
                    logger.warning(f"Funding history {resp.status_code}")
                    break

                data = resp.json()
                if not data:
                    break

                for entry in data:
                    all_rows.append({
                        'timestamp': int(entry['fundingTime']) // 1000,
                        'value': float(entry['fundingRate']),
                    })

                current_start = int(data[-1]['fundingTime']) + 1
                time.sleep(_REQUEST_DELAY)

            except Exception as e:
                logger.warning(f"Funding history error: {e}")
                break

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        logger.info(f"  Funding rate: {len(df)} entries")
        return df

    def _fetch_oi_history(self, symbol: str, period: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Paginate through open interest history."""
        return self._paginate_futures_data(
            f"{BINANCE_FUTURES_DATA}/openInterestHist",
            symbol, period, start_ms, end_ms,
            value_key='sumOpenInterest', label='OI',
        )

    def _fetch_ls_history(self, symbol: str, period: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Paginate through long/short ratio history."""
        return self._paginate_futures_data(
            f"{BINANCE_FUTURES_DATA}/globalLongShortAccountRatio",
            symbol, period, start_ms, end_ms,
            value_key='longShortRatio', label='LS ratio',
        )

    def _fetch_taker_history(self, symbol: str, period: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Paginate through taker buy/sell volume history."""
        if not REQUESTS_AVAILABLE:
            return pd.DataFrame()

        all_rows = []
        current_start = start_ms

        while current_start < end_ms:
            try:
                resp = requests.get(
                    f"{BINANCE_FUTURES_DATA}/takeBuySellVol",
                    params={
                        'symbol': symbol, 'period': period,
                        'startTime': current_start, 'endTime': end_ms, 'limit': 500,
                    },
                    timeout=15,
                )
                if resp.status_code != 200:
                    logger.warning(f"Taker history {resp.status_code}")
                    break

                data = resp.json()
                if not data:
                    break

                for entry in data:
                    all_rows.append({
                        'timestamp': int(entry['timestamp']) // 1000,
                        'taker_buy_vol': float(entry['buyVol']),
                        'taker_sell_vol': float(entry['sellVol']),
                    })

                current_start = int(data[-1]['timestamp']) + 1
                time.sleep(_REQUEST_DELAY)

            except Exception as e:
                logger.warning(f"Taker history error: {e}")
                break

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        logger.info(f"  Taker volume: {len(df)} entries")
        return df

    def _paginate_futures_data(
        self,
        url: str,
        symbol: str,
        period: str,
        start_ms: int,
        end_ms: int,
        value_key: str,
        label: str,
    ) -> pd.DataFrame:
        """Generic paginator for Binance /futures/data/ endpoints."""
        if not REQUESTS_AVAILABLE:
            return pd.DataFrame()

        all_rows = []
        current_start = start_ms

        while current_start < end_ms:
            try:
                resp = requests.get(
                    url,
                    params={
                        'symbol': symbol, 'period': period,
                        'startTime': current_start, 'endTime': end_ms, 'limit': 500,
                    },
                    timeout=15,
                )
                if resp.status_code != 200:
                    logger.warning(f"{label} history {resp.status_code}: {resp.text[:200]}")
                    break

                data = resp.json()
                if not data:
                    break

                for entry in data:
                    all_rows.append({
                        'timestamp': int(entry['timestamp']) // 1000,
                        'value': float(entry[value_key]),
                    })

                current_start = int(data[-1]['timestamp']) + 1
                time.sleep(_REQUEST_DELAY)

            except Exception as e:
                logger.warning(f"{label} history error: {e}")
                break

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        logger.info(f"  {label}: {len(df)} entries")
        return df
