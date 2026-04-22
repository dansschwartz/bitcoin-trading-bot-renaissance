"""
cascade_data_collector.py — Collects Polymarket crowd price data
for validating the Cascade strategy.

Runs every 30 seconds. Records:
  1. Current BTC price (from bot's data feed)
  2. All active crypto direction market prices (from Polymarket Gamma API)
  3. BTC's recent returns (1-bar, 3-bar, 5-bar)

After 1-2 weeks of collection, run cascade_analyze_crowd.py to measure
how fast the crowd reprices after BTC moves.

Storage: SQLite table 'cascade_crowd_data' in the bot's database.
"""

import logging
import time
import threading
import sqlite3
import json
import os
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("cascade_collector")

GAMMA_API = "https://gamma-api.polymarket.com"

# Slug patterns for crypto direction markets
# Format: {asset}-updown-{duration}-{timestamp}
ASSET_SLUGS = ['btc', 'eth', 'sol', 'xrp', 'doge', 'avax', 'link']
DURATIONS = ['5m', '15m']


class CascadeDataCollector:
    """Collects Polymarket crowd prices alongside BTC prices for analysis."""

    def __init__(self, bot, db_path: str = None, poll_interval: int = 30):
        self.bot = bot
        self.poll_interval = poll_interval
        self._running = False
        self._thread = None

        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "data", "renaissance_bot.db")
        self.db_path = db_path

        self._init_db()
        self._request_count = 0
        self._last_rate_reset = time.time()

        logger.info(f"Cascade data collector initialized (poll every {poll_interval}s)")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cascade_crowd_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                btc_price REAL,
                btc_ret_1bar REAL,
                btc_ret_3bar REAL,
                btc_ret_6bar REAL,
                markets_json TEXT
            );

            CREATE TABLE IF NOT EXISTS cascade_market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                asset TEXT NOT NULL,
                duration TEXT NOT NULL,
                market_slug TEXT,
                condition_id TEXT,
                yes_price REAL,
                no_price REAL,
                volume REAL,
                liquidity REAL,
                end_date TEXT,
                seconds_remaining REAL,
                resolved INTEGER DEFAULT 0,
                outcome TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_cascade_crowd_ts
                ON cascade_crowd_data(timestamp);
            CREATE INDEX IF NOT EXISTS idx_cascade_snapshots_ts
                ON cascade_market_snapshots(timestamp, asset);
            CREATE INDEX IF NOT EXISTS idx_cascade_snapshots_slug
                ON cascade_market_snapshots(market_slug);
        """)
        conn.commit()
        conn.close()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Cascade data collector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Cascade data collector stopped")

    def _poll_loop(self):
        while self._running:
            try:
                self._collect_snapshot()
            except Exception as e:
                logger.error(f"Cascade collector error: {e}")
            time.sleep(self.poll_interval)

    def _collect_snapshot(self):
        """Collect one snapshot of BTC price + all active direction markets."""
        now = datetime.now(timezone.utc)

        # Get BTC price and recent returns
        btc_price = self._get_btc_price()
        btc_rets = self._get_btc_returns()

        # Get all active crypto direction markets from Gamma API
        markets = self._fetch_active_direction_markets()

        # Save to DB
        conn = sqlite3.connect(self.db_path)

        # Save BTC state
        conn.execute(
            """INSERT INTO cascade_crowd_data
               (timestamp, btc_price, btc_ret_1bar, btc_ret_3bar, btc_ret_6bar, markets_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                now.isoformat(),
                btc_price,
                btc_rets.get('1bar', 0),
                btc_rets.get('3bar', 0),
                btc_rets.get('6bar', 0),
                json.dumps([m.get('slug') for m in markets]) if markets else '[]',
            )
        )

        # Save each market snapshot
        for m in markets:
            asset = m.get('asset', 'UNKNOWN')
            duration = m.get('duration', 'unknown')

            # Parse prices
            yes_price = 0.5
            no_price = 0.5
            try:
                prices = m.get('outcomePrices', '[]')
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if len(prices) >= 2:
                    yes_price = float(prices[0])
                    no_price = float(prices[1])
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

            # Calculate time remaining
            end_date = m.get('endDate', m.get('end_date_iso', ''))
            seconds_remaining = 0
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                seconds_remaining = (end_dt - now).total_seconds()
            except (ValueError, TypeError):
                pass

            conn.execute(
                """INSERT INTO cascade_market_snapshots
                   (timestamp, asset, duration, market_slug, condition_id,
                    yes_price, no_price, volume, liquidity,
                    end_date, seconds_remaining, resolved, outcome)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
                (
                    now.isoformat(),
                    asset,
                    duration,
                    m.get('slug', ''),
                    m.get('conditionId', m.get('condition_id', '')),
                    yes_price,
                    no_price,
                    m.get('volume', 0),
                    m.get('liquidity', 0),
                    end_date,
                    seconds_remaining,
                )
            )

        conn.commit()
        conn.close()

        if markets:
            logger.debug(
                f"Cascade collector: BTC=${btc_price:.0f} "
                f"ret1={btc_rets.get('1bar', 0):.4f} "
                f"markets={len(markets)}"
            )

    def _fetch_active_direction_markets(self) -> list:
        """
        Fetch all active crypto direction markets from Polymarket Gamma API.
        Returns list of market dicts with asset, duration, prices.
        """
        # Rate limiting: max 50 requests per minute
        now = time.time()
        if now - self._last_rate_reset > 60:
            self._request_count = 0
            self._last_rate_reset = now
        if self._request_count >= 50:
            return []

        markets = []
        try:
            # Query Gamma for active crypto markets
            resp = requests.get(
                f"{GAMMA_API}/markets",
                params={
                    'active': 'true',
                    'closed': 'false',
                    'limit': 100,
                    'tag_id': 21,  # Crypto tag
                },
                timeout=15,
            )
            self._request_count += 1

            if resp.status_code != 200:
                logger.warning(f"Gamma API returned {resp.status_code}")
                return []

            all_markets = resp.json()
            if not isinstance(all_markets, list):
                all_markets = all_markets.get('data', all_markets.get('markets', []))

            # Filter for direction markets (up/down, 5m/15m)
            for m in all_markets:
                slug = m.get('slug', m.get('market_slug', ''))
                question = m.get('question', '')

                # Check if this is a direction market
                is_direction = False
                asset = None
                duration = None

                for asset_slug in ASSET_SLUGS:
                    for dur in DURATIONS:
                        if f'{asset_slug}-updown-{dur}' in slug.lower():
                            is_direction = True
                            asset = asset_slug.upper()
                            duration = dur
                            break
                    if is_direction:
                        break

                # Also try matching by question text
                if not is_direction:
                    q_lower = question.lower()
                    for asset_slug in ASSET_SLUGS:
                        if asset_slug in q_lower and ('up or down' in q_lower or 'direction' in q_lower):
                            is_direction = True
                            asset = asset_slug.upper()
                            duration = '15m' if '15' in q_lower else ('5m' if '5' in q_lower else '15m')
                            break

                if is_direction:
                    m['asset'] = asset
                    m['duration'] = duration
                    markets.append(m)

        except requests.exceptions.RequestException as e:
            logger.warning(f"Gamma API request failed: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch direction markets: {e}")

        return markets

    def _get_btc_price(self) -> float:
        """Get current BTC price from bot's latest data."""
        try:
            # Try bot's latest_prices dict (populated during scan loop)
            if hasattr(self.bot, 'latest_prices'):
                for key in ('BTCUSDT', 'BTC-USD', 'BTC/USDT'):
                    p = self.bot.latest_prices.get(key, 0)
                    if p > 0:
                        return float(p)
            # Try the price_cache from binance_spot_provider
            if hasattr(self.bot, '_price_cache'):
                for key in ('BTCUSDT', 'BTC-USD'):
                    if key in self.bot._price_cache:
                        return float(self.bot._price_cache[key].get('close', 0))
        except Exception:
            pass
        return 0.0

    def _get_btc_returns(self) -> dict:
        """Get recent BTC returns from bot's bar data."""
        rets = {'1bar': 0.0, '3bar': 0.0, '6bar': 0.0}
        try:
            # Try to get from the bot's bar_history for BTCUSDT
            if hasattr(self.bot, 'bar_history'):
                bars = self.bot.bar_history.get('BTCUSDT', [])
                if len(bars) >= 2:
                    rets['1bar'] = bars[-1]['close'] / bars[-2]['close'] - 1 if bars[-2]['close'] > 0 else 0
                if len(bars) >= 4:
                    rets['3bar'] = bars[-1]['close'] / bars[-4]['close'] - 1 if bars[-4]['close'] > 0 else 0
                if len(bars) >= 7:
                    rets['6bar'] = bars[-1]['close'] / bars[-7]['close'] - 1 if bars[-7]['close'] > 0 else 0
        except Exception:
            pass
        return rets

    def get_collection_stats(self) -> dict:
        """Return stats for dashboard."""
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row

            total_snapshots = conn.execute(
                "SELECT COUNT(*) as n FROM cascade_crowd_data"
            ).fetchone()['n']

            total_market_records = conn.execute(
                "SELECT COUNT(*) as n FROM cascade_market_snapshots"
            ).fetchone()['n']

            unique_markets = conn.execute(
                "SELECT COUNT(DISTINCT market_slug) as n FROM cascade_market_snapshots"
            ).fetchone()['n']

            first_ts = conn.execute(
                "SELECT MIN(timestamp) as ts FROM cascade_crowd_data"
            ).fetchone()['ts']

            last_ts = conn.execute(
                "SELECT MAX(timestamp) as ts FROM cascade_crowd_data"
            ).fetchone()['ts']

            conn.close()
            return {
                'total_snapshots': total_snapshots,
                'total_market_records': total_market_records,
                'unique_markets': unique_markets,
                'first_record': first_ts,
                'last_record': last_ts,
                'collecting': self._running,
            }
        except Exception:
            return {'error': 'no data yet', 'collecting': self._running}
