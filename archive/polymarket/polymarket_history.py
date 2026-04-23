"""
polymarket_history.py — Historical 5-minute market data collector.

Fetches resolved Polymarket 5m direction markets and records:
- Asset, window start/end times
- Crowd YES price at market open
- Resolution outcome: UP (1) or DOWN (0)
- Price data from our own five_minute_bars

Stores in SQLite: polymarket_5m_history table.

Usage:
    collector = PolymarketHistoryCollector(db_path="data/renaissance_bot.db")
    result = collector.collect_last_n_hours(hours=72)
    df = collector.get_dataframe()
"""

import sqlite3
import requests
import time
import math
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Confirmed slug format: {asset}-updown-5m-{unix_timestamp}
# Timestamps aligned to 300-second (5-minute) boundaries
# Resolution: outcomePrices[0] = UP outcome price
#   "1" or close to 1.0 → UP won
#   "0" or close to 0.0 → DOWN won

FIVE_MIN_ASSETS = {
    "BTC": {"slug_prefix": "btc-updown-5m", "pair": "BTC/USDT", "product_id": "BTC-USD"},
    "ETH": {"slug_prefix": "eth-updown-5m", "pair": "ETH/USDT", "product_id": "ETH-USD"},
    "SOL": {"slug_prefix": "sol-updown-5m", "pair": "SOL/USDT", "product_id": "SOL-USD"},
    "XRP": {"slug_prefix": "xrp-updown-5m", "pair": "XRP/USDT", "product_id": "XRP-USD"},
    "DOGE": {"slug_prefix": "doge-updown-5m", "pair": "DOGE/USDT", "product_id": "DOGE-USD"},
    "HYPE": {"slug_prefix": "hype-updown-5m", "pair": None, "product_id": None},
    "BNB": {"slug_prefix": "bnb-updown-5m", "pair": "BNB/USDT", "product_id": None},
}

# Only assets where we have ML predictions
CALIBRATION_ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]

WINDOW_SECONDS = 300  # 5 minutes


class PolymarketHistoryCollector:

    GAMMA_BASE = "https://gamma-api.polymarket.com"
    CLOB_BASE = "https://clob.polymarket.com"

    def __init__(self, db_path: str = "data/renaissance_bot.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS polymarket_5m_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT NOT NULL,
                window_start INTEGER NOT NULL,
                window_end INTEGER NOT NULL,
                slug TEXT,
                market_id TEXT,
                condition_id TEXT,

                -- Crowd pricing (what the market thought)
                crowd_yes_open REAL,
                crowd_yes_close REAL,
                crowd_yes_midlife REAL,

                -- Resolution
                resolved INTEGER,          -- 1=UP won, 0=DOWN won
                outcome_yes_final REAL,    -- Final YES price

                -- Price data from our bars
                price_start REAL,
                price_end REAL,
                price_change_pct REAL,

                -- Metadata
                volume REAL,
                collected_at TEXT DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(asset, window_start)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pm5m_asset_time
            ON polymarket_5m_history(asset, window_start)
        """)
        conn.commit()
        conn.close()

    def _get_price_from_bars(self, conn: sqlite3.Connection, pair: str,
                             window_start: int) -> Tuple[Optional[float], Optional[float]]:
        """Get open/close price from five_minute_bars for a window."""
        row = conn.execute("""
            SELECT open, close FROM five_minute_bars
            WHERE pair = ? AND bar_start = ?
        """, (pair, float(window_start))).fetchone()
        if row:
            return float(row[0]), float(row[1])
        return None, None

    def collect_last_n_hours(self, hours: int = 72,
                             assets: Optional[List[str]] = None) -> Dict:
        """
        Collect resolved 5m markets for the last N hours.

        Enumerates every 5-minute slot going backward, constructs the slug,
        looks it up on Gamma API, and records the resolution.
        """
        if assets is None:
            assets = list(FIVE_MIN_ASSETS.keys())

        now = time.time()
        current_slot = int(math.floor(now / WINDOW_SECONDS) * WINDOW_SECONDS)

        # Start from 2 windows ago (ensure resolution is complete)
        start_slot = current_slot - (2 * WINDOW_SECONDS)
        earliest_slot = current_slot - (hours * 3600)

        conn = sqlite3.connect(self.db_path)

        total_collected = 0
        total_skipped = 0
        total_not_found = 0
        total_errors = 0

        for asset in assets:
            config = FIVE_MIN_ASSETS.get(asset)
            if not config:
                continue

            slot = start_slot
            asset_collected = 0
            consecutive_not_found = 0

            while slot >= earliest_slot:
                # Check if already collected
                existing = conn.execute(
                    "SELECT id FROM polymarket_5m_history WHERE asset=? AND window_start=?",
                    (asset, slot)
                ).fetchone()

                if existing:
                    total_skipped += 1
                    slot -= WINDOW_SECONDS
                    consecutive_not_found = 0
                    continue

                # Construct slug and look up
                slug = f"{config['slug_prefix']}-{slot}"

                try:
                    resp = requests.get(
                        f"{self.GAMMA_BASE}/markets",
                        params={"slug": slug},
                        timeout=10,
                    )

                    if resp.status_code == 429:
                        # Rate limited — back off
                        logger.debug(f"Rate limited, sleeping 2s")
                        time.sleep(2)
                        continue

                    if resp.status_code != 200:
                        total_errors += 1
                        slot -= WINDOW_SECONDS
                        time.sleep(0.15)
                        continue

                    markets = resp.json()
                    if not isinstance(markets, list) or not markets:
                        total_not_found += 1
                        consecutive_not_found += 1
                        # If many consecutive not found, these markets may not exist yet
                        if consecutive_not_found > 50:
                            logger.info(
                                f"HISTORY [{asset}]: 50 consecutive not found at "
                                f"slot {slot} — stopping lookback"
                            )
                            break
                        slot -= WINDOW_SECONDS
                        time.sleep(0.1)
                        continue

                    consecutive_not_found = 0
                    market = markets[0]

                    # Check if closed
                    is_closed = market.get("closed", False)
                    if not is_closed:
                        slot -= WINDOW_SECONDS
                        time.sleep(0.1)
                        continue

                    # Extract outcome prices
                    prices = market.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        prices = json.loads(prices)

                    yes_final = float(prices[0]) if prices and len(prices) >= 1 else None

                    # Determine resolution
                    if yes_final is not None:
                        if yes_final >= 0.95:
                            resolved = 1  # UP won
                        elif yes_final <= 0.05:
                            resolved = 0  # DOWN won
                        else:
                            # Not definitively resolved
                            slot -= WINDOW_SECONDS
                            time.sleep(0.1)
                            continue
                    else:
                        slot -= WINDOW_SECONDS
                        time.sleep(0.1)
                        continue

                    # Try to get CLOB price history for crowd pricing
                    token_ids = market.get("clobTokenIds", "[]")
                    if isinstance(token_ids, str):
                        token_ids = json.loads(token_ids)

                    crowd_yes_open = None
                    crowd_yes_mid = None
                    crowd_yes_close = None

                    if token_ids and len(token_ids) >= 1:
                        yes_token = token_ids[0]
                        try:
                            hist_resp = requests.get(
                                f"{self.CLOB_BASE}/prices-history",
                                params={
                                    "market": yes_token,
                                    "startTs": slot,
                                    "endTs": slot + WINDOW_SECONDS,
                                    "fidelity": 1,
                                },
                                timeout=10,
                            )
                            if hist_resp.status_code == 200:
                                hist_data = hist_resp.json()
                                history = hist_data.get("history", [])
                                if history:
                                    crowd_yes_open = float(history[0].get("p", 0))
                                    mid_idx = len(history) // 2
                                    if mid_idx < len(history):
                                        crowd_yes_mid = float(history[mid_idx].get("p", 0))
                                    crowd_yes_close = float(history[-1].get("p", 0))
                        except Exception as e:
                            logger.debug(f"Price history failed for {slug}: {e}")
                        time.sleep(0.1)

                    # Get price data from our bars
                    price_start = None
                    price_end = None
                    price_change_pct = None

                    pair = config.get("pair")
                    if pair:
                        price_start, price_end = self._get_price_from_bars(
                            conn, pair, slot
                        )
                        if price_start and price_end and price_start > 0:
                            price_change_pct = (price_end - price_start) / price_start * 100

                    volume = market.get("volume", 0) or 0

                    conn.execute("""
                        INSERT OR IGNORE INTO polymarket_5m_history
                        (asset, window_start, window_end, slug, market_id, condition_id,
                         crowd_yes_open, crowd_yes_close, crowd_yes_midlife,
                         resolved, outcome_yes_final,
                         price_start, price_end, price_change_pct, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        asset, slot, slot + WINDOW_SECONDS,
                        slug,
                        str(market.get("id", ""))[:50],
                        str(market.get("conditionId", ""))[:50],
                        crowd_yes_open, crowd_yes_close, crowd_yes_mid,
                        resolved, yes_final,
                        price_start, price_end, price_change_pct,
                        float(volume) if volume else 0,
                    ))
                    conn.commit()

                    asset_collected += 1
                    total_collected += 1

                except requests.exceptions.RequestException as e:
                    logger.debug(f"Request failed for {slug}: {e}")
                    total_errors += 1
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Unexpected error for {slug}: {e}")
                    total_errors += 1

                slot -= WINDOW_SECONDS
                time.sleep(0.1)  # ~10 req/sec

            logger.info(f"HISTORY [{asset}]: collected {asset_collected} resolved markets")

        conn.close()

        logger.info(
            f"POLYMARKET HISTORY: collected={total_collected} skipped={total_skipped} "
            f"not_found={total_not_found} errors={total_errors}"
        )

        return {
            "collected": total_collected,
            "skipped": total_skipped,
            "not_found": total_not_found,
            "errors": total_errors,
        }

    def backfill_from_bars(self, assets: Optional[List[str]] = None) -> Dict:
        """
        Build synthetic ground truth from our five_minute_bars data.

        For each 5-minute bar in our database:
        - If close >= open → UP (resolved=1)
        - If close < open → DOWN (resolved=0)

        This gives us ground truth for calibration even without Polymarket data.
        Does NOT overwrite existing Polymarket-sourced records.
        """
        if assets is None:
            assets = CALIBRATION_ASSETS

        conn = sqlite3.connect(self.db_path)
        total = 0

        for asset in assets:
            config = FIVE_MIN_ASSETS.get(asset)
            if not config or not config.get("pair"):
                continue

            pair = config["pair"]

            rows = conn.execute("""
                SELECT bar_start, open, close FROM five_minute_bars
                WHERE pair = ? AND open > 0 AND close > 0
                ORDER BY bar_start
            """, (pair,)).fetchall()

            inserted = 0
            for row in rows:
                bar_start = int(row[0])
                open_price = float(row[1])
                close_price = float(row[2])

                # Skip if already have Polymarket data
                existing = conn.execute(
                    "SELECT id FROM polymarket_5m_history WHERE asset=? AND window_start=?",
                    (asset, bar_start)
                ).fetchone()
                if existing:
                    continue

                resolved = 1 if close_price >= open_price else 0
                pct = (close_price - open_price) / open_price * 100

                conn.execute("""
                    INSERT OR IGNORE INTO polymarket_5m_history
                    (asset, window_start, window_end, slug,
                     resolved, outcome_yes_final,
                     price_start, price_end, price_change_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    asset, bar_start, bar_start + WINDOW_SECONDS,
                    f"{config['slug_prefix']}-{bar_start}",
                    resolved, 1.0 if resolved else 0.0,
                    open_price, close_price, pct,
                ))
                inserted += 1

            conn.commit()
            total += inserted
            logger.info(f"BACKFILL [{asset}]: inserted {inserted} from bars")

        conn.close()
        return {"backfilled": total}

    def get_dataframe(self):
        """Return collected history as a pandas DataFrame."""
        import pandas as pd
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM polymarket_5m_history ORDER BY window_start",
            conn,
        )
        conn.close()
        return df

    def get_stats(self) -> Dict:
        """Summary statistics of collected data."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        total = conn.execute(
            "SELECT COUNT(*) as n FROM polymarket_5m_history"
        ).fetchone()["n"]

        per_asset = conn.execute("""
            SELECT asset, COUNT(*) as n,
                   SUM(resolved) as ups,
                   COUNT(*) - SUM(resolved) as downs,
                   AVG(crowd_yes_open) as avg_crowd_open,
                   SUM(CASE WHEN crowd_yes_open IS NOT NULL THEN 1 ELSE 0 END) as has_crowd,
                   SUM(CASE WHEN price_start IS NOT NULL THEN 1 ELSE 0 END) as has_price,
                   MIN(window_start) as earliest,
                   MAX(window_start) as latest
            FROM polymarket_5m_history
            GROUP BY asset
        """).fetchall()

        conn.close()

        return {
            "total_markets": total,
            "per_asset": [dict(r) for r in per_asset],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    collector = PolymarketHistoryCollector(db_path="data/renaissance_bot.db")

    # Step 1: Collect from Polymarket API
    print("=== Collecting from Polymarket API (last 72 hours) ===")
    result = collector.collect_last_n_hours(hours=72)
    print(f"API collection: {result}")

    # Step 2: Backfill from our own bar data for calibration
    print("\n=== Backfilling from five_minute_bars ===")
    backfill = collector.backfill_from_bars()
    print(f"Backfill: {backfill}")

    # Step 3: Show stats
    print("\n=== Stats ===")
    stats = collector.get_stats()
    print(f"Total markets: {stats['total_markets']}")
    for a in stats['per_asset']:
        up_pct = a['ups'] / a['n'] * 100 if a['n'] > 0 else 0
        print(
            f"  {a['asset']}: {a['n']} markets, {a['ups']} UP ({up_pct:.1f}%), "
            f"crowd_data={a['has_crowd']}, price_data={a['has_price']}"
        )
