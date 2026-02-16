"""
Download historical 5-minute OHLCV candles from Coinbase public REST API.

Saves to CSV files in data/training/ for ML model training.
Supports incremental updates — if CSV exists, only fetches new candles.

Usage:
    python -m scripts.training.fetch_training_data --days 30
    python -m scripts.training.fetch_training_data --days 30 --pairs BTC-USD ETH-USD
"""

import argparse
import json
import logging
import math
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COINBASE_API = "https://api.exchange.coinbase.com/products/{}/candles"
GRANULARITY = 300  # 5 minutes
BATCH_SIZE = 300   # Coinbase max per request
DEFAULT_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"]
DEFAULT_OUTPUT_DIR = "data/training"


def fetch_candles_batch(pair: str, start_ts: int, end_ts: int) -> List[list]:
    """Fetch a batch of candles from Coinbase public REST API.

    Args:
        pair: Trading pair e.g. "BTC-USD"
        start_ts: Unix timestamp for start
        end_ts: Unix timestamp for end

    Returns:
        List of [timestamp, low, high, open, close, volume]
    """
    import urllib.request

    url = COINBASE_API.format(pair)
    params = f"?granularity={GRANULARITY}&start={start_ts}&end={end_ts}"
    req = urllib.request.Request(
        url + params,
        headers={"User-Agent": "RenaissanceBot/1.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data


def download_pair(pair: str, days: int, output_dir: str = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    """Download candles for a single pair and save to CSV.

    Supports incremental updates: if CSV exists, only fetches candles
    newer than the last timestamp in the file.

    Args:
        pair: Trading pair e.g. "BTC-USD"
        days: Number of days of history to fetch
        output_dir: Directory to save CSV files

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{pair}.csv")

    existing_df = None
    last_ts = 0

    # Check for existing data
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        if len(existing_df) > 0:
            last_ts = int(existing_df["timestamp"].max())
            logger.info(f"{pair}: found {len(existing_df)} existing candles, "
                        f"last at {datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()}")

    now = int(time.time())
    target_start = now - (days * 86400)

    # If we have recent enough data, only fetch the gap
    if last_ts > 0 and last_ts > target_start:
        fetch_start = last_ts + GRANULARITY
    else:
        fetch_start = target_start

    if fetch_start >= now:
        logger.info(f"{pair}: already up to date")
        return existing_df

    # Calculate batches needed
    total_candles = math.ceil((now - fetch_start) / GRANULARITY)
    batches = math.ceil(total_candles / BATCH_SIZE)
    logger.info(f"{pair}: fetching ~{total_candles} candles in {batches} batches...")

    all_candles = []
    for i in range(batches):
        batch_end = now - (i * BATCH_SIZE * GRANULARITY)
        batch_start = batch_end - (BATCH_SIZE * GRANULARITY)

        # Don't fetch before our target
        batch_start = max(batch_start, fetch_start)
        if batch_start >= batch_end:
            break

        try:
            candles = fetch_candles_batch(pair, batch_start, batch_end)
            all_candles.extend(candles)
            if (i + 1) % 10 == 0 or i == batches - 1:
                logger.info(f"  {pair}: batch {i+1}/{batches} — {len(all_candles)} candles so far")
            time.sleep(0.3)  # Rate limit
        except Exception as e:
            logger.warning(f"  {pair}: batch {i+1}/{batches} failed: {e}")
            time.sleep(1.0)

    if not all_candles:
        logger.warning(f"{pair}: no new candles fetched")
        return existing_df if existing_df is not None else pd.DataFrame()

    # Parse into DataFrame — Coinbase: [timestamp, low, high, open, close, volume]
    new_df = pd.DataFrame(all_candles, columns=["timestamp", "low", "high", "open", "close", "volume"])
    # Reorder columns to standard OHLCV
    new_df = new_df[["timestamp", "open", "high", "low", "close", "volume"]]
    new_df = new_df.astype({"timestamp": int, "open": float, "high": float,
                            "low": float, "close": float, "volume": float})

    # Merge with existing data
    if existing_df is not None and len(existing_df) > 0:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Deduplicate and sort
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    # Save
    combined.to_csv(csv_path, index=False)
    logger.info(f"{pair}: saved {len(combined)} candles to {csv_path}")

    return combined


def load_cached_data(pair: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> Optional[pd.DataFrame]:
    """Load cached CSV data for a pair if it exists.

    Returns:
        DataFrame or None if not found
    """
    csv_path = os.path.join(output_dir, f"{pair}.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
    return df


def download_all(
    pairs: List[str] = None,
    days: int = 30,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> Dict[str, pd.DataFrame]:
    """Download candles for all pairs.

    Args:
        pairs: List of trading pairs (default: 6 major pairs)
        days: Number of days of history
        output_dir: Directory to save CSV files

    Returns:
        Dict of pair → DataFrame
    """
    if pairs is None:
        pairs = DEFAULT_PAIRS

    results = {}
    total_candles = 0

    for pair in pairs:
        df = download_pair(pair, days, output_dir)
        if df is not None and len(df) > 0:
            results[pair] = df
            total_candles += len(df)

    logger.info(f"Downloaded {total_candles} total candles across {len(results)} pairs")
    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Download training data from Coinbase")
    parser.add_argument("--days", type=int, default=30, help="Days of history (default: 30)")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help="Trading pairs to download")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for CSV files")
    args = parser.parse_args()

    results = download_all(args.pairs, args.days, args.output_dir)

    print(f"\nSummary:")
    print(f"{'Pair':<12} {'Candles':>8} {'Days':>6}")
    print("-" * 28)
    for pair, df in results.items():
        n_days = len(df) * 5 / (60 * 24) if len(df) > 0 else 0
        print(f"{pair:<12} {len(df):>8} {n_days:>6.1f}")


if __name__ == "__main__":
    main()
