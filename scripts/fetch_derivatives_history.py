#!/usr/bin/env python3
"""
Fetch historical derivatives data from Binance Futures + Fear & Greed from Alternative.me.

Downloads and saves CSVs for training ML models with derivatives features.

Output files in data/training/derivatives/:
  - {PAIR}_derivatives.csv  (per-pair: funding_rate, open_interest, long_short_ratio, taker_buy/sell_vol)
  - fear_greed_history.csv  (global daily Fear & Greed index)

Usage:
    python -m scripts.fetch_derivatives_history
    python -m scripts.fetch_derivatives_history --days 365
    python -m scripts.fetch_derivatives_history --pairs BTC-USD ETH-USD
"""

import argparse
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from derivatives_data_provider import DerivativesDataProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_derivatives")

DEFAULT_PAIRS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'AVAX-USD', 'LINK-USD']
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "training", "derivatives")


def main():
    parser = argparse.ArgumentParser(description="Fetch historical derivatives data for training")
    parser.add_argument("--days", type=int, default=730, help="Days of history to fetch (default: 730)")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="Pairs to fetch")
    parser.add_argument("--period", default="5m", help="Granularity for OI/LS/taker data (default: 5m)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    provider = DerivativesDataProvider()

    # Fetch per-pair derivatives data
    for pair in args.pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fetching derivatives for {pair} ({args.days} days, period={args.period})")
        logger.info(f"{'='*60}")

        df = provider.fetch_historical_derivatives(pair, days_back=args.days, period=args.period)

        if df.empty:
            logger.warning(f"  No data for {pair}")
            continue

        out_path = os.path.join(args.output_dir, f"{pair}_derivatives.csv")
        df.to_csv(out_path, index=False)
        size_mb = os.path.getsize(out_path) / 1e6
        logger.info(f"  Saved: {out_path} ({len(df):,} rows, {size_mb:.1f} MB)")

    # Fetch Fear & Greed history (global, not per-pair)
    logger.info(f"\n{'='*60}")
    logger.info("Fetching Fear & Greed history (all available)")
    logger.info(f"{'='*60}")

    fng_df = provider.fetch_historical_fear_greed(limit=0)

    if not fng_df.empty:
        fng_path = os.path.join(args.output_dir, "fear_greed_history.csv")
        fng_df.to_csv(fng_path, index=False)
        logger.info(f"  Saved: {fng_path} ({len(fng_df):,} rows)")
    else:
        logger.warning("  No Fear & Greed data fetched")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    files = [f for f in os.listdir(args.output_dir) if f.endswith('.csv')]
    total_size = 0
    for f in sorted(files):
        path = os.path.join(args.output_dir, f)
        size = os.path.getsize(path)
        total_size += size
        import pandas as pd
        rows = len(pd.read_csv(path))
        logger.info(f"  {f}: {rows:,} rows ({size / 1e6:.1f} MB)")
    logger.info(f"  Total: {len(files)} files, {total_size / 1e6:.1f} MB")
    logger.info(f"\nNext step: run training with --derivatives flag to use this data")


if __name__ == "__main__":
    main()
