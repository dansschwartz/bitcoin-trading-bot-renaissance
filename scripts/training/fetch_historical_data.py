"""
Download 5+ years of 5-minute OHLCV candles from Binance via CCXT.

Uses the Binance public API (no authentication required) to fetch historical
candle data for ML model training. Saves to CSV with incremental resume support.

Pairs and start dates:
  BTC, ETH:       2017-09-01
  LINK, DOGE:     2019-01-01
  SOL, AVAX:      2020-09-01
  All pairs:      through today

Usage:
    python -m scripts.training.fetch_historical_data                    # all pairs, full history
    python -m scripts.training.fetch_historical_data --pairs BTC ETH    # specific pairs only
    python -m scripts.training.fetch_historical_data --resume            # continue interrupted download
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

TIMEFRAME = "5m"
CANDLE_MS = 5 * 60 * 1000  # 5 minutes in milliseconds
BATCH_SIZE = 1000  # Binance max per request
RATE_LIMIT_MS = 100  # 100ms between requests
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "training")

# Binance uses USDT pairs; we map to our BTC-USD naming convention
PAIR_CONFIG = {
    "BTC-USD":  {"symbol": "BTC/USDT",  "start": "2017-09-01"},
    "ETH-USD":  {"symbol": "ETH/USDT",  "start": "2017-09-01"},
    "LINK-USD": {"symbol": "LINK/USDT", "start": "2019-01-01"},
    "DOGE-USD": {"symbol": "DOGE/USDT", "start": "2019-01-01"},
    "SOL-USD":  {"symbol": "SOL/USDT",  "start": "2020-09-01"},
    "AVAX-USD": {"symbol": "AVAX/USDT", "start": "2020-09-01"},
}

ALL_PAIRS = list(PAIR_CONFIG.keys())


def _parse_start_ts(date_str: str) -> int:
    """Parse a YYYY-MM-DD string to Unix milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ts_to_str(ts_ms: int) -> str:
    """Convert Unix milliseconds to human-readable string."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _estimate_total_candles(start_ms: int, end_ms: int) -> int:
    """Estimate total candles between two timestamps."""
    return max(0, (end_ms - start_ms) // CANDLE_MS)


# ══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def create_exchange():
    """Create a CCXT Binance exchange instance."""
    import ccxt
    exchange = ccxt.binance({
        "enableRateLimit": False,  # We handle rate limiting ourselves
        "options": {"defaultType": "spot"},
    })
    return exchange


def download_pair(
    pair: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    resume: bool = True,
) -> Optional[pd.DataFrame]:
    """Download full history for a single pair.

    Args:
        pair: Our pair name e.g. "BTC-USD"
        output_dir: Directory to save CSV
        resume: If True, resume from last downloaded candle

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    if pair not in PAIR_CONFIG:
        logger.error(f"Unknown pair: {pair}. Available: {list(PAIR_CONFIG.keys())}")
        return None

    config = PAIR_CONFIG[pair]
    symbol = config["symbol"]
    start_ms = _parse_start_ts(config["start"])
    now_ms = int(time.time() * 1000)
    total_expected = _estimate_total_candles(start_ms, now_ms)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{pair}_5m_historical.csv")

    # Check for existing data (resume support)
    existing_count = 0
    if resume and os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if len(existing_df) > 0:
                last_ts = int(existing_df["timestamp"].max())
                existing_count = len(existing_df)
                start_ms = last_ts + CANDLE_MS  # Start from next candle
                logger.info(
                    f"{pair}: resuming from {_ts_to_str(last_ts)} "
                    f"({existing_count:,} candles already downloaded)"
                )
                if start_ms >= now_ms:
                    logger.info(f"{pair}: already up to date")
                    return existing_df
        except Exception as e:
            logger.warning(f"{pair}: could not read existing CSV, starting fresh: {e}")

    remaining = _estimate_total_candles(start_ms, now_ms)
    logger.info(
        f"{pair}: downloading {symbol} from {_ts_to_str(start_ms)} to now "
        f"(~{remaining:,} candles)"
    )

    exchange = create_exchange()
    all_candles = []
    since = start_ms
    request_count = 0
    last_progress_count = 0

    while since < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=BATCH_SIZE)

            if not ohlcv:
                break

            all_candles.extend(ohlcv)
            request_count += 1

            # Advance to next batch: use the last candle's timestamp + 1 interval
            last_candle_ts = ohlcv[-1][0]
            since = last_candle_ts + CANDLE_MS

            # Progress logging every 10,000 candles
            total_so_far = existing_count + len(all_candles)
            if total_so_far - last_progress_count >= 10000:
                pct = (total_so_far / total_expected * 100) if total_expected > 0 else 0
                logger.info(
                    f"  {pair}: {total_so_far:,} / ~{total_expected:,} candles ({pct:.0f}%)"
                )
                last_progress_count = total_so_far

            # Rate limiting
            time.sleep(RATE_LIMIT_MS / 1000)

            # If we got fewer than BATCH_SIZE, we've reached the end
            if len(ohlcv) < BATCH_SIZE:
                break

        except Exception as e:
            logger.warning(f"  {pair}: request failed at {_ts_to_str(since)}: {e}")
            time.sleep(2.0)  # Back off on error
            continue

    if not all_candles:
        logger.info(f"{pair}: no new candles fetched")
        if resume and os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return None

    # Convert to DataFrame
    new_df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    new_df = new_df.astype({
        "timestamp": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
    })

    # Merge with existing data if resuming
    if resume and os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    # Deduplicate and sort
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    # Save
    combined.to_csv(csv_path, index=False)
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    logger.info(f"{pair}: saved {len(combined):,} candles to {csv_path} ({size_mb:.1f} MB)")

    return combined


def download_all(
    pairs: Optional[List[str]] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    resume: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Download historical data for all specified pairs.

    Args:
        pairs: List of our pair names (default: all 6)
        output_dir: Directory for CSV files
        resume: Resume interrupted downloads

    Returns:
        Dict of pair → DataFrame
    """
    if pairs is None:
        pairs = ALL_PAIRS

    results = {}
    total_candles = 0

    for i, pair in enumerate(pairs, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(pairs)}] Downloading {pair}")
        logger.info(f"{'='*60}")

        df = download_pair(pair, output_dir=output_dir, resume=resume)
        if df is not None and len(df) > 0:
            results[pair] = df
            total_candles += len(df)

    logger.info(f"\nTotal: {total_candles:,} candles across {len(results)} pairs")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_data_quality(pair: str, df: pd.DataFrame) -> Dict:
    """Run data quality checks on downloaded candles.

    Checks:
      1. Gaps (missing candles where timestamp gap > 5 min)
      2. Duplicate timestamps
      3. OHLCV validity (high >= max(open,close), low <= min(open,close), volume >= 0)

    Args:
        pair: Pair name for logging
        df: DataFrame with OHLCV data

    Returns:
        Dict with quality metrics
    """
    n = len(df)
    report = {"pair": pair, "total_candles": n, "issues": []}

    # 1. Check for gaps
    timestamps = df["timestamp"].values
    diffs = np.diff(timestamps)
    gap_mask = diffs > CANDLE_MS
    n_gaps = int(gap_mask.sum())
    gap_pct = n_gaps / max(n - 1, 1) * 100
    report["gaps"] = n_gaps
    report["gap_pct"] = gap_pct
    logger.info(f"  {pair}: {n:,} candles, {n_gaps} gaps detected ({gap_pct:.3f}%)")

    # 2. Check for duplicates (should be 0 after dedup in download)
    n_dupes = df.duplicated(subset="timestamp").sum()
    report["duplicates"] = int(n_dupes)
    if n_dupes > 0:
        logger.warning(f"  {pair}: {n_dupes} duplicate timestamps found")
        report["issues"].append(f"{n_dupes} duplicates")

    # 3. OHLCV validity
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values

    # high >= max(open, close)
    high_ok = highs >= np.maximum(opens, closes)
    n_high_bad = int((~high_ok).sum())

    # low <= min(open, close)
    low_ok = lows <= np.minimum(opens, closes)
    n_low_bad = int((~low_ok).sum())

    # volume >= 0
    n_vol_bad = int((volumes < 0).sum())

    ohlcv_issues = n_high_bad + n_low_bad + n_vol_bad
    report["ohlcv_invalid"] = ohlcv_issues
    if ohlcv_issues > 0:
        logger.warning(
            f"  {pair}: OHLCV issues — high violations: {n_high_bad}, "
            f"low violations: {n_low_bad}, negative volume: {n_vol_bad}"
        )
        report["issues"].append(f"{ohlcv_issues} OHLCV violations")
    else:
        logger.info(f"  {pair}: OHLCV validity OK")

    return report


def run_quality_checks(pair_dfs: Dict[str, pd.DataFrame]) -> List[Dict]:
    """Run quality checks on all pairs."""
    logger.info("\n" + "=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)

    reports = []
    for pair, df in sorted(pair_dfs.items()):
        report = check_data_quality(pair, df)
        reports.append(report)

    return reports


# ══════════════════════════════════════════════════════════════════════════════
# REGIME DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_regime_distribution(pair: str, df: pd.DataFrame) -> Dict[str, float]:
    """Classify each bar into a regime and report distribution.

    Regime classification:
      - Compute 20-bar ATR and 100-bar SMA
      - Trending: price > SMA + ATR or price < SMA - ATR
      - High volatility: ATR > 1.5x 100-bar average ATR
      - Low volatility: everything else

    Args:
        pair: Pair name for logging
        df: DataFrame with OHLCV data

    Returns:
        Dict of regime → fraction
    """
    if len(df) < 120:
        logger.warning(f"  {pair}: too few bars ({len(df)}) for regime analysis")
        return {"low_vol": 1.0, "trending": 0.0, "high_vol": 0.0}

    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    # True Range
    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # 20-bar ATR (rolling mean of TR)
    atr_20 = pd.Series(tr).rolling(20, min_periods=20).mean().values

    # 100-bar SMA of close
    sma_100 = pd.Series(close).rolling(100, min_periods=100).mean().values

    # 100-bar average ATR (for high-vol threshold)
    avg_atr_100 = pd.Series(tr).rolling(100, min_periods=100).mean().values

    # Classify each bar (only where all indicators are available)
    regimes = []
    for i in range(len(df)):
        if np.isnan(atr_20[i]) or np.isnan(sma_100[i]) or np.isnan(avg_atr_100[i]):
            continue

        price = close[i]
        atr = atr_20[i]
        sma = sma_100[i]
        avg_atr = avg_atr_100[i]

        if price > sma + atr or price < sma - atr:
            regimes.append("trending")
        elif avg_atr > 0 and atr > 1.5 * avg_atr:
            regimes.append("high_vol")
        else:
            regimes.append("low_vol")

    if not regimes:
        return {"low_vol": 1.0, "trending": 0.0, "high_vol": 0.0}

    total = len(regimes)
    dist = {
        "low_vol": regimes.count("low_vol") / total,
        "trending": regimes.count("trending") / total,
        "high_vol": regimes.count("high_vol") / total,
    }
    return dist


def run_regime_analysis(pair_dfs: Dict[str, pd.DataFrame]) -> None:
    """Run regime distribution analysis on all pairs."""
    logger.info("\n" + "=" * 60)
    logger.info("REGIME DISTRIBUTION")
    logger.info("=" * 60)

    print(f"\n{'Pair':<12} {'Low Vol':>10} {'Trending':>10} {'High Vol':>10} {'Bars':>12}")
    print("-" * 58)

    for pair in sorted(pair_dfs.keys()):
        df = pair_dfs[pair]
        dist = compute_regime_distribution(pair, df)
        print(
            f"{pair:<12} "
            f"{dist['low_vol']*100:>9.1f}% "
            f"{dist['trending']*100:>9.1f}% "
            f"{dist['high_vol']*100:>9.1f}% "
            f"{len(df):>11,}"
        )

    print()


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPER
# ══════════════════════════════════════════════════════════════════════════════

def load_historical_csvs(
    pairs: Optional[List[str]] = None,
    data_dir: str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, pd.DataFrame]:
    """Load historical CSVs from data/training/ directory.

    Args:
        pairs: List of pairs to load (default: all available)
        data_dir: Directory containing CSV files

    Returns:
        Dict of pair → DataFrame
    """
    if not os.path.exists(data_dir):
        logger.error(f"Training data directory not found: {data_dir}")
        return {}

    if pairs is None:
        pairs = ALL_PAIRS

    pair_dfs = {}
    for pair in pairs:
        csv_path = os.path.join(data_dir, f"{pair}_5m_historical.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                pair_dfs[pair] = df
                logger.info(f"  Loaded {pair}: {len(df):,} bars")
        else:
            logger.warning(f"  {pair}: no historical CSV found at {csv_path}")

    return pair_dfs


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download 5+ years of 5-minute candle data from Binance via CCXT"
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Pairs to download (short names: BTC ETH SOL DOGE AVAX LINK). Default: all 6",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume interrupted download (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start download from scratch, ignoring existing CSVs",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Parse pairs — accept short names like "BTC" or full names like "BTC-USD"
    pairs = None
    if args.pairs:
        pairs = []
        for p in args.pairs:
            if "-" in p:
                pairs.append(p)
            else:
                pairs.append(f"{p}-USD")

    resume = not args.no_resume

    # Download
    start_time = time.time()
    pair_dfs = download_all(pairs=pairs, output_dir=args.output_dir, resume=resume)

    if not pair_dfs:
        logger.error("No data downloaded!")
        sys.exit(1)

    # Quality checks
    run_quality_checks(pair_dfs)

    # Regime analysis
    run_regime_analysis(pair_dfs)

    # Summary
    elapsed = time.time() - start_time
    total_candles = sum(len(df) for df in pair_dfs.values())
    total_mb = sum(
        os.path.getsize(os.path.join(args.output_dir, f"{p}_5m_historical.csv")) / (1024 * 1024)
        for p in pair_dfs.keys()
        if os.path.exists(os.path.join(args.output_dir, f"{p}_5m_historical.csv"))
    )

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"{'Pair':<12} {'Candles':>12} {'Date Range':<30} {'Size':>8}")
    print("-" * 65)
    for pair in sorted(pair_dfs.keys()):
        df = pair_dfs[pair]
        first = _ts_to_str(int(df["timestamp"].iloc[0]))
        last = _ts_to_str(int(df["timestamp"].iloc[-1]))
        csv_path = os.path.join(args.output_dir, f"{pair}_5m_historical.csv")
        size_mb = os.path.getsize(csv_path) / (1024 * 1024) if os.path.exists(csv_path) else 0
        print(f"{pair:<12} {len(df):>12,} {first} → {last}  {size_mb:>6.1f}MB")

    print(f"\nTotal: {total_candles:,} candles, {total_mb:.0f} MB")
    print(f"Elapsed: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
