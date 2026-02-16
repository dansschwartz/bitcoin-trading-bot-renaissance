"""
Backfill historical 5-minute OHLCV bars from Coinbase public API.
Inserts into five_minute_bars table so the HMM regime detector can train.

Usage:
    python backfill_bars.py              # default: BTC-USD, 500 bars
    python backfill_bars.py --pairs BTC-USD ETH-USD --bars 300
"""

import argparse
import json
import math
import sqlite3
import time
import urllib.request

DB_PATH = "data/renaissance_bot.db"
COINBASE_API = "https://api.exchange.coinbase.com/products/{}/candles"
GRANULARITY = 300  # 5 minutes in seconds
BATCH_SIZE = 300   # Coinbase returns max 300 candles per request


def fetch_candles(pair: str, start: int, end: int) -> list:
    """Fetch candles from Coinbase public REST API."""
    url = COINBASE_API.format(pair)
    params = f"?granularity={GRANULARITY}&start={start}&end={end}"
    req = urllib.request.Request(url + params, headers={"User-Agent": "RenaissanceBot/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    # Coinbase returns: [[timestamp, low, high, open, close, volume], ...]
    return data


def backfill(pair: str, num_bars: int, db_path: str = DB_PATH):
    """Fetch and insert historical bars for a pair."""
    conn = sqlite3.connect(db_path)

    # Check existing bars
    existing = conn.execute(
        "SELECT COUNT(*) FROM five_minute_bars WHERE pair = ?", (pair,)
    ).fetchone()[0]
    print(f"{pair}: {existing} existing bars, fetching {num_bars} historical bars...")

    now = int(time.time())
    all_candles = []
    batches = math.ceil(num_bars / BATCH_SIZE)

    for i in range(batches):
        end = now - (i * BATCH_SIZE * GRANULARITY)
        start = end - (BATCH_SIZE * GRANULARITY)
        try:
            candles = fetch_candles(pair, start, end)
            all_candles.extend(candles)
            print(f"  Batch {i+1}/{batches}: fetched {len(candles)} candles")
            time.sleep(0.3)  # Rate limit courtesy
        except Exception as e:
            print(f"  Batch {i+1}/{batches} failed: {e}")
            break

    if not all_candles:
        print(f"  No candles fetched for {pair}")
        conn.close()
        return 0

    # Deduplicate by timestamp
    seen = set()
    unique = []
    for c in all_candles:
        ts = int(c[0])
        if ts not in seen:
            seen.add(ts)
            unique.append(c)

    # Sort by timestamp ascending
    unique.sort(key=lambda x: x[0])

    # Get existing timestamps to avoid duplicates
    existing_ts = set(
        row[0] for row in conn.execute(
            "SELECT bar_start FROM five_minute_bars WHERE pair = ?", (pair,)
        ).fetchall()
    )

    inserted = 0
    for c in unique:
        # Coinbase format: [timestamp, low, high, open, close, volume]
        bar_start = float(c[0])
        if bar_start in existing_ts:
            continue

        bar_end = bar_start + GRANULARITY
        low, high, open_, close_, volume = float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])

        # Compute log return
        log_return = 0.0
        if open_ > 0:
            log_return = math.log(close_ / open_)

        conn.execute(
            """INSERT INTO five_minute_bars
               (pair, exchange, bar_start, bar_end, open, high, low, close, volume,
                num_trades, vwap, log_return, avg_spread_bps, buy_sell_ratio, funding_rate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pair, "coinbase", bar_start, bar_end, open_, high, low, close_, volume,
             0, (open_ + close_) / 2, log_return, 0.0, 0.0, 0.0),
        )
        inserted += 1

    conn.commit()

    total = conn.execute(
        "SELECT COUNT(*) FROM five_minute_bars WHERE pair = ?", (pair,)
    ).fetchone()[0]
    print(f"  Inserted {inserted} new bars. Total: {total} bars for {pair}")
    conn.close()
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Backfill 5-min OHLCV bars from Coinbase")
    parser.add_argument("--pairs", nargs="+", default=["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"])
    parser.add_argument("--bars", type=int, default=500, help="Number of historical bars to fetch per pair")
    parser.add_argument("--db", default=DB_PATH)
    args = parser.parse_args()

    total_inserted = 0
    for pair in args.pairs:
        n = backfill(pair, args.bars, args.db)
        total_inserted += n

    print(f"\nDone. Inserted {total_inserted} total bars across {len(args.pairs)} pairs.")


if __name__ == "__main__":
    main()
