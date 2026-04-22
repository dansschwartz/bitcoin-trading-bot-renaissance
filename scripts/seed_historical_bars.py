#!/usr/bin/env python3
"""
Seed five_minute_bars table with historical candles from Coinbase.
This lets the HMM regime detector warm up immediately instead of waiting ~17 hours.

Usage:
    python scripts/seed_historical_bars.py [--count 300] [--db-path data/renaissance_bot.db]
"""

import argparse
import json
import math
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone


PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"]
GRANULARITY = 300  # 5 minutes in seconds


def fetch_coinbase_candles(pair: str, count: int = 300) -> list[dict]:
    """Fetch historical 5-minute candles from Coinbase REST API."""
    # Coinbase /products/{id}/candles returns up to 300 candles per request
    # Params: granularity (seconds), start, end (ISO 8601)
    end_time = int(time.time())
    start_time = end_time - (count * GRANULARITY)

    url = (
        f"https://api.exchange.coinbase.com/products/{pair}/candles"
        f"?granularity={GRANULARITY}&start={start_time}&end={end_time}"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "RenaissanceBot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  WARNING: Coinbase failed for {pair}: {e}")
        print(f"  Trying Binance fallback...")
        return fetch_binance_candles(pair, count)

    # Coinbase returns: [[timestamp, low, high, open, close, volume], ...]
    # Sorted newest first — reverse to chronological order
    bars = []
    for candle in reversed(data):
        ts, low, high, opn, close, volume = candle[0], candle[1], candle[2], candle[3], candle[4], candle[5]
        bar_start = float(ts)
        bar_end = bar_start + GRANULARITY
        log_ret = math.log(close / opn) if opn > 0 else 0.0
        vwap = (high + low + close) / 3.0  # Approximation
        bars.append({
            "pair": pair,
            "exchange": "coinbase",
            "bar_start": bar_start,
            "bar_end": bar_end,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "num_trades": 0,
            "vwap": vwap,
            "log_return": log_ret,
        })
    return bars


def fetch_binance_candles(pair: str, count: int = 300) -> list[dict]:
    """Fallback: fetch from Binance if Coinbase fails."""
    # Convert BTC-USD -> BTCUSDT
    symbol = pair.replace("-USD", "USDT")

    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit={count}"
    req = urllib.request.Request(url, headers={"User-Agent": "RenaissanceBot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR: Binance also failed for {pair}: {e}")
        return []

    # Binance returns: [[open_time, open, high, low, close, volume, close_time, ...], ...]
    bars = []
    for candle in data:
        bar_start = candle[0] / 1000.0  # ms -> seconds
        bar_end = bar_start + GRANULARITY
        opn, high, low, close = float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4])
        volume = float(candle[5])
        log_ret = math.log(close / opn) if opn > 0 else 0.0
        vwap = (high + low + close) / 3.0
        bars.append({
            "pair": pair,
            "exchange": "coinbase",  # Store as coinbase since that's what the bot expects
            "bar_start": bar_start,
            "bar_end": bar_end,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "num_trades": int(candle[8]) if len(candle) > 8 else 0,
            "vwap": vwap,
            "log_return": log_ret,
        })
    return bars


def seed_bars(db_path: str, count: int = 300) -> None:
    """Fetch and insert historical bars for all pairs."""
    conn = sqlite3.connect(db_path)

    total_inserted = 0
    for pair in PAIRS:
        print(f"Fetching {count} bars for {pair}...")
        bars = fetch_coinbase_candles(pair, count)
        if not bars:
            print(f"  No bars fetched for {pair}, skipping.")
            continue

        inserted = 0
        for bar in bars:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO five_minute_bars
                       (pair, exchange, bar_start, bar_end, open, high, low, close,
                        volume, num_trades, vwap, log_return)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (bar["pair"], bar["exchange"], bar["bar_start"], bar["bar_end"],
                     bar["open"], bar["high"], bar["low"], bar["close"],
                     bar["volume"], bar["num_trades"], bar["vwap"], bar["log_return"]),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass  # Duplicate — already exists

        conn.commit()
        total_inserted += inserted
        print(f"  Inserted {inserted}/{len(bars)} bars for {pair}")
        time.sleep(0.5)  # Rate limiting

    # Verify
    print(f"\nTotal inserted: {total_inserted}")
    print("\nBar counts per pair:")
    rows = conn.execute(
        "SELECT pair, COUNT(*) FROM five_minute_bars GROUP BY pair ORDER BY pair"
    ).fetchall()
    for pair, cnt in rows:
        print(f"  {pair}: {cnt} bars")

    conn.close()
    print("\nDone! The HMM should activate within 1-2 trading cycles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed historical 5-minute bars")
    parser.add_argument("--count", type=int, default=300, help="Number of bars per pair")
    parser.add_argument("--db-path", default="data/renaissance_bot.db", help="Path to DB")
    args = parser.parse_args()
    seed_bars(args.db_path, args.count)
