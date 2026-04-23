"""
Backtest the reversal strategy on historical 5-minute bar data.

Uses five_minute_bars table to simulate:
1. BTC price trajectory within each 5-minute window
2. Detection of BTC reversals
3. Altcoin divergence at the time of reversal
4. Whether the altcoin ultimately followed BTC's reversal

This gives us an estimated accuracy for the strategy.
"""

import sqlite3
import time
import numpy as np
from collections import defaultdict


def run_backtest(
    db_path: str = "data/renaissance_bot.db",
    days_back: int = 30,
):
    conn = sqlite3.connect(db_path)

    # Check what granularity we have
    tables = [t[0] for t in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    print(f"Available tables: {[t for t in tables if 'bar' in t.lower() or 'candle' in t.lower() or 'ohlc' in t.lower()]}")

    # Get BTC 5-minute bars
    cutoff = int(time.time()) - (days_back * 86400)

    btc_bars = conn.execute("""
        SELECT bar_start, open, high, low, close, volume
        FROM five_minute_bars
        WHERE pair IN ('BTC-USD', 'BTC/USDT') AND bar_start >= ?
        ORDER BY bar_start
    """, (cutoff,)).fetchall()

    if not btc_bars:
        print("No BTC bars found in five_minute_bars table")
        # Try alternate table names
        for t in tables:
            if 'btc' in t.lower() or 'market' in t.lower():
                cols = [c[1] for c in conn.execute(f"PRAGMA table_info({t})").fetchall()]
                cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"  {t}: {cnt} rows, cols={cols[:10]}")
        conn.close()
        return

    print(f"BTC bars: {len(btc_bars)}")

    # For each altcoin, load matching bars (support both pair formats)
    altcoins = {"SOL": ("SOL-USD", "SOL/USDT"), "DOGE": ("DOGE-USD", "DOGE/USDT"), "XRP": ("XRP-USD", "XRP/USDT")}
    alt_bars = {}

    for asset, (pair_dash, pair_slash) in altcoins.items():
        bars = conn.execute("""
            SELECT bar_start, open, high, low, close
            FROM five_minute_bars
            WHERE pair IN (?, ?) AND bar_start >= ?
            ORDER BY bar_start
        """, (pair_dash, pair_slash, cutoff)).fetchall()
        alt_bars[asset] = {b[0]: b for b in bars}
        print(f"{asset} bars: {len(bars)}")

    conn.close()

    # Simulate the strategy
    # For each 5-minute BTC bar, use OHLC to approximate intra-bar trajectory:
    # If close > open: assume went down to low first, then reversed up to close
    # If close < open: assume went up to high first, then reversed down to close
    # This is a simplification but captures the reversal pattern.

    results = defaultdict(lambda: {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0})

    MIN_BTC_SWING = 0.08  # % minimum swing for a "reversal"

    for bar in btc_bars:
        ts, btc_open, btc_high, btc_low, btc_close, btc_vol = bar

        if btc_open <= 0:
            continue

        # Detect reversal from OHLC
        up_swing = (btc_high - btc_open) / btc_open * 100
        down_swing = (btc_open - btc_low) / btc_open * 100
        close_move = (btc_close - btc_open) / btc_open * 100

        # Scenario 1: BTC went DOWN then reversed UP
        # Indicators: low is significantly below open, close is above low
        down_then_up = (
            down_swing >= MIN_BTC_SWING and  # Went down meaningfully
            btc_close > btc_low and          # Reversed (close above low)
            (btc_close - btc_low) / btc_open * 100 >= MIN_BTC_SWING * 0.5  # Reversal is real
        )

        # Scenario 2: BTC went UP then reversed DOWN
        up_then_down = (
            up_swing >= MIN_BTC_SWING and
            btc_close < btc_high and
            (btc_high - btc_close) / btc_open * 100 >= MIN_BTC_SWING * 0.5
        )

        if not down_then_up and not up_then_down:
            continue

        # For each altcoin, check if it followed the reversal
        for asset, pair in altcoins.items():
            alt_bar = alt_bars[asset].get(ts)
            if not alt_bar:
                continue

            alt_ts, alt_open, alt_high, alt_low, alt_close = alt_bar
            if alt_open <= 0:
                continue

            alt_close_move = (alt_close - alt_open) / alt_open * 100

            if down_then_up:
                # BTC reversed UP. We would bet altcoin UP.
                # Did altcoin close UP?
                won = alt_close >= alt_open

                # Estimate entry price based on how extreme the move was
                # Bigger BTC drop before reversal = cheaper entry
                estimated_entry = max(0.10, 0.50 - down_swing * 3)

            elif up_then_down:
                # BTC reversed DOWN. We would bet altcoin DOWN.
                won = alt_close < alt_open
                estimated_entry = max(0.10, 0.50 - up_swing * 3)

            estimated_entry = min(0.25, estimated_entry)  # Cap

            if won:
                pnl = (1.0 - estimated_entry) * 0.99 * (3.0 / estimated_entry) - 3.0
            else:
                pnl = -3.0  # Lost the bet

            results[asset]["total"] += 1
            if won:
                results[asset]["wins"] += 1
            else:
                results[asset]["losses"] += 1
            results[asset]["pnl"] += pnl

    # Print results
    print(f"\n{'='*60}")
    print(f"REVERSAL STRATEGY BACKTEST ({days_back} days)")
    print(f"{'='*60}")
    print(f"\n{'Asset':>6} | {'Bets':>5} | {'Wins':>5} | {'Win%':>6} | {'P&L':>9} | {'Avg Entry':>10}")
    print("-" * 55)

    total_bets = 0
    total_wins = 0
    total_pnl = 0

    for asset in ["SOL", "DOGE", "XRP"]:
        r = results[asset]
        wr = r["wins"] / r["total"] * 100 if r["total"] > 0 else 0
        print(
            f"{asset:>6} | {r['total']:>5} | {r['wins']:>5} | {wr:>5.1f}% | "
            f"${r['pnl']:>+8.2f} | ~$0.35"
        )
        total_bets += r["total"]
        total_wins += r["wins"]
        total_pnl += r["pnl"]

    total_wr = total_wins / total_bets * 100 if total_bets > 0 else 0
    print("-" * 55)
    print(f"{'TOTAL':>6} | {total_bets:>5} | {total_wins:>5} | {total_wr:>5.1f}% | ${total_pnl:>+8.2f}")

    print(f"\nBreakeven accuracy at $0.15 entry: 15%")
    print(f"Breakeven accuracy at $0.20 entry: 20%")
    print(f"Actual win rate: {total_wr:.1f}%")

    if total_wr > 20:
        print(f"\n*** STRATEGY IS PROFITABLE at {total_wr:.0f}% accuracy ***")
        print(f"Expected daily P&L: ~${total_pnl / days_back:.2f}")
    elif total_wr > 15:
        print(f"\nMarginal -- profitable at cheapest entries only")
    else:
        print(f"\nNot profitable with current parameters")

    return results


if __name__ == "__main__":
    run_backtest(days_back=30)
