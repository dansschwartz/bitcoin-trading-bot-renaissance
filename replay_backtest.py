#!/usr/bin/env python3
"""
Replay/backtest harness for the golden path.
Feeds historical OHLCV bars into the bot to validate signals and sizing.
"""

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Dict, Any

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

from renaissance_trading_bot import RenaissanceTradingBot
from enhanced_technical_indicators import PriceData


@dataclass
class BacktestStats:
    total: int = 0
    buy: int = 0
    sell: int = 0
    hold: int = 0


def _load_rows(args) -> Iterable[Dict[str, Any]]:
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required for replay_backtest.py")

    df = pd.read_csv(args.csv)

    for col in [args.timestamp_col, args.open_col, args.high_col, args.low_col, args.close_col, args.volume_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    df = df.copy()
    df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col])

    for _, row in df.iterrows():
        yield {
            "timestamp": row[args.timestamp_col],
            "open": float(row[args.open_col]),
            "high": float(row[args.high_col]),
            "low": float(row[args.low_col]),
            "close": float(row[args.close_col]),
            "volume": float(row[args.volume_col])
        }


async def run_backtest(args) -> None:
    bot = RenaissanceTradingBot(args.config)

    cash = float(args.initial_cash)
    btc = 0.0
    stats = BacktestStats()

    current_day = None
    day_start_equity = cash

    for row in _load_rows(args):
        price_data = PriceData(
            timestamp=row["timestamp"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )

        bot.technical_indicators.update_price_data(price_data)

        market_data = {
            "order_book_snapshot": None,
            "price_data": price_data,
            "technical_signals": bot.technical_indicators.get_latest_signals(),
            "alternative_signals": None,
            "timestamp": price_data.timestamp,
        }

        signals = await bot.generate_signals(market_data)
        weighted_signal, contributions = bot.calculate_weighted_signal(signals)
        decision = bot.make_trading_decision(weighted_signal, contributions)

        stats.total += 1
        if decision.action == "BUY":
            stats.buy += 1
        elif decision.action == "SELL":
            stats.sell += 1
        else:
            stats.hold += 1

        equity = cash + btc * price_data.close
        trade_day = price_data.timestamp.date()
        if current_day != trade_day:
            current_day = trade_day
            day_start_equity = equity
            bot.daily_pnl = 0.0
        else:
            bot.daily_pnl = equity - day_start_equity

        if price_data.close <= 0:
            continue

        current_position_value = btc * price_data.close
        target_value = current_position_value

        if decision.action == "BUY":
            target_value = equity * decision.position_size
        elif decision.action == "SELL":
            target_value = 0.0

        delta_value = target_value - current_position_value

        if delta_value > 0:
            spend = min(cash, delta_value)
            btc += spend / price_data.close
            cash -= spend
        elif delta_value < 0:
            sell_value = min(current_position_value, abs(delta_value))
            btc -= sell_value / price_data.close
            cash += sell_value

    final_equity = cash + btc * price_data.close if stats.total else cash

    print("\nBacktest Summary")
    print("----------------")
    print(f"Bars processed: {stats.total}")
    print(f"BUY decisions: {stats.buy}")
    print(f"SELL decisions: {stats.sell}")
    print(f"HOLD decisions: {stats.hold}")
    print(f"Final equity: {final_equity:,.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay/backtest harness for the golden path")
    parser.add_argument("--csv", required=True, help="Path to CSV with OHLCV data")
    parser.add_argument("--config", default="config/config.json", help="Path to config JSON")
    parser.add_argument("--initial-cash", type=float, default=10000.0, help="Initial cash balance")

    parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name")
    parser.add_argument("--open-col", default="open", help="Open column name")
    parser.add_argument("--high-col", default="high", help="High column name")
    parser.add_argument("--low-col", default="low", help="Low column name")
    parser.add_argument("--close-col", default="close", help="Close column name")
    parser.add_argument("--volume-col", default="volume", help="Volume column name")

    args = parser.parse_args()
    asyncio.run(run_backtest(args))


if __name__ == "__main__":
    main()
